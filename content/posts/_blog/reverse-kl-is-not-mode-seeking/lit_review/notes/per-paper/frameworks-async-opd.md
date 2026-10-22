# Open-source post-training frameworks — async / off-policy on-policy distillation (sweep 2026-09-14)

**Sources: source repositories, cloned `--depth 1` on 2026-09-14.** Not a paper note; the
TEMPLATE.md section order is adapted so that "source" = repo docs + code, and every claim
carries a `path:line` (docs) or `file:line` (code) reference at the pinned commit below.
verl is audited separately and is deliberately **out of scope here**.

| Repo | URL | Commit audited | Commit date |
|---|---|---|---|
| THUDM/slime | https://github.com/THUDM/slime | `4c193f1f37509cca70f0e88807a9305b70f63f4e` | 2026-09-03 |
| NVIDIA-NeMo/RL (nemo-rl) | https://github.com/NVIDIA-NeMo/RL | `feff70f5b6239d90dc703ae79b973ea809550732` | 2026-09-15 |
| alibaba/ROLL | https://github.com/alibaba/ROLL | `192b1a01ea61c113b2deb543f7b115783038dff8` | 2026-08-27 |
| inclusionAI/AReaL | https://github.com/inclusionAI/AReaL | `7187bc03ddb1bdc3dfadcd1045d5d87b9aa54bd5` | 2026-09-14 |
| OpenRLHF/OpenRLHF | https://github.com/OpenRLHF/OpenRLHF | `b117b2b5a19a91e5b0d9f64408b7473c3b5a45a1` | 2026-09-14 |

Conventions follow the sweep template: reverse KL = KL(student ‖ teacher); `rkl_pg` =
sampled-token score-function estimator with per-token advantage sg[log π_T − log π_θ];
`rkl` = full-vocab reverse KL optimized directly; `fkl` = forward KL. "Async" = the
student trains on rollouts sampled by an **earlier version of itself**, teacher fixed.

## Summary table

| Framework | OPD mode? | Async supported? (param, default) | Loss family | Correction on the distillation term | Doc prose on staleness for OPD |
|---|---|---|---|---|---|
| slime | yes (`--use-opd`) | yes, structurally (`train_async.py` 1-rollout lead; `--num-steps-per-rollout`; `--partial-rollout` + `--mask-offpolicy-in-partial-rollout`, both default off; `--update-weights-interval` default 1) | `rkl_pg` only; sampled-token penalty folded into the advantage | none specific to OPD; the enclosing PPO ratio + optional `--use-tis` (`--tis-clip` 2.0) multiply the whole pg_loss | none |
| nemo-rl (MOPD, `adv_estimator: opd`) | yes | **yes, by design** — `grpo.async_grpo.max_trajectory_age_steps` (default 1); SC runtime `sampler.max_staleness_versions` / `max_lookahead_versions` | `rkl_pg` (default) **or** full-vocab `rkl` (`on_policy_distillation.full`) | ICE-POP hard token gate (`truncated_importance_sampling_type: icepop`, bounds 0.2/5.0 in the shipped recipe); full-vocab variant **rejects** all IS knobs | yes, explicit |
| nemo-rl (`run_distillation.py`) | yes | no (fresh rollout each step) | teacher-top-k KL, `kl_type: forward\|reverse\|mixed`, k = `topk_logits_k` 64, renormalized over the teacher's k unless `zero_outside_topk` | n/a | none |
| ROLL | yes (`is_pure_opd` / `use_opd`) | yes, inherited from RLVR/Agentic (`async_generation_ratio`, default 0 = sync) | `rkl_pg` via `kl_penalty` (`kl` k1 default; `k3`, `abs`, `mse`, `full` also available) | none on the OPD term; generic `pg_variant` (tis/topr/cispo/kimi15/ppo) + `train_infer_correction` act on the RL loss | none for OPD (async and R3 docs discuss staleness generally) |
| AReaL (MOPD) | yes | **yes, shipped async** — `rollout.max_head_offpolicyness` (default 0; MOPD example **2**) | `rkl_pg`, multi-teacher weighted | **truncated behavior-policy IS ratio on the distillation loss itself**, `importance_ratio_cap` default 5.0 | yes (async guide; MOPD doc states the cap's purpose) |
| AReaL (legacy `teacher.*` distill mode) | yes | same (`max_head_offpolicyness: 2` in both example YAMLs) | `rkl_pg` (pure-KD branch) / plain RKL penalty (joint branch) | uncapped `exp(logprobs − old_logp)` in pure-KD; **none** in the joint branch | no |
| OpenRLHF | **no** | yes for RL (`--train.async_enable`, `--train.async_queue_size` 1, `--train.partial_rollout_enable`) | n/a | n/a | n/a |

---

## 1. THUDM/slime

### OPD mode
Yes. `--use-opd` plus `--opd-type {sglang,megatron}`
(`slime/utils/arguments.py:1137-1169`), documented at
`docs/en/advanced/on-policy-distillation.md`. Two teacher placements: an external SGLang
server scoring the student's token ids during rollout
(`slime/rollout/on_policy_distillation.py:8-67`), or a second Megatron model forwarded at
train time (`slime/backends/megatron_utils/actor.py:414-425`).

### Objective — `rkl_pg` only
The doc is explicit that the full vocabulary is never enumerated
(`docs/en/advanced/on-policy-distillation.md:28`):

> "slime does not enumerate the full vocabulary to compute this expectation. For each
> sampled token, it uses the Monte Carlo contribution"

with `d̂_t = log π_θ(a_t|h_t) − log π_T(a_t|h_t)` and `Â_t = A_t − λ_opd · d̂_t`
(lines 30-39). Implementation, `slime/backends/megatron_utils/loss.py:695-697`:

```python
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
```

`--opd-kl-coef` defaults to 1.0. The docstring credits
`thinking-machines-lab/tinker-cookbook/.../distillation/train_on_policy.py`
(`loss.py:677`). The penalty is applied **after** the base advantage estimator
(`loss.py:808-814`), so it composes with GRPO/PPO/REINFORCE++/GSPO. There is no
full-vocab or top-k variant. `--opd-kl-coef 1.0` with a 0.0 task reward
(`on_policy_distillation.py:65`) gives pure distillation.

### Async / off-policy
No OPD-specific staleness parameter exists, but nothing blocks OPD from running stale:
* `train_async.py:32-40` launches rollout `rollout_id + 1` before training on
  `rollout_id`, i.e. a one-rollout generation lead; `train_async.py:11` asserts only
  `not args.colocate`, and `train_async.py:9` notes "The framework supports other
  asynchronous approaches such as fully async".
* `--update-weights-interval` (default 1, `arguments.py:549-553`) lets the generator lag
  several rollouts.
* `--num-steps-per-rollout` (`arguments.py:721`) splits one rollout across several
  optimizer steps, so steps 2..N train on rollouts from an older student.
* `--partial-rollout` (default `False`, `arguments.py:468-476`) recycles unfinished
  samples into the data buffer so one response can span policy versions, and
  `--mask-offpolicy-in-partial-rollout` (default `False`, `arguments.py:478-484`) is the
  only explicit staleness control: "If set, only on-policy generated tokens will be used
  in training". It zeroes the whole prior response
  (`slime/rollout/sglang_rollout.py:263-265`):

```python
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length
```

The shipped OPD recipe is sync: `--rollout-batch-size 16 --n-samples-per-prompt 4
--global-batch-size 64` = exactly one optimizer step per rollout
(`examples/on_policy_distillation/run-qwen3-8B-opd.sh:64-69`).

### Off-policy correction on the distillation term
**None specific to OPD.** Note `actor.py:436` forces a student log-prob recompute whenever
`use_opd` is set (`can_reuse_log_probs_in_loss` excludes `use_opd`), so `student_log_probs`
in the KL are the **current** student's scores of stale tokens — the sampled-token
estimator is evaluated off its own sampling distribution with no reweighting. The only
corrections available are (a) the enclosing PPO ratio, applied to the OPD-modified
advantage, and (b) `--use-tis` (default off, `--tis-clip` 2.0, `--tis-clip-low` 0,
`arguments.py:1072-1090`), which targets the train/infer numerics mismatch, not
staleness, and multiplies the entire `pg_loss`
(`slime/backends/megatron_utils/loss.py:894-903`):

```python
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    ...
    pg_loss = pg_loss * tis_weights
```

An IcePop-style zeroing variant is also shipped (`loss.py:907-929`). Because
`old_log_probs` is the training engine's recompute, in async mode this ratio absorbs
staleness incidentally — it is not documented as a staleness correction.

### Doc prose on staleness for OPD
**None.** The OPD doc frames the method as strictly on-policy
(`docs/en/advanced/on-policy-distillation.md:3`): "On-policy distillation (OPD) trains a
student on response tokens sampled from the student's current policy." Staleness is never
discussed in the OPD doc.

---

## 2. NVIDIA-NeMo/RL (nemo-rl)

nemo-rl ships **three** distinct distillation paths. Only the first is async.

### 2a. MOPD — `adv_estimator: opd` on async GRPO (the relevant one)

**OPD mode:** yes, `docs/about/algorithms/mopd.md`, after the MiMo-V2-Flash report
(arXiv 2601.02780). Async is not optional but constitutive
(`docs/about/algorithms/mopd.md:6-9`):

> "MOPD runs on async GRPO and collects rollouts through NeMo Gym, so the agent loop
> drives multi-turn / multi-step interaction. Each token of the resulting student rollout
> is scored by a teacher, and the policy is updated to close the gap with the teacher."

**Async parameters.** Legacy runtime: `grpo.async_grpo.max_trajectory_age_steps`, default
`1` (`nemo_rl/algorithms/grpo.py:254-259`):

```python
class AsyncGRPOConfig(BaseModel, extra="allow"):
    enabled: bool = False
    # Maximum trajectory age in training steps for samples drawn from the
    # async replay buffer. Trajectories older than this are excluded during
    # sampling; buffer sizing also scales with this value.
    max_trajectory_age_steps: int = 1
```

Guidance at `docs/guides/async-grpo.md:220`: "**Age Limits**: Start with
`max_trajectory_age_steps: 1` and increase if needed for higher throughput"; buffer sized
`num_prompts_per_step × max_trajectory_age_steps × 2` (`async-grpo.md:217`); stale
trajectories are evicted (`async-grpo.md:187`). Single-Controller runtime replaces the
single knob with pluggable `StalenessSampler`s — `in_order` (`max_lookahead_versions`),
`weight_fifo`, `ready_first`, `windowed` (`max_staleness_versions`) —
(`docs/guides/single-controller.md:278-295`,
`nemo_rl/algorithms/async_utils/staleness_sampler.py:313-343`); the legacy-to-SC map is
`max_trajectory_age_steps: N` → `sampler.name: in_order`, `sampler.max_lookahead_versions: N`
(`single-controller.md:373`). The MOPD VLM smoke is explicitly async
(`docs/guides/models/nemotron/nemotron-3-super-omni-mopd.md:~66`): "With one-step
asynchronous trajectory staleness, the third step uses trajectories generated after the
first weight update."

**Objective (default): `rkl_pg`.** `docs/about/algorithms/mopd.md:19-32`:

> "Â_t = sg[ log π_teacher(t) − log π_student(t) ]"

> "Maximizing this advantage is reverse-KL minimization — it pushes the student toward the
> teacher's token distribution — but, in this default (top-k) form, it needs only the
> teacher's log-probability for the *sampled* token rather than the full vocabulary
> distribution."

Code (`nemo_rl/algorithms/advantage_estimator.py:644-648`):

```python
        # Â_MOPD,t = sg[log π_teacher - log π_student]  (Equation 8)
        distill_advantages = (teacher_logprobs - prev_logprobs).detach()
        # Apply mask
        advantages = distill_advantages * mask
```

`log π_student` is `prev_logprobs`, the **training engine's** recompute, not the
generator's — so the advantage is the current-student/teacher gap on stale tokens.
The doc warns that zeroing it breaks the estimator (`mopd.md:97-101`).

**Objective (full-vocab option): `rkl`.** `on_policy_distillation.full.enabled: true`
gives the exact full-vocabulary reverse KL (`mopd.md:133-142`):

> "L_t = Σ_v p_student(v) · [ log p_student(v) − log p_teacher(v) ]"

> "This is the K=V limit of the top-k estimator: with the support spanning the whole
> vocabulary the score-function tail term vanishes, so the objective is exact,
> deterministic, and free of that estimator's off-policy bias."

`divergence: reverse_kl`, `teacher_payload: hidden_states|logits`, chunked
(`ChunkedDistributedReverseKLToFixedLogits`,
`nemo_rl/algorithms/loss/utils.py:302-308`).

**Off-policy correction — the sharpest finding in the sweep.** For the sampled-token
form, the correction is an **ICE-POP hard gate**, configured in the MOPD recipe as
`use_importance_sampling_correction: true`, `truncated_importance_sampling_type: icepop`,
`truncated_importance_sampling_ratio_min: 0.2`, `truncated_importance_sampling_ratio: 5.0`
(`examples/configs/recipes/llm/mopd-qwen3-1.7b-3n8g-megatron-pack.yaml:15-25`), with
`disable_ppo_ratio: true` replacing the PPO clip. The doc annotates the intent inline
(`docs/about/algorithms/mopd.md:54-61`):

> "# REINFORCE form (drop the PPO probability-ratio clipping); on-policy correction is
> handled by the ICE-POP gate below instead."
> "# ICE-POP hard gate: zero tokens whose train/inference importance-sampling weight falls
> outside bounds, correcting async off-policy drift."

Implementation (`nemo_rl/algorithms/loss/loss_functions.py:695-714`):

```python
            elif self.truncated_importance_sampling_type == "icepop":
                token_kept_mask = (
                    actor_importance_weights_expanded
                    >= self.truncated_importance_sampling_ratio_min
                ) & (
                    actor_importance_weights_expanded
                    <= self.truncated_importance_sampling_ratio
                )
                ...
                actor_importance_weights_expanded = torch.where(
                    token_kept_mask,
                    actor_importance_weights_expanded,
                    torch.zeros_like(actor_importance_weights_expanded),
                )
```

with `actor_importance_weights_expanded = torch.exp(prev_logprobs - generation_logprobs)`
(`loss_functions.py:656-659`), i.e. training-policy over **generator** policy — exactly the
staleness ratio in async mode. `grpo.py:2482-2484` warns when a MOPD run does not select
icepop. The `nemo_rl/algorithms/opd.py:15-19` module docstring makes the division of
labour explicit:

> "Advantage computation lives in advantage_estimator.OPDAdvantageEstimator.
> IS truncation lives in loss_functions.ClippedPGLoss (ICE-POP mode)."

**For the full-vocab variant, every IS knob is rejected at construction**
(`docs/about/algorithms/mopd.md:196-203`):

> "The policy-gradient and reward-side KL knobs have no code path under this objective and
> are rejected: `disable_ppo_ratio: false`, `ratio_clip_c`, `use_cispo`,
> `force_on_policy_ratio`, `sequence_level_importance_ratios`,
> `use_importance_sampling_correction`, `truncated_importance_sampling_type`,
> `positive_example_nll_weight`, `use_kl_in_reward`, and
> `use_on_policy_kl_approximation`"

So in the *same* async runtime, at the *same* staleness, nemo-rl requires an off-policy
gate for the sampled-token estimator and forbids one for the full-vocab reverse KL,
justifying it as "free of that estimator's off-policy bias" (`mopd.md:136-139`). That is
a framework-level statement of the direction our Tier-A claim argues.

**General async prose** (`docs/guides/async-grpo.md:244-248`):

> "In standard GRPO, we assume trajectories are sampled from π_{θ_old}. However, in async
> GRPO, trajectories are actually sampled from π_{θ_generator}, which is the policy weights
> from N training steps ago (where N ≥ 1 depending on `max_trajectory_age_steps`)."

> "Without importance sampling correction, the GRPO objective becomes fundamentally
> incorrect: 1. **Incorrect probability ratios** ... 2. **Biased gradient estimates**:
> Since we're computing gradients based on samples from the wrong distribution, the policy
> updates become biased and can lead to instability."

And `docs/guides/single-controller.md:86`: "**Enable importance sampling correction**
whenever the sampler admits off-policy data (any `max_staleness_versions > 0` on the
`windowed`/`weight_fifo` samplers, or `max_lookahead_versions > 0` on `in_order`)."

**Caveat:** both shipped MOPD recipes self-distill Qwen3-1.7B, so they are correctness
smoke tests, not evidence (`mopd.md:241-243`): "Because student and teacher are identical,
the OPD loss stays near zero — it is a correctness smoke test, not a demonstration of
distillation gains." No async-vs-sync numbers anywhere.

### 2b. `run_distillation.py` — teacher-top-k KL, synchronous

`docs/about/algorithms/on-policy-distillation.md`. Rollouts are regenerated by the current
student every step; `should_use_async_rollouts` only selects the vLLM asyncio engine
(`nemo_rl/algorithms/distillation.py:818`), not staleness. No replay buffer, no age knob.

Objective: **teacher-top-k KL**, `k = distillation.topk_logits_k` (default 64,
`nemo_rl/algorithms/distillation.py:111`; config `examples/configs/distillation_math.yaml:12`),
ids chosen by the teacher (`teacher_policy.get_topk_logits(...)`, `distillation.py:905-911`),
direction selectable `kl_type: forward | reverse | mixed` with `mixed_kl_weight: 0.5`
(`loss_functions.py:1705-1709`, `1761-1777`). **Renormalization:** teacher log-probs are
always a `log_softmax` over the k gathered logits
(`nemo_rl/distributed/model_utils.py:2236`), and student log-probs likewise
(`model_utils.py:2221-2223`) **unless** `zero_outside_topk: true`, in which case the student
keeps full-vocab log-probs at those ids and the tail is handled by an explicit correction
term `H_rest − log_infinitesimal · P_rest` (`loss_functions.py:1751-1758`, with
`log_infinitesimal = -100`). No IS correction anywhere — the path is on-policy by
construction.

### 2c. `run_xtoken_off_policy_distillation.py` — *not* async in our sense

Despite the name, "off-policy" here means offline corpus distillation across mismatched
tokenizers, explicitly with no generation at all
(`docs/guides/xtoken-off-policy-distillation.md:110-113`): "There is no separate generation
worker (off-policy distillation trains on a fixed dataset) and no colocation toggle."
Loss modes P-KL / gold / H-KL, `loss_fn.reverse_kl` toggles `KL(student ‖ teacher)` vs
`KL(teacher ‖ student)` (`xtoken-off-policy-distillation.md:242`). **Out of scope** for the
async sweep; recorded so the name does not mislead.

---

## 3. alibaba/ROLL

### OPD mode
Yes, two flavours, both layered on the RLVR (or Agentic) pipeline rather than a separate
trainer: `is_pure_opd: true` (teacher KL is the entire signal) and `use_opd: true` (mixed
with external rewards) — mutually exclusive (`roll/configs/base_config.py:615-627,
680-684`). Launcher `examples/start_onpolicy_distill_pipeline.py:52-67`; recipe
`examples/qwen3-8B-onpolicy-distill-megatron/onpolicy_distill_config.yaml`
(Qwen3-8B student ← teacher, DAPO-Math-17k, `adv_estimator: reinforce`, lr 5e-7);
multi-teacher variant `multi_teacher_onpolicy_distill_config.yaml` with per-teacher
`opd_kl_coef` (`roll/configs/worker_config.py:256`, default 1.0 propagated at
`base_config.py:727-730`). Docs:
`docs_roll/docs/User Guides/Pipeline/on_policy_distill_pipeline_start.md`.

### Objective — `rkl_pg`
`on_policy_distill_pipeline_start.md:113-120`:

> "KL(π_θ ‖ π_teacher) = E_{x∼π_θ}[ log π_θ(x_{t+1}|x_{1..t}) − log π_teacher(x_{t+1}|x_{1..t}) ]"

listing "Mode Seeking", "Cannot Be Gamed", "Reduced Exposure Bias" as its advantages. Code
(`roll/utils/functionals.py:1113-1153`), multi-teacher weighted sum then negated into the
advantage:

```python
            kld_i = compute_approx_kl(
                log_probs=data.batch[log_probs_key],
                log_probs_base=data.batch[ref_key],
                action_mask=response_mask,
                kl_penalty=kl_penalty,
            )
            total_weighted_kld += kl_coef_i * kld_i
    # For pure OPD mode, advantage is directly -total_weighted_kld
    if is_pure_opd:
        advantages = -total_weighted_kld
```

(mixed mode: `advantages = advantages - total_weighted_kld`,
`functionals.py:1179`). `kl_penalty` selects the estimator
(`functionals.py:192-204`): `kl` (k1, the default and the only one matching the doc's
equation), `abs`, `mse`, `k3`, and `full` (`F.kl_div(..., log_target=True).sum(-1)`, which
would need full-vocab log-prob tensors — not wired up by the OPD path, which passes
per-token vectors). Effectively **`rkl_pg` with a k1 sampled estimator**. A duplicate
implementation lives at `roll/pipeline/agentic/utils.py:481-554`.

### Which student log-prob enters the KL — a staleness landmine
`roll/utils/functionals.py:1114`:

```python
        log_probs_key = "old_log_probs" if getattr(pipeline_config, "enable_old_logprobs_recompute", False) else "infer_logprobs"
```

`enable_old_logprobs_recompute` defaults to `False` (`base_config.py:578`) and is only
flipped on when a step has multiple backward passes per DP rank or `init_kl_coef > 0`
(`base_config.py:785-822`). The shipped OPD recipe has one backward step per rank, so the
KL is computed against **`infer_logprobs`, the rollout engine's log-probs — i.e. the
behavior (possibly stale) student**, not the current one. In async mode ROLL's OPD reward
is therefore `log π_T − log π_behavior`, which is constant w.r.t. θ and stale, whereas
slime's and nemo-rl's is `log π_T − log π_current`. Worth flagging as a genuine
cross-framework divergence.

### Async / off-policy
Inherited, not OPD-specific. `async_generation_ratio` (`roll/configs/base_config.py:502-507`):

```python
    async_generation_ratio: float = field(
        default=0,
        metadata={
            "help": "The ratio of ahead generation requests in pipeline, 0 means synchronous pipeline."
        },
    )
```

`async_pipeline` is simply `async_generation_ratio > 0` (`base_config.py:822-824`), and the
same `RLVRPipeline`/`AgenticPipeline` the OPD launcher instantiates branches on it
(`roll/pipeline/rlvr/rlvr_pipeline.py:454,477,500,511,524,559`). Constraints when enabled:
`async_generation_ratio >= 1.0`, `generate_opt_level == 1`, and forced
`is_num_return_sequences_expand = True` (`roll/pipeline/rlvr/rlvr_config.py:221-245`).
Staleness bound is the ratio itself — the test asserts
`max(0, current_step - async_generation_ratio) == step` for every sampled rollout
(`tests/distributed/scheduler/test_rollout_scheduler.py:227-228`). Doc
(`docs_roll/docs/User Guides/Advanced Features/async_training.md:173-175`):

> "1. When `async_generation_ratio` is set to a value greater than 0, the framework starts
> asynchronous training mode
> 2. The inference process generates `async_generation_ratio` times more data than needed
> for training in advance"

Nothing forbids `is_pure_opd` + `async_generation_ratio > 0`; equally, no example config
combines them.

### Off-policy correction on the distillation term
**None.** The OPD advantage is written into `data.batch["advantages"]` and then consumed by
the ordinary PG loss, so only the generic RL-side machinery applies: `pg_variant` ∈
{`vanilla`, `ppo`, `tis`, `topr`, `cispo`, `kimi15`} via `ActorPGWorker`
(`docs_roll/docs/User Guides/Configuration/offpolicy_setting.md:5-50`), plus
`train_infer_correction` (IS weight + sequential `response_mask` filters,
`roll/configs/base_config.py:471-479,581-585`) "for off-policy/mismatch handling", and
optional `advantage_clip`. No importance ratio, clip or mask is applied to
`total_weighted_kld` itself.

### Doc prose on staleness for OPD
**None.** The OPD doc's only off-policy discussion contrasts OPD with *offline*
distillation (`on_policy_distill_pipeline_start.md:93-98`), claiming the on-policy column
has **no** distribution shift: "Distribution Shift | Exists (student may enter states
unseen by teacher) | None (learns on own distribution)" — a claim that async training
silently violates and the doc never revisits. Staleness prose exists only in the generic
guides, e.g. `docs_roll/docs/User Guides/Advanced Features/router_replay.md:53,63`: "For
subsequent off-policy mini-batches, fixing the routing constrains policy staleness." /
"R3 mitigates both the training-inference discrepancy and policy staleness simultaneously,
and is the recommended path in ROLL."

---

## 4. inclusionAI/AReaL (brief)

The only framework whose **distillation loss itself** carries an explicit staleness
correction, and whose OPD examples ship async **by default**.

### MOPD (`docs/en/algorithms/mopd.md`, `areal/trainer/mopd/`)
Multi-teacher weighted reverse KL, score-function surrogate (`mopd.md:90-103`):

> "MOPD minimizes the raw weighted reverse KL Σ_j w_j D_KL(π_θ ‖ π_{T_j}) with the
> on-policy score-function surrogate:"
> ```text
> rho(a) = min(exp(log pi_theta(a) - log pi_old(a)), importance_ratio_cap)
> reward(a) = S_T(a) - W * stop_gradient(log pi_theta(a))
> mopd_loss = -mean(rho(a) * reward(a))
> loss = rl_coefficient * rl_loss + distillation_coefficient * mopd_loss
> ```
> "`importance_ratio_cap` defaults to `5.0` and bounds the importance-sampling multiplier
> to prevent exponential overflow."

`rl_coefficient: 0.0` gives pure distillation (`mopd.md:110`). Implementation
(`areal/trainer/mopd/loss.py:95-109`):

```python
    detached_log_ratio = safe_logprobs.detach() - detached_old_logprobs
    bounded_importance_weight = torch.exp(
        detached_log_ratio.clamp(max=math.log(importance_ratio_cap))
    )
    score_reward = torch.where(
        mask,
        detached_teacher_logp - (detached_teacher_weight * safe_logprobs.detach()),
        torch.zeros_like(logprobs),
    )
    # At forward time the carrier is exactly one. Its derivative is
    # d log pi_theta, preserving the score-function gradient even when
    # the detached importance ratio was clipped.
    score_function_carrier = torch.exp(safe_logprobs - safe_logprobs.detach())
    importance_weight = bounded_importance_weight * score_function_carrier
    per_token_loss = -(importance_weight * score_reward)
```

Docstring (`loss.py:43-50`): "Compute a truncated-IS multi-teacher reverse-KL surrogate.
... The behavior-policy importance ratio is capped with a stop-gradient weight to prevent
exponential overflow while retaining the score-function gradient." Crucially `old_logprobs`
is **not** the training recompute but `mopd_behavior_logprobs`, described at
`areal/trainer/ppo/actor.py:1069-1072` as "MOPD targets require immutable rollout behavior
log-probabilities" — i.e. the ratio is exactly π_current/π_behavior, the staleness ratio.
`MOPDLossConfig.importance_ratio_cap` default 5.0 (`areal/api/cli_args.py:3757-3760`).

### Async
`rollout.max_head_offpolicyness` (default `0`, `docs/en/cli_reference.md:580`: "Maximum
off-policyness for the head. If the current version is more than this many versions behind,
the request will not be accepted."). `docs/en/algorithms/async.md:8-9`: "This guide applies
to all algorithms when asynchronous training is enabled (i.e.,
`rollout.max_head_offpolicyness > 0`). Setting `rollout.max_head_offpolicyness=0` reverts
AReaL to synchronous RL. The synchronous setting is useful for debugging but is typically
2x slower than asynchronous training." Tips at `async.md:41-42`: "Higher values increase
throughput but may reduce training stability. Typical range: 2-8 depending on model size
and update frequency." Partial rollouts are also supported (`async.md:20-22`): "a single
trajectory can be segmented across multiple policy versions."

**The shipped distillation recipes are async:** `max_head_offpolicyness: 2` in
`examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml:81` (Qwen3-14B → Qwen3-0.6B) and in
`examples/distillation/gsm8k_grpo_distill_mode_rolloutEngine.yaml:27`. AReaL's general
off-policy machinery (`use_decoupled_loss`, `recompute_logprobs`, `async.md:44-58`) sits on
the RL loss.

### Legacy single-teacher `teacher.*` mode (`docs/en/algorithms/distillation.md`)
Reward `R_{i,t} = log π_T(o_{i,t}) − log π_θ(o_{i,t})` under a GRPO-style ratio
(`distillation.md:47-52`), after KDRL (arXiv 2506.02208). `teacher.engine_type: rollout`
(recommended) or `train` (deprecated), `rl_loss_weight` / `distill_loss_weight` (5e-3 in
the example). Two branches (`areal/trainer/ppo/actor.py:1089-1108`): with
`rl_loss_weight == 0` an **uncapped** ratio is used —
`importance_weight = torch.exp(logprobs - old_logp)`; with `rl_loss_weight > 0` the RKL is
a plain penalty `(logprobs - teacher_logp)` with **no** correction at all. The doc
(`distillation.md:56-60`) says only: "During pure KD, we need to set `rl_loss_weight` to 0,
so the implementation estimates the RKL gradient using importance sampling." Given
`max_head_offpolicyness: 2` in the example, the joint branch trains an uncorrected
sampled-token RKL on 2-version-stale rollouts.

### Not in the source
No async-vs-sync distillation comparison; no ablation of `importance_ratio_cap`; no
full-vocab or top-k distillation objective in either AReaL path.

---

## 5. OpenRLHF/OpenRLHF (brief)

**No on-policy distillation mode.** At `b117b2b` there is no teacher model, no distillation
loss, no KD trainer: `grep -rn "teacher\|knowledge_distill\|gkd" openrlhf/` returns nothing,
and the only repo-wide "distill" hits are the checkpoint name
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` in two example scripts. Trainers are
`ppo_trainer.py`, `ppo_trainer_async.py`, `sft_trainer.py`, `dpo_trainer.py`,
`rm_trainer.py`. (Historical OpenRLHF releases shipped a `train_kd.py`; it is absent here.)

Async RL, for the record: `--train.async_enable` (default `False`,
`openrlhf/cli/train_ppo_ray.py:293`), `--train.async_queue_size` (default `1`, line 294) —
a one-batch generator lead — and `--train.partial_rollout_enable`
(`train_ppo_ray.py:296-302`), which requires `async_enable` (line 718) and whose help text
is the only staleness prose: "Uses vLLM pause/resume for weight sync instead of locking,
allowing generation to overlap with training. In-flight samples may contain tokens from
both old and new weights." Mismatch correction via
`--algo.advantage.is_correction_level {off,token,seq}` (default `off`,
`train_ppo_ray.py:261-265`), documented in-source as "TIS = token clip, ICEPOP = token
mask, seq-mask-tis = seq mask". None of this touches distillation, since there is none.

---

## What none of these frameworks report

* **No async-vs-sync side-by-side numbers for OPD anywhere.** nemo-rl's two MOPD recipes
  self-distill (loss ≈ 0 by construction); AReaL ships async-by-default recipes with no
  sync baseline; slime's only OPD result (76% → 94% pass@1, Qwen3-8B-Base ← Qwen3-32B,
  `docs/en/advanced/on-policy-distillation.md:120-127`) is a sync run with no staleness arm.
* **No framework measures a staleness metric on the distillation term** (no logged
  "off-policyness" of the OPD advantage, no KL-vs-age curve). nemo-rl logs `is_oob_ratio`
  (fraction of ICE-POP-gated tokens) but only for the RL loss path.
* **No framework offers a full-vocab or top-k OPD objective in an async runtime except
  nemo-rl**, and nemo-rl forbids IS correction there rather than comparing the two under
  matched staleness.
* Direction of the argument, where one is made, is uniform: nemo-rl asserts the full-vocab
  (K=V) objective is "free of that estimator's off-policy bias" while the sampled-token
  form needs an ICE-POP gate; AReaL caps the behavior-policy ratio on the sampled-token
  surrogate. Both are consistent with distribution-level RKL tolerating staleness better
  than `rkl_pg`, but neither supplies a measurement.
