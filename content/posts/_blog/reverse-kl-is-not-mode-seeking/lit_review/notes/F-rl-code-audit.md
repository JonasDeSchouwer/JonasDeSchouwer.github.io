# Agent F result: Audit of reverse-KL terms in RL/RLHF framework code

Convention: q = πθ (trained policy), p = π_ref; r = p/q at the sampled token; k1 = −log r = log q − log p; k2 = (log r)²/2; k3 = r − 1 − log r. Clones in scratchpad `audit-rl/`.

---

## 1a. huggingface/trl — classic PPOTrainer (v0.11.4)

**Commit:** `1f8ba929fbb6962249386f7dce20155da8a26cda` (tag v0.11.4; trainer later removed from trl entirely).

**Estimators** — `trl/trainer/ppo_trainer.py`, `_kl_penalty()` L1150–1165 (https://github.com/huggingface/trl/blob/1f8ba929fbb6962249386f7dce20155da8a26cda/trl/trainer/ppo_trainer.py#L1150-L1165):
```python
if self.config.kl_penalty == "kl":
    return logprob - ref_logprob
if self.config.kl_penalty == "abs":
    return (logprob - ref_logprob).abs()
if self.config.kl_penalty == "mse":
    return 0.5 * (logprob - ref_logprob).square()
if self.config.kl_penalty == "full":
    # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
    return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)
```
- "kl" = per-token k1 at sampled tokens; "abs" = |k1|; "mse" = k2.
- "full" = **full-vocab reverse KL**: `F.kl_div(input, target, log_target=True)` = Σ_v exp(target_v)·(target_v − input_v) = D_KL(target‖input); with input=ref, target=policy (both full log-softmax via `logprobs_from_logits(logits, None, gather=False)`, core.py L115–124) → Σ_v q(log q − log p) = **D_KL(q‖p)** exactly per position.

**Where applied** — in-reward only. `compute_rewards()` L1112–1148:
```python
kl = self._kl_penalty(logprob, ref_logprob)
non_score_reward = -self.kl_ctl.value * kl
reward = non_score_reward.clone()
reward[last_non_masked_index] += score   # RM score only on last token
```
Entire pipeline under `torch.no_grad()` (forward passes L748–764; rewards/advantages L768–785). Gradient reaches policy only via advantages (score-function route).

**Defaults:** `kl_penalty: "kl"`, `init_kl_coef: 0.2`, `adap_kl_ctrl: True` (target 6.0, horizon 10000); AdaptiveKLController (utils.py L54–69). **USED-BY-DEFAULT** (k1 in-reward). abs/mse/full: OPT-IN.

**Caveat:** logprobs from raw logits at T=1.0 while generation kwargs are user-supplied — sampling and scored distributions can differ; no correction.

## 1b. trl — "new" PPOTrainer (v0.24.0, last release with it)

**Commit:** `04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45`.

`trl/trainer/ppo_trainer.py` L558–563:
```python
# Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
logr = ref_logprobs - logprobs
kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
non_score_reward = -args.kl_coef * kl
rewards = non_score_reward.clone()
rewards[[actual_start, actual_end]] += scores
```
k1 (default) or k3, per token, **in-reward** (feeds GAE L577–591), all under `torch.no_grad()` (L464). `kl_coef: 0.05`, `kl_estimator: "k1"`.

**Temperature:** generation `temperature=args.temperature+1e-7, top_k=0, top_p=1.0` (L413–417; default 0.7). Policy logprobs from `output.scores` (already temperature-scaled); ref logits `/= args.temperature + 1e-7` (L500); training-epoch logits likewise (L609). Both sides of KL are temperature-T distributions; no sampling/scoring mismatch, but the KL is between temperature-scaled models, not T=1. Padding logprobs overwritten with INVALID_LOGPROB=1.0 (L76, L552–553).

**Status:** in-reward k1 USED-BY-DEFAULT; k3 OPT-IN. Trainer removed ≥0.25.

## 1c. trl — GRPOTrainer (main)

**Commit:** `24db2a825029f7e0e60f6d3c160485565a784503` (v1.10.0.dev0).

`trl/trainer/grpo_trainer.py` L3094–3101:
```python
if self.beta != 0.0:
    ref_per_token_logps = inputs["ref_per_token_logps"]
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    )
    # Importance sampling correction for the KL divergence
    if self.args.use_bias_correction_kl:
        per_token_kl = per_token_kl * coef_1
```
k3 per sampled token; ref logps precomputed under no_grad (L2578, L2651–2678); live `per_token_logps` → k3 grad flows through logq only: ∇ = (1 − r)·∇logq.

**`use_bias_correction_kl` (default True, grpo_config.py L965–975)**, added after DeepSeek-V3.2: multiplies k3 by `coef_1 = exp(log_importance_weights)` — the **differentiable** ratio exp(logq − old_logq) (token-level default; sequence-level broadcast option). Docstring: makes the KL gradient "the unbiased reverse-KL gradient"; changes gradient even fully on-policy (value 1, nonzero grad).

**Placement:** KL outside the clipped surrogate, added after min/clamp (L3145–3146): `per_token_loss = per_token_loss + self.beta * per_token_kl`; PG term `-min(coef_1·A, clamp(coef_1, 1−ε_low, 1+ε_high)·A)` (L3110–3117, ε=0.2). vLLM IS correction multiplies the PG term only, not the KL (L3142–3143).

**Defaults:** `beta: 0.0` (KL OPT-IN on main; β=0 skips ref model). Sampling T=1.0, top_p=1.0, top_k=0 defaults; policy and ref logits divided by sampling temperature (L1532). If user sets top_p/top_k: generation support truncated, logprobs full-softmax — no correction.

**F.kl_div elsewhere on main:** only distillation trainers (`experimental/distillation/distillation_trainer.py` L124–136, `experimental/iw_opd/iw_opd_trainer.py` L149–159; GKD ≤0.24). Same pattern: `F.kl_div(a_log, b_log, log_target=True)` = D_KL(b‖a).

---

## 2. volcengine/verl

**Commit:** `b75f0d06a72b74e471cd4d3402bcbb0ceaa2e0c8` (main).

**Estimators** — `verl/trainer/ppo/core_algos.py`, `kl_penalty()` L2187–2213, `kl_penalty_forward()` L2215–2251:
```python
if kl_penalty in ("kl", "k1"): return logprob - ref_logprob
if kl_penalty == "abs": return (logprob - ref_logprob).abs()
if kl_penalty in ("mse", "k2"): return 0.5 * (logprob - ref_logprob).square()
if kl_penalty in ("low_var_kl", "k3"):
    kl = torch.clamp(ref_logprob - logprob, min=-20, max=20)
    kld = (torch.exp(kl) - kl - 1).contiguous()
    return torch.clamp(kld, min=-10, max=10)
if kl_penalty == "full": raise NotImplementedError
```
Per token at sampled tokens. `"full"` EXISTS-UNUSED (raises). Straight-through family (L2199–2213): `"+"` suffix (e.g. "k3+") returns `backward_score - backward_score.detach() + forward_score.detach()` with backward_score = k2 → value(k3), gradient = k2's (pathwise-unbiased; their comment says so).

**In-reward** — `ray_trainer.py` `apply_kl_penalty()` L78–117: uses `old_log_probs` and `ref_log_prob` (rollout tensors, no graph); `token_level_rewards = token_level_scores - beta * kld`; adaptive/fixed controller. Gated `algorithm.use_kl_in_reward: False` — **OPT-IN**; `algorithm.kl_penalty: kl` (k1), `kl_coef: 0.001`.

**In-loss (GRPO-style)** — `verl/workers/utils/losses.py` L131–142: `kld = kl_penalty(logprob=log_prob(live), ref_logprob(no grad))`, aggregated, `policy_loss += kl_loss * config.kl_loss_coef`. Added after the aggregated dual-clip PPO term — outside clip; KL not multiplied by ratio (no bias correction on verl main). Defaults `actor.use_kl_loss: false`, `kl_loss_coef: 0.001`, `kl_loss_type: low_var_kl` — **OPT-IN globally but set in flagship GRPO recipes** (e.g. `examples/grpo_trainer/run_qwen3_8b_fsdp.sh` L153–155: use_kl_loss=True, coef 0.001, low_var_kl). De-facto verl GRPO KL: in-loss k3, β=0.001, clamps ±20 (log r) and ±10 (k3).

**Temperature/support:** rollout T=1.0, top_k=-1, top_p=1 defaults; all logprob recomputation divides logits by rollout temperature with 1e-8 floor (`transformer_impl.py` L1352, L1445). Prompt tokens excluded by response_mask.

---

## 3. OpenRLHF

**Commit:** `bc71bb19464aca306b33080b2d2bb45d154e2f49` (main).

**Estimator** — `openrlhf/models/utils.py` `compute_approx_kl()` L64–95:
```python
log_ratio = log_probs.float() - log_probs_base.float()
if kl_estimator == "k1": pass
elif kl_estimator == "k2": log_ratio = log_ratio**2 / 2.0
elif kl_estimator == "k3": log_ratio = (-log_ratio).exp() - 1 + log_ratio
return log_ratio.clamp(min=-10, max=10)
```
Output clamp ±10 on every estimator (incl. k1). Per token at sampled tokens.

**History:** v0.5.7 (`efedd74`) had boolean `--use_kl_estimator_k3` (default False → k1) + `--init_kl_coef 0.01` (cli/train_ppo_ray.py L290–298). Main: `--algo.kl.estimator {k1,k2,k3}` default k1 (L421–429), `--algo.kl.init_coef 0.01`, `--algo.kl.use_loss` (default False, "whether to use KL loss from GRPO", L484–485). Sanity messages (L683–688) recommend k2/k3 for in-loss, k1 for in-reward.

**In-reward (default):** `experience_maker.py` L205–210 (KL on rollout tensors, only when not use_loss), `compute_reward()` (models/utils.py L98–126): `reward = last_reward + (-kl_coef * kl)` under `@torch.no_grad()`. Fixed/Adaptive controller (kl_controller.py). RM score clamped to (−10,10). **USED-BY-DEFAULT (k1, 0.01, in-reward) when a ref model is configured.**

**In-loss (opt-in):** `ppo_actor.py` L319–337: live `action_log_probs`, stored base logprobs; aggregated; `loss = actor_loss + kl_loss * kl_ctl`; outside clipped surrogate.

**Temperature/support:** actor logprobs use rollout temperature (models/actor.py L285; default 1.0); vLLM samples same temperature; top-p/top-k passed to vLLM uncorrected (rollout_log_probs kept for separate vllm_kl diagnostic / IS correction of PG term, not KL).

---

## 4. allenai/open-instruct (Tülu 3 / OLMo RLVR)

**Commit (main):** `05ad0a239663bfbeab8b72d949c9534504d49ca5`.

**Current GRPO (`grpo_fast.py`, flagship): in-loss KL.**

Estimators — `model_utils.py` `estimate_kl()` L793–818: returns stack indexed 0–3:
```python
[ref_logprobs_diff,                                   # [0] k1 (logq - logp)
 (ref_logprobs_diff) ** 2 / 2,                        # [1] k2
 torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff, # [2] k3
 ratio * ref_logprobs_diff]                           # [3] importance-weighted k1 (ratio differentiable)
```
Selection — `grpo_utils.py` `compute_grpo_loss()` L504–538: `ref_logprobs_diff = (new_logprobs - ref_logprobs).clamp(-40.0, 40.0)`; comment "We want the KL loss to backpropagate through the model." `new_logprobs` live; ref no-grad (grpo_fast.py L612–615). Applied per token outside DAPO/CISPO clipped PG (grpo_fast.py L718): `per_token_loss_BT = pg_loss_BT + self.args.beta * kl_BT`. PG clip 0.2/0.272 + ρ train-vs-inference correction multiplies PG only, not KL. Estimators 0–2: grad through new_logprobs only; 3: through ratio and diff.

**Defaults:** `beta: 0.05`, `kl_estimator: 2` (**k3 default**), `load_ref_policy: True`. Tülu 3 recipe `scripts/train/tulu3/grpo_fast_8b.sh`: `--beta 0.01 --kl_estimator 2 --temperature 1.0`. **USED-BY-DEFAULT (in-loss k3).** Logits / sampling temperature (grpo_utils.py L560).

**Historical Tülu 3 PPO (paper-era, in-reward)** — `ppo_vllm_thread_ray_gtrl.py` (deleted from main 2025-11-03, `bac9393`; inspected at `9502f163`): kl1/kl2/kl3 selectable, default "kl1", `beta: 0.05`; `non_score_reward = -args.beta * kl`; inside rollout no_grad (L1147) — in-reward, score-function only.

---

## 5. NVIDIA NeMo-Aligner and NeMo-RL

### 5a. NeMo-Aligner (legacy; frozen)

**Commit:** `af2b3bd96ad78190f6ebfb552b9e599e313ce027`.

`nemo_aligner/utils/ppo_utils.py` L63–82:
```python
def calculate_kl_penalty(log_probs_a, log_probs_b, use_absolute_kl=True):
    init_policy_kl = log_probs_a - log_probs_b
    if use_absolute_kl: init_policy_kl = init_policy_kl.abs()
    return init_policy_kl
```
Per-token k1 — **|k1| by default** (`use_absolute_kl: True`, gpt_ppo_actor.yaml L28), coef `initial_policy_kl_penalty: 0.01`, fixed. **In-reward only** (subtracted before GAE); rollout tensors, no grad; gradient via advantages. Sampling T=1.0, top_k 0, top_p 1.0 → full support, no mismatch. **USED-BY-DEFAULT.**

**Full-vocab-KL claim:** NOT in NeMo-Aligner PPO (per-token |k1| only; no F.kl_div in PPO actor). Full-vocab/top-K KLs in NVIDIA code are distillation losses (NeMo-Aligner KD kernel — see Agent E; and nemo-rl below).

### 5b. NVIDIA-NeMo/RL (nemo-rl)

**Commit:** `037fb36bdad40f6336edcdfd39f7c5150b152324`.

**Estimator** — `nemo_rl/algorithms/utils.py` `calculate_kl()` L42–100:
```python
logr = (logprobs_reference - logprobs).clamp(±20)   # input clamp
k1 = -logr | k2 = logr²/2 | k3 = exp(logr) - 1 - logr
if importance_sampling_weights is not None: kl = importance_sampling_weights * kl
kl = kl.clamp(±10)                                   # output clamp
```
Tokens hitting the input clamp get IS weight detached (L75–80).

**In-loss (ClippedPGLossFn)** — `loss_functions.py` L466–521, added L734:
```python
if self.use_on_policy_kl_approximation:
    kl_importance_weights = torch.exp(curr_logprobs_unfiltered - generation_logprobs)
else:
    kl_importance_weights = torch.exp(curr_logprobs_unfiltered - curr_logprobs_unfiltered.detach())
kl = self.reference_policy_kl_penalty * calculate_kl(..., importance_sampling_weights=kl_importance_weights)
loss = actor_loss + kl + ...
```
Even in the default case the KL is multiplied by `exp(x − x.detach())` (value 1, gradient ∇logq) — **always adds the score-function term** to the in-loss KL gradient, citing arXiv:2506.09477. Ref logprobs precomputed (no grad). KL outside clip. Defaults: `reference_policy_kl_penalty: 0.01`, `reference_policy_kl_type: "k3"`, clamps 20/10, ratio_clip 0.2. Flagship `examples/configs/grpo_math_1B.yaml`: k3, 0.01, `use_kl_in_reward: false`. **In-loss k3·exp(x−sg(x)) USED-BY-DEFAULT; in-reward OPT-IN** (`advantage_estimator.py` L248–259: adv −= kl_coef·KL before advantage normalization).

**Support truncation handled explicitly:** with top-k/top-p rollout filtering, actor-loss logprobs are computed WITH filtering (renormalized) but the KL uses `curr_logprobs_unfiltered` vs unfiltered ref (L392–393, L468–474) — only framework in audit distinguishing the two.

**Full-vocab KL in nemo-rl:** `DistillationLossFn` (L1108–1206): per-token KL over top-k slice — "forward": Σ p(log p − log q); "reverse": Σ q(log q − log p) over top-k support + optional entropy correction for excluded tail (L1154–1162); default `kl_type: "mixed"`, weight 0.5 (distillation_math.yaml L17–18). `DraftCrossEntropyLossFn` (L65–105) = full-vocab soft CE (forward-KL gradient). Both distillation to a teacher, not the RL reference penalty.

---

## 6. DeepSeekMath GRPO (paper only)

arXiv:2402.03300 (v3). No official code. Eq. (3): −β·D_KL[πθ‖π_ref] per token, inside per-token sum, outside min/clip. Eq. (4): k3 with r = π_ref/πθ at sampled token. §4.1.1: "instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence … to the loss." β = 0.04 (§4.2). No detach convention specified; open implementations (trl, verl, open-instruct, OpenRLHF, nemo-rl) are the de-facto references — all: ref detached, grad through logq only (nemo-rl adds score-function factor; trl main multiplies by differentiable IS ratio by default).

---

## Cross-framework summary

| Framework (path) | Estimator (default) | Reward/loss | Grad path | Coef | Status |
|---|---|---|---|---|---|
| trl ≤0.11 PPOTrainer | k1 ("kl"; abs/mse/full opt-in; "full"=full-vocab reverse KL) | reward | advantages only | 0.2 adaptive | default |
| trl ≤0.24 PPOTrainer | k1 (k3 opt-in) | reward | advantages only | 0.05 | default |
| trl main GRPOTrainer | k3 · IS-ratio (use_bias_correction_kl=True) | loss, outside clip | ∇logq (ratio differentiable) | β=0.0 (off) | opt-in |
| verl in-reward | k1 (abs/k2/k3 opt-in; "full" raises) | reward | advantages only | 0.001 | opt-in |
| verl in-loss | k3 "low_var_kl" (clamps ±20/±10; "k3+" ST variant) | loss, outside clip | ∇logq only | 0.001 | flagship GRPO recipes |
| OpenRLHF default | k1 (clamp ±10) | reward | advantages only | 0.01 | default |
| OpenRLHF --algo.kl.use_loss | k1/k2/k3 | loss, outside clip | ∇logq only | 0.01 | opt-in |
| open-instruct grpo_fast | k3 (idx 2; k1/k2/ratio·k1 opts; clamp ±40) | loss, outside clip | ∇logq (est.3 also ratio) | β=0.05 (Tülu3 0.01) | default |
| NeMo-Aligner PPO | \|k1\| (use_absolute_kl=True) | reward | advantages only | 0.01 | default |
| nemo-rl ClippedPGLossFn | k3 · exp(x−sg(x)) (clamps ±20/±10) | loss (reward opt-in) | ∇logq incl. score-fn term | 0.01 | default |

Every RL path computes the KL per token at sampled tokens from q-rollouts; the only full-vocab reference-KL implemented is trl-classic's `kl_penalty="full"` (reverse, in-reward, no-grad) and nemo-rl's distillation losses (teacher, top-k). Temperature: trl (PPO/GRPO), verl, OpenRLHF, open-instruct all divide training logits by sampling temperature → KL between temperature-T distributions; trl-classic (≤0.11) is the exception (possible silent mismatch).

## Visitation-term verification for the in-reward family (firsthand, 2026-08-17)

GAE discount defaults, read in the pinned source (all γ=1.0 → the reward channel carries the *undiscounted* future KL penalties into earlier tokens' advantages = the visitation term of the seq-level KL gradient; see lit_review @sec-deriv):

| Framework @ commit | γ | λ | Where verified |
|---|---|---|---|
| trl ≤0.11 @ `1f8ba92` | 1.0 | 0.95 | `ppo_config.py` L166–167; `whiten_rewards: False` L187; GAE `compute_advantages` `ppo_trainer.py` L1166–1192 |
| trl ≤0.24 @ `04fd120` | 1.0 | 0.95 | `ppo_config.py`; rewards L561–566 (`non_score_reward = -kl_coef*kl`), GAE L574–584 |
| OpenRLHF @ `bc71bb1` | 1.0 | 1.0 (ray CLI) | `compute_reward` `models/utils.py` L98–126 (per-token `-kl_coef*kl` + score at EOS); GAE `experience_maker.py` L333+ (docstring: "rewards may include a KL divergence loss term"); `train_ppo_ray.py` `--algo.advantage.gamma 1`, `--algo.advantage.lambd 1` |
| open-instruct PPO @ `9502f16` | 1 | 0.95 | Args L262–264; `non_score_reward = -beta*kl` → `rewards` → GAE loop L1301–1322 |
| NeMo-Aligner @ `af2b3bd` | 1.0 | 0.95 | `gpt_ppo_actor.yaml`; `ppo_utils.py`: `calculate_kl_penalty` (|k1| when `use_absolute_kl`), `calculate_ppo_rewards` (per-token `- penalty_factor*init_policy_kl`), `calculate_advantages_and_returns` (GAE) |
| verl @ `b75f0d0` | 1.0 | 1.0 | `ppo_trainer.yaml` L68/L71; `apply_kl_penalty` `ray_trainer.py` L78+ (token-level rewards); `use_kl_in_reward: False` (opt-in), `kl_penalty: kl`, `kl_coef 0.001` |

## trl-classic `kl_penalty="full"` — G_visit-only gradient (firsthand, 2026-08-17)

The `"full"` penalty is the exact per-position full-vocab reverse KL — `F.kl_div(ref_logprob, logprob, log_target=True).sum(-1)` (`ppo_trainer.py` L1160–1162; PyTorch semantics: KL(target‖input) = Σ_v q(log q − log p)) — computed from ungathered log-softmaxes, entirely under `torch.no_grad()` (L747, L767), placed per token in the reward (L1140), fed to GAE. Returned full logits are sliced `[:, :-1]` (L1061), the identical shift as the gathered logprobs (`logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])`, L1031), so the penalty at index j is a function of exactly the prefix from which action j was sampled. Consequence (derivation in lit_review @sec-deriv / @sec-audit-notes): penalty is prefix-measurable ⇒ own-position score term has zero expectation; `no_grad` ⇒ no pathwise term; idealized expected gradient of the penalty = −β·G_visit **only** (the dual of the Tang–Munos token-level pitfall, which keeps G_local and drops G_visit). Table 1 reclassified C → B accordingly. Not covered by Tang & Munos (checked their arXiv HTML section-by-section, 2026-08-17); not yet cross-checked against Liu et al. (rethinking-KL) or RPG for prior art.

## Framework papers vs. audited estimator options (via WebFetch, 2026-08-17)

The audited frameworks mostly *do* have papers — but they are systems papers that do not describe the KL-estimator options this audit covers, so Table 1's paper-less rows stay uncited by design:

- **verl**: HybridFlow (Sheng et al., arXiv:2409.19256, EuroSys) — single/multi-controller dataflow, actor resharding, throughput; abstract has no KL-estimator content. The `"mse"`/k2, `low_var_kl`/k3, `"k3+"` options are engineering additions in `core_algos.py`.
- **OpenRLHF**: Hu et al., arXiv:2405.11143 — framework design/scheduling/documentation; not about KL estimator choices (k2/k3 opt-ins are code-level).
- **NeMo-Aligner**: Shen et al., COLM 2024, arXiv:2405.01481 — covers RLHF, DPO, SteerLM, SPIN, PEFT; **no mention of KD, the top-K stored-logits kernel, or `bwd_kl`/`fwd_kl`** — the KD path (Agent E) sits outside the paper.
- **torchtune**: now has a paper (Obozov et al., arXiv:2605.21442, May 2026) — mentions KD "with dedicated KD losses" citing Hinton et al. 2015, but never names `ForwardKLLoss`/`ReverseKLLoss` or specifies a direction.
- **trl**: no paper; canonical citation is the GitHub repo (von Werra et al.). **nemo-rl**: no paper found (NVIDIA-NeMo/RL repo only).

Implication: options like `bwd_kl`, `"mse"`, `"full"`, `ReverseKLLoss` entered these libraries as API-symmetry/completeness additions via ordinary PRs, not as research contributions — consistent with Table 1's "exists-unused" classifications.

**Leads not verified**
- "NVIDIA computes full-vocab KL to the reference policy" — not found in NeMo-Aligner or nemo-rl RL paths (distillation losses only). Megatron-LM / canary branches not searched.
- verl Megatron/torchtitan engines, `compute_policy_loss_dppo_kl`, other registered loss modes — not line-audited.
- OpenRLHF remote workers assumed graph-free (standard ray pattern), not traced.
- trl GRPO vLLM `importance_sampling_ratio` computation noted, not line-audited.
- DeepSeekMath equations from ar5iv render, not publisher PDF.
