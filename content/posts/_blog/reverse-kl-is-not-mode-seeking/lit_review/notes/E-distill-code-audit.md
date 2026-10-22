# Agent E result: Audit of reverse-KL distillation implementations (commit-pinned)

Convention: **q = student (trained), p = teacher**; "reverse KL" = D_KL(q‖p). Local clones under scratchpad `audit-distill/`.

---

## 1. MiniLLM — microsoft/LMOps, subdir `minillm`

**Commit:** `4f2a9deb5f08e459fd44c2e4792344d78ca89fc3`. Companion patched transformers fork (teacher-mixed sampling): `t1101675/transformers` branch `minillm`, commit `be0435edd0e45f491fbb66fe9fa630d458a2ace6` (via `minillm/install.sh`).

**Key files:**
- `minillm/minillm/losses.py` — PG loss L58–96, single-step reg L98–108, ppo_loss L126–196, advantages L27–56, pt_loss L198–232. https://github.com/microsoft/LMOps/blob/4f2a9deb5f08e459fd44c2e4792344d78ca89fc3/minillm/minillm/losses.py#L58-L196
- `minillm/minillm/sampler.py` — rollout + rewards + IS weight L39–174. …/minillm/minillm/sampler.py#L68-L131
- `minillm/minillm/reward.py` — teacher log-prob reward L38–83.
- `minillm/minillm/utils.py` — `get_rev_kl` L66–69, `get_x_entropy` L50–63, `get_entropy` L15–28.
- Fork sampling: `src/transformers/generation/utils.py` L3326–3344 (mixing), L3243–3259 (teacher forward per step). https://github.com/t1101675/transformers/blob/be0435edd0e45f491fbb66fe9fa630d458a2ace6/src/transformers/generation/utils.py#L3326-L3344

**What is sampled.** Prompts x from Dolly-style dataset. Responses y token-by-token from **mixture policy** p̃(v|s) = (1−α)·q_old(v|s) + α·p(v|s), α = 0.2 (mixing full probability vectors after logits processing; do_sample, top_k=0, top_p=1.0, T=1.0):
```python
probs = nn.functional.softmax(next_token_scores, dim=-1)
if mix_in_model is not None:
    probs = (1 - mix_in_alpha) * probs + mix_in_alpha * m_probs
```
A second pure-student sample per prompt only logs `rev_kl` (sampler.py L83–101); not in the loss.

**Per-token reward** (sampler.py L77–131, all `torch.no_grad()`): r_t = log p(y_t|s_t) − log q_old(y_t|s_t). reward.py computes log p as `selection_value − logsumexp(logits)` on mean-centered teacher logits (L52–74; teacher logits NOT divided by temperature here — moot at T=1.0). Then `rewards / reward_scaling` (0.5), `torch.clip(rewards, −100, 100)`.

**Advantage / length normalization** (losses.py L27–56): A_t = whiten( (Σ_{t'≥t} γ^{t'−t} r_{t'}) / L_t ), γ=0.95, L_t = # remaining non-pad response tokens (cumsum trick L39–41). Division by remaining length is **unconditional** (L49); `--length-norm` flag only gates normalization of *logged* metrics (L182–184). `.detach()` at L56.

**PG loss** (losses.py L58–96), per token on mixture-sampled tokens only (not full vocab):
```python
log_ratio = (logprobs - old_logprobs) * mask
ratio = torch.exp(log_ratio.float())
ratio = ratio * w
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2).float() * mask) / n
```
L_pg = (1/Σmask)·Σ_t max(−A_t·ρ_t, −A_t·clip(ρ_t, 1±ε)), ρ_t = [q_θ(y_t)/q_old(y_t)]·w_t, ε=0.2. w_t = q_old(y_t)/p̃(y_t) = per-token IS weight for mixture sampling (sampler.py L110–115, no_grad; w≡1 if α None). `logprobs` recomputed under current student (temperature-divided logits, losses.py L144) = **only** grad path; `old_logprobs`, `advantages`, `w` are rollout constants. `ppo-epochs 4` inner epochs (data reuse).

**Single-step decomposition term** (losses.py L98–108; `--single-step-reg`, set in scripts): full-vocab token-level reverse KL at every response position of the mixture rollout: L_reg = (1/Σmask)·Σ_t Σ_{v∈V} q_θ(v|s_t)[log q_θ(v|s_t) − log p(v|s_t)] (as `xent − s_ent`); teacher logits `torch.no_grad()`; gradient flows through q_θ inside and outside the log (no stop-grad).

**Own-token reward retained ⇒ local term double-counted (verified firsthand 2026-08-17).** The paper (arXiv v6 §3.2) assigns R_{t+1} — *excluding* r_t — to the PG part ("(∇ℒ)_Long = 𝔼[Σ_t R_{t+1} ∇log q_θ(y_t)]") with the single-step part computed exactly in its place. The code instead keeps the full R_t in the advantage (`last_rw = rewards[:, t] + gamma * last_rw`, losses.py L43–46, so rw[:,t] includes r_t) *and* adds L_reg on top (`if self.args.single_step_reg: loss += single_step_reg_loss`, losses.py L172–173) — the local gradient component enters twice (once sampled/whitened/length-normalized inside A_t, once exact via L_reg). Alignment is airtight by construction: `rewards = t_rewards + ent_rewards` with `ent_rewards = -logprobs` (sampler.py L122–124), and that same `logprobs` tensor is stored as `old_logprobs` (sampler.py L138, L151), so reward index t contains −log q_old of exactly the token whose IS ratio sits at index t in `_pg_loss`; reward.py asserts `scores.size() == gen_ids.size()` for the teacher side. γ default 0.95 (arguments.py L158), not overridden by the run scripts. **Paper cross-check (v6 HTML, 2026-08-17):** Algorithm 1 updates θ ← θ − η[(∇ℒ)_Single + (∇ℒ)_LongNorm + ∇ℒ_PT] with the long part defined via R_{t+1}; the phrase "single-step regularization" does not appear in the paper — the added-regularizer reading exists only in the code's `--single-step-reg` flag name. ⇒ Table 1 class revised S → B (mild): the shipped estimator deviates unstated from the paper's own Alg. 1 (DistiLLM-2 pattern).

**Logged rev_kl only** (not a loss): k3 estimator `exp(log p − log q) − 1 − (log p − log q)` per sampled token of the pure-student sample (utils.py L66–69).

**Additional LM term** (trainer.py L287): total = rl_loss + `lm_coef`(=1)·pt_loss, pt_loss = 0.5·CE + 0.5·(−Σ_v p·log q) (forward-KL-style teacher CE, full vocab, teacher no_grad) on OpenWebText (`kd_ratio 0.5`).

**USED:** yes. `scripts/gpt2/minillm/train_base_xl.sh` L69–84: `--type minillm --length-norm --single-step-reg --teacher-mixed-alpha 0.2 --reward-scaling 0.5 --cliprange-reward 100`; parallel scripts for llama/llama2/opt/mistral/qwen2.5. https://github.com/microsoft/LMOps/blob/4f2a9deb5f08e459fd44c2e4792344d78ca89fc3/minillm/scripts/gpt2/minillm/train_base_xl.sh#L69-L84

---

## 2. TRL GKDTrainer — huggingface/trl

**Commit:** `24db2a825029f7e0e60f6d3c160485565a784503`. **File moved**: now `trl/experimental/gkd/gkd_trainer.py` (no `trl/trainer/gkd_trainer.py` at this commit).
URL: https://github.com/huggingface/trl/blob/24db2a825029f7e0e60f6d3c160485565a784503/trl/experimental/gkd/gkd_trainer.py#L238-L323

**`generalized_jsd_loss`** (L238–323): s = log_softmax(student_logits/T), t = log_softmax(teacher_logits/T):
```python
if beta == 0:
    jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
elif beta == 1:
    jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
else:
    mixture_log_probs = torch.logsumexp(
        torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
    jsd = beta * kl_teacher + (1 - beta) * kl_student
```
**F.kl_div semantics verified:** `F.kl_div(input, target, log_target=True)` = elementwise exp(target)·(target − input) = KL(target‖input) summed over vocab. So:
- **beta = 0** → KL(p‖q), **forward KL**.
- **beta = 1** → KL(q‖p), **reverse KL**.
- **0<beta<1** → m = (1−β)·q + β·p (β weights the **teacher**); jsd = β·KL(p‖m) + (1−β)·KL(q‖m). Endpoint cases are exact KLs; generic expression → 0 as β→0 or 1 (code switches discontinuously). Matches config docstring (gkd_config.py L66–71), docs (gkd_trainer.md L26), and GKD paper Eq. 1 with P=teacher, Q=student.

**Reduction/masking:** full-vocab sum; mask = shifted `labels != −100` (prompt + post-first-EOS = −100; `generate_on_policy_outputs` L448–489); `batchmean` = sum / #valid positions; under grad accumulation / global `num_items_in_batch` (L306–317).

**Temperature:** `args.temperature` (default **0.9**) used **only for generation** (L216–225: do_sample, top_k=0, top_p=1.0). `compute_loss` calls the loss **without** temperature (L422–428) → loss temperature 1.0 regardless of config (comment L147–148 confirms intentional; Liger fused path also omits it).

**On-policy fraction (lmbda, default 0.5):** per step one coin flip for the whole batch: `random.random() <= lmbda` → batch replaced by completions from **current student** (L501–514); `elif seq_kd` (default False) → **teacher**-generated (L515–528); else dataset completions. Loss always teacher-forced on batch tokens; student logits live; teacher logits `torch.no_grad()` (L406–410, teacher.eval()); no other detach.

**Used in reported experiments:** TRL is NOT the GKD paper's code (paper = Google-internal, T5). De-facto public reference; docs claim Eq. 1 fidelity. **Google's original GKD code: no official release found** (google-research has only `speculative_kd` etc.). Classification: reference implementation; original NOT PUBLIC (as of 2026-08-10).

---

## 3a. DistiLLM — jongwooko/distillm

**Commit:** `d47e77ff9d27721783b32213be38c1204230cc0a`.
URLs: https://github.com/jongwooko/distillm/blob/d47e77ff9d27721783b32213be38c1204230cc0a/distillm/losses.py#L66-L95 , …/finetune.py#L171-L193

**Exact mixtures (λ = `--skew-alpha` = 0.1):**
- `skewed_forward_kl` (L66–78): `mixed_probs = lam * teacher_probs + (1-lam) * student_probs`; loss = −(1/N)Σ_t Σ_v p(v)·log[λp + (1−λ)q](v) = **H(p, λp+(1−λ)q) = KL(p‖λp+(1−λ)q) + H(p)** (teacher-entropy const omitted).
- `skewed_reverse_kl` (L80–95): `mixed_probs = (1-lam) * teacher_probs + lam * student_probs`; loss = (1/N)Σ_t Σ_v q(v)·[log q(v) − log((1−λ)p + λq)(v)] = **KL(q‖(1−λ)p + λq)** exactly.
```python
mixed_probs = (1-lam) * teacher_probs + lam * student_probs
prod_probs  = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
```
Full vocab; mask `label != −100`; per-token mean over batch. Mixture in probability space then `torch.log` (no epsilon). Teacher logits `torch.no_grad()` (finetune.py L172–175); student q differentiable everywhere including inside mixture — **no detach anywhere**. Plain `reverse_kl` (L14–24) = full-vocab KL(q‖p), same masking/grads.

**Adaptive off-policy scheduler** (finetune.py L290–325, L411–418): with `--student-gen --type adaptive-*`: threshold τ from 0.0; each eval, if dev loss rose ≥ 0.1, τ += 0.1 (cap 1.0). Per step r∼U(0,1): r < τ·(1 − step/total) → **fresh batch from current student** (`SampleGenerator.run_sample`, no_grad), train + push to replay buffer (cap 1000); elif r < τ → **replay-buffer** batch; else **ground-truth** batch. Same skew loss regardless of provenance. Total = (1−kd_ratio)·CE + kd_ratio·distil, **kd_ratio = 1.0** in all distillm scripts → pure skew-KL.

**USED:** yes. `scripts/gpt2/distillm/train_0.1B_1.5B.sh` → `adaptive-sfkl`; `0.3B/0.7B` + all `scripts/opt/distillm/*` → `adaptive-srkl`; all `--student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --kd-ratio 1.0`. Caveat: `scripts/openllama2/distillm/train_3B_7B_teacher_lora.sh` uses `--type srkl-aesop` — "aesop" matches no code branch, lacks "adaptive"/"mixed" → no sampling branch triggers → SRKL on fixed data at this commit (static read).

## 3b. DistiLLM-2 — jongwooko/distillm-2 (CALD contrastive)

**Commit:** `fe4cf9bfbb4f83219dd1d69219800164b59685fa`.
URLs: https://github.com/jongwooko/distillm-2/blob/fe4cf9bfbb4f83219dd1d69219800164b59685fa/src/distillm_trainer.py#L1036-L1099 (kd_loss), #L1101-L1242 (get_batch_logps), …/generate/reformat.py#L18-L29

**Data roles:** DPO-style triples; **chosen = teacher-generated response, rejected = student-generated response** (reformat.py L20–29), generated **offline** with vLLM, temperature 0.8 (README L61), one response per prompt, before training.

**Per-token divergences** (loss_type containing "distillm"; teacher = ref_model, `torch.no_grad()` L1282–1288; full vocab; causal shift L1128–1131):
- **Chosen (teacher data)** — skew-forward: m₁ = α₁·p + (1−α₁)·q (log-space logsumexp L1160–1164; **q in m₁ NOT detached**); `tea_pos_kl = (p·(log p − log m₁)).sum(vocab)` = KL(p‖α₁p+(1−α₁)q), α₁ = 0.1.
- **Rejected (student data)** — skew-reverse: m₂ = (1−α₂)·p + α₂·**sg(q)** (L1194–1198: `log_alpha_2 + vocab_logps.detach()`); `ref_pos_kl = (q·(log q − log m₂)).sum(vocab)` = KL(q‖(1−α₂)p+α₂q), α₂ = 0.1, student inside mixture **detached**, outer q(v)·log q(v) live — deliberate stop-gradient asymmetry vs distillm-1.
```python
mix_vocab_logps = torch.logsumexp(torch.stack([
    log_one_minus_alpha_2 + tea_vocab_logps, log_alpha_2 + vocab_logps.detach()], dim=0), dim=0)
ref_pos_kl = (vocab_logps.exp() * (vocab_logps - mix_vocab_logps)).sum(-1)
```
**Aggregation:** `average_log_prob=True` → per-sequence mean over valid tokens (L1230–1234), then batch `.mean()` (L1347).

**Loss** (kd_loss L1070–1078): `distillm_v2`: L = (2−β)·SKL(chosen) + β·SRKL(rejected), β = 1.0 (gradual β 1→1.5 schedule exists L1074–1075 but `self.gradual_beta = False` hardcoded L508) → **L = SKL(teacher-data) + SRKL(student-data)**. `distillm_v1`: SRKL on both. A `gkd` loss_type reimplements JSD(0.9) (L1214–1224).

**Adaptive alpha:** code exists (L1145–1153, L1180–1187) but `self.update_alpha = False`, `self.logp_logq = None` hardcoded (L507–509) → release runs **fixed α₁ = α₂ = 0.1**.

**USED:** yes — `src/run_distillm.py` + `training_configs/deepseek-coder-1.3b-distillm2.yaml` (`loss_type: distillm_v2`); README generate→reformat→train.

---

## 4. Thinking Machines on-policy distillation — thinking-machines-lab/tinker-cookbook

**Commit:** `3d3e9767076d6f44825fb46c094a4e0fce39bba1`. `tinker_cookbook/distillation/train_on_policy.py` (header cites blog); recipe `tinker_cookbook/recipes/distillation/on_policy_distillation.py`.
URLs: https://github.com/thinking-machines-lab/tinker-cookbook/blob/3d3e9767076d6f44825fb46c094a4e0fce39bba1/tinker_cookbook/distillation/train_on_policy.py#L54-L130 , …/recipes/distillation/on_policy_distillation.py#L68-L82

**Mechanics:**
1. Student samples full rollouts (T=1.0; group_size 4). Env reward ≡ 0 (`PromptOnlyEnv.step` returns 0.0, datasets.py L113–116); `compute_advantages` group-centers → env advantages exactly 0.
2. `incorporate_kl_penalty` (L54–130): per **sampled token only** (not full vocab): kl_t = log q_sample(y_t) − log p_teacher(y_t), log q_sample = behavior-policy logprob recorded at sampling time (`datum.loss_fn_inputs["logprobs"]` L89); log p_teacher from `teacher_client.compute_logprobs_async` on the sampled sequence (L80–85, aligned `[1:]` L92). Advantage edited in place:
```python
reverse_kl = [(sampled_logprobs - torch.tensor(teacher_logprobs[1:])) * mask ...]
kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]          # coef = 1.0
if kl_discount_factor > 0:                                                  # default 0.0
    kl_advantages = discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
datum.loss_fn_inputs["advantages"] = ... advantages + kl_advantages
```
   With default discount 0.0, each token's advantage = its **own** per-token −reverse-KL (no future sum). Advantages plain numbers (no autograd). Docstring's p/q naming swapped vs ours ("KL[p||q] = log p − log q", p = student sampled logprobs); computed quantity is log q_student − log p_teacher.
3. `loss_fn = "importance_sampling"` (recipe default L81), executed **server-side in Tinker** (kernel not in repo). Official docs (tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/, fetched 2026-08-10):
```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()
```
   L = −Σ_t exp(log q_θ(y_t) − log q_sample(y_t))·A_t over sampled tokens; grad flows **only through current student's sampled-token logprobs via the ratio**; sampling logprobs, advantages constants. = "**advantage = −(log q − log p) per token, detached; IS policy-gradient surrogate**" — NOT direct backprop of E_y[log q − log p] through log q. `num_substeps=1` default → ratio ≈ 1 at update. No teacher temperature; no clipping in this loss (separate "ppo" loss_fn exists, not the distillation default).

**USED:** yes — public code for the blog; defaults (`kl_penalty_coef=1.0`, discount 0.0, importance_sampling, T=1.0) match blog description. Caveat: Tinker loss kernel closed-source (docs-verbatim only).

**Blog-era run config (fetched 2026-08-21, commit `330b73d8d763b0f65bcf04e643cc6f082e002554` = "On-policy distillation code (#42)", 2025-10-27, same day as blog).** `recipes/distillation/README.md` at that commit is the reproduction guide:
- Pipeline: (1) SFT on OpenThoughts3-1.2M via `off_policy_reasoning.py`; (2) OPD on **DeepMath-103K** (`zwhe99/DeepMath-103K`, train split, `question` field as sole user message, no system prompt, prompt truncated to 1024 tokens, sequential batching, no shuffle) — resolves blog's unnamed OPD prompt source.
- SFT defaults: Qwen3-8B-Base, renderer `qwen3`, batch 128, `max_prompts = 128*3000 = 384,000` ("400k"), max_length 16384, 1 epoch, linear LR schedule; **LR 1e-3 LoRA-128 / 1e-4 full-FT**. README: "~55% AIME'24 using a rank-128 LoRA after 3000 steps"; blog's 60% is the full-FT run.
- OPD README command (reasoning): `dataset=deepmath, groups_per_batch=512, group_size=4, lora_rank=128, learning_rate=1e-4`; **LR 1e-4 LoRA / 5e-5 full-FT**; max_tokens 4096, constant LR, fully synchronous (sampler refreshed each step), eval/save every 20. 512×150 steps = 76.8k ≈ blog's "77K prompts"; blog's "64 prompts/batch default" is the personalization (tulu3) config only.
- README teacher default is **Qwen3-8B** ("~65% AIME'24 … after 100 steps" with LoRA-128); blog headline (60→70% in ~150 steps) is full-FT with Qwen3-32B teacher. Blog footnote: Tulu3 assistant turns regenerated with Qwen3-8B.
- Blog-era code had no temperature plumbing (fixed by #103, 2025-11-23) → sampled at default T=1.0. No AIME evaluator in repo; blog eval protocol unspecified. 2026-06-16 commit #739 retargeted recipe defaults to Qwen3.5-9B (originals retired on Tinker).

---

## 5. torchtune KD recipe (forward-KL baseline)

**Commit:** `bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1`. `torchtune/modules/loss/kd_losses.py`, `recipes/knowledge_distillation_single_device.py` (+ `_distributed.py`).
URLs: https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/loss/kd_losses.py#L11-L114 , …/recipes/knowledge_distillation_single_device.py#L590-L623

- `ForwardKLLoss` (L11–59; docstring: "Implementation of jongwooko/distillm .../losses.py"): loss = −(1/N)Σ_t Σ_v p(v)·log q(v), full vocab, mask `labels != −100`, token-mean = **KL(p‖q) + H(p)** (const omitted). `ForwardKLWithChunkedOutputLoss` (L170–257) chunked.
- `ReverseKLLoss` (L62–114): loss = (1/N)Σ_t Σ_v [q log q − q log p] = **exact KL(q‖p)**, full vocab; `ReverseKLWithChunkedOutputLoss` (L260+); `SymmetricKLLoss` (L117–167) = ratio·FKL + (1−ratio)·RKL.
- Recipe: teacher logits `torch.no_grad()` (L605–606); total = (1−kd_ratio)·CE + kd_ratio·kd_loss, kd_ratio 0.5. Teacher-forced on **fixed dataset tokens** (no student sampling).
- **Usage:** every shipped KD config (`recipes/configs/{qwen2,qwen2_5,qwen3,llama3_2}/*KD_lora*.yaml`) sets `ForwardKLWithChunkedOutputLoss` → **forward-KL USED**; ReverseKL/SymmetricKL **EXISTS-BUT-UNUSED** (grep-verified over configs).

---

## 6. NVIDIA — TensorRT-Model-Optimizer (modelopt) and NeMo-Aligner

### 6a. modelopt — commit `9220fac0535e4afd7e9929822eb332e09ea8a4f8`
- **`modelopt/torch/distill/losses.py` `LogitsDistillationLoss`** (L28–71): `F.kl_div(log_softmax(s/T), softmax(t/T).detach())`·T² = **forward KL(p‖q)**, full vocab; teacher `.detach()` (L60). https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/9220fac0535e4afd7e9929822eb332e09ea8a4f8/modelopt/torch/distill/losses.py#L28-L71 . (Default `reduction="mean"` divides by all elements incl. vocab; HF `KDTrainer` uses `reduction="none"`, vocab-summed, causal-shifted, −100-masked token-mean, `plugins/huggingface.py` L233–246.)
- **Megatron plugin `LogitsKLLoss`** (`plugins/megatron.py` L292–370), used by Megatron/NeMo Minitron-style workflow via `examples/megatron_bridge/distill.py`:
```python
p, q = F.log_softmax(output_student, dim=-1), F.log_softmax(output_teacher, dim=-1)
if self._reverse:
    p, q = q, p
loss = torch.sum(F.kl_div(p, q, reduction="none", log_target=True), dim=-1)
```
  Default `reverse=False` → **forward KL(p‖q)**, full vocab (TP-aware), temperature both (default 1.0). **`reverse=True` exists** (would give KL(q‖p)) but `DistillationConfig` (L50–160) exposes no field and instantiates `LogitsKLLoss(student_cfg, temperature=...)` (L135–136) → **reverse EXISTS-BUT-UNUSED (unreachable from any config/example)**. `TopKLogitsKLLoss` (L373–458): same KL on teacher's global top-K (default 1024) renormalized; same unreachable reverse flag. Teacher `requires_grad_(False)` + no_grad forward.
- **HF plugin liger path:** `liger_jsd_beta` (default 0.0; "0=forward KL, 1=reverse KL", huggingface.py L77–85, ARGUMENTS.md L25) — reverse-capable knob, only Liger fused path; default + all examples 0.0 (forward).
- **Usage:** `examples/llm_distill/main.py` → KDTrainer → LogitsDistillationLoss (forward). Megatron-bridge recipe → LogitsKLLoss default (forward). **Forward USED; reverse EXISTS-BUT-UNUSED.**

### 6b. NeMo-Aligner — commit `af2b3bd96ad78190f6ebfb552b9e599e313ce027`
- Model: `nemo_aligner/models/nlp/gpt/megatron_gpt_knowledge_distillation.py` (L48–57: `kd_loss` ∈ {"fwd_kl" (default), "bwd_kl"}, `target_logits_scale`/`logits_scale` 1.0, `sft_loss_weight` 0, `kd_loss_weight` 1; loss call L115–131, loss-mask token-mean). Kernel: `nemo_aligner/utils/distributed.py` `_TopKLogitsCrossEntropy` (custom autograd.Function, L510–760). URLs: https://github.com/NVIDIA/NeMo-Aligner/blob/af2b3bd96ad78190f6ebfb552b9e599e313ce027/nemo_aligner/utils/distributed.py#L510-L760 , …/megatron_gpt_knowledge_distillation.py#L115-L131
- **Data:** teacher's **top-K logits precomputed offline into dataset** (`topk_logits`, `topk_token_ids`; tutorial K=5) — no teacher forward at train time.
- **Loss (top-K restricted & renormalized):** S = teacher's top-K ids at position t; p̃ = softmax over S of scaled teacher logits; q̃ = softmax over S of student logits gathered at those ids. `fwd_kl` (L616–627): Σ_{k∈S} p̃_k(log p̃_k − log q̃_k). **`bwd_kl` (L609–614):**
```python
# loss = sum_{k=1}^{K} prob_k * (log prob_k - log target_prob_k)
kd_loss = (probs * (predicted_logits_topk - log_sum_exp_logits_topk - target_logits + log_sum_exp)).sum(-1)
```
  = Σ_{k∈S} q̃_k(log q̃_k − log p̃_k) — **reverse KL restricted to teacher's top-K support, both renormalized on S**. Fixed dataset tokens (teacher-forced); KL over the K-subset "vocab", not sampled tokens.
- **Gradient:** hand-written backward (L682–760): `bwd_kl`: ∂L/∂z_k = q̃_k[(log q̃_k − log p̃_k) − Σ_j q̃_j(log q̃_j − log p̃_j)] at the K ids, exactly the analytic gradient of the restricted reverse KL wrt student top-K logits; zero elsewhere. `fwd_kl`: q̃_k − p̃_k. Teacher = dataset constants.
- **Usage:** default + documented tutorial use `fwd_kl` (docs/user-guide/knowledge-distillation.rst L252, L332). **`bwd_kl` EXISTS (fully implemented incl. analytic gradient) but no shipped config/tutorial invokes it → EXISTS-BUT-UNUSED.** Whether Minitron/Nemotron production distillations ran this path or the modelopt Megatron plugin: INDETERMINATE from these repos.

### 6c. NeMo-RL (NVIDIA-NeMo/RL) — main @ `36b5999a725a` (2026-08-19)
Two coexisting distillation paths, added independently:
- **`nemo_rl/algorithms/distillation.py` + `loss/loss_functions.py` `DistillationLossFn`** — top-k **distribution-level** logit matching on student rollouts (imports GRPO rollout/refit machinery; commit 2025-09-29 "feat: add on policy distillation algorithm" #1006 — predates MiMo/Cascade 2/Ultra; Qwen3-style). `kl_type` ∈ {"forward", "reverse", "mixed"}; operates on `student_topk_logprobs`/`teacher_topk_logprobs`, optional `zero_outside_topk` correction (`log_infinitesimal = -100`). Defaults: `examples/configs/distillation_math.yaml` → `kl_type: "mixed"`, `mixed_kl_weight: 0.5`. **Shipped recipe `examples/configs/recipes/llm/distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1.yaml` sets `kl_type: reverse` → top-k distribution-level RKL USED by a shipped config** — unlike the exists-but-unused pattern in §6a/§6b/torchtune. NB: this is the estimator family Ultra §3.3.5 reports as failed *for MOPD*; NeMo-RL ships it alongside OPD as a separate algorithm.
- **`opd.py` + `advantage_estimator.py` `OPDAdvantageEstimator`** — sampled-token RKL advantage (MOPD); merged from the ultra-v3 drop via #2780 (2026-06-25), details in notes/D §1.

---

## Leads not verified
- Tinker `importance_sampling` kernel closed-source; formula from official docs page. Whether blog experiments used exactly repo defaults: unverified.
- Google original GKD code: no official release found (not exhaustively provable).
- MiniLLM transformers fork: pinned HEAD is a later rebase; paper-era fork commit not identified.
- distillm `srkl-aesop` reading: static inspection only.
- NVIDIA Minitron/Nemotron production runs: which OSS path/loss actually used — indeterminate from repos.
- TRL GKD relocation commit not traced (shallow clone). TRL also ships experimental `gold`, `iw_opd`, `distillation` trainers — out of scope.
