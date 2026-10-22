# Xin et al. 2026 — Escaping the KL Agreement Trap in On-Policy Distillation (KAT)

**arXiv 2606.09471** (Jun 2026). Haoran Xin, Anhao Zhao, Ying Sun, Jin Li, Xiaoyu Shen, Hui Xiong. Content from arXiv abs + HTML via WebFetch [FETCH 2026-08-18]. Incorporated: lit_review @sec-folklore-crit (=== entry) + estimator-criticism clause in @sec-criticism.

```bibtex
@article{xin2026agreementtrap,
  author  = {Haoran Xin and Anhao Zhao and Ying Sun and Jin Li and Xiaoyu Shen and Hui Xiong},
  title   = {Escaping the {KL} Agreement Trap in On-Policy Distillation},
  journal = {arXiv preprint arXiv:2606.09471},
  year    = {2026}
}
```

## Setting

Eq. 2 [FETCH]: ℒ_OPD(θ) = E_{x, ŷ∼π_θ(·|x)}[Σ_{t=1}^T D_KL(π_θ(·|x,ŷ_{<t}) ‖ π_T(·|x,ŷ_{<t}))] — token-level reverse KL on student rollouts (the frozen-visitation objective of lit_review @sec-deriv).

## The trap

- Abstract [FETCH]: "When the student drifts into an unrecoverable prefix, the teacher may locally agree with the degraded state, producing low reverse KL but little corrective training signal." "We identify this persistent regime as a low-KL agreement trap." "Tokens during and after such traps produce less useful supervision signals."
- Body [FETCH]: "Conditioned on this degraded state, the teacher's token-level distribution may carry little corrective information, instead producing continuations that are plausible under the erroneous prefix and therefore close to the student's distribution."
- Quantitative: "More than 60% of rollouts exhibit at least one extended low-KL segment"; post-trap KL stays below pre-segment level.

## Fix (KAT)

Online termination: sliding-window mean z_t = (1/W)Σ_{i=t−W+1}^t d_i of per-token divergences (Eq. 4); terminate rollout when c ≥ T consecutive windows satisfy z_t < s_k, a dynamic training-adaptive calibrated threshold (Eq. 7). Results: +2.66 avg@k, +3.43 pass@k, −59.73% mean rollout length, four math benchmarks.

## Relevance

Same genus as the paper's compromise latching: teacher's *conditionals* cannot veto a globally bad trajectory — a prefix-coupling pathology invisible to the unconditional folklore picture. They treat it as a data-filtering problem, not a divergence-direction problem: key differentiation point for paper.typ.

## Async / off-policy (sweep 2026-09-14)

Re-read from the **primary source**: PDF `https://arxiv.org/pdf/2606.09471` (curl + pypdf, 13 pages, 2026-09-14). Stamp on p. 1: `arXiv:2606.09471v1 [cs.LG] 8 Jun 2026` — v1 is the only version (abs page, 2026-09-14). The FETCH-based content above checks out against the PDF (abstract quotes, the ">60% of rollouts" figure in §3.1, and the headline "+2.66% avg@k / +3.43% pass@k / −59.73% rollout length" from the abstract). Corrections and additions below.

### Verdict: **out of scope for async. Tier D only.**

Training is strictly on-policy — one optimizer update per rollout batch, with the sampler resynced every step. No stale student data, no replay of rollouts, no mini-batch inner loop, no importance ratio anywhere in the paper.

### (a) Exact hyperparameter statements

- App. C, "Distillation objective": "We run **pure on-policy distillation (λ = 1.0)** with reverse KL and softmax temperature τ = 1.0."
- App. C, "Student rollouts": "**Student weights are synchronized into the vLLM engine after every optimizer step.**" Generation is colocated: "the student generates completions via a colocated vLLM engine (TP = 1, `gpu_memory_utilization` = 0.3 for the 1.7B student and 0.2 for the 4B student) under nucleus sampling with T = 1.0, top-p = 0.95, top-k disabled, and one completion per prompt."
- App. C, "Optimization": "We train the 1.7B student for 3 epochs and the 4B student for 2 epochs. **Per-device train batch size is 1 with 64 gradient-accumulation steps**, yielding an effective batch of 128 sequences with 2 GPUs (1.7B) and 256 sequences with 4 GPUs (4B)." — gradient *accumulation*, i.e. one optimizer step per rollout batch; the "epochs" are prompt epochs over DAPO-Math-17K, each drawing fresh rollouts.
- AdamW β1 = 0.9, β2 = 0.999, weight decay 0, grad clip 1.0, cosine schedule with 3% linear warm-up, peak LR 1e-6, dropout off, 3 seeds averaged (App. C). Hardware: "a single node with 4×NVIDIA A800-SXM4 80 GB GPUs (NVLink)".

### (b) Verbatim statements about stale / off-policy student data

There are none. *stale*, *staleness*, *asynchronous*, *replay*, *old policy*, *behavior policy*, *importance* (as a ratio), *ppo_epochs* do not occur in the text. "off-policy" occurs 3×, always meaning teacher-generated static data:

- §1: "This mitigates the exposure bias that **off-policy distillation** (Kim and Rush, 2016) typically suffers from, while yielding far higher sample efficiency than sparse outcome-reward reinforcement learning."
- §2.2: "The expectation is taken over student-generated trajectories: supervision is dense (every generated token receives a teacher signal) and is **anchored on the student's own state distribution**. This combination is what gives OPD its sample-efficiency advantage over outcome-only RL methods, and its robustness advantage over **off-policy KD that trains only on a static dataset**."
- §6 (Related Work): "On-policy distillation (OPD) trains students on self-generated states with dense teacher supervision, **reducing off-policy train–test mismatch** while improving sample efficiency over sparse reward RL (Agarwal et al., 2024)."

The FIFO buffer in §4.2/App. C is **not** a rollout replay buffer — it stores scalar per-rollout KL minima used to calibrate the termination threshold: "a FIFO buffer of M = 4096 per-rollout minimum-window means is maintained on each worker" (App. C); "The buffer stores the most recent B minimum scores, providing a rolling estimate of the current low-KL scale **under the evolving student policy**" (§4.2). The only thing that goes stale in KAT is that threshold statistic, which is precisely why it is recalibrated by quantile each step.

### (c) Estimator — correction to the section above

The note above labels the OPD objective "Eq. 2"; in the PDF the objective in §2.2 is **unnumbered**, and Eq. (1) is the per-token reverse KL. Eq. (1), §2.1, is explicitly **distribution-level over the full vocabulary**:

```
d_t = D_KL(p_t ‖ q_t) = Σ_{v∈V} p_t(v) log( p_t(v) / q_t(v) )        (1)
```

with `p_t(v) = π_θ(v|h_t)`, `q_t(v) = π_T(v|h_t)`, `h_t = (x, ŷ_<t)`. So KAT is **not** `rkl_pg`; it is a truncated-support distribution-level reverse KL. App. C, "Distillation objective": "To control the cost of the divergence over a ∼150K-token vocabulary, **both distributions are restricted to a truncated top-K support (K = 64) and re-normalized, with an explicit tail bucket carrying the residual probability mass; we find this approximation to match the full-vocabulary loss to within ≈0.5% relative error while reducing the loss-side FLOPs by more than two orders of magnitude.**" Whose top-K (student, teacher, or union) is **not stated** — the paper says only "both distributions are restricted to a truncated top-K support". The tail bucket is a design detail worth borrowing. Alg. 1 line 10 hedges, "Compute (**or estimate**) `d_t = D_KL(p_t ‖ q_t)`", but the implementation section commits to the truncated distribution-level form.

### (d) Other source discrepancy

§5.1 ("Models") says "the students are Qwen3-0.7B and Qwen3-1.6B", but every results table (Tables 1, 2, Fig. 6) and App. C name **Qwen3-1.7B-Base and Qwen3-4B-Base**, and App. C gives their HF URLs. §5.1 is wrong; use 1.7B/4B. Teacher is Qwen3-8B throughout, fixed. Training prompts: DAPO-Math-17K; eval AMC, MATH500, MinervaMath, AIME24.

### Useful for the paper anyway (tier D)

KAT is a clean citation for "published OPD does one update per rollout batch and resyncs the sampler every step", and — via App. C — for "truncated top-K reverse KL tracks the full-vocabulary loss to ≈0.5% at 1/100 the loss-side FLOPs", which is independent support for the distribution-level estimator being cheap enough to be the default.
