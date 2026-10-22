# Zheng & Jiang 2026 — Blockwise Policy-Drift Gating for On-Policy Distillation

**arXiv 2606.24084v1** (23 Jun 2026). Liwen Zheng, Haiyun Jiang (both "Independent Researcher"). Sources: full PDF (`curl https://arxiv.org/pdf/2606.24084`, pypdf extraction, 8 pages, no appendix) and arXiv HTML for equation LaTeX, fetched 2026-09-14. Async-OPD sweep 2026-09-14.

```bibtex
@article{zheng2026blockwise,
  title         = {Blockwise Policy-Drift Gating for On-Policy Distillation},
  author        = {Liwen Zheng and Haiyun Jiang},
  journal       = {arXiv preprint arXiv:2606.24084},
  year          = {2026},
  eprint        = {2606.24084},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.24084}
}
```

## Tier

- **C** — side-by-side numbers for gated vs ungated reused-rollout OPD (Table 1) but NO fully-on-policy (single-epoch) baseline and no explanation beyond a "plausible explanation" paragraph (Sec. 6).
- **D** — quotable statements on rollout reuse creating an old-current (behavior vs current student) mismatch even in synchronous OPD (Sec. 1, Sec. 2 "Rollout reuse and freshness-aware OPD", Sec. 6); it also restates AsyncOPD's KL-direction claim (Sec. 2).
- **F (weak)** — the staleness-mitigation rule is the paper's main point, but the "staleness" is only 2-PPO-epoch reuse inside a synchronous pipeline; the authors explicitly disclaim async execution (Sec. 6: "Our experiments do not test asynchronous execution, sample buffering, or refresh policies").
- Not A: no argument that distribution-level RKL tolerates staleness better than sampled-token; the gate is applied identically to sampled-token OPD and Teacher-TopK/LSM and helps both (Table 1), with no direction-dependent analysis. Not B: no mechanism-level explanation of why drift hurts. Not E/G.

## Setting

- Student: Qwen3-1.7B-Base. Teacher: "Qwen3-4B-Base-GRPO", "a post-RL Qwen3-4B-Base checkpoint trained with GRPO for mathematical reasoning [24]" (Sec. 4 "Models and codebase"). Teacher fixed: yes ("It does not change teacher targets, teacher top-K supports, or the rollout policy", Abstract; "Teacher signals are computed online on the student-sampled rollout prefixes during OPD training", Sec. 4).
- Codebase: "based on the public `revisiting_opd` codebase with local policy-drift-gating modifications" (Sec. 4) — i.e. Fu et al. 2026 (arXiv 2603.25562, ref [10]).
- Data: DAPO-Math-17k prompts, "1,791,700 math rows; this is a row count from the released parquet table, not a count of unique prompts" (Sec. 4).
- Training config (Sec. 4): max prompt 2048, max response 16384, rollout group size 8, top-p 0.9, temperature 1.0, Teacher-TopK K=32 for LSM rows, lr 2e-6, "two PPO epochs", "train batch size of 4", one A800 80GB per run, fixed 200 steps for every variant.
- Eval (Sec. 4 "Evaluation"): vLLM, n=8, temp 1.0, top-p 0.9, 16384 max tokens; MATH500, AIME24, AIME25 (MathArena), AMC23 (40 problems). Primary metric pass@8; avg@8 auxiliary.
- Six trained variants (Sec. 4 "Main matrix"): {sampled-token OPD, Teacher-TopK/LSM} x {no gate, Block64 gate, NewlineSpan gate}.

## Async mechanism

- What is stale: within a synchronous PPO-style loop, the same rollout batch is optimized for two epochs, so on the second pass the *current* student pi_theta differs from the *behavior* student pi_old that sampled the tokens. "In practical PPO-style training, the same student rollout may be reused across multiple optimization epochs. During reuse, the current student can assign different probabilities to the same response tokens than the behavior student that generated them. This old-current shift is not a teacher-reliability signal, but it can indicate how stale a piece of rollout data is for the current update." (Sec. 1). "Even in a synchronous OPD pipeline, this creates a local old-current distinction during rollout reuse." (Sec. 2).
- Reuse setup: "two PPO epochs" over a batch of 4 prompts x 8 rollouts (Sec. 4). Mini-batch size and the number of gradient updates per rollout batch are NOT stated; only ">= 2 passes" is implied.
- How staleness is measured: per-token sampled-path log-ratio, Eq. 1 (Sec. 3):
  `d_{i,t}=\log\pi_{\theta}(y_{i,t}\mid x_{i},y_{i,<t})-\log\pi_{\mathrm{old}}(y_{i,t}\mid x_{i},y_{i,<t}).`
  "We use 'policy drift' as shorthand for this sampled-path old-current shift; it is not a full distributional KL estimate." "In the implementation used here, log pi_old is recomputed by the actor before policy update, so the ratio is based on raw actor log-probabilities rather than top-p-renormalized rollout probabilities. The teacher model is not used in this ratio." (Sec. 3). Inference (not stated): since pi_old is recomputed by the actor right before the update, d=0 and the gate is uniform on the first epoch; the gate only acts on the second epoch.
- Range studied: a single staleness level (2 epochs). No sweep over epochs, no 1-epoch (fully on-policy) control, no measured drift statistics ("This v1 also focuses on downstream benchmark results rather than gate-distribution diagnostics such as log-ratio percentiles, effective token ratios, or gradient-norm tails", Sec. 7).

## Objective

- Two base losses, both inherited unchanged from `revisiting_opd` and written only abstractly as `\ell^{\mathrm{base}}_{i,t}` with response mask `m_{i,t}` (Sec. 2 "Sampled-token OPD and local support matching"). The paper never writes out either base loss or names its KL direction.
  - "sampled-token OPD" — the standard Thinking-Machines-style sampled-token estimator; in our vocabulary `rkl_pg` (per-token teacher-minus-student log-prob on the sampled token), assuming the `revisiting_opd` default. Flag: direction not stated in this paper; only the related-work paragraph mentions "pure reverse-KL imitation" (Sec. 2, describing [14]).
  - "Teacher-TopK/LSM" — `teacher-top-k` (K=32) local support matching from Fu et al. [10]: "replaces a one-token teacher signal with a truncated teacher support at each visited prefix" (Sec. 2). "In our LSM runs, the teacher top-K set, teacher probabilities, and normalization are unchanged by policy-drift gating." (Sec. 2).
- Gated objective, Eq. 4 (Sec. 3):
  `\mathcal{L}_{\mathrm{gate}}=\frac{\sum_{i,t}m_{i,t}\tilde{g}_{i,t}\ell^{\mathrm{base}}_{i,t}}{\sum_{i,t}m_{i,t}+\epsilon}.`
  "The denominator remains the original response-mask denominator. Detachment prevents the gate from becoming an additional objective on the current policy; it only controls how much each base OPD position loss contributes." (Sec. 3).

## Off-policy correction

- No importance ratio, no clipping, no masking. Instead a detached, sign-symmetric soft gate on aggregated drift. Eq. 2 (Sec. 3): `g=\exp(-\tau|s|)` with `\tau=1`, where s is the block/span mean of `d_{i,t}` (signed). Eq. 3 (Sec. 3):
  `\bar{g}=\frac{\sum_{i,t}m_{i,t}g_{i,t}}{\sum_{i,t}m_{i,t}+\epsilon},\qquad\tilde{g}_{i,t}=\operatorname{stopgrad}\left(\frac{g_{i,t}}{\bar{g}+\epsilon}\right).`
  "The normalized gate can be larger than one. Its purpose is to preserve the average loss scale over valid response tokens while changing the relative emphasis of positions with different old-current shifts." (Sec. 3).
- Granularity (Sec. 3 "Granularity"): "Block64-Gate: split responses into fixed 64-token blocks, average d_{i,t} inside each block, and broadcast the gate to tokens in the block." "NewlineSpan-Gate: split responses using newline delimiters, merge short spans, split long spans, and fall back to fixed chunks when no delimiter is present."
- Explicitly NOT an IS correction: "the method is a policy-drift-aware weighting rule, not an unbiased off-policy estimator." (Sec. 7). Token-level `exp(d_t)` IS weights are named as a natural but noisy baseline (Sec. 1) and a clipped Token-IS control is deferred to future work (Sec. 7, Sec. 8). Caveat noted by authors: "The gate score for a block averages signed log-probability shifts. Opposite-sign token shifts can therefore cancel within a block" (Sec. 7).

## Results: async vs sync

No fully on-policy (1-epoch) baseline exists; every row uses 2-epoch reuse. The comparison is ungated-reuse vs gated-reuse. Table 1 (pass@8, %, 200 steps, single seed):

| Method | MATH500 | AIME24 | AIME25 | AMC23 | Avg. |
|---|---|---|---|---|---|
| Base student (Qwen3-1.7B-Base) | 77.2 | 13.3 | 10.0 | 60.0 | 40.1 |
| Sampled-token OPD (ungated, 2 epochs) | 85.8 | 20.0 | 13.3 | 80.0 | 49.8 |
| + Block64 gate | 86.4 | 20.0 | 20.0 | 80.0 | 51.6 |
| + NewlineSpan gate | 86.0 | 26.7 | 16.7 | 75.0 | 51.1 |
| Teacher-TopK/LSM (ungated, 2 epochs) | 86.0 | 23.3 | 20.0 | 70.0 | 49.8 |
| + Block64 gate | 86.4 | 20.0 | 26.7 | 80.0 | 53.3 |
| + NewlineSpan gate | 86.2 | 20.0 | 26.7 | 80.0 | 53.2 |
| Teacher reference (Qwen3-4B-Base-GRPO) | 90.8 | 46.7 | 33.3 | 90.0 | 65.2 |

- Abstract states the OPD gain as mean pass@8 "0.4978 to 0.5160". Auxiliary avg@8 for sampled-token OPD: "Block64 improves mean avg@8 from 0.3157 to 0.3231" (Sec. 5); avg@8 for the other rows is not reported.
- Throughput/wall-clock: not reported.
- Scale caveat: AIME24/25 have 30 problems, so 20.0 -> 26.7 is two problems; single seed (Sec. 7).

## Explanation offered

- Why reused tokens should be down-weighted (heuristic, no theory): "it is an orthogonal weighting mechanism that asks whether reused rollout tokens should contribute equally when the current student has moved away from the behavior student." (Sec. 6). "This old-current shift is not a teacher-reliability signal, but it can indicate how stale a piece of rollout data is for the current update." (Sec. 1).
- Why sampled-token OPD is fragile in the first place (support/narrow-view mechanism, from [10]): "a sampled token can be a narrow view of the teacher distribution, and student-generated prefixes can move away from regions where the teacher provides reliable local guidance." (Sec. 1).
- Why block granularity: "A plausible explanation for the Block64 advantage is that old-current student drift is locally coherent but tokenwise noisy. Individual token log-ratio shifts can be dominated by lexical, formatting, or tokenizer-level effects, whereas response-level averages wash out stale regions in long reasoning traces. Fixed 64-token blocks provide a middle scale: they smooth token-level spikes while preserving enough locality to identify reasoning segments whose behavior-policy likelihood has changed under rollout reuse." (Sec. 6). "Token-level importance-ratio weights based on exp(d_t) are a natural baseline, but they can be noisy. Sequence-level weighting is smoother but may be too coarse for long reasoning traces." (Sec. 1).
- What the gain is: "The auxiliary avg@8/pass@8 comparison further suggests that blockwise gating mainly improves sample quality or solve-rate robustness rather than uniformly expanding rare-sample coverage" (Sec. 6).
- Direction-dependence (secondhand, attributed to AsyncOPD [23], not tested here): "AsyncOPD studies how stale data affects OPD under asynchronous execution and shows that KL direction changes the stale-data problem: teacher-weighted forward KL is more robust to stale rollouts, whereas student-weighted reverse KL is more vulnerable [23]." (Sec. 2). Mechanism identified in this paper: distribution shift between behavior and current student (student-weighting of the estimator), handled by reweighting rather than by changing the objective.

## Quotes

- Abstract: "This paper introduces blockwise policy-drift gating, a lightweight student-only old-current drift controller for OPD under rollout reuse."
- Abstract: "The results identify local old-current policy drift as a practical control signal for reused OPD rollouts and motivate block-level gating as a simple default for improving solve-rate robustness."
- Sec. 1: "In practical PPO-style training, the same student rollout may be reused across multiple optimization epochs. During reuse, the current student can assign different probabilities to the same response tokens than the behavior student that generated them."
- Sec. 1: "The method is best understood as a detached loss weighting rule: it changes which reused rollout positions dominate optimization, not the teacher supervision, the teacher support, or the rollout distribution."
- Sec. 2: "PPO-style optimization commonly performs multiple minibatch or epoch updates over data sampled by a behavior policy [3]. Even in a synchronous OPD pipeline, this creates a local old-current distinction during rollout reuse. Freshness-aware OPD is the closest neighboring line of work. f-OPD studies stale rollouts in asynchronous or delayed-update settings, decomposes objective discrepancy into rollout drift and supervision drift, and introduces sample-level freshness control [22]."
- Sec. 2: "Our setting is narrower and deliberately orthogonal. We do not design an asynchronous training system, a teacher-cache strategy, or a sample refresh policy. Instead, we keep the teacher supervision fixed and study how old-current student log-probability shifts should be aggregated before being used as detached weights on OPD token losses."
- Sec. 6: "Policy-drift gating is not a replacement for teacher-support fixes such as LSM, and it is not a full stale-rollout correction framework."
- Sec. 6: "The empirical advantage of Block64 on sampled-token OPD and on LSM pass@8 supports block-level drift gating as a practical default for this rollout-reuse setting. This is a solve-rate robustness result rather than a claim that any single gate dominates every secondary metric or training budget."
- Sec. 6: "Our experiments do not test asynchronous execution, sample buffering, or refresh policies. They instead isolate a smaller question inside synchronous rollout reuse: whether old-current student log-probability shifts, aggregated locally along the sampled response path, are useful as loss weights for OPD token losses."
- Sec. 7: "The current matrix also does not include granularity and ratio-weighting controls such as Token-Gate, a clipped token-level importance-ratio baseline (Token-IS), or Sequence-Gate. Without those controls, we should not claim that 64-token blocks are universally optimal."
- Sec. 8: "These results show that old-current student drift is a useful signal for OPD under rollout reuse, and that the granularity of this signal should be evaluated primarily through problem-level solve rate."

## Not in the source

- No fully on-policy control (1 PPO epoch) and no sweep over epochs/staleness, so the size of the reuse penalty itself is unknown; only gated-vs-ungated at a fixed 2-epoch reuse.
- No IS-ratio baseline (Token-IS, clipped) — deferred to v2 (Sec. 7/8).
- Base losses never written out; KL direction of "sampled-token OPD" and the exact LSM formula are inherited from `revisiting_opd` [10] without restatement.
- Mini-batch size / number of optimizer steps per rollout batch not stated; whether pi_old is refreshed per mini-batch or per rollout batch not stated (only "recomputed by the actor before policy update").
- No drift diagnostics (log-ratio percentiles, effective token ratio, gradient-norm tails) despite these being the paper's own control signal (Sec. 7).
- Single seed; no confidence intervals; no training curves; no throughput numbers.
- No per-benchmark avg@8 for LSM rows; avg@8 only quoted as a mean for OPD vs OPD+Block64.
- Teacher is a different-size model from the student (4B vs 1.7B), so no same-size teacher/student comparison.
