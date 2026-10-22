# Fu et al. 2026 — Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes

**arXiv 2603.25562v2** (26 Mar 2026, last revised 27 Apr 2026). Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, Dongbin Zhao (CASIA SKL-MAIS / UCAS School of AI; Fudan; independent). Sources: full v2 PDF (`curl https://arxiv.org/pdf/2603.25562`, pypdf extraction, 24 pages incl. Appendices A–H) and `https://arxiv.org/html/2603.25562` for equation LaTeX, fetched 2026-09-14. Async-OPD sweep 2026-09-14.

```bibtex
@article{fu2026revisiting,
  title         = {Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes},
  author        = {Yuqian Fu and Haohuan Huang and Kaiwen Jiang and Jiacai Liu and Zhuo Jiang and Yuanheng Zhu and Dongbin Zhao},
  journal       = {arXiv preprint arXiv:2603.25562},
  year          = {2026},
  eprint        = {2603.25562},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2603.25562}
}
```

## Tier

- **D (weak)** — two quotable statements that touch the behavior-vs-current-policy gap: the Appendix A "Training–inference mismatch" limitation, and the EMA-PG sampling-policy correction factor `s_{i,t}=sg(pi_theta/pi_infer)` in Appendix G.3.1. Both are about the *rollout engine* (vLLM top-p) vs the training engine, not about a student checkpoint that has since moved. Nothing else.
- **Not A / B / C / E / F / G.** No async pipeline, no replay buffer, no stale rollouts, no sync-vs-async numbers, no explanation of staleness. The paper is about the *estimator* (sampled-token vs teacher-top-K truncated reverse KL) under a fully synchronous trainer.
- **Correction to the task brief:** this paper has **no "off-policy cold start" fix**. The string "cold" does not occur anywhere in v2. Its three fixes are (i) teacher top-K local support matching, (ii) top-p rollout sampling, (iii) special-token masking (§3.1–3.2). "Off-policy" appears in this paper only as a synonym for *teacher-generated SFT data* (§C: "rely on off-policy supervision from fixed datasets"), i.e. out of scope per the sweep rule. The nearest thing to an "off-policy cold start" in this cluster is TrOPD's cosine-annealed teacher-prefix guidance (arXiv 2606.01249 §4.3), not this paper.
- **Value to the sweep is as the parent codebase of Zheng & Jiang 2026 (Blockwise Policy-Drift Gating, arXiv 2606.24084)**, which reuses `revisiting_opd` with "two PPO epochs". See "Async mechanism" for what this paper itself states about updates per rollout batch.

## Setting

- Students: Qwen2.5-7B-Instruct (math §4.2, ALFWorld §4.3); Qwen2.5-1.5B-Instruct (WebShop §G.1).
- Teachers, all **fixed**: OpenThinker3-7B (math; "an SFT model built on Qwen2.5-7B-Instruct", §2.2); GiGPO-Qwen2.5-7B-It-ALFWorld; GiGPO-Qwen2.5-1.5B-It-WebShop (Table A1).
- Domain/benchmarks: DAPO-Math-17k training; AIME24 (avg@32), MATH500, AIME24/25, Minerva, OlympiadBench (pass@1); ALFWorld; WebShop.
- Framework: "We build our training pipeline on verl (Sheng et al., 2025) and the verl-agent framework (Feng et al., 2025), and conduct all experiments on a node with 8 NVIDIA H100 GPUs." (§F). Code and blog links given on p.1.

## Async mechanism

**None by design.** The student trains on rollouts from its own current inference policy at every step; there is no buffer, no pipelined generator, no stale checkpoint. Two residual behavior-vs-current effects are present and are the only staleness-adjacent content:

1. **Rollout-engine vs training-engine mismatch** (acknowledged, uncorrected in the main method). Appendix A: "Prefixes are generated under a rollout policy, e.g., top-p sampling in the vLLM engine, while the training engine updates the model without correcting for this sampling process. This may introduce a training–inference mismatch (Liu et al., 2025; Yao et al., 2025), which we leave to ongoing work." Note the rollout-level expectation in Eq. (8) / Eq. A7.Ex24 is written over `pi_{theta,infer}` / `pi_infer`, not `pi_theta` — so the paper is explicit that the sampling policy is a *different* distribution from the trained one, but supplies no correction in Variants 1–3.
2. **Mini-batching within a rollout batch.** Table A1 reports "Batch size 128 / Mini batch size 64" for all three settings, i.e. **2 mini-batch gradient updates per rollout batch** in verl terms; on the second mini-batch the current student is one optimizer step away from the behavior student that sampled the tokens. *(Inference, flagged: the paper never states `ppo_epochs`, never uses the words "epoch", "mini-batch update count", "old policy", "stale" or "behavior policy", and reports no `pi_old` ratio anywhere. The `128/64` reading is the standard verl semantics, not a claim made in the text.)* Zheng & Jiang 2026 §4 reuse this codebase and state "two PPO epochs" with "train batch size of 4"; that is **their** configuration, not one stated here.
- Staleness measured: not at all. No `pi_old` log-ratio, no drift metric, no off-policyness statistic. The only "clipping" diagnostic plotted (Figure A6c, "Response Clip Ratio") is the fraction of rollouts truncated at the 16384-token length cap, **not** a PPO ratio-clip fraction.
- Range of staleness studied: none (single configuration, no ablation).

## Objective

Reverse KL throughout; conventions match ours (reverse KL = KL(student ‖ teacher), student = `pi_theta`, teacher = `q`).

- Sequence-level target (§2.1, unnumbered):
  `J_{\mathrm{OPD}}(\theta)=\mathbb{E}_{x\sim D}\left[D_{\mathrm{KL}}\left(\pi_{\theta}(\cdot\mid x)\,\|\,q(\cdot\mid x)\right)\right],`
- Baseline = **`rkl_pg`** (sampled-token score-function estimator), Eq. (2):
  `\hat{g}_{\mathrm{tok}}=\sum_{t=1}^{T}r_{t}s_{t}.`  with `s_{t}=\nabla_{\theta}\log\pi_{\theta}(y_{t}\mid c_{t}),\qquad r_{t}=\log\frac{\pi_{\theta}(y_{t}\mid c_{t})}{q(y_{t}\mid c_{t})}` (§2.1). Discounted interpolation Eq. (3) `\hat{g}_{\gamma}=\sum_{t=1}^{T}\left(\sum_{t^{\prime}=t}^{T}\gamma^{t^{\prime}-t}r_{t^{\prime}}\right)s_{t}` with gamma=0 recovering token-level and gamma=1 the causal sequence-level estimator.
- Proposed method = **`teacher-top-k`** truncated reverse KL with renormalization inside the support ("local support matching", LSM), K=32. Support Eq. (6) `S(c_{i,t})=\mathrm{TopK}_{q}(c_{i,t})`; renormalization Eq. (7); objective Eq. (8):
  `\mathcal{L}_{\mathrm{LSM}}=\mathbb{E}_{x,\,\{o_{i}\}\sim\pi_{\theta,\mathrm{infer}}}\left[\frac{1}{\sum_{i=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i}|}\sum_{v\in S(c_{i,t})}\hat{\pi}_{\theta}(v\mid c_{i,t})\log\frac{\hat{\pi}_{\theta}(v\mid c_{i,t})}{\hat{q}(v\mid c_{i,t})}\right].`
- Full-vocab reference Eq. (4) `\mathcal{L}_{\mathrm{full}}(c_{t})=\sum_{v\in\mathcal{V}}\pi_{\theta}(v\mid c_{t})\log\frac{\pi_{\theta}(v\mid c_{t})}{q(v\mid c_{t})}` is stated but never trained (memory).
- §G.3.1 defines five support variants: teacher-top-K renorm (default), student-top-K+sampled renorm, teacher-top-K+sampled renorm, and two EMA-PG variants (below).

## Off-policy correction

- **Main method (Variants 1–3): none.** No importance ratio, no clipping, no masking of stale tokens. Renormalization inside the truncated support is a *bias* fix, not an off-policy fix: "we apply a separate softmax over the logits inside the support set, so gradients do not directly propagate to tokens outside this set" (§3.2).
- **Only IS-like term in the paper** is the EMA-PG tail correction of Variants 4–5 (§G.3.1), imported from Zhang & Ba 2026:
  `r_{i,t}=\frac{\pi_{\theta}(y_{i,t}\mid c_{i,t})}{\operatorname{sg}\!\left(\pi_{\theta}(y_{i,t}\mid c_{i,t})\right)},\qquad s_{i,t}=\operatorname{sg}\!\left(\frac{\pi_{\theta}(y_{i,t}\mid c_{i,t})}{\pi_{\mathrm{infer}}(y_{i,t}\mid c_{i,t})}\right),`
  `\ell_{\mathrm{tail}}(c_{i,t},y_{i,t};\mathcal{S})=s_{i,t}\,\mathbf{1}\!\left[y_{i,t}\notin\mathcal{S}(c_{i,t})\right]\,r_{i,t}\,\operatorname{sg}\!\left(\log\frac{\pi_{\theta}(y_{i,t}\mid c_{i,t})}{q(y_{i,t}\mid c_{i,t})}\right).`
  with `\ell_{\mathrm{head}}(c_{i,t};\mathcal{S})=\sum_{v\in\mathcal{S}(c_{i,t})}\pi_{\theta}(v\mid c_{i,t})\log\frac{\pi_{\theta}(v\mid c_{i,t})}{q(v\mid c_{i,t})}` and `\ell_{\mathrm{EMA}}=\ell_{\mathrm{head}}+\ell_{\mathrm{tail}}`. Text: "where `s_{i,t}` is the sampling-policy correction (optionally clipped, as in EMA-PG)" (§G.3.1). The clip value is **not given**. This is a current-vs-sampling-policy ratio, exactly the quantity that becomes non-trivial under staleness — but it is applied only to the out-of-support tail, and only in the two ablation variants.
- Empirically the corrected variants are **worse**: "the EMA-PG variants, despite their unbiasedness motivation, do not translate into better empirical performance here" (§G.3.2).

## Results: async vs sync

**None.** No async or stale-rollout condition exists; every row is synchronous on-policy. The only comparison bearing on off-policy correction is the support-variant ablation, Table A3 (single-task, Qwen2.5-7B-Instruct <- OpenThinker3-7B, 400 steps; `avg@32` and `pass@1`):

| Variant (§G.3.1) | AIME24 avg@32 | MATH500 avg@32 | Avg. pass@1 (5 math) |
|---|---|---|---|
| teacher top-K (default, no correction) | 23.6 | 80.4 | 41.7 |
| student top-K + sampled token (renorm) | 22.3 | 82.4 | 41.9 |
| teacher top-K + sampled token (renorm) | 22.4 | 81.6 | 42.9 |
| student top-K + sampled token (EMA-PG, IS-corrected tail) | 21.1 | 82.0 | 41.2 |
| teacher top-K + sampled token (EMA-PG, IS-corrected tail) | 20.7 | 81.4 | 38.1 |

Multi-task (Table A3b, avg pass@1 over 5 math benchmarks): teacher top-K 41.7; student top-K renorm 28.4; teacher top-K renorm 26.9; student top-K EMA-PG 36.2; teacher top-K EMA-PG 33.7 (ALFWorld 95.3 / 95.3 / 94.5 / 93.8 / 96.1).

Headline method result (not staleness-related), Table 2/3 and Abstract: "+19.8% performance gain over standard sampled-token OPD baselines". WebShop (Table A2): sampled-token OPD 73.0 task score / 50.0 success → LSM 75.1 / 57.8.

Throughput: not reported anywhere.

## Explanation offered

No staleness explanation exists. The paper's mechanism claims are about *support mismatch and estimator variance*, and are worth recording because they are the closest cousins of the staleness argument:

- Variance of the MC estimator vs horizon (§2.1): "Under bounded rewards and bounded gradients, the worst-case variance upper bound of token-level OPD scales as O(T^2), whereas the sequence-level estimator scales as O(T^4)." and "stronger future coupling leads to substantially higher gradient variance and less stable optimization" (Figure 1).
- Support/reliability drift along a rollout (§2.2): "This proxy becomes unreliable on prefixes that are common under the student but uncommon for the teacher. In such regions, tokens with high teacher probability can remain rewarded even after the trajectory has drifted into repetition, self-resetting reasoning, or other meaningless continuations." and "We hypothesize that two factors amplify this issue: sharp teacher distributions, where modest teacher–student mismatch can produce large log-ratio values, and growing teacher–student divergence along long rollouts. Consistent with this view, Figure 4 shows that the distribution of teacher–student gaps becomes wider later in the sequence."
- Why distribution-level beats sampled-token (§3.1): "this replaces a one-token point estimate with a distribution-level comparison over teacher-supported candidates at the same prefix. The update is therefore no longer determined by the sign and magnitude of a single sampled-token log-ratio". *(This is a same-prefix variance/robustness argument, not a staleness argument; it does not mention who sampled the prefix.)*
- Why unconstrained sampling is avoided (§3.2): "Unconstrained sampling occasionally produces very low-probability tokens, creating prefixes on which the teacher signal becomes less informative and optimization less stable. Top-p sampling keeps trajectories closer to typical continuations and makes the teacher signal more reliable."

## Quotes

Every staleness/off-policy statement in the paper, in order:

- §1: "Recent public reports ... suggest a broader shift toward supervision on model-generated trajectories, or closely related on-policy variants, alongside off-policy distillation and reinforcement learning."
- §4.4: "we ... discuss possible causes, including support-set construction and remaining off-policy effects, in Appendix A."
- §A (Training–inference mismatch): "Prefixes are generated under a rollout policy, e.g., top-p sampling in the vLLM engine, while the training engine updates the model without correcting for this sampling process. This may introduce a training–inference mismatch (Liu et al., 2025; Yao et al., 2025), which we leave to ongoing work."
- §A (Teacher matching remains an imperfect proxy): "Closing that gap may require stronger rollout control, better handling of distribution shift, better use of teacher uncertainty, and combinations with outcome-verifiable rewards."
- §B (Continual learning as a testbed): "that regime would also stress exactly the issues surfaced in this paper: distribution shift, teacher staleness, and the accumulation of approximation error over long adaptation horizons." — note: **"teacher staleness" here means the fixed teacher becoming out of date as the student moves to new domains in continual learning**, not stale student rollouts. Same paragraph: "whether teacher-based objectives remain useful once the student keeps moving away from the teacher's original domain."
- §B (Relation to other stabilization directions): "This work is complementary to directions such as reward-hacking mitigation, EMA-anchor stabilization with top-K KL (Zhang & Ba, 2026), perturbation-based off-policy correction (Ye et al., 2026a), and logit-level fusion between teacher and student rollouts (Zhang et al., 2026). These methods address different parts of the same broader problem: how to keep teacher-derived signals useful once teacher and student policies begin to diverge (Li et al., 2026)."
- §C: "Many widely used post-training methods for language models rely on off-policy supervision from fixed datasets, including supervised fine-tuning on demonstration data ... and conventional knowledge distillation on teacher-provided targets" — i.e. "off-policy" = teacher SFT data, the sweep's out-of-scope sense.
- §G.3.1: "where `s_{i,t}` is the sampling-policy correction (optionally clipped, as in EMA-PG)."

## Not in the source

- No `ppo_epochs` / updates-per-rollout-batch statement; only "Batch size 128 / Mini batch size 64" (Table A1). Whether `pi_old` is refreshed per mini-batch, and whether the `pi_infer` in Eq. (8) is a cached rollout log-prob or a recomputed actor log-prob, are both unstated.
- No stale-rollout, replay-buffer, or asynchronous-generator experiment; no sync-vs-async baseline; no off-policyness metric; no throughput/wall-clock numbers.
- The EMA-PG sampling-policy correction `s_{i,t}` has **no clip value given** and is never applied to the default method.
- No "off-policy cold start" fix, and no teacher-prefix / mixed-prefix condition of any kind (contra the task brief).
- No ablation on top-p vs the IS correction that would isolate the training–inference mismatch it acknowledges.
- Single seed; no confidence intervals on Tables 2–4 / A2–A3; AIME24/25 are 30-problem benchmarks so pass@1 steps are 3.3 points.
