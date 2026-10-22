# DeepSeek-V4 — full-vocabulary multi-teacher OPD as the consolidation stage

**arXiv 2606.19348v1** (only version; arXiv submission history: 26 Apr 2026). DeepSeek-AI. Content below extracted verbatim from the arXiv HTML (`https://arxiv.org/html/2606.19348`, downloaded and parsed locally, 2026-08-19; re-verified line-by-line against a fresh HTML + PDF pull 2026-09-14); equations from the MathML `alttext` LaTeX [HTML]. No training code released.

```bibtex
@article{deepseek2026v4,
  title   = {{DeepSeek-V4}: Towards Highly Efficient Million-Token Context Intelligence},
  author  = {{DeepSeek-AI}},
  journal = {arXiv preprint arXiv:2606.19348},
  year    = {2026}
}
```

**§1 claim (`S1.p7.1`) [HTML]:** "a single unified model is trained through on-policy distillation, wherein the unified model acts as the student learning to optimize the reverse KL loss with teacher models."

## §5.1 Post-Training Pipeline (overview) [HTML]

"Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)." — OPD citations resolve (via `bib.bib35`, `bib.bib103`) to the Thinking Machines post (Lu & TML Lab 2025) and MiniLLM (Gu et al. 2024). §5.1.1: specialists built by adapting the V3.2 pipeline (SFT + RL per domain).

## §5.1.2 On-Policy Distillation [HTML]

- **Eq. 29:** ℒ_OPD(θ) = Σ_{i=1}^N w_i · D_KL(π_θ ∥ π_{E_i}), for expert set {π_{E_1},…,π_{E_N}}. "w_i represents the assigned weight for each expert, typically determined by the relative importance of the expert." Values unstated.
- **On-policy [RAW]:** "Computing the reverse KL loss D_KL(π_θ ∥ π_{E_i}) requires sampling training trajectories from the student π_θ to maintain on-policy learning."
- **Scale:** "more than ten teacher models covering various domains are employed to distill a single student model"; consolidation is "via logits-level alignment, practically circumventing the performance degradation often encountered in traditional weight-merging or mixed RL techniques."
- **Rejects the sampled-token advantage family [RAW]:** "prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate at each token position, and reuse RL framework by replacing sg[log (π_{E_i}(y_t|x,y_{<t}) / π_θ(y_t|x,y_{<t}))] (sg represents the stop gradient operation) as the per-token advantage estimate in the policy loss calculation. Although this approach is resource-efficient, it leads to high variance in gradient estimation and often causes training instability."
- **Their choice [RAW]:** "Therefore, we adopt full-vocabulary logit distillation in our OPD. Preserving the complete logit distribution in calculating reverse KL loss yields more stable gradient estimates and ensures faithful distillation of the teachers' knowledge."

## §5.2.2 Efficient Teacher Scheduling for Full-Vocabulary OPD [HTML]

- "naively materializing logits for a vocabulary size |V| > 100k across all teachers is prohibitive" → cache only last-layer teacher hidden states; "passed through the corresponding prediction head module to reconstruct the full logits on the fly" (exact reconstruction, not top-K).
- "we order training samples by teacher index during data dispatching … at most one teacher head resides in device memory at any given time" → each sample is scored by a single domain teacher; the i-sum of Eq. 29 is realized by data routing.
- **Kernel [RAW]:** "the exact KL divergences between teacher and student logits are computed using a specialized TileLang kernel."
- §5.2.3: rollouts come from a "preemptible and fault-tolerant LLM generation service for RL/OPD rollout" (separate inference service; token-granular WAL).

## Estimator-audit reading

Claimed objective is sequence-level (Eq. 29, policy-level KLs); the stated computation is the *integrand*: exact full-vocabulary per-position KL on student-sampled trajectories — GKD's computation on Tinker/Nemotron's data distribution. **Unstated (no loss-level equations, no code):** token aggregation (sum vs length-normalized mean), the gradient path (pathwise-only presumed; no PG/advantage route — that family is explicitly rejected), any train–inference IS correction for the rollout service, sampling temperature, w_i values. Under the pathwise-only reading: value unbiased for Eq. 29 (chain identity), 𝔼[∇] = frozen-visitation gradient Σ_t 𝔼_{s_t}[∇D_KL,t] — the same partial gradient as Tinker/Nemotron/MiMo, minus their sampled-token variance. This is exactly the configuration Nemotron 3 Ultra §3.3.5 reported inferior to its sampled-token estimator (full-vocab RKL on student rollouts) — V4 ships it as the frontier counterexample.

## Cross-references

- Cites TML post + MiniLLM as OPD precedents — not MiMo-V2-Flash (cf. [[mimo-v2-flash]], which the Nemotron code is written against).
- Chronology: after Nemotron 3 Ultra (2606.15007), same month (June 2026).
- Contrast rows: Nemotron 3 Ultra MOPD / Nemotron-Cascade 2 / Tinker (sampled-token advantage, the family V4 rejects); GKD (same integrand, dataset/mixed prefixes).

**Not in the report:** hyperparameters (w_i, temperature, steps), any acknowledgment of the visitation-term / partial-gradient issue, any off-policy correction for OPD.

---

## Async / off-policy (sweep 2026-09-14)

**Sources for this section:** arXiv HTML `https://arxiv.org/html/2606.19348` and PDF `https://arxiv.org/pdf/2606.19348`, curl + local parse, 2026-09-14. Section ids are LaTeXML ids. (Housekeeping: the arXiv abs page dates v1 to 26 Apr 2026 although the identifier is a 2606 one; the "same month (June 2026)" chronology line above is based on the identifier. Not load-bearing for anything in this sweep.)

**Headline: DeepSeek-V4 says nothing about asynchrony or staleness in its OPD stage.** Full-text term counts over the HTML: `off-policy` **0**, `off policy` **0**, `stale`/`staleness` **0**, `importance sampling` **0**, `importance ratio` **0**, `policy lag` **0**, `behavior policy` **0**. `asynchron*` occurs **once** and is about I/O, not rollouts (§5.2.2, quoted below). `replay` occurs twice and both are sandbox command replay (§5.2.5), not experience replay. `clip` occurs once and is QK-Clip in the pre-training optimizer (§3), not policy clipping.

### Tier

**none.**

- Not **D/E/F/G** — the report never states that OPD rollouts come from an earlier student version, so there is no async-OPD usage or statement to record. (E would otherwise apply trivially: this is a production frontier tech report whose *entire* consolidation stage is OPD. It is excluded because the async property is absent, not because the OPD is.)
- Not **B/C** — no async-vs-sync numbers, no staleness sweep, no OPD ablation of any kind.
- Not **A**, but **A-adjacent and worth flagging**: V4 *does* argue that distribution-level (full-vocabulary) reverse KL is better-behaved than the sampled-token estimator — "it leads to high variance in gradient estimation and often causes training instability" vs "Preserving the complete logit distribution … yields more stable gradient estimates" (§5.1.2, `S5.SS1.SSS2.p2.1`). That is an **estimator-variance** argument, made in a purely on-policy setting. It never mentions staleness, lag, or off-policy data, so it does **not** discharge tier A, which requires the argument to be about staleness tolerance. Do not cite it as an A. It is the closest thing in the sweep to the A-direction premise, and is a useful *ingredient* for our own argument — but the staleness step is ours, not DeepSeek's.

### Setting

- **Student:** one unified model per size (DeepSeek-V4-Pro and DeepSeek-V4-Flash). "a single unified model is trained through on-policy distillation" (§1, `S1.p7.1`).
- **Teachers:** "more than ten teacher models covering various domains are employed to distill a single student model" (§5.1.2, `S5.SS1.SSS2.p1.2`), each a domain specialist from §5.1.1 (SFT then GRPO RL, "maintaining hyper-parameters closely aligned with our prior research", `S5.SS1.SSS1.p1.1`). Domains named in §1: "mathematics, coding, agent, and instruction following".
- **Teacher fixed: yes.** Specialists are finished checkpoints; §5.1.2 only forwards through them. Teacher weights are "offloaded to a centralized distributed storage and are loaded on demand during the teacher forward pass" (§5.2.2, `S5.SS2.SSS2.p1.1`) — read-only. In scope.
- **Stage role:** OPD replaces RL outright. "the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation" (§5.1, `S5.SS1.p1.1`).
- **Benchmarks:** §5.3 (standard suite) and §5.4 (Chinese writing, search, white-collar, code agent). None reported per-OPD-configuration.
- **Framework:** in-house, built on the V3.2 stack — "we integrate the same distributed training stack described in Section 3.4 and the rollout engine introduced earlier for efficient auto-regressive sampling" (§5.2, `S5.SS2.p1.1`). No code released.

### Async mechanism

**None stated.** The report describes the rollout path in three places and none of them introduces student-version staleness:

1. **Student sampling is asserted to be on-policy, without qualification** (§5.1.2, `S5.SS1.SSS2.p1.2`): "Computing the reverse KL loss $\text{D}_{\text{KL}}\left(\pi_{\theta}\parallel\pi_{E_{i}}\right)$ requires sampling training trajectories from the student $\pi_{\theta}$ to maintain on-policy learning." No caveat, no lag, no version accounting follows.
2. **The one `asynchronous` in the OPD infrastructure is I/O, not rollouts** (§5.2.2, `S5.SS2.SSS2.p1.1`): "All parameters and hidden state loading/offloading operations proceed asynchronously in the background, without blocking computation on the critical path." This is teacher-weight and hidden-state prefetch overlap. **Do not mine this sentence for an async-OPD claim** — it is the single `asynchron*` hit in the paper and it is about hiding storage latency.
3. **Preemption/WAL is fault tolerance, and the stated correctness concern is length bias, not staleness** (§5.2.3, `S5.SS2.SSS3.p2.1`–`p3.1`): the generation service checkpoints each request token-by-token and resumes decoding from the persisted KV cache and WAL. Verbatim: "Importantly, it is mathematically incorrect to regenerate unfinished requests from scratch, as this introduces length bias. Because shorter responses are more likely to survive interruption, regenerating from scratch makes the model more prone to producing shorter sequences whenever an interruption occurs." The report never says whether the student is updated during a preemption window, so whether a resumed trajectory is mixed-version is **unknowable from the source**. Note the contrast with Kimi K3: K3's partial rollout pauses generation *in order to* let the optimizer run (deliberate staleness); V4's pause is caused by cluster preemption and the design goal is to make resumption *indistinguishable* from uninterrupted generation.
4. **Teacher-side caching introduces no staleness** (§5.2.2): last-layer teacher hidden states are cached in a centralized buffer during the forward pass and re-projected through the teacher head at training time. The teacher is frozen, so a cached teacher activation equals a fresh one.

**Staleness measured:** not measured. **Range studied:** n/a.

### Objective

- **Vocabulary: `rkl`** — full-vocabulary reverse KL directly optimized (not `rkl_pg`, not top-k). Direction matches the sweep convention: $\mathrm{D_{KL}}(\pi_\theta\,\|\,\pi_{E_i})$, student first. No letter deviation.
- **Eq. 29** (§5.1.2, display `S5.E29`), verbatim LaTeX:
  ```latex
  \mathcal{L}_{\text{OPD}}(\theta)=\sum_{i=1}^{N}w_{i}\cdot\text{D}_{\text{KL}}\left(\pi_{\theta}\parallel\pi_{E_{i}}\right).
  ```
- **Exactness is claimed explicitly** (§5.2.2, `S5.SS2.SSS2.p1.1`): "Finally, the exact KL divergences between teacher and student logits are computed using a specialized TileLang kernel, which accelerates the computation and curtails dynamic memory allocation." Combined with on-the-fly full-logit reconstruction from cached hidden states, this is genuine full-vocabulary RKL — no top-k truncation anywhere.
- **The sampled-token family is named and rejected** (§5.1.2, `S5.SS1.SSS2.p2.1`), verbatim: "In handling the above OPD objective, prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate at each token position, and reuse RL framework by replacing $\texttt{sg}\big[\log\frac{\pi_{E_{i}}(y_{t}|x,y_{<t})}{\pi_{\theta}(y_{t}|x,y_{<t})}\big]$ (sg represents the stop gradient operation) as the per-token advantage estimate in the policy loss calculation. Although this approach is resource-efficient, it leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD."
  That rejected estimator is **exactly** Kimi K3 Eq. 15 and GLM-5 Eq. 2 — the three 2026 frontier reports in this sweep split 2–1 on it, with V4 the dissenter.
- **Token aggregation, temperature, $w_i$ values, step count: all unstated** (unchanged from the audit above).

### Off-policy correction

**None — and none is needed under the report's own account**, since it claims pure on-policy sampling. Concretely, the report contains:
- no importance ratio (per-token or per-sequence) anywhere in §5.1.2 or §5.2;
- no PPO-style clipping and no trust region (the single `clip` in the paper is QK-Clip in the pre-training optimizer, §3, unrelated);
- no IcePop-style masking, no `pop(·)` operator, no rollout-version tagging, no staleness-based sample dropping;
- **no train–inference mismatch correction for OPD**, even though the rollouts are produced by a *separate* inference service (§5.2.3) and the training path uses FP4 QAT weights. §5.2.1 (`S5.SS2.SSS1.p3.1`) argues the mismatch away on the *sampling* side rather than correcting it in the loss: "During the inference and rollout phases of RL training, which do not involve backward passes, we directly use native FP4 quantized weights instead of simulated quantization. This ensures that model behavior during sampling is fully consistent with online deployment" — note this aligns rollout with *deployment*, which is not the same as aligning rollout with the *training* forward pass. GLM-5 (Eq. 1's `pop(ρ)`) and Kimi K3 (shared MXFP4/MXFP8 scheme) both address this explicitly; V4 does not.

### Results: async vs sync

**none.** No such comparison exists in the report. There is also no sync-OPD-vs-RL ablation, no full-vocab-vs-sampled-token ablation (the rejection in §5.1.2 is argued, not measured), and no OPD training curve.

### Explanation offered

**On staleness: nothing** (there is no staleness claim to explain).

**On the estimator choice** — the one mechanism argument the report does make, verbatim (§5.1.2, `S5.SS1.SSS2.p2.1`):
> "Although this approach is resource-efficient, it leads to high variance in gradient estimation and often causes training instability."
> "Therefore, we adopt full-vocabulary logit distillation in our OPD. Preserving the complete logit distribution in calculating reverse KL loss yields more stable gradient estimates and ensures faithful distillation of the teachers' knowledge."

Mechanism identified: **variance of the Monte-Carlo (sampled-token) estimator of the per-position KL**, plus a fidelity claim ("faithful distillation of the teachers' knowledge") that reads as a support-coverage argument — one sampled token per position sees only a sliver of the teacher's distribution. No derivation, no variance bound, no measurement. Asserted, not shown.

### Quotes

Every statement in the report bearing on on-policy-ness, async, or rollout provenance, with section ids. (The list is short because the report is silent on the sweep's core question.)

- §1 `S1.p7.1`: "Finally, to integrate these distinct proficiencies, a single unified model is trained through on-policy distillation, wherein the unified model acts as the student learning to optimize the reverse KL loss with teacher models."
- §5.1 `S5.SS1.p1.1`: "Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD; Lu and Lab, 2025; Gu et al., 2024)."
- §5.1.2 `S5.SS1.SSS2.p1.1`: "This is achieved by having the student learn from the output distributions of teacher models on its own generated trajectories."
- §5.1.2 `S5.SS1.SSS2.p1.2`: "Computing the reverse KL loss $\text{D}_{\text{KL}}\left(\pi_{\theta}\parallel\pi_{E_{i}}\right)$ requires sampling training trajectories from the student $\pi_{\theta}$ to maintain on-policy learning." ← the only on-policy-ness claim, and it is unqualified.
- §5.1.2 `S5.SS1.SSS2.p2.1`: "Although this approach is resource-efficient, it leads to high variance in gradient estimation and often causes training instability."
- §5.2 `S5.SS2.p1.1`: "These designs enable efficient execution of ultra-long-context RL and OPD merging tasks involving over ten distinct teacher models, thereby substantially accelerating the iteration cycle for model releases."
- §5.2.1 `S5.SS2.SSS1.p3.1`: "During the inference and rollout phases of RL training, which do not involve backward passes, we directly use native FP4 quantized weights instead of simulated quantization. This ensures that model behavior during sampling is fully consistent with online deployment…"
- §5.2.2 `S5.SS2.SSS2.p1.1`: "All parameters and hidden state loading/offloading operations proceed asynchronously in the background, without blocking computation on the critical path." ← the only `asynchron*` in the paper; **I/O, not rollouts**.
- §5.2.3 `S5.SS2.SSS3.p1.1`: "To maximize GPU resource utilization while enabling rapid hardware provisioning for high-priority tasks, our GPU cluster employs a cluster-wide preemptive task scheduler, where any running task may be preempted at any time. … To this end, we implement a preemptible and fault-tolerant LLM generation service for RL/OPD rollout."
- §5.2.3 `S5.SS2.SSS3.p2.1`: "During preemption, we pause the inference engine and save the KV cache of unfinished requests. Upon resumption, we use the persisted WALs and saved KV cache to continue decoding."
- §5.2.3 `S5.SS2.SSS3.p3.1`: "Importantly, it is mathematically incorrect to regenerate unfinished requests from scratch, as this introduces length bias. Because shorter responses are more likely to survive interruption, regenerating from scratch makes the model more prone to producing shorter sequences whenever an interruption occurs."
- §5.2.4 `S5.SS2.SSS4.p1.1`: "We introduce targeted optimizations for efficient RL and OPD on million-token sequences. During the rollout phase, we adopt a preemptible and fault-tolerant rollout service, detailed in Section 5.2.3."

### Not in the source

- **Whether the OPD rollouts are ever stale.** The report asserts on-policy sampling and then never revisits it. In particular: no statement on whether the student is updated while a preempted rollout is paused (§5.2.3), no weight-sync cadence between trainer and the generation service, no version tagging. Given a separate generation service (§5.2.3) and a preemptive cluster, *some* lag is plausible — but the report gives no basis to assert it, and this note must not.
- **No sync/async comparison, no staleness sweep, no lag metric** → V4 contributes no async data point. Its value to this review is the opposite polarity: a frontier-scale **existence proof for `rkl` (full-vocab reverse KL) as the shipped OPD objective**, plus an on-record argument against the `rkl_pg` family on variance grounds.
- **No off-policy correction of any kind** — so if one wanted to run V4's objective asynchronously, the report offers no recipe, and its stated rationale (exact, low-variance gradients from student-visited states) is silent about what happens when those states are visited by a stale student.
- **No loss-level equation** beyond Eq. 29's policy-level KL: token aggregation, the gradient path, sampling temperature, and $w_i$ remain unstated (as recorded in the "Estimator-audit reading" above).
- **No acknowledgment of the visitation-term / partial-gradient issue**, i.e. that differentiating only the integrand on frozen student-visited states is not the full gradient of Eq. 29. This matters here more than for the `rkl_pg` papers, because V4's claim of exactness ("the exact KL divergences") is about the *per-position integrand*, not about the sequence-level objective it writes down.
