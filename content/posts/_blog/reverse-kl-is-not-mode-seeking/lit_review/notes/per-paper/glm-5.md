# GLM-5-Team 2026 — GLM-5: From Vibe Coding to Agentic Engineering

**arXiv 2602.15763v2** (v1 17 Feb 2026, v2 24 Feb 2026; v2 current as of 2026-09-14). GLM-5-Team (arXiv author field; first author Aohan Zeng). Affiliation as printed: "Zhipu AI & Tsinghua University". Sources: PDF `https://arxiv.org/pdf/2602.15763` and arXiv HTML `https://arxiv.org/html/2602.15763` (curl + local parse, 2026-09-14); equations from the MathML `alttext` LaTeX; abs page for version history (2026-09-14). The HTML `<title>` lowercases the subtitle ("GLM-5: from Vibe Coding to Agentic Engineering"); the bib entry keeps "From". Section ids below are LaTeXML ids (e.g. `S3.SS5.p1.1` = §3.5, first paragraph). Release URL given by the paper: "Code, models, and more information are available at https://github.com/zai-org/GLM-5." (abstract, `abstract1.1`). The training framework `slime` (§3.6) is open source at `https://github.com/THUDM/slime` — **that URL is not in the paper**, it is from the repo itself; the report names slime but gives no link and no GLM-5 training configs.

```bibtex
@article{glm2026glm5,
  title   = {{GLM-5}: From Vibe Coding to Agentic Engineering},
  author  = {{GLM Team}},
  journal = {arXiv preprint arXiv:2602.15763},
  year    = {2026}
}
```

## Tier

**none** — for the on-policy distillation stage specifically.

§3.5 "On-Policy Cross-Stage Distillation" contains **no** statement about asynchrony, staleness, rollout versions, replay, or off-policy data. It is not tier D (no quotable async/sync statement *about the OPD stage*), not E/F/G (the report never says the OPD stage was run asynchronously), and not A/B/C (no theory, no async-vs-sync numbers, no ablation of any kind for this stage). See "Not in the source".

Why this is a deliberate finding rather than an omission of reading:
- GLM-5 *does* run a fully asynchronous, decoupled RL framework — but it scopes that to **agentic RL** (§3.3, §4.1), not to distillation. "To facilitate agentic performance of GLM-5, we develop a fully asynchronous and decoupled RL framework and optimize GLM-5 in coding and search agent tasks." (§3.3, `S3.SS3.p1.1`).
- §3.5 builds its loss on **Eq. 1**, the Reasoning-RL backbone of §3.2 — not on Eq. 3, the asynchronous agentic objective of §4.1.2. §4.1.2 draws the line explicitly: "Unlike the synchronous RL training setting in Section 3, in the asynchronous setting, rollout engines may undergo multiple updates during a single trajectory generation" (`S4.SS1.SSS2.Px2.p1.1`). That places the Eq.-1 family — and therefore §3.5's OPD — on the synchronous side, **by contrast rather than by direct assertion**. The report never writes "the distillation stage is synchronous".
- The async machinery of §4.1.2 (IS ratio to $\pi_{\text{rollout}}$, double-sided masking, version-lag dropping) is nonetheless the most directly relevant vocabulary in the sweep. It is recorded below under "Adjacent (agentic RL, NOT the OPD stage)" and must **not** be attributed to GLM-5's distillation stage.

## Setting

- **Student:** GLM-5 (MoE with DeepSeek Sparse Attention, MLA + MTP; §2.1). The OPD stage is the final post-training step applied to the model coming out of the RL stages.
- **Teachers:** the model's *own earlier checkpoints* — "the final checkpoints from the preceding training stages serve as teacher models" (§3.5, `S3.SS5.p1.1`), i.e. SFT, Reasoning RL and General RL endpoints. This is self-distillation across stages, not a larger external teacher.
- **Teacher fixed: yes.** The teachers are frozen finished checkpoints; nothing updates them during §3.5. In scope for this sweep. (Contrast the *pipeline* view: because the teachers are earlier versions of the same model, GLM-5's OPD is arguably the closest thing in the sweep to "learning from an earlier version of yourself" — but the staleness there is **intentional and fixed at stage boundaries**, not an artifact of a pipelined generator, so it is not the sweep's notion of async.)
- **Purpose:** anti-forgetting, not capability transfer. "In our multi-stage RL pipeline, sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities. To mitigate this issue, we perform on-policy cross-stage distillation as the final stage" (§3.5, `S3.SS5.p1.1`); "By leveraging on-policy cross-stage distillation as the final refinement, GLM-5 effectively mitigates capability regression while harnessing the performance gains from each training stage." (§3 intro, `S3.p1.1`, referring to Figure 5).
- **Prompts:** "the training prompts are sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions" (§3.5) — proportions unstated.
- **Pipeline order** (§3, Figure 5): SFT (§3.1) → Reasoning RL (§3.2, synchronous, Eq. 1) → Agentic RL (§3.3, asynchronous, details in §4) → General RL (§3.4) → **On-Policy Cross-Stage Distillation (§3.5)**.
  - Note an internal inconsistency with the executive summary, which says the distillation runs *throughout*: "Crucially, we utilized On-Policy Cross-Stage Distillation throughout this process to prevent catastrophic forgetting" (§1, `S1.SS0.SSS0.Px2.p1.1`) versus §3.5's "as the final stage". §3.5 is the technical section and is preferred.
- **Framework:** `slime` (§3.6). "GLM-5 leverages this capability to support a broad range of domains and training paradigms, including but not limited to reasoning RL, general RL, agentic RL, and on-policy distillation, all within a unified training stack." (§3.6.1, `S3.SS6.SSS1.p2.1`) — OPD is a first-class task *in* the async-capable stack, but this sentence is about task coverage, not about whether the OPD stage's rollouts were stale.
- **Benchmarks:** §6 (reasoning, coding, agentic). None reported per-OPD-configuration.

## Async mechanism

**For the OPD stage: none stated.** The report gives no pipelined trainer/generator, no replay buffer, no precomputed rollouts, no mixing ratio and no staleness metric for §3.5. The only quantities given for the stage are:

> "During training, the group size in the GRPO algorithm is configured to 1 to increase data throughput, and the batch size is set to 1024. This is feasible at this stage because it is no longer necessary to maintain a large group of samples per prompt to estimate advantages; the advantage is computed directly from the gap with the teacher models instead." (§3.5, `S3.SS5.p3.1`)

Group size 1, batch size 1024. Whether the 1024-sample batch is consumed in one optimizer step or split into minibatches (which would make even the synchronous setting mildly off-policy through Eq. 1's $r_{i,t}$ ratio) is **not stated**.

**The one mismatch the stage does acknowledge is engine-side, not version-side** (§3.5, `S3.SS5.p3.1`):

> "Currently, we utilize the inference engine to fetch teachers' logits. In the future, we plan to migrate the inference backend to the training engine and uniformly adopt the Multi-Query Attention (MQA) mode of MLA for inference ( $\pi_{\theta_{\text{teacher}}}^{\text{infer}}\rightarrow\pi_{\theta_{\text{teacher}}}^{\text{train}}$ )."

i.e. the *teacher*'s probabilities in Eq. 2 are inference-engine probabilities, and GLM-5 flags closing that train/infer gap as future work. This is a precision/kernel mismatch between two evaluations of the *same* weights — categorically different from training on rollouts produced by an *earlier* student, and it must not be counted as staleness.

### Adjacent (agentic RL, §4.1 — NOT the OPD stage)

Recorded because it is the sweep's richest vocabulary from this lab, and because the OPD stage explicitly does **not** use it.

- **Mechanism:** decoupled trainer/generator with periodic weight push. "we decouple the training engine and the inference engine onto different GPU devices. The inference engine continuously generates trajectories. Once the number of generated trajectories reaches a predefined threshold, the batch is sent to the training engine to update the model. To reduce policy lag and keep the training approximately on-policy, the model weights used by the rollout engine are periodically synchronized with those of the training engine. The training engine updates the model parameters and pushes the new weights back to the inference engine every $K$ gradient updates." (§4.1.1, `S4.SS1.SSS1.p1.1`). $K$ is not given.
- **What is stale:** the behaviour student, and mixed-version *within* a trajectory: "rollout engines may undergo multiple updates during a single trajectory generation" (§4.1.2). "it also means that different trajectories may be generated by different versions of the model, introducing a severe off-policy issue" (§4.1.1).
- **Staleness measured, and used:** per-response version list. "for each response we record the sequence of model versions involved, $(w_{0},\ldots,w_{k})$ with $w_{0}<\cdots<w_{k}$ . Let $w^{\prime}$ denote the current policy version. We discard a sample if its oldest rollout version is too stale, i.e., if $w^{\prime}-w_{0}>\tau$ , where $\tau$ is a predefined threshold." (§4.1.2, `S4.SS1.SSS2.Px3.p1.1`). $\tau$ is not given — so the staleness *range* is again unquantified.
- **Optimizer reset on each sync:** "Since the weight update considers a different optimization problem due to the changing rollout policy, we also reset the optimizer after each weight update of the inference engine." (§4.1.1).

## Objective

- **Vocabulary: `rkl_pg`** — sampled-token score-function estimator of reverse KL, inserted as the per-token advantage in a GRPO-family policy loss. Same family as Kimi K3 Eq. 15 and the Thinking Machines recipe; the family DeepSeek-V4 §5.1.2 rejects.
- **How it is defined** (§3.5, `S3.SS5.p1.1`): "adopting an on-policy distillation algorithm [14; 52; 51; 28] to swiftly recover the skills acquired in earlier SFT and RL stages… The training loss can be obtained by replacing the advantage term in Eq. 1 with the following formula ('sg' stands for the stop gradient operation, e.g., .detach()):" — citations are [14] MiniLLM (Gu et al.), [52] Qwen3 technical report, [51] MiMo-v2-flash (arXiv 2601.02780), [28] Thinking Machines Lab on-policy distillation post.
- **Eq. 2** (display `S3.E2`), verbatim LaTeX from the HTML `alttext`:
  ```latex
  \hat{A}_{i,t}=\text{sg}\left[\log\frac{\pi_{\theta_{\text{teacher}}}^{\text{infer}}(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta}^{\text{train}}(y_{i,t}\mid x,y_{i,<t})}\right].
  ```
- **Eq. 1**, the host loss it is substituted into (§3.2, displays `S3.E1`/`S3.E1Xa`), verbatim:
  ```latex
  \mathcal{L}(\theta)=-\mathbb{E}_{x\sim\mathcal{D},\{y_{i}\}_{i=1}^{G}\sim\pi^{\text{infer}}_{\theta_{\text{old}}}(\cdot\mid x)}
  \Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_{i}|}\sum_{t=1}^{|y_{i}|}\operatorname{pop}(\rho_{i,t},1/\beta,\beta)
  \cdot\min\!\left(r_{i,t}\hat{A}_{i,t},\operatorname{clip}\!\left(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}}\right)\hat{A}_{i,t}\right)\Bigg],
  ```
  with (`S3.EGx1`, `S3.EGx3`)
  ```latex
  \rho_{i,t}=\frac{\pi_{\theta_{\text{old}}}^{\text{train}}(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}^{\text{infer}}(y_{i,t}\mid x,y_{i,<t})}
  r_{i,t}=\frac{\pi_{\theta}^{\text{train}}(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}^{\text{train}}(y_{i,t}\mid x,y_{i,<t})},\quad \hat{A}_{i,t}=\frac{R_{i}-\operatorname{mean}(R_{1},\dots,R_{G})}{\operatorname{std}(R_{1},\dots,R_{G})}
  ```
  Aggregation is therefore **token-mean within a sequence, then mean over the group** ($\frac1G\sum_i\frac1{|y_i|}\sum_t$) — one of the few reports in the sweep that writes its normalization down. Under §3.5 the second $\hat A$ definition is replaced by Eq. 2 and $G=1$.
- **Direction check (convention: reverse KL = KL(student ‖ teacher)).** Consistent: $\hat A_{i,t}=\mathrm{sg}\log(\pi_{\text{teacher}}/\pi_\theta)$ on student-sampled $y$, maximized by policy gradient, is the sampled-token estimator of $-\mathrm{KL}(\pi_\theta\|\pi_{\text{teacher}})$. No deviation from the sweep's letters.
- **No full-vocabulary and no top-k variant is mentioned** for the OPD stage. (The "top-k" discussion in §3.2 is the DSA *indexer*'s $k=2048$ attention sparsity — an architecture matter, unrelated to truncated-support KL. Do not conflate.)

## Off-policy correction

**For the OPD stage:** whatever Eq. 1 carries, and nothing added. §3.5 replaces only "the advantage term in Eq. 1", so the surviving machinery is:

1. **IcePop-style train/infer mask** `pop(ρ, 1/β, β)` (Eq. 1 + `S3.EGx2`), verbatim:
   ```latex
   \operatorname{pop}(\rho_{i,t},1/\beta,\beta)=\begin{cases}\rho_{i,t},&1/\beta\leq\rho_{i,t}\leq\beta,\\ 0,&\text{otherwise}.\end{cases}
   ```
   "The operator $\operatorname{pop}(\cdot)$ suppresses samples whose mismatch ratio deviates excessively" (§3.2, `S3.SS2.SSS0.Px1.p2.1`). Per-token, zeroing (not clipping) out-of-band tokens. **This corrects the training-engine vs inference-engine mismatch, not staleness** — $\rho$ is the ratio of two evaluations of the *same* weights $\theta_{\text{old}}$.
2. **PPO ratio + clip** on $r_{i,t}=\pi_\theta^{\text{train}}/\pi_{\theta_{\text{old}}}^{\text{train}}$ — this *is* a genuine per-token importance ratio against the behaviour policy, so it would absorb minibatch-level staleness if §3.5 takes more than one gradient step per rollout batch. The report does not say whether it does.
3. **Hyperparameters:** stated only for §3.2 — "we set hyperparameters $\beta=2,\epsilon_{\text{low}}=0.2,\epsilon_{\text{high}}=0.28$ . Training is performed entirely on-policy with a group size of 32 and a batch size of 32." (`S3.SS2.SSS0.Px1.p4.1`). **§3.5 does not restate them**, and does not repeat the "entirely on-policy" claim. Whether $\beta,\epsilon$ carry over to the OPD stage is unstated.
4. **No IcePop-style mask against a rollout policy, no version tagging, no staleness filter, no $\pi_{\text{rollout}}$ ratio** in §3.5 — all of those live in §4.1.2 and are scoped to agentic RL.

**Adjacent (agentic RL, §4.1.2 — not the OPD stage).** "Direct double-sided importance sampling for token clipping", Eqs. 3–5, verbatim:
```latex
L(\theta)=\mathbb{E}_{t}\left[f(r_{t}(\theta),\epsilon_{l},\epsilon_{h})\hat{A}_{t}\log\pi_{\theta}(a_{t}|s_{t})\right]
r_{t}(\theta)=\exp\left(\log\pi_{\theta}(a_{t}|s_{t})-\log\pi_{\text{rollout}}(a_{t}|s_{t})\right)
f(x;\epsilon_{\ell},\epsilon_{h})=\begin{cases}x,&\text{if }1-\epsilon_{\ell}<x<1+\epsilon_{h}\\ 0,&\text{otherwise}\end{cases}
```
Ratio is against $\pi_{\text{rollout}}$ (the cached rollout log-probs), with $\pi_{\theta_{\text{old}}}$ removed entirely; out-of-band tokens are **masked to zero**, not clipped. $\epsilon_\ell,\epsilon_h$ values are not given. Explicitly IcePop-adjacent: "This shares similarities with the IcePop mechanism [44], yet our strategy is simpler by further removing the $\pi_{\theta_{\text{old}}}$ and achieving more stable training." ([44] = "Every step evolves: scaling reinforcement learning for trillion-scale thinking model", arXiv 2510.18855.)

## Results: async vs sync

**none.** No table, figure or sentence compares the OPD stage under different staleness settings, and no synchronous/asynchronous OPD baseline exists. §3.5 reports **no numbers at all** — not even an anti-forgetting before/after on the stages it is meant to recover. §6's benchmark tables report only the final GLM-5 against external models.

The async-RL section likewise reports no async-vs-sync *scores*; its claims are throughput/stability claims without a numeric baseline table.

## Explanation offered

**For the OPD stage: none** — the stage carries no mechanism discussion of staleness because it claims no staleness. The only "why" it offers is about the group size: "This is feasible at this stage because it is no longer necessary to maintain a large group of samples per prompt to estimate advantages; the advantage is computed directly from the gap with the teacher models instead." (§3.5, `S3.SS5.p3.1`) — a useful observation for the review (a dense per-token teacher signal removes the need for a group baseline, so OPD can run $G=1$ at large batch), but not a staleness argument.

**For agentic RL (adjacent, §4.1.2):** the mechanism offered is bias-vs-tracking-cost, i.e. accept a bounded off-policy bias rather than maintain checkpoint history —
> "In the experiments, we find that reusing rollout log-probabilities accepts a controlled degree of off-policy bias to circumvent the need for historical policy tracking while boosting training stability." (`S4.SS1.SSS2.Px2.p4.1`)
plus tail-truncation of the staleness distribution ("overly long trajectories can become highly off-policy, which may destabilize training", `S4.SS1.SSS2.Px3.p1.1`) and exact action alignment ("we find token-in-token-out is critical for asynchronous RL training because it preserves exact action-level correspondence between what was sampled and what is optimized", `S4.SS1.SSS2.Px1.p1.1`). Identified mechanism: **distribution shift bounded by masking + tail dropping**, with no appeal to estimator properties.

## Quotes

**On the OPD stage (§3.5) — the complete set; note that none of them mentions async, stale, replay, or off-policy:**
- `S3.SS5.p1.1`: "In our multi-stage RL pipeline, sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities. To mitigate this issue, we perform on-policy cross-stage distillation as the final stage, adopting an on-policy distillation algorithm [14; 52; 51; 28] to swiftly recover the skills acquired in earlier SFT and RL stages (Reasoning RL and General RL)."
- `S3.SS5.p1.1`: "Specifically, the final checkpoints from the preceding training stages serve as teacher models, where the training prompts are sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions."
- `S3.SS5.p1.1`: "The training loss can be obtained by replacing the advantage term in Eq. 1 with the following formula ('sg' stands for the stop gradient operation, e.g., .detach()):"
- `S3.SS5.p3.1`: "Currently, we utilize the inference engine to fetch teachers' logits. In the future, we plan to migrate the inference backend to the training engine and uniformly adopt the Multi-Query Attention (MQA) mode of MLA for inference."
- `S3.SS5.p3.1`: "During training, the group size in the GRPO algorithm is configured to 1 to increase data throughput, and the batch size is set to 1024. This is feasible at this stage because it is no longer necessary to maintain a large group of samples per prompt to estimate advantages; the advantage is computed directly from the gap with the teacher models instead."
- `S3.p1.1`: "By leveraging on-policy cross-stage distillation as the final refinement, GLM-5 effectively mitigates capability regression while harnessing the performance gains from each training stage."
- `S1.SS0.SSS0.Px2.p1.1`: "Crucially, we utilized On-Policy Cross-Stage Distillation throughout this process to prevent catastrophic forgetting, ensuring the model retains its sharp reasoning edge while becoming a robust generalist."

**The boundary statement that puts §3.5 on the synchronous side (by contrast, not by assertion):**
- `S4.SS1.SSS2.Px2.p1.1`: "Unlike the synchronous RL training setting in Section 3, in the asynchronous setting, rollout engines may undergo multiple updates during a single trajectory generation, which renders the tracking of exact behavior probabilities $\pi_{\theta_{\text{old}}}$ computationally prohibitive."
- `S3.SS2.SSS0.Px1.p4.1`: "Training is performed entirely on-policy with a group size of 32 and a batch size of 32." (§3.2 Reasoning RL — the stage whose Eq. 1 §3.5 reuses. Not restated for §3.5.)

**Framework-level, OPD named inside an async-capable stack:**
- `S3.SS6.SSS1.p2.1`: "GLM-5 leverages this capability to support a broad range of domains and training paradigms, including but not limited to reasoning RL, general RL, agentic RL, and on-policy distillation, all within a unified training stack."
- `S3.SS6.SSS2.p1.1`: "a single straggling trajectory can stall synchronization points (e.g., batch completion, buffer readiness, trainer updates) and directly determine wall-clock progress." (slime's motivation for tail-latency work — describes the synchronous cost the async path avoids.)

**Adjacent async-RL statements (agentic RL only; do NOT attribute to the OPD stage):**
- `S3.SS3.p1.1`: "To facilitate agentic performance of GLM-5, we develop a fully asynchronous and decoupled RL framework and optimize GLM-5 in coding and search agent tasks. Naive synchronous RL suffers from severe GPU idle time during long-horizon agent rollouts."
- `S3.SS3.p2.1`: "To maintain training stability under asynchronous off-policy conditions, we introduce two key mechanisms."
- `S4.SS1.SSS1.p1.1`: "To reduce policy lag and keep the training approximately on-policy, the model weights used by the rollout engine are periodically synchronized with those of the training engine. The training engine updates the model parameters and pushes the new weights back to the inference engine every $K$ gradient updates."
- `S4.SS1.SSS1.p1.1`: "While asynchrony could significantly improve overall training efficiency, it also means that different trajectories may be generated by different versions of the model, introducing a severe off-policy issue."
- `S4.SS1.SSS1.p1.1`: "Since the weight update considers a different optimization problem due to the changing rollout policy, we also reset the optimizer after each weight update of the inference engine."
- `S4.SS1.SSS2.Px1.p1.1`: "We find token-in-token-out is critical for asynchronous RL training because it preserves exact action-level correspondence between what was sampled and what is optimized…"
- `S4.SS1.SSS2.Px2.p2.1`: "Tokens falling outside this interval are entirely masked from gradient computation to prevent instabilities caused by extreme policy divergence. This shares similarities with the IcePop mechanism [44], yet our strategy is simpler by further removing the $\pi_{\theta_{\text{old}}}$ and achieving more stable training."
- `S4.SS1.SSS2.Px2.p4.1`: "In the experiments, we find that reusing rollout log-probabilities accepts a controlled degree of off-policy bias to circumvent the need for historical policy tracking while boosting training stability."
- `S4.SS1.SSS2.Px3.p1.1`: "In asynchronous RL, overly long trajectories can become highly off-policy, which may destabilize training. To filter out these severely off-policy samples, we log the policy weight version used by the rollout engine at generation time."

## Not in the source

- **The report never says whether the on-policy cross-stage distillation stage used asynchronous / stale rollouts.** It is silent: §3.5 contains no occurrence of "asynchronous", "off-policy", "stale", "replay", "lag" or "version". The synchronous reading rests on (a) §3.5 reusing Eq. 1 rather than Eq. 3, and (b) §4.1.2 calling Section 3 "the synchronous RL training setting" — both inferences, neither a direct statement. **Flag this as an inference wherever it is used in the review.**
- **No sync-vs-async comparison for OPD**, no staleness sweep, no lag metric → tiers A/B/C unavailable from this report.
- **No numbers of any kind for §3.5** — no before/after on the capabilities it is supposed to restore, no training curves, no cost. The anti-forgetting claim ("effectively mitigates capability regression") is asserted, never measured.
- **No hyperparameters carried over explicitly:** $\beta$, $\epsilon_{\text{low}}$, $\epsilon_{\text{high}}$, temperature, number of OPD steps, teacher mixing proportions, and the weights over the several teacher checkpoints are all unstated for §3.5.
- **No minibatching detail** — with batch 1024 and group 1, whether the batch is split across several optimizer steps (making $r_{i,t}\neq1$ in Eq. 1 and thus mildly off-policy even in the sync setting) is not said.
- **$K$ (weight-push interval) and $\tau$ (staleness-drop threshold) are never given numerically** in §4.1, so even the agentic-RL staleness *range* is unquantified — GLM-5 supplies the mechanism vocabulary but no numbers for it.
- **No full-vocabulary or truncated-support variant** is discussed for OPD, and no reason is given for choosing the sampled-token estimator over them; contrast DeepSeek-V4 §5.1.2, which argues against this exact estimator on variance grounds.
