# MiMo-V2-Flash (Xiaomi) — origin of the MOPD paradigm

**arXiv 2601.02780v2** (v1 submitted 2026-01-06, v2 revised 2026-01-08). Xiaomi LLM-Core Team (listed as "LLM-Core Xiaomi" on the PDF title page; ~150 named contributors on the arXiv abstract page, first five Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen). MoE, 309B total / 15B active. Content below originally from the arXiv HTML via WebFetch (2026-08-17); **all equations re-verified 2026-09-14 against the v2 PDF** (`curl -L https://arxiv.org/pdf/2601.02780`, pypdf) **and the v2 HTML** — the [FETCH] transcriptions below were confirmed correct, so the [FETCH] caveat is retired.

```bibtex
@article{xiaomi2026mimov2,
  author  = {{Xiaomi LLM-Core Team}},
  title   = {MiMo-V2-Flash Technical Report},
  journal = {arXiv preprint arXiv:2601.02780},
  year    = {2026}
}
```

**Abstract claim [RAW]:** "MiMo-V2-Flash introduces a novel Multi-Teacher On-Policy Distillation (MOPD) paradigm" where "domain-specialized teachers … provide dense and token-level reward, enabling the student model to perfectly master teacher expertise."

## §4.4 MOPD objective (explicit reverse KL)

- **Eq. 5 [FETCH]:** "The reverse KL divergence loss between student and teacher is defined as:" ℒ_reverse-KL(θ) = −𝔼[log π_domain_x(y_t|x,y_<t) / π_θ(y_t|x,y_<t)] — explicit "reverse KL" naming; token-level expectation, sequence framing in the preamble before importance sampling.
- **Eq. 6 (gradient of Eq. 5) [added 2026-09-14, was missing from this note]:** "The gradient is:" $\nabla_\theta \mathcal{L}_{\text{reverse-KL}}(\theta) = -\mathbb{E}_{x\sim\mathcal{D}, y_t\sim\pi_\theta(\cdot|x,y_{<t})}\big[\log\frac{\pi_{\text{domain}_x}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}\,\nabla_\theta \log \pi_\theta(y_t|x,y_{<t})\big]$ — the score-function form. Note it carries **no $\nabla_\theta$ of the $-\log\pi_\theta$ inside the bracket and no visitation/prefix term**, which is precisely the frozen-visitation surrogate flagged in the estimator-audit reading below.
- **Eq. 7 (surrogate loss) [FETCH]:** ℒ_MOPD(θ) = −𝔼[1/|y| Σ_t w_t Â_MOPD,t log π_θ(y_t|x,y_<t)] — REINFORCE-style score-function update (no PPO clip), per-sequence length normalization 1/|y|. Rollouts y ∼ μ_θ(·|x) (inference-engine copy of the student).
- **Eq. 8 (gated IS weight + advantage) [FETCH]:** w_t(θ) = sg[π_θ(y_t)/μ_θ(y_t)] if ε_low ≤ ratio ≤ ε_high, else 0 (hard gate = IcePop-style zero-out; "we apply training-inference importance sampling and discard tokens that exhibit large discrepancies"); Â_MOPD,t = sg[log π_domain_x(y_t|x,y_<t) / π_θ(y_t|x,y_<t)] — **per-token only, no return-to-go, no discount**.
- **Eq. 9 (ORM blending) [FETCH]:** Â_MOPD,t = sg[log π_domain_x / π_θ] + α Â_ORM.
- No acknowledgment of the partial-gradient / visitation-term issue (Tang & Munos pitfall 2) in the fetched section.

## Estimator-audit reading

Same family as Tinker / Nemotron MOPD: sampled-token log-ratio as a stop-gradient per-token advantage in a score-function update ⇒ frozen-visitation surrogate (G_local only) of the stated seq-level reverse KL. Closest published form to Nemotron-Cascade 2's Eqs. 2–4 (truncated-IS REINFORCE, per-seq length norm). **Correction 2026-09-14:** this note previously said MiMo differs from Nemotron in that "the IS weight is a hard gate (zero-out) rather than sg[r]·1[bounds]" — that is wrong, the two are the *same* object (Nemotron's $w_t=\mathrm{sg}[r_t]\mathbf{1}[\epsilon_\text{low}\le r_t\le\epsilon_\text{high}]$ is exactly MiMo's piecewise `sg[ratio] if in band else 0`). The real differences are that **Nemotron publishes the band $[0.5,2.0]$ while MiMo publishes no numeric $\epsilon$**, and that **MiMo's default objective adds an ORM advantage (Eq. 9) while Nemotron's $\mathcal{L}_\text{MOPD}$ is pure RKL**.

## Cross-references

- NVIDIA-NeMo/RL `ultra-v3` `OPDAdvantageEstimator` docstring cites this paper's Eq. 8 for its advantage and Eq. 9 for ORM blending (see D-tech-reports.md §1) — i.e., NVIDIA's public MOPD implementation is written against MiMo's equations.
- Chronology: Jan 2026, predates Nemotron-Cascade 2 (2603.19220) and Nemotron 3 Ultra (2606.15007).

**TODO resolved 2026-09-14.** §4.4 was read from the v2 PDF and cross-checked against the v2 HTML: the Eq. 5/7/8/9 transcriptions above are verbatim-correct. Answers to the open questions:
- **ε_low / ε_high: no numeric values anywhere in the paper.** §4.4 states the gate symbolically only; there is no hyperparameter table for MOPD. (Contrast Nemotron-Cascade 2, which gives $[0.5, 2.0]$.)
- **α (the ORM blending weight in Eq. 9): no numeric value given.**
- **MOPD sampling temperature: not stated.** Nor are MOPD batch size, rollout count per prompt, learning rate, or step count — Fig. 6 shows ~50 training steps on the x-axis, and that is the only MOPD-scale quantity in the report.
- **Multi-teacher weighting: no explicit weights.** Teacher selection is per-prompt by domain — $\pi_{\text{domain}_x}$ is "the teacher policy specialized for the domain of prompt $x$" (§4.4) — i.e. a hard routing by prompt domain, not a weighted mixture over teachers. Table 7 tags each benchmark's best teacher as RL, SFT, or Self.
- **"Zhao et al. (2025)" resolved:** the citation behind "we apply training-inference importance sampling and discard tokens that exhibit large discrepancies" is X. Zhao et al., *"Small leak can sink a great ship—boost RL training on MoE with icepop!"*, Sep 2025, https://ringtech.notion.site/icepop. So this note's earlier "IcePop-style zero-out" reading is **confirmed by the paper's own citation**, not just by shape.

## Async / off-policy (sweep 2026-09-14)

Sources: v2 PDF (`https://arxiv.org/pdf/2601.02780`, pypdf) and `https://arxiv.org/html/2601.02780v2`, both fetched 2026-09-14. arXiv **2601.02780v2** (v1 2026-01-06, v2 2026-01-08). Xiaomi LLM-Core Team.

```bibtex
@article{xiaomi2026mimov2,
  author  = {{Xiaomi LLM-Core Team}},
  title   = {MiMo-V2-Flash Technical Report},
  journal = {arXiv preprint arXiv:2601.02780},
  year    = {2026}
}
```

### Tier

- **E** — async OPD used in a production tech report: **YES, partially and system-level.** §4.6 is titled "RL Infrastructures" and opens "Our RL (**and MOPD**) infrastructure…", so §4.6.2's partial-rollout-with-staleness machinery is declared to cover the MOPD stage. This is a frontier production report shipping a *bounded-staleness* OPD stage.
- **D** — quotable statements about async/sync OPD: **YES.** Two independent ones: §4.6.2's staleness-limiting / staleness-aware-TIS sentence, and §2.3.1's argument that on-policy is preferable but expensive.
- **G** — async used but not the main point: **YES.** Staleness is a throughput device buried in the infrastructure section; the paper's thesis is the MOPD paradigm itself.
- **A** — no. No argument that distribution-level RKL tolerates staleness better or worse than sampled-token `rkl_pg`. The objective is sampled-token throughout, so the comparison is never posed.
- **B, C** — no. No async-vs-sync numbers at any staleness level.
- **F** — no.

**Answer to the sweep's question.** There is **no statement "MOPD is robust to off-policy data"** in this paper; the opposite lean is stated. But asynchrony *is* present, in one specific and bounded form: **partial rollout**, which splits overlong trajectories across optimization steps so that a batch contains tokens generated by an earlier student. It is (a) explicitly staleness-limited, (b) explicitly corrected by staleness-aware truncated importance sampling, (c) quantified nowhere, and (d) never ablated.

### Setting

- Student: MiMo-V2-Flash, MoE 309B total / 15B active, hybrid SWA (128-token window, 5:1 ratio) + MTP; SFT checkpoint is the MOPD starting point.
- Teachers: a suite of domain-specialized models from **Stage 2** ("We train a suite of domain-specialized teacher models through independent RL optimization", §4.1), covering search/code/general-tool agentic tasks and math/general-reasoning/safety. Per-prompt hard routing to $\pi_{\text{domain}_x}$.
- **Teacher fixed? Yes, within the reported run.** Stage 2 (train teachers) strictly precedes Stage 3 (MOPD), and teacher–student co-evolution is listed as a *property the framework supports* and as *future work*, not as something executed here: "MOPD naturally supports a teacher-student co-evolution cycle" (§4.1) and "we plan to scale the compute for the iterative co-evolution of teachers and students in MOPD to fully unlock its potential" (§6, Conclusion/Future Work). Figure 3 does draw "Replace" arrows on the domain-teacher box, but no run with a changing teacher is reported. **In scope.**
- Benchmarks: AIME 2025, HMMT Feb 2025, LiveCodeBench, MMLU-Pro, GPQA-Diamond, HLE, Arena-Hard, SWE-Bench Verified/Multilingual, τ²-Bench, BrowseComp, LongBench V2, MRCR.
- Framework: **SGLang** inference engine + **Megatron-LM** training engine, FP8 for both (§4.6). Three custom modules: Rollout Routing Replay / R3 (§4.6.1), Data Scheduler (§4.6.2), Toolbox + Tool Manager (§4.6.3). Extends the **Seamless Rollout Engine** (Xia et al., 2025). Not open-sourced (weights + MTP weights are; the RL stack is not).

### Async mechanism

**Partial rollout with bounded staleness**, §4.6.2 (Data Scheduler). The load-bearing sentences, verbatim:

> "We integrate partial rollout (Fu et al., 2025; Kimi Team, 2025b) to partition overlong trajectories across steps, while limiting staleness and the proportion of partial samples in each batch. By employing staleness-aware truncated importance sampling for partial rollout, we significantly accelerate RL training without sacrificing model quality."

**What is stale.** A single long trajectory is *cut across optimization steps*: its early tokens are generated by student version $\theta_{k}$, generation is suspended, the student is updated, and the remainder is generated by $\theta_{k+1}, \theta_{k+2}, \dots$. So within one training batch, the behavior policy for a partial sample's prefix is an **earlier version of the student**, and different tokens of the *same sequence* can have different behavior policies. This is the AReaL/Kimi-k1.5 partial-rollout design — the two citations confirm it: **Fu et al., 2025 = AReaL, "a large-scale asynchronous reinforcement learning system for language reasoning", arXiv 2505.24298**; **Kimi Team, 2025b = Kimi k1.5, arXiv 2501.12599**. Citing an explicitly *asynchronous* RL system for this mechanism is the strongest evidence that the staleness here is temporal, not numerical.

**Two independent staleness knobs are named** (both unquantified):
1. a cap on staleness itself ("limiting staleness"), i.e. a max policy-version lag per sample;
2. a cap on the mixing ratio ("the proportion of partial samples in each batch"), i.e. fresh/stale batch composition.

**Distinguish from the two *non*-temporal mismatches in the same paper**, which must not be conflated:
- **Train/inference-engine mismatch** ($\pi_\theta$ vs $\mu_\theta$, §4.4 Eqs. 7–8): same weights, two engines, numerical gap — same object as Nemotron's $r_t$. Corrected by the Eq. 8 gate.
- **MoE routing inconsistency** (§4.6.1): "MoE models suffer from inconsistent expert routing across rollout and training due to numerical precision issues." Fixed structurally by **Rollout Routing Replay (R3)** — "to train RL using the same routed experts from rollout" — plus a request-level prefix cache that "avoids re-prefilling or inter-request output cache sharing, ensuring sampling consistency for routed experts" (§4.6.1). R3 is an MoE-specific *determinism* fix, not a staleness correction.

**How staleness is measured:** not measured, not reported. **Range of values studied:** **not stated** — no max-lag value, no partial-sample fraction, no buffer size, no off-policyness metric. This is the single largest gap in the paper for our purposes.

**Also present but unrelated:** §4.6.3, "sequence-level asynchronous reward computation" — async *reward/tool* execution overlapped with rollout, not async rollout generation.

### Objective

`rkl_pg` — sampled-token score-function estimator of reverse KL, with the direction named explicitly by the paper ("The reverse KL divergence loss between student and teacher"). Conforms to our convention: Eq. 5 is $-\mathbb{E}_{y_t\sim\pi_\theta}\log\frac{\pi_{\text{domain}_x}}{\pi_\theta} = \mathbb{E}_{y_t\sim\pi_\theta}\log\frac{\pi_\theta}{\pi_{\text{domain}_x}} = \mathrm{KL}(\text{student}\,\|\,\text{teacher})$. No deviation to flag.

Verbatim, §4.4 "Technical Formulation of MOPD" (all five equations; re-verified against v2 PDF and v2 HTML):

Preamble: "Let $\pi_\theta$ denote the target student policy optimized in the training engine, $\mu_\theta$ denote the student sampling policy adopted in the inference engine, and $\pi_{\text{domain}_x}$ denote the teacher policy specialized for the domain of prompt $x$ sampled from distribution $\mathcal{D}$. Let $\mathrm{sg}[\cdot]$ denote the stop-gradient operator. The reverse KL divergence loss between student and teacher is defined as:"

$$\mathcal{L}_{\text{reverse-KL}}(\theta) = -\mathbb{E}_{x\sim\mathcal{D},\, y_t\sim\pi_\theta(\cdot|x,y_{<t})} \log \frac{\pi_{\text{domain}_x}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})} \tag{5}$$

"The gradient is:"

$$\nabla_\theta \mathcal{L}_{\text{reverse-KL}}(\theta) = -\mathbb{E}_{x\sim\mathcal{D},\, y_t\sim\pi_\theta(\cdot|x,y_{<t})}\left[\log \frac{\pi_{\text{domain}_x}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})} \nabla_\theta \log \pi_\theta(y_t|x,y_{<t})\right] \tag{6}$$

"Following Zhao et al. (2025), we apply training-inference importance sampling and discard tokens that exhibit large discrepancies. We then define the surrogate loss of MOPD as:"

$$\mathcal{L}_{\text{MOPD}}(\theta) = -\mathbb{E}_{x\sim\mathcal{D},\, y\sim\mu_\theta(\cdot|x)}\left[\frac{1}{|y|}\sum_{t=1}^{|y|} w_t\, \hat{A}_{\text{MOPD},t}\, \log \pi_\theta(y_t|x,y_{<t})\right] \tag{7}$$

"where"

$$w_t(\theta) = \begin{cases} \mathrm{sg}\!\left[\dfrac{\pi_\theta(y_t|x,y_{<t})}{\mu_\theta(y_t|x,y_{<t})}\right], & \epsilon_{\text{low}} \le \dfrac{\pi_\theta(y_t|x,y_{<t})}{\mu_\theta(y_t|x,y_{<t})} \le \epsilon_{\text{high}} \\[2ex] 0, & \text{other} \end{cases} \qquad \hat{A}_{\text{MOPD},t} = \mathrm{sg}\!\left[\log\frac{\pi_{\text{domain}_x}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}\right] \tag{8}$$

"By default, we combine the advantages of MOPD with other types of advantages, such as those computed using Outcome Reward Models (ORMs), including GRPO (Shao et al., 2024). Let $\hat{A}_{\text{ORM}}$ denote the advantages computed by the ORMs; the final advantages are given by:"

$$\hat{A}_{\text{MOPD},t} = \mathrm{sg}\!\left[\log\frac{\pi_{\text{domain}_x}(y_t|x,y_{<t})}{\pi_\theta(y_t|x,y_{<t})}\right] + \alpha \hat{A}_{\text{ORM}} \tag{9}$$

Structural notes: per-sequence length normalization $1/|y|$ (a length-normalized variant, not a pure token-mean over the batch); REINFORCE-style score function with **no PPO clip**; the only gradient path is $\log\pi_\theta(y_t|\cdot)$ (both $w_t$ and $\hat{A}$ carry $\mathrm{sg}$); $\hat{A}$ is **per-token only — no return-to-go, no discount**. **Eq. 9 is the default**, so the shipped MOPD objective is RKL *plus* an ORM/GRPO advantage, not pure RKL; Fig. 6's "MOPD w/o ORM" curve is the $\alpha=0$ ablation.

### Off-policy correction

Two logically distinct corrections, only one of which is written down as an equation.

**(1) Train/inference-engine mismatch — Eq. 8, fully specified in form, unspecified in value.**
- Ratio: **per-token**, $\pi_\theta(y_t|x,y_{<t}) / \mu_\theta(y_t|x,y_{<t})$. Not per-sequence.
- Treatment: a **hard gate / zero-out**, not a clamp. Inside $[\epsilon_{\text{low}}, \epsilon_{\text{high}}]$ the weight is $\mathrm{sg}[\text{ratio}]$; outside, it is exactly $0$ — the token is discarded. This is **IcePop**, per the paper's own citation (Zhao et al., 2025 = the IcePop note, https://ringtech.notion.site/icepop): "we apply training-inference importance sampling and **discard tokens that exhibit large discrepancies**."
- **Values: none given.** No numeric $\epsilon_{\text{low}}$ / $\epsilon_{\text{high}}$ anywhere in the paper (verified over v2 PDF and v2 HTML).
- Relation to Nemotron-Cascade 2: Nemotron's Eq. 3 $w_t = \mathrm{sg}[r_t]\mathbf{1}[\epsilon_{\text{low}} \le r_t \le \epsilon_{\text{high}}]$ is *the same object* as MiMo's Eq. 8 — this note previously described the two as differing ("hard gate rather than $\mathrm{sg}[r]\cdot\mathbf{1}$"), but the indicator form and the piecewise-zero form are identical. The real difference is that **Nemotron publishes the band $[0.5, 2.0]$ and MiMo publishes none.**

**(2) Staleness from partial rollout — named, never written down.** "By employing **staleness-aware truncated importance sampling** for partial rollout…" (§4.6.2). No equation, no definition of "staleness-aware", no statement of whether it is a separate weight from Eq. 8's $w_t$ or a widening of the same band as a function of lag, and no values. **This is the paper's most significant omission for the async sweep.**

**(3) MoE routing (structural, not statistical):** R3 replays rollout-time expert routing into the training forward pass (§4.6.1) so the ratio in Eq. 8 is not contaminated by expert-selection divergence. Overhead "negligible".

Masks beyond the Eq. 8 gate: none stated. No KL-to-reference penalty in the MOPD objective.

### Results: async vs sync

**None.** No side-by-side at any staleness level, no throughput numbers for the staleness mechanism, no ablation of the staleness cap or the partial-sample fraction. The only claim is unquantified and unsupported by a reported measurement: "we significantly accelerate RL training **without sacrificing model quality**" (§4.6.2) — no speedup factor, no quality delta, no baseline.

For completeness, the MOPD results that *are* reported (all at whatever the production staleness setting was — the paper never distinguishes a sync arm):

| Benchmark | Student before MOPD | Best teacher (type) | Student after MOPD | Δ (student − teacher) |
|---|---|---|---|---|
| AIME 2025 | 89.3 | 93.9 (RL) | 94.1 | +0.2 |
| HMMT Feb. 2025 | 76.9 | 82.6 (RL) | 84.4 | +1.8 |
| LiveCodeBench | 77.5 | 82.6 (RL) | 83.2 | +0.6 |
| MMLU-Pro | 84.7 | 84.7 (Self) | 84.9 | +0.2 |
| GPQA-Diamond | 84.9 | 84.9 (Self) | 84.3 | −0.6 |
| HLE (w/o Tool) | 21.2 | 21.2 (Self) | 22.1 | +0.9 |
| Arena-Hard (Hard Prompt) | 50.0 | 50.0 (Self) | 54.1 | +4.1 |
| Arena-Hard (Creative Writing) | 90.1 | 90.1 (Self) | 86.2 | −3.9 |
| SWE-Bench Verified | 67.8 | 74.2 (RL) | 73.4 | −0.8 |
| τ²-Bench | 75.9 | 79.6 (RL) | 80.3 | +0.7 |
| τ²-Bench (Telecom) | 92.7 | 95.0 (RL) | 95.3 | +0.3 |
| BrowseComp | 42.5 | 51.7 (SFT) | 45.4 | **−6.3** |

Source: **Table 7**, §4.1. BrowseComp is the conspicuous failure (−6.3 vs its SFT teacher), and Creative Writing regresses −3.9 below the student's own pre-MOPD score. The paper offers no diagnosis for either, and in particular never asks whether staleness contributed — the natural suspicion for long agentic search trajectories, which are exactly the ones partial rollout would chop across steps.

**Figure 6** (§4.5.1, x-axis = training step, 0–50): three arms, "ORM", "MOPD w/o ORM", "MOPD", plus a teacher line, on AIME 2025 (84→94%) and LiveCodeBench (72→84%). This is an *objective* ablation ($\alpha=0$ vs $\alpha>0$ in Eq. 9), **not** an async ablation.

### Explanation offered

No explanation of *why* bounded staleness is acceptable for MOPD is given. The paper's reasoning runs the other way — it treats on-policy as the desirable regime and asynchrony as a throughput concession:

- §2.3.1 ("Accelerating RL Training"), verbatim: "**It enables efficient and effective RL with small batches.** Current RL training relies on large-batch, off-policy algorithms to maximize throughput (Liu et al., 2025; Schulman et al., 2017; Zheng et al., 2025). **However, on-policy training is generally more stable and effective**, yet its small batches underutilize GPU resources. MTP mitigates this limitation by scaling token-level parallelism instead of batch size, **making small-batch, on-policy RL training more practical**."
  - Mechanism named: none — asserted, with PPO/GSPO/DeepSeek cited as the off-policy status quo. The interesting move is architectural: MTP (speculative decoding) is deployed **specifically so they can afford to stay on-policy**. That is a production team spending architecture budget to *avoid* staleness.
- §2.3.1: "It mitigates GPU idleness from long-tail stragglers. As the rollout phase progresses, long-tail stragglers that process long sequences with small batch sizes (often approaching 1) can cause significant GPU idleness."
  - This is the pressure that partial rollout answers. The paper therefore has two independent long-tail remedies — MTP (make generation faster, stay sync) and partial rollout (accept bounded staleness) — and never compares them.
- §4.1 ("Effective and Efficient" bullet), verbatim: "**By learning from its own distribution, the student avoids the exposure bias and distribution mismatch common in off-policy methods trained on static datasets.**"
  - Mechanism named: **exposure bias / distribution mismatch**. Crucially this is an argument against *offline* distillation (fixed teacher-generated corpora), a different axis from rollout staleness — a batch of 5-step-stale student samples still comes from very near the student's own distribution. The paper does not distinguish the two senses of "off-policy", and this sentence should not be quoted as evidence about staleness tolerance.
- §4.1 (Stage 3), verbatim: "Rather than merging model parameters or generating static offline datasets from experts, we formulate multi-teacher knowledge integration as an on-policy reinforcement learning process. **The student model samples from its own evolving distribution** and receives token-level supervision from domain-specific teachers through KL divergence rewards."
- §4.1 ("Effective and Efficient"): "on-policy distillation using dense, token-level rewards from teacher logits ensures stable credit assignment and rapid convergence."
  - Mechanism named: **credit-assignment density** — orthogonal to staleness.

No theory, no variance analysis, no support-mismatch argument, no staleness-tolerance claim in either direction.

### Quotes

Every verbatim statement bearing on async / sync / stale / off-policy / replay, with section id.

1. §4.6 (RL Infrastructures, opening): "Our RL (and MOPD) infrastructure uses SGLang (Zheng et al., 2024) as the inference engine and Megatron-LM (Shoeybi et al., 2019) as the training engine. We adopt FP8 for both training and inference." ← **establishes that §4.6.2 applies to MOPD**
2. §4.6.2 (Data Scheduler): "**We integrate partial rollout (Fu et al., 2025; Kimi Team, 2025b) to partition overlong trajectories across steps, while limiting staleness and the proportion of partial samples in each batch.**" ← the core async statement
3. §4.6.2: "**By employing staleness-aware truncated importance sampling for partial rollout, we significantly accelerate RL training without sacrificing model quality.**" ← the core correction statement
4. §4.6.2: "In dynamic sampling, as sequences return for reward computation, we reference historical pass rates and, if necessary, assign new prompts to GPUs with load balancing."
5. §4.6.2: "For MiMo-V2-Flash, we extend the Seamless Rollout Engine (Xia et al., 2025) and implement a Data Scheduler to seamlessly schedule fine-grained sequences instead of micro-batches, addressing GPU idleness in distributed MoE training."
6. §4.6.2: "The Data Scheduler supports data source-specific configurations (sample quota, scheduling priority, length limits, temperature) and fits pass rates to accept samples by configured ratios. Priority-based scheduling overlaps reward computation and inference across data sources with different time patterns, ensuring high GPU utilization."
7. §2.3.1 (Accelerating RL Training): "Current RL training relies on large-batch, off-policy algorithms to maximize throughput (Liu et al., 2025; Schulman et al., 2017; Zheng et al., 2025). However, on-policy training is generally more stable and effective, yet its small batches underutilize GPU resources. MTP mitigates this limitation by scaling token-level parallelism instead of batch size, making small-batch, on-policy RL training more practical."
8. §2.3.1: "It mitigates GPU idleness from long-tail stragglers. As the rollout phase progresses, long-tail stragglers that process long sequences with small batch sizes (often approaching 1) can cause significant GPU idleness (Gao et al., 2025; Zhong et al., 2025)."
9. §4.1 (Effective and Efficient): "By learning from its own distribution, the student avoids the exposure bias and distribution mismatch common in off-policy methods trained on static datasets." ← about *offline* datasets, not staleness
10. §4.1 (Stage 3): "Rather than merging model parameters or generating static offline datasets from experts, we formulate multi-teacher knowledge integration as an on-policy reinforcement learning process. The student model samples from its own evolving distribution…"
11. §4.4: "Following Zhao et al. (2025), we apply training-inference importance sampling and discard tokens that exhibit large discrepancies."
12. §4.6.1 (R3): "MoE models suffer from inconsistent expert routing across rollout and training due to numerical precision issues (He and Lab, 2025; Yao et al., 2025). We propose Rollout Routing Replay (R3) (Ma et al., 2025) to train RL using the same routed experts from rollout, making its overhead negligible through optimized data types and communication overlapping."
13. §4.6.1: "For multi-turn agent training, we employ a request-level prefix cache during rollout. This cache stores KVCache and MoE routed experts from prior turns, allowing them to be reused for subsequent generation steps of the same request. Unlike the commonly-used radix cache in current inference engines, our request-level prefix cache avoids re-prefilling or inter-request output cache sharing, ensuring sampling consistency for routed experts."
14. §4.6.3 (Toolbox / Tool Manager): "Integrated with the rollout engine, Tool Manager coordinates with Toolbox to accelerate training through environment pre-warming and **sequence-level asynchronous reward computation**." ← async *reward*, not async rollouts
15. §4.1 (Iterative Co-Evolution): "MOPD naturally supports a teacher-student co-evolution cycle. Distilled student models can re-enter the specialized RL stage to produce stronger teachers, which in turn provide higher-quality supervision for the next generation of students…" ← capability claim only
16. §6 (Conclusion): "we plan to scale the compute for the iterative co-evolution of teachers and students in MOPD to fully unlock its potential." ← confirms co-evolution is **future work**; teachers fixed in this report

Relevant bibliography resolutions (§References): **Fu et al., 2025** = "AReaL: A large-scale asynchronous reinforcement learning system for language reasoning", arXiv 2505.24298. **Kimi Team, 2025b** = "Kimi k1.5: Scaling reinforcement learning with LLMs", arXiv 2501.12599. **Zhao et al., 2025** = X. Zhao et al., "Small leak can sink a great ship—boost RL training on MoE with icepop!", Sep 2025, https://ringtech.notion.site/icepop. **Gao et al., 2025** = "Rollpacker: Mitigating long-tail rollouts for fast, **synchronous** RL post-training", arXiv 2509.21009. **Ma et al., 2025** = "Stabilizing MoE reinforcement learning…" (the R3 paper).

Absent from the entire document (verified over v2 PDF text and a full-document v2 HTML pass, 2026-09-14): any occurrence of "replay buffer", "rollout lag", "behavior policy", "policy version", "robust to off-policy", or any numeric staleness bound. The word "staleness" occurs exactly twice, both in quotes 2–3.

### Not in the source

- **No numeric staleness bound.** "limiting staleness" with no max-lag in steps, tokens, or policy versions; "the proportion of partial samples in each batch" with no ratio. The entire async configuration is unreported.
- **No definition of "staleness-aware truncated importance sampling".** Named once, never formalized. It is not clear whether it is a distinct weight from Eq. 8's $w_t$, a lag-dependent widening of $[\epsilon_{\text{low}}, \epsilon_{\text{high}}]$, or a per-segment sequence-level ratio.
- **No numeric $\epsilon_{\text{low}}$ / $\epsilon_{\text{high}}$, and no $\alpha$** for Eq. 9's ORM blend.
- **No sync baseline.** "without sacrificing model quality" is asserted with no comparison run, no quality delta, and no speedup factor.
- **No fraction of tokens discarded** by the Eq. 8 gate, and no measured distribution of the ratio.
- **No MOPD training hyperparameters at all** — no batch size, rollouts per prompt, learning rate, optimizer, temperature, or step count (Fig. 6's ~50-step x-axis is the only scale cue). Contrast Nemotron-Cascade 2, which gives a full MOPD hyperparameter column (Appendix B, Table 8).
- **No diagnosis of the two MOPD regressions** in Table 7 (BrowseComp −6.3, Arena-Hard Creative Writing −3.9), and no test of whether long-trajectory staleness is implicated — the obvious hypothesis, since BrowseComp trajectories are precisely the "overlong trajectories" §4.6.2 partitions across steps.
- **No RL stack release.** Weights and MTP weights are open-sourced; the Data Scheduler / partial-rollout implementation is not, so none of the above can be recovered from code.
- **No acknowledgment of the partial-gradient / frozen-visitation issue** (Tang & Munos pitfall 2) — Eq. 6 silently drops the visitation term, and staleness in the prefix makes the dropped term *more* wrong, which the paper does not discuss.
