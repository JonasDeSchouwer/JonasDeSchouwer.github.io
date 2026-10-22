# Nemotron-Cascade 2 (arXiv 2603.19220)

```bibtex
@article{nvidia2026nemotroncascade2,
  author  = {{NVIDIA}},
  title   = {Nemotron-Cascade 2: Post-Training {LLM}s with Cascade {RL} and Multi-Domain On-Policy Distillation},
  journal = {arXiv preprint arXiv:2603.19220},
  year    = {2026}
}
```

All content below read from the arXiv HTML via WebFetch, 2026-08-19; **re-verified 2026-09-14 against the v2 PDF** (`curl -L https://arxiv.org/pdf/2603.19220`, pypdf) and the v2 HTML. Repro artifacts, code lineage, and stage order are in `notes/D-tech-reports.md` §2.

## MOPD objective (§4.4, Eqs. 2–4, verbatim)

Setup prose: "For each prompt x, we sample a response y=(y₁,…,y_T)∼π^inf(·|x). We then select a domain teacher π^{domain_i} for that training example, where domain_i indicates the capability domain associated with the chosen teacher. Writing s_t=(x,y_{<t}) for the decoding state at step t, we define the token-level distillation advantage using reverse-KL as"

**Eq. 2 (advantage):**

$$a_t^{\text{MOPD}} = \log \pi^{\text{domain}_i}(y_t \mid s_t) - \log \pi^{\text{train}}(y_t \mid s_t)$$

Prose after Eq. 2: "Intuitively, this term is positive when the domain teacher assigns a higher probability to the sampled token than the current training policy, and therefore serves as a dense token-level distillation advantage that converges toward 0 during training. The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary."

**Eq. 3 (truncated importance weighting):** "Because responses are sampled under π^inf but optimized under π^train, we apply truncated importance weighting to account for train–inference mismatch:"

$$r_t = \frac{\pi^{\text{train}}(y_t \mid s_t)}{\pi^{\text{inf}}(y_t \mid s_t)}, \quad w_t = \mathrm{sg}[r_t] \, \mathbf{1}[\epsilon_{\text{low}} \leq r_t \leq \epsilon_{\text{high}}]$$

Values: "For truncated importance weighting, we set ϵ_low=0.5 and ϵ_high=2.0."

**Eq. 4 (surrogate loss):** "We then optimize the surrogate objective"

$$\mathcal{L}_{\text{MOPD}} = -\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi^{\text{inf}}(\cdot \mid x)} \left[\frac{1}{|\mathcal{V}(y)|} \sum_{t \in \mathcal{V}(y)} w_t \, \mathrm{sg}[a_t^{\text{MOPD}}] \log \pi^{\text{train}}(y_t \mid s_t)\right]$$

"where 𝒱(y) is the set of valid response tokens retained by the token mask."

Note: sg appears on a_t in Eq. 4 (not in Eq. 2's definition) and on r_t in Eq. 3; the only gradient path is log π^train.

## Definitions and facts (§4.4)

- π^inf / π^train: "Let π^inf denote the student policy used for response generation in the inference engine, and let π^train denote the student policy optimized by the training engine."
- Loss composition: L_MOPD is the whole stage objective — no task reward, SFT term, or KL-to-reference mixed in.
- Teachers (all self-derived): math = initial SFT ckpt; RLHF teacher = RLHF-stage ckpt; multi-domain = post-IF-RL+MDRL ckpt. Prompts "from the RL training data pools (RLHF, IF-RL, and Multi-domain), as well as from AceReason-Math for math."
- Hparams (§4.4): 128 prompts/update × 4 rollouts = "effective batch size of 512 responses"; lr 2e-6 with linear warm-up over first 30 steps from 2e-7; "Training typically converges within 40-50 optimization steps."
- Batch-shape variant (§4.4, was missing from this note): "In later experiments, we find that using 512 prompts with rollout size 1 yields slightly more stable optimization while producing similar final results."
- **Internal inconsistency (flagged 2026-09-14):** §4.4 prose says lr $2\times10^{-6}$; Appendix B Table 8 (MOPD column) says lr $3\times10^{-6}$. Table 8 also gives Steps 52 (matching Table 3's "MOPD 52") against §4.4's "40-50", and max response length 98K. Batch size 128 / rollout size 4 agree.
- **CORRECTED 2026-09-14 — the MOPD sampling temperature *is* stated.** The previous version of this note claimed "no sampling temperature stated for MOPD rollouts (checked exhaustively)". That is wrong: **Appendix B, Table 8** ("Training hyperparameters of Nemotron-Cascade-2-30B-A3B in Cascade RL (IF-RL, Multi-domain RL, MOPD)") has an explicit MOPD column reading `Temperature 1.0` and `Top-p 1.0`. The §4.4 *prose* indeed omits it — that is what the earlier check saw — but the appendix table supplies it. `lit_review.typ` Table 1's "Sampled from" cell was updated from "T unstated" to "$T = 1.0$, top-p $1.0$ (Table 8)" in the same pass.

## Async / off-policy (sweep 2026-09-14)

Sources for this section: v2 PDF (`https://arxiv.org/pdf/2603.19220`, pypdf text extraction) and `https://arxiv.org/html/2603.19220v2`, both fetched 2026-09-14. arXiv **2603.19220v2** (v1 2026-03-19, v2 2026-03-22); PDF cover date 2026-03-16. Authors: Zhuolin Yang, Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander Bukharin, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping (NVIDIA).

```bibtex
@article{nvidia2026nemotroncascade2,
  author  = {{NVIDIA}},
  title   = {Nemotron-Cascade 2: Post-Training {LLM}s with Cascade {RL} and Multi-Domain On-Policy Distillation},
  journal = {arXiv preprint arXiv:2603.19220},
  year    = {2026}
}
```

### Tier

- **E** — async OPD used in a production tech report: **NO, the opposite.** This is a production tech report whose OPD stage is explicitly and strictly *synchronous*. It qualifies as a tier-E *negative* data point: a frontier MOPD run that deliberately refuses staleness.
- **D** — quotable statements about async/sync OPD: **YES.** §4.1.2 and §4.7.2 contain direct, quotable justifications for choosing strict on-policy over off-policy/stale training ("improved stability and higher accuracy", "mitigates entropy collapse", "fully on-policy for stability").
- **A, B, C** — no. No staleness argument, no async-vs-sync numbers, no ablation.
- **F, G** — no. Nothing in the paper is async at the rollout level.

**Answer to the sweep's question:** the MOPD stage is **not asynchronous**. The truncated importance weighting with $\epsilon_{\text{low}}=0.5$, $\epsilon_{\text{high}}=2.0$ corrects **only train/inference-engine numerical mismatch on fresh rollouts**, exactly as the note's §4.4 transcription says. The paper states elsewhere, in full generality over the whole Cascade RL process (of which MOPD is one stage), that each rollout batch receives **exactly one gradient update**.

### Setting

- Student: Nemotron-Cascade-2-30B-A3B (30B MoE, 3B activated), initialized from the SFT checkpoint and carried through the Cascade RL pipeline.
- Teachers (3, all self-derived, **fixed** during MOPD): math = the initial SFT checkpoint; RLHF = a checkpoint optimized through RLHF from the initial SFT checkpoint; multi-domain = a checkpoint after the IF-RL + Multi-domain RL stages. Teachers are frozen checkpoints selected before the stage; they are not trained during MOPD. In scope.
- Domain/benchmarks: AIME25 (avg@64), ArenaHard v2 (Hard Prompt / Creative Writing), plus the full report suite.
- Framework: **NeMo-RL** ("We conduct our training using the Nemo-RL repository (NVIDIA, 2025)", §4.1.2).
- Stage position: MOPD sits between Multi-domain RL and RLHF in the Cascade (Figure 2, §4.1).

### Async mechanism

**None.** There is no stale-rollout mechanism anywhere in the paper.

What *is* present, and what must not be mistaken for staleness:

1. **Train/inference-engine mismatch (§4.4).** Two *copies of the same weights* live in two engines: $\pi^{\text{inf}}$ in the inference engine (generation) and $\pi^{\text{train}}$ in the training engine (optimization). The paper's own framing: "Let $\pi^{\text{inf}}$ denote the student policy used for response generation in the inference engine, and let $\pi^{\text{train}}$ denote the student policy optimized by the training engine." The gap between them is *numerical* (kernel/precision/engine differences), not temporal. §4.4 names the cause explicitly as "train–inference mismatch", never as staleness, lag, or off-policyness.

2. **One gradient step per rollout batch (§4.1.2).** "At each iteration, we generate a group of $G$ rollouts from the current policy $\pi_\theta$ and then perform a *single* gradient update. This ensures that the policy used for data collection always matches the one being updated, making the importance sampling ratio exactly 1." So there is no inner-epoch / mini-batch reuse either — the usual second source of off-policyness in PPO-style loops is absent by construction.

**Scope check (does §4.1.2 cover MOPD?).** Yes. §4.1.2 opens "Throughout the entire Cascade RL process…", §4 is titled "Cascade RL and Multi-Domain On-Policy Distillation", and §4.1 says "we integrated the Multi-Domain On-Policy Distillation (MOPD) along the Cascade RL process". MOPD is §4.4, a stage *within* the Cascade. Appendix B Table 8 likewise groups MOPD with IF-RL and Multi-domain RL under "Cascade RL".

**Tension worth recording.** §4.1.2 claims the single-update scheme makes "the importance sampling ratio exactly 1", yet §4.4 still computes $r_t = \pi^{\text{train}}/\pi^{\text{inf}}$ and truncates it to $[0.5, 2.0]$. These are consistent only if §4.1.2's "exactly 1" is read *algorithmically* (same weights, no policy lag) while §4.4's $r_t$ is the *numerical* residual between two engines holding those same weights. A band as wide as $[0.5, 2.0]$ is a large tolerance for pure numerical noise, but nothing in the paper attributes any part of it to staleness.

**Staleness measured:** not measured. **Range studied:** N/A (single point: zero staleness).

**The one "asynchronous" in the paper is unrelated** (§4.7.2, Code RL): "we deploy an asynchronous reward verification server that completes each batch in 427.2 seconds across 384 CPU cores." That is async *reward verification*, a CPU-side sandbox pool, not async rollout generation; and it is in the Code RL stage, not MOPD.

### Objective

`rkl_pg` — sampled-token score-function estimator of reverse KL, per the sweep vocabulary. Explicitly *not* full-vocab: "The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary" (§4.4).

Direction check: the advantage is $\log \pi^{\text{teacher}} - \log \pi^{\text{student}}$, used as a per-token advantage on student-sampled tokens. Descending it drives the student toward the teacher under the student's own visitation — i.e. KL(student ‖ teacher), reverse KL by our convention. Conforms.

Verbatim (§4.4, Eqs. 2–4):

$$a_t^{\text{MOPD}} = \log \pi^{\text{domain}_i}(y_t \mid s_t) - \log \pi^{\text{train}}(y_t \mid s_t) \tag{2}$$

$$r_t = \frac{\pi^{\text{train}}(y_t \mid s_t)}{\pi^{\text{inf}}(y_t \mid s_t)}, \quad w_t = \mathrm{sg}[r_t] \, \mathbf{1}[\epsilon_{\text{low}} \leq r_t \leq \epsilon_{\text{high}}] \tag{3}$$

$$\mathcal{L}_{\text{MOPD}} = -\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi^{\text{inf}}(\cdot \mid x)} \left[\frac{1}{|\mathcal{V}(y)|} \sum_{t \in \mathcal{V}(y)} w_t \, \mathrm{sg}\big[a_t^{\text{MOPD}}\big] \log \pi^{\text{train}}(y_t \mid s_t)\right] \tag{4}$$

The GRPO objective used by every *other* Cascade stage is Eq. 1 (§4.1.2), REINFORCE with group-normalized rewards and no KL term. MOPD replaces the group-normalized reward with $\mathrm{sg}[a_t^{\text{MOPD}}]$; $\mathcal{L}_{\text{MOPD}}$ is the whole stage objective (no task reward, no SFT term, no KL-to-reference).

### Off-policy correction

**Importance ratio: yes, but for engine mismatch, not staleness.**

- Form: **per-token**, $r_t = \pi^{\text{train}}(y_t\mid s_t) / \pi^{\text{inf}}(y_t\mid s_t)$ (Eq. 3). Not per-sequence.
- Treatment: **truncation by masking** — $w_t = \mathrm{sg}[r_t]\,\mathbf{1}[\epsilon_{\text{low}} \le r_t \le \epsilon_{\text{high}}]$. The ratio is *stop-gradiented* and *zeroed outside the band* (not clipped-to-boundary as in PPO). Tokens outside the band contribute nothing.
- Values: "For truncated importance weighting, we set $\epsilon_{\text{low}}=0.5$ and $\epsilon_{\text{high}}=2.0$" (§4.4).
- Stated purpose, verbatim: "Because responses are sampled under $\pi^{\text{inf}}$ but optimized under $\pi^{\text{train}}$, we apply truncated importance weighting to account for train–inference mismatch" (§4.4). **"train–inference mismatch" is the only justification given.** No staleness, lag, or replay is invoked.
- IcePop-style masks: the $\mathbf{1}[\cdot]$ factor is functionally a zero-out mask on discrepant tokens (the same shape as IcePop / MiMo's Eq. 8 gate), but the paper does not name IcePop and keeps $\mathrm{sg}[r_t]$ as a multiplicative weight *inside* the band, where MiMo's gate is `sg[ratio] or 0`. Nemotron's Eq. 3 and MiMo's Eq. 8 are in fact the same object.
- Additional mask: $\mathcal{V}(y)$, "the set of valid response tokens retained by the token mask" (§4.4) — a validity/length mask, not an off-policy correction.
- Gradient path: through $\log \pi^{\text{train}}(y_t\mid s_t)$ only ($\mathrm{sg}$ on both $r_t$ and $a_t^{\text{MOPD}}$).

### Results: async vs sync

**None.** The paper contains no async-vs-sync comparison at any staleness level, no throughput-vs-quality trade curve, and no staleness ablation — because it never trains on stale data.

The only comparisons reported for the MOPD stage are *objective* comparisons at zero staleness:

| Comparison | Benchmark | Baseline | MOPD | Steps | Location |
|---|---|---|---|---|---|
| MOPD vs GRPO (math-only) | AIME25 (avg@64) | GRPO 89.9 → 91.0 @ 25 steps | 92.0 within 30 steps (recovers teacher level) | — | Fig. 3(c), §4.4 |
| MOPD vs RLHF | ArenaHard v2, Hard Prompt | Initial 71.5; RLHF 81.7 @100, 80.7 @160 | **85.5** @ 52 | 52 | Table 3, §4.4 |
| MOPD vs RLHF | ArenaHard v2, Creative Writing | Initial 40.6; RLHF 68.6 @100, 71.2 @160 | **71.0** @ 52 | 52 | Table 3, §4.4 |

Fig. 3(a) tracks "Reverse KL" (≈0.03 → ≈0.01 over ~60 steps) and Fig. 3(b) `grad_norm` — training-dynamics diagnostics, not staleness measurements.

### Explanation offered

The paper explains why it is **synchronous**, and the reasons are exactly the ones an async-tolerance study would need to rebut.

- Policy-match / unit-ratio argument (§4.1.2, verbatim): "At each iteration, we generate a group of $G$ rollouts from the current policy $\pi_\theta$ and then perform a *single* gradient update. This ensures that the policy used for data collection always matches the one being updated, making the importance sampling ratio exactly 1. This on-policy setup contributes to stable RL training and mitigates entropy collapse."
  - Mechanism named: **entropy collapse** plus generic instability. Notably the stated failure mode of off-policyness here is *distributional collapse of the student*, not estimator variance.
- Stability/accuracy assertion (§4.1.2, verbatim): "we use Group Relative Policy Optimization (GRPO) algorithm (Shao et al., 2024) with **strict on-policy training** following Nemotron Cascade (Wang et al., 2025). We adopt on-policy training for improved stability and higher accuracy."
  - Mechanism named: none beyond the assertion; presented as inherited practice from Nemotron-Cascade 1.
- Stage-level stability claim for MOPD itself (§4.1.1, "Stabilization through On-policy Distillation", verbatim): "We find that Multi-domain On-policy Distillation (§4.4) serves as a critical stabilization point in this ordering. It is effective at recovering benchmark performance that may have regressed during earlier, more specialized stages of the cascade RL, leading to a more balanced and robust final policy model."
- Why MOPD converges fast (§4.4, verbatim): "MOPD provides a dense token-level distillation advantage, whereas GRPO relies on a sparse sequence-level outcome reward that is shared across all generated tokens. This makes MOPD substantially more sample- and step-efficient in practice." And: "These results show that the dense token-level advantage in on-policy distillation lead to much faster training convergence."
  - Mechanism named: **credit-assignment density**, orthogonal to staleness.
- Why same-family teachers help (§4.4, verbatim): "because these teachers are derived from the same SFT initialization, they share the same tokenizer and vocabulary as the student, reducing distribution shift and avoiding additional alignment issues."
  - Mechanism named: **support/distribution mismatch** — but between *student and teacher*, not between behavior and current student.

No theory, no variance analysis, no staleness-tolerance argument in either direction.

### Quotes

Every statement in the paper bearing on async / sync / stale / off-policy / replay, with location.

1. §4.1.2 (RL Training Configuration): "Throughout the entire Cascade RL process, we use Group Relative Policy Optimization (GRPO) algorithm (Shao et al., 2024) with strict on-policy training following Nemotron Cascade (Wang et al., 2025). We adopt on-policy training for improved stability and higher accuracy. We conduct our training using the Nemo-RL repository (NVIDIA, 2025)."
2. §4.1.2: "At each iteration, we generate a group of $G$ rollouts from the current policy $\pi_\theta$ and then perform a *single* gradient update. This ensures that the policy used for data collection always matches the one being updated, making the importance sampling ratio exactly 1."
3. §4.1.2: "This on-policy setup contributes to stable RL training and mitigates entropy collapse."
4. §4.1.2: "In addition, we remove KL divergence term entirely, which simplifies the GRPO objective to the standard REINFORCE objective (Williams, 1992) with group-normalized rewards and token-level loss (Yu et al., 2025)."
5. §4.4 (MOPD): "Let $\pi^{\text{inf}}$ denote the student policy used for response generation in the inference engine, and let $\pi^{\text{train}}$ denote the student policy optimized by the training engine."
6. §4.4: "Because responses are sampled under $\pi^{\text{inf}}$ but optimized under $\pi^{\text{train}}$, we apply truncated importance weighting to account for train–inference mismatch." ← **the whole justification for Eq. 3; "mismatch", not staleness**
7. §4.4: "For truncated importance weighting, we set $\epsilon_{\text{low}} = 0.5$ and $\epsilon_{\text{high}} = 2.0$."
8. §4.4: "In later experiments, we find that using 512 prompts with rollout size 1 yields slightly more stable optimization while producing similar final results."
9. §4.7.2 (Code RL, **not MOPD**): "We adopt the strict binary reward function to avoid potential reward hacking and keep the whole training to be fully on-policy for stability."
10. §4.7.2 (Code RL, **not MOPD**): "To support the resulting verification throughput of $128\times16 = 2{,}048$ code executions per RL step, we deploy an asynchronous reward verification server that completes each batch in 427.2 seconds across 384 CPU cores." ← the *only* occurrence of "asynchronous" in the paper; refers to reward verification, not rollout generation.
11. §4.4: "The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary."

Absent from the entire document (verified by exhaustive grep over the v2 PDF text and by a full-document pass over the v2 HTML, 2026-09-14): *stale*, *staleness*, *off-policy*, *replay buffer*, *partial rollout*, *rollout lag*, *behavior policy*, *mini-batch*, *pipelined*, *disaggregated*. The string "async"/"asynchronous" occurs exactly once, at quote 10.

### Not in the source

- **No stale-rollout experiment of any kind.** No sync baseline vs async arm, because there is no async arm.
- **No staleness tolerance statement** — the paper never says whether MOPD *could* tolerate off-policy data; it simply never generates any.
- **No decomposition of the $[0.5, 2.0]$ band.** With one gradient step per batch and identical weights in both engines, the band should be absorbing pure numerical/engine noise, yet a factor-of-2 tolerance is very wide for that. The paper gives no measured distribution of $r_t$, no fraction of tokens masked out, and no ablation over $\epsilon$.
- **No throughput / wall-clock data for MOPD.** Timing is reported only for the Code RL reward server (427.2 s/batch, §4.7.2), so the cost of insisting on synchrony cannot be recovered.
- **No rollout-generation architecture.** NeMo-RL is named but colocated-vs-disaggregated placement, weight-sync protocol, and generation backend are unstated.
- **Internal hyperparameter inconsistency** (see Definitions and facts above): §4.4 prose lr $2\times10^{-6}$ vs Appendix B Table 8 lr $3\times10^{-6}$ for MOPD; §4.4 "40-50 steps" vs Table 8 / Table 3 "52".
- **No MOPD teacher-selection rule at the example level.** "We then select a domain teacher $\pi^{\text{domain}_i}$ for that training example" — the routing is by the prompt's source pool, but no mixing weights or tie-breaking are given.
