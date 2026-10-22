# Xie, Zhang, Xie & Yang 2026 — Trust Region Policy Distillation (TOP-D)

**arXiv 2607.04751v1** (6 Jul 2026; only version as of 2026-09-14). Zhengpeng Xie (internship), Li Lyna Zhang (project leader), Zeke Xie (HKUST-GZ, corresponding), Mao Yang (Microsoft). Sources: PDF `https://arxiv.org/pdf/2607.04751` (curl + pypdf text; p. 9 rendered with pymupdf at 600–700 dpi to read Fig. 5), arXiv HTML `https://arxiv.org/html/2607.04751v1` (equation LaTeX via `alttext` attributes), abs page via WebFetch — all 2026-09-14. No code release and no project URL anywhere in the paper.

```bibtex
@article{xie2026trustregion,
  title   = {Trust Region Policy Distillation},
  author  = {Zhengpeng Xie and Li Lyna Zhang and Zeke Xie and Mao Yang},
  journal = {arXiv preprint arXiv:2607.04751},
  year    = {2026}
}
```

## Tier

- **B** — a genuine 2×2 over {OPD reward, TOP-D reward} × {1 update per rollout batch, 16 mini-batch updates per rollout batch} in Fig. 5 (curves only, no table), with an explanation: the unbounded reward makes the gradient variance diverge (Thm 4.2), which is what makes data reuse unsafe; the bounded reward restores monotonic improvement for the internal iterations (Thm 4.9).
- **D** — several quotable statements about "safely break[ing] the strict on-policy data-reuse barrier of standard OPD" (§1, §3.2, §3.3, §5.3, §6).
- **F** — off-policy data reuse ("internal trust region iterations") is one of the paper's two headline contributions, alongside the proximal teacher; the two are presented as a joint design. Partial F: the reuse is *within* a rollout batch, not a pipelined generator or a replay buffer across steps.
- Not **A** — the theory is about the *magnitude/variance* of the per-token reward (bounded `log(αρ+1−α)` vs unbounded `log ρ`), not about distribution-level vs sampled-token estimators. Both TOP-D and the standard-OPD baseline here are sampled-token `rkl_pg`; the paper never runs a full-vocab or top-k variant, so it says nothing about which *estimator* tolerates staleness better.
- Not **C** (an explanation is given), not **E** (academic paper, 4×8 H200 nodes, App. "Computational Resources"), not **G** (it is a main point).

## Setting

- Students: Qwen3-8B-Base and Qwen3-1.7B-Base (§5.1 "Models and baselines").
- Teachers: Qwen3-30B-A3B-Instruct-2507 (main) and Qwen3-14B (second column of Table 3). **Teacher fixed: yes** — π\* is an external, frozen model; the "proximal teacher" `π̃*` is *not* a trained model but an algebraic interpolation constructed at each step from π\* and the behaviour student, and it never appears in the implementation ("Thanks to mathematics, we do not need to explicitly construct a proximal teacher to obtain the new reward `r̃_k`", §3.1).
- Data: DAPO-Math-17k for training; eval on AIME24/25/26 (avg@32), AMC23 / MATH-500 / Olympiad (avg@8) (§5.1 "Datasets and benchmarks", Tables 2–4).
- Baselines: GRPO, DAPO, standard OPD (§5.1). Note the RLVR baselines *also* use 16 mini-batches per rollout batch (Table 5), so only the OPD baseline is strictly on-policy.
- Framework/code: **not named.** §1 claims compatibility with Megatron-LM [15] / PyTorch FSDP [22] and vLLM [8] / SGLang [23], but never states which was used. Hardware: "4 compute nodes, with each node containing 8 H200 GPUs (amounting to 32 GPUs in total)" (App. "Computational Resources").

## Async mechanism

**Mini-batch reuse of one rollout batch — no pipelining, no replay buffer, no cross-step staleness.**

- The student's own earlier weights are the stale behaviour policy: `π_θold ← π_θ` is refreshed once per global step (Alg. 1 line 2), the batch is rolled out from `π_θold` (line 5), and the batch is then consumed in 16 mini-batch gradient steps (lines 10–13, Table 5). So mini-batch *m* (m = 0…15) trains `π_θ` on samples drawn from a student that is *m* optimizer steps old.
- Exact numbers (§5.1 "Implementation details" and Table 5): group size G = 8; "the global batch size is set to 512 prompts (4096 samples) and the mini-batch size is set to 32 prompts (256 samples)"; **Mini-batches = 16**; **Off-policy epoch = 1**. 512/32 = 16 gradient updates per rollout batch ⟹ staleness range **0–15 optimizer steps**, never more. E in Alg. 1 ("internal epochs") is 1 in all reported runs.
- Standard-OPD baseline row of Table 5: "Mini-batch size 512, Mini-batches 1, Off-policy epoch –, ε_low –, ε_high –" ⟹ exactly **1 update per rollout batch, strictly on-policy, no importance ratio at all**.
- Also stale within a global step: **the reward itself**. Alg. 1 line 6 computes `r̃^i_k ← log(α π*(y^i_k|x,y^i_<k)/π_θold(y^i_k|x,y^i_<k) + 1−α)` — the *behaviour* student is in the denominator, once, before the inner loop. So the proximal teacher is anchored at `π_θold` (`π̃*_k = T(π_k) = απ* + (1−α)π_k`, Eq. 11, "let `π̃*_k = T(π_k)` be the **fixed** proximal teacher", §4.3) and the token-level advantages `Â^i_k` are frozen across all 16 inner updates.
- Not studied: staleness > 1 rollout batch, buffer size, mixing ratios, any "off-policyness" metric, or an epoch sweep (E is never varied).

## Objective

**`rkl_pg`** — sampled-token score-function estimator of reverse KL, with the per-token reward algebraically smoothed by the proximal teacher, then wrapped in a GRPO/PPO-style clipped surrogate with group-normalised token-level advantages.

Reverse KL in our convention (student ‖ teacher), Eq. (2):

```latex
\theta^{*}=\arg\min_{\theta}\mathcal{L}(\theta)=\arg\min_{\theta}\mathbb{E}_{x\sim\mathcal{D}_{x}}\left[\mathcal{D}_{\mathrm{KL}}(\pi_{\theta}(\cdot\mid x),\pi^{*}(\cdot\mid x))\right]
=\arg\max_{\theta}\mathbb{E}_{x\sim\mathcal{D}_{x},y\sim\pi_{\theta}(\cdot\mid x)}\left[\log\frac{\pi^{*}(y\mid x)}{\pi_{\theta}(y\mid x)}\right]=\arg\max_{\theta}J(\theta)
```

Score-function form, Eq. (3), with `ρ_k = π*(y_k|x,y_<k)/π_θ(y_k|x,y_<k)` and per-token reward `r_k = log ρ_k`:

```latex
\nabla_{\theta}J(\theta)=\mathbb{E}_{x\sim\mathcal{D}_{x},y\sim\pi_{\theta}(\cdot\mid x)}\left[\sum_{t=1}^{\lvert y\rvert}\nabla_{\theta}\log\pi_{\theta}(y_{t}\mid x,y_{<t})\cdot\left(\sum_{k=t}^{\lvert y\rvert}\log\rho_{k}\right)\right]
```

"However, the cumulative reward `Σ_k r_k` introduces an inherent length bias as well as high variance. In practice, a common approach is to use the immediate reward `r_k` directly." (§2.1).

**Proximal teacher** (unnumbered, §3.1): `π̃*(y_k|x,y_<k) = απ*(y_k|x,y_<k) + (1−α)π_θ(y_k|x,y_<k)`, α ∈ (0,1). The smoothed reward, **Eq. (5)** — this is the `log(αx + 1 − α)` bound the paper is built on:

```latex
\tilde{r}_{k}=\log\frac{\tilde{\pi}^{*}(y_{k}\mid x,y_{<k})}{\pi_{\theta}(y_{k}\mid x,y_{<k})}=\log\left(\alpha\frac{\pi^{*}(y_{k}\mid x,y_{<k})}{\pi_{\theta}(y_{k}\mid x,y_{<k})}+1-\alpha\right)=\log\left(\alpha\rho_{k}+1-\alpha\right)
```

with footnote 4: "`log(αx+1−α) ≥ log(1−α)` when `x ∈ (0,+∞)` and `α ∈ (0,1)`". α = 1 recovers standard OPD. Interpolating in *log*-probability space instead would give only `r̃_k = α r_k`, Eq. (6), "which is merely a naive scaling of the original OPD reward" (§3.1).

**Token-level return**, Eq. (8) (immediate reward + length-normalised future return):

```latex
\tilde{R}_{k}^{i}=\tilde{r}_{k}^{i}+\frac{1}{\lvert y^{i}\rvert-k}\sum_{j=k+1}^{\lvert y^{i}\rvert}\tilde{r}_{j}^{i}
```

then `Â^i_k = (R̃^i_k − µ)/σ` with µ, σ taken over **all tokens of all G responses for that prompt** ("token-level normalization across the responses generated for a given prompt", Fig. 4 caption; "µ = mean{R̃^i_k} and σ = std{R̃^i_k}, where i = 1,…,G and k = 1,…,|y^i|", §3.2). Because the mean is subtracted, the optimised direction is a GRPO surrogate built on the RKL per-token reward, not the RKL gradient itself — the paper does not comment on this.

**Variance bound, Thm 4.2 / Eq. (10)**: `Var(g̃_k) ≤ M²|V| max{(log(1−α))², C*α}`, with "`As α→1⁻`, TOP-D recovers standard OPD, with its upper bound diverging (`(log(1−α))² → ∞`) to reflect OPD's unbounded variance." (Remark 4.3).

## Off-policy correction

**PPO-style token-level importance ratio with symmetric clipping.** Eq. (7):

```latex
J(\theta)=\mathop{\mathbb{E}}_{x\sim\mathcal{D}_{x},\ \{y^{i}\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x)}\left[\frac{1}{\sum_{i=1}^{G}\lvert y^{i}\rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert y^{i}\rvert}\left\{\min\left[p_{t}^{i}\cdot\hat{A}_{t}^{i},\mathrm{clip}\left(p_{t}^{i},1-\epsilon,1+\epsilon\right)\cdot\hat{A}_{t}^{i}\right]\right\}\right]
```

- Ratio: **per-token**, `p^i_t = π_θ(y^i_t|x,y^i_<t) / π_θold(y^i_t|x,y^i_<t)` (§3.2).
- Clipping: symmetric, **ε_low = ε_high = 0.2** (Table 5). No Clip-Higher (DAPO's 0.28 is used only for the DAPO baseline).
- No IcePop-style mask, no ratio truncation, no sequence-level ratio, no KL penalty to a reference policy, no truncated-support/top-k reward.
- The *reward* is not importance-corrected: `r̃` is computed against `π_θold` once per rollout batch (Alg. 1 line 6) and held fixed; only the policy-gradient term carries the ratio.
- Standard OPD baseline: **no correction of any kind** — and none needed, since it takes 1 update per batch (Table 5).

## Results: async vs sync

The clean comparison is the Fig. 5 ablation on **Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct-2507**, which forms a 2×2 over reward type and data reuse. Values below are read off the rendered figure (±0.5 pp); the paper gives no table for Fig. 5.

| Reward | Data reuse | Fig. 5 curve | AIME24 | AIME25 | AIME26 |
|---|---|---|---|---|---|
| OPD `log ρ` (α=1) | none (1 update/batch) | "Standard OPD" (grey), ends ≈280 steps | ≈6.4 end (peak ≈8.9 early, dips to ≈2 at step 100) | ≈6.3 end (peak ≈7.6) | ≈4.8 end (peak ≈5.8) |
| OPD `log ρ` (α=1) | 16 mini-batches | "TOP-D (α = 1.0)" (pink), ends ≈180 steps | ≈8.3 end, plateau 7–8.5, noisy | ≈8.8 end | ≈6.3 end |
| TOP-D `log(αρ+1−α)` | none | "TOP-D w/o off-policy" (tan), ends ≈200 steps | ≈11.3 end | ≈8.3 end | ≈7.2 end (peak ≈7.8) |
| TOP-D `log(αρ+1−α)` | 16 mini-batches | "TOP-D (α = 0.1/0.2/0.3)" (orange/red), ends ≈410 steps | ≈18–19.6 end; ≈16.5–17 at step 200 | ≈16.5–17 end; ≈13 at step 200 | ≈11.4–13 end; ≈11 at step 200 |

At a **matched ~200 steps** the ordering is TOP-D (≈17 / ≈13 / ≈11) > TOP-D w/o off-policy (≈11.3 / ≈8.3 / ≈7.2) > TOP-D α=1.0 (≈8.3 / ≈8.8 / ≈6.3) ≳ standard OPD (≈7 / ≈6.5 / ≈5). Reuse alone (grey → pink) buys ≈ +1.5 to +2.5 pp; the bounded reward alone (grey → tan) buys ≈ +2.0 to +4.9 pp; together they buy ≈ +9 to +14 pp — i.e. the two interventions are strongly **super-additive** in this figure, which is the paper's central empirical claim, though it never quantifies the interaction.

α sensitivity: α ∈ {0.1, 0.2, 0.3} curves are indistinguishable in all three panels — "the training stability, convergence speed, and final performance exhibit marginal differences across this interval" (§5.3).

Main-table endpoints (not an async/sync contrast; standard OPD there is the 1-update-per-batch configuration and TOP-D the 16-mini-batch one, so the gap confounds reward and reuse):

| Student ← Teacher | Method | AIME24 | AIME25 | AIME26 | AMC23 | MATH-500 | Olympiad |
|---|---|---|---|---|---|---|---|
| Qwen3-8B-Base ← Qwen3-30B-A3B-Instruct-2507 (Table 2) | OPD | 24.58 | 23.33 | 25.42 | 76.88 | 87.98 | 59.29 |
| | TOP-D | **50.42** (+25.84) | **34.06** (+10.73) | **44.06** (+18.64) | **88.13** (+11.25) | **91.23** (+3.25) | **64.67** (+5.38) |
| Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct-2507 (Tables 3, 4) | OPD | 8.96 | 7.50 | 5.94 | 45.70 | 71.13 | 35.83 |
| | TOP-D | **20.31** (+11.35) | **17.71** (+10.21) | **13.75** (+7.81) | **54.38** (+8.68) | **77.33** (+6.20) | **43.06** (+7.23) |
| Qwen3-1.7B-Base ← Qwen3-14B (Table 3) | OPD | 7.81 | 8.33 | 5.63 | — | — | — |
| | TOP-D | **12.81** (+5.00) | **14.69** (+6.36) | **9.69** (+4.06) | — | — | — |

Throughput / wall-clock: **none reported**. The reuse claim is about sample efficiency only; the paper repeatedly states "TOP-D introduces zero additional computational overhead" (Abstract, §1) but gives no step-time or GPU-hour comparison.

## Explanation offered

The mechanism is **reward-magnitude-driven gradient variance**, not support mismatch or estimator variance in the usual sense:

- Why standard OPD cannot reuse: "In extreme cases where the teacher assigns a near-zero probability to a generated token (`π* → 0`), the immediate reward `r_k = log ρ_k` diverges to negative infinity. This unbounded penalty destabilizes the policy gradient, making the optimization process extremely fragile." (§3.1)
- "While standard OPD relies on continuous online sampling, it fundamentally operates in a strictly on-policy manner. In the naive formulation, once the student policy is updated, the previously generated trajectories are immediately discarded. This results in notoriously low sample efficiency, lacking the capacity to reuse data." (§3.2)
- Why TOP-D can: "Built upon this principle, the *internal trust region iterations* decouple the behavior and target policies, breaking the strict on-policy data-reuse barrier. Crucially, these mechanisms are not merely empirical heuristics but share deep theoretical synergy." (§3.3) — i.e. reuse is only licensed *after* the reward has been bounded.
- Theory closing the loop (§4.3, Remark 4.5, Remark 4.10): the proximal-teacher operator contracts at rate (1−α) but leaves an asymptotic gap `limsup d(π_{k+1}, π*) ≤ ε_∞/α` (Thm 4.4, Eq. 13); "to break through the precision ceiling without compromising stability (i.e., keeping α small), the only viable path is to strictly minimize the single-step optimization error ‖ε_k‖₁. This theoretical bottleneck motivates the necessity of introducing *internal trust region iterations*" (Remark 4.5). The monotonic-improvement bound (Thm 4.9, Eq. 14) then gives a decreasing sequence `E[D_KL^(0)] ≥ E[D_KL^(1)] ≥ … ≥ E[D_KL^(n)]` (Eq. 15) over the inner mini-batch updates against the *fixed* proximal teacher `π̃*_k`, and Pinsker + Jensen turn `E[D_KL^(n)] ≤ ½δ²` into `‖ε_k‖₁ ≤ δ`.
- The variance bound is the formal statement of "α is a variance dial": "This effectively designates α as a direct variance controller. As `α→1⁻`, TOP-D recovers standard OPD, with its upper bound diverging (`(log(1−α))² → ∞`) to reflect OPD's unbounded variance. Conversely, as `α→0⁺`, the variance vanishes." (Remark 4.3)
- Empirical reading of the ablation: "by disabling the internal trust region iterations (denoted as *w/o off-policy*), we force the optimization into a strictly on-policy regime. This leads to a drastic drop in sample efficiency, requiring significantly more training steps to converge and climb the performance curve." (§5.3)
- Note the theory does **not** cover the actual algorithm's staleness: Thm 4.9 is a TRPO-style bound in total variation, and the implementation replaces it with PPO clipping at ε = 0.2 with no stated correspondence between ε and the TV radius.

## Quotes

- "We propose TOP-D, a plug-and-play distillation paradigm that smooths the unbounded reward signal and **safely breaks the strict on-policy data-reuse barrier of standard OPD**." (§1, contributions)
- "Inspired by modern trust region algorithms [11, 12, 17], we analogously decouple the *behavior policy* `π_θold` and the *target policy* `π_θ` to inherit their high sample efficiency." (§3.2)
- "To execute the trust region iterations, we follow the same off-policy setting as RLVR by dividing the same batch of data into multiple mini-batches, where the global batch size is set to 512 prompts (4096 samples) and the mini-batch size is set to 32 prompts (256 samples)." (§5.1, Implementation details)
- "Standard OPD suffers from two fundamental bottlenecks: optimization fragility and low sample efficiency. TOP-D systematically resolves both through a unified design." (§3.3)
- "On the other hand, by disabling the internal trust region iterations (denoted as *w/o off-policy*), we force the optimization into a strictly on-policy regime. This leads to a drastic drop in sample efficiency, requiring significantly more training steps to converge and climb the performance curve." (§5.3)
- "These empirical results clearly demonstrate that both the *external proximal teacher* and the *internal trust region iterations* are indispensable components for achieving optimization stability and rapid performance improvement." (§5.3)
- "Theoretically, TOP-D bounds gradient variance via an external proximal teacher and **guarantees monotonic improvement with safe off-policy data reuse** through internal trust region iterations." (§6)
- "Recent studies have devoted considerable engineering efforts to mitigate this instability through various techniques, including teacher-student mixed sampling [3], reward clipping [7], top-p sampling [2], **off-policy cold starts** [9], and scaling to full-vocabulary supervision [1]. However, these mitigation strategies remain largely empirical heuristics…" (§1) — [7] = REOPOLD, [9] = Li et al. 2604.13016, [2] = Fu et al. 2603.25562, [1] = DeepSeek-V4.
- Table 1 labels the sampling of "Supervised Fine-Tuning" as "off-policy" and both OPD and TOP-D as "on-policy" — i.e. the paper's own taxonomy table does *not* count mini-batch reuse as off-policy.

## Not in the source

- **No run of "standard OPD + off-policy reuse" under the OPD hyperparameters.** The closest thing is "TOP-D (α = 1.0)", which is the OPD reward *with* 16 mini-batches, token-level normalised advantages and ε = 0.2 clipping — so it also differs from the standard-OPD baseline in advantage estimation and clipping, not only in reuse. The 2×2 in the Results table above is therefore not perfectly controlled.
- **No numeric table for Fig. 5** — the ablation exists only as learning curves; every reuse-vs-no-reuse number here is read off the plot.
- **"Training Steps" is never defined.** With 16 mini-batches per rollout batch, a "step" could be a global rollout batch or a single optimizer update, and the two readings differ by 16× in rollout budget. Table 5's "Off-policy epoch 1" and the limitation "approximately 200 to 400 update steps" point weakly to global steps, but the paper never says. The curves also terminate at different step counts (≈180 / 200 / 280 / 410), so no two curves are compared at a matched budget of either kind.
- **No sweep over the amount of staleness**: mini-batch count (16) and epochs (E = 1) are fixed; there is no 2/4/8/32-mini-batch curve, no clip-ε sweep, and no measurement of how far `π_θ` drifts from `π_θold` (no clip fraction, no KL(π_θ‖π_θold), no "off-policyness" metric).
- **No cross-step staleness at all**: no pipelined generator, no replay buffer, no lag between the rollout snapshot and the trainer beyond one batch — so the paper says nothing about the async-OPD regime proper.
- No throughput or wall-clock numbers; "zero additional computational overhead" is asserted but never measured.
- No error bars / seeds anywhere (all tables are single runs), no full-vocab or top-k OPD variant, no framework or code release, and Appendix B omits the number of training steps per method.
- The advantage is z-normalised (mean subtracted) but the paper still calls the objective reverse-KL minimisation; the effect of the baseline subtraction on the RKL interpretation is not discussed.
