# Chen, Zhang & Wu 2026 — f-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control

**arXiv 2605.17862v1** (18 May 2026; only version as of 2026-09-14). Xianwei Chen, Shimin Zhang, Jibin Wu (The Hong Kong Polytechnic University). Sources: PDF `https://arxiv.org/pdf/2605.17862` (curl + pypdf text, pages 2/7/8 rendered with pymupdf, 2026-09-14); arXiv HTML and abs page via WebFetch (2026-09-14). No code release, no project URL (abs page, 2026-09-14). The paper's own name is "𝒇-OPD" (italic f); written `f-OPD` here.

```bibtex
@article{chen2026fopd,
  title   = {f-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control},
  author  = {Xianwei Chen and Shimin Zhang and Jibin Wu},
  journal = {arXiv preprint arXiv:2605.17862},
  year    = {2026}
}
```

## Tier

- **B** — side-by-side sync vs async (lag 0/2/4/8) numbers on three tasks (Fig. 3a, Fig. 4, Tables 1–2) with an explanation (rollout drift + supervision drift decomposition, Prop. 3.2; entropy destabilisation, §5.2; horizon compounding, Prop. A.4).
- **D** — many quotable statements on staleness/async OPD (§1, §5.2, App. D).
- **F** — async OPD *is* the paper: the failure mode of vanilla async OPD and a fix (freshness weighting + rollout-anchored KL + adaptive refresh) are the main contribution.
- Not **A**: the theory (Prop. 3.2) bounds the sync/async objective gap for a generic loss ℓ; it never compares sampled-token vs distribution-level estimators, so it says nothing about which objective tolerates staleness better.
- Not **E** (academic paper, ~16k H100-hours, App. B.1), not **C**/**G** (explanation is given; async is the main point).

## Setting

- Students / teachers (§5.1 "Protocols", App. B.1 "Student and teacher checkpoints", Table 3): reasoning = Qwen2.5-Math-7B-Instruct ← Qwen2.5-Math-72B-Instruct; tool-use and coding-agent = Qwen3-8B ← Qwen3-Coder-30B-A3B-Instruct. "All teachers are frozen." (App. B.1). Table 1 rows reproduced by the authors "share the same base model (Qwen3-8B)" (§5.3).
- Teacher fixed: **yes**. "Teacher labels are greedily generated in every setting, so supervision drift reflects asynchronous state drift rather than teacher sampling noise." (App. B.1). App. A.5: "The key point is that this term measures supervision drift, not teacher-parameter drift."
- Protocol: "We follow the student-teacher distillation setup introduced by Thinking Machines Lab [40]." (§5.1). Rollouts: "temperature 0.7 and top-p 0.95 for reasoning and tool use, and temperature 0.6 for coding." (App. B.1).
- Tasks (§5.1 "Task taxonomies", App. B.1): (i) short-horizon reasoning — DAPO-Math-17K prompts, eval MATH500 Avg@1; (ii) medium-horizon tool-use — same prompts in a "deterministic ReTool-style code-interpreter harness", ≤6 tool calls / ≤10 assistant turns, eval held-out 1,000 olympiad problems Avg@4 + invalid-tool rate + peak-final drop; (iii) long-horizon coding agent — mini-SWE-agent scaffold, eval 250 SWE-bench Verified issues (50 pilot), single-run resolve rate + repeated-loop / premature-stop / post-patch-regression rates. Five seeds everywhere. "Collapse is operationalized as a run whose final task score is more than 40% below its peak and remains below 90% of peak for the last 10% of training." (App. B.1).
- Schedule/compute: "uniform 400-step training schedule" (App. B.1); "about 16k H100-80GB GPU-hours in total: 3k for reasoning, 5k for tool-use, and 8k for coding. Reasoning and tool-use runs use 16 H100-80GB GPUs, while coding runs use 24" (App. B.1).
- Framework/code: not named. §1 cites HybridFlow/verl [8], slime [9], NeMo RL [10] as "modern OPD systems [that] widely adopt asynchronous mode", but the paper never says which system it used; no code URL.

## Async mechanism

- Pipelined trainer/generator with an active replay buffer (Fig. 1a: rollout, teacher supervision and training on separate "resource groups" RG1–RG3, overlapped). "Under asynchronous execution, training proceeds on an active buffer B_t composed of samples generated and graded at current or earlier policy update steps." (§3.1).
- What is stale: **the behaviour student** π_θ^{r(i)} that generated sample i vs the current student π_θ^t (rollout drift), and — for multi-turn tasks — the **teacher-conditioning context** c_i^{r(i)} cached at grading time vs the context c_i^t the same sample would get if relabelled now (supervision drift). The teacher itself is fixed; supervision drift is context drift caused by stale student actions, i.e. still student-side staleness: "stale early actions can create the context drift that later enlarges D_i^sup" (App. A.4); "If the teacher is fixed, the effective context is unchanged, and labeling is deterministic, then π_teacher(·|c_i^t) = π_teacher(·|c_i^{r(i)}) ⟹ D_i^sup = 0." (Remark A.6). For pure reasoning D^sup is not even logged: "Supervision-side replay diagnostics are logged only when current relabeling contexts can be reconstructed reliably; in the current protocol, that means the tool-use ablations and the coding-agent runs, not pure reasoning." (App. B.1).
- Staleness measures (§3.3, §4.1, Eq. 10–11): update lag τ_i = t − r(i) (steps); replay diagnostics on the stored prefixes at aligned token positions: D_i^roll = mean_h KL(π_θ^t(·|x_{i,h}) ‖ π_θ^{r(i)}(·|x_{i,h})) (note: current ‖ rollout), D_i^sup = mean_h KL(π_teacher(·|c_{i,h}^t) ‖ π_teacher(·|c_{i,h}^{r(i)})); freshness f_i = (τ_i+1)^{-1} exp(−α√D_i^roll − β√D_i^sup) (Eq. 11). Both KLs are "monotone surrogates rather than exact estimators" of the TV terms in Prop. 3.2 (§3.3; via Pinsker).
- Lag definition: "Lag k means that the optimizer is allowed to apply roughly k update steps between the snapshot used for rollout generation and the current training state consuming that sample." (App. B.1 "Async pipeline definition").
- Range studied: lag ∈ {0, 2, 4, 8} for the failure analysis (§5.2, Fig. 3, Fig. 5); all method comparisons "use the hardest asynchronous setting, lag = 8" (App. B.1 "Compared methods"). Measured drift at lag 8 (Fig. 3b, read from plot): rollout KL ≈ 0.8e-2 (reasoning), 1.4e-2 (tool-use), 2.1e-2 (coding); supervision KL ≈ 6e-2 (tool-use), 9–10e-2 (coding). Buffer size, rollouts per prompt and batch size are not given.

## Objective

- Vocabulary: **unspecified, presumably `rkl_pg`** — the paper says only that it follows the Thinking Machines Lab OPD setup [40] (§5.1), whose per-token loss is the sampled-token reverse KL used as a per-token advantage. Nowhere in the paper is ℓ instantiated; "reverse", "forward KL" (only for SFT in §2), "top-k" and "importance" never appear in an objective context. Assumption 3.1 evaluates the loss "on a finite support where both student and teacher probabilities are bounded away from zero", which reads like a truncated-support analysis device, not an implementation statement.
- Sync objective, Eq. 1: `J_sync(π_θ^t) = E_{x∼d^t}[ ℓ(π_θ^t(·|x), π_teacher(·|x)) ]`, "where d^t denotes the mixture distribution over trajectory prefixes induced by rolling out π_θ^t across token positions" (§3.1).
- Async objective, Eq. 2: `J_async(π_θ^t) = E_{i∼B_t}[ ℓ(π_θ^t(·|x_i), π_teacher(·|c_i^{r(i)})) ]`; discrepancy Eq. 3: `Δ^t = |J_async(π_θ^t) − J_sync(π_θ^t)|`.
- Sample loss, Eq. 12: `ℓ_i^t = (1/|U_i^sup|) Σ_{h∈U_i^sup} ℓ(π_θ^t(·|x_{i,h}), π_teacher(·|c_{i,h}^t))` — note the *current* context c^t is used in the loss (the stale cached supervision only enters the diagnostics).
- f-OPD objective, Eq. 16: `J(π_θ^t) = E_{i∼C(B_t)}[ σ(f_i − ξ) ( ℓ_i^t + λ R_i^anchor ) ]`, σ = ReLU, R_i^anchor = D_i^roll, C(B_t) = adaptive refresh (Eq. 14–15: refresh when mean freshness f̄^t ≤ κ_f or aligned-coverage ratios M̄^{t,roll} < κ_roll or M̄^{t,sup} < κ_sup; "at most once every 200 optimizer steps", App. B.1).
- Direction of the KLs that *are* written: the anchor/diagnostic KL is KL(current student ‖ rollout student) on stored prefixes (Eq. 7, 10); teacher-side is KL(teacher | current ctx ‖ teacher | stale ctx). Neither is the distillation loss itself.

## Off-policy correction

- **No importance ratio, no clipping, no IcePop-style mask** anywhere (checked full text). The vanilla async baseline "is the bare overlapped pipeline with no freshness-aware control" (App. B.1) — i.e. stale samples are trained on as if on-policy.
- What f-OPD does instead (§4.2, Alg. 1): (1) sample-level freshness weight/gate `w_i = σ(f_i − ξ)` (Eq. 13, ReLU: samples below threshold ξ get weight 0); (2) rollout-anchored regulariser `λ D_i^roll = λ · KL(π_θ^t ‖ π_θ^{r(i)})` on the stale prefixes, "as a trust-region constraint on the stale buffer" (App. D), also multiplied by w_i; (3) adaptive buffer refresh (Eq. 15). Baselines: "Async OPD + fixed refresh" every 10 optimizer steps (called "hard refresh" in Table 1) and "lag-only weighting" `f_i^{lag-only} = (τ_i+1)^{-1}, w_i = σ(f_i^{lag-only} − ξ)` (App. B.1).
- Values of α, β, λ, ξ, κ_f, κ_roll, κ_sup: not reported — "All thresholds and score coefficients are selected on the task-specific pilot split and then fixed for the reported runs." (App. B.1).

## Results: async vs sync

All async numbers are at lag 8 unless stated; five-seed means; throughput = "completed training samples per wall-clock hour, normalized to synchronous OPD" (§5.3).

**Table 1 (coding agent, Qwen3-8B, SWE-bench Verified):**

| Method | Throughput | Resolve ↑ | Post-patch reg. ↓ | Collapse |
|---|---|---|---|---|
| Vanilla SFT | 2.18× | 39.1 | — | 0/5 |
| Vanilla GRPO | 0.92× | 40.2 | — | — |
| Sync OPD | 1.00× | 41.8 | 1.6 | 0/5 |
| Async OPD | 1.61× | 26.8 | 12.1 | 3/5 |
| Async + hard refresh (every 10 steps) | 1.15× | 35.1 | 6.6 | 1/5 |
| Async + lag-only weighting | 1.54× | 34.9 | 7.6 | 2/5 |
| f-OPD | 1.46× | 39.4 | 2.6 | 0/5 |

**Table 2 (ablation, cumulative on top of async OPD):**

| Method | Tool-use Acc avg@4 | Peak-final drop | Coding resolve | Post-patch reg. | Collapse |
|---|---|---|---|---|---|
| Sync OPD | 74.8 | 2.7 | 41.8 | 1.6 | 0/5 |
| Async OPD | 61.6 | 10.4 | 26.8 | 7.4 | 3/5 |
| + freshness weighting | 70.8 | 4.3 | 36.5 | 4.4 | 1/5 |
| + rollout-anchored reg. | 73.2 | 3.4 | 38.1 | 3.2 | 1/5 |
| + adaptive refresh (= f-OPD) | 74.4 | 3.1 | 39.4 | 2.6 | 0/5 |

(Inconsistency: async-OPD post-patch regression is 12.1 in Table 1 but 7.4 in Table 2 for the same row.)

**Fig. 3(a), vanilla async OPD, performance as % of lag-0 (read from the plot; text only gives "roughly 40%" for coding at lag 8):**

| Task | lag 0 | lag 2 | lag 4 | lag 8 |
|---|---|---|---|---|
| Reasoning (MATH500) | 100 | ≈99 | ≈94 | ≈89 |
| Tool-use | 100 | ≈98 | ≈93 | ≈83 |
| Coding agent | 100 | ≈93 | ≈79 | ≈60 |

**Fig. 4 (training curves, 400 steps, read from the plot):** reasoning — sync ends ≈80, async peaks ≈78 near step 200 then decays to ≈70, f-OPD ≈79; tool-use — sync ≈74, async peaks ≈73 then decays to ≈62, f-OPD ≈74; coding — sync ≈42, async peaks ≈35 then ≈27, f-OPD ≈39. Fig. 4(d) throughput: sync 1.00×, async 1.61×, f-OPD 1.46×. Fig. 1(b): async sits at relative efficiency ≈1.6 with normalised performance ≈0.88 (reasoning) / ≈0.7 (tool-use) / ≈0.5 (coding); f-OPD at ≈1.45 with ≈0.95–0.98 on all three.

**Fig. 3(c)/Fig. 5 (entropy):** task-average token entropy falls from ≈0.95 to ≈0.2 for lag 0 by step 400; lag 8 flattens near ≈0.3–0.4 with a wide seed band from step ≈250 on. Fig. 5: "Reasoning remains comparatively stable even as lag increases, while tool use and coding progressively develop wider late-training uncertainty bands."

**Fig. 6 (coding failure rates, %):** async OPD highest on invalid-tool / repeat-loop / premature-stop / regression-after-fix (up to ≈12–14%), lag-only score intermediate, f-OPD lowest (values not in text).

## Explanation offered

Mechanism: **distribution shift of the prefix occupancy (student-side staleness) + context drift of the teacher's conditioning in multi-turn tasks**, compounding with horizon; manifests as a peak-then-decay curve and entropy instability / partial collapse. No variance-of-estimator, support-mismatch or student-weighting argument; no teacher-parameter drift.

- "Δ^t ≤ C_roll TV(d^t, d^{stale,t}) [rollout drift] + C_sup E_{i∼B_t}[TV(π_teacher(·|c_i^t), π_teacher(·|c_i^{r(i)}))] [supervision drift]" (Prop. 3.2, Eq. 5). "rollout drift, arising when the active buffer reflects prefixes drawn from the older student policy rather than the current policy; and supervision drift, arising when teacher supervision is attached to outdated conditioning contexts that no longer align with the current contexts." (§3.2).
- "Unlike reward-estimation bias in off-policy RL, Δ^t arises from optimizing on prefixes generated by older student policies and, potentially, on teacher supervision tied to outdated conditioning contexts." (§3.2).
- "all three tasks exhibit a clear and monotonic performance degradation as policy lag increases, confirming that policy staleness is inherently harmful to asynchronous OPD fidelity. Moreover, this degradation becomes substantially more severe as the interaction horizon grows. Particularly for the coding-agent task, performance drops by roughly 40% under a lag of 8." (§5.2).
- "Both drifts grow systematically with policy lag and interaction horizon, and supervision drift scales more steeply with lag than rollout drift, highlighting the importance of controlling teacher-context staleness in long-horizon OPD." (§5.2).
- "asynchronous OPD exhibits progressively less stable task-averaged entropy dynamics as lag increases. The highest-lag setting (lag = 8, red) develops the widest late-training uncertainty band, indicating that some runs retain high-entropy exploration whereas others move toward partial collapse. This suggests that policy drift under asynchronous execution not only amplifies objective mismatch but also destabilizes generation dynamics, making entropy a practical early-warning signal for training instability." (§5.2).
- Horizon compounding: "TV(d^t, d^{stale,t}) ≤ K_H E_{i∼B_t}[δ_i^roll] ≤ K_H E_{i∼B_t}[√(½ D_i^roll)]" with K_H = O(HL) (Prop. A.4); "small early action differences alter later prefixes, which in turn alter the observations, tool outputs, and teacher-conditioning contexts seen downstream." (App. A.4).
- Lag is a budget, not a measure: "Update lag is not a divergence per se, but a budget: it bounds rather than measures policy drift, with the bound becoming looser as per-step errors accumulate." (App. A.3). "lag alone is insufficient in OPD: samples with identical age can exhibit vastly different mismatches due to heterogeneous drift from both the student rollout policy and the teacher supervision." (§1) — backed by lag-only weighting (34.9) < f-OPD (39.4) in Table 1.
- Why the fix works (§5.4): "freshness weighting recovers most of the lost task quality (+9.2 avg@4 accuracy, +9.7 coding resolve) and reduces the collapse ratio to 1/5, confirming that down-weighting stale samples directly counters rollout drift. Rollout-anchored regularization further closes the gap by stabilizing per-update drift (+2.4 avg@4 accuracy, +1.6 coding resolve), while adaptive refresh delivers the final gains and eliminates collapse entirely."
- Scope statement: "asynchronous OPD becomes unstable when the student outruns its data and supervision stream" (App. D). "our analysis is a discrepancy bound under bounded-regularity assumptions, not a convergence theory ... neither claim eliminates asynchronous bias outright." (App. D).

## Quotes

- Abstract: "asynchronous execution is necessary for system efficiency, but structurally deviates from the ideal on-policy objective."
- Abstract: "f-OPD consistently achieves task performance comparable to synchronous optimization while largely retaining the throughput advantages of asynchronous execution. Our results establish the first recipe for achieving a performance–efficiency trade-off in OPD".
- §1: "these processes should be executed synchronously in principle so that each training step remains fully on-policy. Empirically, however, strict synchronization leads to substantial system under-utilization, especially in long-horizon settings where rollout and supervision become increasingly expensive. As a result, modern OPD systems [8–10] widely adopt asynchronous mode (Fig. 1(b)) to improve throughput".
- §1: "this efficiency gain comes at the cost of staleness: buffered trajectories and their associated supervision may no longer match the current student policy when used for optimization. Such a mismatch in freshness can distort the effective on-policy objective, destabilize optimization, and ultimately degrade task performance. This creates a fundamental performance–efficiency trade-off in OPD: improving system utilization through asynchrony often deteriorates optimization fidelity, while preserving on-policy consistency requires sacrificing throughput."
- §1: "this freshness mismatch introduced by asynchronous execution in OPD remains poorly understood and largely unaddressed. An intuitive way to mitigate it is to periodically refresh the replay buffer, preventing stale rollouts from continuously corrupting policy updates. Yet such a hard refresh introduces its own dilemma: refreshing too aggressively discards potentially useful learning signals, while refreshing too conservatively allows stale optimization to accumulate."
- §2: "Our setting extends this line of work to teacher-supervised OPD, where freshness mismatch arises jointly from stale student rollouts and outdated teacher supervision."
- §3: "Although the freshness mismatch arises mainly from systems-level asynchrony, it can be analyzed as an objective-level discrepancy."
- §4.2: "Weighting is a first layer of defense: it reduces further deviation caused by over-optimizing on stale samples. However, it does not explicitly constrain how far the current policy can deviate from the rollout policy associated with the surviving samples, nor can it repair a mismatch that is already present in an active buffer that has become globally stale."
- §5.3: "Vanilla asynchronous OPD delivers a 1.61× throughput gain over synchronous OPD, but task quality collapses to a 26.8 resolve with a 60% collapse rate across runs. Simplified freshness-aware variants (hard refresh & lag-only weighting) mitigate this degradation only partially and remain well below the synchronous ceiling. In contrast, f-OPD attains a 1.46× throughput gain while preserving 94% synchronous resolve and incurring only a 2.6-point post-patch regression, with no observed collapse."
- §5.3: "f-OPD inherits the task quality of synchronous OPD and much of the throughput advantage of asynchronous OPD simultaneously, recovering the on-policy fidelity that vanilla asynchronous mode sacrifices without paying the full throughput cost of synchronous execution."
- §6: "Its effectiveness is established across tasks of varying interaction horizons, with the most pronounced gains on long-horizon coding agents."
- App. A.1: "The resulting bounds should therefore be read as conservative control justifications rather than a complete optimization theory for asynchronous OPD."
- App. B.1: "A rollout batch is generated by a student snapshot, labeled by the frozen teacher, and consumed by a later student snapshot."
- App. D: "In static labeling settings with a fixed teacher and identical context, D_i^sup is near-zero by construction; its value lies in asynchronous relabeling or trajectory-conditioned regimes. We therefore frame supervision drift as a generally applicable term whose contribution may be negligible in some settings, not as a universally dominant factor."

## Not in the source

- The per-token distillation loss ℓ is never instantiated: no statement of reverse vs forward KL, sampled-token vs full-vocab vs top-k, or KL direction; only "we follow the ... Thinking Machines Lab [40]" setup. No comparison across objectives, so nothing on whether `rkl` / student-top-k tolerates lag better than `rkl_pg`.
- No importance-sampling ratio, clipping, or masking is mentioned for any variant (the async baseline is uncorrected).
- No learning rate, batch size, rollouts per prompt, sequence length, buffer size, optimiser, or the numeric values of α, β, λ, ξ, κ_f, κ_roll, κ_sup ("selected on the task-specific pilot split").
- Training framework and hardware pipeline unspecified beyond GPU counts; no code.
- Lag-2/4 numbers exist only as plotted points (Fig. 3a); f-OPD is evaluated only at lag 8, so no f-OPD-vs-lag curve. Absolute sync/async accuracies for reasoning appear only in Fig. 4.
- "Teacher labels are greedily generated" is not explained (what a greedy label is when the teacher supplies a distribution over the student's tokens).
- Internal inconsistencies: async-OPD post-patch regression 12.1 (Table 1) vs 7.4 (Table 2); Table 1 calls the baseline "hard refresh", App. B.1 "fixed refresh"; App. B.1 refers to "a refresh rate of 100.0 in the systems table", which does not exist in the paper; App. B.1 contains two near-duplicate "Metrics, uncertainty, and compute" paragraphs; §5.2 says "four OPD variants" meaning four lag values.
- No wall-clock or step-time numbers behind the 1.00×/1.61×/1.46× throughput; the hard-refresh baseline's throughput (1.15×) is from "a simple amortized overlap model", not measured (App. B.1).
