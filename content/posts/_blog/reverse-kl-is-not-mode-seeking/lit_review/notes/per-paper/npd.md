# Rang, Bi et al. 2026 — Near-Policy: Accelerating On-Policy Distillation via Asynchronous Generation and Selective Packing

**arXiv 2605.05940v1** (7 May 2026). Miao Rang*, Zhenni Bi*, Hang Zhou, Kai Han, Xuechun Wang, An Xiao, Xinghao Chen, Yunhe Wang, Hanting Chen† (Huawei Technologies; Tianjin University). Sources: PDF v1 (curl + pypdf text, pages 7 and 17 rendered with pymupdf, 2026-09-14); arXiv abs page for title/version. The method is called "NPD" (Near-Policy Distillation) throughout; the phrase "selective packing" appears only in the title — in the body it is the Δ-IFD filter (§3.2) followed by SFT-style sequence packing (§3.1). No code ("Codes will be released soon", abstract).

```bibtex
@article{rang2026npd,
  title   = {Near-Policy: Accelerating On-Policy Distillation via Asynchronous Generation and Selective Packing},
  author  = {Miao Rang and Zhenni Bi and Hang Zhou and Kai Han and Xuechun Wang and An Xiao and Xinghao Chen and Yunhe Wang and Hanting Chen},
  journal = {arXiv preprint arXiv:2605.05940},
  year    = {2026}
}
```

## Tier

- **B** — side-by-side async (NPD) vs sync ("On-Policy", GKD, DistiLLM-2) numbers (Table 1, Fig 3, Tables 4/9/10) WITH an explanation (§3.2, §4.5, App. C.1: bounded KL(current student ‖ generating student) ≤ ε ≈ 0.10 = "near-policy regime"; Δ-IFD removes the high-variance tail; a sparse refresh "resets the policy lag").
- **D** — many quotable statements on policy lag / near- vs off-policy (collected below).
- **F** — asynchronous generation decoupled from training IS the paper's main point (title, contribution 1, Algorithm 1).
- Not A: no argument about distribution-level vs sampled-token estimators (the objective is teacher-top-k forward KL + CE; nothing is compared against `rkl_pg`). Not C (explanation is offered). Not E strictly (method paper, not a tech report), though the recipe produced the released openPangu-Embedded-1B-RL model on a "1.2 million" sample "internal SFT dataset" (§4.2, Table 7). Not G.

## Setting

- Students / teachers (teacher always fixed, used as "a pure evaluator" doing a prefill pass, §3.1):
  - Main: openPangu-Embedded-7B → openPangu-Embedded-1B, "internal SFT dataset comprising 1.2 million high-quality samples", 10 epochs (§4.1–4.2, Table 2/8). All 11-benchmark AVG numbers (56.28 SFT → 64.37 NPD → 68.73 after GRPO) are this pair.
  - Sample-wise comparison: Qwen2.5-Math-7B-Inst → Qwen2.5-Math-1.5B on a 50k subset of MetaMathQA ("MetaQA Yu et al. [2023]"), 1 epoch (Table 1, §4.2).
  - Code: Qwen2.5-Coder-7B-Inst → Qwen2.5-Coder-1.5B on WizardCoder (Table 17, App. D.3).
  - Scale: "72B teacher" → openPangu-7B (Table 15); thinking: openPangu-7B-think → openPangu-1B-think at 32K context (Table 16).
- Hyper-parameters (§4.1): AdamW, wd 0.1, grad-clip 1.0, packing to 32,768 tokens, global batch 2M tokens, cosine lr 1e-5 → 1e-6, 10 epochs. λ_KD = 0.9, teacher top-k = 10 (App. B.4, Tables 11–12; ablation on "a single stage (Direct Fast Response) of SFT", 8-benchmark average).
- Benchmarks (§4.1): MMLU, CMMLU, C-Eval, IF-Eval, CLUEWSC, GSM8K, MATH-500, DROP, GPQA-Diamond, MBPP, HumanEval.
- Framework: vLLM-style batch generation ("e.g., vLLM", §3.1), then a standard SFT trainer with packing on Ascend NPUs (all times in NPU-hours). Generation engine for the student, teacher prefill, and SFT trainer are three separate stages, not a co-scheduled RL system.

## Async mechanism

- What is stale: the rollouts. Algorithm 1 (App. A): every K epochs, the *current* student samples one response per prompt for the whole prompt set ("D_S ← {(x,y) | x ∈ D_x, y ∼ p_θ^S(·|x)}", line 7; greedy by default — App. D.4 calls the default "a single greedy path" and the multi-path variant "K = 3: 1 greedy, 2 sampled"), Δ-IFD filters, sequences are packed, the teacher prefills top-k logits once (line 11), and the (sequence, teacher-logit) pairs are stored in a "Replay buffer D_train" (line 3, 12). The student then trains on that buffer for K epochs (lines 15–21) before the next refresh. So the behaviour policy is the student checkpoint at the last refresh; the trained policy drifts for up to K epochs of SFT-style updates. "single-rollout-multiple-updates" (App. E).
- Staleness knob = "Update Freq." = "the number of policy refreshes per 10-epoch cycle" (§4.5, Table 4): 0 (rollouts from the initial student reused for all 10 epochs), 1 ("a single strategic update (at epoch 5)"), 2, 3 (Table 10). Note Algorithm 1 line 5 refreshes at epoch 1 as well, so freq 1 = generate at epochs 1 and 6.
- Staleness metric: "D_KL(π_learner ‖ π_generator)" between "the active learner and the rollout generator" (App. C.1; Fig 5 y-label "D_KL(π_current ‖ π_gen)"). Values read from Fig 5 (x-axis 1150–11500 steps ≈ 10 epochs): On-Policy = 0 (dashed reference); "Off-Policy" 0.20 → 0.30; "Near-Policy w/o Δ-IFD" 0.09 → 0.12; "Near-Policy" 0.08 → 0.10; "Dynamic Near-Policy" (one weight sync at ≈ step 6900) 0.085 → drops to 0.05 → 0.065 at step 11500. Text: "When a weight sync occurs (e.g., around step 6900), the KL divergence sharply drops from 0.09 to 0.05, effectively resetting the policy lag" (App. C.1).
- Defined regime: "D_KL(π_current^(t) ‖ π_gen^(t)) ≤ ε … Unlike strictly on-policy training (ε = 0) … or off-policy training (ε → ∞) … NPD empirically maintains a strict boundary of ε ≈ 0.10" (App. C.1). What "Off-Policy" in Fig 5 is (fixed teacher/SFT corpus? never-refreshed rollouts?) is not defined.
- Mixing: none — each refresh replaces the buffer (Algorithm 1 line 12); no fresh/stale mixing.

## Objective

`fkl` on **teacher-top-k** (k = 10) plus a CE term on the student's own sampled tokens; both computed on packed sequences in an SFT trainer. Not reverse KL, not sampled-token.

Eq. (1): $L_{\text{total}} = (1-\lambda)\cdot L_{CE} + \lambda\cdot L_{KD}$, "λ ∈ [0,1] … trade-off between the hard-label supervision (L_CE) and the teacher's soft distributional guidance" (§3.1). Default λ = 0.9 (Table 11).

Eq. (2): $L_{KD} = \sum_{c \in V_{\text{top-}k}} p_T(c)\,\log\frac{p_T(c)}{p_S(c)}$, "Let V_top-k ⊂ V denote the subset of indices corresponding to the k largest logits of the teacher model. The distillation loss is reformulated as the KL divergence computed exclusively over this high-confidence support" (§3.1). Direction: KL(teacher ‖ student) = forward KL in our convention. Algorithm 1 line 17: "p_T ← Softmax(O_T), p_S ← Softmax(O_S)" where O_T are the stored top-k teacher logits — so p_T is renormalised over the top-k; whether p_S is renormalised over the same support or is the full-vocab student softmax is not stated.

The "hard-label" L_CE(S) (Algorithm 1 line 19) is cross-entropy on the packed *student-generated* sequences S, i.e. maximum likelihood of the behaviour student's own samples (a self-SFT term), with weight 1−λ = 0.1. The paper does not comment on this.

Top-k ablation (Table 12, App. B.4): k = 5/10/15/20 → 54.78/55.29/54.24/54.40; "training time remains almost identical across different k values".

## Off-policy correction

None. No importance ratio, no clipping, no per-token/sequence mask based on log-prob mismatch. The stabilisers are (i) the Δ-IFD sample filter and (ii) sparse refreshes; nothing in Eq. (1)–(2) or Algorithm 1 depends on the behaviour-policy log-probs.

Δ-IFD filter (§3.2): IFD(Q,A) = L_θ(A|Q) / L_θ(A) (Eq. 3, ratio of conditioned to unconditioned CE of the response); "ΔIFD = IFD_Teacher − IFD_Student" (Eq. 4). Keep only "The Proximal Learning Zone (0 ≤ ΔIFD ≤ τ)" with τ = 0.8 best (Table 5: τ = 0.9/0.8/0.7 → 62.86/63.05/62.60; §3.2 and Table 3 use [0,1]). Rejects:
- "The Degenerate Zone (ΔIFD < 0)": "the teacher exhibits higher confidence than the student … typically arises from short, meaningless responses (e.g., trivial patterns or empty sequences) … the student generates them with high uncertainty due to policy degradation."
- "The Cognitive Disconnect Zone (ΔIFD > τ)": "the teacher model exhibits extreme uncertainty regarding the response (i.e., IFD_Teacher ≫ IFD_Student) … while the student generates the content with high confidence, the teacher evaluates it as highly improbable or erroneous. Including such conflicting data creates a substantial mismatch … forcing high-variance optimization steps."
Filter rate (Table 13, "12M proprietary dataset"): Degeneration 2.60 % early / 3.77 % late; Proximal 96.20 / 96.19; Cognitive Disconnect 1.20 / 0.04. "approximately 96% of rollouts are safely retained" (App. C.2). Cost: Δ-IFD computation 5 h + filtering 0.04 h of 366 h total (Table 19).

## Results: async vs sync

Sync baselines are all *other people's* on-policy methods (GKD, DistiLLM, DistiLLM-2) or an unspecified "standard on-policy distillation" (Fig 3); NPD is never run in a K→0 (per-step refresh) mode of its own pipeline.

| Comparison | Benchmark | Sync | Async (NPD) | Throughput | Source |
|---|---|---|---|---|---|
| Qwen2.5-Math-7B-Inst → 1.5B, 50k MetaMathQA, 1 epoch | GSM8K / MATH500 | GKD 80.17 / 73.60 (6.54 NPU-h); DistiLLM 81.15 / 73.80 (12.00); DistiLLM-2 81.27 / 74.20 (12.00); teacher 89.31 / 80.00; student 77.33 / 62.00; Supervised KD 78.62 / 75.00 (1.17); SeqKD 82.41 / 72.60 (2.06) | 82.87 / 76.60 | 1.48 NPU-h per epoch → "8.1× training speedup compared to DistiLLM-2" (12.00/1.48); 4.4× vs GKD | Table 1, §4.2 |
| openPangu-7B → 1B, fixed steps | MATH-500 vs steps 25–250 | "On-Policy" ≈ 56 → ≈ 67 | ≈ 56 → ≈ 67, mostly ≤ 1 pt above | — | Fig 3(a) |
| same, fixed wall-clock | MATH-500 vs 8-NPU-hours 0–140 | ≈ 55.5 (10 h) → 60.5 (30 h) → 64 (50 h) → 65.8 (110–135 h) | ≈ 62.5 (0 h) → 66.5 (10 h) → 68.5 (30 h) → 69 (50–80 h) → 72 (110 h) → 71.5 (135 h) | — | Fig 3(b) (values read off the plot) |
| Qwen2.5-Coder-7B-Inst → 1.5B, WizardCoder | HumanEval / MBPP / AVG | GKD 40.85 / 61.90 / 51.38; DistiLLM 39.63 / 62.17 / 50.90; DistiLLM-2 42.24 / 62.70 / 52.47 | 43.58 / 63.01 / 53.29 | — | Table 17 |

Staleness ablations (openPangu 7B → 1B, 11-benchmark AVG, 10 epochs):

| Update freq. per 10 epochs | NPD (full) | NPD w/o Δ-IFD | NPU-hours (full) | Source |
|---|---|---|---|---|
| 0 | 62.78 | 61.63 | 335 | Tables 4, 10 |
| 1 | **64.37** | 63.26 | 659 | |
| 2 | 64.17 | **63.43** | 983 | |
| 3 | – | 63.12 | | Table 10 |

Component ablation (Table 9): Standard KD 58.79 → Packed Distillation on student rollouts 61.63 (+2.84) → +Δ-IFD 62.78 (+1.15) → +Sparse Update 64.37 (+1.59). Filter ablation (Table 3): unfiltered 61.63; PPL-weighted 61.69; IFD_Teacher∈[0,1] 62.69; IFD_Student∈[0,1] 62.60; ΔIFD∈(0,∞) 61.77; ΔIFD∈(−∞,1) 61.76; ΔIFD∈[0,1] 62.78. Downstream GRPO (Table 6/14): SFT init 56.28 → 59.86 (+3.58); NPD w/o Δ-IFD 63.43 → 65.91 (+2.48); NPD 64.37 → 68.73 (+4.36).

Cost breakdown (Table 19, 10 epochs, 1.2M samples, 8-NPU node): Student Rollout 300 h (82.01 %), Teacher Annotation 18 h (4.92 %), Δ-IFD 5 h, Filtering 0.04 h, Packing 0.75 h, Training 42 h (11.48 %). Note this sums to ≈ 366 h, whereas Table 4 lists 335/659/983 NPU-hours for 0/1/2 refreshes — the accounting units differ (8-NPU-node-hours vs NPU-hours) and are not reconciled.

Inconsistency to note: in Fig 3(b) the NPD curve already sits at ≈ 62.5 at t ≈ 0 h while both curves start at ≈ 56 in Fig 3(a); the wall-clock axis therefore appears to exclude the (dominant, 82 %) rollout-generation time or plots a different run. The text does not say.

## Explanation offered

Mechanisms invoked: (1) bounded distribution shift between generating and current student ("near-policy regime", KL ≤ 0.10); (2) variance — rare out-of-distribution samples dominate gradients, removed by Δ-IFD; (3) sparse refreshes reset the lag; (4) too-frequent refreshes add "unnecessary perturbations". No theory beyond the empirical KL bound.

- "However, asynchronous updates inevitably introduce policy lag and sample noise, which can cause the behavior to drift from near-policy toward off-policy. … By filtering extreme out-of-distribution samples, Δ-IFD prevents noise from dominating the gradients, ensuring updates remain within a safe proximal learning zone." (Abstract)
- "First, we implement sparse student updates, ensuring that the generation policy never deviates significantly from the training policy. Second, we introduce a dynamic filtering mechanism based on Δ-IFD. … This prevents high-variance noise from dominating the gradients and ensures the student updates within a safe proximal learning zone, maintaining a near-policy state despite the asynchronous updates." (§1)
- "Crucially, this heuristic filtering serves as a robust stabilizer against the policy lag inherent in asynchronous frameworks. Even if the data-generating policy drifts from the current training state, the Δ-IFD criterion effectively intercepts extreme out-of-distribution samples. Empirically (Fig 5), without Δ-IFD, noisy samples trigger high-variance updates, driving the KL divergence to 0.12. In contrast, Δ-IFD effectively intercepts out-of-distribution noise, suppressing the divergence within 0.1." (§3.2)
- "Despite the asynchronous nature of our data generation, the student's policy updates remain as precise and effective as those derived from synchronous, strictly on-policy gradients." (§4.3, fixed-steps result, Fig 3a)
- "a single strategic update (at epoch 5) yields a significant 1.59% performance boost, whereas increasing the frequency to two or more leads to saturation or even slight regression. This phenomenon suggests that excessive synchronization not only incurs higher computational costs but may also introduce unnecessary perturbations that disrupt student learning stability." (§4.5)
- "Contrary to strict on-policy paradigms (e.g., PPO) that demand frequent, high-cost updates to minimize distributional shift, our results demonstrate that the theoretical upper bound of on-policy optimization can be effectively approximated via sparse, periodic updates. Consequently, we establish that frequent synchronization is not strictly necessary" (§4.5)
- "the Sparse Update mechanism contributes the most among our novel additions (+1.59), as it physically resets the policy lag and prevents asynchronous training collapse. The Δ-IFD filter provides an additional significant boost (+1.15) by strictly pruning harmful variance." (App. B.2)
- "Theoretically, a policy update should be dynamically triggered whenever the KL divergence approaches our predefined ε boundary (e.g., 0.10). However, our extensive empirical evaluations reveal that uniformly allocating 1 or 2 updates across a standard 10-epoch training cycle naturally intercepts the critical points just before severe policy drift occurs." (App. B.3)
- "off-policy training exhibits severe distribution drift, with the KL divergence rapidly diverging to 0.30. In contrast, our proposed NPD method (green line) strictly maintains the KL divergence below 0.10." (App. C.1)
- "Importantly, this filtered 4% corresponds to a high-variance and potentially poisonous long tail. Enforcing updates on these anomalous samples can lead to explosively compounding gradient variance and, in turn, destabilize training. Therefore, our design emphasizes precision over recall" (App. C.2)
- "as illustrated in Fig 4, our filtering scheme effectively suppresses loss spikes inherent to unfiltered training. By purging samples where the student is statistically overconfident or the teacher is uncertain, the 0 ≤ ΔIFD ≤ 1 criterion significantly stabilizes the optimization landscape" (§4.4; Fig 4 shows CE loss over ~12k steps, w/o Δ-IFD spikier)

## Quotes

- "we propose Near-Policy Distillation (NPD), an asynchronous approach that decouples student generation from training. This reformulation enables Supervised Fine-Tuning (SFT) with sequence packing." (Abstract)
- "Although effective in correcting distribution shift, these methods rely on synchronous RL-like frameworks that necessitate simultaneous teacher-student inference. Consequently, they inherit the inefficiencies of RL, specifically the inability to utilize sequence packing and the low token utilization rate caused by continuous generation." (§1)
- "However, the asynchronous nature of NPD introduces a new challenge: policy lag. Since the data generation policy differs from the current training policy, samples may become noisy or drift toward off-policy distributions, potentially destabilizing the student's alignment." (§1)
- "achieving the performance benefits of on-policy distillation with the efficiency of off-policy training (8.1× speedup)." (§1, contributions)
- "Overall, on-policy distillation better corrects mismatch under the student's distribution, but its throughput is often limited by online generation and teacher evaluation overhead." (§2)
- "Unlike on-policy methods that require synchronized teacher generation, our decoupled framework allows the teacher to act as a pure evaluator. Crucially, this enables the application of sequence packing during the annotation phase." (§3.1)
- "Specifically, under identical training conditions, NPD achieves a 8.1× training speedup compared to DistiLLM-2, validating its capability to deliver on-policy-level performance with off-policy-level efficiency." (§4.2)
- "under identical training steps, the accuracy trajectories on Math500 for both NPD and on-policy methods exhibit a highly consistent upward trend. This observation is critical: it demonstrates that NPD effectively retains the optimization quality of on-policy methods." (§4.3)
- "By decoupling the generation phase and pre-computing Student-Generated Outputs, NPD eliminates the computational bottleneck associated with real-time, step-by-step sampling." (§4.3)
- "capturing the relative knowledge gap is critical for preventing policy lag and ensuring effective knowledge transfer in asynchronous distillation." (§4.4)
- "Our framework demonstrates high robustness and is largely insensitive to the specific update frequency." (App. B.3)
- "Based on our empirical results, we formally define the Near-Policy regime: At any training step t, the distance between the current student policy (π_current) and the data-generating policy (π_gen) is strictly bounded by a small threshold ε: D_KL(π_current^(t) ‖ π_gen^(t)) ≤ ε" (App. C.1)
- "Traditional strict on-policy methods necessitate synchronous inference and optimization (i.e., a 'one-rollout-per-update' paradigm), which severely underutilizes expensive training NPUs by leaving them in prolonged idle states. In contrast, NPD not only fully decouples the student rollout process asynchronously—enabling horizontal scaling with additional resources to accelerate inference—but also implements a 'single-rollout-multiple-updates' strategy for data utilization." (App. E)
- Fig 5 caption: "Unlike standard off-policy methods that suffer from severe drift, NPD maintains strict divergence bounds (< 0.10), with sparse updates ensuring forced policy synchronization."

Convention flags: the paper's "on-policy" baselines (GKD, DistiLLM-2) use reverse/skew/contrastive objectives, whereas NPD's own L_KD is teacher-top-k *forward* KL — the async-vs-sync comparison in Table 1 therefore also changes the objective. Its lag metric is KL(current student ‖ generating student), not a teacher–student divergence.

## Not in the source

- No sync (per-step refresh, K → 0) run of NPD's own objective/pipeline; the on-policy baselines are third-party methods (Table 1, 17) or an unnamed "standard on-policy distillation" (Fig 3) whose objective, batch size and refresh cadence are not given.
- No async-vs-sync comparison for a sampled-token / reverse-KL objective; nothing about `rkl_pg` vs distribution-level losses.
- No importance weighting or stale-token masking, and no ablation of one.
- Fig 3 numbers only as curves; the wall-clock axis of Fig 3(b) is not reconciled with Fig 3(a) or with Table 19's rollout cost (82 % of time).
- "Off-Policy" curve in Fig 5 is undefined; the per-step KL values are not tabulated.
- Whether p_S in Eq. (2) is renormalised over the teacher's top-k; the sampling temperature for rollouts (default appears to be greedy, App. D.4); number of prompts per refresh beyond "the prompts dataset".
- Exact epoch of each refresh for freq 2/3; whether the optimizer state or lr schedule is reset at refresh.
- Code not released; the 1.2M / "12M proprietary" (Table 13) datasets are internal.
- Standard errors / seeds for any table.
