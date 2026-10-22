# Wu, Han, Cai 2026 — Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation

**arXiv 2604.13010v2** (v1 14 Apr 2026, v2 8 May 2026). Yecheng Wu, Song Han, Han Cai (NVIDIA; corresponding hcai@nvidia.com — arXiv metadata and references.bib spell the third author "Hai Cai", the PDF says "Han Cai"). Code: `github.com/jet-ai-projects/Lightning-OPD` (not inspected). Sources: PDF v2 text via pypdf + Fig. 3 page rendered with pymupdf and read visually, 2026-09-14.

```bibtex
@article{wu2026lightning,
  title   = {Lightning {OPD}: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation},
  author  = {Yecheng Wu and Song Han and Hai Cai},
  journal = {arXiv preprint arXiv:2604.13010},
  year    = {2026}
}
```

## Scope verdict

IN SCOPE (extreme staleness). The "offline" rollouts are sampled from π_ref, the SFT-initialised student, which is exactly the step-0 checkpoint of the OPD student ("the student π_θ is initialized from π_ref and trained on D_OPD", Fig. 2 caption; Alg. 1 line 10). So the behaviour policy is the student's own earlier (step-0) version, frozen for all 150 steps. Rollouts are NOT from the teacher and NOT from a foreign model. Teacher is fixed throughout (§3.1 "Let π_T denote a fixed teacher model"). The only "other-model" element is that the SFT data (Stage 1) must come from the same teacher π_T — that is a constraint on how π_ref is built, not on whose rollouts are used.

## Tier

- **B** — side-by-side offline (step-0 student rollouts, no live teacher) vs online OPD on 5 benchmarks × 2 scales (Table 1), cost (Table 2), across SFT checkpoints (Fig. 3c), and a 2×2 teacher-consistency grid for each paradigm (Table 4), with an explanation (Thms 3.5–3.9, Fig. 3a drift measurement, §4.5 asymmetry argument).
- **D** — many quotable statements on offline/stale-rollout OPD (see Quotes).
- **F** — offline OPD is the paper's main point.
- Not A: the theory bounds the gap between the uncorrected stale sampled-token gradient (`rkl_pg`, w ≡ 1) and the on-policy one by G·σ_A·sqrt(χ²(π_θ‖π_ref)) (Thm 3.5) and gives a covariance/trust-region decomposition (Thm 3.7); it never contrasts sampled-token vs distribution-level (full-vocab / top-k) RKL under staleness. It is an A-adjacent argument for *why rkl_pg tolerates staleness at all* (small drift + implicit regularisation), not for which estimator tolerates it better.
- Not E/G (not a production report; async is the main point).

## Setting

- Students / teachers (§4.1, §4.4): Qwen3-4B-Base ← Qwen3-8B; Qwen3-8B-Base ← Qwen3-32B; Qwen3-30B-A3B-Base (MoE) ← Qwen3-30B-A3B-Thinking-2507 (offline only; online OPD "OOM" on one 8×H100 node, Table 3). Ablation teachers: QwQ-32B crossed with Qwen3-8B (4B) / Qwen3-32B (8B) (Table 4).
- Teacher fixed: yes (§3.1).
- Pipeline (§3.2, Alg. 1): Stage 1 SFT of π_base on trajectories x ∼ π_T(·|q) for q ∈ OpenThoughts-3 prompts (Eq. 5–6; 3000 steps, lr 8e-5, Table 5) → π_ref. Stage 2 preprocessing: one response per prompt x ∼ π_ref, teacher log-probs log π_T(a_t|s_t) stored once (Eq. 7). Stage 2 training: 150 steps, batch 256, max response 4096, lr 2e-6 constant, wd 0.1, Adam β=(0.9, 0.98), rollout temp 0.8, top-p 1.0, advantage clip [−10, 10] (Table 6).
- OPD prompts: DAPO-Math-17k (math); 30K subset of EpiCoder-func-380k (code) (§4.1). Code OPD is initialised from the math-OPD checkpoint, not from π_ref (App. B).
- Benchmarks: AIME 2024/2025, HMMT 2025 (avg pass@1 over 32 samples), LiveCodeBench v5/v6 (4 samples); temp 0.6, top-p 0.95, 32,768 / 40,960 max tokens (§4.1).
- Framework: LlamaFactory (SFT), slime (OPD) (§4.1). "Standard OPD and Lightning OPD share identical training settings in the OPD stage, differing only in the source of rollouts" (§4.1).

## Async mechanism

- What is stale: the *entire* rollout set. Behaviour policy = π_ref = student at OPD step 0; current student = π_θ after k ∈ [0, 150] updates. No refresh, no replay mixing, no fresh fraction — "Lightning OPD fixes the rollout distribution to π_ref" (§3.1, Eq. 4). The precomputed *teacher* log-probs are exact (teacher is fixed), so the only staleness is in the sampling distribution of tokens.
- Staleness in steps: 0 → 150 optimizer steps at lr 2e-6, batch 256 (Table 6). With one response per prompt and 17K math prompts, 150 × 256 = 38,400 samples means each math rollout is revisited ≈2.3 times (≈1.3× for the 30K code set) — my arithmetic from §4.1 + Table 6; the paper does not state epochs.
- How measured: (i) theory, sequence-level χ²(π_θ‖π_ref) = E_{π_ref}[w²] − 1 with w(x;θ) = π_θ(x|q)/π_ref(x|q) (Thm 3.5, App. A); (ii) empirically, *per-token* importance weight w_t = π_θ/π_ref on the fixed data, Fig. 3a (Qwen3-4B): mean "drops from 1 to 0.94 within the first 20 steps and then plateaus", std "rises sharply in the same early phase before stabilizing" at ≈0.09 (read off the right axis; text says "below 0.1") (App. C.1).
- Range studied: only the two endpoints (fully online vs fully offline-from-step-0). No intermediate staleness levels, no buffer/mix ratio. Fig. 3c varies the *quality* of π_ref (SFT checkpoints 500–3000 steps), not the staleness.

## Objective

`rkl_pg` — sampled-token score-function estimator of reverse KL with the per-token advantage under stop-gradient, no importance weighting, clipped to [−10, 10]. Letters/direction match our convention: the paper calls it "Reverse-KL Advantage Computation" (Fig. 2) and proves J_on(θ) = −KL(π_θ‖π_T) (Thm 3.6, App. A.3), i.e. KL(student‖teacher).

Verbatim (§3.1):

- Eq. (2): $A_t(\theta) = \log \pi_T(a_t|s_t) - \log \pi_\theta(a_t|s_t)$
- Eq. (3): $J_{\mathrm{on}}(\theta) = \mathbb{E}_{q\sim p,\, x\sim \pi_\theta}\left[\sum_{t=1}^{T} A_t(\theta)\right]$
- Eq. (4): $J_{\mathrm{off}}(\theta) = \mathbb{E}_{q\sim p,\, x\sim \pi_{\mathrm{ref}}}\left[\sum_{t=1}^{T} A_t(\theta)\right]$
- "Following standard OPD practice [8, 9, 22], the advantage A_t(θ) is treated as a fixed scalar (stop-gradient) when computing parameter updates; throughout our analysis, ∇J_on and ∇J_off denote the resulting advantage-weighted policy gradients" (§3.1). Per-trajectory gradient $f(x;\theta) := \sum_t A_t(\theta)\,\nabla \log \pi_\theta(a_t|s_t)$; $\nabla J_{\mathrm{off}}(\theta) = \mathbb{E}_{x\sim\pi_{\mathrm{ref}}}[f(x;\theta)]$ (§3.1, Prop. A.1).
- Alg. 1 line 13: "Compute A_t(θ) = log π_T(a_t|s_t) − log π_θ(a_t|s_t), clip to [−τ, τ]"; τ = 10 (Table 6).
- Sequence-level sum of per-token advantages, no length normalisation stated; no group/baseline subtraction.

Side-note (mine): because A_t carries stop-gradient, the surrogate ∇J_off = E_{π_ref}[Σ_t A_t ∇log π_θ] drops the −E_{π_ref}[Σ_t ∇log π_θ] term that would appear in the true gradient of Eq. 4. On-policy that term is zero in expectation; under π_ref sampling it is not, so the "stop-gradient" convention is doing work in the offline case. The paper does not discuss this.

## Off-policy correction

None. "the offline gradient ∇J_off(θ) = E_{x∼π_ref}[f(x;θ)] is the special case w ≡ 1" (§3.1; Prop. A.1). No per-token or per-sequence importance ratio, no truncation, no IcePop-style masks. The only bounded quantity is the advantage clip |A_t| ≤ 10 (Alg. 1 line 13, Table 6), which the authors use to satisfy Assumption 3.1 ("automatically satisfied under advantage clipping (|A_t| ≤ τ gives σ_A ≤ Tτ)", §3.3). Importance weights are computed only as a *diagnostic* (Fig. 3a). App. D even argues correction is pointless for the failure mode they care about: "no importance sampling correction or conservatism mechanism can remove it [the teacher-mismatch bias]".

Ambiguity: §3.3 says the χ² bound "remains small under standard KL regularization" and App. A.2 says "advantage clipping and KL regularization keep χ² small throughout training", yet Thm 3.7 / Fig. 3a are sold as stabilising "without an explicit KL penalty" and Table 6 lists no KL coefficient. Whether a KL-to-π_ref term is in the loss is not stated; the most literal reading is none.

## Results: async vs sync

**Table 1** (pass@1; "OPD" = online, fresh rollouts every step; "Lightning OPD" = step-0 rollouts, no live teacher; identical OPD hyperparameters):

| Student ← Teacher | Method | AIME24 | AIME25 | HMMT25 | Math avg | LCB v5 | LCB v6 | Code avg |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Base ← Qwen3-8B | SFT | 56.7 | 52.1 | 34.0 | 47.6 | 33.8 | 31.5 | 32.6 |
| | ExOPD [10] | 61.0 | 56.0 | 34.4 | 50.5 | – | 29.0 | – |
| | OPD (sync) | 65.4 | 57.9 | 39.9 | 54.4 | 44.2 | 39.3 | 41.8 |
| | Lightning OPD (offline) | 68.1 | 58.4 | 39.8 | 55.4 | 42.8 | 40.3 | 41.5 |
| Qwen3-8B-Base ← Qwen3-32B | SFT | 63.7 | 51.7 | 36.9 | 50.8 | 44.7 | 36.8 | 40.8 |
| | OPD (sync) | 68.5 | 59.0 | 39.4 | 55.6 | 47.3 | 41.2 | 44.2 |
| | Lightning OPD (offline) | 69.9 | 59.2 | 41.9 | 57.0 | 49.5 | 43.9 | 46.7 |

Offline ≥ online in 8 of 10 cells; deficits only 4B HMMT (−0.1) and 4B LCB v5 (−1.4). §4.2: "achieves performance on par with standard OPD across all settings, and in several cases marginally exceeds it."

**Table 2** (GPU hours, full OPD stage): 4B: OPD 72 vs Lightning 20 (3.6×); 8B: 120 vs 30 (4.0×). Lightning breakdown 4B/8B: rollout collection 10/10, teacher log-prob precompute 2/4, OPD training 8/16.

**Table 3** (Qwen3-30B-A3B MoE, offline only): SFT 66.8/63.2/44.6 (avg 58.2), LCB 39.4/33.0 (36.2) → Lightning 71.0/66.3/48.3 (61.9), LCB 60.8/54.4 (57.6). Online OPD "(OOM)".

**Table 4** (AIME24; rows = SFT-stage teacher, cols = OPD-stage teacher):

| Student | Paradigm | SFT teacher | OPD = same-family | OPD = QwQ-32B |
|---|---|---|---|---|
| 4B (Qwen3-8B vs QwQ-32B) | online OPD | Qwen3-8B | **65.4** | 62.4 |
| | | QwQ-32B | 61.2 | **62.8** |
| | Lightning (offline) | Qwen3-8B | **68.1** | 62.5 |
| | | QwQ-32B | 59.3 | **63.1** |
| 8B (Qwen3-32B vs QwQ-32B) | online OPD | Qwen3-32B | **68.5** | 64.8 |
| | | QwQ-32B | 65.0 | **66.5** |
| | Lightning (offline) | Qwen3-32B | **69.9** | 63.1 |
| | | QwQ-32B | 62.1 | **68.7** |

Diagonal (consistent teacher): offline ≥ online in all 4 cells. Off-diagonal (mismatched): offline ≤ online in all 4 cells (4B: 62.5 vs 62.4, 59.3 vs 61.2; 8B: 63.1 vs 64.8, 62.1 vs 65.0). §4.5: "mismatching from Qwen3-32B SFT to QwQ-32B OPD drops Lightning OPD by 6.8 points but standard OPD by only 3.7 points."

**Fig. 3c** (Qwen3-4B, AIME24 vs SFT checkpoint used as π_ref; values read from the rendered figure labels):

| SFT steps | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| SFT | 42.3 | 51.0 | 54.7 | 56.0 | 56.0 | 56.7 |
| OPD (sync) | 56.4 | 61.1 | 64.4 | 64.6 | 63.6 | 65.4 |
| Lightning (offline) | 60.7 | 62.3 | 65.2 | 65.6 | 67.7 | 68.1 |

Offline ≥ online at every SFT checkpoint, gap 0.6–4.3 points. **Fig. 3b**: offline AIME24 56.7 → ≈68 within ~50 steps, then flat to 150 (Qwen3-4B). No online learning curve shown.

## Explanation offered

Mechanism, in the paper's terms: (1) distribution shift between π_ref and π_θ stays small in OPD, so the uncorrected stale gradient stays close to the on-policy one; (2) the missing IS weight acts as an implicit trust region; (3) both have the same fixed point when the teacher is representable; (4) failures of offline OPD are attributed to *teacher inconsistency* (SFT-data teacher ≠ OPD teacher), not to staleness per se.

- §1: "recent empirical studies suggest that RL-trained models remain surprisingly close to their SFT initialization ... We observe a similar phenomenon in OPD training, where the student's distribution exhibits only modest drift from the SFT reference throughout the OPD stage. This observation suggests a practical offline alternative [26]: precompute the teacher's log-probabilities once over SFT rollouts prior to training and reuse these values throughout the OPD process".
- Thm 3.5 (Eq. 8): $\|\nabla J_{\mathrm{on}}(\theta) - \nabla J_{\mathrm{off}}(\theta)\|_2 \le G\,\sigma_A\,\sqrt{\chi^2(\pi_\theta\|\pi_{\mathrm{ref}})}$, with χ² = E_{π_ref}[w²] − 1; G bounds ‖∇log π_θ(a_t|s_t)‖, σ_A² bounds E_{π_ref}[(Σ_t|A_t|)²]; requires supp(π_θ) ⊆ supp(π_ref) (Assumption 3.2, "holds naturally when π_θ is initialized from π_ref"). "At initialization π_θ = π_ref, χ² = 0 and the two gradients coincide exactly; the bound grows with drift but remains small under standard KL regularization." (§3.3). Cor. A.2: "the gap grows at rate O(ηk) after k gradient steps with step size η."
- Thm 3.6: J_on(θ) = −KL(π_θ‖π_T); "When π_T ∈ Π_Θ, A_t(θ*) = 0 almost surely, and θ* is a shared zero of both the online and offline OPD updates."
- Thm 3.7: $\nabla J_{\mathrm{off}}(\theta) = \nabla J_{\mathrm{on}}(\theta) - \mathrm{Cov}_{\pi_{\mathrm{ref}}}[w(x;\theta), f(x;\theta)]$. "The covariance term vanishes at initialization (w ≡ 1) and grows with drift, empirically acting as a trust-region effect that stabilizes training without an explicit KL penalty (Figure 3a)." App. A.4: "As π_θ drifts from π_ref, w becomes large on trajectories that π_θ over-weights relative to π_ref; on those same trajectories f (a gradient vector) also tends to have large magnitude. The covariance vector Cov[w, f] therefore grows in norm with drift and, when projected onto the drift direction θ − θ_ref, opposes further movement away from π_ref. This produces a restoring-force effect".
- App. A.3 heuristic decomposition: KL(π_θfinal‖π_T) ≈ ε_approx (model capacity, irreducible) + ε_opt (→0) + O(Gσ_A sqrt(χ²)) (offline–online gap); "Switching from offline to online rollouts reduces only the third term, not ε_approx, which is the dominant term determined by model capacity."
- App. C.1 (Fig. 3a): "The mean remaining close to 0.94 indicates that the student policy stays near the reference distribution throughout training, and the standard deviation stabilizing below 0.1 indicates that per-token distributional shift is consistently small. Both quantities remaining bounded confirms the implicit regularization of Theorem 3.7, which shows that fixing rollouts to π_ref automatically constrains both the magnitude and spread of policy drift without any explicit KL penalty."
- Teacher-consistency (why naive offline fails): Thm 3.8 (Eq. 9) $\|\nabla J_{\mathrm{on}} - \nabla J_{\mathrm{off}}\|_2 \le G(\sigma_A+\sigma_\Delta)\sqrt{\chi^2(\pi_\theta\|\pi_{\mathrm{ref}})}$ with Δ_t = log π_T^SFT(a_t|s_t) − log π_T^OPD(a_t|s_t); plus a persistent offline bias ≤ Gσ_Δ "independent of χ²"; Thm 3.9 (Eq. 10): the same Gσ_Δ bias hits online OPD at initialisation. §1: "we find that the issue does not primarily stem from the offline approximation itself, but rather from a more fundamental condition that has been overlooked in prior OPD work, which we term teacher consistency." Formally: "Teacher consistency requires these two teachers [SFT-data generator and OPD reference] to be the same model" (§1); Assumption 3.4 "When teacher consistency holds (π_T^SFT = π_T^OPD), Δ_t = 0 everywhere and σ_Δ = 0."
- Why offline is hurt MORE by mismatch (§4.5): "This asymmetry arises because a mismatched SFT teacher corrupts π_ref in two roles simultaneously for Lightning OPD, both as the reference distribution and as the source of fixed rollouts, whereas standard OPD refreshes rollouts from the current student at every step and can partially recover."
- Why offline slightly beats online is not explained beyond "implicit regularization ... prevents policy drift" (abstract) and the Fig. 3c remark that ordering is stable across SFT budgets.

## Quotes

- Abstract: "We investigate whether OPD can be performed offline by precomputing teacher log-probabilities once over SFT rollouts and reusing them during training. We find that naively doing so fails to reliably match standard OPD, and trace the root cause to a previously overlooked condition we term teacher consistency".
- Abstract: "We prove that, under teacher consistency, Lightning OPD shares the same optimum as standard OPD, with bounded gradient discrepancy and an implicit regularization effect that helps prevent policy drift."
- §1: "On-policy training is defined by the student's current rollout distribution, which evolves at every gradient step, making the teacher appear indispensable."
- §1: "In practice, however, naively applying this offline precomputation fails to reliably match the performance of standard OPD."
- §1: "We show that such teacher inconsistency introduces a gradient bias that degrades performance for both offline and online OPD, with a more pronounced effect on the offline variant."
- §1: "the gradient discrepancy between the two remains bounded throughout training, and the offline objective introduces an implicit regularization effect that naturally prevents policy drift without requiring any explicit penalty."
- §2: "Lightning OPD is similar only in optimizing over a fixed dataset but receives dense token-level teacher supervision rather than sparse rewards, so its central challenge is teacher consistency rather than OOD value estimation."
- §3.1: "Both objectives share the same advantage function and differ only in the response distribution."
- §3.2: "In contrast, standard OPD requires a live teacher server throughout training, querying it at every step for log-probabilities on freshly sampled rollouts from the current policy π_θ."
- §3.3 (after Thm 3.6): "When π_T ∉ Π_Θ, Theorem 3.5 ensures the offline update stays close to the online one as long as drift remains small, suggesting comparable capacity-limited behavior in practice."
- §4.1: "Standard OPD and Lightning OPD share identical training settings in the OPD stage, differing only in the source of rollouts that standard OPD samples rollouts online from the current student while Lightning OPD reuses rollouts precomputed from π_ref before training begins."
- §4.2: "The central finding is that Lightning OPD, despite eliminating the live teacher server entirely during training, achieves performance on par with standard OPD across all settings, and in several cases marginally exceeds it. This validates our theoretical analysis that under teacher consistency, the offline approximation preserves the same performance optimum as standard OPD."
- §4.5: "Notably, teacher inconsistency imposes a larger penalty on Lightning OPD than on standard OPD."
- §4.5: "Teacher consistency is consequently a more critical design requirement for Lightning OPD than for standard OPD."
- §5: "More generally, our results suggest that much of the practical benefit of on-policy distillation can be retained without the conventional deployment burden of maintaining a live teacher throughout training."
- App. A.2 remark: "Although χ² ≥ KL, advantage clipping and KL regularization keep χ² small throughout training."
- App. D: "Neither problem exists in Lightning OPD, where the teacher supplies dense per-token log-probability supervision for all token sequences, leaving no OOD region and no sparse reward to cause high-variance estimation. Conservatism is therefore neither necessary nor applicable. The real obstacle to offline OPD is teacher inconsistency."
- App. D: "Theorem 3.6 and our empirical results suggest that Lightning OPD's performance ceiling is primarily capacity-limited, determined by model capacity Π_Θ relative to the teacher rather than by the rollout distribution."
- App. D: "Lightning OPD instead collects rollouts from the student's own policy π_ref and evaluates the teacher's per-token log-probabilities on these student-generated sequences. This means the teacher provides corrective signals on exactly the distribution the student will encounter during inference, which is the core advantage of on-policy methods over off-policy ones [8]. The distinction is preserved in Lightning OPD even though teacher log-probabilities are precomputed."
- App. D on Rang et al. [26]: "No analysis is provided for when or why the offline approximation is reliable."
- App. E: "teacher consistency requires that SFT training data be generated by the same teacher used for OPD; when adopting a new teacher, this necessitates regenerating the SFT dataset".

## Not in the source

- No intermediate staleness: only "fresh every step" vs "frozen at step 0"; no replay buffer, mixing ratio, or k-step-lag sweep. No test of refreshing rollouts once mid-run.
- No sampled-token vs full-vocab / top-k comparison; the estimator is rkl_pg throughout. No forward-KL variant.
- No importance correction tried, so no evidence for whether IS/clipping would help or hurt; the Thm 3.7 covariance term is *argued* to be a helpful trust region but never ablated (e.g. offline + IS weights vs offline without).
- Whether an explicit KL(π_θ‖π_ref) penalty is in the loss is contradictory (§3.3 / App. A.2 mention "KL regularization"; Thm 3.7 and Fig. 3a caption say "without an explicit KL penalty"; Table 6 lists no coefficient).
- Fig. 3a importance weights are per-token; the theory is sequence-level χ². No sequence-level χ² or KL(π_θ‖π_ref) number reported; no drift curve for online OPD to compare against.
- No online learning curve (Fig. 3b is offline only); no seeds/variance; the 0.2–2.7-point offline advantages are within plausible noise of 32-sample AIME pass@1 and are not tested.
- Code stage: rollouts are "sampled from π_ref" (§4.1) but the student is initialised from the math-OPD checkpoint (App. B), so for code the behaviour policy ≠ step-0 student unless re-sampled; not stated.
- Number of epochs over the fixed rollout set not stated (≈2.3 for math by arithmetic); no per-prompt multiple responses, no group baseline.
- ExOPD baseline numbers (Table 1) are quoted from [10], presumably different SFT/teacher; not a controlled comparison.
- Why offline *exceeds* online on the diagonal is not analysed beyond the implicit-regularisation story; the possibility that the online run is under-tuned (same lr, no KL) is not discussed.
- Wall-clock / throughput of the online run is aggregated to GPU hours; no breakdown of teacher-server vs trainer cost for the online side.
