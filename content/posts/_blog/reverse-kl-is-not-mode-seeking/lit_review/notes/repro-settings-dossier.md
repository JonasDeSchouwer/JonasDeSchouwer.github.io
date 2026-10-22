# Reproduction-settings dossier (raw extraction blocks)

Per-work field extractions feeding the "Reproducing the Tinker recipe" table in lit_review.typ. Each block: agent-extracted from primary sources (paper HTML + appendices + code repos), every value with a source location or NOT REPORTED. Assembled 2026-08-21.

---

## Geometry of OPD — arXiv 2606.07082v3 (read v3, 14 Jun 2026)

**META:** Code: NO repo (searched; none linked). Checkpoints: none released ("do not redistribute"; no URL).

**EVAL (Table 2):** AIME 2024 avg@8. n_seeds, prompting, grading, stop, temperature, top_p, top_k, max_tokens: NOT REPORTED. NB: **zero numeric accuracy values in the paper** — performance only as unlabeled curves (Figs. 6, 8); Table 1's numbers are bf16 update sparsity, not accuracy. App F evaluates on "five reasoning benchmarks" named only inside figure images.

**SHARED infra (Table 2):** Qwen3 family; bf16 training, fp32 grad accum, fp32 attn softmax; H200 80GB; dropout 0.0.

**OPD defaults (Table 4; C.2: all variant runs follow Table 4 except the changed factor):**
- Data: dapo-math-17k; ordering NOT REPORTED.
- On-policy: "Student-generated, on-policy" (current policy per §G.2); refresh cadence NOT REPORTED.
- 300 prompts/step × 8 samples/prompt; rollout temperature 1.0; **global batch size 64** (tension with 2400 seqs/step unexplained — report as stated). top_p/top_k/max_tokens/stop/prompt-template: NOT REPORTED.
- Steps: 299 (endpoint at eval plateau, C.1); ckpt steps {63,127,191,255,299}.
- LR 1e-6 constant; Adam, wd 0.1, betas (0.9, 0.98), eps NOT REPORTED (RLVR table has 1e-15); full FT (inferred — no LoRA mention).
- Objective: teacher token log-probabilities, OPD coef 1.0, "Advantage estimator: GRPO", KL loss disabled, entropy disabled (Table 4).
- **Advantage (App G.2 Eqs. 34–35):** A_OPD = −r, r = log π_θ(a|s) − log π_T(a|s) — "sampled-token log-probability gap, not a full-vocabulary KL"; pure OPD = α=1 in g_α = E[A^(α) ∇log π_θ]; α-mixing runs use "raw signal mixing without per-batch std normalization" → whether the default baseline group/std-normalizes is AMBIGUOUS vs Table 4's "GRPO" label. Conceptual loss App D Eq. 19 = Σ_t KL(student‖teacher) with direction left noncommittal in a caveat.
- Clipping: off-policy protocol sets IS-clip ε=1.0 ("effectively inactive"); baseline ε NOT REPORTED. Loss mask: response tokens (inferred from G.2). Framework: NOT REPORTED (clues: Megatron-style ckpt name "iter_0005375"; sglang router w/ 2 teacher replicas for off-policy arm).

**RLVR arm (Table 5):** same init; 1024 prompts × 8; temp 1.0; 1023 steps; GRPO w/ binary DeepScaler math reward; clip ε=0.2/ε_high=0.28 (DAPO clip-higher); zero-variance batch skip; batch 512; LR 1e-6 const; Adam wd 0.1 (0.9,0.98) eps 1e-15.

**G1 (main, 8B←32B):** teacher Qwen3-32B; student Qwen3-8B from SFT anchor iter_0005375.
- SFT anchor (Table 3): Qwen3-8B-Base; Dolci-Think SFT data (cites Olmo 3, 2512.13961); 5 epochs (≈5375 steps, inferred from ckpt name; C.1 "approximately 5k steps"); batch 256; LR 1e-5 cosine→1e-6, warmup 10%; Adam wd 0.1 betas (0.9, 0.95); "Qwen3 loss mask" (whole statement); max_length NOT REPORTED; full FT inferred. Post-SFT AIME: NOT REPORTED numerically.
- Post-OPD AIME: NOT REPORTED numerically. bf16 sparsity: OPD 51.6% / SFT 8.1% / RLVR 77.2% (Table 1).

**G1-MoE sub-variant:** teacher Qwen3-30B-A3B only change (Tables 6–7); sparsity 48.6%.

**G3 (4B, teacher sweep):** student Qwen3-4B from "Qwen3-4B-SFT" (**provenance NOT REPORTED**); teachers Qwen3-8B/14B/32B; sparsity 50.3/51.1/51.7%. AIME per teacher NOT REPORTED.

**G4 (14B):** student Qwen3-14B from "Qwen3-14B-SFT" (provenance NOT REPORTED); teacher Qwen3-32B; sparsity 56.6%. AIME NOT REPORTED.

(Also exist: code domain w/ DeepCoder + LCB v5, sparsity 57.1%; multi-seed robustness runs.)

---

## OPD+ — arXiv 2606.01039v1 (read v1, 31 May 2026; HTML=PDF, 9 pp)

**META:** Code: NO (no link; awesome-lists index paper-only; "only one line change of code is needed" but base impl never linked). Checkpoints: not mentioned (Table 2 is best-checkpoint, so intermediates existed).

**SHARED (B1+B2):** teacher Qwen3-8B; student Qwen3-8B-Base "initialized by supervised fine-tuning (SFT)" (App C.1). SFT dataset NOT NAMED; all SFT hyperparams NOT REPORTED. Post-SFT row (Table 2, avg@32 best-ckpt): AIME'24 50.94, AIME'25 38.61, HMMT'25 25.52, ToolUse 24.63.
- Eval: avg@32 ("mean per-sample score as estimate of single-sample accuracy"), best-checkpoint selection; n_seeds (1 run/arm inferred from Fig. 1); prompting/extraction/grading/stop/temp/top_p/top_k/max_tokens all NOT REPORTED.
- Objective: u = q_teacher/p_student per sampled token; reward −f(u), γ=0 (no reward-to-go; cites Tinker that γ>0 doesn't help). RKL OPD baseline advantage A = ln u = log q − log p (stop-grad); OPD+ corrected w(u) = −f(u)+u·f′(u) → RKL: ln u − 1; FKL: u; JSD: ½ln((1+u)/2) (Table 1). ⚠ paper has sign inconsistencies in §3.2 prose and App C.2 ("+1"); Table 1 + w=−f+uf′ is the consistent pair; constant = state-indep. baseline either way.
- Loss: PPO-like IS surrogate, ρ_θ = p_θ/p_θ0, NO clipping mentioned. Loss mask / optimizer / LR (8B runs) / LoRA rank (8B) / rollout temp/top_p/top_k/max_tokens/stop / sync cadence: NOT REPORTED.
- Framework: Tinker implied not stated ("default hyperparameter setup ... recommended by tinker of the Thinking Machine Labs", App C.1 — scoped to math; none of the defaults enumerated). Tinker LoRA-only ⇒ LoRA implied, rank unstated (rank-64 LoRA mention is the OPSD teacher only).

**B1 8B math:** DeepMath rollouts (subset/size/ordering NOT REPORTED); 100 on-policy steps; "total batch size of 64 groups" (64 prompts/batch inferred; group size NOT REPORTED). Results (avg@32 best-ckpt, Table 2): RKL OPD 64.27 / 48.54 / 34.06 (AIME24/25/HMMT25), avg 48.96; RKL OPD+ 64.69 / 48.85 / 33.44, avg 48.99; FKL OPD collapsed (clipped to SFT), FKL+ 59.79/46.88/32.50; JSD OPD collapsed, JSD+ 63.12/50.42/33.02. C.2: RKL+ ≈64% AIME'24 by step 20; RKL plateaus/degrades after step 60; JSD+ still climbing at 100.

**B2 8B tool-use:** ToolAlpaca ("ToolUse task", Tang et al. 2023); 80 steps × 256 rollouts/step ("identical architecture settings"; decomposition NOT REPORTED; tinker-defaults claim worded for math only). Eval: 68-problem test split. Results: SFT 24.63; FKL OPD 0.00 (§4 text), FKL+ 59.01; JSD OPD 0.00, JSD+ 60.29; RKL OPD 59.97, RKL+ 61.44.

(OPSD 1.7B self-distill — not a table column but only explicit HPs in paper: teacher LoRA rank 64, lr 5e-6, 1024-token generation budget.)

---

## LGR / "Your Teacher Can't Help You Here" — arXiv 2605.30833v1 (read v1; cached txt verified vs HTML)

**Primary-source recheck, 2026-09-12:** [Tables 1–2](https://arxiv.org/html/2605.30833v1#S4.T2), R1-Distill-Qwen-1.5B / Skywork-OR1-Math-7B / Polaris: main (9K) mean@8/pass@8 is 46.67/73.33, 35.42/50.00, 34.17/63.33 for AIME24/25/26; 39K is 46.25/73.33, 36.75/60.00, 34.92/66.67. The per-year maxima therefore come from different rollout caps.

**META:** Code: YES — github.com/zui-jiang/LGR (footnote p.1). Repo = SLIME fork + SGLang fork + ONE launch script `scripts/exp258-distill-1.5B-v3.sh` (1.5B LGR-topk arm only; no 7B, no plain-OPD/LGR scripts). Checkpoints: none ("does not release new assets"). Single training run per arm (checklist: no error bars). Hardware: 4×8 H20.

**EVAL (math):** AIME-24/25/26, HMMT-25/26 (Feb sessions). 8 responses/problem, mean@8 + pass@8 (App A). temp=1 (App A); top_p=1 [repo]; top_k NOT REPORTED; max_tokens 39k math (36k code) (App A; repo 39,936). Prompting: NOT REPORTED in paper; repo: chat template. Grading: NOT REPORTED; repo `--eval-rm-type deepscaler` (boxed-answer verify, inferred). Stop: NOT REPORTED. In-training eval: AIME-24 every 10 steps [repo].

**SFT:** none by authors — released R1-Distill checkpoints used as-is (§4.1, App A).

**OPD shared:** framework SLIME + SGLang (2 patches: tree-attn masks, temp-scaled prefill logits) on Megatron backend [repo]. Fully on-policy: 1 optimizer step per fresh rollout batch (`--num-steps-per-rollout 1`; REOPOLD note "staleness=1 (fully on-policy)"). Data: Polaris (App A); 53K version pinned in repo; ordering: shuffled [repo]; 300 steps (Tab 4; repo cap 3000 ⚠); batch 128 prompts × **1 rollout/prompt** (Tab 4). Prompt: chat template, max prompt 1024 [repo]. Rollout: temp 1.0 student AND teacher (App A; B.3 ablations (1,0.2),(0.8,0.8) degraded — run on Qwen3-1.7B side-config); top_p/top_k/stop NOT REPORTED; max response 9k for main arms (Tab 1 "OPD (9k)"; Table 2 sweep 3k/9k/16k/39k). LR: 1.5B 1e-5, 7B 3e-6 (Tab 4); warmup 0; constant [repo]. Full FT (inferred, no LoRA anywhere). Optimizer AdamW β(0.9,0.98) wd 0.1 (Tab 4). Advantage (Eq. 3, stop-grad §2.1): A_t = 1 + log π_θ − log π_T (note the +1). Loss: RKL Eq. 1 grad = Σ ∇log π_θ · A_t. topk arm: union student-top-10 ∪ teacher-top-10, renormalized (App A; "without it training diverges"); ⚠ repo uses topk 20/20 + PPO eps-clip 0.2/0.28. LGR extras: K=8, τ=0.2, γ=1.0, N=8 (Tab 4). Loss mask: NOT REPORTED (response-only inferred). GRPO rows = public ckpts (DeepScaler-1.5B; SkyWork-OR1-Math-7B), not trained.

**L1 math 1.5B:** teacher Skywork-OR1-Math-7B; student R1-Distill-Qwen-1.5B; LR 1e-5. AIME-24 mean@8/pass@8: Base 24.17/50.00; GRPO(public) 38.33/66.67; OPD(9k) 45.83/80.00; OPD(topk) 45.83/70.00; LGR 46.67/73.33; LGR(topk) 44.17/80.00; teacher 67.50/83.33. (Full 5-benchmark rows + Table 2 length sweep in agent report; OPD 39k AIME-24 43.33.)

**L3 math 7B:** teacher R1-Distill-Qwen-32B; student R1-Distill-Qwen-7B; LR 3e-6; else per Tab 4 (⚠ no 7B repo script). AIME-24: Base 52.08/80.00; GRPO(public=SkyWork) 67.50/83.33; OPD(9k) 62.08/76.67; OPD(topk) 57.50/80.00; LGR 61.67/83.33; LGR(topk) 61.67/80.00; teacher R1-32B 80.83/93.33. NB: GRPO public ckpt beats all distillation arms at 7B; LGR < OPD(9k) on AIME-24.

---

## AOPD — arXiv 2605.06387v3 (read v3, 13 May 2026; HTML+PDF; NO hyperparameter table — setup lives in §6.1 "Training Pipeline")

**META:** Code: NO (grepped; only awesome-lists index it). Checkpoints: NOT REPORTED.

**EVAL:** AIME'24, AIME'25, HMMT'25-Feb; Pass@1 & Pass@4. n_seeds NOT REPORTED. Pass@1 averaging NOT STATED; **inferred 16 samples/problem** (all 54 Pass@1 values lie on the 100/480 grid). Prompting/extraction/grading/stop/temp/top_p/top_k/max_tokens: all NOT REPORTED. App D.2: eval every 10 steps, P@1/4/8 curves. Tool-use: ToolAlpaca test, success rate (Table 2).

**SFT warm-up (shared):** init = respective Base; dataset "OpenThoughts" (Guha et al. ICLR 2026; version/subset/size/response-provenance NOT REPORTED); batch 128 (§6.1) ⇒ 3K steps ≈ 384K samples, 1K ≈ 128K (inferred); LoRA never mentioned (full FT inferred); max_length/loss-mask/template/LR/optimizer NOT REPORTED (stated LR 1e-5 is distillation-phase only).

**OPD phase (shared):** init = SFT ckpt; dataset DeepMath (103K cite; split/size/filtering NOT REPORTED); **90 steps** math; **LR 1e-5, batch 512** ("uniformly set", §6.1 — prompts-vs-rollouts ambiguous; rollouts/prompt NOT REPORTED); on-policy implied, freshness NOT REPORTED; sampling params/prompt construction/framework/optimizer/loss-mask/hardware: all NOT REPORTED.
- **Advantage (Eq. 1):** A_t = sg[log P_T(y_t|c_t) − log P_S(y_t|c_t)], y_t∼P_S ("reverse KL … K1 estimator"; OPD attributed to TML).
- **Loss (Eq. 2):** L_OPD = −E[(1/|y|) Σ_t A_t·log P_S(y_t|c_t)] — **length-normalized**, NO IS ratio, NO clipping, NO KL penalty.
- AOPD arm: tokens w/ A_t≤0 (τ=0 threshold) get truncated FKL on teacher top-K support, K=32, β=1. ⚠ Eq. 7 (main; unnormalized teacher probs, 1/K prefactor) ≠ App B Eq. 13 (renormalized, no 1/K) — unreconciled. Baselines: SeqKD, OPD, GKD (JSD, K=32), ExOPD.
- +40-step ToolAlpaca continual stage (8B-3K arms only, Table 2).

**Results (AIME'24 P@1; fuller rows in agent report):**
- A1-3K (32B→8B, 3K warm-up): SFT 58.96 → OPD 61.46 → AOPD 66.87 (avg P@1 45.90/49.31/53.40; GKD 66.25).
- A1-1K: SFT 45.21 → OPD 43.13 (BELOW SFT) → AOPD 53.75.
- A2-3K (8B→4B): SFT 52.29 → OPD 57.08 → AOPD 61.04.
- A2-1K (App D.1 Tab 3): SFT 36.25 → OPD 46.25 → AOPD 51.25. ⚠ Tab 3 SeqKD row digit-identical to Tab 1 4B-3K SeqKD row.
- App B.2: β=0 (RKL-on-top-K intervention) collapses to 33.7 avg P@1 vs 53.4 (β=1), entropy collapse.

---

## EOPD / Entropy-Aware OPD — arXiv 2603.07079v3 (ICML 2026; read v3, 12 Jun 2026; cached txt verified)

**META:** Code: YES — github.com/WLS04/EOPD (abstract). Caveat: full verl snapshot; referenced run script `examples/on_policy_distillation/on_policy_it.sh` MISSING; loss implemented in `core_algos.py` (`compute_policy_loss_on_policy_distill`) + `on_policy_distill_trainer.py` + teacher server `recipe/gkd/teacher/`. Checkpoints: not released.

**IDENTITY:** teacher Qwen3-8B non-thinking, enforced via chat template with pre-filled empty `<think>\n\n</think>` block (App A); students Qwen3-{0.6B,1.7B,4B}-Base (raw — **no SFT stage**, confirmed by absence).

**EVAL (E1–E3 identical):** 6 math benchmarks incl. AIME24 & AIME25 (+MATH500, AMC23, Minerva, OlympiadBench); Qwen2.5-Math eval pipeline, zero-shot, prompt suffix "Please reason step by step, and put your final answer within \boxed{}."; Avg@8 + Pass@8 (8 samples/question; Pass@k curves to k=128); temp 1.0, top-p 0.8, max 8192; n_seeds/top_k/stop/eval-template NOT REPORTED. OOD (E2 only): GPQA-D, MMLU-Pro (5-shot P@1), AlpacaEval 2.0.

**OPD/EOPD training (shared):** framework verl (KD arm: DistillKit). PPO-style iterations: B=128 prompts, B_mini=32 ⇒ 4 grad steps/rollout iteration (π_θold refresh per iteration). **1 rollout/prompt** (GRPO arm: 8). Rollout sampling: temp 1.0, top-p 1.0 (Qwen), max response 4096; top-k/stop NOT REPORTED. Prompt (repo-derived): single user msg = problem + " Let's think step by step and output the final answer within \boxed{}." LR 3e-6 cosine, AdamW (betas/wd/clip NOT REPORTED; verl defaults inferred). Full FT (inferred). Hardware 4×A100-80GB.
- **Advantage (§2.3):** Â_t = log π_te(x_t|c_t) − log π_θold(x_t|c_t) (detached; repo line matches).
- **Loss (Eqs. 7–8):** PPO-clipped surrogate on that advantage: Ã_t = max(−r_t·Â_t, −clip(r_t,1−ε,1+ε)·Â_t), r_t = π_θ/π_θold; **ε NOT REPORTED** (verl default 0.2 inferred). Token-mean aggregation over response tokens (Alg. 1).
- **EOPD arm:** + α·1[H_te>τ]·FKL on teacher top-k=16 renormalized; τ=0.8, α=1.0 (§4; repo README example shows topk 32 ⚠ — paper authoritative).
- KD arm (Tab 8): DistillKit, CE 0.5 + FKL 0.5 top-k 16, LR 1e-5, batch 128, 3 epochs, cutoff 4096.

**E1 0.6B/MATH:** data MATH (paper silent on split; repo: MATH-lighteval train, 7.5k inferred); 3 epochs (≈176 iters inferred). AIME24 Avg@8/Pass@8: KD 2.19/6.67, GRPO 4.58/10.00, OPD 2.50/10.00, EOPD 4.17/13.33. (MATH500: OPD 50.09/73.20, EOPD 52.02/76.00.)
**E2 1.7B/MATH:** identical config. AIME24: KD 10.06/20.00, GRPO 9.17/20.00, OPD 8.33/20.00, EOPD 10.42/23.33. (MATH500 OPD 67.76, EOPD 68.73.) Ablation workhorse (GKD λ, OPD-FKL, FKL placement, τ/α/k, OOD Tab 3).
**E3 4B/DAPO:** data "DAPO-Math-14k" = open-r1/DAPO-Math-17k-Processed config "en" (14,116 rows; repo-verified); **2 epochs** (≈220 iters). AIME24: KD 12.50/30.00, GRPO 14.85/30.00, OPD 18.33/26.67, EOPD 17.92/36.67. (MATH500 OPD 78.81, EOPD 80.20.)

**Gaps (confirmed NOT REPORTED):** clip ε; MATH split/size; eval seeds/top-k/stop; optimizer betas/wd/warmup; step counts (epochs only); data ordering (repo shuffle:True). Repo teacher-server sampling snippets (temp 0.7–0.8/top-p 0.95) unrelated/conflicting — paper App A authoritative.

---

## SCOPE — arXiv 2604.10688v2 (read v2, 30 May 2026; cached p2.txt + shallow-cloned repo)

**Primary-source recheck, 2026-09-12:** [Table 1](https://arxiv.org/html/2604.10688v2#S4.T1): S1 AIME24 42.7/77.9 and AIME25 30.4/50.9 (avg@32/pass@32); OPD 40.2/75.0 and 28.9/48.5. The [official README](https://github.com/machine981/SCOPE#main-results) labels these results Skywork-OR1-Math-7B, but the paper and deployment example say Skywork-OR1-7B. Grouping under Math-7B is provisional; checkpoint equivalence is unresolved.

**META:** Code: YES — github.com/machine981/SCOPE (verl fork; full scripts + eval harness; README admits ckpt/env issues). Checkpoints: SCOPE arms only — HF Machine981/SCOPE-Qwen3-1.7B, SCOPE-Deepseek-R1-Distill-Qwen-1.5B (no OPD/GRPO/KD arms). Framework: verl (FSDP + vLLM rollout) [repo; paper silent]. Infra: 20×A100-80GB (16 train + 4 teacher-serve; teacher scored via vLLM completions API echo=True — never generates).

**EVAL (shared; Table 4 + §4.1):** MATH500, AIME24, AIME25, AMC23, Minerva, OlympiadBench; Avg@32 + Pass@32; temp 0.6, top-p 0.95, top-k 20, max 32768, 32 samples/prompt; repetition penalty 1.0 (1.5B) / **1.08 (Qwen3-1.7B-Base**, "severe repetition issues", App B.1 — eval-time). n_seeds NOT REPORTED; repo Pass@32 = bootstrap (1000 resamples, seed 42), Avg@32 plain mean. Prompting [repo]: question + boxed tip, single user turn, no system prompt, chat template (⚠ literal doubled-brace `\boxed{{}}` bug). Grading [repo]: math_robust/math_verify/prime_math per benchmark. Stop: EOS/max. ⚠ committed eval driver in modified state (rep-penalty commented out; 4×6 rollouts) — Table 4 authoritative.

**SFT:** none for any arm (straight into GRPO/KD/OPD/SCOPE from released/base ckpts).

**OPD/SCOPE training (Table 3 + repo):** fully on-policy: 256 prompts/batch = mini-batch 256, 1 update epoch/batch (ρ=1; old_log_probs recomputed with training engine). Data: DeepMath (split/size NOT REPORTED; repo `sample_deepmath_train.json`, filter prompt ≤16384 chars). Steps/epochs NOT REPORTED in paper (repo: total_epochs=10, save_freq=8). Ordering: NOT REPORTED (verl shuffle:True default). **Rollouts/prompt: OPD baseline 1, SCOPE 8, GRPO 8** (Table 3 + Table 8 caption). Rollout: temp 0.6; max prompt 4096 (⚠ repo scripts 2048), max completion 12288; top_p/top_k unset → verl defaults 1.0/−1. LR **5e-5 constant**, weight decay 0.01 (Table 3); optimizer NOT NAMED (repo AdamW (0.9,0.999), 0 warmup). Full FT (FSDP bf16; no LoRA). Loss mask: response tokens, token-mean [repo].
- **Loss (§3.1 Eqs. 1–6, NO clipping in paper):** ρ_i,t = π_θ/π_old; correct-branch ℒ_MLE = −Σ_t ρ_i,t; wrong-branch ℒ_OPD = Σ_t ρ_i,t(log π_θ̄ − log π_T) (θ̄ detached; "reverse KL ... token-level log-ratio as a negative advantage"); PPL-softmax weights w^stu (correct, τ=1.0) / w^tea (wrong); binary verifier reward routes branches. Plain-OPD baseline = ℒ_OPD on all trajectories, 1 rollout/prompt.
- ⚠ **Repo deviations from equations:** scripts use `DISTILL_LOSS_TYPE=low_var_kl` ⇒ advantage = −k3 clamp(exp(Δ)−Δ−1, ±10), Δ=log π_T−log π_old — NOT the plain log-ratio; flows through PPO clipped surrogate (ε=0.2, inert at ρ=1). Group weights ×|Ω| (mean 1, not sum 1). PPL sign flags in scripts internally inconsistent — paper equations authoritative.

**S1 (1.5B math):** teacher Skywork-OR1-7B; student R1-Distill-Qwen-1.5B (released). AIME24 Avg@32/Pass@32 (Table 1): Base 29.4/76.5; GRPO 35.5/68.3; KD 26.6/71.4; **OPD 40.2/75.0; SCOPE 42.7/77.9**. 6-bench avg: OPD 52.3/73.1, SCOPE 55.2/75.0. (TACO code extension: SCOPE 44.6 avg A@32 vs OPD 42.6.) Per-step: GRPO 459s, OPD 227.5s, SCOPE 641.9s (teacher scoring 200s).

**S2 (Qwen3 math):** teacher "Qwen3-8B-Instruct" as printed (§4.1) — no such release; repo TEACHER=Qwen3-8B ⇒ actual = Qwen3-8B (inferred); teacher think-mode NOT REPORTED. Student Qwen3-1.7B-Base raw (no SFT). Same HPs as S1. AIME24: Base 3.7/26.7; GRPO 7.5/27.2; KD 4.6/18.9; **OPD 12.2/31.5; SCOPE 13.3/31.5**. 6-bench avg: OPD 30.6/55.9, SCOPE 32.5/58.6 (KD Pass@32 avg 46.8 < base 49.2).

---

## Rethinking OPD — arXiv 2604.13016v2 (read v2, 15 Apr 2026; cached txt/html + thunlp/OPD repo)

**Primary-source recheck, 2026-09-12:** [Section 3.2/Fig. 4](https://arxiv.org/html/2604.13016v2#S3.SS2) confirm R7 (R1-Distill-1.5B / Skywork-OR1-Math-7B / DAPO-Math-17K). Fig. 4 aggregates AIME24, AIME25 and AMC23; 16.9% gap recovery is not an individual AIME score. [Section 4.1/Fig. 6](https://arxiv.org/html/2604.13016v2#S4.SS1) report no improvement for the R1-Distill-7B control, with unlabelled AIME curves. Neither row should be omitted for lack of numeric table entries.

**META:** Code: YES — github.com/thunlp/OPD (verl v0.7.0 [repo-stated, paper silent]; SFT via LLaMA-Factory v0.9.5; eval from thunlp/JustRL). Released: HF lllyx/Qwen3-1.7B-SFT, lllyx/Qwen3-4B-Base-GRPO (teacher), dataset lllyx/OpenThought3-Qwen3-4B (200K rollouts). NO OPD-trained student ckpts. Hardware 8×A800-80GB. ICML 2026 FoGen wksp.

**SHARED DEFAULTS (App A.2 Table 2 verbatim):** train temp 1.0 | global batch 64 | mini batch 64 | rollout n=4 | LogProb top-K 16 | strategy Student Top-K | top-p 1.0 | max prompt 1024 | max response 7168 | LR 1e-6 | 1 epoch | KL coef 0.0.
- Strictly on-policy: 1 grad update/rollout batch; ratio≡1 (old_log_prob=log_prob.detach() in this regime [repo]). Full-param FSDP (fp32 dtype), no LoRA; AdamW verl defaults (inferred), constant LR no warmup [repo]. Data DAPO-Math-17K, **shuffle=False (sequential)** [repo], 1 epoch ≈270 steps (inferred). 64×4=256 seqs/step. Teacher logits temp 1.0 [repo]. Loss mask: response tokens, token-mean. No format/correctness reward.
- Default template = "Original DAPO Template" ("Solve the following math problem step by step. ... Answer: ..."); non-thinking instruct students need enable_thinking=False (README).
- **Objective (default Student Top-k, k=16; §2.2 Eq. 5):** L = E[Σ_t KL(p̄_t^(S_t) ‖ q̄_t^(S_t))], S_t = student top-16, both renormalized. Sampled-token variant Eq. 3 = TML estimator (LOG_PROB_TOP_K=0 fallback). **Implementation = advantage-PG, not differentiable KL** [repo]: per (t, v∈top-16) detached reward −(log p−log q)·p̄_t(v) (un-renormalized logs ⚠ slight deviation from Eq. 5) used directly as advantage (`token_reward_direct`, no baseline), PPO-vanilla loss, ratio 1 ⇒ score-function gradient of subset RKL. Fig 19 calls it "PG Loss".

**EVAL:** AIME'24, AIME'25, AMC'23; avg@16 (16 samples, per-request seed=idx); temp 0.7, top-p 0.95, max 31,744; eval prompt = JustRL-style boxed (differs from training template); grading math-verify-based (JustRL pipeline). ⚠ README: verl in-training validation under-reads ~5–7pp.

**Settings (defaults unless noted; ALL post-OPD results figure-only — no result tables in paper):**
- R1 1.7B-Base←Qwen3-4B(NT): Fig 2/17; underperforms R2.
- R2 1.7B-Base←Qwen3-4B-Base-GRPO: teacher author-trained (App A.1 Table 1: GRPO on DAPO-17K, boxed template, batch 64, n=8, lr 1e-6, 1 epoch; ckpt released). §5.2 data variant: DAPO vs size-matched DeepMath subset deduped vs DAPO (exact + semantic ≥0.6 cosine, all-mpnet-base-v2/FAISS; App C.3); DAPO wins (Fig 10).
- R3 cold-start: init lllyx/Qwen3-1.7B-SFT; OPD on ≈30K held-out OpenThoughts3 math prompts (repo parquet confirms); control = same from Base. SFT-init wins throughout (Fig 8); post-SFT score figure-only. **R3 SFT stage (App C.1 + Table 3):** 200K OpenThoughts3 math prompts × 1 response by Qwen3-4B(NT) (temp 0.7, top-p 0.95, max 12,288, boxed template; truncated/degenerate filtered; rejection sampling ≤3 [repo]); full-param SFT, template qwen3, 1 epoch, cutoff 14,336, global batch 64 (8×8), lr 1e-5 cosine, warmup 0.05, BF16, LLaMA-Factory (ZeRO-2, FA2, Liger [repo]).
- R4 1.7B(NT)←Qwen3-4B(NT): limited gains (Fig 4R). R5 1.7B(NT)←Qwen3-4B-NT-RL-Math (external, 2602.12125): strong gains (Fig 4R).
- R6 R1-Distill-1.5B←R1-Distill-7B: fails ("no improvement despite stronger teacher"); AUROC 0.75 (§6.2). R7 ←Skywork-OR1-Math-7B: strong gains (Fig 4L).
- R8 ←JustRL-1.5B (workhorse): recovers >80% of gap (§4.1); overlap 72%→>91%; overlap mass 97–99%; AUROC 0.73. Arms: top-k support ablation (student/overlap/non-overlap, Fig 7); template variant (JustRL-aligned wins, ~80%→~85% recovery); max-length sweep {0.5K–15K}, 200 steps each (3K/7K best; 10K/15K late collapse); teacher-continuation probe (+0.37@1K → +0.02@16K prefix); k∈{1,4,16,64} vs sampled-token (§6.3, Fig 15 figure-only: sampled ≈ k≥4; top-1 worst).
- R9 JustRL-1.5B←R1-Distill-1.5B: regresses to pre-RL level (Fig 5). R10 ←R1-Distill-7B: near-identical regression.
- R11 R1-Distill-7B←Skywork-OR1-Math-7B (improves) / ←R1-Distill-14B (little improvement) (App B.3, Fig 20).

---

## TML blog + cookbook — blog 2025-10-27; cookbook commit 330b73d8 (agent-verified vs blog, SDK wheels, docs, HF)

**⚠ TEACHER FINDING:** blog footnote ("On-policy distillation" subsection): "We actually use Qwen3-8B as a teacher, as it performs slightly better. However, for the purpose of comparing compute, we can still measure the FLOPs of the 32B model." ⇒ headline run's actual teacher = Qwen3-8B; Qwen3-32B is the FLOPs-accounting fiction.

**META:** Code: recipes/distillation/ + distillation/ modules; LoRA-on-Tinker path only (full-FT was TML-internal, no public full-FT code). Checkpoints: none on HF; **Dec 19 2025 README (commit 8f32fc851c) added Tinker-hosted ckpts** for LoRA runs: SFT ranks 8/32/128 `tinker://.../weights/final`; OPD ranks 8/32/128 `tinker://.../sampler_weights/000080` (step-80!). Nothing for full-FT headline.

**Algorithm (shared):** fully synchronous; sampler refreshed after EVERY optimizer step (1-step lag corrected by IS ratio). A_t = −1.0·(log π_sampled − log π_teacher)·mask; env rewards ≡0; discount 0 (blog: >0 didn't help). Loss = unclipped importance_sampling: −Σ exp(target_lp − sampling_lp)·A ("ppo" exists, unused). Loss mask: sampled completion tokens. Teacher: compute_logprobs only. Rollout sampling: NOTHING passed ⇒ tinker SDK 0.2.2 defaults **temp 1.0, top_p 1.0, top_k −1**. Stop: renderer stop seq; ⚠ blog-day default resolves Qwen3-8B-Base → `role_colon` renderer (stop "\n\nUser:"); SFT stage explicitly `qwen3` (<|im_end|>); current main resolves renderer from ckpt — intended qwen3. Prompt: no system prompt, raw question, truncated 1024 tok. Data: DeepMath-103K question field, SEQUENTIAL (no shuffle). group_size 4. num_substeps 1. **Optimizer: Adam(β1 0.9, β2 0.95, eps 1e-8); blog-era SDK had NO weight_decay/grad_clip fields at all.** SFT LR schedule linear→0; OPD constant.

**EVAL AIME'24: NOT REPORTED at blog time** (no n/metric/temp/prompting/grading anywhere; blog-day repo has no wired-in AIME eval — only generic inspect_ai bridge + math_rl grader, evaluator_builders empty). **June 16 2026 README (Qwen3.5 rerun): "evaluated with temperature=1.0, top_p=1.0, top_k=-1, and max_tokens=64000"** — samples/metric still unstated; strictly covers the rerun. GPQA-D numbers in blog table = Qwen3 TR Table 21 quotes, not TML measurements.

**T1 (full-FT headline):** student Qwen3-8B-Base; teacher: presented 32B, actual 8B (footnote). SFT: OpenThoughts3-1.2M "400k prompts" (cookbook 128·3000=384k); full FT LR 1e-4 (README); batch 128 [C/I]; 3000 steps [I]; max_len 16384; ALL_ASSISTANT_MESSAGES; qwen3 renderer; linear decay; 1 epoch. Post-SFT AIME'24 60% [blog]. OPD: init = SFT ckpt; DeepMath (README; blog doesn't name it); ~150 steps ≈77K prompts, 4 samples/prompt [blog footnote]; ⇒ ≈512 prompts/batch [arith, matches README]; LR 5e-5 full-FT (README); max_tokens 4096 [C]; rest shared. Post-OPD AIME'24 70% [blog]. No other TML-measured evals.

**T2 (cookbook LoRA-128):** README commands: SFT lr 1e-3, batch 128, rank 128, 3000 steps → ~55% AIME'24; OPD lr 1e-4, groups_per_batch 512, rank 128, teacher default Qwen/Qwen3-8B → ~65% "after 100 steps". Init via load_checkpoint_path (explicit in Dec-2025 README w/ tinker URI). Batch 512×4=2048 seqs/step. Rest shared. Tinker-hosted ckpts ranks 8/32/128 (SFT final; OPD step-80).

---

## One-shot OPD (Rethinking OPD II) — arXiv 2609.04172v1 (read v1, 3 Sep 2026; pypdf text + Thinking-Space/One-Shot-OPD repo @ 6b6b1782, 2026-09-09)

Full note with figure readouts and code snippets: `per-paper/one-shot-opd.md`.

**META:** Code: YES — github.com/Thinking-Space/One-Shot-OPD (vendored verl 0.9.0; vllm 0.28.0, torch 2.13.0, transformers 5.10.4; Python 3.12). Released: HF `Thinking-Space/UltraData-IF-1.5B` (IF teacher) only; NO OPD-trained students; data rebuilt by `data/prep/` from ModelScope. Hardware: one 8×H100/A100-80GB node, teacher shares the actor pool.

**SHARED DEFAULTS (Table 3 = README = `recipe/_common.sh`):** rollout batch 64 | mini batch 64 | responses/prompt 1 | KL coef 0 | LogProb top-K 0 (sampled) / 16 (§3 math runs), strategy student top-K, weights `student_p` (renormalized over the K) | loss agg token-mean | train temp 1.0 | top-p 1.0 | top-k −1 [code] | AdamW β (0.9, 0.999), wd 0.01 | LR 1e-6 constant, no warmup | grad clip 1.0 | max prompt 1024 math (paper; launcher default 4096, 8192 in oneshot mode ⚠) | max response 7168 (paper+code; README says 7680 ⚠) | `data.shuffle=True` | 1 epoch, steps via `total_training_steps` (300 in Figs 3–8, 1000 in Figs 1/14; 17,917/64 = 280 per epoch [inf.]) | strictly on-policy (mini batch = batch, ppo_epochs 1 ⇒ ratio ≡ 1, PPO clip inactive) | advantage `token_reward_direct` (reward used as-is, no baseline/discount); loss `vanilla_topk` (PPO-clipped per candidate, summed over K); entropy coef 0 | teacher: frozen FSDP forward, bf16, temp 1.0, k1 estimator.

**Objective (Eq. 3, code):** A_t = Σ_{v∈top16(q)} q̃(v) · [log p_T(v) − log q(v)], q̃ = q renormalized over the 16; raw (un-renormalized) log-probs in the gap. Non-math domains + the §4–5 analysis runs use the sampled-token Eq. 2 (= TML's estimator).

**PROMPT:** bare problem as one user turn, DAPO "Answer: …" wrapper stripped, no system prompt, no boxed instruction [code]; model chat template with `<｜Assistant｜><think>\n` generation prompt [tokenizer_config]. One-shot corpus = DAPO row 22 ("medium", 4/8 pre-training pass rate) × 64 rows.

**EVAL:** MATH-500, AMC'23, AIME'25 avg@16, cap 31,744; in-training verl loop temp 0.7, top-p 0.9 [code; README: all reference numbers produced under these]. **AIME'24 NOT evaluated.** Grading `ttrl_mathverify`: boxed rule grader, else math-verify on the last 2,000 chars [code]. n_seeds NR (1 run/arm inferred). Other domains: LCB v6 avg@3 (65,536), Multi-IF turn-3 (16,384/turn), BFCL v3 avg@8 (4,096).

**F1 math 1.5B (workhorse):** student DeepSeek-R1-Distill-Qwen-1.5B as released (no SFT); teacher JustRL-DeepSeek-1.5B (`hbx/JustRL-DeepSeek-1.5B`: GRPO from the same checkpoint on DAPO-Math-17k, 16K cap, 4,000+ steps; card AIME24@32 29.90 → 52.60). Data DAPO-Math-17K (17,917 unique). Results (Fig 3 read-off, step 0 → 300, top-16 objective): MATH-500 0.840 → full 0.891 / one-shot 0.883 (teacher 0.916); AIME'25 0.231 → 0.343 / 0.337 (0.386); AMC'23 0.706 → 0.861 / 0.835 (0.882); macro-avg 59.1 → 69.8 / 68.5 (72.7). Step 1000 (Fig 1): 72.1 / 68.4. Sampled-token runs (§4.1 + fn 1): one-shot 59.1 → 66.9, full-data 70.8 at step 300. Top-16 overlap 0.82 → 0.95.
- Sub-variants: query difficulty t59/t22/t56 (Fig 6, comparable gains); response cap {3k,5k,7k}; temp {1.0,0.6,0.2}; 4-/16-shot via BGE-M3 clusters (16-shot = full-data, coverage 98.9%); off-policy fixed-64-trajectory runs (§4.2/5.3); template-only input `<|begin_of_sentence|><|User|><think>\n` and WildChat prompts ≈ real queries (§7.1); one-shot RLVR comparison (GRPO 8×64, §7.2).

**F2 family checks (math, one-shot, MATH-500+AMC avg, Fig 4):** Llama-3.2-3B-Instruct ← GT-Llama-3.2-3B-Instruct-MATH 28.2 → 40.2; OLMo-3-7B-Instruct-DPO ← OLMo-3-7B-Instruct 70.8 → 82.4; R1-Distill-1.5B 77.1 → 85.5.

**F3 other domains (R1-Distill-1.5B unless noted, sampled-token):** code ← Nemotron-Research-Reasoning-Qwen-1.5B-v2-RLVE (launcher default `DeepCoder-1.5B-Preview` ⚠), Open-R1 Codeforces (launcher: TACO ⚠), LCB v6 recovery 73%; IF ← UltraData-IF-1.5B (author GRPO, step 600), Multi-IF recovery 66%; agentic Qwen2.5-Coder-1.5B-Instruct ← Hammer2.1-1.5b, xLAM-60K, BFCL v3 recovery 64%. MOPD (math+code+IF, Table 8 not extracted): 43.5 → 52.8 full-data, 50.1 one-shot, 52.9 16-shot per domain; separate full-data OPD 53.8.

**RELATION TO OUR RUNS:** the pairing Part I marks as failing (R6: R1-1.5B ← R1-7B; R11: R1-7B ← R1-14B "little improvement") is ours; this paper's working teacher is an RL continuation of the identical student on the same prompts.
