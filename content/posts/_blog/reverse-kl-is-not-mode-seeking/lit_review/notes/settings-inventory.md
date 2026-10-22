# Settings inventory — OPD reproduction-table columns

Enumeration of every distinct training-run *setting* (future table column) per work. A setting = one (student+init, teacher, data/domain, FT regime) configuration; loss/method arms run on the same configuration are listed as arms, not settings. Sub-variants (changed init, data control, teacher-grid cell) listed under their parent. Properties per setting to be extracted in a later pass.

Cite-only works excluded (2605.17497, 2606.10385, surveys 2604.00626, 2606.22793). Versions read noted per work. ⚠ = detail from a summarizing fetch or otherwise UNVERIFIED — re-verify during property extraction.

## TML blog + tinker-cookbook (Lu & Thinking Machines Lab, Oct 27 2025; cookbook commit 330b73d8)

- **T1 — Math headline (full FT)**: Qwen3-8B-Base, SFT on 384k OpenThoughts-3 (QwQ-32B traces) → OPD, teacher Qwen3-32B, DeepMath-103K. Blog "Distillation for reasoning". 60→70% AIME'24.
- **T2 — Math cookbook (LoRA-128)**: same pipeline, LoRA rank 128, teacher Qwen3-8B, README hyperparams. ~55→~65% @100 steps. Cookbook README.
- **T3 — LoRA-rank sweep**: SFT and OPD compute-efficiency comparison across LoRA ranks (rank 32 highlighted), teacher Qwen3-8B (blog footnote). Sub-variants by rank. Blog figure experiment-on-policy-distillation-loras.
- **T4 — RL-teacher distillation (Discussion)** ⚠: teacher = authors' RL-trained checkpoint (Qwen3-8B-Base + rank-128-LoRA RL on DeepMath), distilled back into Qwen3-8B-Base on DeepMath; auxiliary run: the RL training itself. Blog "dense supervision" Discussion, experiment-self-distillation figure.
- **T5 — Single-prompt data reuse (Discussion)** ⚠: same RL-teacher lineage; 1 DeepMath prompt, 20 consecutive steps × 256 rollouts. experiment-self-distillation-multiepoch figure.
- **T6 — Personalization mid-train (SFT-only)**: Qwen3-8B (assistant) mid-trained on internal docs + Tulu3 mixtures, ratio sweep; full-FT and LoRA variants. Blog "personalization".
- **T7 — Personalization OPD recovery**: student = mid-trained Qwen3-8B (70/30 docs/chat), teacher = original Qwen3-8B, Tulu3 prompts. Cookbook README tulu3 command (groups_per_batch 64).
- **T8 — SFT-on-own-rollouts control (SFT-only)** ⚠: Qwen3-32B SFT on its own T=1.0 Tulu3 samples, LR sweep. Blog Discussion (continual learning), experiment-sft-on-policy figure.

## AOPD — Asymmetric On-Policy Distillation (2605.06387v3)

- **A1 — 32B→8B math**: Qwen3-8B-Base + OpenThoughts SFT warm-up; teacher Qwen3-32B; DeepMath. Sub-variants: 3K-step ("strong") vs 1K-step ("weak") warm-up, both main-text. Arms: SFT ckpt, SeqKD, OPD, ExOPD, GKD, AOPD. §6.1, Table 1.
- **A2 — 8B→4B math**: Qwen3-4B-Base + 3K warm-up; teacher Qwen3-8B; DeepMath. Sub-variant: 1K warm-up (App D.1, Table 3). Same arms.
- **A3 — 8B tool-use continual**: init = each method's math-distilled 3K/8B checkpoint from A1; ToolAlpaca, 40 steps. Teacher unstated ⚠ (presumably Qwen3-32B). Arms: OPD, ExOPD, GKD, AOPD. §6.3, Table 2.
- LoRA never mentioned; full FT implied ⚠.

## OPD+ (2606.01039v1)

- **B1 — 8B math**: Qwen3-8B-Base SFT-initialized (SFT data unnamed ⚠); teacher Qwen3-8B; DeepMath rollouts; Tinker-recommended defaults (Tinker = LoRA-only service, so LoRA implied ⚠). Arms: {OPD, OPD+} × {FKL, JSD, RKL} + SFT baseline. §4, App C.1, Table 2.
- **B2 — 8B tool-use**: same pair; ToolAlpaca ("ToolUse task [14]"), 80 steps × 256 rollouts; 68-problem test split. Same 6 arms. Table 2.
- **B3 — OPSD 1.7B self-distill math**: Qwen3-1.7B (instruct); teacher = same model + fixed rank-64 LoRA conditioned on privileged info (problem + reference solution); training prompt set unnamed ⚠. Arms: OPSD vs OPSD+ × divergences. Table 2 last column.

## Lightning OPD (2604.13010v2; github jet-ai-projects/Lightning-OPD)

- **C1 — 8B→4B math+code**: Qwen3-4B-Base, SFT on OpenThoughts-3 prompts w/ Qwen3-8B responses; teacher Qwen3-8B; OPD on DAPO-Math-17k + 30K EpiCoder subset (mixed vs separate runs unstated ⚠). Arms: SFT, online OPD, Lightning (offline) OPD. Sub-variants: 2×2 SFT-teacher × OPD-teacher grid {Qwen3-8B, QwQ-32B} (Table 4, math only).
- **C2 — 32B→8B math+code**: Qwen3-8B-Base, SFT w/ Qwen3-32B responses; teacher Qwen3-32B; same data/arms. Sub-variants: 2×2 grid {Qwen3-32B, QwQ-32B}.
- **C3 — MoE 30B-A3B math+code**: Qwen3-30B-A3B-Base; teacher Qwen3-30B-A3B-Thinking-2507; same pipeline; arms SFT + Lightning only (online OPD OOM). §4.4, Table 3.
- Full-parameter per GitHub configs (paper doesn't state) ⚠.

## Tokenizer Barrier (2606.09456v1)

- **D1 — Qwen3-8B→Llama-3.1-8B**: Llama-3.1-8B base + authors' SFT on 400K OpenThoughts; teacher Qwen3-8B (cross-tokenizer); OPD on 20K OpenThoughts+DeepMath prompts. Arms: SFT, +ALM, +CDM, +OPD (token-mapping). Table 1.
- **D2 — DeepSeek-R1→R1-Distill-Qwen-7B**: student R1-Distill-Qwen-7B (fresh OpenThoughts SFT per "all settings" sentence, ambiguous ⚠); teacher DeepSeek-R1 (variant unspecified ⚠). Same arms.
- **D3 — Same-tokenizer control Qwen3→Qwen3**: teacher Qwen3-8B; student = unspecified Qwen3 model + own SFT init ⚠. Arms: SFT, OPD. §5.5, Table 3.

## Geometry of OPD (2606.07082v3)

- **G1 — Main 8B←32B math**: Qwen3-8B from authors' SFT anchor (Qwen3-8B-Base SFT'd on Dolci-Think); teacher Qwen3-32B; dapo-math-17k. Arms: OPD, RLVR (GRPO same config), rank-16 projected OPD, token-sparsified, off-policy variant, OPD/RLVR interpolation, multi-seed. Sub-variant: MoE teacher Qwen3-30B-A3B. Tables 3–7.
- **G2 — SFT-anchor run**: Qwen3-8B-Base → SFT on Dolci-Think (produces shared anchor; also rank-16 projected-SFT arm). Table 3, §4.2.
- **G3 — 4B student, teacher sweep**: Qwen3-4B from "Qwen3-4B-SFT" (provenance unstated ⚠); teachers Qwen3-{8B, 14B, 32B}; math. Table 6.
- **G4 — 14B student**: Qwen3-14B from "Qwen3-14B-SFT" ⚠; teacher Qwen3-32B; math. Tables 1, 6.
- **G5 — Code**: Qwen3-8B from same anchor; teacher Qwen3-32B; DeepCoder data; LiveCodeBench v5. App C.2.

## SCOPE (2604.10688v2)

- **S1 — Math 1.5B**: DeepSeek-R1-Distill-Qwen-1.5B (as released); teacher Skywork-OR1-7B; DeepMath. Arms: base, GRPO, KD, OPD, SCOPE. Table 1.
- **S2 — Math Qwen3**: Qwen3-1.7B-Base; teacher "Qwen3-8B-Instruct" (paper's naming) ⚠; DeepMath. Same arms.
- **S3 — Code 1.5B**: same pair as S1; TACO (25,202 problems); HumanEval/Codeforces/LCB eval. App B.2, Table 2.

## Anti-Self-Distillation (2605.11609v1)

Privileged self-teacher throughout (same network conditioned on prompt + verified solution + correctness feedback); DAPO-Math-17k, 200 steps for math settings. Arms: base, +GRPO, +SD, +AntiSD.

- **N1 — Qwen3-8B math** (sub-variant: continual AntiSD from GRPO@200 ckpt, +50 steps)
- **N2 — Qwen3-4B-Instruct-2507 math** (sub-variant: continual AntiSD)
- **N3 — Olmo-3-7B-Instruct math**
- **N4 — Olmo-3-7B-Think math**
- **N5 — Qwen3-30B-A3B math**
- **N6 — Qwen3-8B code**: Dolci-RLZero code-RL data; HumanEval+/MBPP+; arms +GRPO, +AntiSD only. §4.1, Table 2.

## Rubric-based OPD (2605.07396v1)

Student Qwen3-4B instruct (no SFT stage) unless noted; GRPO-style RL on rubric rewards.

- **P1 — Black-box math**: teacher GPT-5.2-chat-latest (API); DAPO-Math-17K. Arms: ROPD, T-Judge, OVD, GAD. Sub-variants: reward-construction ablations (m=1, no rubric sharing, no blind scoring). Table 1.
- **P2 — Black-box science**: RaR-Science-20K; ckpt selection GPQA-Diamond.
- **P3 — Black-box medical**: RaR-Medical-20K; selection HealthBench.
- **P4 — White-box math**: teacher Qwen3-30B-A3B; DAPO-Math-17K; arms ROPD, SFT, LOPD (logit OPD ← the TML-family baseline), ExOPD. Verifier identity contradictory between §3.1 and App E ⚠. Table 2.
- **P5 — Gemma cross-arch**: student Gemma3-4B-it; teacher GPT-5.2; math (dataset inferred DAPO ⚠). Arms: base, OVD, GAD, ROPD. Table 3.

## Entropy-Aware OPD (2603.07079v3, ICML 2026)

Teacher E1–E3: Qwen3-8B non-thinking. verl (on-policy) + DistillKit (off-policy).

- **E1 — Qwen3-0.6B-Base / MATH**. Arms: KD, GRPO, OPD, EOPD.
- **E2 — Qwen3-1.7B-Base / MATH** (workhorse): + many objective arms (entropy bonus, advantage shaping, GKD λ, OPD-FKL, full/random FKL placement, BD-KD, AKL, ToDi, τ/α/top-k ablations).
- **E3 — Qwen3-4B-Base / DAPO** ("DAPO-Math-14k" per Table 2 ⚠ — 14k as printed, not 17k).
- **E4 — Llama cross-family**: Llama-3.2-3B-Instruct ← Llama-3.1-8B-Instruct; data unnamed (likely MATH) ⚠. App G, Table 10.

## Your Teacher Can't Help You Here / LGR (2605.30833v1)

SLIME + SGLang; math data Polaris, code data DeepScaler-code, uniform across sizes. Main arms per setting: OPD (9k), OPD topk (student∪teacher top-10 renorm), JSD β=0.2, REOPOLD, LGR, LGR topk. GRPO rows = public checkpoints, not author-trained.

- **L1 — Math 1.5B**: R1-Distill-Qwen-1.5B ← Skywork-OR1-Math-7B; + max-gen-length sweep {3k,9k,16k,39k}, confidence-metric and γ sweeps, future-KL GAE arms, FKLD/JSD/RKLD arms.
- **L2 — Code 1.5B**: R1-Distill-Qwen-1.5B ← DeepCoder-14B.
- **L3 — Math 7B**: R1-Distill-Qwen-7B ← R1-Distill-Qwen-32B.
- **L4 — Code 7B**: same pair, code.
- **L5 — FAILED Qwen3 pair**: Qwen3-1.7B ← Qwen3-32B, "highly unstable"; still used for App B.2 renormalization and B.3 rollout-temperature ablations ⚠ (data presumed Polaris).
- **L6 — Homogeneous diagnostic**: models never named ⚠; teacher = RLVR-trained version of student base; ≥80 steps for Figs 5–6.

## Rethinking OPD (2604.13016v2)

All math; default data DAPO-Math-17K; default objective Student Top-k OPD (k=16). Full-parameter SFT stated only for cold-start.

- **R1 — Qwen3-1.7B-Base ← Qwen3-4B (Non-thinking)**, DAPO. §3.1.
- **R2 — Qwen3-1.7B-Base ← Qwen3-4B-Base-GRPO** (author-trained teacher via zero-RL GRPO on DAPO; auxiliary run App A.1). Sub-variant: DeepMath subset deduped vs DAPO (§5.2, App C.3).
- **R3 — Cold start**: Qwen3-1.7B-SFT (full-param SFT on 200K Qwen3-4B-NT rollouts over OpenThoughts3 math prompts; auxiliary run App C.1) ← Qwen3-4B (NT), OPD on ~30K held-out OpenThoughts prompts. Control arm: same OPD from Qwen3-1.7B-Base init. §5.1.
- **R4 — Qwen3-1.7B (NT, instruct) ← Qwen3-4B (NT)**, DAPO. §3.2.
- **R5 — Qwen3-1.7B (NT) ← Qwen3-4B-Non-Thinking-RL-Math** (external teacher from arXiv 2602.12125, NOT author-trained), DAPO. §3.2.
- **R6 — R1-Distill-1.5B ← R1-Distill-7B**, DAPO (same-pipeline, fails). §3.2, §4.1.
- **R7 — R1-Distill-1.5B ← Skywork-OR1-Math-7B**, DAPO (succeeds). §3.2.
- **R8 — R1-Distill-1.5B ← JustRL-1.5B** (workhorse): top-k family arms, template variant, max-length sweep {0.5K…15K}, top-k∈{1,4,16,64} vs sampled-token. §4–6.
- **R9 — JustRL-1.5B ← R1-Distill-1.5B** (reverse distillation to own pre-RL ckpt). §3.3.
- **R10 — JustRL-1.5B ← R1-Distill-7B**. §3.3.
- **R11 — R1-Distill-7B ← {Skywork-OR1-Math-7B (a: succeeds), R1-Distill-14B (b: stagnates)}**, App B.3.

## Totals

TML 8; AOPD 3; OPD+ 3; Lightning 3; TokBarrier 3; Geometry 5; SCOPE 3; AntiSD 6; ROPD 5; EOPD 4; LGR 6; Rethinking 12 (R11 counts a/b) ⇒ **61 settings** across 12 works, before promoting any sub-variants to their own columns.
