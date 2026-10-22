# H — Async OPD sweep (2026-09-14): every candidate checked

Scope: student trains on rollouts sampled by an EARLIER version of itself (pipelined trainer/generator, replay buffer, precomputed rollouts from an earlier student checkpoint, multi-epoch reuse, mixed fresh/stale batches); teacher fixed. Out: rollouts from the teacher or another model (SFT-style "off-policy distillation"), trainable/self teachers, RL-only async findings.
Convention: reverse KL = KL(student ‖ teacher). Objective vocabulary: `rkl_pg` (sampled-token score-function estimator), `rkl` (full-vocab), student-top-k / teacher-top-k, `fkl`.
Tiers: A proof/argument that distribution-level RKL tolerates staleness better than `rkl_pg` (or the reverse); B async-vs-sync numbers with explanation; C numbers without; D quotes; E production use; F async is the paper's main point; G async used incidentally.
Every in-scope source has a per-paper note in `per-paper/` written from the primary source only. Discovery: arXiv/web search (queries logged below), the two awesome-OPD lists, the two OPD surveys, and a keyword screen (asynchron/stale/off-policy/replay/ppo_epoch/importance/behavior policy/old policy/epochs/reuse) over the full text of every OPD paper in `references.bib` (`scratchpad/async/screen/screen.tsv`).

## In scope (per-paper note exists)

| Source | arXiv / URL | Note | Tiers |
|---|---|---|---|
| AsyncOPD (Kang, Galim et al., Microsoft) | 2606.24143 | per-paper/asyncopd.md | A (reversed), B, D, F |
| Lightning OPD (Wu et al., NVIDIA) | 2604.13010 | per-paper/lightning-opd.md | B, D, F (A-adjacent) |
| NPD / Near-Policy Distillation (Rang, Bi et al., Huawei) | 2605.05940 | per-paper/npd.md | B, D, F |
| f-OPD (Chen, Zhang, Wu, PolyU) | 2605.17862 | per-paper/f-opd.md | B, D, F |
| Blockwise Policy-Drift Gating (Zheng) | 2606.24084 | per-paper/blockwise-drift-gating.md | C, D, F(weak) |
| Offline on-policy KD (Rang et al. 2025) | 2509.26497 | per-paper/offline-onpolicy-kd.md | C, E, F |
| DistiLLM (Ko et al. 2024) | 2402.03898 | per-paper/distillm.md | B, C, D, F |
| DistiLLM-2 (Ko et al. 2025) | 2503.07067 | per-paper/distillm-2.md | C, G |
| Nemotron 3 Ultra (NVIDIA) | 2606.15007 | per-paper/nemotron-3-ultra.md | D, E, G (192K = max generation length; staleness one step) |
| Nemotron-Cascade 2 (NVIDIA) | 2603.19220 | per-paper/nemotron-cascade-2.md | D (negative E: synchronous by design) |
| MiMo-V2-Flash (Xiaomi) | 2601.02780 | per-paper/mimo-v2-flash.md | D, E, G (partial rollout) |
| Kimi K3 (Moonshot) | 2607.24653 | per-paper/kimi-k3.md | D, E, G (partial rollout) |
| GLM-5 (Zhipu) | 2602.15763 | per-paper/glm-5.md | none (distillation on the synchronous backbone) |
| DeepSeek-V4 | 2606.19348 | per-paper/deepseek-v4.md | none (silent; A-adjacent variance argument only) |
| Thinking Machines OPD blog + tinker-cookbook | thinkingmachines.ai/blog/on-policy-distillation | per-paper/tinker-opd-blog.md | D (recipe sync-only; async RL trainer not wired in) |
| Qwen3 technical report | 2505.09388 | per-paper/qwen3.md | none for OPD (its off-policy quote is about GRPO) |
| GKD (Agarwal et al.) | 2306.13649 | per-paper/gkd.md | none |
| MiniLLM (Gu et al.) | 2306.08543 | per-paper/minillm.md | D, G (4 inner epochs, clipped ratio) |
| verl async OPD recipe (docs + code) | verl.readthedocs.io/…/async-on-policy-distill | per-paper/verl-async-opd.md | D, F |
| slime / NeMo-RL / ROLL / AReaL / OpenRLHF (code audit) | repos | per-paper/frameworks-async-opd.md | D (nemo-rl asserts A), G (AReaL, ROLL, slime); OpenRLHF none |
| Formula-driven OPD survey | 2606.22793 | per-paper/formula-driven-opd-survey.md | D |
| Revisiting OPD failure modes | 2603.25562 | per-paper/revisiting-opd-failures.md | D (weak); 128/64 batch = 2 updates per batch (inferred) |
| Decoupling KL and Trajectories | 2605.16826 | per-paper/decoupling-kl-trajectories.md | A (partial), D |
| Trust Region OPD | 2606.01249 | per-paper/trust-region-opd.md | none (off-policy = teacher prefix); D (weak) |
| KD with Training Wheels (Liu et al., Amazon) | 2502.17717 | per-paper/kd-training-wheels.md | G (weak), D (weak) |
| TOP-D / Trust Region Policy Distillation (Xie et al.) | 2607.04751 | per-paper/top-d.md | B, D, F |
| Entropy-Aware OPD (Jin et al.) | 2603.07079 | per-paper/entropy-aware-opd.md | D, G |
| KL Agreement Trap (Xin et al.) | 2606.09471 | per-paper/kl-agreement-trap.md (appended) | D only; strictly sync |

## Checked and excluded (primary source read; deciding sentence)

- **ReOPD / Multi-Turn OPD with Prefix Replay** (2607.04763v3, Microsoft/UvA): prefixes are teacher-collected, never stale-student. §3: "We instead train off-environment from a fixed pool of pre-collected teacher trajectories … The student occupancy that the ideal objective targets is therefore never sampled". The only staleness studied is on the teacher-prefix side (Table 7). No tier-D statement.
- **Less is More: Early Stopping Rollout for OPD** (2605.27028): "off-policy" = the student prefix is off-distribution for the *teacher* under standard synchronous OPD. §2: "the student prefix y^S_{<t}, which is off-policy to the teacher, may move increasingly far away from the teacher's own high-probability reasoning regions." Standard sync loop, no replay, no ratio.
- **Rethinking Continual Experience Internalization** (2606.04703, RUCBM): changing self-teacher, refreshed every iteration (§3 Eq. 3: "The same policy, when conditioned on E^(k), serves as an experience-aware teacher"); "off-policy" = forward KL on teacher-generated trajectories (Eq. 1). Out on both counts.
- **Scaling SDPO (trajectory.ai field notes)**: self-distillation (teacher = the model itself with feedback), i.e. a changing teacher. Out.
- **Experience replay for LLM RL** (2604.08706), **INTELLECT-3** (2512.16144), **Missing Old Logits in Async Agentic RL** (2605.12070), **Align and Filter: Async On-Policy RL** (2603.01365), **ROLL Flash** (2510.11345), **Staleness-constrained rollout coordination** (2601.12784), **Relax** (2604.11554), **RhymeRL** (2508.18588), HF blog "Keep the tokens flowing", Luke Huang "Is frontier async RL solved": RL-only asynchrony, no distillation stage. Out by scope rule.
- **AdaSwitch** (2510.07842): "off-policy" = fixed dataset/teacher targets, never stale student. §2.2: "Off-policy methods use fixed targets from datasets, while on-policy methods generate y from the student model itself"; the off-policy baseline is SupervisedKD on ground-truth sequences (Appendix baselines). AdaSwitch itself splices *fresh* teacher tokens into the student's generation token-by-token — no reuse, no buffer, no ratio. Out.
- **Unified View of Off-Policy and On-Policy Learning** (2604.07941v2, Zhao et al., Nankai/Huawei): taxonomy survey, no experiments, and the string "KL" never occurs in the paper — so no theory or statement about sampled-token vs distribution-level or forward vs reverse KL under a stale behavior student. Stale student data is only *classified*: §2.2 puts "replayed traces from earlier policies" in the off-policy regime, and the on-policy-teacher-guided-transfer row of Table 7 notes it "Weakens when cached traces dominate fresh correction". Boundary labels, no mechanism, secondary source. Out.
- **Nemotron 3 Super** (2604.12374) and the **Nemotron 3 family white paper** (2512.20856): no on-policy distillation stage at all — neither report contains the string "KL"; teachers appear only as SFT data generators ("using Qwen3-Coder-480B-A35B-Instruct as the teacher model", 2604.12374 §3.1.1). The asynchrony is RL-only: "We use an asynchronous GRPO setup in which training and inference are decoupled across separate GPU devices … we restrict the inference workers to be at most one step behind the latest model version" and "we mask the importance sampling ratio computed from the training and inference logprobs" (§3.2.4); "The result is a single rollout can have tokens and log probabilities from differently aged policies" (§3.2.5). The only self-generated stage is a "short semi-on-policy SFT stage of 350 steps … where we collect roll-outs from the model and truncate 12% of reasoning traces to random reasoning budgets" (§3.1.3) — SFT on own rollouts, no teacher. Out by the RL-only rule.
- **KDFlow** (2603.01875): KD framework paper (decoupled SGLang teacher / FSDP2 student); "off-policy" = static-dataset KD, on-policy mode is per-step; keyword screen finds no staleness/async statement. Out.
- **StableOPD / Demystifying OPD: length inflation** (2604.08527): "off-policy" = golden SFT data mixed into the OPD minibatch (§4.1 "mixture distillation"). Out.
- **vOPD / KL for a KL** (2605.07865): control-variate baseline; "off-policy" only in the exposure-bias sense. Out.
- **Draft-OPD** (2605.29343): "replay" = re-running the drafter from verification-exposed error positions on target-assisted rollouts; prefixes are not stale-student. Out.
- **OEL / Online Experiential Learning** (2603.16856): "off-policy context distillation" = samples from the knowledge-augmented model (changing self-teacher). Out.
- **On the Geometry of OPD** (2606.07082v3): App. G.2 "In the off-policy variant, rollouts are generated by the teacher" — teacher rollouts, not a stale student. Out.
- **SCOPE** (2604.10688): Table 3 "Update Epochs per Batch 1"; the async sentence is about overlapping teacher-logprob acquisition. Out (D: none).
- **Cross-tokenizer OPD** (2606.09456): PPO-style ratio present; mini-batches per batch not stated; "asynchronous" refers to tokenizer chunk alignment. Out.
- **MOPD multi-teacher paper** (2606.30406): "data generated from each rollout is used for a single gradient update and then discarded"; async = teacher prefill overlap. Out.
- **SOD** (2605.07725): batch 64 / mini-batch 16 (4 updates) never discussed; "Rollout is performed asynchronously via vLLM" is the engine. G data point only.
- **REOPOLD** (2603.11137): 4 updates per rollout step (math), 2 (visual, agentic); clipped fraction < 0.2% "the policy does not deviate significantly from the behavior policy". G data point.
- **Rethinking OPD I** (2604.13016), **PowerOPD** (2606.17199): strictly one update per rollout batch; tier-D quotes on the estimator axis only (Rethinking §2.2 taxonomy; PowerOPD fn. 2 on exact expectation vs sampled-token).
- Secondary write-ups not used as sources: Maxime Labonne "Nemotron 3 Ultra: what distillation can't fix" (substack), VESSL blog "How open models actually learn new abilities", AI Weekly on ReOPD, the two awesome-OPD README lists (used for discovery only), OPD survey 2604.00626 (song2026survey; its only async content is the DistiLLM replay-buffer sentence in §2.3 and the Lightning-OPD speedup sentence in §4.1).

## Search log (2026-09-14)

WebSearch: "asynchronous on-policy distillation LLM stale rollouts replay buffer 2026"; "off-policy on-policy distillation reverse KL stale student samples"; "offline on-policy distillation OR asynchronous OPD"; "on-policy distillation off-policy importance sampling staleness teacher logits reuse rollouts technical report"; "near-policy distillation OR freshness-aware OR policy-drift gating"; tech-report queries (MiniMax, Qwen3.5, Kimi K2.5, Ling/Ring, Klear, Nemotron, Step, Hunyuan, Seed, Llama, Olmo 3) — no additional async-OPD tech reports surfaced; framework queries (slime, verl, ROLL, nemo-rl); "on-policy distillation replay OR reuse rollouts OR sample reuse OR multiple epochs".
