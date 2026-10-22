# Fu et al. 2026 — Rethinking On-Policy Distillation of Large Language Models II: One Training Example

**arXiv 2609.04172v1** (3 Sep 2026). Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan-ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, Chaojun Xiao (Tsinghua / THUNLP; Part II of Rethinking-OPD @li2026rethinking). Sources: PDF text (pypdf, 2026-09-09), repo `github.com/Thinking-Space/One-Shot-OPD` @ `6b6b1782` (2026-09-04, cloned 2026-09-09), HF model cards (curl, 2026-09-09). Incorporated: paper.typ Appendix "Reproducing Tinker Recipe" table, column F.

```bibtex
@article{fu2026oneshot,
  title   = {Rethinking On-Policy Distillation of Large Language Models II: One Training Example},
  author  = {Zixuan Fu and Bingxiang He and Yuxin Zuo and others},
  journal = {arXiv preprint arXiv:2609.04172},
  year    = {2026}
}
```

## Claim

One-shot OPD (a single training query, repeated 64× per batch) "keeps improving for hundreds of steps and recovers most of full-data OPD's gain across task domains and model families" — 72% of full-data OPD's gain at step 1000 on math (Fig. 1). Mechanism: state coverage (one query reaches 71.5% of the K=200 state clusters full-data OPD visits; 16 semantically distinct queries reach 98.9% and match full-data) vs. a decaying "absorption rate" that is the same for 1 or 17k queries ("data-overfed but algorithm-starved").

## Student–teacher pairs (Table 1, §3.1)

| Domain | Student | Teacher |
|---|---|---|
| Math | DeepSeek-R1-Distill-Qwen-1.5B (as released) | JustRL-DeepSeek-1.5B (`hbx/JustRL-DeepSeek-1.5B`) |
| Math (family check) | Llama-3.2-3B-Instruct | GT-Llama-3.2-3B-Instruct-MATH |
| Math (family check) | OLMo-3-7B-Instruct-DPO | OLMo-3-7B-Instruct |
| Code | R1-Distill-1.5B | Nemotron-Research-Reasoning-Qwen-1.5B-v2-RLVE (launcher default `DeepCoder-1.5B-Preview` — differs from paper) |
| IF | R1-Distill-1.5B | UltraData-IF-1.5B (author-trained GRPO from the same student, step 600; the only released HF artefact) |
| Agentic | Qwen2.5-Coder-1.5B-Instruct | Hammer2.1-1.5b |

§3.1: "For each domain, we pair a student with a post-trained teacher from the same family." No SFT stage anywhere.

**Teacher provenance (JustRL model card, `hbx/JustRL-DeepSeek-1.5B`, curl 2026-09-09):** `base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; "Standard GRPO with binary outcome rewards … DAPO-Math-17k without filtering or dynamic sampling … 16K context cap … Only 'clip higher'"; 4,000+ steps. Card table (own harness, POLARIS-based + CompassVerifier-3B): AIME24 @32: R1-Distill-1.5B 29.90 → JustRL 52.60; AIME25 @32: 22.40 → 38.75; AMC23 @32: 63.82 → 91.02; MATH-500 @4: 84.90 → 91.65. I.e. the teacher is the *same checkpoint after RL on the same prompt set the OPD stage uses*.

## Math data (§3.1, App. A.1, `data/prep/`)

- Training set: DAPO-Math-17K, deduplicated to 17,917 unique prompts (`prepare_dapo_math500.py`: "each problem ships 100 times; keep the first"); the DAPO "Solve the following math problem step by step … Answer: $Answer" prefix/suffix is **stripped** — the row is the bare problem as a single user turn (`prompt: [{"content": problem, "role": "user"}]`), no system prompt.
- One-shot corpus: `math/DAPO-oneshot/t22.parquet` = DAPO row 22 repeated 64 times (`prepare_dapo_oneshot.py`, `--repeat 64` = train_batch_size; answer "37"). Selection (App. A.1): 8 rollouts of the untrained student on the first 64 DAPO queries; t59 (8/8 easy), t22 (4/8 medium, the default), t56 (0/8 hard). "yet all three produce comparable benchmark gains in Figure 6."
- 4-/16-shot: BGE-M3 clustering of DAPO-17K, one representative per cluster, seeded from t22.
- Validation set: MATH-500 test (raw `problem` field, same schema), plus AMC'23 / AIME'25 via `prepare_amc23_aime25.py`.
- Chat template: verl `RLHFDataset` applies the model's `apply_chat_template(..., add_generation_prompt=True)`; the R1-Distill tokenizer's template ends the prompt with `<｜Assistant｜><think>\n` (tokenizer_config.json, curl 2026-09-09). `data.shuffle=True` for `full` and `oneshot` modes (`mode_args`).

## Objective (§2.2 Eq. 2–3, code)

Paper: full-distribution OPD Eq. 1 = Σ_i KL(π_θ(·|s_i) ‖ π_T(·|s_i)); sampled-token estimator Eq. 2 A_i = log π_T(y_i|s) − log π_θ(y_i|s); top-k estimator Eq. 3 A_i^{top-k} = Σ_{v∈V_i} π̃_θ(v|s)[log π_T(v|s) − log π_θ(v|s)], V_i = TopK(π_θ, k), π̃_θ renormalized over V_i. "The mathematical-reasoning runs of this section optimize the top-k advantage of Eq. 3 with k = 16; the code, instruction-following, and agentic runs optimize the sampled-token advantage of Eq. 2." Footnote 1: the §4–5 analysis runs use Eq. 2 "so their absolute accuracies differ slightly".

Code (`verl/verl/workers/fsdp_teacher_worker.py` @ 6b6b178):

```python
# compute_top_k_reward (lines 159-195); strategy only_stu, weight_mode student_p
kl = student_log_probs - teacher_log_probs           # (…, K) raw log-probs of the student's top-K ids
weights = compute_reward_weights(student_log_probs, teacher_log_probs, valid, weight_mode, normalize)
#   student_p + normalize=True -> exp(lp - logsumexp(lp)) = student probs renormalized over the K ids
return -kl * weights
# lines 418-420: sampled-token fallback (LOG_PROB_TOP_K=0)
rm_scores = -(student_log_probs - teacher_log_probs)
```

`verl/verl/trainer/ppo/opd_core_algos.py`: `token_reward_direct` = "the advantage is simply the reward itself (masked) — there is no baseline to subtract and no discounting to apply"; `vanilla_topk` = PPO clipped objective per (position, candidate), summed over K, `agg_loss(..., loss_agg_mode)`; the per-candidate log-probs come from the student's own top-K forward (`losses.py:107 old_log_prob = data["student_top_k_log_probs"]`). With mini batch = rollout batch and `ppo_epochs: 1` the ratio is ≡ 1 → score-function gradient of the student-top-16 truncated reverse KL. `kl_estimator: k1`, `teacher_temperature: 1.0`.

## Training defaults (Table 3 = README = `recipe/_common.sh`)

| Item | Value | Source |
|---|---|---|
| Rollout batch / mini batch | 64 / 64 (`MINI_BATCH_SIZE`, `data.train_batch_size`) | paper+code |
| Responses per prompt | 1 (`N_RESPONSES`) | paper+code |
| KL coef / entropy coef | 0.0 (`USE_KL=False`) / 0 (verl default) | paper / code |
| LogProb top-K | 0 (sampled) default; 16 for the §3 math runs; strategy `only_stu`, weights `student_p` | paper+code |
| Loss aggregation | `token-mean` | paper+code |
| Rollout temperature / top-p / top-k | 1.0 / 1.0 / −1 (verl `rollout.yaml` defaults) | paper / code |
| Optimizer | AdamW β=(0.9, 0.999), wd 0.01; grad clip 1.0; LR 1e-6 constant, no warmup (`optim/fsdp.yaml` defaults + `lr=1e-6` in `common_args`) | paper+code |
| Max prompt length | 1024 (math) per paper; **launcher default 4096, 8192 in `oneshot` mode** — `opd.sh`'s math branch sets none (README's "the math launcher sets 1024 itself" is not in the code) | discrepancy |
| Max response length | 7168 (paper Table 3, `MAX_RESP_LENGTH`); README table says 7680 | discrepancy |
| Epochs / steps | `TOTAL_EPOCHS=1`; steps set by `trainer.total_training_steps` (README example); 300 steps in Figs 3–8, 1000 in Figs 1/14; 17,917/64 = 280 steps per epoch (inf.) | code / paper |
| Framework | vendored verl 0.9.0 (FSDP actor, param+optimizer offload, dynamic bsz; vLLM 0.28 rollout, `enforce_eager`; frozen FSDP teacher, bf16, `param_offload`); Python 3.12, torch 2.13, transformers 5.10.4 | README |
| Hardware | one 8-GPU H100/A100-80GB node, teacher shares the actor pool | README |
| Data ordering | `data.shuffle=True` | code |
| Save / test freq | 50 / 20 steps | code |

## Evaluation (§3.1, `_common.sh`)

- Math: MATH-500, AMC 2023, AIME 2025, avg@16, response cap 31,744. **AIME 2024 is not evaluated.** "mathematics figures report validation accuracy macro-averaged over MATH-500, AMC 2023, and AIME 2025".
- In-training verl validation loop: `VAL_N=16`, `VAL_TEMPERATURE=0.7`, `VAL_TOP_P=0.9`, `MAX_VAL_RESP_LENGTH=31744`, `do_sample=True` — comment: "avg@N, not greedy. Every reference number was produced under these settings." README: "The in-training validation loop gives avg@16 at temperature 0.7."
- Grading: `reward.custom_reward_function = opd_mixed_verify.reward_func` → math fallback `ttrl_mathverify.reward_func`: strict boxed grader (`ttrl_math.compute_score`) first; if `format_score != 1.0`, `math_verify.compute_score` on the last 2,000 characters.
- Other domains: LCB v6 avg@3 (cap 65,536), Multi-IF turn-3 over 8 languages (16,384/turn, outside verl), BFCL v3 avg@8 (cap 4,096).

## Results (math, R1-Distill-1.5B ← JustRL-1.5B, top-16 objective)

Read off Fig. 3 (`notes/figs/oneshot-opd-fig3-math.png`, 2026-09-09), values at step 0 → 300:

| Benchmark (avg@16) | Student | Full-data OPD | One-shot OPD | Teacher |
|---|---|---|---|---|
| MATH-500 | 0.840 | 0.891 (peak 0.898 @200) | 0.883 | 0.916 |
| AIME 2025 | 0.231 | 0.343 (peak 0.386 @250) | 0.337 | 0.386 |
| AMC 2023 | 0.706 | 0.861 | 0.835 | 0.882 |
| macro-avg (text §3.2) | 59.1 | 69.8 | 68.5 | 72.7 |

Fig. 1 / §3.2 (1000 steps, `notes/figs/oneshot-opd-fig1.png`): full-data 72.1, one-shot 68.4 ("72% of full-data OPD's gain"), teacher 72.7. Sampled-token runs (§4.1, footnote 1): one-shot 59.1 → 66.9, full-data 70.8 at step 300. Dynamics (Fig. 3): top-16 overlap ratio 0.82 → 0.95, overlap advantage −0.009 → ≈0, |entropy gap| → ≈0 for both runs. Fig. 4 (MATH-500+AMC avg, one-shot, final ckpt): R1-Distill-1.5B 77.1 → 85.5; Llama-3B-It 28.2 → 40.2; OLMo-7B-It-DPO 70.8 → 82.4. Fig. 6: gain insensitive to query difficulty, response cap {3k, 5k, 7k}, temperature {1.0, 0.6, 0.2}. §7.2: one-shot RLVR (GRPO, 8 rollouts × 64 queries) gains less than half of OPD's over 1000 steps. §7.1: `<|begin_of_sentence|><|User|><think>\n` as the *entire* training input, or WildChat prompts, "track the real-query baseline" (59.1 → 69.8) on a third to a half of the rollout tokens; a `<think>\n</think>\n` scaffold collapses to short replies.

## Released artefacts

HF collection `Thinking-Space/one-shot-opd` (WebFetch 2026-09-09): only `Thinking-Space/UltraData-IF-1.5B` (the IF teacher) + the paper. No OPD-trained students, no math teacher (JustRL is `hbx/`). Data must be rebuilt with `data/prep/` (ModelScope mirrors).

## Relevance to our runs

The recipe's load-bearing choice is the teacher: an RL continuation of the *identical* student checkpoint on the *same* prompt set (top-16 overlap already 0.82 at step 0). Part I @li2026rethinking reports exactly the pairings we run as the failing ones: R6 R1-Distill-1.5B ← R1-Distill-7B "no improvement despite stronger teacher", R11 R1-Distill-7B ← R1-Distill-14B "little improvement" (see `repro-settings-dossier.md`). Beyond the teacher, the recipe is small-batch (64 rollouts/update), LR 1e-6 with AdamW wd 0.01 + clip 1.0, student-top-16 reward instead of the sampled token, 7,168-token rollouts, and it reports MATH-500/AMC/AIME'25 at temp 0.7 — never AIME'24.
