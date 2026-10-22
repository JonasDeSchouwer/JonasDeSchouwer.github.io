# Agent B result: Reverse-KL Knowledge Distillation of LLMs — Literature Slice

**Convention used throughout:** reverse KL (RKL) = D_KL(q‖p), q = student, p = teacher; forward KL (FKL) = D_KL(p‖q). Every paper below was checked against this convention; deviations are flagged (spoiler: all core papers agree with it).

---

## 1. MiniLLM — Gu, Dong, Wei, Huang (ICLR 2024; arXiv 2306.08543)

**Title flag (citation hygiene):** v1 (Jun 2023) was titled "Knowledge Distillation of Large Language Models"; the ICLR 2024 paper is "MiniLLM: Knowledge Distillation of Large Language Models"; arXiv v5–v6 (Nov 2025 / Jan 2026) **retitled it "MiniLLM: On-Policy Distillation of Large Language Models"** — which now collides with GKD's title. Cite by arXiv ID to disambiguate.

```bibtex
@inproceedings{gu2024minillm,
  title     = {Mini{LLM}: Knowledge Distillation of Large Language Models},
  author    = {Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2306.08543},
  archivePrefix = {arXiv},
  note      = {arXiv v6 (2026) retitled ``MiniLLM: On-Policy Distillation of Large Language Models''}
}
```
Links: https://arxiv.org/abs/2306.08543 — Code: https://github.com/microsoft/LMOps/tree/main/minillm

**Objective (as stated, Sec. 2.1, Eq. 1):** θ = argmin_θ KL[q_θ‖p] = argmin_θ −E_{x∼p_x, y∼q_θ}[log p(y|x)/q_θ(y|x)]. Sequence-level reverse KL as the **primary objective**, with y sampled from the student → optimized by **policy gradient** (Sec. 2.2). The sequence-level PG estimator is decomposed per timestep; prefixes come from student rollouts (mixed with teacher, below). Estimation: long-horizon part via sampled trajectories; the single-step term is computed exactly by full-vocabulary summation.

**Stated motivation (their convention matches ours):** "Minimizing forward KLD causes q_θ to assign unreasonably high probabilities to the void regions of p and produces very unlikely samples under p during free-run generation" (Sec. 1); reverse KL is mode-seeking: "q_θ assigns high probabilities to p's large modes and ignore the small ones" (Sec. 2.1).

**Three techniques (Sec. 2.2), each fixing an estimator-level pathology:**
- *Single-step decomposition:* splits reward into exact immediate term r_t (full-vocab sum) + future term, reducing PG variance — "error in front tokens accumulates."
- *Teacher-mixed sampling:* rollouts from p̃ = α·p + (1−α)·q_θ (α = 0.2) to prevent "reward hacking when q_θ produces degenerated sentences … that receive high scores from the teacher (e.g., repeated phrases)."
- *Length normalization:* normalizes future reward by remaining length because "long sequences tend to have small R_{t+1}, which encourages the model to produce short responses."

**Empirical effects:** Beats word-level FKL KD and SeqKD on Rouge-L and GPT-4-feedback win comparisons across GPT-2 120M → LLaMA-13B students (Table 1; e.g., 76.4 vs 73.6 GPT-4 feedback for LLaMA-7B on SelfInst). **Exposure bias:** ExAccErr much lower, plateaus at ~150 tokens while baselines grow (Fig. 6, Sec. 3.3). **Calibration:** ECE on SST2 0.099 vs SeqKD 0.243 (Table 2) — notable because RKL is usually accused of *hurting* calibration; here it improves it vs FKL baselines. **Diversity:** distinct-4 ≈ preserved (99.0 vs teacher 99.3, Table 3). Gains concentrate on longer responses (Fig. 7).

**Acknowledged negative/limitation (Sec. 3.3):** "Models optimized by minimizing reverse KLD are likely to lose modes, which affects generation diversity." Their defense: "generating one correct response is sufficient … for scenarios demanding high truthfulness and reliability," plus Table 3 diversity metrics. Weaker on very short (≤5-token) targets.

---

## 2. GKD — Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem (ICLR 2024; arXiv 2306.13649)

```bibtex
@inproceedings{agarwal2024gkd,
  title     = {On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  author    = {Agarwal, Rishabh and Vieillard, Nino and Zhou, Yongchao and Stanczyk, Piotr and Ramos, Sabela and Geist, Matthieu and Bachem, Olivier},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2306.13649},
  archivePrefix = {arXiv}
}
```
Links: https://arxiv.org/abs/2306.13649 — No official code released with the paper; de facto reference implementation is Hugging Face TRL's `GKDTrainer`.

**Objective:** token-level divergence conditioned on prefixes, computed **full-vocabulary per position** (no policy gradient), on a mixture of data sources: with probability λ the prefixes come from student rollouts (on-policy), otherwise from the fixed dataset. Divergence family: generalized JSD, D_JSD(β)(P‖Q) = β·KL(P‖βP+(1−β)Q) + (1−β)·KL(Q‖βP+(1−β)Q) (Sec. 2, Eq. 1); after rescaling, β→0 recovers forward KL and β→1 recovers reverse KL. **Terminology matches ours:** P = teacher, Q = student; "forward KL" = KL(P‖Q). λ ∈ [0,1] "controls the student data fraction, that is, the fraction of on-policy student-generated outputs" (Sec. 3.1).

**Stated motivation:** the mismatch argument — "distilling auto-regressive student models using such approaches results in train-inference distribution mismatch. This is because the partial sequences encountered by the student … at inference can be quite different from the ones seen during training" (Sec. 3.1). On RKL: it "prioritize[s] the tokens where the teacher assigns high probability, which can avoid low-quality generations but at the expense of less diverse generations" (Sec. 3.1); "[reverse KL] is mode-seeking as it forces Q_θ to be zero where P is zero and hence makes it concentrate on one of the modes" (Fig. A.16 caption).

**Empirical picture — no uniform winner:** "optimal divergence seems to be task-dependent" (Sec. 3.1); "choice of divergence is temperature-dependent" (Sec. 4.1); "Mode-seeking divergences perform better when evaluat[ing] via temperature sampling while choice of divergence doesn't affect performance much with greedy sampling" (App. A.3). Headline gains are attributed to **on-policy sampling**, not to a specific divergence: 2.1× (XSum), 1.7× (WMT), 1.9× (GSM8K) relative gains (Fig. 1). Key nuance vs MiniLLM: GKD decouples "sample from student" from "use reverse KL" and finds the former is what matters robustly.

---

## 3a. DistiLLM — Ko, Kim, Chen, Yun (ICML 2024; arXiv 2402.03898)

```bibtex
@inproceedings{ko2024distillm,
  title     = {Disti{LLM}: Towards Streamlined Distillation for Large Language Models},
  author    = {Ko, Jongwoo and Kim, Sungnyun and Chen, Tianyi and Yun, Se-Young},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  eprint    = {2402.03898},
  archivePrefix = {arXiv}
}
```
Links: https://arxiv.org/abs/2402.03898 — Code: https://github.com/jongwooko/distillm

**Objective:** token-level, full-vocab, skewed divergences: **SKL** D^(α)_SKL(p, q_θ) = KL(p ‖ αp + (1−α)q_θ) (skewed forward) and **SRKL** D^(α)_SRKL(p, q_θ) = KL(q_θ ‖ (1−α)p + αq_θ) (skewed reverse), α = 0.1 optimal. Prefixes: fixed dataset, plus student-generated outputs (SGOs) drawn from a **replay buffer** with adaptive probability (adaptive off-policy) — student generation happens only occasionally; adaptive probability φ increases when validation loss degrades.

**Critique of MiniLLM/GKD (estimator-level):** on-policy SGO generation "faces low sample efficiency and high generation time as it constantly prompts the student to produce new training sequences," with SGO generation "reaching up to 80%" of total training time; also "the optimal divergence seems to be task-dependent, which requires additional efforts to inconveniently select a proper loss function."

**Theory:** skewing bounds the gradient ratio (SKL gradient coefficient (1−α)·r_{p,q̃_θ} vs KLD's unbounded r_{p,q_θ}), giving stable gradients; Theorem 1 bounds the empirical estimator error with optimal trade-off at intermediate α.

**Empirical:** SRKL consistently best on instruction-following (Dolly/Self-Instruct/Vicuna/S-NI/UnNI, 25.21–30.11 ROUGE-L); competitive on SAMSum and IWSLT vs GKD; **2.5–4.3× training speedup** vs MiniLLM/GKD (Fig. 7). Negative result: naive off-policy applied to vanilla KLD/RKLD *drops* 1–2 ROUGE-L (Table 4) — replay only works combined with skewing.

## 3b. DistiLLM-2 — Ko, Chen, Kim, Ding, Liang, Zharkov, Yun (ICML 2025 Spotlight; arXiv 2503.07067)

```bibtex
@inproceedings{ko2025distillm2,
  title     = {Disti{LLM}-2: A Contrastive Approach Boosts the Distillation of {LLM}s},
  author    = {Ko, Jongwoo and Chen, Tianyi and Kim, Sungnyun and Ding, Tianyu and Liang, Luming and Zharkov, Ilya and Yun, Se-Young},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  eprint    = {2503.07067},
  archivePrefix = {arXiv}
}
```
Links: https://arxiv.org/abs/2503.07067 — Code: https://github.com/jongwooko/distillm-2

**Objective — exact data/direction pairing (Eq. 2):** L = (1/2|D|) Σ [(1−β)·D^(α_t)_SKL(x, y_t) + β·D^(α_s)_SRKL(x, y_s)] — **skewed forward KL on teacher-generated responses y_t; skewed reverse KL on student-generated responses y_s.** Rationale (Sec. 3.1.2): "using SKL for teacher responses … where most of p(y_t|x) ≫ 0 and using SRKL for student responses, y_s, where … most of p(y_s|x) ≃ 0"; "KL increases q_θ(·|x) in regions where p(·|x) are high (i.e., pulling-up effect) … RKL attempts to reduce the ratio q_θ(·|x)/p(·|x), consequently q_θ(·|x) decreases in region[s] where p(·|x) [is] small (i.e., pushing-down effect)."

**Negative results:** (i) vanilla interpolation γ·SKL + (1−γ)·SRKL on the *same* data type "does not improve performance compared to using either SKL or SRKL alone" (Sec. 3.1.2) — the gain comes from the data-loss pairing, not from mixing divergences; (ii) DPKD-style contrastive ratio maximization "excessively decreases the likelihood of q_θ(y_s|x) (e.g., 91.25 in terms of NLL), causing the student to lose pre-trained information" (Sec. 3.1.1, Fig. 1b).

**Empirical:** SOTA small students across instruction-following/math/code; e.g., Mistral-7B→Danube2-1.8B: AlpacaEval 74.04%, avg 56.45% (Table 2); +3.5–4.5 pts over DistiLLM on instruction-following; extensions to preference alignment and VLMs.

---

## 4. f-DISTILL — Wen, Li, Du, Mou (ACL 2023; arXiv 2307.15190)

```bibtex
@inproceedings{wen2023fdistill,
  title     = {f-Divergence Minimization for Sequence-Level Knowledge Distillation},
  author    = {Wen, Yuqiao and Li, Zichao and Du, Wenyu and Mou, Lili},
  booktitle = {Association for Computational Linguistics (ACL)},
  year      = {2023},
  eprint    = {2307.15190},
  archivePrefix = {arXiv}
}
```
Links: https://arxiv.org/abs/2307.15190 — Code: https://github.com/MANGA-UOFA/fdistill

**Objective:** sequence-level f-divergence, decomposed into tractable step-wise word-level losses (Theorem 1: exact decomposition for KL/RKL/JS; TVD is *upper-bounded* by its step-wise loss). Four variants: KL (samples from teacher p), RKL (samples from student q_θ), JS and TVD (samples from both). **Terminology matches ours:** J_RKL = D_KL(q_θ‖p) (Eq. 7). Pre-LLM-scale students (T5/BART-class seq2seq).

**Empirical:** "Symmetric distilling losses (JS and TVD) are consistently better than asymmetric ones (KL and RKL) across all datasets except for WMT16 EN-RO" — where they explain MT is "analogous to learning a uni-modal distribution, where mode averaging does not occur." Winners: DART → TVD; XSum → TVD (RKL second); WMT En-Ro → JS (KL second); Commonsense Dialogue → JS.

**On RKL specifically:** "RKL distillation has the opposite problem, known as mode collapsing, where the student only learns one or a few modes of the teacher distribution" (Sec. 2.2). Evidence: RKL has consistently **higher coverage risk** than KL (DART: 0.59 vs 0.49, Table 3) — it undercovers the teacher; but on the strongly one-to-many Commonsense Dialogue it achieves much better likelihood risk (0.29 vs KL 0.43). Case study (App. C): "KL and RKL distillations fail to yield coherent responses … KL distillation yields short and generic utterances."

---

## 5. Rethinking KL in KD (AKL) — Wu, Tao, Wang, Yang, Zhao, Wong (COLING 2025; arXiv 2404.02657) — key criticism-of-folklore source

```bibtex
@inproceedings{wu2025rethinking,
  title     = {Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models},
  author    = {Wu, Taiqiang and Tao, Chaofan and Wang, Jiahao and Yang, Runming and Zhao, Zhe and Wong, Ngai},
  booktitle = {International Conference on Computational Linguistics (COLING)},
  year      = {2025},
  eprint    = {2404.02657},
  archivePrefix = {arXiv}
}
```
Links: https://arxiv.org/abs/2404.02657 — Code: https://github.com/wutaiqiang/LLM_KD_AKL. **Terminology matches ours** (FKL = KL(p‖q_θ), RKL = KL(q_θ‖p)).

**Central argument:** "Contrary to prior assertions that reverse Kullback-Leibler (RKL) divergence is mode-seeking and thus preferable over the mean-seeking forward Kullback-Leibler (FKL) divergence, this study empirically and theoretically demonstrates that neither mode-seeking nor mean-seeking properties manifest in KD for LLMs." The classical picture assumes "(1) the student distribution q_θ follows the Gaussian distribution, and (2) both teacher p and student q_θ are continuous" — both fail for softmax-over-vocabulary distributions where the student can represent any categorical distribution. Shared optimum: "For both FKL and RKL, the sufficient and necessary condition [to] converge is q_θ(Y_j|y_<t) = p(Y_j|y_<t) ∀j" (Eqs. 5–8) and "RKL and FKL share the same optimization objective and both converge after a sufficient number of epochs."

**Evidence:** toy discrete distributions, identical inits, 200 epochs → both FKL and RKL converge to the target regardless of shape (Fig. 2). The real difference is transient optimization behavior: "FKL focuses on the head part and RKL focuses on the tail part at the beginning epochs" (Fig. 3). They explicitly target the folklore chain: "Recent methods (Gu et al.; Agarwal et al.; Kim et al.) argue that RKL is more suitable than FKL … since RKL avoids learning too many long-tail variants" — and argue this reasoning is flawed in the discrete setting. (MiniLLM excluded from empirical comparison "since it pre-trains on extra corpus.")

**Proposal — AKL (Eq. 15):** token-adaptive convex combination, weights proportional to head-gap vs tail-gap: AKL = [g_head/(g_head+g_tail)]·FKL + [g_tail/(g_head+g_tail)]·RKL, head = smallest token set accumulating ≥ μ = 0.5 teacher mass, gaps ϵ = |p(z) − q_θ(z)|. Fixed-dataset prefixes, full-vocab, token-level.

**Empirical:** GPT-2-1.5B→120M and LLaMA-6.7B→TinyLLaMA-1.1B on Dolly/S-NI/UnNI: AKL beats SFT, SeqKD, FKL, RKL, SKL, SRKL, and equal-weight FKL+RKL (e.g., UnNI 31.97 vs 30.48, p < 0.001). GPT-4 eval (Fig. 5): AKL improves **both diversity and quality** — direct counter-evidence to "you must trade diversity for quality when picking a direction."

**Caveat for the review:** their theory concerns *representable* optima; it does not address capacity-limited students where q cannot match p — exactly the regime the mode-seeking argument invokes. Empirical scale (120M–1.1B students, ≤10k-example instruction data) is modest.

---

## 6. Thinking Machines Lab — "On-Policy Distillation" (blog, Oct 27, 2025)

```bibtex
@misc{lu2025onpolicydistillation,
  title        = {On-Policy Distillation},
  author       = {Lu, Kevin and {Thinking Machines Lab}},
  howpublished = {Thinking Machines Lab: Connectionism (blog)},
  year         = {2025},
  month        = oct,
  url          = {https://thinkingmachines.ai/blog/on-policy-distillation/}
}
```

**Objective:** per-token reverse KL on student-sampled trajectories: KL(π_θ‖π_teacher) = E_{x∼π_θ}[log π_θ(x_{t+1}|x_{1..t}) − log π_teacher(x_{t+1}|x_{1..t})] — "We sample trajectories from the *student* model and use a high-performing teacher to grade *each token*." Note the estimator: sampled log-ratio at chosen tokens (RL-style dense per-token reward), not a full-vocab divergence per position. Conceptually GKD's on-policy setting (λ = 1) with pure reverse KL; cites Agarwal et al. 2023 and Gu et al. (MiniLLM), plus DAgger (Ross et al. 2010) and the Qwen3 technical report. **Terminology matches ours.**

**Motivation:** dense reward vs sparse RL — "RL provides very sparse feedback"; "Dense supervision greatly improves compute efficiency"; RKL is "mode seeking": "it learns one specific behavior (the teacher's) instead of spreading its distribution across several suboptimal options"; and "unhackable" — low reverse KL always corresponds to teacher-desirable behavior; also reduces exposure bias.

**Results:** Qwen3-8B-Base student, Qwen3-32B teacher, AIME'24: off-policy distillation 55.0%; +RL 67.6% (17,920 GPU-hours); +on-policy distillation **74.4% (1,800 GPU-hours)**; "distillation reaches the teacher's level of performance approximately 7-10x faster than RL," ~9–30× total-cost reduction depending on accounting. **Continual-learning/restoration experiment:** Qwen3-8B mid-trained on internal docs (Internal QA 18→36%, IF-eval 85→79%); on-policy distillation *from the earlier version of the same model* on Tulu3 prompts restores IF-eval to 83% while pushing Internal QA to 41% — "on-policy distillation recovers nearly full performance on IF-eval without losing any knowledge."

---

## 7. Other verified papers

**Speculative KD** — Xu, Han, Wang, Le, Madeka, Li, Wang, Agarwal, Lee, Pfister (ICLR 2025; arXiv 2410.11325). Code: https://github.com/google-research/google-research/tree/master/speculative_kd
```bibtex
@inproceedings{xu2025skd, title={Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling}, author={Xu, Wenda and Han, Rujun and Wang, Zifeng and Le, Long T. and Madeka, Dhruv and Li, Lei and Wang, William Yang and Agarwal, Rishabh and Lee, Chen-Yu and Pfister, Tomas}, booktitle={ICLR}, year={2025}, eprint={2410.11325}, archivePrefix={arXiv}}
```
Divergence-direction relevance is *indirect*: objective written teacher-first, "minimize D(M_t‖M_s)(y|x)" (Sec. 3), i.e., forward-direction; contribution is the **data distribution**: student proposes tokens, teacher resamples any token outside its top-K ("if y_i ⊄ topK(M_t(·|y_<i, x_j)) then y_i ∼ M_t"). Critique of pure on-policy KD (important estimator-level negative result): "the student may generate low-quality samples that are out-of-distribution (OOD) for the teacher, especially early in training, leading to inaccurate teacher feedback" (Sec. 1) — with dramatic numbers (translation COMET: on-policy KD 36.1 vs SKD 75.3 from an instruction-tuned Gemma-2B init; worse with weaker Qwen-0.5B students).

**ImitKD (precursor)** — Lin, Wohlwend, Chen, Lei (EMNLP 2020; arXiv 2009.07253). Code: https://github.com/asappresearch/imitkd
```bibtex
@inproceedings{lin2020imitkd, title={Autoregressive Knowledge Distillation through Imitation Learning}, author={Lin, Alexander and Wohlwend, Jeremy and Chen, Howard and Lei, Tao}, booktitle={EMNLP}, year={2020}, eprint={2009.07253}, archivePrefix={arXiv}}
```
DAgger-style: sample sequences from a mixture of the dataset and student generations (annealed), then per-token cross-entropy against the teacher's full distribution: ℓ = −Σ_v π*(v|y_<t, x)·log π(v|y_<t, x) — i.e., **forward KL on student-influenced prefixes**. No discussion of reverse KL. Establishes that "student-sampled prefixes fix exposure bias" predates and is orthogonal to the reverse-KL choice.

**ABKD** — Wang, Yang, Wang, Wang, Xu, Huang (ICML 2025 Spotlight; arXiv 2505.04560). Code: https://github.com/ghwang-s/abkd
```bibtex
@inproceedings{wang2025abkd, title={ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via $\alpha$-$\beta$-Divergence}, author={Wang, Guanghui and Yang, Zhiyong and Wang, Zitai and Wang, Shi and Xu, Qianqian and Huang, Qingming}, booktitle={ICML}, year={2025}, eprint={2505.04560}, archivePrefix={arXiv}}
```
Frames FKL vs RKL as two entangled effects — "the Hardness-Concentration effect … focusing on modes with large errors, and the Confidence-Concentration effect … focusing on modes with high student confidence" — FKL exhibits both too weakly, RKL too strongly; α-β-divergence "offers a smooth interpolation between FKLD and RKLD." Another "both extremes are wrong, interpolate" result.

**Pointers (verified metadata only):**
- Survey: Xu, Li, Tao et al., "A Survey on Knowledge Distillation of Large Language Models," arXiv 2402.13116 (v4 Oct 2024) — pointer only.
- BiLD — Li, Zhou, Song (COLING 2025; arXiv 2406.13555): top-k logit-difference KL loss; direction unspecified in abstract; weak relevance.
- RKLU — Wang, Zi, Sun, Zhao, Qin (NAACL 2025, 2025.naacl-long.60): reverse-KL-based distillation for **unlearning** — RKL's pushing-down behavior repurposed to remove target content while preserving utility.
- BabyLlama-RKL — Shi, Matusevych, Nissim, arXiv 2410.22081 (BabyLM Challenge 2): RKL distillation in low-data pre-training; "under reverse KL divergence, a single-teacher model often outperforms or matches multiple-teacher models across most tasks."
- DRKL — Luong, Tran, Chen, "Diversity-Aware Reverse KL Divergence for LLM Distillation," arXiv 2604.00223 (2026): gradient analysis showing RKL's non-target gradients "consistently push the target logit upward even when the student already matches the teacher" → overconfidence/low diversity; decouples target/non-target terms; beats RKL and FKL on OPT-6.7B→1.3B (avg ROUGE-L 23.11 vs 21.07/20.91) while improving distinct-2 and self-BLEU.
- On-policy-distillation survey — Song, Zheng et al., arXiv 2604.00626 (2026): formalizes OPD as "f-divergence minimization over student-sampled trajectories," connects to KL-constrained RL.
- ICLR 2025 blogpost "On LLM Knowledge Distillation — A Comparison between Forward KL and Reverse KL" (https://d2jud02ci9yv69.cloudfront.net/2025-04-28-llm-knowledge-distil-157/blog/llm-knowledge-distil/, also on OpenReview): Qwen2.5 1.5B/7B on UltraChat; finds "there's not much difference between forward KL and reverse KL" at token level with SFT loss, FKL converged faster — independent secondary support for Wu et al.

---

# Direct answers

**(a) Where people actually choose reverse KL, and why.** Four recurring situations. (1) *Capacity-gap generative KD* (MiniLLM): student can't cover the whole teacher distribution, so let it concentrate on teacher modes rather than "assign unreasonably high probabilities to the void regions of p"; precision/truthfulness valued over coverage. (2) *On student-generated data specifically*: when prefixes/sequences come from the student, reverse-KL-shaped losses are the natural choice — DistiLLM-2 makes this explicit (SRKL on student responses where p ≃ 0, SKL on teacher responses where p ≫ 0), and TML's on-policy distillation is pure RKL on student rollouts. (3) *As a dense, unhackable RL substitute* (TML): per-token reverse KL to a teacher is a process-level reward that is cheap (7–10× fewer steps than RL, 9–30× cost) and can't be reward-hacked. (4) *Selective suppression*: unlearning (RKLU) and behavior restoration/continual learning (TML), exploiting RKL's pushing-down effect. Note the sequence-level RKL objective is also exactly reward-maximization with reward log p(y|x) plus student-entropy regularization, which is why the RL toolchain (policy gradient in MiniLLM, dense reward in TML) attaches to it.

**(b) Empirical picture of RKL vs FKL vs mixtures.** No credible source shows reverse KL uniformly winning. MiniLLM shows RKL+PG beating FKL word-level KD and SeqKD on instruction-following with better calibration, lower exposure bias, roughly preserved distinct-n diversity — but gains are entangled with on-policy sampling and three estimator fixes. GKD cleanly separates the axes: the on-policy *data* axis robustly helps while the divergence is task- and temperature-dependent (mode-seeking helps under temperature sampling; irrelevant under greedy). Wen et al.: symmetric divergences (JS/TVD) beat both KL and RKL except on near-unimodal MT; RKL has a coverage deficit. DistiLLM: *skewed* RKL best on instruction-following, but attributes vanilla RKL/FKL problems to gradient/estimator instability; sequel shows mixtures only help when direction is matched to data source. Wu et al. and ABKD: the two directions are transient-optimization variants of the same objective (head vs tail focus); adaptive/interpolated combinations beat both, improving quality *and* diversity. Strongest headline numbers for pure RKL: TML's on-policy results (AIME'24 74.4% at 1/10 the GPU-hours of RL) — but the comparison is vs RL and off-policy KD, not vs on-policy forward KL. Net: current consensus ≈ "sample from the student; the divergence direction matters less than folklore claims; best practice is skewed/adaptive/data-matched combinations."

**(c) Criticisms of reverse-KL distillation.** *Divergence-level:* (i) mode collapse / coverage loss — admitted by MiniLLM itself ("likely to lose modes"), quantified by Wen et al. (higher coverage risk) and GKD ("at the expense of less diverse generations"); (ii) the mode-seeking justification may be vacuous for LLMs — Wu et al.: Gaussian/continuous assumptions fail for softmax distributions, FKL and RKL share the same optimum, differences are early-training head-vs-tail effects (echoed by ICLR 2025 blogpost); (iii) over-concentration/overconfidence — ABKD and DRKL (non-target gradients push the target logit up even at the optimum). *Estimator-level:* (i) MiniLLM's PG estimator is expensive, sample-inefficient, unstable — DistiLLM: SGO generation up to 80% of training time; 2.5–4.3× speedup available without PG; (ii) naive sequence-level RKL has degenerate optima requiring patches — MiniLLM's teacher-mixed sampling (anti reward-hacking on repetitions) and length normalization (anti short-response bias) are evidence; (iii) student-sampled data can be garbage early in training — SKD: OOD student samples get "inaccurate teacher feedback" (on-policy KD 36.1 COMET vs 75.3 interleaved); (iv) off-policy reuse of student samples breaks vanilla RKL (DistiLLM Table 4) unless skewed. Clean structure: the *data question* (student vs teacher vs dataset prefixes) is settled in favor of student involvement; the *direction question* is contested; the *optimization question* (PG vs token-level full-vocab vs sampled log-ratio) is where most practical failures live.

**Terminology flags:** All core sources (MiniLLM, GKD, DistiLLM/-2, Wen, Wu, TML, DRKL) use reverse KL = KL(student‖teacher), matching our convention — no swaps found. Two notation cautions: GKD's JSD(β) recovers FKL/RKL only in *rescaled* limits (JSD(β)/β as β→0, symmetrically as β→1), and JSD(β) is not symmetric in (P,Q) for β ≠ 0.5; MiniLLM's 2026 arXiv retitle collides with GKD's title.

# Leads not verified

- ImitKD venue: EMNLP 2020 confirmed via paper text; main vs Findings not checked in ACL Anthology.
- DPKD ("Direct Preference Knowledge Distillation", critiqued in DistiLLM-2) — not fetched.
- "Kim et al. 2024" cited by Wu et al. as a third RKL-advocating method — resolved: PromptKD (Kim, Jang, Yang; EMNLP 2024 Findings, arXiv 2402.12842). Motivates RKL with one sentence citing only f-GAN ("as it exhibits mode-seeking behavior (Nowozin et al. 2016) and benefits generation tasks"); student-generated pseudo-targets via GKD's exposure-bias insight; no folklore figure; its Appendix D/Table 12 ablation on the two prompt-tuning losses finds "barely any significant difference" between directions (student-loss direction not ablated).
- BiLD's internal KL direction(s) — full text not fetched.
- Qwen3 Technical Report Table 21 — TML's source for off-policy/RL/on-policy comparison; not independently fetched.
- BOND (Sessa et al. 2024) — Jeffreys-divergence best-of-n distillation; not fetched (covered by RL agent).
- DRKL code URL and venue — not in fetched excerpt; "AB divergence (Wang et al., 2025)" baseline mapped to ABKD but not confirmed.
- Cross-tokenizer KD (ULD etc.) — skipped as not divergence-direction-relevant, not verified by fetch.
- On-policy distillation survey (2604.00626) body — only abstract fetched.
- MiniLLM quotes come from current (v6) arXiv HTML; wording may differ slightly in ICLR 2024 camera-ready.
