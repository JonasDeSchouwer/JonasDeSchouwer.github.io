# Evidence inventory: reverse-KL objectives in LM training, fine-tuning, and RL

**Convention.** Reverse KL (RKL) = D_KL(q‖p), where q is the model being trained (student/policy) and p is the target (teacher/reference). Forward KL (FKL) = D_KL(p‖q). Sources deviating in terminology or letter assignment are flagged inline. Full quotes, section numbers, and BibTeX for every entry live in `notes/A–F` (raw research dossiers); this file is the curated index plus the draft comparison tables.

**Verification status.** Quotes marked verbatim were checked against primary sources (fetched PDFs/HTML or cloned code at pinned commits) during collection. Items still needing a re-check before final drafting are listed in §8.

---

## 1. Theory & folklore (→ notes/A)

Canonical claims: FKL is zero-avoiding / inclusive / mass-covering / "mean-seeking"; RKL is zero-forcing / exclusive / "mode-seeking", underestimates support and variance. All core arguments are **theoretical**, resting on two rarely-stated assumptions: (i) q is capacity-limited relative to p (unimodal q on multimodal p); (ii) you obtain the global minimizer, not what SGD finds.

| Source | Key | What it contributes |
|---|---|---|
| Minka 2005, MSR-TR-2005-173 | `minka2005divergence` | α-divergence family (α→0 = RKL, α→1 = FKL); zero-forcing/inclusive-exclusive terminology; nuance: RKL optimum tracks the mode "with the largest mass (not the mode which is highest)"; Fig. 1 (Gaussian fit to 2-Gaussian mixture across α) |
| Bishop PRML 2006 §10.1.2 | `bishop2006prml` | **Fig 10.3** (the ur-figure: (a) FKL covers; (b),(c) RKL — two distinct local minima) and Fig 10.2. **FLAG: Bishop calls KL(p‖q) "the reverse KL"** — opposite of our convention (his default is the VI direction) |
| Murphy PML1 2022 §6.2.6 / PML2 2023 §5.1.4 | `murphy2022pml1`, `murphy2023pml2` | Modern statement (matches our convention); M-projection vs I-projection; moment-matching proof; PML2 carries the caveat "minimizing the reverse KL does not always result in an overly compact approximation [Tur+08]" |
| MacKay 2003 ITILA Ch. 33 | `mackay2003itila` | Why VI uses RKL: tractability (eq. 33.9; FKL objective G "usually intractable"); Ex. 33.5 compactness "assertion"; Ex. 33.7 three-RKL-minima example |
| Blei et al. 2017, arXiv 1601.00670 | `blei2017variational` | VI review; "variational inference generally underestimates the variance of the posterior … a consequence of its objective function" |
| Turner & Sahani 2011 | `turner2011two` | Literally calls compactness a "folk-theorem"; cites Turner–Berkes–Sahani–MacKay 2008 "Counter-examples to variational free-energy compactness folk theorems" (UCL TR; not locatable online) |
| Huszár 2015, arXiv 1511.05101 | `huszar2015how` | Generative-model folklore: MLE/FKL "overgeneralise[s]"; RKL "very conservatively try to avoid any behaviour that is unlikely under P"; JS_π interpolation; GAN ≈ π=0.5 |
| Theis et al. 2016, arXiv 1511.01844 | `theis2016note` | Likelihood ⊥ sample quality; Fig. 1 (KLD vs MMD vs JSD Gaussian fits) |
| Goodfellow 2017, arXiv 1701.00160 §3.2.5 | `goodfellow2016nips` | States the GAN folklore then **refutes it**: "The reverse KL prefers to generate from as many modes … as the model is able to; it does not prefer fewer modes in general. This suggests that mode collapse is driven by a factor other than the choice of divergence." Figs 14–15 |
| Arjovsky & Bottou 2017, arXiv 1701.04862 | `arjovsky2017towards` | Thm 2.5: −log D GAN loss ≈ RKL-dominated; RKL "assigns an extremely high cost to generating fake looking samples, and an extremely low cost on mode dropping" |
| Nowozin et al. 2016 (f-GAN), arXiv 1606.00709 | `nowozin2016fgan` | Divergence choice matters "when the generative model is misspecified"; Table 3 Gaussian-on-GMM fits across 7 divergences |
| Li & Farnia 2023 (AISTATS) | `li2023modeseeking` | Formal mode-seeking definition under which JS qualifies and "standard KL" does not — labels must be re-derived per divergence |
| Soletskyi et al. 2024, arXiv 2410.13300 | `soletskyi2024theoretical` | RKL mode collapse happens even with **well-specified** families: "a consequence of the optimization procedure", not expressivity |
| Chan et al. 2022 (JMLR) | `chan2022greedification` | RL greedification: RKL stronger improvement guarantees; FKL more exploration; differences often modest (quotes unverified) |
| Schulman 2020 blog | `schulman2020klapprox` | k1 = −log r, k2 = ½(log r)², k3 = (r−1)−log r, x∼q, r = p/q; explicitly for **diagnostics**, not gradients |
| Blogs: Ghosh; Kristiadi 2016; E. Jang 2019; A. Jones; janus 2022 (LessWrong) | — | The folklore canon; janus's "Mysteries of mode collapse" carries an in-post correction (text-davinci-002 was **not** RLHF-trained) |
| Wu et al. 2024/25 (AKL), arXiv 2404.02657 | `wu2025rethinking` | Strongest LLM-specific pushback: "neither mode-seeking nor mean-seeking properties manifest in KD for LLMs"; FKL/RKL share the optimum for softmax students; differences are transient (head vs tail early in training) |

**Standard figures used to explain the folklore:** Bishop Fig 10.2/10.3; Minka Fig 1 & 3; Murphy PML1 Fig 6.3, PML2 Figs 5.1–5.2 (public notebooks exist); Goodfellow Figs 14–15; Theis Fig 1; Huszár Fig 1; MiniLLM Fig 2 (toy GMM for LLM KD); GKD Fig A.16; Ari Seff's animation.

---

## 2. Reverse KL in LLM distillation — papers (→ notes/B)

| Source | Objective (as stated) | Key findings / quotes |
|---|---|---|
| **MiniLLM** Gu et al., ICLR 2024, arXiv 2306.08543 | Sequence-level RKL as primary objective, policy gradient; teacher-mixed sampling (α=0.2), single-step decomposition, length normalization | FKL "assign[s] unreasonably high probabilities to the void regions of p"; beats FKL-KD/SeqKD on instruction-following; better ECE (0.099 vs 0.243), lower exposure bias; self-admits "models optimized by minimizing reverse KLD are likely to lose modes". Note: arXiv v6 (2026) retitled "MiniLLM: On-Policy Distillation of LLMs" — colliding with GKD's title |
| **GKD** Agarwal et al., ICLR 2024, arXiv 2306.13649 | Token-level generalized JSD(β) (β→0 FKL, β→1 RKL after rescaling), full vocab; on-policy fraction λ | Decouples data source from divergence: on-policy sampling gives the robust gains; "optimal divergence seems to be task-dependent"; "choice of divergence is temperature-dependent" (mode-seeking helps under temperature sampling, irrelevant under greedy) |
| **DistiLLM** Ko et al., ICML 2024, arXiv 2402.03898 | Skew divergences: SKL = KL(p‖αp+(1−α)q), SRKL = KL(q‖(1−α)p+αq), α=0.1; adaptive off-policy replay | SRKL best on instruction-following; 2.5–4.3× faster than MiniLLM/GKD (SGO generation "up to 80%" of their training time); naive off-policy breaks vanilla KL/RKL (Table 4) |
| **DistiLLM-2** Ko et al., ICML 2025, arXiv 2503.07067 | SKL on **teacher**-generated responses + SRKL on **student**-generated responses (Eq. 2) | Direction–data pairing is the point: same-data interpolation "does not improve performance compared to using either SKL or SRKL alone" |
| **f-DISTILL** Wen et al., ACL 2023, arXiv 2307.15190 | Sequence-level f-divergences, stepwise decomposition (KL/RKL exact, TVD bounded) | "Symmetric distilling losses (JS and TVD) are consistently better than asymmetric ones (KL and RKL)" except near-unimodal MT; RKL has higher coverage risk ("mode collapsing") |
| **AKL** Wu et al., COLING 2025, arXiv 2404.02657 | Adaptive head/tail-gap-weighted FKL+RKL combination, fixed-dataset prefixes | See §1: anti-folklore theory + AKL improves both diversity and quality |
| **TML On-Policy Distillation** blog, Oct 2025 | Per-token RKL on student rollouts as dense reward (sampled-token estimator) | AIME'24: 74.4% at 1,800 GPU-h vs RL 67.6% at 17,920 GPU-h; "7-10x faster than RL"; RKL "unhackable"; behavior-restoration experiment (IF-eval 79→83 without knowledge loss) |
| **SKD** Xu et al., ICLR 2025, arXiv 2410.11325 | Interleaved student-propose/teacher-resample data, forward-direction objective | Anti-on-policy negative result: early student samples are OOD for teacher → "inaccurate teacher feedback" (on-policy KD 36.1 vs SKD 75.3 COMET) |
| **ImitKD** Lin et al., EMNLP 2020, arXiv 2009.07253 | FKL on student-influenced prefixes (DAgger-style) | Proves "student-sampled prefixes" predates and is orthogonal to reverse KL |
| **ABKD** Wang et al., ICML 2025, arXiv 2505.04560 | α-β-divergence interpolation | FKL/RKL = hardness- vs confidence-concentration, both extreme; interpolation wins |
| Also: BabyLlama-RKL (2410.22081), RKLU unlearning (NAACL 2025), DRKL (2604.00223), OPD survey (2604.00626), ICLR'25 blogpost ("not much difference between forward KL and reverse KL") | | |

---

## 3. Reverse KL in RL / alignment — papers (→ notes/C)

**Regularizer lineage (KL-in-reward, never argued against alternatives originally):** Ziegler 2019 (R = r − β log(π/ρ); "entropy bonus … prevents the policy from moving too far from the range where r is valid"); Stiennon 2020 ("deterring it from collapsing to a single mode" — note: RKL as collapse-*preventer* vs unregularized RL); InstructGPT 2022 ("per-token KL penalty … to mitigate over-optimization"; alignment tax mitigated by PPO-ptx, not the KL); Bai 2022 (λ_KL = 0.001, "might actually be wholly unnecessary"; reward ∝ √KL).

**Variational view:** Korbak et al. 2022 (arXiv 2205.11275): KL-regularized RL ≡ minimizing **reverse** KL to tilted posterior π* ∝ π_ref·e^{r/β} (ELBO); "naïve maximisation of the standard RL objective unavoidably leads to distribution collapse".

**Distribution-matching critique of RKL:** GDC (Khalifa 2021: FKL/DPG alternative; degeneration framing); **f-DPG** (Go et al., ICML 2023, arXiv 2302.08215): RKL-DPG shows "low diversity but high quality … ('mode collapse')"; entropy crash is transient when well-specified, severe under misspecification; "JS-DPG performing best overall"; FKL suffers high-variance gradients.

**DPO family:** DPO (arXiv 2305.18290) = closed-form RKL-constrained objective; motivation quote credits the constraint with "maintaining the generation diversity and preventing mode-collapse". **f-DPO** (arXiv 2309.16240, ICLR 2024): "Beyond Reverse KL" — RKL lowest entropy/highest win-rate, FKL opposite, JSD between; "divergence constraints directly influence expected calibration error"; cites Santurkar 2023 (RLHF-under-RKL narrows political views). f-PO (2410.21662); EXO (2402.00856, lead).

**Best-of-N matching:** **BOND** (arXiv 2407.14622) — Jeffreys divergence explicitly because "the forward KL may produce over-spread distributions, the backward KL can lead to policy and entropy collapses". **FLAG: BOND's Eq. 11 uses p = trained policy, q = target** — letters swapped vs our convention (direction agrees). vBoN (2407.06057) = pure RKL to the BoN target.

**GRPO:** DeepSeekMath (arXiv 2402.03300) Eq. 4 = k3 per token **in the loss** ("instead of adding KL penalty in the reward … avoiding complicating the calculation of Â"); β=0.04. DeepSeek-R1 restates it.

**Estimator-gradient analyses (load-bearing for §5):**
- **Tang & Munos 2025** (arXiv 2506.09477): differentiating KL value-estimates as losses is "generally incorrect": k1-as-loss → zero-mean noise; **k3-as-loss → E[∇] = ∇KL(π_ref, π), "the positions of the two distributions are reversed"** — a forward-KL gradient whose "regularization effect … is completely incidental"; k2-as-loss → unbiased RKL gradient on-policy; token-level losses "produce a partial gradient at best".
- **RPG** (arXiv 2505.17508): off-policy importance weightings per estimator/direction; "the widely-used k3 penalty is exactly the unnormalized KL"; "corrects an off-policy importance-weighting mismatch in GRPO's KL term".
- **Liu et al.** (arXiv 2510.01555): "'k1 in reward' (like PPO) is the principled loss for Reverse KL"; on-policy "'k2 as loss' … gradient-equivalent to 'k1 in reward'"; "'k3 as loss' (like GRPO) is merely a first-order, biased approximation".

**Empirical effects:** Kirk et al. (ICLR 2024, arXiv 2310.06452): RLHF "substantially decreases the diversity of outputs sampled for a given input"; across-input mode collapse; **increasing the RKL coefficient does NOT restore diversity** (§6.3). GPT-4 report Fig. 8: "post-training hurts calibration significantly" (attributed to post-training generally, not the KL direction). Gao et al. 2023 (arXiv 2210.10760): "KL penalty ineffectiveness" for the gold-reward-vs-KL frontier. ProRL (arXiv 2505.24864): the mirror image — RKL-to-reference (+ reference resets) *fights* entropy collapse in RLVR.

---

## 4. Technical reports (→ notes/D)

| Report | Stage | Direction (our convention) | Sampled from |
|---|---|---|---|
| **Nemotron 3 Ultra** (arXiv 2606.15007) | MOPD post-training | **reverse KL**, sampled-token estimator as PPO-style advantage | student rollouts |
| Nemotron 3 Ultra §3.3.5 | ablation: full-vocab/top-k logit matching | full-distribution RKL estimator — **failed** (quote below) | student prefixes |
| Nemotron 3 Ultra | MTP boosting | forward KL (T=2) | own rollouts |
| **Nemotron-Cascade 2** (2603.19220) | MOPD in Cascade RL | **reverse KL**, sampled-token ("computed only on the student-sampled token rather than over the full vocabulary") | student |
| Minitron (2407.14679) | prune-retrain KD | forward KL; **R-KLD tried and worse** (Table 16: 2.155 vs 2.665) | dataset |
| Nemotron-H / Nano 2 (2504.03624 / 2508.14444) | prune-retrain KD | "forward KL divergence loss exclusively" | dataset |
| Qwen3 (2505.09388) | strong-to-weak distillation | phase 1 SFT-on-teacher-samples; phase 2 on-policy "minimize the KL divergence" on logits — **direction unspecified** | teacher / student |
| Gemma 2 (2408.00118) | pretraining KD | forward (CE to teacher soft targets, equation given) | dataset |
| Gemma 2 | post-training | "distillation from the teacher on the student's distribution" citing GKD+MiniLLM — unspecified | student |
| Gemma 3 (2503.19786) | pretraining KD | forward/CE on **sampled 256-logit support**, renormalized | dataset |
| Llama 3.2 1B/3B (blog) | post-pruning KD | logit targets, unspecified | dataset |
| DeepSeek-R1 (2501.12948) | distill to small models | SFT on 800k teacher samples | teacher |
| Apple AFM 2024 (2407.21075) | core pretrain | teacher top-1 label mix (0.9) | dataset |
| Apple AFM 2024 | RLHF (MDLOO) | reverse-KL penalty **written out** (rare) | student |
| Gemini 1.5 Flash (2403.05530) | "online distilled" from Pro | unspecified | unspecified |
| Tülu 3 / OLMo 2 | RLVR | reverse-KL penalty (Eq. 4, boilerplate) | student |
| Checked, no KD/KL stage: Phi-4, MiniCPM, Hunyuan, GLM-4, SmolLM2, InternLM2, Yi, Kimi K2 (OPMD squared log-ratio ≠ KL), Llama 3, ERNIE 4.5, Falcon 3 (blog-level KD, unspecified), Seed-OSS, Ministral, Grok, GPT-4o mini | | | |

**Failure / negative quotes (verbatim, primary):**

1. **Nemotron 3 Ultra §3.3.5** ("Logit matching"): "A natural alternative to the sampled-token objective is distribution-level distillation, where the student is trained to match the teacher's predictive distribution over the top-k tokens or the full vocabulary at each prefix. In our preliminary experiments, these objectives did not improve MOPD performance and consistently underperformed the sampled-token objective on some agentic benchmarks such as Terminal Bench. We hypothesize that full-distribution matching may impose an overly strong local constraint on prefixes sampled from the student policy, particularly when those prefixes have limited support under the teacher distribution. In this regime, teacher logits can become poorly calibrated or less informative, and matching the full distribution may amplify noise from off-support states. By contrast, the sampled-token objective applies supervision only to realized actions and may therefore provide a more stable on-policy learning signal." — NB: their shipped objective **is** reverse KL; what failed is the **full-distribution estimator** of it on student prefixes.
2. **Minitron §4.3**: "we experiment with Kullback-Leibler divergence (KLD), MSE, cosine similarity and reverse KLD (R-KLD) … We observe from Table 15 and 16 that KLD is the best choice for pruned base model training." (A.3: "KLD consistently outperforms R-KLD".)
3. **Apple 2024 §3.2.2**: "We did not find a distillation loss to be helpful here for AFM-on-device, unlike in core pre-training." (distillation generally, not direction-specific)
4. Research-adjacent (unverified quotes, leads): Entropy-Aware OPD (2603.07079); "Your Teacher Can't Help You Here" (2605.30833); Rethinking OPD (2604.13016).

---

## 5. Implementation audits (→ notes/E, notes/F)

### Table 1 — Sources (draft)

| # | Report/model | Stage | Claimed objective | Implementation @ commit | Used in reported experiments? | Evidence |
|---|---|---|---|---|---|---|
| 1 | MiniLLM | KD fine-tuning | seq-level RKL | microsoft/LMOps `minillm` @ `4f2a9de` (losses.py, sampler.py) | **USED** (run scripts set all flags) | notes/E §1 |
| 2 | GKD | KD fine-tuning | token-level JSD(β) incl. RKL at β=1 | huggingface/trl `experimental/gkd/gkd_trainer.py` @ `24db2a8` | reference impl; **paper code not public** (Google-internal) | notes/E §2 |
| 3 | DistiLLM | KD fine-tuning | skew SKL/SRKL | jongwooko/distillm @ `d47e77f` | **USED** (scripts: adaptive-s(f/r)kl) | notes/E §3a |
| 4 | DistiLLM-2 | KD fine-tuning | SKL(teacher-data)+SRKL(student-data) | jongwooko/distillm-2 @ `fe4cf9b` | **USED** (`distillm_v2` config) | notes/E §3b |
| 5 | TML on-policy distillation | post-training | per-token RKL on student rollouts | tinker-cookbook @ `3d3e976` (train_on_policy.py); loss kernel server-side | **USED** (public code for blog; kernel closed-source, docs-verbatim) | notes/E §4 |
| 6 | Nemotron 3 Ultra / Cascade 2 | MOPD | per-token RKL as advantage | NVIDIA-NeMo/RL: Ultra OPD **fully public on main** (#2780, from ultra-v3); Cascade 2 variant **unreleased** (dataset loader only) | Ultra: code USED, verifiable; Cascade 2: report-only | notes/D, notes/E §6c, notes/F §5b |
| 7 | torchtune KD | KD recipe | forward KL (RKL class exists) | pytorch/torchtune @ `bd2a0fc` kd_losses.py | ForwardKL **USED** in all configs; **ReverseKLLoss EXISTS-UNUSED** | notes/E §5 |
| 8 | NVIDIA modelopt | Minitron-style KD | forward KL | TensorRT-Model-Optimizer @ `9220fac` | forward USED; `reverse=True` flag **unreachable from configs** | notes/E §6a |
| 9 | NeMo-Aligner KD | offline top-K KD | fwd_kl default; `bwd_kl` = top-K-renormalized RKL w/ analytic gradient | NeMo-Aligner @ `af2b3bd` distributed.py L510–760 | fwd USED (tutorial); **bwd_kl EXISTS-UNUSED** | notes/E §6b |
| 10 | RLHF classic (trl ≤0.11 PPO) | RLHF regularizer | RKL penalty | trl @ v0.11.4 `_kl_penalty` | k1-in-reward USED-BY-DEFAULT; "full" (full-vocab RKL) opt-in | notes/F §1a |
| 11 | trl new PPO (≤0.24) | RLHF regularizer | RKL penalty | trl @ v0.24.0 | k1-in-reward default; k3 opt-in | notes/F §1b |
| 12 | trl GRPOTrainer (main) | RLVR regularizer | RKL penalty | trl @ `24db2a8` | β=0 default (opt-in); when on: k3·IS-ratio in-loss | notes/F §1c |
| 13 | verl | RLVR regularizer | RKL penalty | verl @ `b75f0d0` | in-loss k3 ("low_var_kl") set in flagship GRPO recipes; in-reward k1 opt-in | notes/F §2 |
| 14 | OpenRLHF | RLHF regularizer | RKL penalty | OpenRLHF @ `bc71bb1` | k1-in-reward USED-BY-DEFAULT; in-loss opt-in | notes/F §3 |
| 15 | open-instruct (Tülu 3/OLMo) | RLVR regularizer | RKL penalty | @ `05ad0a2` grpo_fast; paper-era PPO @ `9502f16` | grpo_fast in-loss k3 default (Tülu3 recipe β=0.01); paper-era k1-in-reward | notes/F §4 |
| 16 | NeMo-Aligner PPO | RLHF regularizer | RKL penalty | @ `af2b3bd` ppo_utils.py | **\|k1\|**-in-reward USED-BY-DEFAULT | notes/F §5a |
| 17 | nemo-rl | RLVR regularizer | RKL penalty | NVIDIA-NeMo/RL @ `037fb36` | in-loss k3·exp(x−sg(x)) USED-BY-DEFAULT (cites Tang & Munos) | notes/F §5b |
| 18 | DeepSeekMath GRPO | RLVR regularizer | RKL penalty | paper only (Eq. 3–4); no official code | — | notes/C §6, notes/F §6 |

### Table 2 — Estimator audit (draft; classifications mine, derivations to be written out in the review)

Notation: prefixes s_t, tokens y_t; "sampled-token" = single gathered logprob per position; "full-vocab" = Σ_v. sg[·] = stop-gradient.

| # | Expression computed | Sampled var ∼ dist | Scope | Gradient path | Support/decoding mods | Resulting objective / E[∇] | Classification (confidence) |
|---|---|---|---|---|---|---|---|
| 1 | MiniLLM: clipped-IS PG loss, A_t = whiten(Σγ^{t'−t}r_{t'}/L_t), r_t = log p−log q_old (sg), + full-vocab per-position RKL reg + FKL LM term | y ∼ (1−α)q_old + αp (α=0.2) | seq-level PG + token full-vocab reg | ∇ through live log q at sampled tokens (ratio) and through full-vocab reg; rewards/advantages sg | reward scaling 0.5, reward clip ±100, PPO clip 0.2, γ=0.95, whitening, length-norm, teacher-mix | PG estimator family of ∇(seq-RKL) with deliberate reward-shaping/variance mods; not exact ∇RKL | **correct for stated surrogate** (med) |
| 2 | TRL GKD β=1: full-vocab KL(q‖p) per position via F.kl_div(t, s, log_target) (direction verified) | prefixes: dataset w.p. 1−λ, student rollouts w.p. λ (batch-level coin) | token-level, full-vocab | ∇ through q everywhere; teacher sg; **no ∇ through prefix sampling** | temperature applied at generation (0.9) but **not** in loss; top_k=0/top_p=1 | exact token-KL under sg data distribution — GKD's stated objective | **correct for claimed objective** (high) |
| 3 | DistiLLM SRKL = KL(q‖(1−α)p+αq), full-vocab, α=0.1 | prefixes: dataset / student replay (adaptive) | token-level, full-vocab | ∇ through q incl. inside mixture; teacher sg | none (prob-space mixture, no ε) | exact skew-RKL (stated objective) | **correct for claimed objective** (high) |
| 4 | DistiLLM-2: SKL on teacher data (q in mix live) + SRKL on student data with **q inside mixture sg** | responses generated offline (teacher T=0.8 / student) | token-level, full-vocab, per-seq mean | outer q live; q inside m₂ detached | fixed α=0.1 (adaptive code disabled) | gradient ≠ exact ∇SRKL (detach); deliberate | **correct for stated surrogate** (med; derivation needed) |
| 5 | Tinker OPD: A_t = −sg[log q_sample(y_t) − log p(y_t)]; loss = −Σ exp(log q_θ−log q_sample)·A_t | y ∼ q (T=1) | token-level, sampled-token | ∇ only through log q_θ via IS ratio; A_t sg | no clipping in loss; T=1; teacher logprobs on sampled seq | on-policy E[∇] = Σ_t E[k1_t·∇log q_t] = exact ∇ of token-KL at frozen student-visited prefixes; omits prefix-visitation term of seq-RKL | **correct for stated surrogate** (high) |
| 6 | Nemotron MOPD: Â_t = sg[log π_T − log π_prox], PPO clip + IcePop masking (Cascade 2: truncated IS weights) | y ∼ π_θ (async) | token-level, sampled-token | policy-gradient route (advantage sg) | clipping, token masking | same family as #5 (sampled-token RKL advantage) | **correct for stated surrogate** (med — code partially public) |
| 7 | torchtune ReverseKLLoss: full-vocab KL(q‖p) | dataset prefixes | token-level, full-vocab | ∇ through q; teacher sg | −100 masking | exact token-RKL | correct for claimed; **EXISTS-UNUSED** (high) |
| 8 | NeMo-Aligner bwd_kl: Σ_{k∈topK} q̃(log q̃−log p̃), both renormalized on teacher top-K; hand-written analytic backward | dataset tokens (teacher-forced) | token-level, top-K support | analytic grad wrt student logits at K ids; zero elsewhere | **top-K restriction + renormalization changes which KL** | exact RKL of the K-renormalized distributions (≠ full RKL) | correct for restricted objective; **EXISTS-UNUSED** (high) |
| 9 | k1-in-reward (trl PPO both, OpenRLHF, NeMo-Aligner w/o abs, verl opt-in, Tülu3 paper-era): reward −= β(log q_old−log p) | y ∼ q (rollout) | token-level, sampled-token, in-reward | all KL tensors no_grad; gradient via advantages (score-function) | clamps (±10 OpenRLHF), adaptive coefs, temperature-scaled logprobs (see notes/F) | on-policy E[∇] = β∇KL(q‖p) contribution (score-function route; exact per-token given GAE caveats) | **correct for claimed objective** (high; NeMo-Aligner's \|k1\| is a different penalty — flag) |
| 10 | k3-as-loss, plain (verl low_var_kl; open-instruct est.2; trl GRPO with bias-correction OFF; DeepSeekMath as written): β·(e^{logp−logq}−(logp−logq)−1), ref sg, ∇ through log q | y ∼ q | token-level, sampled-token, in-loss | ∇k3 = (1−p/q)∇log q | clamps ±20/±10 (verl, nemo-rl), ±40 (open-instruct) | **E[∇] = ∇KL(p‖q) — forward-KL gradient** (Tang & Munos; my check: E_q[(1−r)∇log q] = −Σp∇log q = ∇KL(p‖q)) | **biased / inconsistent with claimed reverse-KL objective** (high; value estimate unbiased, gradient direction flipped) |
| 11 | k3 × differentiable IS ratio (trl GRPO default `use_bias_correction_kl`; nemo-rl exp(x−sg(x)) surrogate; open-instruct est.3 = ratio·k1) | y ∼ q | token-level, sampled-token, in-loss | ∇ through both ratio and k3/k1 | as above | on-policy E[∇(ratio·k3)] = E[k3∇log q]+E[∇k3] = ∇KL(q‖p) — **unbiased reverse-KL gradient** (derivation to be written out) | **correct for claimed objective** (med-high; off-policy needs RPG-style weighting) |
| 12 | k2-as-loss (verl "mse"/"k2", OpenRLHF k2, "k3+" straight-through) | y ∼ q | token-level, sampled-token, in-loss | ∇k2 = (log q−log p)∇log q | clamps | E[∇] = ∇KL(q‖p) on-policy (unbiased; Tang & Munos) | **correct for claimed objective** (high) |
| 13 | trl-classic "full": full-vocab KL(q‖p) in-reward, no_grad | y ∼ q (prefix visitation) | token-level, full-vocab, in-reward | via advantages only | none | reverse-KL penalty with exact per-position values | correct for claimed; OPT-IN (high) |

**One-line derivations backing rows 9–12 (to be expanded in the review):** with r = p/q at a sampled token, E_q[∇log q] = 0 gives: ∇KL(q‖p) = E[k1·∇log q] (hence k1-in-reward and k2-as-loss are both unbiased on-policy; they coincide for each token's own penalty term, but k1-in-reward's reward-to-go channel additionally carries the visitation cross-terms that k2-as-loss omits); ∇(k3) = (1−r)∇log q with E = −Σp∇log q = ∇KL(p‖q) (forward); adding the differentiable ratio restores E[∇(ratio·k3)] = ∇KL(q‖p). Naive backprop of sampled log q−log p (no ratio, no advantage) has E[∇] = E[∇log q] = 0 — pure noise. Token-level estimators of sequence-level KL additionally omit the prefix-visitation gradient (Tang & Munos "partial gradient"; Nemotron/Tinker use per-token advantages, i.e. optimize token-KL under frozen visitation).

**Support/decoding modifications found that change which KL is estimated:** temperature scaling of both sides (trl PPO 0.7 default; GKD generation 0.9 vs loss at T=1); top-K restriction + renormalization (NeMo-Aligner KD, modelopt TopK, nemo-rl distillation with tail-entropy correction, Gemma 3's 256-sampled-logit support); log-ratio/output clamps (verl ±20/±10, OpenRLHF ±10, open-instruct ±40, MiniLLM reward clip ±100); nemo-rl is the only framework that deliberately computes the KL on **unfiltered** logprobs when rollout top-k/top-p filtering is on.

---

## 6. Alternative divergences introduced to fix FKL/RKL shortcomings

Minka α-family (the original knob); Huszár JS_π; GKD JSD(β); f-DISTILL JS/TVD ("symmetric … consistently better"); DistiLLM skew-KL (bounded gradients); AKL adaptive head/tail; ABKD α-β; BOND Jeffreys ("inherits the best of both"); f-DPO/f-PO divergence families for preference optimization; f-DPG (JS best overall); Rényi-VI, Markovian Score Climbing (forward-KL VI) as the mirror-image fix.

---

## 7. Direct answers (condensed; full versions in notes/A–F "Direct answers")

1. **Folklore:** use FKL for coverage/likelihood, RKL for sharp high-precision generation tolerant of dropped modes. Theoretical basis: zero-forcing under capacity constraint at the global optimum. Empirical basis: thin and mixed; strongest LLM-scale counter-evidence: Wu et al. (shared optimum, transient difference), GKD (task/temperature-dependent), Goodfellow (GAN mode collapse not divergence-driven), Soletskyi (optimization-driven collapse).
2. **Motivations/figures:** void-regions argument (MiniLLM), train-inference mismatch (GKD — actually a data argument, not a direction argument), unhackability + density (TML), tractability (VI/RLHF: only direction estimable from on-policy samples with logprob access). Figures: Bishop 10.2/10.3, Minka Fig 1, MiniLLM Fig 2, GKD A.16, Goodfellow 14.
3. **Where RKL is actually used in training:** (a) on-policy post-training distillation (MiniLLM, TML, Qwen3?, Nemotron 3 Ultra/Cascade 2 — the only production stages that say "reverse KL"); (b) RLHF/RLVR regularizer (universal, boilerplate); (c) DPO implicitly; (d) unlearning/behavior-restoration. **Never** found in pretraining-scale distillation (all forward/CE).
4. **Empirical effects:** wins on instruction-following quality/win-rate (MiniLLM, GKD-RKL variants, DistiLLM-SRKL), large compute savings vs RL (TML 7–10×, Qwen3 10×); costs: per-input diversity ↓ (Kirk), entropy ↓ (f-DPG/f-DPO), coverage risk ↑ (f-DISTILL); calibration: mixed (MiniLLM ECE better than FKL-KD baselines; f-DPO links ECE to divergence choice; GPT-4 post-training calibration loss not direction-attributed); RLVR: RKL-to-reference *preserves* entropy vs unregularized RL (ProRL).
5. **Failure quotes:** Nemotron 3 Ultra §3.3.5 (full-distribution RKL estimator failed; sampled-token shipped), Minitron (R-KLD worse for base-model compression), Apple 2024 (distillation unhelpful in continued pretraining), DistiLLM (naive off-policy RKL breaks), SKD (pure on-policy KD gets garbage teacher feedback early).
6. **Implementations:** see Tables 1–2. Headline: the k3-as-loss pattern (GRPO default in verl/open-instruct; trl before bias-correction) optimizes a forward-KL gradient in expectation while being labeled reverse KL; k1-in-reward and k2-as-loss are the on-policy-correct patterns; production on-policy distillation (TML, Nemotron) uses sampled-token score-function estimators that are exact for token-KL at frozen prefixes but omit the visitation gradient; full-vocab RKL classes exist in torchtune/NeMo-Aligner/modelopt but are unused by any shipped config — with one exception found 2026-08-19: NeMo-RL main ships a recipe (`distillation-qwen3-32b-to-1.7b-base…v1.yaml`) running its top-k distribution-level on-policy distillation with `kl_type: reverse` (notes/E §6c).

---

## 8. Gaps / re-checks before drafting

- Re-fetch Schulman blog directly (proxy-only so far) before quoting verbatim.
- [FETCH]-marked quotes in notes/C (f-DPO Table 2 numbers, DeepSeekMath β=0.04, Kirk "first rigorous", f-DPG §3.4 sentence) — re-verify wording.
- Nemotron 3 Ultra: verify NeMo-RL `ultra-v3` branch code path details for MOPD (only the guide was located).
- Qwen3 on-policy KL direction: genuinely unspecified; consider noting TML blog's claim that it follows the same recipe (Table 21 lead).
- DeepSeek-V3.2 "ρk3" (importance-ratio-corrected KL) primary source (drives trl's `use_bias_correction_kl`).
- EXO (2402.00856), Santurkar 2023, Jaques 2017, Turner et al. 2008 TR, DPKD, BiLD direction, ICLR'25 blogpost authors — unfetched leads.
- Derivations for Table 2 rows 1, 4, 11 (MiniLLM PG bias terms; DistiLLM-2 detach; ratio·k3) to be written out in the review with the sign conventions triple-checked.
