# Agent C result: Reverse KL in RLHF / Alignment / RL Fine-Tuning — Annotated Source Dossier

**Convention used throughout:** "reverse KL" (RKL) = D_KL(q‖p), q = policy being trained, p = reference/teacher. "Forward KL" (FKL) = D_KL(p‖q). Terminology flags mark any source deviating in letters or naming.

**Verification status:** [RAW] = verified character-for-character against the paper's HTML fetched during the agent session; [FETCH] = extracted by the WebFetch summarizer from the correct page but not re-verified against raw HTML (near-verbatim; re-check before final publication).

---

## 1. Reverse KL as a REGULARIZER in classic RLHF

### 1.1 Ziegler et al. 2019 — "Fine-Tuning Language Models from Human Preferences"
- Links: https://arxiv.org/abs/1909.08593 , https://ar5iv.labs.arxiv.org/html/1909.08593
- **Math (Sec. 2, Eq. 2):** RL performed on a *modified reward*, not a loss term: R(x,y) = r(x,y) − β·log[π(y|x)/ρ(y|x)], y sampled from π (policy), ρ = initial LM. Sequence-level as written; penalty enters the reward channel, optimized with PPO. Expectation of the penalty is β·KL(π,ρ) — policy first ⇒ **reverse KL** in our convention. **Flag:** comma notation "KL(π,ρ)" for KL(π‖ρ).
- **Motivation [RAW], Sec. 2:** "To keep π from moving too far from ρ, we add a penalty with expectation β KL(π,ρ) (see table 10 for what happens without this). That is, we perform RL on the modified reward R(x,y)=r(x,y)−β log π(y|x)/ρ(y|x). (2) We either choose a constant β or vary it dynamically to achieve a particular value of KL(π,ρ); see section 2.2. This term has several purposes: it plays the role of an entropy bonus, it prevents the policy from moving too far from the range where r is valid, and in the case of our style continuation tasks it also is an important part of the task definition: we ask humans to evaluate style, but rely on the KL term to encourage coherence and topicality."
- **Adaptive β [FETCH], Sec. 2.2:** log-space proportional controller targeting a fixed KL(π,ρ) value (K_β = 0.1). Table 10 shows degeneration without the penalty.

```bibtex
@article{ziegler2019finetuning,
  title={Fine-Tuning Language Models from Human Preferences},
  author={Ziegler, Daniel M. and Stiennon, Nisan and Wu, Jeffrey and Brown, Tom B. and Radford, Alec and Amodei, Dario and Christiano, Paul and Irving, Geoffrey},
  journal={arXiv preprint arXiv:1909.08593}, year={2019}}
```

### 1.2 Stiennon et al. 2020 — "Learning to summarize from human feedback"
- Links: https://arxiv.org/abs/2009.01325 , https://ar5iv.labs.arxiv.org/html/2009.01325
- **Math (Sec. 3.4):** "we include a term in the reward that penalizes the KL divergence between the learned RL policy π^RL_φ … and this original supervised model π^SFT" [RAW]; R(x,y) = r_θ(x,y) − β log[π^RL_φ(y|x)/π^SFT(y|x)]. KL **in the reward**; reward-model score arrives only at end-of-sequence (footnote 7: "the reward model only gives rewards for entire summaries, and not at intermediate time steps" [RAW]); KL log-ratio written at sequence level. Policy-first ⇒ reverse KL.
- **Motivation [RAW], Sec. 3.4:** "This KL term serves two purposes. First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode. Second, it ensures the policy doesn't learn to produce outputs that are too different from those that the reward model has seen during training."
- Note: the *original* framing is that the RKL penalty **prevents** collapse (relative to unregularized reward maximization) — in tension with later literature blaming reverse KL (relative to distribution matching) for collapse. The baseline of comparison differs.

```bibtex
@inproceedings{stiennon2020learning,
  title={Learning to summarize from human feedback},
  author={Stiennon, Nisan and Ouyang, Long and Wu, Jeff and Ziegler, Daniel M. and Lowe, Ryan and Voss, Chelsea and Radford, Alec and Amodei, Dario and Christiano, Paul},
  booktitle={NeurIPS}, year={2020}, note={arXiv:2009.01325}}
```

### 1.3 Ouyang et al. 2022 — InstructGPT
- Links: https://arxiv.org/abs/2203.02155 , https://ar5iv.labs.arxiv.org/html/2203.02155
- **Math (Sec. 3.5, their Eq. 2) [RAW]:** objective(φ) = E_{(x,y)∼D_{π^RL_φ}}[r_θ(x,y) − β log(π^RL_φ(y|x)/π^SFT(y|x))] + γ E_{x∼D_pretrain}[log π^RL_φ(x)]. Samples from the trained policy; KL enters via the reward; pretraining-mix term (PPO-ptx) is a separate loss term.
- **Per-token, and why [RAW], Sec. 3.5:** "In addition, we add a per-token KL penalty from the SFT model at each token to mitigate over-optimization of the reward model."
- **Alignment tax [RAW], Sec. 4:** "This is an example of an 'alignment tax' since our alignment procedure comes at the cost of lower performance on certain tasks that we may care about." Mitigation: "We can greatly reduce the performance regressions on these datasets by mixing PPO updates with updates that increase the log likelihood of the pretraining distribution (PPO-ptx), without compromising labeler preference scores." Note: tax mitigated by an added *forward-flavored* likelihood term on pretraining data, not by strengthening the RKL penalty.

```bibtex
@inproceedings{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll L. and Mishkin, Pamela and others},
  booktitle={NeurIPS}, year={2022}, note={arXiv:2203.02155}}
```

### 1.4 Bai et al. 2022 — Anthropic HH
- Links: https://arxiv.org/abs/2204.05862 , https://ar5iv.labs.arxiv.org/html/2204.05862
- **Math (Sec. 4.1, Eq. 4.1) [RAW]:** "we also follow other work [Stiennon et al., 2020] and apply an empirically-estimated KL penalty term in the reward, with the total reward given by r_total = r_PM − λ_KL D_KL(policy ‖ policy_0)". KL **in the reward**; policy-first ⇒ reverse KL; "empirically-estimated" (Monte-Carlo, not exact).
- **Coefficient / near-irrelevance [RAW], Sec. 4.1:** "In practice we use a very small value of λ_KL = 0.001, which likely has a very minor impact during most of RL training (as D_KL < 100 typically), and might actually be wholly unnecessary."
- **Reward–KL scaling [RAW], abstract:** "we investigate the robustness of RLHF training, and identify a roughly linear relation between the RL reward and the square root of the KL divergence between the policy and its initialization." (Elaborated Sec. 4.3.)

```bibtex
@article{bai2022training,
  title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback},
  author={Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Nova and others},
  journal={arXiv preprint arXiv:2204.05862}, year={2022}}
```

---

## 2. The variational / Bayesian-inference view

### 2.1 Korbak, Perez, Buckley 2022 — "RL with KL penalties is better viewed as Bayesian inference" (Findings of EMNLP 2022)
- Links: https://arxiv.org/abs/2205.11275 , https://ar5iv.labs.arxiv.org/html/2205.11275
- **Math:** J_KL-RL(θ) = E_{x∼π_θ}[r(x)] − β D_KL(π_θ, π_0) (Eq. 3; comma notation, policy first ⇒ reverse KL). Optimal policy [RAW], Eq. 5: "π*_KL-RL(x) = (1/Z) π_0(x) exp(r(x)/β)". Key identity [RAW], Eq. 7: "the KL-regularised RL objective can be cast as minimising the KL divergence between the LM π_θ and this target distribution π*_KL-RL: J_KL-RL(θ) ∝ −D_KL(π_θ, π*_KL-RL)" — KL-regularized RLHF **is reverse-KL variational inference toward the tilted posterior**. "Minimising this new divergence coincides with a variational inference… J_KL-RL(θ) is the evidence lower bound (ELBO) on the log likelihood of π_θ being optimal under r, assuming a prior π_0." [RAW]
- **Degeneration of pure RL [RAW], abstract:** "naïve maximisation of the standard RL objective unavoidably leads to distribution collapse: turning the LM into a degenerate distribution."
- **Anti-RL framing [RAW]:** "RL is not an adequate formal framework for problems such as fine-tuning LMs." … "The Bayesian perspective moves KL-regularised RL closer to other divergence-minimisation-based approaches to fine-tuning LMs (Khalifa et al., 2021) and, more broadly, to other divergence-minimisation-based accounts of control (Levine, 2018; Hafner et al., 2020)."
- **Lineage:** control-as-inference (Levine 2018; Hafner et al. 2020); KL-control ancestry in LMs via Jaques et al. 2017 (Sequence Tutor), acknowledged by GDC: "Jaques et al. (2017); Ziegler et al. (2019) propose a conservative fine-tuning approach moderated by a KL penalty between the trained policy and the original LM, discouraging large deviations." [RAW, Khalifa et al. 2021 Sec. 1]. Todorov/Kappen NOT cited by name in Korbak's main text — don't cite them via this paper.

```bibtex
@inproceedings{korbak2022rl,
  title={RL with KL penalties is better viewed as Bayesian inference},
  author={Korbak, Tomasz and Perez, Ethan and Buckley, Christopher L},
  booktitle={Findings of EMNLP}, year={2022}, note={arXiv:2205.11275}}
```

---

## 3. Distributional-control line that criticizes reverse KL

### 3.1 Khalifa, Elsahar, Dymetman 2021 — GDC (ICLR 2021)
- Links: https://arxiv.org/abs/2012.11635 , https://ar5iv.labs.arxiv.org/html/2012.11635
- **Math:** two-stage. (i) Target p = argmin_{c∈C} D_KL(c, a) (Eq. 1) — an EBM p(x) ∝ a(x)e^{λ·φ(x)}. (ii) Train policy π_θ toward p with **DPG minimizing CE(p, π_θ)**, i.e. **forward KL** D_KL(p‖π_θ), importance-weighted gradient −E_{x∼q}[(p(x)/q(x)) ∇_θ log π_θ(x)]. Trained-policy divergence is forward KL — the explicit alternative to Ziegler-style RKL fine-tuning.
- **Criticism of reward maximization / KL-penalty fine-tuning [RAW], Sec. 1:** "However, such an optimization process is not infallible; Liu et al. (2016a) noted that it often leads to 'degeneration', producing poor examples that improve the average reward but forgo coherence and fluency. This degeneration is often diagnosed as an effect of deviating too much from the original pretrained LM during optimization. Consequently, prior work has regarded proximity to the pretrained model as a prescription for sample quality." On PPLM's KL penalty [RAW]: "the authors show that balancing policy deviations from the original LM while also satisfying the control conditions is delicate. To combat degeneration they had to combine the KL penalty with post-norm fusion, reranking, and early-stopping procedures."
- GDC does **not** frame its critique as "reverse vs forward KL" by name; the explicit direction comparison is in f-DPG.

```bibtex
@inproceedings{khalifa2021distributional,
  title={A Distributional Approach to Controlled Text Generation},
  author={Khalifa, Muhammad and Elsahar, Hady and Dymetman, Marc},
  booktitle={ICLR}, year={2021}, note={arXiv:2012.11635}}
```

### 3.2 Go et al. 2023 — f-DPG (ICML 2023)
- Links: https://arxiv.org/abs/2302.08215 , https://ar5iv.labs.arxiv.org/html/2302.08215
- **Math:** min_θ D_f(π_θ‖p), p an EBM target (e.g. p(x) ∝ a(x)exp(r(x)/β)). Instances: forward KL = KL(p‖π_θ) (recovering GDC/DPG), reverse KL = KL(π_θ‖p), JS, TV. Notation matches our convention. They state RLHF-style RL-with-KL-penalties is the reverse-KL instance [FETCH; corroborated by Korbak Eq. 7].
- **Mode collapse under RKL [RAW], Sec. 4.1:** "…additional metrics on generated sentences, which show low diversity but high quality for RKL-DPG, compared to other f-DPGs, suggesting it captures a subset of the target distribution ('mode collapse'), as commonly observed in other generative models (Huszar, 2015; Che et al., 2017; Mescheder et al., 2018)." And: "This can lead π_θ(x) to concentrate on high-probability regions of p(x), at the cost of diversity, which can also be seen in the low entropy of the generated samples." [RAW]
- **Mode-seeking dynamics [RAW], GPT-2 distillation experiment:** "RKL-DPG shows a significant drop of entropy in the initial phase, but still converges to the distribution of p(x). … we link the drop to the mode-seeking behaviour. However, since we are not in the mis-specified scenario, the model can recover, and cover the rest of the distribution." (Collapse is severe under *misspecification*.)
- **Which divergence wins [RAW]:** "the three remaining variants of DPG (KL, TV and JS) consistently minimize all four tracked divergences, with JS-DPG performing best overall." Fig. 7 Pareto frontiers: RKL best at reward/alignment, worst at diversity; pure FKL suffers high-variance gradients: "KL-DPG lags considerably… We link this to the high variance of the KL-DPG pseudo-reward" [RAW].

```bibtex
@inproceedings{go2023aligning,
  title={Aligning Language Models with Preferences through f-divergence Minimization},
  author={Go, Dongyoung and Korbak, Tomasz and Kruszewski, Germ{\'a}n and Rozen, Jos and Ryu, Nahyeon and Dymetman, Marc},
  booktitle={ICML}, year={2023}, note={arXiv:2302.08215}}
```

---

## 4. DPO as implicitly reverse-KL-constrained; f-DPO

### 4.1 Rafailov et al. 2023 — DPO
- Links: https://arxiv.org/abs/2305.18290 , https://ar5iv.labs.arxiv.org/html/2305.18290
- **Math [RAW]:** Eq. 3: max_{π_θ} E_{x∼D, y∼π_θ(y|x)}[r_φ(x,y)] − β D_KL[π_θ(y|x) ‖ π_ref(y|x)] — explicit reverse KL, policy first. Eq. 4: π_r(y|x) = (1/Z(x)) π_ref(y|x) exp((1/β) r(x,y)). DPO's *implicit* constraint is exactly this RKL-regularized objective (sequence-level).
- **Motivation quote [RAW], Sec. 3:** "The added constraint is important, as it prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse to single high-reward answers." Also [RAW]: "The standard approach … has been to construct the reward function r(x,y)=r_φ(x,y)−β(log π_θ(y|x)−log π_ref(y|x)), and maximize using PPO."

```bibtex
@inproceedings{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D. and Ermon, Stefano and Finn, Chelsea},
  booktitle={NeurIPS}, year={2023}, note={arXiv:2305.18290}}
```

### 4.2 Wang et al. 2023 — f-DPO ("Beyond Reverse KL…", ICLR 2024)
- Links: https://arxiv.org/abs/2309.16240 , https://openreview.net/forum?id=2cRzmWXK9N
- **Math:** replaces DPO's RKL constraint with D_f(π_θ, π_ref); via KKT the partition function drops for f with 0 ∉ dom(f′) (JSD, forward KL, α-divergences), yielding L = E[−log σ(β f′(π_θ(y_w|x)/π_ref(y_w|x)) − β f′(π_θ(y_l|x)/π_ref(y_l|x)))] [FETCH, Eq. 3]. **Flag:** in D_f(p‖q)=Σ_x q f(p/q), their reverse KL is f(u)=u log u and forward KL is f(u)=−log u — f-divergence "‖" order does not visually match KL subscript order.
- **Criticism of RKL [RAW], Sec. 1:** "The mode-seeking property of reverse KL divergence tends to reduce diversity in generation (Wiher et al., 2022; Khalifa et al., 2021; Perez et al., 2022; Glaese et al., 2022), which, although beneficial for optimizing alignment performance, may limit the model's potential (e.g., user engagement). Additionally, Santurkar et al. (2023) observe that finetuning LLMs with RLHF under the reverse KL regularization will result in a limited range of political views."
- **Findings:** [RAW, abstract] "Empirically, adopting these divergences ensures a balance between alignment performance and generation diversity. Importantly, f-DPO outperforms PPO-based methods in divergence efficiency, and divergence constraints directly influence expected calibration error (ECE)." [FETCH, Sec. 5] IMDB: FKL highest predictive entropy, RKL lowest; Anthropic HH (Table 2): RKL 67.19% win-accuracy w/ lowest entropy (12.25), FKL 54.30% w/ highest entropy (13.01); JSD in between.

```bibtex
@inproceedings{wang2024beyond,
  title={Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints},
  author={Wang, Chaoqi and Jiang, Yibo and Yang, Chenghao and Liu, Han and Chen, Yuxin},
  booktitle={ICLR}, year={2024}, note={arXiv:2309.16240}}
```

### 4.3 Han et al. — f-PO ("f-PO: Generalizing Preference Optimization with f-divergence Minimization")
- Link: https://arxiv.org/abs/2410.21662 . [FETCH, abstract]: "f-PO minimizes f-divergences between the optimized policy and the optimal policy, encompassing a broad family of alignment methods… unifies previous algorithms like DPO and EXO."

```bibtex
@inproceedings{han2025fpo,
  title={f-PO: Generalizing Preference Optimization with f-divergence Minimization},
  author={Han, Jiaqi and Jiang, Mingjian and Song, Yuxuan and Ermon, Stefano and Xu, Minkai},
  booktitle={AISTATS}, year={2025}, note={arXiv:2410.21662; venue per arXiv listing—verify}}
```

---

## 5. BOND and vBoN: distribution matching to Best-of-N

### 5.1 Sessa et al. 2024 — BOND
- Links: https://arxiv.org/abs/2407.14622 , https://arxiv.org/html/2407.14622v1
- **Framing [RAW, abstract]:** "BOND is a distribution matching algorithm that forces the distribution of generations from the policy to get closer to the Best-of-N distribution. We use the Jeffreys divergence (a linear combination of forward and backward KL) to balance between mode-covering and mode-seeking behavior, and derive an iterative formulation that utilizes a moving anchor for efficiency."
- **Objective:** π_BOND = argmin_π D(π ‖ π_BoN); Sec. 3.3 "Connection with standard RLHF"; App. A.3: "Backward KL and policy gradient equivalence" (backward-KL-to-π_BoN ≡ a KL-regularized RLHF problem with a particular reward).
- **Argument for Jeffreys [RAW], Sec. 4.2, Eq. 11:** "Jeffreys^β(p‖q) := (1−β)·KL(q‖p) [Forward KL] + β·KL(p‖q) [Backward KL]. … Notably, when fine-tuning policy p, the forward KL(q‖p) encourages that generations likely under q are also likely under p, thus encouraging a mode-covering behavior. Instead, the reverse KL(p‖q) is well-known to have a mode-seeking effect, steering policy p to produce generations that have a high likelihood according to q (Agarwal et al., 2024). While the forward KL may produce over-spread distributions, the backward KL can lead to policy and entropy collapses. Instead, we empirically show that the Jeffreys divergence inherits the best of both divergences, producing better aligned policies."
- **TERMINOLOGY FLAG (important):** in BOND's Eq. 11, **p is the trained policy and q is the target** — letters swapped relative to our convention. Their "backward KL"/"reverse KL" = KL(policy‖target) = **our reverse KL** (direction agrees; letters don't). "Backward" and "reverse" used interchangeably.
- **Estimation [RAW], Eq. 12:** forward KL(π_BoN‖π) = E_{y∼π_BoN}[log π_BoN(y) − log π(y)] "can be seen as a supervised fine-tuning loss on the Best-of-N samples"; backward KL = E_{y∼π}[log π(y) − log π_BoN(y)], with log π_BoN in closed form via π_BoN(y) = π_ref(y)·p_≤(y)^{N−1}·Σ_{i=1}^{N}[p_<(y)/p_≤(y)]^{i−1} [FETCH; raw-verified up to Eq. 12], p_≤(y) = reward quantile of y under π_ref (Monte-Carlo estimated).

```bibtex
@article{sessa2024bond,
  title={BOND: Aligning LLMs with Best-of-N Distillation},
  author={Sessa, Pier Giuseppe and Dadashi, Robert and Hussenot, L{\'e}onard and Ferret, Johan and Vieillard, Nino and Ram{\'e}, Alexandre and others},
  journal={arXiv preprint arXiv:2407.14622}, year={2024}}
```

### 5.2 Amini, Vieira, Ash, Cotterell — vBoN ("Variational Best-of-N Alignment")
- Link: https://arxiv.org/abs/2407.06057
- [FETCH, abstract]: "…derive the distribution induced by the BoN algorithm. Then propose to fine-tune the language model to minimize backward KL divergence to the BoN distribution." … "in the controlled generation task, vBoN appears more frequently on the Pareto frontier of reward and KL divergence compared to other alignment methods." **Flag:** "backward KL" = KL(π‖π_BoN) = our reverse KL. The pure-reverse-KL counterpart to BOND's Jeffreys choice.

```bibtex
@article{amini2024variational,
  title={Variational Best-of-N Alignment},
  author={Amini, Afra and Vieira, Tim and Ash, Elliott and Cotterell, Ryan},
  journal={arXiv preprint arXiv:2407.06057}, year={2024}}
```

---

## 6. GRPO: the k3 estimator in the loss

### 6.1 Shao et al. 2024 — DeepSeekMath (GRPO)
- Links: https://arxiv.org/abs/2402.03300 , https://ar5iv.labs.arxiv.org/html/2402.03300
- **Math (Eq. 3):** GRPO objective = per-token clipped-ratio surrogate with group-relative advantage Â_{i,t}, minus β·D_KL[π_θ‖π_ref] **as a separate loss term inside the per-token average** (not in the reward).
- **Reward-vs-loss placement [RAW], Sec. 4.1:** "Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of Â_{i,t}."
- **The estimator [RAW], Eq. 4:** "we estimate the KL divergence with the following unbiased estimator (Schulman, 2020): D_KL[π_θ‖π_ref] = π_ref(o_{i,t}|q,o_{i,<t})/π_θ(o_{i,t}|q,o_{i,<t}) − log[π_ref(o_{i,t}|q,o_{i,<t})/π_θ(o_{i,t}|q,o_{i,<t})] − 1, which is guaranteed to be positive." This is Schulman's k3 with r = π_ref/π_θ, per token, samples from (old) policy. β = 0.04 for DeepSeekMath-RL [FETCH, Sec. 4.2]. Paper calls it "unbiased estimator" of KL *value*; it is then differentiated as a loss — see Sec. 7 for why that changes what is optimized.

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and others},
  journal={arXiv preprint arXiv:2402.03300}, year={2024}}
```

### 6.2 DeepSeek-R1 (2501.12948)
- [FETCH, Sec. 2.2.1]: R1 paper restates GRPO objective with −β D_KL(π_θ‖π_ref) (Eq. 1) and the k3 formula (Eq. 2). No explicit statement about β's value or dropping KL for R1-Zero. (Community/DAPO claims about removing KL are NOT sourced from the R1 paper; see ProRL quote for the published "recent works have argued for the removal of the KL penalty".)

```bibtex
@article{deepseekr1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={{DeepSeek-AI} and Guo, Daya and others},
  journal={arXiv preprint arXiv:2501.12948}, year={2025}}
```

---

## 7. KL estimators and their gradients

### 7.1 Schulman 2020 — "Approximating KL Divergence" (blog)
- Link: http://joschu.net/blog/kl-approx.html (fetched raw)
- **Setup [RAW]:** "KL[q, p] = Σ_x q(x) log(q(x)/p(x)) = E_{x∼q}[log(q(x)/p(x))]" — samples from q, r = p(x)/q(x). With q = policy, p = reference, KL[q,p] = our **reverse KL**; k1/k2/k3 estimate the *value*, not the gradient (post explicitly about diagnostics: "This is a reasonable choice if KL is just being used as a diagnostic, as is often the case in reinforcement learning." [RAW]).
- **Estimators [RAW]:** k1 = −log r ("unbiased… has high-variance, as it's negative for half of the samples, whereas KL is always positive"); k2 = ½(log r)² ("lower variance but is biased"; expectation is an f-divergence with f''(1)=1, hence "low bias" near p≈q); k3 = (r−1) − log r, from control variate −log r + λ(r−1) with λ=1: "since log is concave, log(x) ≤ x−1… the expression above is guaranteed to be positive. It measures the vertical distance between log(x) and its tangent." Bregman view; general recipe f(r) − f'(1)(r−1); for KL[p,q] (other direction): r log r − (r−1) [RAW].

```bibtex
@misc{schulman2020klapprox,
  title={Approximating KL Divergence},
  author={Schulman, John},
  howpublished={\url{http://joschu.net/blog/kl-approx.html}}, year={2020}}
```

### 7.2 Tang & Munos 2025 — "On a few pitfalls in KL divergence gradient estimation for RL" (arXiv 2506.09477)
- Links: https://arxiv.org/abs/2506.09477 , https://arxiv.org/html/2506.09477v1
- **Abstract [RAW]:** "The first major pitfall is to differentiate through the KL estimate as loss functions to minimize KL divergence. We show that such implementations are generally incorrect and do not produce the desired KL gradient. Secondly, we show that some implementations do not account for the sequential nature of the estimation problem and produce a partial gradient at best."
- **Derivation conclusions [RAW]:** target is KL(π, π_ref) (comma notation, policy first = our reverse KL).
  - k1-as-loss: "E_{y∼π}[∇KL̂_vanilla] = E_{y∼π}[∇log π(y)] = 0. In other words, using such a gradient estimate amounts to adding a zero-mean noise to the aggregate gradient estimate." (Eq. 2)
  - **k3-as-loss:** "E[∇KL̂_var-reduced] = ∇KL(π_ref, π) ≠ ∇KL(π, π_ref). Very importantly, the positions of the two distributions are reversed compared to the target KL divergence" (Eq. 3) — **differentiating k3 through π yields the FORWARD-KL gradient** ∇D_KL(π_ref‖π). Also [RAW]: "when combined with reward maximization, the variance reduced estimate converges to a different policy than the target optimal policy. It is important to understand that the regularization effect of the variance reduced estimate is completely incidental! The regularizer comes from the control variate term of the estimate…"
  - k2-as-loss: produces the unbiased reverse-KL gradient [FETCH: "the squared KL divergence estimate, despite being a biased estimate itself, produces an unbiased KL gradient estimate when differentiated"].
  - Correct reverse-KL gradient needs pathwise + score-function parts: ∇KL = E[∇KL̂] + E[KL̂ · ∇log π(y)] [FETCH]; token-level implementations "produce a partial gradient at best" [RAW, abstract].

```bibtex
@inproceedings{tang2025pitfalls,
  title={On a few pitfalls in KL divergence gradient estimation for RL},
  author={Tang, Yunhao and Munos, R{\'e}mi},
  booktitle={ICML (per header in preprint)}, year={2025}, note={arXiv:2506.09477}}
```

### 7.3 Zhang et al. 2025 — RPG ("On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning", arXiv 2505.17508)
- Links: https://arxiv.org/abs/2505.17508
- **Abstract [FETCH, near-verbatim]:** "KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator (k1/k2/k3), is scattered across the literature… under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? … RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used k3 penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term…" Authors: Yifan Zhang, Yifeng Liu, Huizhuo Yuan, Yang Yuan, Quanquan Gu, Andrew Chi-Chih Yao.

```bibtex
@article{zhang2025rpg,
  title={On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning},
  author={Zhang, Yifan and Liu, Yifeng and Yuan, Huizhuo and Yuan, Yang and Gu, Quanquan and Yao, Andrew Chi-Chih},
  journal={arXiv preprint arXiv:2505.17508}, year={2025}}
```

### 7.4 Liu et al. 2025 — "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization" (arXiv 2510.01555)
- Links: https://arxiv.org/abs/2510.01555 , https://openreview.net/forum?id=keCnsHtION
- **Claims [FETCH]:** "we prove that the conventional 'k1 in reward' (like PPO) is the principled loss for Reverse KL (RKL) regularization. We further establish a key finding: under on-policy conditions, the 'k2 as loss' formulation is, in fact, gradient-equivalent to 'k1 in reward'." And: "the recently adopted 'k3 as loss' (like GRPO) is merely a first-order, biased approximation of the principled loss." On GRPO: "…exemplifying the flawed practice of transferring value estimation principles to loss design."

```bibtex
@article{liu2025rethinking,
  title={Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization},
  author={Liu, Kezhao and Liu, Jason Klein and Chen, Mingtao and Liu, YiMing},
  journal={arXiv preprint arXiv:2510.01555}, year={2025}}
```

### 7.5 Practitioner notes (secondary but precise)
- Xihuai Wang, "Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness" (Dec 2025): https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html — [FETCH] "Directly using k3 (without ρ): E_{q_θ}[∇k3] = ∇D_KL(p ∥ q_θ) (forward KL), i.e., the wrong direction for reverse-KL regularization." Also: k1-in-reward and k2-as-loss KL-gradient terms are "the same random variable — identical sample by sample". "Value unbiasedness does not imply gradient correctness."
- Xiaobo Yang, "Gradient Estimation of KL Divergence in LLM RL": https://xiaobo-yang.github.io/posts/kl_grad/ (found, not fetched — lead).

---

## 8. Empirical effects of reverse-KL-regularized RLHF

### 8.1 Kirk et al. — "Understanding the Effects of RLHF on LLM Generalisation and Diversity" (ICLR 2024, arXiv 2310.06452)
- Links: https://arxiv.org/abs/2310.06452
- **Setup:** LLaMA-7B (+ OPT sizes); summarization (TL;DR→CNN/DailyMail) and instruction following (AlpacaFarm); per-input diversity = avg diversity of K=16 samples per input; across-input diversity = diversity of one sample across N inputs [RAW, Eqs. 2–3].
- **Findings [RAW, Sec. 1 summary]:** "RLHF improves in-distribution performance… and also OOD performance in comparison to SFT. • However, RLHF substantially decreases the diversity of outputs sampled for a given input compared to SFT. • Even when sampling outputs for different inputs, RLHF produces less diverse text on some metrics…" (per-input drop is the large one; across-input differences "much smaller" [RAW, Fig. 6 caption]).
- **KL-coefficient ablation (direction-relevant negative result) [RAW], Sec. 6.3:** "Motivated by the fact that the KL penalty coefficient (see Eq. 1) encourages the RLHF policy to stay closer to the SFT policy, we investigate whether adjusting this coefficient trades off between generalisation and diversity. The results show that this does not work – increasing the KL penalty coefficient leads to a drop in performance as expected, but also to a drop in per-input diversity, rather than a gain." Turning up the *reverse* KL penalty does not restore diversity.
- [FETCH, Sec. 6.2]: "we believe that this is the first rigorous empirical demonstration of across-input mode collapse emerging from RLHF training specifically."

```bibtex
@inproceedings{kirk2024understanding,
  title={Understanding the Effects of RLHF on LLM Generalisation and Diversity},
  author={Kirk, Robert and Mediratta, Ishita and Nalmpantis, Christoforos and Luketina, Jelena and Hambro, Eric and Grefenstette, Edward and Raileanu, Roberta},
  booktitle={ICLR}, year={2024}, note={arXiv:2310.06452}}
```

### 8.2 OpenAI 2023 — GPT-4 Technical Report (calibration)
- Links: https://arxiv.org/abs/2303.08774
- **[RAW], Limitations:** "Interestingly, the pre-trained model is highly calibrated (its predicted confidence in an answer generally matches the probability of being correct). However, after the post-training process, the calibration is reduced (Figure 8)." **Figure 8 caption [RAW]:** "Left: Calibration plot of the pre-trained GPT-4 model on a subset of the MMLU dataset. … Right: Calibration plot of the post-trained GPT-4 model on the same subset of MMLU. The post-training hurts calibration significantly."
- **Caveat:** attributed to "the post-training process" (RLHF pipeline), not the KL direction. (f-DPO's ECE finding connects calibration to divergence-constraint choice.)

```bibtex
@article{openai2023gpt4,
  title={GPT-4 Technical Report},
  author={{OpenAI}},
  journal={arXiv preprint arXiv:2303.08774}, year={2023}}
```

### 8.3 Alignment tax — see InstructGPT (1.3).

### 8.4 Entropy collapse in RLVR and the KL term — ProRL (arXiv 2505.24864)
- **[RAW], Sec. 2:** "We then address key challenges in prolonged RL training, such as entropy collapse and instability, by introducing a KL divergence penalty and periodic resets of the reference policy." **Sec. 2.3.1 [RAW]:** "While DAPO and temperature adjustment help slow entropy collapse, we find that explicit regularization via a KL divergence penalty provides a stronger and more stable solution. Specifically, we incorporate a KL penalty between the current policy π_θ and a reference policy π_ref: L_KL-RL(θ) = L_GRPO(θ) − β D_KL(π_θ‖π_ref). (4) This penalty not only helps maintain entropy but also serves as a regularizer… Recent works [4,7,5,18] have argued for the removal of the KL penalty, citing that models naturally diverge during training on chain-of-thought reasoning tasks."
- **Relevance:** in RLVR the *reverse* KL-to-reference is deployed as an entropy-*preserving* device (reference has higher entropy than the collapsing policy), with periodic reference resets — mirror image of the alignment-diversity critique. Ties entropy collapse to presence/absence of KL, not its *direction*.

```bibtex
@article{liu2025prorl,
  title={ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models},
  author={Liu, Mingjie and Diao, Shizhe and Lu, Ximing and Hu, Jian and Dong, Xin and Choi, Yejin and Kautz, Jan and Dong, Yi},
  journal={arXiv preprint arXiv:2505.24864}, year={2025}}
```

### 8.5 Forward-KL / alternative-divergence regularizers proposed to fix diversity
- f-DPO, f-DPG, BOND/Jeffreys, f-PO (above).
- Theory: **Wu, Shi, Yang, Shen, "f-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses"** (arXiv 2605.06977, May 2026): [FETCH, abstract] "While most existing approaches rely on the reverse KL-regularization, recent empirical studies have begun exploring alternative divergences (e.g., forward KL, chi-squared) as regularizers in RLHF. However, a unified theoretical understanding of general f-divergence regularization remains under-explored. … establishing … the first performance bounds for online RLHF under general f-divergence regularization."
- **Excluded as not divergence-motivated:** Diverse Preference Optimization (DivPO, Lanchantin et al., arXiv 2501.18101) — a *pair-selection* method; cite only as evidence of the diversity problem.

### 8.6 Opinion narrowing (secondhand)
- Santurkar et al. 2023 ("Whose opinions do language models reflect?") cited *by f-DPO* as observing RLHF-under-RKL narrows political views. Attribution to reverse KL is f-DPO's framing; Santurkar's paper not fetched (lead).

---

## 9. Reward hacking / KL as insufficient safeguard

### Gao, Schulman, Hilton 2023 — "Scaling Laws for Reward Model Overoptimization" (arXiv 2210.10760, ICML 2023)
- **KL as distance measure [RAW], Sec. 1:** gold reward "as a function of the Kullback–Leibler divergence from the initial policy to the optimized policy KL := D_KL(π ‖ π_init)"; "because it is a quadratic metric of distance (Bai et al., 2022, Section 4.3), we will define d := sqrt(D_KL(π‖π_init))". Policy-first ⇒ reverse KL as the *measuring stick* of optimization pressure.
- **KL penalty as safeguard — negative result [RAW], Sec. 1 bullet:** "**KL penalty ineffectiveness.** In our reinforcement learning setup, using a KL penalty increases the proxy reward model score that can be achieved for a given KL divergence, but this does not correspond to a measurable improvement in the gold RM score–KL_RL frontier. However, we note this result could be particularly sensitive to hyperparameters." Also [RAW]: "KL penalty for all RL experiments is set to 0 except for in section 3.6."
- **Direction note:** never contrasts KL directions; claim is the (reverse-)KL penalty changes *where along the KL axis* you land, not the gold-reward-vs-KL frontier.

```bibtex
@inproceedings{gao2023scaling,
  title={Scaling Laws for Reward Model Overoptimization},
  author={Gao, Leo and Schulman, John and Hilton, Jacob},
  booktitle={ICML}, year={2023}, note={arXiv:2210.10760}}
```

---

## Note: "KL in the reward" vs "KL in the loss" — what sources actually say

- **DeepSeekMath [RAW]:** KL in loss "instead of adding KL penalty in the reward … avoiding complicating the calculation of Â_{i,t}" — motivation is implementation simplicity, no equivalence claim.
- **Tang & Munos [RAW/FETCH]:** true reverse-KL gradient has pathwise + score-function parts (E[KL̂·∇log π]); differentiating a KL *value estimate* as a loss drops/garbles the score-function part (k1→0; k3→forward-KL gradient), token-level losses give "a partial gradient at best". KL-in-reward routes the penalty through the policy-gradient/advantage channel = exactly the score-function part.
- **Liu et al. 2510.01555 [FETCH]:** "'k1 in reward' (like PPO) is the principled loss for Reverse KL"; on-policy "'k2 as loss' … gradient-equivalent to 'k1 in reward'"; "'k3 as loss' (like GRPO) is merely a first-order, biased approximation."
- **RPG 2505.17508 [FETCH]:** derives off-policy importance weighting needed per estimator/direction; "identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term"; "the widely-used k3 penalty is exactly the unnormalized KL."
- **Xihuai Wang blog [FETCH]:** k1-in-reward and k2-as-loss KL-gradient contributions are "the same random variable — identical sample by sample" (on-policy).
- One-line summary (sourced, not derived): *on-policy, reward-channel k1 ≡ loss-channel k2 for reverse KL; loss-channel k3 optimizes the forward KL in expectation; equivalences break off-policy without importance-weight corrections.*

---

# Direct answers

**(a) Role of reverse KL in RLHF, and who justifies the direction how.** In the canonical pipeline (Ziegler → Stiennon → InstructGPT → Bai), reverse KL D_KL(π‖π_ref) is a *regularizer added to the reward* channel of PPO, and the direction is never argued for against alternatives — it is the direction that is (i) estimable from on-policy samples with only log-prob access, and (ii) justified operationally: keep π "from moving too far from the range where r is valid" (Ziegler), "entropy bonus … deterring it from collapsing to a single mode" plus keeping outputs in-distribution for the reward model (Stiennon), "mitigate over-optimization of the reward model" (InstructGPT), stabilization at tiny λ_KL that "might actually be wholly unnecessary" (Bai). The principled justification of the *direction* comes later, from the variational view: Korbak et al. show the KL-regularized objective equals maximizing an ELBO, i.e. minimizing the **reverse** KL D_KL(π_θ‖π*) to the tilted posterior π* ∝ π_ref·exp(r/β); DPO inherits exactly this RKL-constrained objective in closed form; f-DPO adds that reverse KL is *algebraically special* (partition function cancels; other f's need the 0∉dom(f′) trick).

**(b) Empirical effects reported.** Reward improves roughly linearly in sqrt(KL) (Bai; Gao). Diversity: RKL-based RLHF substantially reduces per-input output diversity and shows across-input mode collapse signs (Kirk); RKL instances of distribution matching produce "low diversity but high quality … 'mode collapse'" and initial entropy crashes (f-DPG), lowest predictive entropy among divergences (f-DPO), "policy and entropy collapses" (BOND); RLHF under RKL narrows expressed opinion range (Santurkar via f-DPO). Calibration: GPT-4 post-training "hurts calibration significantly" (Fig. 8) — attributed to post-training generally; f-DPO reports divergence constraint "directly influence[s] ECE." Alignment tax mitigated by pretraining-likelihood mixing, not the KL term (InstructGPT). Counterpoints: increasing the RKL coefficient does *not* recover diversity (Kirk 6.3); in RLVR the RKL-to-reference penalty (with resets) is used to *fight* entropy collapse (ProRL) — RKL is entropy-preserving relative to unregularized reward maximization, yet mode-seeking relative to forward-KL/distribution-matching alternatives.

**(c) Criticisms — divergence vs estimator.** (1) *Of the divergence:* reverse KL is mode-seeking; with imperfect policy class/optimization it collapses onto high-probability regions of the target, sacrificing coverage/diversity (GDC lineage, f-DPG, f-DPO, BOND); Korbak et al. criticize the *unregularized* objective more radically while endorsing the RKL-regularized one as Bayesian inference. Gao et al.: as a *safeguard* the RKL penalty does not improve the gold-reward-vs-KL frontier. (2) *Of estimators/gradients:* k1/k2/k3 fine as value estimators (Schulman), but differentiating value estimates as losses is "generally incorrect" (Tang & Munos): k1-loss = zero-mean noise; k3-loss (GRPO Eq. 4) has expected gradient ∇KL(π_ref, π) — forward-KL gradient whose "regularization effect … is completely incidental"; k2-loss (or k1-in-reward) gives correct reverse-KL gradient on-policy; off-policy all need importance-weight corrections (RPG). So some deployed "reverse-KL" systems (GRPO-style k3-in-loss) are, in expected-gradient terms, doing forward-KL regularization.

**(d) Alternative divergences introduced, and why.** Forward KL / distribution matching to explicit target (GDC via DPG; f-DPG's KL-DPG): mass-covering, avoids degeneration-by-truncation, but high-variance gradients, weaker reward. JS and TV (f-DPG): JS "performing best overall." f-divergence families in preference optimization (f-DPO: JSD, FKL, α-divergences; f-PO unifies DPO/EXO). Jeffreys (BOND): "the forward KL may produce over-spread distributions, the backward KL can lead to policy and entropy collapses," combination "inherits the best of both"; target = Best-of-N distribution (vBoN keeps pure reverse KL to same target). Theory: general f-divergence-regularized online RLHF bounds (Wu et al. 2605.06977).

---

# Leads not verified

- Ziegler Table 10 contents (behavior without KL penalty).
- Bai Sec. 4.3 detailed sqrt-KL derivation text.
- Korbak appendix (Eqs. 11–17); Todorov/Kappen not found — cite KL-control lineage via Jaques et al. 2017 (unfetched) and Levine 2018.
- Jaques et al. 2017 (Sequence Tutor) — direction of its KL-control term not verified first-hand.
- f-DPG Sec. 3.4 sentence (RLHF ≡ RKL minimization of EBM target) — [FETCH] only.
- f-DPO Table 2 numbers — [FETCH] only.
- DeepSeekMath β = 0.04 — [FETCH] only.
- Kirk "first rigorous empirical demonstration of across-input mode collapse" — [FETCH] only.
- Santurkar et al. 2023 (arXiv 2303.17548) — original not fetched.
- DeepSeek-V3.2 "ρk3" importance-ratio-corrected KL — search results only; no primary source fetched.
- Xiaobo Yang blog — found, not fetched.
- "Future-KL Regularized GRPO" (arXiv 2601.10201); "Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization" (arXiv 2605.11491) — not fetched; check KL-direction ties.
- EXO ("Towards Efficient Exact Optimization of Language Model Alignment," arXiv 2402.00856) — referenced by f-PO as reverse/forward-KL comparison point in offline alignment; not fetched — worth adding.
- vBoN closed form (log-CDF reward interpretation) — body not fetched.
