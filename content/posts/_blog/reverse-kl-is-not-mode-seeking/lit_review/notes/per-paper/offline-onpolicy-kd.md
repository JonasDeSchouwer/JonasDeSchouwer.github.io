# Rang, Bi et al. 2025 — Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation

**arXiv 2509.26497v1** (30 Sep 2025, primary class cs.CV). Miao Rang†, Zhenni Bi†, Hang Zhou, Hanting Chen, An Xiao, Tianyu Guo, Kai Han*, Xinghao Chen#, Yunhe Wang# (Huawei Noah's Ark Lab; † equal contribution, * project leader, # corresponding). Model card: `https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-1B-V1.1` (not inspected). No training code released. Sources: PDF v1 text via pypdf, HTML v1 (`arxiv.org/html/2509.26497`) for equations/table cells/section ids, and Fig. 1 (`2509.26497v1/fig/pipeline-new.PNG`) read visually, 2026-09-14.

```bibtex
@article{rang2025revealing,
  title   = {Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation},
  author  = {Miao Rang and Zhenni Bi and Hang Zhou and Hanting Chen and An Xiao and Tianyu Guo and Kai Han and Xinghao Chen and Yunhe Wang},
  journal = {arXiv preprint arXiv:2509.26497},
  year    = {2025}
}
```

## Scope verdict

IN SCOPE (extreme staleness, the "generate once, train for the whole stage" case). The behaviour policy is the student's own *earlier* checkpoint: the SFT-finished student makes "a single inference pass over the training corpus" (§2 Knowledge Distillation), the teacher labels that fixed corpus offline, and the KD stage then trains on it. The teacher is a fixed 7B model, never updated (§3.3 bullet 2). The starred flagship run refreshes the generator twice mid-training (§4.3 Distillation Strategy, Table 5 caption), i.e. the multi-round "regenerate every N steps" variant, so the paper contains exactly two staleness levels (1 generation vs 3 generations) with numbers. This is the earlier, CE+KL-soft-logit ancestor that Lightning OPD (2604.13010) cites as ref [26].

## Tier

- **C** — two staleness levels side by side with numbers: "Distillation by Student" (one generation, AVG 62.28) vs "Distillation by Student*" (generator refreshed twice, AVG 63.43), Table 5 rows 4–5. The accompanying explanation is a single asserted sentence ("creates a dynamic, self-correcting loop", §4.3 p3), not an analysis or measurement — so C, not B. There is **no online/sync OPD row anywhere in the paper**, so the headline offline-vs-online comparison is claimed but never run.
- **D** — several quotable statements contrasting offline-on-policy with true online on-policy KD (§2 p3, §3.3 p2); see Quotes.
- **E** — this is a production tech report: openPangu Embedded-1B-KD is a shipped Huawei edge model ("All the training and deployment are conducted on Ascend hardware", §5), and offline on-policy KD is the post-training recipe that produced it.
- **F** — offline on-policy KD is one of the paper's two headline contributions (title, abstract, Fig. 1, §3.3, §4.3).
- Not A: no argument, proof, or measurement about which KL estimator tolerates staleness; the paper never discusses the drift between the generating checkpoint and the current student at all.
- Not B (explanation is an assertion, no numbers behind it), not G (async *is* the main point).

## Setting

- Student: openPangu Embedded-1B (base model from Pangu Embedded, Chen et al. 2025), post-trained to openPangu Embedded-1B-SFT then openPangu Embedded-1B-KD (§1, §3).
- Teacher: "a powerful teacher model(e.g., a 7B parameter model sharing the same tokenizer)" (§3.3 bullet 2). **The teacher is never named** — no model id, no provenance, no eval of the teacher itself.
- Teacher fixed: yes. Only the student's generator is refreshed; the teacher runs as an offline logit annotator.
- Domain/benchmarks (§4.1): MMLU, CMMLU, C-Eval, IF-Eval, CLUEWSC (general); GSM8K, MATH-500 (math); DROP, GPQA-Diamond (reasoning); MBPP, HumanEval (code). Ablation tables 3–4 use an 8-benchmark average (MMLU, CMMLU, CEval, BBH, GSM8K, MATH, MBPP, HumanEval). All evals greedy decoding, 8K decode length (Table 3/4/5 captions).
- Post-training data (§3.1): mixed open-source instruction data + industrial queries + synthetic problems, reasoning:non-reasoning = 3:1, MinHash-LSH dedup + entropy-guided ZIP selection.
- SFT setup (§4.1): two curriculum stages, 10 epochs each, AdamW wd 0.1, grad clip 1.0, max seq len 32K with packing; stage 1 global batch 4M tokens, cosine 2e-5 → 2e-6 with 200-iter warmup; stage 2 batch 2M tokens, cosine 1e-5 → 1e-6.
- KD setup: "the distillation learning rate kept consistent with that used in SFT" (§4.3 KD Loss Weight). **Number of KD epochs/steps is not stated anywhere.**
- Framework/hardware: Ascend (§5). No framework named, no code.

## Async mechanism

- What is stale: the *entire* rollout corpus D_s. Behaviour policy = the SFT-finished student (the KD stage's own step-0 checkpoint); current policy = the student after k KD updates. "The process commences with our SFT-trained student model, which performs inference on the queries from the original training dataset. This generates an intermediate on-policy dataset, D_s" (§3.3, S3.I1.i1.p1.1). Fig. 1 draws this as a straight arrow from "Stage2 Fast Response" into "Offline On-policy Distillation" with no refresh loop.
- The teacher targets are stale only in the same sense: because the teacher is fixed and it is conditioned on the stored student prefixes, the top-10 teacher logits are *exact* for the data actually trained on. All staleness lives in the sampling distribution of the prefixes, none in the supervision.
- Refresh schedule: the default variant regenerates **never** (one pass). The starred flagship variant regenerates **twice** during the KD stage: "By periodically updating the student model with its latest parameters during the training process—in our case, twice—and re-generating responses for subsequent distillation" (§4.3 p3), and Table 5's caption: "The asterisk (*) denotes that the student model is updated twice with its latest parameters during training for response generation." So 1 vs 3 generation rounds (initial + 2 refreshes); the paper does not say at which steps the refreshes happen.
- How staleness is measured: **not at all**. No importance-weight statistics, no KL(π_θ‖π_gen), no drift plot, no step/token count for how long a batch of rollouts is reused, no epoch count for the KD stage. The only handle on "off-policyness" is the qualitative claim that the student-generated corpus "effectively simulates an on-policy data distribution" (§2 p3).
- Range studied: two points (never regenerated, regenerated twice). No intermediate schedule, no buffer, no fresh/stale mixing.

## Objective

Forward KL from the fixed teacher to the student, truncated to the **teacher's top-k** (default k = 10), added to a cross-entropy term. In the sweep's vocabulary: **`fkl`, teacher-top-k (k = 10), on stale student-generated text, with a CE auxiliary at weight 1 − λ_KD = 0.1**. Not `rkl`, not `rkl_pg`.

Composite loss (§3.3 "Training with a Composite Loss Function", displayed equation S3.Ex1, **unnumbered**):

```latex
L_{total}=(1-\lambda_{KD})\cdot L_{CE}+\lambda_{KD}\cdot L_{KD}
```

KD term (displayed equation S3.Ex2, **unnumbered**):

```latex
D_{KL}(P\parallel Q)=\sum_{c\in C}P(c)\log\frac{P(c)}{Q(c)}
```

with, verbatim, "a probability distribution Q (from the student) ... a reference probability distribution P (from the teacher)" (§3.3 p1.2). So P = teacher, Q = student, i.e. KL(teacher ‖ student) = **forward KL** under this review's convention (reverse KL = KL(student ‖ teacher)). **Direction flag:** the paper calls the divergence "asymmetric" but never says forward or reverse, and its prose gloss ("quantifies how a probability distribution Q (from the student) differs from a reference ... P") reads like the reverse direction; the formula is unambiguous and is the forward one. The sum is written over "a vocabulary of classes C", but the stored supervision is only "the logits for the top-k most probable tokens" of the teacher (§3.3 bullet 2), so C is the teacher's top-10 in practice — the paper never states how the truncated distribution is renormalised or how the student's mass outside the top-10 is handled.

Hyperparameters (§4.3, Tables 3–4, both tuned on the *label* variant, not the student variant): λ_KD = 0.9 best (Table 3: 0.5 → 53.94, 0.6 → 54.07, 0.7 → 53.88, 0.8 → 54.43, **0.9 → 55.29**, 1.0 → 54.44 — pure KD is worse than KD+CE); top-k = 10 best (Table 4: 5 → 54.78, **10 → 55.29**, 15 → 54.24, 20 → 54.40), and "training time remains almost identical across different k values".

## Off-policy correction

**None.** No importance ratio, no per-token or per-sequence weights, no clipping or truncation, no drift mask. The KD stage is literally an SFT loop with an extra loss term: "Compared to the standard SFT process, our method introduces only an additional Knowledge Distillation (KD) loss term" (§3.3 p2). The only equations in the paper are the two above; neither contains a ratio. The paper never acknowledges that training on stale rollouts is a bias at all.

## Results: async vs sync

**No sync (online OPD) baseline exists anywhere in the paper.** The efficiency advantage over online OPD is asserted ("markedly improves training efficiency over online counterparts without a significant compromise in model performance", §2 p3) with **zero** throughput, wall-clock, or accuracy numbers for an online run.

What the paper does provide is (a) a data-source comparison (off-policy label / off-policy teacher / offline-on-policy student) and (b) one staleness comparison. Table 5, all rows = a Stage-3 KD pass on 'fast-response' data added after the 2-stage SFT, greedy, 8K decode:

| Method (Table 5, §4.3) | Rollout source | MMLU | C-Eval | GSM8K | MATH-500 | DROP | MBPP | HumanEval | **AVG** |
|---|---|---|---|---|---|---|---|---|---|
| Stage2 SFT (no KD) | — | 63.21 | 58.51 | 70.89 | 56.20 | 28.73 | 52.53 | 59.15 | **56.28** |
| Stage3 Distillation by Label | ground-truth labels (off-policy) | 62.68 | 58.83 | 72.10 | 65.20 | 30.24 | 57.20 | 62.80 | **58.79** |
| Stage3 Distillation by Teacher | teacher's own generations (off-policy, seq-KD) | 61.74 | 62.83 | 74.30 | 61.20 | 36.59 | 55.25 | 60.37 | **59.55** |
| Stage3 Distillation by Student | student, generated **once** (stale) | 65.91 | 64.30 | 76.72 | 71.40 | 40.99 | 58.75 | 67.07 | **62.28** |
| Stage3 Distillation by Student* | student, generator refreshed **twice** | 67.28 | 66.55 | 77.33 | 73.80 | 40.95 | 61.09 | 65.85 | **63.43** |

(Full 11-benchmark rows also include CMMLU / IF-Eval / CLUEWSC / GPQA-Diamond; the AVG column is over all 11.)

Readings: offline-on-policy (stale student data, no correction, no refresh) beats both off-policy sources by +2.7 to +3.5 AVG and beats the SFT start by +6.00. Refreshing the generator twice buys a further **+1.15 AVG** (62.28 → 63.43), improving 8 of 11 benchmarks but *decreasing* DROP (40.99 → 40.95), GPQA-Diamond (44.95 → 44.44) and HumanEval (67.07 → 65.85) — i.e. fresher rollouts help but the one-shot-stale version already captures ~84% of the gain over SFT.

Table 1 (§4.1) reports the shipped model: openPangu Embedded-1B-KD (= the Distillation by Student* recipe) AVG 63.43 vs its SFT parent 56.28 and vs Qwen3-1.7B 63.69, Qwen2.5-1.5B 55.10, Qwen3-0.6B 44.93.

Table 6 (§4.4) is a pipeline-ordering ablation and uses only the *label* (off-policy) KD variant, so it says nothing about staleness.

## Explanation offered

The paper's explanation is about *on-policy-ness* (prefix distribution matching), never about *staleness tolerance*. Mechanism claimed: support/distribution mismatch between teacher supervision and the student's own state.

- "This finding highlights the benefit of aligning the teacher's guidance with the student's current output space. By conditioning distillation on the student's response, the teacher provides corrective and refining signals on a distribution immediately relevant to the student's state. This alignment minimizes the distributional mismatch between the two models, creating a more stable learning signal and facilitating more effective knowledge transfer." (§4.3, Distillation Strategy, S4.SS3.SSS0.P0.SPx3.p2.1)
- "Crucially, rather than performing greedy decoding, we preserve the teacher's full predictive distribution by recording the logits for the top-k most probable tokens. This constrained conditioning strategy is the cornerstone of our approach. By compelling the teacher to generate guidance from a probabilistic space already accessible to the student, we effectively minimize the intrinsic distributional divergence between the two models. This facilitates a more stable and efficient knowledge transfer." (§3.3 bullet 2, S3.I1.i2.p1.1)
- Framing of the problem as exposure bias: "A fundamental challenge in applying KD to autoregressive models is the exposure bias phenomenon (Ross et al., 2011)—a discrepancy between the training distribution, which is conditioned on ground-truth sequences, and the inference distribution, where the model conditions on its own generated outputs." (§2 p3)
- Why *stale* student data suffices — the only sentence that touches it: "This method effectively simulates an on-policy data distribution, thereby alleviating exposure bias while circumventing the costly iterative loop inherent to true on-policy methods." (§2 p3). "Simulates" is the whole argument; there is no bound, no drift measurement, and no discussion of when the simulation breaks down.
- Why regenerating helps: "This creates a dynamic, self-correcting loop that maximizes the efficiency of knowledge transfer and significantly boosts the final performance of the student model." (§4.3 p3)
- A separate, non-accuracy motivation for going offline: data control. "Unlike the online version, which cannot dynamically adjust and optimize the quality of the generated logits, the offline version allows for selecting high-quality data during the preprocessing stage to generate teacher model guidance information." (§3.3 p2)

## Quotes

Every statement in the paper bearing on offline / stale / on-policy distillation, with section ids:

- §2 p3 (S2.p3.1): "To directly mitigate this issue, on-policy distillation methods have been introduced. These approaches train the student on sequences it generates dynamically, with the teacher providing supervision on these self-generated samples (Agarwal et al., 2024). While this methodology effectively aligns the training and inference distributions, its iterative nature—requiring repeated generation and training steps—incurs substantial computational overhead."
- §2 p3 (S2.p3.1): "To harness the benefits of on-policy training while maintaining offline efficiency, our work introduces a novel offline adaptation of this principle. Our framework is executed in a straightforward two-stage process: first, the student model conducts a single inference pass over the training corpus to generate a complete set of responses. Second, conventional logits-based distillation is performed using this student-generated dataset, with the teacher model providing the supervisory soft labels."
- §2 p3 (S2.p3.1): "This method effectively simulates an on-policy data distribution, thereby alleviating exposure bias while circumventing the costly iterative loop inherent to true on-policy methods. The result is a simple and scalable framework that markedly improves training efficiency over online counterparts without a significant compromise in model performance."
- §3.3 p2 (S3.SS3.p2.1): "To decouple the generation of teacher model guidance data from the training process of the student model, and to allow independent optimization of the training data and process, we design an offline version of on-policy distillation."
- §3.3 p2 (S3.SS3.p2.1): "Unlike the online version, which cannot dynamically adjust and optimize the quality of the generated logits, the offline version allows for selecting high-quality data during the preprocessing stage to generate teacher model guidance information."
- §3.3 p2 (S3.SS3.p2.1): "Compared to the standard SFT process, our method introduces only an additional Knowledge Distillation (KD) loss term."
- §3.3 bullet 1 (S3.I1.i1.p1.1): "The process commences with our SFT-trained student model, which performs inference on the queries from the original training dataset. This generates an intermediate on-policy dataset, D_s, where each sample consists of a query and the corresponding student-generated response. This initial step, which we refer to as distillation by student, ensures that the subsequent teacher guidance is grounded in the student's own output distribution."
- §3.3 bullet 2 (S3.I1.i2.p1.1): "For each position n in a student-generated sequence, the teacher model is conditioned on the prefix of the preceding n−1 tokens to predict the distribution for the n-th token."
- §4.3 Distillation Strategy p1 (S4.SS3.SSS0.P0.SPx3.p1.1): "The third strategy is an offline on-policy approach, ”Distillation by Student”, where the teacher provides target logits based on the student model's own generated response."
- §4.3 p2 (S4.SS3.SSS0.P0.SPx3.p2.1): "the results demonstrate the clear superiority of the ”Distillation by Student” strategy, which achieves a more substantial performance increase of 6%."
- §4.3 p3 (S4.SS3.SSS0.P0.SPx3.p3.1): "By periodically updating the student model with its latest parameters during the training process—in our case, twice—and re-generating responses for subsequent distillation, the model's accuracy is further improved to 63.43%."
- Table 5 caption (S4.T5): "The asterisk (*) denotes that the student model is updated twice with its latest parameters during training for response generation."
- Abstract: "Our post training pipeline consists of curriculum-based supervised fine-tuning (SFT) and offline on-policy knowledge distillation."

## Not in the source

- **No online/sync OPD baseline at all** — the central claim ("markedly improves training efficiency over online counterparts without a significant compromise in model performance", §2 p3) has no supporting run, no accuracy number, and no throughput/wall-clock number. The paper's "offline vs online" comparison is rhetorical.
- **No staleness metric of any kind**: no importance-ratio statistics, no KL between the generating and current student, no count of how many optimizer steps or epochs a rollout batch is reused for. The KD stage's epoch/step count is never stated, so the *magnitude* of the staleness in the winning run cannot be recovered from the paper.
- **No off-policy correction, and no acknowledgement that one might be needed.** No ratio, clipping, or masking appears in either equation.
- **Teacher never identified** ("e.g., a 7B parameter model sharing the same tokenizer", §3.3) — so the teacher–student gap cannot be calibrated against other papers.
- Truncation semantics unspecified: with only the teacher's top-10 logits stored, the paper does not say whether P is renormalised over the top-10, whether the student's Q is renormalised over the same support, or how the residual mass is treated. Equation S3.Ex2 is written over the full vocabulary C, which contradicts the top-k storage.
- The CE term's targets are unspecified for the student variant: on student-generated sequences, "standard supervised learning" (§3.3) would mean CE against the student's own sampled tokens (self-reinforcement), but the paper never says whether L_CE uses the ground-truth response or the student's generated one.
- Both KD hyperparameters (λ_KD = 0.9, top-k = 10) were tuned on the **label** (off-policy) variant "to accelerate the experimental cycle" (§4.3) and then reused for the student variant without re-tuning.
- Refresh schedule for the starred run is unspecified (which steps, what fraction of training between refreshes), and no 1-refresh / 3-refresh / N-refresh sweep is run — only 0 vs 2.
- No seeds, no error bars, no repeated runs; all differences are single-run greedy-decode numbers.
- No ablation of the KD direction (forward vs reverse KL), no sampled-token/policy-gradient variant, and no full-vocab variant.
- No cost accounting for the offline phase (teacher forward passes over the whole corpus, top-10 logit storage size) against the online alternative.
