# Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist & Bachem 2024 — On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

**arXiv 2306.13649v3** (17 Jan 2024; ICLR 2024). Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem (Google DeepMind; Mila; Univ. of Toronto). Sources: PDF `https://arxiv.org/pdf/2306.13649` (curl + pypdf text; pages 4 and 8 rendered with pymupdf to read Algorithm 1 and Figures 7–8, 2026-09-14).

```bibtex
@inproceedings{agarwal2024gkd,
  author    = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos and Matthieu Geist and Olivier Bachem},
  title     = {On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  eprint    = {2306.13649},
  archivePrefix = {arXiv}
}
```

## Async / off-policy (sweep 2026-09-14)

## Tier

- **none** for the async question. GKD regenerates student samples from the *current* student at every training step (Algorithm 1, line 6, inside the `for each step k` loop) and takes exactly one gradient step per batch (line 10). Nothing in the paper is stale, replayed, buffered, pipelined, or reused across updates.
- Explicitly **not** a staleness result: the λ knob mixes fresh student rollouts with a **fixed dataset of ground-truth or teacher-generated sequences**, not with the student's own earlier rollouts. "For output sequences, GKD uses a mixture of fixed dataset, either teacher-generated or ground-truth, and on-policy student-generated sequences." (§3.1). The λ < 1 arm is therefore off-policy *w.r.t. the teacher/data distribution*, which is the SFT axis, not the staleness axis.
- Worth keeping in the review anyway as the **origin of the "student data fraction" framing** that async work reuses, and because the λ sweep is the closest published thing to a "fraction of on-policy data" dose-response curve (Figures 7, 8) — with the caveat above stated every time it is cited.

## Setting

- Students: T5-Small (77M), T5-Base (250M), T5-Large (800M), and FLAN-T5 variants; all initialized by supervised FT (App. A.2). Teacher: supervised-FT T5-XL (~3B) / FLAN T5-XL.
- Teacher fixed: **yes** throughout.
- Domains/benchmarks: XSum (ROUGE-2), WMT en→de (BLEU), GSM8K with 4-shot CoT (accuracy w/ calculator), and task-agnostic instruction tuning on FLAN (MMLU, BBH). Also self-distillation on GSM8K (App. A.1, Fig. A.11).
- Framework/code: T5/FLAN-T5 checkpoints, Adafactor optimizer (App. A.2). No code release stated. Batch sizes 32 (XSum, WMT), 128 (FLAN); 40K–100K training steps.

## Async mechanism

**None.** Algorithm 1 (§3.1, p. 4) verbatim:

```
1: Given: Teacher model p_T, Student Model p_S^θ, Dataset (X, Y) containing (input, output) pairs
2: Hyperparameters: Student data fraction λ ∈ [0, 1], Divergence D, Learning rate η
3: for each step k = 1, . . . , K do
4:    Generate a random value u ∼ Uniform(0, 1)
5:    if u ≤ λ then
6:        Sample inputs x from X and generate outputs y ∼ p_S^θ(·|x) to obtain B = {(x_b, y_b)}_{b=1}^B
7:    else
8:        Sample batch of inputs and outputs from (X, Y) to obtain B = {(x_b, y_b)}_{b=1}^B.
9:    end if
10:   Update θ to minimize L_GKD: θ ← θ − η (1/B) Σ_{(x,y)∈B} ∇_θ D(p_T ‖ p_S^θ)(y|x)
11: end for
```

Line 6 samples from `p_S^θ` — the *current* θ — inside the step-k loop, and line 10 applies one update to that same batch. Staleness is therefore identically zero and the number of gradient steps per rollout batch is exactly one. There is no buffer, no epoch loop, no lag parameter, and no "off-policyness" metric anywhere in the paper.

The paper is explicit that the student distribution is expected to move under it: "Moreover, as the student evolves during training, the data it generates also improves in quality." (§3.1) — i.e. freshness is treated as a feature, never as a cost to be amortized.

The one place the paper acknowledges the price of freshness (App. A.2, "Computational cost of GKD"):

> "On GSM8K, the computational overhead from student sampling is approximately 1.8×, 2× and 2.2× compared to sampling from a fixed dataset of outputs, for a student-teacher ratio of 38×, 12× and 3.8×."

> "Overall, performance benefits from on-policy GKD might be worth the compute cost, especially when combined with RLHF."

This 1.8–2.2× is exactly the sampling bubble that async OPD exists to hide, and GKD pays it in full rather than pipelining it.

## Objective

Generic `D(p_T ‖ p_S^θ)` — **teacher first**; GKD's "forward KL" is KL(teacher ‖ student) and its "reverse KL" is KL(student ‖ teacher), matching our convention. Full-vocabulary token-level (teacher logits are available), never a sampled-token estimator, and no top-k truncation.

On-policy KD loss, Eq. (4) (§3.1):

```latex
L_{OD}(\theta) := \mathbb{E}_{x\sim X}\Big[\mathbb{E}_{y\sim p_S(\cdot|x)}\big[\mathcal{D}_{KL}\big(p_T\|p_S^\theta\big)(y|x)\big]\Big]
```

GKD objective (§3.1, boxed, unnumbered — sits between Eq. (4) and the "Choice of Divergence" paragraph):

```latex
L_{\mathrm{GKD}}(\theta) := (1-\lambda)\,\mathbb{E}_{(x,y)\sim(X,Y)}\big[\mathcal{D}(p_T\|p_S^\theta)(y|x)\big] \;+\; \lambda\,\mathbb{E}_{x\sim X}\Big[\mathbb{E}_{y\sim p_S(\cdot|x)}\big[\mathcal{D}(p_T\|p_S^\theta)(y|x)\big]\Big]
```

"On-policy and supervised KD are instantiations of GKD with divergence D set to forward KL and student data fractions λ to 1 and 0 respectively." Divergences explored: forward KL, reverse KL, and JSD(β) for β ∈ {0.1, 0.5, 0.9} (§4).

## Off-policy correction

**None, and by design.** GKD deliberately drops the score-function/importance machinery:

> "where we do *not* backpropagate through the student's sampling distribution p_S(·|x), similar to on-policy imitation. Not backpropagating through the sampling makes the training stable and computationally efficient." (§3.1, after Eq. 4)

> "Akin to on-policy KD, we do not backpropagate gradients through the student's sampling process." (§3.1)

No importance ratio, no clipping, no truncation, no mask. The λ < 1 branch trains on a fixed dataset with *no* correction either — it is simply a different (teacher/ground-truth) data distribution under the same divergence. GKD's contrast with MiniLLM makes the design choice explicit (§5):

> "However, we argue that GKD is simpler and more stable, being closer to supervised training, since it does not backpropagate through the student's sampling process. Indeed, MiniLLM relies on a number of stabilizing tricks, to tackle high variance, reward hacking, and generation length bias."

## Results: async vs sync

**None** — no stale-student arm exists. What the paper *does* report is the fresh-student vs fixed-teacher-dataset mixing sweep. Reproduced here with the caveat that the λ = 0 end is teacher/ground-truth data, **not** stale student data:

Figure 7 (§4.3, p. 8) — GSM8K accuracy *improvement* over the SFT student, T5-XL → T5-Base (teacher 27.9, student 10.16 greedy):

| Student data fraction λ | Forward KL | JSD(0.1) | JSD(0.5) | JSD(0.9) | Reverse KL |
|---|---|---|---|---|---|
| 100% (all fresh student) | **8.8** | 7.6 | 7.0 | 6.7 | 8.0 |
| 50% (mixed) | 6.8 | 6.2 | 6.4 | 5.9 | 6.9 |
| 0% (fixed dataset only) | 4.7 | 5.1 | 5.1 | 5.0 | 5.7 |

Figure 8 (§4.3, p. 8) — GSM8K test accuracy vs λ, FLAN T5-XL → T5-Small (values read off the plot, ±0.1):

| λ | 0% | ~12% | 25% | 50% | 75% | ~90% | 100% |
|---|---|---|---|---|---|---|---|
| Forward KL | 7.2 | 6.9 | 7.2 | 8.85 | 9.2 | 9.65 | **11.2** |
| Reverse KL | 8.4 | 7.95 | 8.0 | 9.05 | 9.2 | 9.4 | **10.15** |

Caption: "**Varying on-policy data on GSM8K**. As we increase fraction of student-generated data beyond 25%, performance typically improves." Note the non-monotone dip between 0% and 25% in both curves — a small amount of student data is *worse* than none.

## Explanation offered

The paper's mechanism for why fresh student data helps is **train-inference distribution mismatch / exposure bias**, i.e. support mismatch plus compounding error — the same mechanism the async literature invokes for *staleness*, applied here to the teacher-vs-student data gap:

> "distilling auto-regressive student models using such approaches results in train-inference distribution mismatch. This is because the partial sequences encountered by the student during the auto-regressive generation phase at inference can be quite different from the ones seen during the training phase. Since predictions at any step are contingent upon previous steps in auto-regressive models, this mismatch can have a cascading effect where error in prediction at early step can affect the future predictions, resulting in poor quality text generation." (§3, p. 3–4)

> "When using on-policy data during distillation, the student receives token-specific feedback from the teacher's logits on the erroneous tokens in its self-generated output sequences. This enables a form of feedback loop akin to what we observe in RL, which helps minimize the train-inference distribution mismatch." (§3.1)

On the divergence choice (§3.1, "Choice of Divergence in GKD") — relevant to which objective family should be robust where:

> "forward KL ... requires the student to cover the entire support of the teacher token-level distribution p_T(.|y<n, x). In doing so, the student might end up assigning probability mass to tokens v which have low probability under p_T(.|y<n, x), which can result in hallucination and low-quality generations. When the student has much lower model capacity than the teacher, this issue is likely to happen with temperature sampling. Alternatively, mode-seeking divergences, such as reverse KL, prioritize the tokens where the teacher assigns high probability, which can avoid low-quality generations but at the expense of less diverse generations for a given input. Our experiments indicate that optimal divergence seems to be task-dependent."

Nothing in any of this is about lag between the behaviour student and the current student.

## Quotes

Every statement bearing on the on-policy/off-policy data axis:

- §3.1: "on-policy imitation approaches (e.g. Ross et al., 2011) iteratively collect sequences using the student policy, obtain expert labels for those sequences, and then retrain the student on this dataset. Despite their popularity in robotics and deep RL, on-policy approaches are typically not used for distilling auto-regressive models."
- §3.1: "In on-policy KD, the training is done on output sequences that the student is likely to generate. During training, we use a temperature of γ = 1 to encourage diversity in student generated sequences."
- §3.1: "Moreover, as the student evolves during training, the data it generates also improves in quality."
- §3.1: "For output sequences, GKD uses a mixture of fixed dataset, either teacher-generated or ground-truth, and on-policy student-generated sequences."
- §3.1: "λ ∈ [0, 1] is a hyper-parameter that controls the *student data fraction*, that is, the fraction of on-policy student-generated outputs."
- §4: "For student data fraction λ, we try λ = 1 (On-policy), λ = 0.5 (Mixed) and λ = 0 (Supervised). In particular, we are interested in the on-policy variants (λ = 1), which have not been previously explored."
- §4.2 (Results, WMT): "Moreover, using purely on-policy and mixed data distributions consistently outperform GKD variants only using a fixed supervised dataset, showing the importance of generating on-policy output sequences from the student."
- §4.3 (Results, GSM8K): "We observe that when using only the fixed CoT dataset or mixing it with student-generated CoTs, performance consistently falls short of using solely the student-generated CoTs."
- §4.3 / Fig. 8 caption: "As we increase fraction of student-generated data beyond 25%, performance typically improves."
- §4.4: "Additionally, Figure 8 shows that performance consistently improves as the proportion of on-policy data increases, provided that at least 25% of the data is on-policy."
- §5 (Related work, on ImitKD): "ImitKD (Lin et al., 2020) identifies this connection by sampling sequences from both the student and a fixed dataset but does not push the idea further. Unlike GKD, ImitKD does not explore purely on-policy data collection, nor does it integrate RL fine-tuning. ... ImitKD can be viewed as GKD with forward KL and a non-increasing schedule on λ, a simple choice being λ = 0.5."
- App. A.2: "the computational overhead from student sampling is approximately 1.8×, 2× and 2.2× compared to sampling from a fixed dataset of outputs"

## Not in the source

- **No stale-student experiment at all.** Nothing generates student samples once from the initial student and reuses them; nothing caches, buffers, or replays student rollouts; no run takes more than one gradient step per sampled batch. The word "stale" and the concepts "replay", "buffer", "asynchronous", "lag" do not appear.
- **λ is not a staleness knob** and the paper never suggests otherwise. Citing Fig. 7/8 as evidence about stale data would be a misreading — the λ = 0 end is a *fixed teacher/ground-truth* dataset.
- No off-policy correction is given because none is needed at zero lag; the paper offers no guidance on what would be needed if lag were introduced.
- No throughput or wall-clock numbers beyond the 1.8–2.2× sampling overhead ratio — no proposal to hide it, and no measurement of generator idle time.
- No analysis of how the choice of divergence (forward KL / JSD / reverse KL) would interact with data staleness — only with the teacher-vs-student data gap.
