// ML paper template (NeurIPS-like single-column layout).
// Fill in the metadata below and add content under each section.

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#let paper-title = [Your Reverse-KL Is Not Mode-Seeking]
#let authors = (
  (name: "Jonas De Schouwer", affiliation: "Stanford University", email: "jonasds@stanford.edu"),
)
#let abstract-body = [
  // Abstract text goes here.
]
#let keywords = ()

// Inline comment box for draft notes, e.g. #jonas[Check this claim.]
#let jonas(body) = block(
  width: 100%,
  fill: rgb("#fdecec"),
  stroke: 0.5pt + rgb("#e8a9a9"),
  radius: 3pt,
  inset: 0.7em,
  breakable: false,
)[
  #text(weight: "bold", fill: rgb("#a33030"))[Jonas] \
  #body
]

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

#set document(title: paper-title, author: authors.map(a => a.name))

#set page(
  paper: "us-letter",
  margin: (x: 1.5in, y: 1in),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, spacing: 1.1em)

#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 11pt)
#show heading.where(level: 3): set text(size: 10pt)
#show heading: set block(above: 1.4em, below: 0.9em)

#show quote.where(block: true): it => block(
  width: 100%,
  stroke: (left: 3pt + luma(175)),
  fill: luma(249),
  inset: (left: 12pt, right: 10pt, y: 8pt),
  radius: (top-right: 2pt, bottom-right: 2pt),
  above: 1.1em,
  below: 1.1em,
  {
    text(size: 0.85em, fill: luma(64), it.body)
    if it.attribution != none {
      align(right, text(size: 7.5pt, fill: luma(110))[— #it.attribution])
    }
  },
)

#set math.equation(numbering: "(1)")
#set figure(gap: 0.8em)
#show figure.caption: set text(size: 9pt)
#show table: set text(size: 9pt)

#set footnote.entry(separator: line(length: 30%, stroke: 0.5pt))
#set cite(style: "chicago-author-date")  // e.g. (Vaswani et al. 2017)

#show link: set text(fill: rgb("#00008b"))
#show ref: set text(fill: rgb("#00008b"))

// ---------------------------------------------------------------------------
// Title block
// ---------------------------------------------------------------------------

#align(center)[
  #block(text(size: 17pt, weight: "bold", paper-title))
  #v(1em)
  #grid(
    columns: authors.len(),
    column-gutter: 2.5em,
    ..authors.map(a => [
      *#a.name* \
      #a.affiliation \
      #text(size: 9pt, font: "DejaVu Sans Mono", link("mailto:" + a.email, a.email))
    ]),
  )
  #v(1em)
]

// ---------------------------------------------------------------------------
// Abstract
// ---------------------------------------------------------------------------

#align(center, text(size: 12pt, weight: "bold")[Abstract])
#block(inset: (x: 2em), {
  abstract-body
  if keywords.len() > 0 {
    v(0.6em)
    [*Keywords:* #keywords.join(", ")]
  }
})
#v(1em)

// ---------------------------------------------------------------------------
// Main body
// ---------------------------------------------------------------------------

#jonas[
Overarching goal:
1. Identify widely held belief in the comunity
2. Show it is false
3. Offer a corrected or better view
]


= Hook

In the past few years, reverse KL has become a common objective for on-policy distillation and its variants @gu2024minillm @agarwal2024gkd @wen2023fdistill @ko2024distillm @ko2025distillm2 @sessa2024bond @lu2025onpolicydistillation @xu2025skd @jin2026entropy @song2026survey @nvidia2026nemotron3ultra @nvidia2026nemotroncascade2 @deepseek2026v4 @kimi2026k3 @glm2026glm5, and as a regularizer in RL @ziegler2019finetuning @stiennon2020learning @ouyang2022training @bai2022training @korbak2022rl @rafailov2023direct @wang2024beyond @shao2024deepseekmath @apple2024afm @lambert2024tulu3 @olmo2025olmo2 @deepseek2025v32.

As a motivation, they refer to the "folklore" knowledge that reverse-KL is mode-seeking, whereas forward-KL is mode-covering.


#quote(block: true, attribution: [@gu2024minillm · #emph[MiniLLM: Knowledge Distillation of Large Language Models]])[Minimizing reverse KLD has been shown to cause the *mode-seeking* behavior in generative modeling [Hus15, NCT16, CDP+18, LPSK23], where $q_theta$ assigns high probabilities to $p$'s large modes and ignore the small ones.]

#quote(block: true, attribution: [@agarwal2024gkd · #emph[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]])[Given model capacity mismatch, when approximating $P(C)$ using a distribution $Q_theta (C)$, minimizing the reverse and forward KL results in *mean and mode-seeking* behavior (Figure A.16).]

#quote(block: true, attribution: [@wen2023fdistill · #emph[f-Divergence Minimization for Sequence-Level Knowledge Distillation]])[RKL distillation has the opposite problem, known as *mode collapsing*, where the student only learns one or a few modes of the teacher distribution. This is because the RKL loss would be large, if $q_theta (Y')$ is high but $p(Y')$ is low for some $Y'$.]

#quote(block: true, attribution: [@jin2026entropy · #emph[Entropy-Aware On-Policy Distillation of Language Models]])[[the reverse KL objective] penalizes generated tokens that the teacher considers unlikely, but ignores teacher modes that the student does not visit. It induces *mode-seeking* behavior, encouraging the student to concentrate probability mass on a single high-likelihood mode of the teacher while ignoring others [29].]

#quote(block: true, attribution: [@sessa2024bond · #emph[BOND: Aligning LLMs with Best-of-N Distillation]])[…the reverse KL$(p parallel q)$ is well-known to have a *mode-seeking* effect, steering policy $p$ to produce generations that have a high likelihood according to $q$ (Agarwal et al., 2024). While the forward KL may produce *over-spread distributions*, the backward KL can lead to policy and entropy collapses.]

#quote(block: true, attribution: [@lu2025onpolicydistillation · #emph[On-Policy Distillation]])[[Reverse KL] is *"mode seeking"*: it learns one specific behavior (the teacher's) instead of spreading its distribution across several suboptimal options.]


However, is this true? Little interactive quiz!


== Quiz: Knowledge Distillation Toy Setting


[Replace by simpler description, and minimalistic figure]

*Experimental setup*
We create a synthetic database of $N = 10,000$ book → author associations and attempt to distill it from an open-book teacher into a closed-book student. The student must therefore internalize all 10,000 associations. The teacher is a frozen `Qwen2.5-3B-Instruct` @qwen2024qwen25 that sees a 10-pair list containing the queried book and is instructed to answer with its author, or `unknown` if the book is not in the list; two natural sentences in the prompt (a warning that the list may contain transcription errors, and a demonstration of answering `unknown`) make the teacher hedge visibly without any artificial modification of its output distribution. The student is the base model `Qwen2.5-3B`, trained on the closed-book prompt `By whom was <book> written?` using a KL divergence objective. We compare exact full-vocabulary FKL and RKL distillation on the answer tokens against an SFT baseline trained on the ground-truth answers, under a fixed budget of 10 epochs. Hyperparameters are tuned per method on a held-out seed, selecting the best validation NLL among checkpoints whose MMLU @hendrycks2021mmlu stays within 1.5 points of the untrained baseline — forbidding configurations that buy task NLL by destroying general capability. We report mean ± std over 5 seeds, evaluating after every epoch on 1,000 held-out pairs under an intentionally paraphrased prompt (`The author of <book> is`).

*Teacher characterization*
[Add table like figs/teacher_characterization.png, with the ]

@tab-teacher summarizes the teacher's behavior over the training prompts of the five evaluation seeds. The teacher is a near-oracle for the task — median P(correct author) is 0.99997 — with a small, heavy-tailed hedge. `unknown` receives 0.69% of the mass on average, but only $1.4 times 10^(-6)$ in the median context.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    table.header([*First-token probability*], [*Median*], [*Arithmetic mean*], [*Geometric mean*]),
    [P(correct author)], [0.99997], [0.9796], [0.8983],
    [P(`unknown`)], [$1.4 times 10^(-6)$], [$6.9 times 10^(-3)$], [$6.4 times 10^(-7)$],
  ),
  caption: [
    Teacher first-token statistics under the training prompts, pooled over the 5 evaluation seeds (10,000 prompts per seed; fp32 log-probabilities). P(`unknown`) sums the `unknown` and `Unknown` first tokens. Per-seed variation is negligible.
  ],
) <tab-teacher>


Two columns, each with options:
1. Column 1: FKL
  - Option 1: FKL diverges
  - Option 2: FKL produces "unknown" for all contexts.
  - Option 3: FKL learns a smoothed version of the teacher's distribution, i.e. for each context, the student spreads its probability mass over all the authors.
  - Option 4: FKL learns the author perfectly for each context.

2. Column 2: RKL
  - Option 1: RKL diverges
  - Option 2: RKL produces "unknown" for all contexts.
  - Option 3: RKL learns a smoothed version of the teacher's distribution, i.e. for each context, the student spreads its probability mass over all the authors.
  - Option 4: RKL learns the author perfectly for each context.

[A submit button.]

[After the submit button is clicked the following two charts are shown:]


#grid(columns: 2, gutter: 6pt,
  [#figure(
    image("figs/val_nll_e3.png"),
    caption: [Validation NLL of the correct answers under the paraphrased prompt (mean ± std over 5 seeds).],
  ) <fig-e1-nll>],
  [#figure(
    image("figs/val_p1_unknown_e3.png"),
    caption: [Mean first-token probability of `unknown` under the validation prompt. The teacher's own arithmetic mean is 0.69%.],
  ) <fig-e1-unknown>],
)





#pagebreak()

In this blog post, we:
1. Uncover the origins of this folklore and show why it is incorrect for modern language model post-training (section @sec-folklore)
2. We provide an alternative framework for language model training under reverse-KL and forward-KL objectives, based on a decomposition of the expressivity bottleneck into 2 components:
  (1) an output-space expressivity bottleneck
  (2) a cross-context coupling expressivity bottleneck. We derive the optimal element of the model class under both the reverse-KL and forward-KL divergence objectives. (section @sec-framework)
3. We show how this framework explains the result of the toy experiment (section @sec-experiments)


= The Folklore: Mode-Seeking and Mode-Covering KL Divergences
<sec-folklore>

0. If $q$ is a student model we are training, and $p$ is a ground-truth distribution we are trying to learn, then the #emph[reverse KL divergence] is $D_"KL" (q||p) = EE_(x~q(dot)) [log q(x)/p(x)]$. The #emph[forward KL divergence] is $D_"KL" (p||q) = EE_(x~p(dot)) [log p(x)/q(x)]$.

1. In the past few years, reverse KL has become a common objective for on-policy distillation and its variants @gu2024minillm @agarwal2024gkd @wen2023fdistill @ko2024distillm @ko2025distillm2 @sessa2024bond @lu2025onpolicydistillation @xu2025skd @jin2026entropy @song2026survey @nvidia2026nemotron3ultra @nvidia2026nemotroncascade2 @deepseek2026v4 @kimi2026k3 @glm2026glm5, and as a regularizer in RL @ziegler2019finetuning @stiennon2020learning @ouyang2022training @bai2022training @korbak2022rl @rafailov2023direct @wang2024beyond @shao2024deepseekmath @apple2024afm @lambert2024tulu3 @olmo2025olmo2 @deepseek2025v32.

2. As a motivation, these papers refer to certain "folklore" knowledge that reverse-KL is mode-seeking, whereas forward-KL is mode-covering. This folklore stems from older work in variational inference, most notably @minka2005divergence @bishop2006prml, and to a lesser extent @huszar2015how @mackay2003itila @goodfellow2016nips @blei2017variational.

// #quote(block: true, attribution: [@minka2005divergence · #emph[Divergence Measures and Message Passing]])[These divergences are zero-forcing because $p(x) = 0$ forces $q(x) = 0$. In other words, they avoid "false positives"… Zero-forcing emphasizes modeling the tails, rather than the bulk of the distribution, which tends to underestimate the variance of p.]

// #quote(block: true, attribution: [@bishop2006prml · #emph[Pattern Recognition and Machine Learning]])[…a variational treatment based on the minimization of $"KL"(q parallel p)$ will tend to find one of these modes. By contrast, if we were to minimize $"KL"(p parallel q)$, the resulting approximations would average across all of the modes and … would lead to poor predictive distributions.]


#quote(block: true, attribution: [@gu2024minillm · #emph[MiniLLM: Knowledge Distillation of Large Language Models]])[Minimizing reverse KLD has been shown to cause the *mode-seeking* behavior in generative modeling [Hus15, NCT16, CDP+18, LPSK23], where $q_theta$ assigns high probabilities to $p$'s large modes and ignore the small ones.]

#quote(block: true, attribution: [@agarwal2024gkd · #emph[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]])[Given model capacity mismatch, when approximating $P(C)$ using a distribution $Q_theta (C)$, minimizing the reverse and forward KL results in *mean and mode-seeking* behavior (Figure A.16).]

#quote(block: true, attribution: [@wen2023fdistill · #emph[f-Divergence Minimization for Sequence-Level Knowledge Distillation]])[RKL distillation has the opposite problem, known as *mode collapsing*, where the student only learns one or a few modes of the teacher distribution. This is because the RKL loss would be large, if $q_theta (Y')$ is high but $p(Y')$ is low for some $Y'$.]

#quote(block: true, attribution: [@jin2026entropy · #emph[Entropy-Aware On-Policy Distillation of Language Models]])[[the reverse KL objective] penalizes generated tokens that the teacher considers unlikely, but ignores teacher modes that the student does not visit. It induces *mode-seeking* behavior, encouraging the student to concentrate probability mass on a single high-likelihood mode of the teacher while ignoring others [29].]

#quote(block: true, attribution: [@sessa2024bond · #emph[BOND: Aligning LLMs with Best-of-N Distillation]])[…the reverse KL$(p parallel q)$ is well-known to have a *mode-seeking* effect, steering policy $p$ to produce generations that have a high likelihood according to $q$ (Agarwal et al., 2024). While the forward KL may produce *over-spread distributions*, the backward KL can lead to policy and entropy collapses.]

#quote(block: true, attribution: [@lu2025onpolicydistillation · #emph[On-Policy Distillation]])[[Reverse KL] is *"mode seeking"*: it learns one specific behavior (the teacher's) instead of spreading its distribution across several suboptimal options.]



#grid(
  columns: (1fr, 1fr),
  column-gutter: 1em,
  row-gutter: 0.8em,
  align: (x, y) => if y == 0 { bottom } else { top },
  image("figs/fig_minillm_3peak.svg", width: 100%),
  image("figs/fig_bishop_3panel.svg", width: 100%),
  [
    #show figure: it => it.caption
    #figure(
      kind: image,
      caption: [
        A multimodal 1-D distribution is fit with a single Gaussian. The forward-KL fit is #emph[mode-covering] and covers all three peaks; the reverse-KL fit is #emph[mode-seeking] and locks onto the dominant right-hand mode. \
        Similar figures appear in @bishop2006prml @agarwal2024gkd @wen2023fdistill @wang2024beyond
      ],
    )[] <fig-folklore-1d>
  ],
  [
    #show figure: it => it.caption
    #figure(
      kind: image,
      caption: [
        A bimodal 2-D distribution is fit with a single Gaussian. *Left*: the forward-KL fit covers both modes; *Middle/Right:* two distinct local minima of reverse KL, one per mode. \
        Similar figures appear in @bishop2006prml @huszar2015how
      ],
    )[] <fig-folklore-2d>
  ],
)



3. However, this folklore is based on a highly oversimplified model, where an unconditional distribution $p(x)$ is being fit with a model $q(x)$ that is capacity-limited w.r.t. $p(x)$.
This is not realistic for language models for 2 reasons:

1. Language models fit a distribution $p(v|c)$ with a model $q(v|c)$, where both distributions are conditional on a context $c$.
2. *The expressivity bottleneck* for language models *is not in the output-space*: since the output space is discrete (dimension=V) and language models almost always have a learnable embedding for each token in their vocabulary, for any context $c$, the model $q(v|c)$ is able to fit the distribution $p(v|c)$ to arbitrary precision.

#jonas[
  Disclaimer:
  The folklore is _technically_ correct when viewing the language model as modelling an unconditional distribution $p(x)$ over *sequences*, and hence training with a sequence-level KL divergence. This is different from a token-level KL divergence, since one would need to take into account that changing the student at token t will change the probability of the entire sequence, hence influence the gradient at tokens s>t. \
  However, the vast majority of academic language model paper are trained with token-level KL divergences, as well as all technical reports of industry models we checked.
]


= Cross-Context Coupling as an Alternative Framework <sec-framework>

As discussed in the previous section, the "mode-seeking" vs. "mode-covering" folklore is based on a highly oversimplified model, where an unconditional distribution $p(x)$ is being fit with a model $q(x)$ that is capacity-limited w.r.t. $p(x)$.

*The real expressivity bottleneck in LLMs is that all context-conditioned distributions are produced by the same finite model*. If two contexts receive indistinguishable internal representations, their output distributions cannot be chosen independently. We call this shared-representation constraint #emph[cross-context coupling].

*An effective context-information bottleneck.* To describe this constraint without tying it to a particular model component, we let $Z$ represent whatever information about the context the model can use when choosing its output distribution. Contexts that map to the same $Z$ are effectively indistinguishable and must share the same prediction $q(v|z)$. This abstraction does not specify why the model fails to distinguish them: the cause could be its architecture, limited capacity, parameter sharing, or incomplete optimization.

The resulting dependency structure is $V arrow.l C arrow.r Z arrow.r hat(V)$, where $C ~ cal(D)$ is the context, $V ~ p(v|C)$ is the target token, $Z ~ p(z|C)$ is the information available to the model, and $hat(V) ~ q(v|Z)$ is its prediction.

To isolate the effect of this bottleneck, we take the model's pattern of context distinguishability as fixed and allow $q(v|z)$ to be any categorical distribution.


*Forward KL yields an arithmetic mixture.* The expected forward-KL objective decomposes as

$
cal(L)_"FKL"
&= EE_(c~cal(D)) EE_(z~p(z|c)) [D_"KL" (p(v|c) || q(v|z))]
\
&= EE_(c,z) [D_"KL" (p(v|c) || p(v|z))]
+ EE_(c,z) EE_(v~p(v|c)) [log p(v|z) / q(v|z)]
\
&= I(V;C | Z)
+ EE_(z~p(z)) EE_(v~p(v|z)) [log p(v|z) / q(v|z)]
\
&= I(V;C | Z)
+ EE_(z~p(z)) [D_"KL" (p(v|z) || q(v|z))]
$

The first term, $I(V;C|Z)$, is the irreducible cost of discarding context information in $Z$; it does not depend on the decoder. The second is minimized by

$
q^*_"FKL" (v|z) = p(v|z) = EE_(c~p(c|z)) [p(v|c)]
$ <eq-fkl-opt>

Thus, when forward KL must use one prediction for contexts that are indistinguishable at $z$, it chooses their #emph[arithmetic mixture].

*Reverse KL yields a geometric mixture.* Reverse KL combines the same context-conditioned targets differently. Define their normalized geometric mixture as
$
tilde(p)(v|z) = 1/A(z) exp(EE_(c~p(c|z)) [log p(v|c)]),
quad A(z) = sum_v exp(EE_(c~p(c|z)) [log p(v|c)]).
$
The expected reverse-KL objective then becomes

$
cal(L)_"RKL"
&= EE_(c~cal(D)) EE_(z~p(z|c)) [D_"KL" (q(v|z) || p(v|c))]
\
&= EE_(c,z) [D_"KL" (q(v|z) || tilde(p)(v|z))]
+ EE_(c,z) EE_(v~q(v|z)) [log (tilde(p)(v|z)) / p(v|c)]
\
&= EE_(z~p(z)) [D_"KL" (q(v|z) || tilde(p)(v|z))]
+ EE_(z~p(z)) EE_(v~q(v|z)) [log tilde(p)(v|z) - EE_(c~p(c|z)) [log p(v|c)]]
\
&= EE_(z~p(z)) [D_"KL" (q(v|z) || tilde(p)(v|z))]
+ EE_(z~p(z)) EE_(v~q(v|z)) [- log A(z)]
\
&= EE_(z~p(z)) [D_"KL" (q(v|z) || tilde(p)(v|z))]
+ EE_(z~p(z)) [- log A(z)]
$

Again, the second term is fixed by the bottleneck, so the optimum is

$
q^*_"RKL" (v|z) = tilde(p)(v|z)
= 1/A(z) exp(EE_(c~p(c|z)) [log p(v|c)]).
$ <eq-rkl-opt>

The distinction is clearest in the fully context-confused limit. If $Z$ is independent of $C$, the model must use one output distribution for every context. Forward KL averages probabilities: a token retains the mass it receives on average across contexts. Reverse KL averages #emph[log]-probabilities: a token is suppressed when it is assigned very low probability in the other contexts sharing the representation. Its optimum therefore favors tokens with broad cross-context support.

4. Corollary: This produces a failure mode that the unconditional mode-seeking picture does not predict. Under a RKL objective, a desired token in a context $c$ may be disproportionally weighted down if they are highly unlikely in other contexts.
Conversely, tokens that are merely acceptable in many contexts may be disproportionally upweighted, even if they are not a mode of any individual $p(v|c)$. We term this behavior #emph[cross-context compromise latching]. The experiment below tests this prediction directly.

Further, we hypothesize that this phenomenon results in models trained with reverse KL to predict more "filler" tokens than models trained with forward KL.


= Experiments <sec-experiments>

== Experiment 1: Cross-Context Compromise Latching - Toy Setting

To investigate to what extent the compromise latching pathology occurs in realistic settings, we set up a tractable experiment in a knowledge-distillation setting.

*Experimental setup*
We create a synthetic database of $N = 10,000$ book → author associations and attempt to distill it from an open-book teacher into a closed-book student. The student must therefore internalize all 10,000 associations. The teacher is a frozen `Qwen2.5-3B-Instruct` @qwen2024qwen25 that sees a 10-pair list containing the queried book and is instructed to answer with its author, or `unknown` if the book is not in the list; two natural sentences in the prompt (a warning that the list may contain transcription errors, and a demonstration of answering `unknown`) make the teacher hedge visibly without any artificial modification of its output distribution. The student is the base model `Qwen2.5-3B`, trained on the closed-book prompt `By whom was <book> written?` using a KL divergence objective. We compare exact full-vocabulary FKL and RKL distillation on the answer tokens against an SFT baseline trained on the ground-truth answers, under a fixed budget of 10 epochs. Hyperparameters are tuned per method on a held-out seed, selecting the best validation NLL among checkpoints whose MMLU @hendrycks2021mmlu stays within 1.5 points of the untrained baseline — forbidding configurations that buy task NLL by destroying general capability. We report mean ± std over 5 seeds, evaluating after every epoch on 1,000 held-out pairs under an intentionally paraphrased prompt (`The author of <book> is`).

*Teacher characterization*
@tab-teacher summarizes the teacher's behavior over the training prompts of the five evaluation seeds. The teacher is a near-oracle for the task — median P(correct author) is 0.99997 — with a small, heavy-tailed hedge. `unknown` receives 0.69% of the mass on average, but only $1.4 times 10^(-6)$ in the median context.


= The Cross-Context Coupling Framework Predicts Compromise Latching

In the (unrealistic) fully context-confused limit, @sec-framework predicts that the FKL-optimal student outputs the #emph[arithmetic] mixture of the teacher's per-context first-token distribution. This mixture assigns `unknown` its average mass of *0.74%*. // mean on the held-out tuning seed
The RKL-optimal student outputs the #emph[normalized geometric] mixture, which on the same logits assigns *99.2%* of its mass `unknown` tokens. The reason is that `unknown` is the only token with non-negligible probability under #emph[every] context. The framework therefore predicts that, to the extent the student conflates contexts, reverse-KL training should collapse the student onto `unknown` — even though the teacher grants it less than 1% mass for each context.



== Experiment 1b: Toy Setting without the Hedging Prompt

#grid(columns: 2, gutter: 6pt,
  [#figure(
    image("figs/val_nll_e4.png"),
    caption: [Validation NLL of the correct answers under the paraphrased prompt (mean ± std over 5 seeds), distilling from the teacher prompted #emph[without] the two hedging sentences.],
  ) <fig-e1b-nll>],
  [#figure(
    image("figs/val_p1_unknown_e4.png"),
    caption: [Mean first-token probability of `unknown` under the validation prompt, same setting. The teacher's own arithmetic mean is 0.03%.],
  ) <fig-e1b-unknown>],
)

*Results*
#emph[RKL never learns the mapping], while FKL does. MMLU is untouched for RKL and stays within the base±1.5 bounds for FKL. For RKL, the first-token probability of `unknown` climbs monotonically to 0.3–0.45 (@fig-e1-unknown), amplifying the teacher's 0.7% hedge (arithmetic mean) by \~50× -- a collapse toward the geometric-mixture optimum. One step up the tuned learning-rate grid ($3 times 10^(-6)$), the collapse is near-total within a single epoch (P(`unknown`) = 0.998), in quantitative agreement with the 99.2% prediction above. FKL, by contrast, is faithful: its P(`unknown`) settles at 1–2%, close to the teacher's own hedge. This is cross-context compromise latching exactly as predicted by @sec-framework: reverse KL latches onto the one token that is "not-so-bad" in every context -- a hedge the teacher produces through natural prompting alone -- and it does so #emph[instead of], not in addition to, learning the 10,000 desired associations.



= Conclusion

Reverse KL is routinely described as *mode-seeking* and forward KL as *mode-covering*. That picture is inherited from variational inference and the textbook exercise of approximating a fixed, multimodal *unconditional* distribution with a deliberately under-capacity family (@sec-folklore). In this blog post, we showed that this analogy is misleading for the token-level objectives used in modern language-model distillation and KL-regularized RL.

The binding constraint is not output-space capacity but #emph[cross-context coupling]: the requirement that a single finite model assign one prediction to every context that shares the same internal representation (@sec-framework).
Reframing the problem around that bottleneck yields a sharper distinction than mode-seeking versus mode-covering. When contexts are pooled at $Z$, forward-KL distillation targets their #emph[arithmetic] mixture of teacher conditionals; reverse-KL distillation targets their #emph[normalized geometric] mixture. In the limit where the student treats all contexts alike, geometric pooling elevates tokens that are merely acceptable in many contexts. We call the resulting failure mode #emph[cross-context compromise latching]: the student latches onto a cross-context compromise that need not be a mode of any single $p(v|c)$, and may outright prevent learning context-specific structure.

Our book–author distillation experiment makes this concrete (@sec-experiments). A near-oracle open-book teacher assigns less than 1% first-token mass to `unknown` on average, yet a closed-book student trained with reverse KL learns to predict `unknown` all the time, while failing to internalize the 10,000 associations. Forward KL tracks the teacher's hedge and learns the mapping.

We do not claim that reverse KL is never useful. We do claim that invoking mode-seeking as the default intuition for token-level reverse KL in on-policy distillation and RL obscures the mechanism that matters in practice. A corrected folklore is not "reverse KL seeks modes," but "reverse KL geometrically pools teacher conditionals over whatever contexts the student confuses."




// ---------------------------------------------------------------------------
// Bibliography
// ---------------------------------------------------------------------------

#bibliography("refs.bib", title: "References", style: "chicago-author-date")
