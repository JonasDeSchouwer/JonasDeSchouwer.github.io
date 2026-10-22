#set page(margin: 2.2cm, numbering: "1")
#set text(size: 10pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)", supplement: [Eq.])
#show link: set text(fill: blue.darken(35%))
#show cite: set text(fill: blue.darken(35%))
#show quote.where(block: true): it => block(
  width: 100%,
  stroke: (left: 3pt + luma(175)),
  fill: luma(249),
  inset: (left: 12pt, right: 10pt, y: 8pt),
  radius: (top-right: 2pt, bottom-right: 2pt),
  above: 1.1em,
  below: 1.1em,
  {
    text(fill: luma(64), it.body)
    if it.attribution != none {
      v(5pt)
      align(right, text(size: 7.5pt, fill: luma(110))[— #it.attribution])
    }
  },
)
#show table: set text(size: 8.5pt)

#let sidenote(body) = {
  set text(size: 0.8em, fill: luma(120))
  body
}

#align(center)[
  #text(size: 16pt, weight: "bold")[Reverse-KL Objectives in the Training, Fine-Tuning, and Reinforcement Learning of Language Models]

  #v(2mm)
  A literature review, with an audit of what implementations actually optimize
  #v(1mm)
  #text(size: 9pt, fill: gray.darken(30%))[August 2026 · sources verified against primary texts and commit-pinned code; full quote dossiers in `lit_review/notes/`]
]

#v(4mm)

#outline(title: [Contents], indent: auto)

#v(4mm)

= Introduction and conventions <sec-intro>

Throughout this review, *reverse KL* (RKL) means
$ D_"KL" (q parallel p) = bb(E)_(x tilde.op q)[log q(x) - log p(x)], $ <eq-def>
where $q$ is always the distribution *being trained* (student or policy) and $p$ the *target* (teacher or reference). *Forward KL* (FKL) is $D_"KL" (p parallel q)$. We flag every source whose terminology or letter assignment deviates. The review covers (i) the folklore on choosing between the two directions and its evidentiary basis (@sec-folklore), (ii) where reverse KL actually appears in LM training — research methods, alignment pipelines, and production technical reports (@sec-practice), (iii) a code-level audit of what the associated implementations compute and which gradients they take, with derivations (@sec-estimators), (iv) reported empirical effects (@sec-empirical), and (v) criticism and alternative divergences (@sec-criticism), ending with a synthesis (@sec-synthesis).

*Terminology flags.* Non-trivially many sources deviate from the convention above; these deviations are a real hazard when reading this literature side by side.

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + gray,
  table.header([*Source*], [*Deviation*]),
  [PRML @bishop2006prml],
  [Calls $D_"KL" (p parallel q)$ "the reverse Kullback-Leibler divergence" (p. 467) — the *opposite* of our convention, because his default direction is the variational-inference objective $D_"KL" (q parallel p)$.],
  [BOND @sessa2024bond],
  [Eq. 11 uses $p$ for the *trained policy* and $q$ for the target — letters swapped relative to ours. Their "backward KL" $= D_"KL" ("policy" parallel "target")$ agrees with our reverse KL in direction; "backward" and "reverse" are used interchangeably.],
  [vBoN @amini2024variational],
  ["Backward KL" $=$ our reverse KL.],
  [#cite(<ziegler2019finetuning>, form: "prose"); #cite(<korbak2022rl>, form: "prose")],
  [Comma notation "$"KL"(pi, rho)$" for $D_"KL" (pi parallel rho)$; policy first, so direction matches ours.],
  [#cite(<schulman2020klapprox>, form: "prose")],
  [Writes $"KL"[q,p]$ with the *first* argument the sampled distribution; with $q$ the policy this is our reverse KL.],
  [f-DPO @wang2024beyond],
  [In f-divergence notation $D_f (p parallel q) = sum_x q f(p slash q)$, their reverse KL is $f(u)=u log u$ and forward KL is $f(u)=-log u$ — the $parallel$ order of the f-divergence does not visually match the KL subscript order.],
  [Minitron @muralidharan2024minitron],
  [Bare "KLD" means *forward* KL (explicit in their Table 15 caption: "forward KLD loss").],
  [#cite(<minka2005divergence>, form: "prose")],
  ["Exclusive" divergences ($alpha < 1$) $=$ reverse-KL-like; "inclusive" $=$ forward-KL-like. His $alpha$-parametrization ($alpha=0$ RKL, $alpha=1$ FKL) differs from Bishop's ($alpha=-1$ RKL, $alpha=+1$ FKL).],
  [Tinker cookbook (code)],
  [Docstring in `train_on_policy.py` writes "KL[p‖q] = log p − log q" with $p$ denoting the *student's* sampled logprobs; the computed quantity is $log q_"student" - log p_"teacher"$, i.e. our reverse KL. Naming swapped, computation consistent.],
)

= The folklore and its foundations <sec-folklore>

== The standard claims and the arguments behind them <sec-standard-claims>

The folklore, in one sentence: *forward KL is zero-avoiding / inclusive / mass-covering ("mean-seeking"); reverse KL is zero-forcing / exclusive / mode-seeking; use forward KL when coverage or likelihood matters, reverse KL when you want sharp, conservative, high-precision outputs and can tolerate dropped modes.* This section collects the canonical statements source by source — Minka and Bishop in detail; Murphy, MacKay, Goodfellow, and Huszár more briefly — asking three questions of each: what exactly is claimed, what argument (or source) supports it, and which assumptions are made explicit.

*Minka* @minka2005divergence *(technical report; the primary source).* The claims are stated for the $alpha$-divergence family (Zhu–Rohwer parametrization: $alpha = 0$ is reverse KL, $alpha = 1$ is forward KL), first as the outcome of a worked example — a *global* minimization over a deliberately too-small family:

#quote(block: true, attribution: [@minka2005divergence · #emph[Divergence Measures and Message Passing]])[The original distribution $p(x)$ is a mixture of two Gaussians, one tall and narrow, the other short and wide. The approximation $q(x)$ is required to be a single (scaled) Gaussian, with arbitrary mean, variance, and scale factor. For different values of $alpha$, figure 1 plots the global minimum of $D_alpha (p parallel q)$ over $q$. … When $alpha$ is a large negative number, the best approximation represents only one mode, the one with largest mass (not the mode which is highest). When $alpha$ is a large positive number, the approximation tries to cover the entire distribution, eventually forming an upper bound when $alpha arrow infinity$.]

(that figure is reproduced, with our recreation, in @fig-alpha). The argument for the general claim is a direct analysis of the objective — which term of the divergence formula penalizes what:

#quote(block: true, attribution: [@minka2005divergence · #emph[Divergence Measures and Message Passing]])[The properties observed in this example are general, and can be derived from the formula for $alpha$-divergence. Start with the mode-seeking property for $alpha << 0$. It happens because the valleys of $p$ force the approximation downward. … These divergences are zero-forcing because $p(x) = 0$ forces $q(x) = 0$. In other words, they avoid "false positives," to an increasing degree as $alpha$ gets more negative. … The cost of excluding an $x$, i.e. setting $q(x) = 0$, is $p(x) slash (1 - alpha)$. Therefore $q$ will keep the areas of largest total mass, and exclude areas with small total mass. Zero-forcing emphasizes modeling the tails, rather than the bulk of the distribution, which tends to underestimate the variance of $p$.]

#quote(block: true, attribution: [@minka2005divergence · #emph[Divergence Measures and Message Passing]])[When $alpha >= 1$, a different tendency happens. These divergences want to cover as much of $p$ as possible. Following the terminology of Frey et al. (2000), these divergences are inclusive ($alpha < 1$ are exclusive). Inclusive divergences require $q > 0$ whenever $p > 0$, thus avoiding "false negatives."]

The only external source in these passages is Frey et al. (2000) — for the *terminology*, not the behavior. Two nuances from the same section are rarely quoted downstream: the zero-forcing optimum keeps the *largest-mass* region rather than the highest peak (above), and its variance tracks a single component even when many overlap ("if $p$ has 100 identical Gaussians in a row, forming a plateau, the optimal $q$ is only as wide as one of them"). Most importantly, Minka's own use-recommendation — the "when to use which" half of the folklore — is explicitly *conditioned on capacity*, in his concluding remarks:

#quote(block: true, attribution: [@minka2005divergence · #emph[Divergence Measures and Message Passing]])[If the approximating family is a good fit to the true distribution, then it doesn't matter which divergence measure you use, since all will give similar results. The only consideration at that point is computational complexity. If the approximating family is a poor fit to the true distribution, then you are probably safest to use an exclusive divergence, which only tries to model one mode. With an inclusive divergence, message passing probably won't converge at all. If the approximating family is a medium fit to the true distribution, then you need to consider the inference goal.]

The exact-minimization assumption also surfaces explicitly, if in passing: his theorem on the mass of the optimal $q$ "applies to an exact minimization over $tilde(Z)$, which is generally not possible."

*Bishop* @bishop2006prml *(PRML §10.1; the textbook articulation).* The setting is variational inference, and both framing assumptions — an intractable target and a deliberately restricted family — are stated before any behavioral claim:

#quote(block: true, attribution: [@bishop2006prml · #emph[Pattern Recognition and Machine Learning]])[[W]e shall suppose the model is such that working with the true posterior distribution is intractable. We therefore consider instead a restricted family of distributions $q(Z)$ and then seek the member of this family for which the KL divergence is minimized.]

The argument for zero-forcing is an analysis of the KL integrand:

#quote(block: true, attribution: [@bishop2006prml · #emph[Pattern Recognition and Machine Learning]])[[T]here is a large positive contribution to the Kullback-Leibler divergence … from regions of $Z$ space in which $p(Z)$ is near zero unless $q(Z)$ is also close to zero. Thus minimizing this form of KL divergence leads to distributions $q(Z)$ that avoid regions in which $p(Z)$ is small. Conversely, the Kullback-Leibler divergence $"KL"(p parallel q)$ is minimized by distributions $q(Z)$ that are nonzero in regions where $p(Z)$ is nonzero.]

supported by two worked examples (both reproduced in @sec-figures). From the factorized fit to a correlated Gaussian (his Fig. 10.2) he draws the compactness claim:

#quote(block: true, attribution: [@bishop2006prml · #emph[Pattern Recognition and Machine Learning]])[[T]he variance of $q(z)$ is controlled by the direction of smallest variance of $p(z)$ … It is a general result that a factorized variational approximation tends to give approximations to the posterior distribution that are too compact.]

From the single-Gaussian fit to a bimodal $p$ (his Fig. 10.3) he draws the mode-locking claim and the folklore's practice recommendation, with the multimodality premise stated up front:

#quote(block: true, attribution: [@bishop2006prml · #emph[Pattern Recognition and Machine Learning]])[In practical applications, the true posterior distribution will often be multimodal, with most of the posterior mass concentrated in some number of relatively small regions of parameter space. … [A] variational treatment based on the minimization of $"KL"(q parallel p)$ will tend to find one of these modes. By contrast, if we were to minimize $"KL"(p parallel q)$, the resulting approximations would average across all of the modes and, in the context of the mixture model, would lead to poor predictive distributions (because the average of two good parameter values is typically itself not a good parameter value).]

Note "will tend to find *one of* these modes": the reverse-KL solutions in his own figure are distinct local minima — panel (c) is captioned "[a]s in (b) but showing a different local minimum" — so which mode is found is an initialization accident (@fig-1d). His $alpha$-divergence summary states the general claim with a hedge ("typically") doing quiet work:

#quote(block: true, attribution: [@bishop2006prml · #emph[Pattern Recognition and Machine Learning]])[Then for $alpha <= -1$ the divergence is zero forcing, so that any values of $x$ for which $p(x) = 0$ will have $q(x) = 0$, and typically $q(x)$ will under-estimate the support of $p(x)$ and will tend to seek the mode with the largest mass. Conversely for $alpha >= 1$ the divergence is zero-avoiding, … and typically $q(x)$ will stretch to cover all of $p(x)$, and will over-estimate the support of $p(x)$.]

No external source is cited for any of these claims; the argument is self-contained (the integrand analysis plus the worked examples). One terminology hazard for modern readers: Bishop reserves "the reverse Kullback-Leibler divergence" for $"KL"(p parallel q)$ — the *opposite* of current LLM usage — since his default direction is the variational one.

*Murphy* @murphy2022pml1 @murphy2023pml2 *(the modern textbook transmission).* PML1 (§6.2.6) fixes the now-standard vocabulary — forward KL "will be zero-avoiding or mode-covering, and will typically over-estimate the support of $p$", reverse KL "is called zero-forcing or mode-seeking behavior … will typically under-estimate the support of $p$" — and PML2 states the mechanism:

#quote(block: true, attribution: [@murphy2023pml2 · #emph[Probabilistic Machine Learning: Advanced Topics]])[$q$ tends to cover both modes as it must be nonvanishing everywhere $p$ is; this is called mode-covering or zero-avoiding behavior … to prevent $D_"KL" (q parallel p)$ from becoming infinite, we must have $q = 0$ whenever $p = 0$, which creates mode-seeking or zero-forcing behavior]

The supporting argument is the most explicit in the canon: PML2 §5.1.4 derives both optima exactly in restricted families — the moment-matching optimum of forward KL for exponential-family $q$, and the precision-diagonal optimum of factorized reverse KL on a Gaussian target — the same derivations behind our recreations (@sec-recreations). Murphy is also the only canonical source to attach a counter-example caveat to the compactness claim:

#quote(block: true, attribution: [@murphy2023pml2 · #emph[Probabilistic Machine Learning: Advanced Topics]])[Note, however, that minimizing the reverse KL does not always result in an overly compact approximation, as explained in [Tur+08].]

where [Tur+08] is Turner, Berkes, Sahani, and MacKay's counter-examples report (@sec-folklore-crit).

*MacKay* @mackay2003itila *(ITILA Ch. 33; the tractability articulation).* The direction is chosen because it is the *evaluable* one: the variational free energy differs from $D_"KL" (Q parallel P)$ by a constant (his eq. 33.9), and "for a range of interesting energy functions, and for sufficiently simple approximating distributions, the variational free energy can be efficiently evaluated" — while the forward objective earns a bracketed warning:

#quote(block: true, attribution: [@mackay2003itila · #emph[Information Theory, Inference, and Learning Algorithms]])[[Note that in general it is not possible to evaluate the objective function $G$, because integrals under the true distribution $P(x)$ are usually intractable.]]

The compactness claim is stated as an unqualified tendency — "[t]he approximations given by variational free energy minimization always tend to be more compact than the true distribution" — but is explored through exercises rather than proved (Ex. 33.5; Ex. 33.7 constructs a problem with *three* distinct reverse-KL minima), and his printed solution states the capacity assumption in the same breath as the claim:

#quote(block: true, attribution: [@mackay2003itila · #emph[Information Theory, Inference, and Learning Algorithms]])[The best possible variational approximation is of course the target distribution $P$. Assuming that this is not possible, a good variational approximation is more compact than the true distribution. In contrast, a good sampler is more heavy tailed than the true distribution.]

*Goodfellow* @goodfellow2016nips *(GAN tutorial; folklore stated, then retracted).* The tutorial records the folklore as a belief:

#quote(block: true, attribution: [@goodfellow2016nips · #emph[NIPS 2016 Tutorial: Generative Adversarial Networks]])[Previously, many people (including the author) believed that GANs produced sharp, realistic samples because they minimize the Jensen-Shannon divergence while VAEs produce blurry samples because they minimize the KL divergence between the data and the model.]

Its Fig. 14 (reproduced in @fig-llm-originals) restates the two-direction claim with the capacity assumption unusually explicit in the caption itself:

#quote(block: true, attribution: [@goodfellow2016nips · #emph[NIPS 2016 Tutorial: Generative Adversarial Networks]])[The two directions of the KL divergence are not equivalent. The differences are most obvious when the model has too little capacity to fit the data distribution. … Because a single Gaussian cannot capture the true data distribution, the choice of divergence determines the tradeoff that the model makes.]

and the same section then withdraws the explanatory claim on empirical grounds:

#quote(block: true, attribution: [@goodfellow2016nips · #emph[NIPS 2016 Tutorial: Generative Adversarial Networks]])[GANs often choose to generate from very few modes; fewer than the limitation imposed by the model capacity. The reverse KL prefers to generate from as many modes of the data distribution as the model is able to; it does not prefer fewer modes in general. This suggests that mode collapse is driven by a factor other than the choice of divergence.]

*Huszár* @huszar2015how *(the generative-models articulation the LLM papers cite).* The claims are behavioral, aimed at sample quality rather than inference, and the reverse-KL claim carries its own support condition in the trailing clause:

#quote(block: true, attribution: [@huszar2015how · #emph[How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?]])[Minimising $"KL"[P parallel Q]$ often leads to models that overgeneralise, and sometimes produce samples that are very unlikely under P. … If we train a generative model by minimising $"KL"[Q parallel P]$, the model will very conservatively try to avoid any behaviour that is unlikely under P. This comes at the cost of ignoring modes of P completely, unless those additional modes can be modelled without introducing probability mass in regions where P has none.]

No citation is attached to these characterizations; the argument is a reading of which distribution weights the log-ratio in each objective, and the note's contribution is the interpolation knob ("minimising $"JS"_pi$ divergence for a range of $pi in (0, 1)$ allows us to interpolate between the behaviour of $"KL"[P parallel Q]$ and $"KL"[Q parallel P]$"; Fig. 1 reproduced in @fig-js). #cite(<theis2016note>, form: "prose") supplied the corresponding constructions for evaluation, concluding "log-likelihood and visual appearance of samples are therefore largely independent."

*Taking stock.* Across all six sources the argument has one shape: characterize the exact minimizer of each divergence over a family too small to contain the target. The two load-bearing assumptions are stated by the sources themselves, though rarely by those who cite them (@sec-folklore-llm): (i) *capacity limitation* — explicit in Minka's good-fit/poor-fit conditioning, Bishop's "restricted family", MacKay's "[a]ssuming that this is not possible", and Goodfellow's caption; and (ii) *exact global minimization* — Minka plots "the global minimum" and flags exact minimization as "generally not possible", while Bishop's own figure lands the reverse-KL fit in one of several local minima. Alongside the behavioral claims runs a separate, non-behavioral reason for the reverse direction: it is the *tractable* one — MacKay's bracketed note, Bishop's "working with the true posterior distribution is intractable", and the VI literature generally @blei2017variational — an asymmetry that recurs unchanged in RLHF, where reverse KL is the only direction estimable from on-policy samples given log-prob access to $p$ (@sec-rl).

== The standard figures <sec-figures>

The folklore is transmitted by a small set of recurring figures, worth knowing by name: Bishop PRML Fig. 10.2 (factorized $q$ on a correlated Gaussian, both directions) and Fig. 10.3 (single Gaussian on a bimodal $p$: (a) forward covers, (b)/(c) reverse locks onto one of two local minima) @bishop2006prml; Minka Fig. 1 (Gaussian fit across the $alpha$ family from mode-matching to covering) @minka2005divergence; Murphy's redraws with public notebooks (PML1 Fig. 6.3; PML2 Figs. 5.1–5.2) @murphy2022pml1 @murphy2023pml2; Goodfellow Figs. 14–15 @goodfellow2016nips; Theis et al. Fig. 1 @theis2016note; Huszár Fig. 1 (generalized-JS sweep) @huszar2015how; and, in the LLM-distillation literature, MiniLLM Fig. 2, which fits "a Gaussian mixture with a Gaussian distribution using forward KLD and reverse KLD" @gu2024minillm, and GKD Fig. A.16 (its caption is quoted in @sec-folklore-llm) @agarwal2024gkd. @fig-1d[Figures]–@fig-js[] show the originals, with our numerical recreations collected in @sec-recreations; @fig-llm-originals shows the GAN-era redrawing, and each LLM paper's own folklore figure is reproduced beside its entry in @sec-folklore-llm. The originals are reproduced from their sources for private study and commentary in this unpublished working document; the recreations are our own (generation script at `figs/make_figures.py`).

#figure(
  image("figs/originals/bishop_fig10_3.png", width: 86%),
  caption: [A single Gaussian fit to a bimodal $p$: the original Fig. 10.3, reproduced from PRML @bishop2006prml (p. 469, © 2006 Springer Science+Business Media): (a) the forward-KL fit covers both modes; (b)/(c) two distinct local minima of reverse KL, one per mode (blue $= p$, red $= q$). Our recreations: @fig-1d-ours[Figures] and @fig-1d-panels[].],
) <fig-1d>

#figure(
  image("figs/originals/bishop_fig10_2.png", width: 62%),
  caption: [A factorized (diagonal-covariance) Gaussian $q$ fit to a correlated Gaussian $p$: the original Fig. 10.2, reproduced from PRML @bishop2006prml (p. 468, © 2006 Springer Science+Business Media): green $= p$, red $= q$; their (a) minimizes $"KL"(q parallel p)$, their (b) minimizes $"KL"(p parallel q)$. Our recreation: @fig-2d-ours.],
) <fig-2d>

#figure(
  image("figs/originals/minka_fig1.png", width: 100%),
  caption: [Gaussian fits minimizing the $alpha$-divergence, Zhu–Rohwer parametrization ($alpha = 0$: reverse KL; $alpha = 1$: forward KL): the original Fig. 1, reproduced from Minka's technical report @minka2005divergence (© Microsoft Research); "$alpha arrow -infinity$ prefers matching one mode, while $alpha arrow infinity$ prefers covering the entire distribution." Our recreation: @fig-alpha-ours.],
) <fig-alpha>

#figure(
  image("figs/originals/huszar_fig1.png", width: 84%),
  caption: [Gaussian fits minimizing Huszár's generalized $"JS"_pi$: the original Fig. 1, reproduced from @huszar2015how (arXiv:1511.05101): a 2-D mixture $P$ (panel A) fit by a single isotropic Gaussian minimizing $"JS"_pi [P parallel Q]$ for $pi = 0.1, 0.5, 0.99$ (panels B–D). Our recreation: @fig-js-ours.],
) <fig-js>

#figure(
  image("figs/originals/goodfellow_fig14.png", width: 56%),
  caption: [The same construction as redrawn in the GAN literature: Fig. 14, reproduced from the GAN tutorial @goodfellow2016nips. The LLM-era redrawings appear beside their entries in @sec-folklore-llm.],
) <fig-llm-originals>

=== Our own recreations <sec-recreations>

Our numerical recreations of the constructions in @fig-1d[Figures]–@fig-js[], and of MiniLLM's toy figure (@fig-minillm-orig).

#figure(
  image("figs/fig_fkl_rkl_1d.svg", width: 74%),
  caption: [Our numerical recreation of @fig-1d on an asymmetric mixture. The forward-KL fit is the (exact) moment match and over-covers; the *global* reverse-KL minimum sits on the right-hand component, which carries the larger mass ($0.6$), *not* on the higher left-hand peak — the fit prefers "the one with largest mass (not the mode which is highest)" @minka2005divergence.],
) <fig-1d-ours>

#figure(
  image("figs/fig_bishop_3panel.svg", width: 100%),
  caption: [Our panel-by-panel recreation of @fig-1d: a bimodal 2-D mixture $p$ (blue) fit by a single full-covariance Gaussian $q$ (red). (a) The forward-KL fit is the (exact) moment match and covers both modes; (b)/(c) the two local minima of reverse KL, each locked onto one mode.],
) <fig-1d-panels>

#figure(
  image("figs/fig_factorized_2d.svg", width: 64%),
  caption: [Our recreation of @fig-2d ($rho = 0.9$; $1 sigma$ and $2 sigma$ contours; blue $= p$, red $= q$). The reverse-KL optimum matches the precision diagonal and is over-compact (variance $1 - rho^2 = 0.19$); the forward-KL optimum matches moments and over-covers.],
) <fig-2d-ours>

#figure(
  image("figs/fig_alpha_sweep.svg", width: 100%),
  caption: [Our recreation of the $alpha$-divergence sweep in @fig-alpha, on our asymmetric mixture, with $alpha = plus.minus 5$ standing in for $alpha arrow plus.minus infinity$. The family interpolates from mode-locking to mass-covering.],
) <fig-alpha-ours>

#figure(
  image("figs/fig_js_sweep.svg", width: 76%),
  caption: [Our 1-D recreation of the $"JS"_pi$ sweep in @fig-js. As $pi arrow 0$ the objective behaves like (rescaled) forward KL; as $pi arrow 1$, like reverse KL — the interpolation Huszár proposed as the "ideal perceptual objective" knob.],
) <fig-js-ours>

#figure(
  image("figs/fig_minillm_3peak.svg", width: 74%),
  caption: [Our recreation of MiniLLM's own folklore figure (@fig-minillm-orig): a three-peak Gaussian mixture (weights $0.1, 0.15, 0.75$) fit with a single Gaussian. The forward-KL fit is the (exact) moment match and covers all three peaks; the reverse-KL fit locks onto the dominant right-hand mode.],
) <fig-minillm-ours>

== How LLM training papers motivate reverse KL <sec-folklore-llm>

Which papers appear here is itself a methodological choice. An earlier draft of this section collected papers *because* they invoke the mode-seeking folklore — a selection rule that cannot show how reverse-KL adopters actually motivate the choice, since it filters out every adopter who reasons differently, or not at all. The entries below are instead selected on *usage*: the papers and production reports of @sec-practice that adopt reverse KL, argue about its direction, or reject it, grouped by the motivation they actually give. Two caveats remain: silent adopters leave nothing to quote (they are listed explicitly in @sec-motive-precedent), and any quote collection still finds folklore vocabulary more easily than silence.

=== The folklore as the stated motivation <sec-motive-folklore>

The folklore is the dominant stated motivation in the 2023–24 research literature — sometimes with the textbook argument spelled out, more often by citation chain or the bare words "mode-seeking". Each entry reproduces the exact citations the paper attaches to the claim; where a paper draws its own folklore figure, we reproduce it beside the entry. Notably, none of these papers restates the capacity-limitation or global-optimality assumptions on which the original argument rests (@sec-standard-claims) when invoking the folklore; the closest approach is PromptKD's ablation conjecture, which rediscovers the capacity condition empirically rather than citing it.

*MiniLLM* @gu2024minillm *(distillation).* The introduction motivates the switch to reverse KL by indicting the forward direction:

#quote(block: true, attribution: [@gu2024minillm · #emph[MiniLLM: Knowledge Distillation of Large Language Models]])[Minimizing forward KLD causes $q_theta$ to assign unreasonably high probabilities to the void regions of $p$ [MG19] and produces very unlikely samples under $p$ during free-run generation [Hus15].]

and the method section states the reverse direction's virtue with a four-citation cluster:

#quote(block: true, attribution: [@gu2024minillm · #emph[MiniLLM: Knowledge Distillation of Large Language Models]])[Minimizing reverse KLD has been shown to cause the mode-seeking behavior in generative modeling [Hus15, NCT16, CDP+18, LPSK23], where $q_theta$ assigns high probabilities to $p$'s large modes and ignore the small ones.]

In MiniLLM's bibliography, [Hus15] is Huszár's note @huszar2015how, [NCT16] is f-GAN @nowozin2016fgan, [MG19] is Malinin and Gales's reverse-KL training of prior networks @malinin2019reverse, and [CDP+18] and [LPSK23] are a symmetric-VAE paper (Chen et al., 2018) and a vision self-distillation paper (Lee et al., 2023); the introduction's version of the void-regions claim additionally cites [M+05] — Minka's report, listed there as "Tom Minka et al. … Technical report, Citeseer, 2005". MiniLLM also draws the folklore itself (@fig-minillm-orig).

#figure(
  image("figs/originals/minillm_fig2.png", width: 44%),
  caption: [MiniLLM's own folklore figure (its Fig. 2): a three-peak Gaussian mixture fit with a single Gaussian under forward and reverse KLD. Reproduced from @gu2024minillm. Our recreation: @fig-minillm-ours.],
) <fig-minillm-orig>

*GKD* @agarwal2024gkd *(distillation).* The background section states the claim in full, citing only the paper's own appendix figure:

#quote(block: true, attribution: [@agarwal2024gkd · #emph[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]])[Given model capacity mismatch, when approximating $P(C)$ using a distribution $Q_theta (C)$, minimizing the reverse and forward KL results in mean and mode-seeking behavior (Figure A.16).]

That figure's caption is the textbook claim verbatim:

#quote(block: true, attribution: [@agarwal2024gkd · #emph[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]])[[Reverse KL] is mode-seeking as it forces $Q_theta$ to be zero where $P$ is zero and hence makes it concentrate on one of the modes.]

with the trade-off stated in the body text:

#quote(block: true, attribution: [@agarwal2024gkd · #emph[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]])[[Mode-seeking divergences] prioritize the tokens where the teacher assigns high probability, which can avoid low-quality generations but at the expense of less diverse generations for a given input.]

None of these statements carries a citation; the sole source anywhere in the chain is the caption's closing "See Le (2017) to replicate this plot", which resolves in GKD's bibliography to a personal blog note @le2017reverseforward. Fig. A.16 is reproduced in @fig-gkd-orig.

#figure(
  image("figs/originals/gkd_figA16.png", width: 72%),
  caption: [GKD's own folklore figure (its Fig. A.16), whose caption is quoted above. Reproduced from @agarwal2024gkd.],
) <fig-gkd-orig>

*PromptKD* @kim2024promptkd *(distillation).* The third member of the "recent methods" trio the critics name below, it gives the direction exactly one sentence:

#quote(block: true, attribution: [@kim2024promptkd · #emph[PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning]])[Regarding the two objectives, we opt for minimizing the reverse KL divergence instead of the forward KL divergence to measure the discrepancy, as it exhibits mode-seeking behavior (Nowozin et al. 2016) and benefits generation tasks.]

The sole citation is f-GAN @nowozin2016fgan. The two axes are motivated separately, GKD-style: the student-generated pseudo-targets rest on "the insight (Agarwal et al. 2024) that incorporating sentences that the model can generate during free-run generation into the training process can mitigate exposure bias" — a data argument — while the sentence above carries the direction for the two prompt-tuning objectives; the student loss (reverse KL as well) gets no argument of its own. It draws no folklore figure, and its own appendix ablation — swapping each prompt-tuning loss to forward KL — deflates the choice:

#quote(block: true, attribution: [@kim2024promptkd · #emph[PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning]])[[E]xperimental results indicate that using reverse KL divergence yields the best performance. However, there is barely any significant difference. We conjecture that since the model distribution being trained is derived from the teacher, resulting in similar or even more modes in distribution, which prevent undesirable behaviors such as mode-covering even during forward KL divergence minimization.]

The conjecture is Minka's capacity conditioning rediscovered empirically (@sec-standard-claims): the trained distribution here is the prompted *teacher*, which has capacity to spare — so the direction stops mattering. (The student-loss direction is not ablated.)

*f-DISTILL* @wen2023fdistill *(distillation).* It names the failure mode directly, citing no source for it:

#quote(block: true, attribution: [@wen2023fdistill · #emph[f-Divergence Minimization for Sequence-Level Knowledge Distillation]])[RKL distillation has the opposite problem, known as mode collapsing, where the student only learns one or a few modes of the teacher distribution. This is because the RKL loss would be large, if $q_theta (Y')$ is high but $p(Y')$ is low for some $Y'$.]

Its Fig. 1 draws the folklore as "typical phenomena" (@fig-fdistill-orig).

#figure(
  image("figs/originals/fdistill_fig1.png", width: 60%),
  caption: [f-DISTILL's own folklore figure (its Fig. 1): "Typical phenomena of (a) KL distillation and (b) reverse KL distillation. Gray curve: teacher distribution. Green curve: student distribution." Reproduced from @wen2023fdistill.],
) <fig-fdistill-orig> PRML @bishop2006prml does appear in f-DISTILL's bibliography — but attached to the remark that cross-entropy training minimizes a KL, not to this claim.

*DistiLLM-2* @ko2025distillm2 *(distillation).* It gives the mechanism-level version that motivates its direction–data pairing:

#quote(block: true, attribution: [@ko2025distillm2 · #emph[DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs]])[KL increases $q_theta (dot|x)$ in regions where $p(dot|x)$ are high (i.e., pulling-up effect) … RKL attempts to reduce the ratio $q_theta (dot|x) slash p(dot|x)$, consequently $q_theta (dot|x)$ decreases in region[s] where $p(dot|x)$ [is] small (i.e., pushing-down effect).]

In the paper that passage opens "As shown in Figure 1, KL increases …": the support is the paper's own toy figure, with no external source. Panel (a) of that figure — reproduced in @fig-distillm2-orig — analyzes, per its caption, "long-tailed toy data introduced in Wu et al. (2024)": the toy data of the folklore's critics (@sec-folklore-crit), repurposed to argue the folklore's mechanism.

#figure(
  image("figs/originals/distillm2_fig1a.png", width: 40%),
  caption: [DistiLLM-2's own folklore figure (its Fig. 1(a)): KL and RKL "analyzed for long-tailed toy data introduced in Wu et al. (2024)". Reproduced from @ko2025distillm2.],
) <fig-distillm2-orig>

*Entropy-aware OPD* @jin2026entropy *(distillation, 2026).* The 2026 wave states the folklore as its own finding:

#quote(block: true, attribution: [@jin2026entropy · #emph[Entropy-Aware On-Policy Distillation of Language Models]])[However, we show that the mode-seeking property of reverse KL reduces generation diversity and yields unstable learning signals when the teacher distribution has high entropy.]

The abstract sentence carries no citation ("we show"); the body's statement of the claim does carry one:

#quote(block: true, attribution: [@jin2026entropy · #emph[Entropy-Aware On-Policy Distillation of Language Models]])[Since the expectation is taken over student samples, [the reverse KL objective] penalizes generated tokens that the teacher considers unlikely, but ignores teacher modes that the student does not visit. It induces #emph[mode-seeking] behavior, encouraging the student to concentrate probability mass on a single high-likelihood mode of the teacher while ignoring others [29].]

Its [29] is Minka's report @minka2005divergence — making this one of the few invoking papers to reach the primary source.

*AKL (Wu et al.)* @wu2025rethinking *(distillation; the critics).* They document the citation chain explicitly before attacking it:

#quote(block: true, attribution: [@wu2025rethinking · #emph[Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models]])[Therefore, recent methods (Gu et al., 2023; Agarwal et al., 2024; Kim et al., 2024) argue that RKL is more suitable than FKL to align the logits from LLMs since RKL avoids learning too many long-tail variants.]

Here Gu et al., 2023 is MiniLLM, Agarwal et al., 2024 is GKD, and Kim et al., 2024 is PromptKD @kim2024promptkd (above). Wu et al. are also the only invoking authors who return to the primary source — their analysis opens "Based on the analysis in Minka et al. (2005), reverse KL is zero forcing because $p = 0$ forces $q = 0$" — and their Fig. 1 (@fig-wu-orig) is captioned "The toy example in Gu et al. (2023)": MiniLLM's Gaussian figure, redrawn by the folklore's critics.

#figure(
  image("figs/originals/wu_fig1.png", width: 44%),
  caption: [Wu et al.'s Fig. 1 — "The toy example in Gu et al. (2023)": MiniLLM's figure redrawn by the folklore's critics. Reproduced from @wu2025rethinking.],
) <fig-wu-orig>

*f-DPO* @wang2024beyond *(alignment).* Motivating divergence constraints beyond reverse KL:

#quote(block: true, attribution: [@wang2024beyond · #emph[Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints]])[The mode-seeking property of reverse KL divergence tends to reduce diversity in generation (Wiher et al., 2022; Khalifa et al., 2021; Perez et al., 2022; Glaese et al., 2022), which, although beneficial for optimizing alignment performance, may limit the model's potential (e.g., user engagement).]

Its four citations resolve to a study of decoding strategies (Wiher, Meister, and Cotterell, TACL 2022), GDC @khalifa2021distributional, red-teaming (Perez et al., EMNLP 2022), and Sparrow (Glaese et al., 2022) — alignment and decoding papers, not divergence theory. f-DPO also draws its own folklore figure (@fig-fdpo-orig).

#figure(
  image("figs/originals/fdpo_fig1.png", width: 30%),
  caption: [f-DPO's own folklore figure (its Fig. 1): "The mode seeking and mass covering behaviors of reverse KL and forward KL." Reproduced from @wang2024beyond.],
) <fig-fdpo-orig>

*BOND* @sessa2024bond *(alignment).* It declares the folklore common knowledge while blaming both directions in one breath (recall BOND's swapped letters, @sec-intro: their $p$ is the trained policy):

#quote(block: true, attribution: [@sessa2024bond · #emph[BOND: Aligning LLMs with Best-of-N Distillation]])[…the reverse KL$(p parallel q)$ is well-known to have a mode-seeking effect, steering policy $p$ to produce generations that have a high likelihood according to $q$ (Agarwal et al., 2024). While the forward KL may produce over-spread distributions, the backward KL can lead to policy and entropy collapses.]

The sole citation backing "well-known" is (Agarwal et al., 2024) — GKD again. BOND's figures are training-time KL curves, not folklore toys.

*f-DPG* @go2023aligning *(alignment).* It ties the LLM-era claim back to its generative-models lineage:

#quote(block: true, attribution: [@go2023aligning · #emph[Aligning Language Models with Preferences through f-divergence Minimization]])[…low diversity but high quality for RKL-DPG, compared to other f-DPGs, suggesting it captures a subset of the target distribution ("mode collapse"), as commonly observed in other generative models (Huszar, 2015; Che et al., 2017; Mescheder et al., 2018).]

Its lineage citations are Huszár @huszar2015how and two GAN papers: mode-regularized GANs (Che et al., ICLR 2017) and a convergence analysis (Mescheder et al., ICML 2018).

*Thinking Machines Blog* @lu2025onpolicydistillation

#quote(block: true, attribution: [@lu2025onpolicydistillation · #emph[On-Policy Distillation]])[[Reverse KL] is "mode seeking": it learns one specific behavior (the teacher's) instead of spreading its distribution across several suboptimal options.]

*Minitron* @muralidharan2024minitron *(technical report; secondhand).* It transmits the folklore as received wisdom while reporting the opposite result in its own regime:

#quote(block: true, attribution: [@muralidharan2024minitron · #emph[Compact Language Models via Pruning and Knowledge Distillation]])[Recent work has shown R-KLD [15, 27] to be a better fit than KLD in the SFT/instruction-following setting, and Agarwal et al. [2] claim the choice of loss is task-dependent.]

— followed two sentences later by the finding that forward KL wins for pruned-base-model retraining (@sec-reports). In Minitron's numbered bibliography, [15] is MiniLLM @gu2024minillm, [27] is DistiLLM @ko2024distillm, and [2] is GKD @agarwal2024gkd. It draws no folklore figure.

Traced one hop back, the citation chain is short and narrow. The technical report cites the LLM-distillation papers; the alignment papers cite GKD and empirical alignment work; the distillation papers cite the 2015–2018 generative-models literature (Huszár, f-GAN, GAN analyses — PromptKD's entire support is its lone f-GAN citation) or nothing at all (GKD's body text, f-DISTILL, DistiLLM-2); and the two figure-level invocations rest on blog posts (GKD's Fig. A.16 on Le's 2017 note @le2017reverseforward; TML's sidenote on Jang's 2016 tutorial @jang2016variational). The toy figure propagates like the claims: six of the invokers draw one (MiniLLM, GKD, f-DISTILL, DistiLLM-2, f-DPO, and Wu et al. — the last redrawing MiniLLM's, with DistiLLM-2 borrowing the critics' toy data). The textbook treatments are nearly absent: no invoking paper cites MacKay or Murphy; f-DISTILL lists PRML @bishop2006prml but attaches it elsewhere; and Minka is reached only three times — by MiniLLM's [M+05], by the critics @wu2025rethinking, and by entropy-aware OPD's [29] @jin2026entropy.

=== Tractability and operational control: the RLHF lineage <sec-motive-tract>

The pipeline that made reverse KL ubiquitous (@sec-rl) never argues the direction at all: reverse KL is the only direction estimable from on-policy samples given log-prob access to the reference — the same tractability asymmetry that fixed the direction of variational inference (@sec-standard-claims). The stated justifications are operational, and none of them cites a source:

#quote(block: true, attribution: [@ziegler2019finetuning · #emph[Fine-Tuning Language Models from Human Preferences]])[This term has several purposes: it plays the role of an entropy bonus, it prevents the policy from moving too far from the range where r is valid, and … it also is an important part of the task definition.]

#quote(block: true, attribution: [@stiennon2020learning · #emph[Learning to Summarize from Human Feedback]])[First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode. Second, it ensures the policy doesn't learn to produce outputs that are too different from those that the reward model has seen during training.]

DPO inherits the same reverse-KL constraint in closed form and justifies it the same way:

#quote(block: true, attribution: [@rafailov2023direct · #emph[Direct Preference Optimization: Your Language Model is Secretly a Reward Model]])[The added constraint is important, as it prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse to single high-reward answers.]

Note the sign: relative to *unregularized* reward maximization, the same intuitions the folklore turns against reverse KL are invoked *for* it — the folklore supplies the vocabulary for both readings. f-DPO adds an algebraic motivation: among divergence constraints, reverse KL is the one whose partition function cancels in DPO-style derivations @wang2024beyond.

=== RL synergy, unhackability, density, and compute <sec-motive-synergy>

- *Thinking Machines blog* @lu2025onpolicydistillation — the post's motivation portfolio leads not with the folklore but with RL structure and reward safety:

  #quote(block: true, attribution: [@lu2025onpolicydistillation · #emph[On-Policy Distillation]])[Reverse KL has natural synergy with RL, which generally optimizes a form of sequence-level reverse KL induced by the reward model. However, unlike most reward models in practice, the reverse KL is "unhackable" in the sense that low KL always corresponds to a high probability of desirable behavior from the teacher model's point of view.]

  with density and compute close behind ("distillation reaches the teacher's level of performance approximately 7-10x faster than RL"; @sec-distill). The folklore appears third in the list, as an argument *for* reverse KL:

  #quote(block: true, attribution: [@lu2025onpolicydistillation · #emph[On-Policy Distillation]])[[Reverse KL] is "mode seeking": it learns one specific behavior (the teacher's) instead of spreading its distribution across several suboptimal options.]

  The only source attached to "mode seeking" is a sidenote — "See Eric Jang's post for more discussion of mode seeking behaviors" — linking Jang's 2016 variational-inference tutorial @jang2016variational; the neighbouring claim that reverse KL "reduces exposure bias" links Bengio et al.'s scheduled-sampling paper (arXiv:1506.03099). The post's graphics are method diagrams; it draws no folklore figure.

- *Qwen3* @qwen2025qwen3 — efficiency and exploration, with no direction argument (the direction is not even stated; @sec-reports):

  #quote(block: true, attribution: [@qwen2025qwen3 · #emph[Qwen3 Technical Report]])[…distillation achieves significantly better performance than reinforcement learning while requiring approximately only 1/10 of the GPU hours. Furthermore, distillation from teacher logits enables the student model to expand its exploration space and enhance its reasoning potential, as evidenced by the improved pass\@64 scores.]

- *Nemotron-Cascade 2* @nvidia2026nemotroncascade2 — operational stabilization:

  #quote(block: true, attribution: [@nvidia2026nemotroncascade2 · #emph[Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation]])[[Multi-domain On-policy Distillation] serves as a critical stabilization point … effective at recovering benchmark performance that may have regressed during earlier, more specialized stages.]

=== Derived optimality <sec-motive-optimality>

The variational reading motivates the direction as a theorem rather than a preference: the KL-regularized RLHF objective *is* reverse-KL variational inference toward the tilted posterior @korbak2022rl, and EXO selects reverse KL because it is "required by the alignment objective" — while arguing that DPO's own objective "corresponds to the forward KL" @ji2024towards (both in @sec-rl).

=== Estimator-level motivations <sec-motive-estimator>

Some adopters motivate not the divergence but the estimator built on it.

- *DistiLLM* — selects *skewed* reverse KL for bounded gradients and estimator stability, not mode behavior @ko2024distillm (@sec-distill).

- *Nemotron 3 Ultra* — keeps the divergence but selects the sampled-token estimator after full-distribution matching underperformed (@sec-reports) @nvidia2026nemotron3ultra.

- *DeepSeek-V4* — argues the opposite side of the same estimator question:

  #quote(block: true, attribution: [@deepseek2026v4 · #emph[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence]])[…we adopt full-vocabulary logit distillation in our OPD. Preserving the complete logit distribution in calculating reverse KL loss yields more stable gradient estimates and ensures faithful distillation of the teachers' knowledge.]

- *DeepSeek-V3.2* — keeps the RLHF penalty while correcting its k3 estimator (@sec-deriv) @deepseek2025v32.

=== Adoption by precedent, without a direction argument <sec-motive-precedent>

By 2026 the frontier consolidation stages adopt reverse-KL on-policy distillation with no direction argument anywhere in the report: full-text search finds no mode-seeking or mass-covering language in DeepSeek-V4, Kimi K3, or GLM-5.

- *Kimi K3* — consolidation:

  #quote(block: true, attribution: [@kimi2026k3 · #emph[Kimi K3: Open Frontier Intelligence]])[We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model [76, 136, 29].]

  its precedents resolving to the Thinking Machines post [76], MiMo-V2-Flash [136], and DeepSeek-V4 [29].

- *GLM-5* — cross-stage recovery:

  #quote(block: true, attribution: [@glm2026glm5 · #emph[GLM-5: From Vibe Coding to Agentic Engineering]])[To mitigate this issue, we perform on-policy cross-stage distillation as the final stage, adopting an on-policy distillation algorithm [14; 52; 51; 28] to swiftly recover the skills acquired in earlier SFT and RL stages (Reasoning RL and General RL).]

  with precedents resolving to MiniLLM [14], Qwen3 [52], MiMo-V2-Flash [51], and the Thinking Machines post [28].

- *DeepSeek-V4* — states the objective bare:

  #quote(block: true, attribution: [@deepseek2026v4 · #emph[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence]])[…a single unified model is trained through on-policy distillation, wherein the unified model acts as the student learning to optimize the reverse KL loss with teacher models.] The folklore chain of @sec-motive-folklore has become a precedent chain: at the frontier, the direction is inherited from Qwen3, the Thinking Machines post, and MiniLLM the way an optimizer default is — and the argument, when one is given at all, is operational rather than distributional.

The folklore chain of @sec-motive-folklore has become a precedent chain: at the frontier, the direction is inherited from Qwen3, the Thinking Machines post, and MiniLLM the way an optimizer default is — and the argument, when one is given at all, is operational rather than distributional.

== Criticisms and complications <sec-folklore-crit>

The folklore has accumulated serious qualifications, several from within its own canon.

*Capacity.* If $q$ can represent $p$ exactly, every f-divergence has the same global optimum; the direction distinction is a misspecification phenomenon. f-GAN states it for generative models:

#quote(block: true, attribution: [@nowozin2016fgan · #emph[f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization]])[…when the generative model is misspecified and does not contain the true distribution, the divergence function used for estimation has a strong influence on which model is learned.]

#cite(<wu2025rethinking>, form: "prose") make the LLM-specific version precise — the classical picture assumes a Gaussian/continuous $q$, but a softmax head can represent any categorical distribution, so FKL and RKL share the optimum:

#quote(block: true, attribution: [@wu2025rethinking · #emph[Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models]])[Contrary to prior assertions that reverse Kullback-Leibler (RKL) divergence is mode-seeking and thus preferable over the mean-seeking forward Kullback-Leibler (FKL) divergence, this study empirically and theoretically demonstrates that neither mode-seeking nor mean-seeking properties manifest in KD for LLMs.]

The observable difference is transient — "FKL focuses on the head part and RKL focuses on the tail part at the beginning epochs". (Scope caveat: their theory concerns representable optima and their experiments use small students; it does not settle the capacity-limited regime the folklore invokes.)

*Mode collapse is not (only) the divergence's fault.* #cite(<goodfellow2016nips>, form: "prose"), refuting the GAN version of the folklore that he had himself helped spread:

#quote(block: true, attribution: [@goodfellow2016nips · #emph[NIPS 2016 Tutorial: Generative Adversarial Networks]])[GANs often choose to generate from very few modes; fewer than the limitation imposed by the model capacity. The reverse KL prefers to generate from as many modes of the data distribution as the model is able to; it does not prefer fewer modes in general. This suggests that mode collapse is driven by a factor other than the choice of divergence.]

Conversely, #cite(<arjovsky2017towards>, form: "prose") show the popular $-log D$ GAN loss is reverse-KL-dominated, with the sharpest theoretical statement of the asymmetry:

#quote(block: true, attribution: [@arjovsky2017towards · #emph[Towards Principled Methods for Training Generative Adversarial Networks]])[…this KL assigns an extremely high cost to generating fake looking samples, and an extremely low cost on mode dropping.]



*Optimization, not objective.* #cite(<soletskyi2024theoretical>, form: "prose") show reverse-KL mode collapse in VI occurs even when "the variational family is statistically well-specified":

#quote(block: true, attribution: [@soletskyi2024theoretical · #emph[A Theoretical Perspective on Mode Collapse in Variational Inference]])[…mode collapse is not a by-product of insufficient model expressivity, but a consequence of the optimization procedure.]

Together with Bishop's multiple local minima and MacKay's exercise exhibiting three distinct reverse-KL minima @mackay2003itila, this relocates much of the folklore from the objective to the dynamics.

*The compactness "folk-theorem".* #cite(<turner2011two>, form: "prose") call the VI compactness claim a folk-theorem outright:

#quote(block: true, attribution: [@turner2011two · #emph[Two Problems with Variational Expectation Maximisation for Time-Series Models]])[It is well known that variational approximations, like those derived in the vEM E-Step, tend to be compact (MacKay, 2003). In other words, the variational approximation has a tendency to have a smaller entropy than the true distribution. The evidence for this folk-theorem is reviewed in the next section.]

They also cite counter-examples (Turner, Berkes, Sahani and MacKay, 2008); Murphy's PML2 carries the same caveat @murphy2023pml2. #cite(<li2023modeseeking>, form: "prose") show that under a rigorous definition of mode-seeking, Jensen-Shannon qualifies while "standard KL" — the forward/MLE direction in their GAN estimation setting — does not; the labels have to be re-derived per divergence rather than assumed. In RL, #cite(<chan2022greedification>, form: "prose") find the direction's practical stakes context-dependent: reverse KL carries stronger policy-improvement guarantees, forward KL more exploration, with often modest differences.

*Folklore about LLM mode collapse specifically.* The most-cited folklore source for "RLHF causes mode collapse", janus's _Mysteries of mode collapse_ @janus2022mode, documents striking sharpening (token confidences "frequently above 99%") but carries an in-post correction that the model in question, `text-davinci-002`, "was not trained with RLHF" (it was supervised FeedME tuning) — the phenomenon is real, its original attribution was wrong, and no KL objective is implicated by that source. Rigorous evidence on RLHF and diversity is deferred to @sec-empirical.

Three 2025–2026 papers sharpen the criticism inside the LLM training stack itself — the first at the level of the KL-regularized RL *objective*, the other two at the level of the on-policy-distillation *estimator*, where the failure mode is conditional (prefix-coupled) rather than modal.

=== KL-regularized RL specifies unimodal targets by design <sec-crit-gxchen>

- #cite(<gxchen2025modecollapse>, form: "prose") show the mode-seeking/mass-covering labels do not transfer to KL-*regularized* reward maximization: both directions admit closed-form optimal-target families (their Eqs. 1–5),
$ J_beta (pi_theta) = bb(E)_(y tilde.op pi_theta) [R(y)] - beta D_"KL" (pi_theta parallel pi_"ref"), quad quad G_beta (y) prop pi_"ref" (y) thin e^(R(y) slash beta); $
$ J_"fwd" (pi_theta) = bb(E)_(y tilde.op pi_theta) [R(y)] - beta D_"KL" (pi_"ref" parallel pi_theta), quad quad G_"fwd" (y) = (beta thin pi_"ref" (y)) / (Lambda - R(y)), quad Lambda > max_y R(y), $
  so which modes survive is a property of the *target family* — set by the regularization strength $beta$ and the relative scales of rewards and reference probabilities — not of the divergence direction.
- Under the common configuration (equal verifiable rewards, small $beta$), the specified target is essentially unimodal: "by construction, KL-regularized RL with equal rewards never promotes a low support answer" — making "diversity collapse a natural consequence of correctly solving the RL problem", with either KL direction.
- For this review: a *target-design* explanation of mode collapse in the regularizer role, complementing the capacity critique of @wu2025rethinking and the optimization critique of @soletskyi2024theoretical — a third mechanism by which the folklore's picture misattributes the observed behavior to the divergence.
- Their fix, Mode Anchored Reward Augmentation (MARA), edits reward magnitudes ($macron(r)_i = R(z) + beta (log pi_"ref" (z) - log pi_"ref" (y_i))$ for high-reward samples) so the induced target spreads mass across all high-reward modes — again independent of KL direction.

#quote(block: true, attribution: [@gxchen2025modecollapse · #emph[KL-Regularized Reinforcement Learning is Designed to Mode Collapse]])[It is commonly believed that optimizing the reverse KL divergence results in 'mode seeking', while optimizing forward KL results in 'mass covering', with the latter being preferred if the goal is to sample from multiple diverse modes. We show -- mathematically and empirically -- that this intuition does not necessarily transfer well to doing reinforcement learning with reverse/forward KL regularization.]

#quote(block: true, attribution: [@gxchen2025modecollapse · #emph[KL-Regularized Reinforcement Learning is Designed to Mode Collapse]])[…commonly used settings such as low regularization strength and equal verifiable rewards tend to specify unimodal target distributions, meaning the optimization objective is, by construction, non-diverse.]

=== The low-KL "agreement trap": local agreement without corrective signal <sec-crit-agreementtrap>

- #cite(<xin2026agreementtrap>, form: "prose") study the standard on-policy distillation objective (their Eq. 2) — the token-level reverse KL on student rollouts, i.e. the frozen-visitation objective of @sec-deriv,
$ cal(L)_"OPD" (theta) = bb(E)_(x, thin hat(y) tilde.op pi_theta (dot|x)) [ sum_(t=1)^T D_"KL" (pi_theta (dot|x, hat(y)_(<t)) parallel pi_T (dot|x, hat(y)_(<t))) ]. $
- The trap: once a rollout drifts into an unrecoverable prefix, the *conditional* teacher locally agrees with the degraded state — the per-token reverse KL is low precisely where supervision is most needed. More than 60% of rollouts exhibit at least one extended low-KL segment, and once entered, the divergence tends to stay below its pre-segment level.
- The pathology is a property of prefix-conditioning, with no analog in the folklore's unconditional fitting picture: per-prefix conditionals cannot veto a globally bad trajectory. Their remedy treats it as a *data* problem, not a divergence-direction problem: KAT monitors the sliding-window mean $z_t = 1/W sum_(i=t-W+1)^t d_i$ of per-token divergences (their Eq. 4) and terminates the rollout once enough consecutive windows fall below a calibrated threshold (their Eq. 7), improving avg\@k by 2.66, pass\@k by 3.43, and cutting rollout length by 59.7% on four math benchmarks.

#quote(block: true, attribution: [@xin2026agreementtrap · #emph[Escaping the KL Agreement Trap in On-Policy Distillation]])[When the student drifts into an unrecoverable prefix, the teacher may locally agree with the degraded state, producing low reverse KL but little corrective training signal. … We identify this persistent regime as a low-KL agreement trap.]

#quote(block: true, attribution: [@xin2026agreementtrap · #emph[Escaping the KL Agreement Trap in On-Policy Distillation]])[Conditioned on this degraded state, the teacher's token-level distribution may carry little corrective information, instead producing continuations that are plausible under the erroneous prefix and therefore close to the student's distribution.]

=== Degenerate agreement and student-induced teacher hacking <sec-crit-teacherhacking>

- #cite(<yu2026mismatch>, form: "prose") analyze the same objective family (their Eq. 1, the sampled-token estimator of @sec-table1); their sharper claim is that the agreement signal is *gameable*: because the student controls the visited contexts, it can manufacture prefixes that bend the teacher's conditionals toward its own behavior.
- Concretely, repetition loops: repeated prefixes make the teacher increasingly favor the same continuation, driving the measured token-level KL to a median of 0.0004 nats on globally degenerate responses. This is the pathology MiniLLM's teacher-mixed sampling @gu2024minillm patched in 2023 — reward hacking via repetition (@sec-distill) — resurfacing at 2026 scale under the sampled-token estimator, now with a mechanism: conditional acceptance by the teacher of student-manufactured contexts.
- Their fix (TIDE) is asymmetric by token type: bounded Hellinger-shaped suppression of student-*excess* tokens and analytic teacher top-K cross-entropy injection for *deficit* tokens,
$ g_"exc" = -bb(E) [ sum_t m_t^- thin "sg"[h(a_t)] thin log q_theta (o_t|s_t) ], quad h(a) = 2(e^(a slash 2) - 1), $
$ cal(L)_"def" = -bb(E) [ sum_t m_t^+ sum_(v in cal(K)_t) macron(p)_t (v) log q_theta (v|s_t) ], $
  with $m_t^(minus.plus)$ masking excess/deficit positions and $cal(K)_t$ the teacher top-K (their §3, Eq. 4). Under strong mismatch (Qwen3-8B $arrow.r$ Qwen3-1.7B-Base): Avg\@8 6.9% $arrow.r$ 20.3%, response length 22,395 $arrow.r$ 7,294 tokens, missing-boxed-answer rate 65.5% $arrow.r$ 5.4%.

#quote(block: true, attribution: [@yu2026mismatch · #emph[Mismatch Matters: On-Policy Distillation Beyond Token Agreement]])[…degenerate agreement, where students exploit repetitive loops to achieve near-perfect token agreement with the teacher despite globally flawed responses…]

#quote(block: true, attribution: [@yu2026mismatch · #emph[Mismatch Matters: On-Policy Distillation Beyond Token Agreement]])[…the student can reduce the measured teacher–student divergence by generating contexts that make the teacher's outputs more aligned with its own…]

= Reverse KL as a training objective in practice <sec-practice>

== Distillation research <sec-distill>

*MiniLLM* @gu2024minillm made sequence-level reverse KL the *primary* distillation objective: $min_theta "KL"[q_theta parallel p]$ with responses sampled from the student, optimized by policy gradient. Its motivation restates the folklore (quoted in @sec-folklore-llm), plus a values claim:

#quote(block: true, attribution: [@gu2024minillm · #emph[MiniLLM: Knowledge Distillation of Large Language Models]])[[G]enerating one correct response is sufficient … for scenarios demanding high truthfulness and reliability.]

Three estimator-level patches (single-step decomposition, teacher-mixed sampling, length normalization) are needed to make the vanilla objective train at all; each patches a concrete pathology (variance accumulation, reward hacking via repetition, short-response bias) — evidence, before any code audit, that the naive sequence-level estimator is degenerate (@sec-estimators). The paper itself concedes the divergence-level cost:

#quote(block: true, attribution: [@gu2024minillm · #emph[MiniLLM: Knowledge Distillation of Large Language Models]])[Models optimized by minimizing reverse KLD are likely to lose modes, which affects generation diversity.]

(Citation hygiene: arXiv v6 of 2026 retitled the paper "MiniLLM: On-Policy Distillation of Large Language Models", colliding with GKD's title; cite by arXiv ID 2306.08543.)

*GKD* @agarwal2024gkd decoupled the two ingredients MiniLLM had fused: (i) *where prefixes come from* (a fraction $lambda$ of batches use student rollouts — motivated by "train-inference distribution mismatch", an argument about data, not divergence direction, inherited from ImitKD @lin2020imitkd) and (ii) *which divergence* (generalized JSD($beta$) interpolating FKL at $beta=0$ to RKL at $beta=1$, computed full-vocabulary per position). Its empirical conclusion is deflationary for the direction question: the on-policy data axis gives the robust gains, while "the optimal divergence seems to be task-dependent" and "the choice of divergence is temperature-dependent" — mode-seeking helps under temperature sampling and hardly matters under greedy decoding.

*PromptKD* @kim2024promptkd carries the recipe into parameter-efficient form: reverse KL throughout, on student-generated pseudo-targets, with a soft prompt (0.0007% of the teacher's parameters) tuned so the teacher emits "student-friendly" distributions. Its own ablation is deflationary in the same way as GKD's: swapping directions makes "barely any significant difference" (quoted in @sec-folklore-llm).

*DistiLLM* @ko2024distillm attacks the estimator, not the direction: skewed divergences $D_"SKL"^((alpha)) = "KL"(p parallel alpha p + (1-alpha) q)$ and $D_"SRKL"^((alpha)) = "KL"(q parallel (1-alpha) p + alpha q)$ bound the gradient ratio and stabilize training, while an adaptive replay scheme avoids MiniLLM/GKD's cost of constant student generation ("up to 80%" of training time). SRKL is their best instruction-following loss. *DistiLLM-2* @ko2025distillm2 makes the direction–data pairing explicit: skewed *forward* KL on teacher-generated responses, skewed *reverse* KL on student-generated responses, on the pulling-up/pushing-down grounds quoted in @sec-folklore-llm; notably, interpolating the two losses on the *same* data "does not improve performance compared to using either SKL or SRKL alone." f-DISTILL @wen2023fdistill had already found symmetric losses (JS, TVD) "consistently better than asymmetric ones (KL and RKL)" outside near-unimodal machine translation, with RKL showing the worst teacher coverage ("mode collapsing"). Speculative KD adds the complementary failure mode of pure on-policy data: early student samples are out-of-distribution for the teacher, "leading to inaccurate teacher feedback" @xu2025skd.

*On-policy distillation as an RL substitute.* The Thinking Machines blog @lu2025onpolicydistillation popularized per-token reverse KL on student rollouts as a *dense, cheap, unhackable* replacement for RLVR:

#quote(block: true, attribution: [@lu2025onpolicydistillation · #emph[On-Policy Distillation]])[We sample trajectories from the student model and use a high-performing teacher to grade each token.]

Reported results: Qwen3-8B-Base reaching AIME'24 74.4% in 1,800 GPU-hours where RL took 17,920 GPU-hours for 67.6% — "distillation reaches the teacher's level of performance approximately 7-10x faster than RL" — plus a behavior-restoration experiment (recovering instruction-following lost in mid-training without losing the newly acquired knowledge). This estimator (sampled-token log-ratio as a per-token advantage) is exactly the one the MOPD line productionized (MiMo-V2-Flash, then Nemotron; @sec-reports), and is audited in @sec-estimators.

*The 2026 on-policy-distillation wave.* Within a year the recipe hardened into standard frontier practice (@sec-reports) and acquired its own literature. A survey formalizes OPD as f-divergence minimization over student-generated data @song2026survey; entropy-aware variants respond to the finding that reverse KL "reduces generation diversity and yields unstable learning signals when the teacher distribution has high entropy" (quoted in @sec-folklore-llm) @jin2026entropy; and two failure-mode studies delimit the recipe. #cite(<liu2026teacher>, form: "prose") identify supervision-fidelity decay:

#quote(block: true, attribution: [@liu2026teacher · #emph[Your Teacher Can't Help You Here: Combating Supervision Fidelity Decay in On-Policy Distillation]])[…as student-generated prefixes lengthen, the teacher's next-token distribution becomes less confident and less discriminative. Consequently, the teacher-dependent corrective signal in reverse-KL distillation weakens, causing student drift to compound across long reasoning chains.]

and #cite(<li2026rethinking>, form: "prose") delimit when OPD works at all:

#quote(block: true, attribution: [@li2026rethinking · #emph[Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe]])[We observe a striking failure mode: a stronger teacher can completely fail to improve a student, even when a weaker teacher succeeds from lower initial alignment.]

with the governing conditions:

#quote(block: true, attribution: [@li2026rethinking · #emph[Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe]])[…(i) the student and teacher should share compatible thinking patterns; and (ii) even with consistent thinking patterns and higher scores, the teacher must offer genuinely new capabilities beyond what the student has seen during training.]

== RL and alignment <sec-rl>

*The regularizer lineage.* In the canonical RLHF pipeline the reverse KL to a reference policy enters as a penalty in the *reward channel* of PPO: $R(x,y) = r(x,y) - beta log [pi(y|x) slash rho(y|x)]$ @ziegler2019finetuning. The direction is never argued against alternatives in this lineage; the justifications are operational:

#quote(block: true, attribution: [@ziegler2019finetuning · #emph[Fine-Tuning Language Models from Human Preferences]])[This term has several purposes: it plays the role of an entropy bonus, it prevents the policy from moving too far from the range where r is valid, and … it also is an important part of the task definition.]

#quote(block: true, attribution: [@stiennon2020learning · #emph[Learning to Summarize from Human Feedback]])[First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode. Second, it ensures the policy doesn't learn to produce outputs that are too different from those that the reward model has seen during training.]

Note the historical irony: relative to *unregularized* reward maximization, the reverse-KL penalty was introduced to *prevent* collapse; the later literature blames the same term for collapse relative to *distribution-matching* alternatives (@sec-criticism). InstructGPT makes the penalty per-token:

#quote(block: true, attribution: [@ouyang2022training · #emph[Training Language Models to Follow Instructions with Human Feedback]])[In addition, we add a per-token KL penalty from the SFT model at each token to mitigate over-optimization of the reward model.]

Anthropic, observing that reward grows linearly in $sqrt("KL")$, barely needed the term at all:

#quote(block: true, attribution: [@bai2022training · #emph[Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback]])[In practice we use a very small value of $lambda_"KL" = 0.001$, which likely has a very minor impact during most of RL training (as $D_"KL" < 100$ typically), and might actually be wholly unnecessary.]

#cite(<gao2023scaling>, form: "prose") later reported a negative result for the penalty's protective value:

#quote(block: true, attribution: [@gao2023scaling · #emph[Scaling Laws for Reward Model Overoptimization]])[KL penalty ineffectiveness. In our reinforcement learning setup, using a KL penalty increases the proxy reward model score that can be achieved for a given KL divergence, but this does not correspond to a measurable improvement in the gold RM score–KL#sub[RL] frontier. However, we note this result could be particularly sensitive to hyperparameters.]



*The variational view* supplies the principled justification of the direction after the fact: the KL-regularized objective is an ELBO, i.e. exactly reverse-KL variational inference toward the tilted posterior, "$J_"KL-RL" (theta) prop -D_"KL" (pi_theta, pi^*_"KL-RL")$" with $pi^* prop pi_0 exp(r slash beta)$ @korbak2022rl — reverse because $pi_theta$ is the variational approximation. DPO inherits precisely this reverse-KL-constrained objective in closed form, crediting the constraint broadly:

#quote(block: true, attribution: [@rafailov2023direct · #emph[Direct Preference Optimization: Your Language Model is Secretly a Reward Model]])[The added constraint is important, as it prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse to single high-reward answers.]

f-DPO observes reverse KL is also *algebraically* special in DPO-style derivations (the partition function cancels) @wang2024beyond. Two later results complicate the DPO picture. EXO formalizes the RLHF-as-reverse-KL equivalence and then argues the opposite for DPO itself:

#quote(block: true, attribution: [@ji2024towards · #emph[Towards Efficient Exact Optimization of Language Model Alignment]])[…we formally demonstrate that the DPO objective corresponds to the forward KL which is essentially different from the reverse KL required by the alignment objective.]

And SimPO drops the reference anchor altogether ("eliminates the need for a reference model"):

#quote(block: true, attribution: [@meng2024simpo · #emph[SimPO: Simple Preference Optimization with a Reference-Free Reward]])[Although SimPO does not impose KL regularization, we find that a combination of practical factors ensures effective learning from preference data while maintaining generalization, leading to an empirically low KL divergence from the reference model.]

*GRPO* moved the KL from the reward into the loss, using Schulman's k3 estimator per token:

#quote(block: true, attribution: [@shao2024deepseekmath · #emph[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models]])[Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of $hat(A)_(i,t)$.]

What gradient that actually yields is the subject of @sec-deriv. Since 2025, RLVR practice has split three ways over the term:

- *Remove it.* DAPO, under a section titled "Removing KL Divergence":

  #quote(block: true, attribution: [@yu2025dapo · #emph[DAPO: An Open-Source LLM Reinforcement Learning System at Scale]])[However, during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary. Therefore, we will exclude the KL term from our proposed algorithm.]

  Dr. GRPO drops it for the same regime reason, plus cost:

  #quote(block: true, attribution: [@liu2025understanding · #emph[Understanding R1-Zero-Like Training: A Critical Perspective]])[However, RL-tuning reasoning models typically employs rule-based verifiers as r (Lambert et al., 2024), eliminating the concerns of distributional shift. This allows us to remove the KL term, which not only saves the memory and computation required by $pi_"ref"$ during training, but also potentially leads to better performance for R1-Zero-like training (Hu et al., 2025).]

  Magistral reaches the same verdict:

  #quote(block: true, attribution: [@mistral2025magistral · #emph[Magistral]])[Eliminating KL divergence. The KL divergence penalty constrains the online policy from deviating too much from a reference policy, helping to maintain alignment with the initial model. However, in GRPO, the policy diverges substantially regardless, and maintaining a copy of the reference model for KL computation incurs a compute cost we find unjustified. We remove the KL penalty entirely.]

  Nemotron-Cascade 2 does the same ("we remove KL divergence term entirely" @nvidia2026nemotroncascade2), and GSPO brackets it out without argument ("we omit the KL regularization term hereinafter for brevity, as it is not the focus of this paper" @zheng2025gspo).

- *Keep and repair it.* DeepSeek-V3.2 retains the penalty, corrects the k3 estimator with an importance-sampling ratio (@sec-deriv), and notes that "different domains benefit from varying strengths of KL regularization" @deepseek2025v32; FRPO repairs its per-token credit assignment with a future-KL return-to-go @yao2026futurekl (@sec-deriv).

- *Re-introduce it.* ProRL brings it back (with reference resets) precisely to *fight* entropy collapse @liu2025prorl.

== Production technical reports <sec-reports>

The sweep of technical reports (full dossier: `notes/D`) yields a clean pattern, summarized below. The type column combines the direction (our convention; "unspecified" means the report's text does not determine it) with the estimator support (full-vocab / truncated-support / sampled-token); the next column records how the divergence enters the training loss.

#table(
  columns: (1.0fr, 1.3fr, 0.8fr, 1.1fr, 0.5fr),
  stroke: 0.4pt + gray,
  table.header([*Report / stage*], [*Objective (as stated)*], [*Type*], [*Introduced via*], [*Sampled from*]),
  [MiMo-V2-Flash — MOPD @xiaomi2026mimov2], [introduces MOPD: "reverse KL divergence loss" (Eq. 5)], [sampled-token *reverse KL* (explicit)], [gated $"sg"[log(pi_"domain" slash pi_theta)]$ as per-token advantage in a length-normalized REINFORCE-style update (Eqs. 7–9)], [student rollouts],
  [Nemotron 3 Ultra — MOPD @nvidia2026nemotron3ultra], [explicit "negative reverse-KL objective" (Eq. 1)], [sampled-token *reverse KL* (explicit)], [$"sg"[log(pi_(T_i) slash pi_"prox")]$ as per-token advantage in a PPO-clipped update with IcePop masking (Eqs. 2–3)], [student rollouts],
  [Nemotron-Cascade 2 — MOPD @nvidia2026nemotroncascade2], ["token-level distillation advantage using reverse-KL … only on the student-sampled token"], [sampled-token *reverse KL* (explicit)], [per-token advantage in a truncated-IS, length-normalized REINFORCE update (Eqs. 2–4)], [student rollouts],
  [Nemotron 3 Ultra — MTP head], ["temperature-scaled forward-KL loss" (T$=$2)], [full-vocab forward KL], [directly optimized (auxiliary MTP loss; CE to the gold token disabled)], [own rollouts],
  [Qwen3 — off-policy distillation @qwen2025qwen3], ["outputs of teacher models … for response distillation"], [SFT on teacher samples (no KL)], [—], [teacher],
  [Qwen3 — on-policy distillation @qwen2025qwen3], ["aligning its logits with those of a teacher model … to minimize the KL divergence"], [logit-level KL, direction unspecified], [directly optimized (no equation given)], [student rollouts],
  [DeepSeek-V4 — consolidation OPD @deepseek2026v4], ["Computing the reverse KL loss … requires sampling training trajectories from the student" (Eq. 29; multi-teacher)], [*full-vocab reverse KL* (explicit)], [directly optimized (exact per-position KL via TileLang kernel; the per-token-advantage route explicitly rejected)], [student rollouts],
  [Kimi K3 — MOPD @kimi2026k3], [MOPD "to consolidate these domain-specialized capabilities … into a unified model"], [sampled-token *reverse KL* (implicit)], [per-token reward $"sg"[log(pi_"teacher" slash pi_theta)]$ clipped to $plus.minus R_"max"$ (Eq. 15)], [student rollouts],
  [GLM-5 — cross-stage distillation @glm2026glm5], [on-policy cross-stage distillation; earlier-stage checkpoints as teachers], [sampled-token *reverse KL* (implicit)], [per-token advantage $hat(A)_(i,t) = "sg"[log(pi_"teacher" slash pi_theta)]$], [student rollouts],
  [DeepSeek-V3.2 — RL (GRPO) @deepseek2025v32], [KL penalty kept; k3 estimator corrected by importance ratio (Eq. 7)], [sampled-token *reverse KL* (regularizer)], [importance-ratio-corrected k3 in the GRPO loss (cf. @eq-rhok3)], [student],
  [Magistral — RLVR @mistral2025magistral], ["We remove the KL penalty entirely"], [removed], [—], [—],
  [Llama 4 — pretraining codistillation @meta2025llama4], [Maverick from Behemoth; "novel distillation loss function that dynamically weights the soft and hard targets"], [unspecified], [directly optimized], [dataset],
  [Gemma 2 — pretraining KD @gemmateam2024gemma2], [$min_(P_S) sum_x -P_T (x|x_c) log P_S (x|x_c)$], [full-vocab forward KL (CE to soft targets)], [directly optimized (pretraining loss)], [dataset],
  [Gemma 2 — post-training @gemmateam2024gemma2], ["distillation from the teacher on the student's distribution", citing GKD and MiniLLM], [unspecified], [unspecified], [student],
  [Gemma 3 — pretraining KD @gemmateam2025gemma3], [CE on 256 sampled logits, renormalized], [truncated-support forward KL (256 sampled logits)], [directly optimized (pretraining loss)], [dataset],
  [Minitron @muralidharan2024minitron], [logit KD; "KLD is the best choice"; R-KLD ablated and worse], [full-vocab forward KL (reverse tried, failed)], [directly optimized ($cal(L)_"logits"$ term)], [dataset],
  [Nemotron-H / Nano 2 @nvidia2025nemotronh @nvidia2025nemotronnano2], ["forward KL divergence loss exclusively"], [full-vocab forward KL], [directly optimized (Minitron loss formulation)], [dataset],
  [Llama 3.2 1B/3B @meta2024llama32], [teacher "logits … used as token-level targets"], [logit-level, direction unspecified], [unspecified (blog-level)], [dataset],
  [DeepSeek-R1 distillation @deepseek2025r1], [SFT on 800k teacher-generated samples, "only SFT … no RL stage"], [SFT on teacher samples (no KL)], [—], [teacher],
  [Apple AFM 2024 — pretraining @apple2024afm], [teacher top-1 labels mixed 0.9 with true labels], [top-1 (hard-label) forward-flavored], [directly optimized (CE to label mix)], [dataset],
  [Apple AFM 2024 — RLHF (MDLOO) @apple2024afm], [$r_phi (x,y) - beta D_"KL" (pi_theta ( dot |x) parallel pi_"ref" (dot|x))$ — direction written out], [*reverse KL* (regularizer; estimator unstated)], [$beta$-penalty on the reward in the RL objective], [student],
  [Gemini 1.5 Flash @geminiteam2024gemini15], ["online distilled" from 1.5 Pro (cites GKD)], [unspecified], [unspecified], [unspecified],
  [Tülu 3 / OLMo 2 @lambert2024tulu3 @olmo2025olmo2], [$r_phi - beta "KL"[pi_theta parallel pi_"ref"]$ (boilerplate)], [*reverse KL* (regularizer; estimator unnamed)], [penalty in reward as written; k1 in PPO reward in paper-era open-instruct code], [student],
)

Reports checked with *no* KD/KL distillation stage described: Llama 3, Phi-4, MiniCPM, Hunyuan-Large/TurboS, GLM-4, SmolLM2, InternLM2, Yi, OLMo 2 (which explicitly notes distillation as a road not taken), Kimi K2 (whose OPMD objective is a squared log-ratio to the previous policy, not a KL) @kimi2025k2, ERNIE 4.5, Falcon 3 (blog-level only), Ministral, Grok, GPT-4o mini.

Three regularities stand out. First, *explicit reverse-KL training appears only in on-policy post-training distillation* — the MOPD paradigm is introduced by Xiaomi's MiMo-V2-Flash (January 2026), with an explicit "reverse KL divergence loss" estimated by a gated sampled-token advantage in a length-normalized REINFORCE-style update (its Eqs. 5–9) @xiaomi2026mimov2; the Nemotron MOPD stages ship the same sampled-token estimator, "computed only on the student-sampled token rather than over the full vocabulary" @nvidia2026nemotroncascade2, and NVIDIA's public implementation cites MiMo's Eq. 8 for its advantage (@sec-audit-notes); by mid-2026 the recipe is standard frontier practice: DeepSeek-V4 replaces its mixed-RL consolidation stage with explicit multi-teacher reverse-KL distillation over full-vocabulary logits @deepseek2026v4, while Kimi K3's MOPD and GLM-5's cross-stage distillation ship the same objective implicitly, as a stop-gradient log-ratio advantage on student rollouts @kimi2026k3 @glm2026glm5. Second, *pretraining-scale distillation is uniformly forward-KL or unspecified* (Gemma 2/3, Minitron line, Apple, Llama 3.2). Third, the reverse-KL RLHF regularizer is universal but boilerplate; almost no report specifies the estimator — DeepSeek-V3.2, which specifies *and corrects* its k3 estimator, is the exception @deepseek2025v32.

*Failure reports.* The most informative quote in the sweep is Nemotron 3 Ultra's §3.3.5 — a negative result for *full-distribution* reverse-KL distillation on student prefixes, from the team whose shipped objective is reverse KL at sampled tokens:

#quote(block: true, attribution: [@nvidia2026nemotron3ultra · #emph[Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning]])[A natural alternative to the sampled-token objective is distribution-level distillation, where the student is trained to match the teacher's predictive distribution over the top-$k$ tokens or the full vocabulary at each prefix. In our preliminary experiments, these objectives did not improve MOPD performance and consistently underperformed the sampled-token objective on some agentic benchmarks such as Terminal Bench. We hypothesize that full-distribution matching may impose an overly strong local constraint on prefixes sampled from the student policy, particularly when those prefixes have limited support under the teacher distribution. In this regime, teacher logits can become poorly calibrated or less informative, and matching the full distribution may amplify noise from off-support states.]

Note what failed and what did not: the divergence (reverse KL) is retained; the *estimator* (full-vocabulary or top-$k$ integrand versus sampled token) is what made the difference — the report does not say whose top-$k$; NVIDIA's public top-$k$ path, nemo-rl's `DistillationLossFn`, takes the ids from the *teacher* (@sec-support). The question is not settled at the frontier: DeepSeek-V4 ships exactly the configuration Nemotron found inferior — full-vocabulary reverse KL on student rollouts — as its consolidation stage @deepseek2026v4. The complementary production negative result is Minitron's, in the opposite regime (teacher-forced base-model compression), with a large gap (LM loss 2.155 forward vs 2.665 reverse for the depth-pruned 8B):

#quote(block: true, attribution: [@muralidharan2024minitron · #emph[Compact Language Models via Pruning and Knowledge Distillation]])[…we experiment with Kullback-Leibler divergence (KLD), MSE, cosine similarity and reverse KLD (R-KLD) … We observe from Table 15 and 16 that KLD is the best choice for pruned base model training.]

Apple reports a direction-agnostic negative:

#quote(block: true, attribution: [@apple2024afm · #emph[Apple Intelligence Foundation Language Models]])[We did not find a distillation loss to be helpful here [continued pretraining] for AFM-on-device, unlike in core pre-training.]

== Student-sampled objectives <sec-student-objectives>

Student-sampled stages from the production reports (@sec-reports) and the estimator audit (@sec-table1), grouped by estimator family. Parentheses mark the support the KL is taken over.

#table(
  columns: (1.15fr, 1fr, 1fr),
  stroke: 0.4pt + gray,
  table.header([*rkl_pg*], [*rkl*], [*fkl*]),
  [
    MiniLLM - KD (sampled-token) \
    Tinker - OPD (sampled-token) \
    MiMo-V2-Flash - MOPD \
    Nemotron-Cascade 2 - MOPD \
    Kimi K3 - MOPD \
    GLM-5 - cross-stage distillation \
    InstructGPT / Ziegler - (within RLHF) \
    Tülu 3 / OLMo 2 (within RL) \
    DeepSeekMath - (within GRPO) \
    DeepSeek-V3.2 - (within GRPO) \
    Nemotron 3 Ultra - MOPD (attempted teacher-top-k) \
  ],
  [
    GKD - KD (full-vocab) \
    DistiLLM - SRKL (full-vocab) \
    DistiLLM-2 - SRKL (full-vocab) \
    DeepSeek-V4 - OPD (full-vocab)
  ],
  [
    GKD - KD (full-vocab) \
  ]
)

== Async OPD <sec-async>

Every source below was read from its primary text (per-paper notes in `notes/per-paper/`; the full candidate list with exclusion verdicts is `notes/H-async.md`). The *tier* column uses the scheme of the sweep:
- *A* an argument that distribution-level reverse KL (full-vocab or student-top-$k$) tolerates stale student rollouts better than the sampled-token `rkl_pg` estimator, or the reverse;
- *B* an async-versus-sync comparison with an explanation;
- *C* such a comparison without one;
- *D* quotable statements;
- *E* async OPD in a production report;
- *F* async OPD as the paper's main point;
- *G* async OPD used incidentally.
Staleness is always student-side: the teacher is fixed in every row.

#table(
  columns: (1.05fr, 0.42fr, 1.35fr, 0.95fr, 1.1fr, 1.15fr),
  stroke: 0.4pt + gray,
  table.header([*Source*], [*Tiers*], [*Async mechanism, staleness*], [*Objective*], [*Correction*], [*Outcome*]),
  [AsyncOPD @kang2026asyncopd], [A (reversed), B, D, F], [controlled replay: always consume the oldest of $k$ pre-generated rollout batches, $k in {0, dots, 128}$ train-batch steps (4 optimizer updates each); streaming pipeline with queue depth 4], [`rkl_pg` ($m$ local next-token draws per position, $m in {1,4,16,64}$); student-top-$k$ RKL renormalized; teacher-top-$k$ FKL; full-vocab RKL defined, never run async], [per-token ratio $q_theta slash q_"old"$ at the fixed prefix, advantage recomputed at learner time, unclipped; PPO clip, frozen advantage, Decoupled PPO and M2PO ablated], [mixed: `rkl_pg` with IS is nearly flat to staleness 128; the PPO-style frozen-advantage surrogate loses 15 points; stale student-top-$k$ cannot be corrected; teacher-top-$k$ FKL is flattest but lower at every staleness; 1.6–3.8$times$ throughput],
  [f-OPD @chen2026fopd], [B, D, F], [pipelined rollout / teacher / trainer groups with an active buffer; lag $in {0,2,4,8}$ optimizer steps], [unspecified; follows the Tinker recipe, so `rkl_pg`], [none in vanilla async; f-OPD adds freshness weights, a KL anchor to the rollout student, and adaptive refresh], [fails uncorrected: SWE-bench resolve 41.8 (sync) $arrow$ 26.8 (lag 8), 3/5 collapses; f-OPD 39.4 at 1.46$times$],
  [NPD @rang2026npd], [B, D, F], [rollouts regenerated 0, 1 or 2 times per 10 epochs; buffer replaced at each refresh; lag measured as $D_"KL" (pi_"current" parallel pi_"gen") <= 0.10$], [teacher-top-10 FKL + CE in a packed SFT trainer], [none; $Delta$-IFD sample filter], [works: 11-benchmark avg 62.78 / 64.37 / 64.17 for 0 / 1 / 2 refreshes; 8.1$times$ faster than DistiLLM-2],
  [Lightning OPD @wu2026lightning], [B, D, F; A-adjacent bound], [rollouts precomputed once from the step-0 student and frozen for all 150 steps], [`rkl_pg`, advantage clipped to $plus.minus 10$], [none ($w equiv 1$); drift measured, mean ratio 0.94], [works when the SFT and OPD teachers coincide ($>=$ online in 8/10 cells, 3.6–4$times$ fewer GPU-h); fails with a mismatched teacher (4/4 cells)],
  [TOP-D @xie2026trustregion], [B, D, F], [16 mini-batch updates per rollout batch (0–15 steps stale); reward frozen against $pi_(theta_"old")$], [`rkl_pg` with the bounded reward $log(alpha rho + 1 - alpha)$], [per-token PPO ratio, clip 0.2], [reuse alone gives $approx$ +2 points, bounding alone $approx$ +2 to +5, both together +9 to +14 (Fig. 5; single runs)],
  [DistiLLM @ko2024distillm], [B, C, D, F], [replay buffer of 1000 student sequences (tokens only), draw probability $phi (1 - t slash T)$], [full-vocab skew RKL / skew FKL], [none], [works for SRKL ($<=$ 0.8 ROUGE-L loss at 1.55$times$ less time); the same replay costs ImitKD and GKD 1–2 points (Tab. 4)],
  [DistiLLM-2 @ko2025distillm2], [C, G], [batched: all student responses sampled once per epoch ($approx$ 390 steps stale by epoch end; my derivation)], [full-vocab SRKL on student text, SKL on teacher text], [none], [on-policy $>=$ batched $>=$ replay for every method except DistiLLM (Tab. 12)],
  [Offline on-policy KD @rang2025revealing], [C, E, F], [one generation pass by the SFT student, teacher top-10 logits precomputed; flagship regenerates twice], [teacher-top-10 FKL + CE], [none], [62.28 (no refresh) vs 63.43 (two refreshes); no online baseline; shipped as openPangu-Embedded-1B],
  [Blockwise drift gating @zheng2026blockwise], [C, D], [two PPO epochs per rollout batch], [`rkl_pg` and teacher-top-32 (inherited from @fu2026revisiting)], [detached gate $exp(-|s|)$ on block-mean drift; no ratio], [gating helps under reuse (+1.8 / +3.5 pass\@8); no one-epoch control],
  [Entropy-aware OPD @jin2026entropy], [D, G], [4 mini-batch updates per rollout batch], [`rkl_pg` + entropy-gated teacher-top-16 FKL], [ratio + clip on the sampled-token half only; nothing on the FKL half], [not compared],
  [MiniLLM @gu2024minillm], [D, G], [4 inner epochs per 256-sample collection ($approx$ 16 updates; App. B.1 and code, absent from Alg. 1)], [sequence RKL: full-vocab single-step term + sampled-token long-term term], [per-token ratio $q_theta slash tilde(p)$, clip 0.2 (also covers the teacher-mixed sampler)], [not compared],
  [Nemotron 3 Ultra @nvidia2026nemotron3ultra], [D, E, G], [one-step off-policy pipeline of rollout, teacher-scoring and learner workers (§3.6.1)], [`rkl_pg`, advantage against a proximal policy], [$"sg"[pi_"prox" slash pi_"behav"]$ untruncated $times$ PPO clip on $pi_theta slash pi_"prox"$ $times$ IcePop mask; no values given], [works (no sync arm); distribution-level variants "consistently underperformed" on Terminal Bench (§3.3.5)],
  [Kimi K3 @kimi2026k3], [D, E, G], [partial rollouts paused and resumed across iterations; MOPD "seamlessly integrates" into that loop (§4.1.3)], [`rkl_pg`, reward clipped to $plus.minus R_max$], [a per-token regularizer deferred to Kimi K2.5; no ratio stated], [works (no sync arm); unspecified top-$k$ variants gave "no clear advantage"],
  [MiMo-V2-Flash @xiaomi2026mimov2], [D, E, G], [partial rollouts with a staleness cap and a partial-sample cap (§4.6.2)], [`rkl_pg` + ORM advantage], [IcePop gate for train–inference mismatch; "staleness-aware truncated importance sampling", never formalized], [works (claimed, unquantified)],
  [Nemotron-Cascade 2 @nvidia2026nemotroncascade2], [D (E, negative)], [none: one update per rollout batch, ratio "exactly 1" (§4.1.2)], [`rkl_pg`], [truncated ratio $[0.5, 2]$ for train–inference mismatch only], [synchronous by design, for stability],
  [Tinker blog + cookbook @lu2025onpolicydistillation], [D], [none in the recipe; the sibling RL trainer has an async mode that the distillation script never imports], [`rkl_pg`], [unclipped per-token ratio against sampling-time logprobs], [not compared],
  [verl async recipe @verl2025asyncopd], [D, F], [one-step-off / two-step-off schedulers (1–2 optimizer steps)], [teacher-top-256 FKL (default), RKL, mixture, JSD], [none], [not compared],
  [NeMo-RL MOPD @nemorl2026mopd], [D; A asserted], [async GRPO runtime, max trajectory age 1 step], [`rkl_pg` (default) or full-vocab RKL], [IcePop gate $[0.2, 5]$ required for `rkl_pg`; every IS knob rejected for the full-vocab loss], [not compared (self-distillation smoke tests)],
  [AReaL MOPD @areal2026mopd], [G], [`max_head_offpolicyness` 2 in the shipped example], [`rkl_pg`], [capped behaviour ratio (cap 5) on the distillation loss itself], [not compared],
  [ROLL, slime @roll2026opd @slime2026opd], [G], [async inherited from RL (`async_generation_ratio`; one-rollout lead)], [`rkl_pg`], [none on the OPD term (ROLL evaluates the log-ratio at the behaviour student)], [not compared],
  [Formula-driven survey @zhang2026formula], [D], [none run; names recent-policy replay, cached old-policy rewards, repeated actor updates], [both routes], [names ratio, clip, trust region; no equation], [normative only],
  [Decoupling KL and trajectories @zhao2026decoupling], [A (partial), D], [none run; DAgger aggregation over earlier students formalized in App. E, then disclaimed], [full-vocab FKL and RKL], [none: "No action-level importance sampling is required"], [not compared for staleness],
  [Training Wheels @liu2025trainingwheels], [G (weak)], [replay buffer + 4 updates per batch, one sentence], [PCL soft-Bellman fit of RKL], [none by design], [not compared],
)

Reports and papers checked and found synchronous or silent on staleness: GLM-5 (its distillation stage sits on the synchronous Eq. 1 backbone) @glm2026glm5, DeepSeek-V4 (no off-policy term anywhere) @deepseek2026v4, Qwen3 (the only rollout-reuse statement concerns the GRPO stage, and its "off-policy distillation" is SFT on teacher samples) @qwen2025qwen3, GKD @agarwal2024gkd, Trust Region OPD (its "off-policy" is a teacher prefix) @xing2026tropd, and the strictly one-update-per-batch studies @li2026rethinking @xin2026agreementtrap @zhao2026power @ma2026mopd @zheng2026scope. A quieter form of staleness is routine: several verl-based papers take several mini-batch updates per rollout batch without comment (four in @ko2026reopold, which also reports the clipped fraction staying below 0.2%; four in @zhong2026sod; two in @fu2026revisiting by its batch and mini-batch sizes).

=== Side-by-side comparisons (tiers B and C)

#table(
  columns: (0.8fr, 1.25fr, 0.9fr, 1.5fr, 1.35fr),
  stroke: 0.4pt + gray,
  table.header([*Source*], [*Setup*], [*Staleness axis*], [*Sync $arrow$ async*], [*Explanation offered*]),
  [AsyncOPD @kang2026asyncopd], [Qwen3-4B-Base $arrow.l$ Qwen3-30B-A3B-Instruct-2507, DeepMath, Avg\@32 on AIME24/25 + AMC'23; teacher scores cached on a finite action set], [0–128 train-batch steps, always the oldest batch], [slopes per doubling of staleness, Table 1: `rkl_pg` with recomputed advantage + unclipped IS $-0.73$ (Avg 38.3 $arrow$ 31.4); with 64 local draws $-0.43$ (38.5 $arrow$ 34.6); PPO-style frozen advantage + clip $-1.93$ (38.6 $arrow$ 23.8); frozen advantage without clip collapses to 0 on AIME at staleness 16; stale student-top-$k$ $-1.81$; teacher-top-$k$ FKL $-0.23$ (34.8 $arrow$ 32.9); throughput 1.6–3.8$times$ at a median accuracy delta of $-0.10$#sidenote[The paper prints only the slopes; the endpoint values are read from its published figure data and reproduce every printed slope (note `asyncopd.md`). No seeds are reported.]], [support ownership: FKL is weighted by the fixed teacher and its top-$k$ never misses the cache, RKL is student-weighted and the current student's support drifts off the cached scores; a sampled action is correctable by the ratio, a stale top-$k$ set is not; recomputing the advantage flattens the ratio tail so clipping only removes signal],
  [f-OPD @chen2026fopd], [Qwen3-8B; MATH500, tool-use, SWE-bench Verified; Tinker-style loss, no correction], [lag 0 / 2 / 4 / 8 optimizer steps], [relative to lag 0 at lag 8: $approx$ 89% reasoning, $approx$ 83% tool-use, $approx$ 60% coding ("roughly 40%" drop); SWE-bench resolve 41.8 $arrow$ 26.8, collapse 3/5; async throughput 1.61$times$], [rollout drift (prefix occupancy of an older student) plus supervision drift (teacher contexts cached at grading time), bounded by TV terms (Prop. 3.2) that compound with horizon; entropy destabilizes with lag],
  [NPD @rang2026npd], [openPangu-7B $arrow.l$ 1B; teacher-top-10 FKL + CE; no correction], [0 / 1 / 2 refreshes per 10 epochs], [62.78 / 64.37 / 64.17 (11-benchmark avg); at equal steps NPD tracks the on-policy curve on MATH-500 (Fig. 3a)], [near-policy regime $D_"KL" (pi_"current" parallel pi_"gen") <= 0.10$; the $Delta$-IFD filter removes the high-variance tail; a sparse refresh "resets the policy lag"],
  [Lightning OPD @wu2026lightning], [Qwen3-4B / 8B after SFT; `rkl_pg`, no ratio], [0 (online) vs whole run on step-0 rollouts], [AIME24 4B 65.4 $arrow$ 68.1, 8B 68.5 $arrow$ 69.9; 72 $arrow$ 20 and 120 $arrow$ 30 GPU-h; with a mismatched SFT teacher offline is worse in 4/4 cells], [$norm(nabla J_"on" - nabla J_"off") <= G sigma_A sqrt(chi^2 (pi_theta parallel pi_"ref"))$ (Thm. 3.5); the offline gradient equals the online one minus a covariance term read as an implicit trust region (Thm. 3.7); bias enters only through teacher inconsistency],
  [TOP-D @xie2026trustregion], [Qwen3-1.7B-Base $arrow.l$ Qwen3-30B-A3B; `rkl_pg`, PPO ratio clip 0.2], [1 vs 16 mini-batch updates per rollout batch], [AIME24 at $approx$ 200 steps: reuse alone $approx$ +2, bounded reward alone $approx$ +2 to +5, both $approx$ +9 to +14 (Fig. 5, read from the plot; single runs, budgets not matched)], [reuse is safe only once the per-token reward is bounded: the raw $log(pi^* slash pi_theta)$ diverges when the teacher assigns near-zero mass, and the inner updates form a decreasing KL sequence against a fixed proximal teacher (Thm. 4.9)],
  [DistiLLM @ko2024distillm], [GPT-2 / OpenLLaMA students; full-vocab SRKL; no correction], [fresh vs replay buffer of 1000 sequences], [SRKL: −0.26 / 0.00 / +0.14 / −0.73 / −0.76 ROUGE-L on five sets at 2.58$times arrow$1.67$times$ relative time; ImitKD (FKL) −1.01 / −0.76 / −1.90; GKD (JSD) −0.86 / +0.05 / −1.08 (Tab. 2, 4)], ["Off-policy approach's success stems from the fast convergence speed of S(R)KL": a fast objective front-loads its learning, so late replay carries little residual gradient; buffer capacity 1000 balances overfitting against "outdated SGO … high bias" (App. E.8)],
  [DistiLLM-2 @ko2025distillm2], [GPT-2 XL $arrow.l$ GPT-2, Dolly, ROUGE-L], [on-policy / batched per epoch / replay buffer], [MiniLLM 23.84 / 23.75 / 23.41; GKD 23.75 / 23.21 / 22.89; DistiLLM 26.12 / 26.11 / 26.12; DistiLLM-2 26.37 / 26.20 / 26.13 (Tab. 12, single runs)], [none of its own; cites DistiLLM],
  [Offline on-policy KD @rang2025revealing], [openPangu 1B; teacher-top-10 FKL + CE], [0 vs 2 regenerations], [62.28 $arrow$ 63.43 avg; no online arm], [none ("simulates an on-policy data distribution")],
  [Blockwise gating @zheng2026blockwise], [Qwen3-1.7B-Base $arrow.l$ Qwen3-4B-Base-GRPO], [two PPO epochs, gated vs ungated], [pass\@8 avg: `rkl_pg` 49.8 $arrow$ 51.6, teacher-top-32 49.8 $arrow$ 53.3; no one-epoch arm], [heuristic: drift "can indicate how stale a piece of rollout data is"],
)

Two readings of this table matter for the estimator question. First, the only controlled staleness sweep (AsyncOPD) finds the sampled-token estimator *more* robust than a distribution-level one, but under a constraint that does not apply to a trainer with the teacher available at update time: its teacher scores are cached at rollout time on a finite action set, so the current student's top-$k$ has no teacher scores and the stale set cannot be repaired, while a sampled action can be reweighted by the ratio. Its own headline figure (forward KL flat, reverse KL falling) is produced with the frozen-advantage PPO surrogate that it later rejects; with the recommended surrogate the reverse-KL curve sits above forward KL at every staleness level. Second, in every setting where the teacher is re-evaluated at update time on stored tokens (DistiLLM, DistiLLM-2, NPD, offline on-policy KD), distribution-level losses are trained on stale prefixes with no correction at all, and the recorded cost is within a point; the same replay costs the score-function baseline in DistiLLM-2's Tab. 12 the most of any method, though by a margin that single runs cannot resolve. Uncorrected `rkl_pg` on stale rollouts fails outright in f-OPD (lag 8, long-horizon agents) and in AsyncOPD's frozen-advantage-no-clip arm (staleness 16), and succeeds in Lightning OPD (whole run on step-0 rollouts) where the drift stays small (mean ratio 0.94) and the advantage is bounded.

=== Folk knowledge (tier D)

#table(
  columns: (1.6fr, 1.5fr, 0.5fr),
  stroke: 0.4pt + gray,
  table.header([*Claim*], [*Who says it*], [*Best tier*]),
  [Asynchrony is required for throughput and staleness is its price], [f-OPD §1; AsyncOPD §1; verl recipe ("trading some strict on-policy guarantees"); NPD App. E ("one-rollout-per-update … leaving [training NPUs] in prolonged idle states")], [B],
  [Teacher-weighted forward KL is robust to stale rollouts, student-weighted reverse KL is vulnerable], [AsyncOPD abstract and §4.2; repeated second-hand by @zheng2026blockwise], [B, but established with the surrogate the paper rejects],
  [A sampled-token surrogate on stale rollouts needs a ratio, a clip, or a trust region, or it is not the current student's gradient], [formula survey §4.1, §4.3; AsyncOPD Eq. 5; Nemotron 3 Ultra §3.3.1; entropy-aware OPD §2.3; TOP-D §3.3; MiniLLM §2.2], [B],
  [A distribution-level loss at a fixed prefix needs no action-level correction, whoever sampled the prefix], [Decoupling KL App. F; NeMo-RL MOPD documentation; implicit in DistiLLM (tokens-only buffer) and verl (no ratio on the async path)], [A (asserted), C],
  [Staying strictly on-policy is a deliberate stability choice], [Nemotron-Cascade 2 §4.1.2; Tinker blog; MiMo-V2-Flash §2.3.1 ("on-policy training is generally more stable and effective"); GKD], [D],
  [Stale rollouts are tolerable with a per-token trust region or gate], [Kimi K3 §4.1.2; MiMo-V2-Flash §4.6.2; Nemotron 3 Ultra §3.3.1; AReaL MOPD docs], [E],
  [Replay works because the student drifts little or the objective converges fast], [DistiLLM §3.2; Lightning OPD App. C.1; NPD App. C.1; REOPOLD (clipped fraction $<$ 0.2%)], [B],
  [Several updates per rollout batch is routine and unexamined], [MiniLLM App. B.1; entropy-aware OPD §5.1; blockwise gating §2; TOP-D §5.1], [G],
)

#quote(block: true, attribution: [@kang2026asyncopd · #emph[AsyncOPD: How Stale Can On-Policy Distillation Be?]])[Once the learner receives a teacher-scored cache, it can recompute current-student log probabilities on cached actions, but it cannot recover teacher scores for actions that were never scored. … Reweighting within $S_"old"$ cannot recover these missing teacher scores, so stale sparse top-$k$ remains a support-mismatched approximation, not an exact correction of the current top-$k$ objective.]

#quote(block: true, attribution: [@kang2026asyncopd · #emph[AsyncOPD]])[We study sparse and Monte Carlo OPD estimators, not dense full-vocabulary KL in the asynchronous setting. Although dense KL avoids cached-support mismatch, it is difficult to implement efficiently when rollout, teacher scoring, and learner updates are decoupled.]

#quote(block: true, attribution: [@chen2026fopd · #emph[f-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control]])[Vanilla asynchronous OPD delivers a 1.61× throughput gain over synchronous OPD, but task quality collapses to a 26.8 resolve with a 60% collapse rate across runs. … all three tasks exhibit a clear and monotonic performance degradation as policy lag increases, confirming that policy staleness is inherently harmful to asynchronous OPD fidelity.]

#quote(block: true, attribution: [@wu2026lightning · #emph[Lightning OPD]])[The central finding is that Lightning OPD, despite eliminating the live teacher server entirely during training, achieves performance on par with standard OPD across all settings, and in several cases marginally exceeds it. … Notably, teacher inconsistency imposes a larger penalty on Lightning OPD than on standard OPD.]

#quote(block: true, attribution: [@ko2024distillm · #emph[DistiLLM]])[Off-policy approach's success stems from the fast convergence speed of S(R)KL while other loss functions cannot be achieved. … unlike other baselines (Lin et al., 2020; Agarwal et al., 2024) that suffer performance drops when switching from on-policy to off-policy, our method maintains its efficacy.]

#quote(block: true, attribution: [@zhang2026formula · #emph[A Formula-Driven Survey and Research Agenda for On-Policy Distillation]])[Without small policy drift, importance correction, PPO-style ratios, or a trust-region approximation, this old-policy surrogate should not be described as the exact current-student sequence reverse-KL gradient.]

#quote(block: true, attribution: [@zhao2026decoupling · #emph[Decoupling KL and Trajectories]])[No action-level importance sampling is required because the action expectation is already taken under $q_theta (dot | s_t)$ rather than under the teacher.]

#quote(block: true, attribution: [@nemorl2026mopd · #emph[NeMo-RL, `docs/about/algorithms/mopd.md`] (on the full-vocabulary variant)])[This is the K=V limit of the top-k estimator: with the support spanning the whole vocabulary the score-function tail term vanishes, so the objective is exact, deterministic, and free of that estimator's off-policy bias.]

#quote(block: true, attribution: [@nvidia2026nemotron3ultra · #emph[Nemotron 3 Ultra]])[In our implementation, MOPD is executed asynchronously. Rollout workers, teacher-scoring workers, and learner workers run in a pipeline. A trajectory may therefore be generated by a stale behavior policy $pi_"behav"$, while the learner optimizes a newer student snapshot. To stabilize this setting, we decouple the behavior policy from the proximal policy $pi_"prox"$ used as the trust-region center.]

#quote(block: true, attribution: [@kimi2026k3 · #emph[Kimi K3]])[Under our partial rollout scheme, an individual long-horizon trajectory naturally spans multiple iterations, introducing data staleness that threatens training stability. Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization. … This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks.]

#quote(block: true, attribution: [@nvidia2026nemotroncascade2 · #emph[Nemotron-Cascade 2]])[At each iteration, we generate a group of $G$ rollouts from the current policy $pi_theta$ and then perform a single gradient update. This ensures that the policy used for data collection always matches the one being updated, making the importance sampling ratio exactly 1. This on-policy setup contributes to stable RL training and mitigates entropy collapse.]

#quote(block: true, attribution: [@lu2025onpolicydistillation · #emph[On-Policy Distillation] (Thinking Machines blog)])[This process turns training on one's own samples into off-policy training over time, which causes the same error accumulation and divergence over long sequences that happens with off-policy training. … On-policy distillation always stays on-policy, and since the teacher stays fixed, the student converges on the teacher's desirable behavior.]

#quote(block: true, attribution: [@gu2024minillm · #emph[MiniLLM], App. B.1])[Similar to [OWJ+22], we collect 256 sentences at once and adopt 4 inner epochs when doing the policy optimization. The clipping rate $epsilon$ is set to 0.2.]

Three terminology traps run through these quotes. "Off-policy distillation" means SFT on teacher samples in Qwen3, GKD, the Rethinking-OPD papers and Trust Region OPD, and stale student rollouts in DistiLLM, TOP-D and the async papers; Qwen3 uses both senses in one report (§4.2 vs §4.5). "Partial rollout" (Kimi K3, MiMo-V2-Flash) is staleness within a single trajectory, whose prefix was generated by an earlier student. And Nemotron 3 Ultra's "192K" is its maximum generation length, not a staleness figure; its stated staleness is one optimizer step for RL and MOPD alike (§3.6.1).

=== Tier A: what can be argued <sec-async-tierA>

No source proves that one estimator tolerates staleness better than another, and the one controlled sweep finds the sampled-token estimator the more robust under its cache constraint. Two sources assert the opposite direction without testing it (Decoupling KL App. F; NeMo-RL's MOPD documentation), Lightning OPD bounds the uncorrected `rkl_pg` error without comparing estimators, and the surveys leave the comparison open. The following is ours.

Fix a prefix $s$ sampled by the behaviour student $q_"old"$ and let the teacher $p$ be fixed. Following @sec-deriv, every implementation in the table above drops the visitation term and optimizes the frozen-visitation surrogate
$ hat(J)(theta) = bb(E)_(s tilde.op d_"old") [D(theta; s)], quad D(theta; s) = sum_a q_theta (a|s) log (q_theta (a|s)) / (p(a|s)), $ <eq-async-surrogate>
where $d_"old"$ is the prefix distribution of whichever student generated the batch. Staleness enters @eq-async-surrogate in two places. The *prefix* term, $d_"old"$ in place of $d_theta$, is common to every estimator: it is the term f-OPD bounds by $C_"roll" "TV"(d^t, d^("stale", t))$ and Lightning OPD by $G sigma_A sqrt(chi^2)$, it is exactly zero in synchronous training only because $d_"old" = d_theta$, and no action-level device touches it. The *action* term differs by estimator:

- *Full-vocabulary RKL.* $nabla_theta D(theta; s)$ is computed exactly at the stored prefix from the current student and the teacher's logits at $s$. It does not depend on which action was sampled at $s$, so there is no behaviour-policy ratio, no clipping and no variance from the action draw. This is the content of Decoupling KL's App. F, and it is why DistiLLM can store tokens only and re-run both models on the stale sequence.
- *Student-top-$k$ RKL.* The same holds if the support $S_theta = "TopK"(q_theta (dot|s))$ is formed at update time and the teacher is evaluated on it. If instead the teacher was scored at rollout time on $S_"old" = "TopK"(q_"old")$, the current support has missing scores; AsyncOPD shows this is uncorrectable by reweighting and that it collapses at moderate staleness. The distinction is therefore not the estimator but *when the teacher is queried*: our trainer evaluates the teacher on the current student's top-$k$ at every update, which is the first case.
- *Sampled-token `rkl_pg`.* $nabla_theta D(theta; s) = bb(E)_(a tilde.op q_theta (dot|s)) [A_theta (a, s) nabla log q_theta (a|s)]$ with $A_theta = log q_theta - log p$ under stop-gradient, plus the score-function tail that vanishes only when the expectation is under $q_theta$ (the "K=V limit" of the NeMo-RL note). With $a tilde.op q_"old"$ the unbiased form needs $rho = q_theta (a|s) slash q_"old" (a|s)$ and an advantage recomputed at update time (AsyncOPD Eq. 5 and 7); its variance carries the ratio and grows with staleness, and the two usual variance devices, a frozen advantage and a clipped ratio, both bias it (AsyncOPD Fig. 3: the frozen-advantage form without clipping collapses at staleness 16, and with clipping loses 15 points at 128).

So, conditional on the teacher being evaluated at update time, the distribution-level estimators are exact in the action term at any staleness and inherit only the prefix-term bias, whereas `rkl_pg` adds a ratio-variance term and, once clipped or frozen, a bias term. That is the argument for the hypothesis; its empirical status is that no published run isolates it. The nearest evidence is indirect: DistiLLM-2's Tab. 12, where the sampled-token MiniLLM row loses the most from replay of any method, and the clean survival of uncorrected full-vocab losses on stale prefixes in four independent papers, against uncorrected `rkl_pg` failing in f-OPD at lag 8. The variable that the literature does not control is the prefix term, which those papers hold small by design (bounded replay in DistiLLM, near-policy KL $<=$ 0.10 in NPD, mean ratio 0.94 in Lightning OPD) and which f-OPD lets grow with horizon. An experiment that fixes the prefix distribution and varies only the estimator is the missing row of the comparison table.

= Estimators and implementations <sec-estimators>

This section is the core of the review: for each implementation associated with a reverse-KL training stage, what expression is computed, what is sampled, which gradient is taken, and whether the result is consistent with the claimed objective. Code references are commit-pinned; classifications use the categories *correct for the claimed objective* / *correct for a stated surrogate* / *biased or inconsistent with the claimed objective* / *indeterminate*, each with a confidence level.

== Taxonomy: which "reverse KL" is being estimated <sec-taxonomy>

Write $s_t = (x, y_(<t))$ for a prefix and $q_theta (dot|s_t)$, $p(dot|s_t)$ for next-token distributions. The chain rule gives the exact identity
$ D_"KL" (q_theta (y|x) parallel p(y|x)) = sum_t bb(E)_(s_t tilde.op q_theta) [ D_"KL" (q_theta (dot|s_t) parallel p(dot|s_t)) ], $ <eq-chain>
so *sequence-level* reverse KL equals the sum of *token-level conditional* KLs weighted by the student's own prefix-visitation distribution. This equivalence is exact for the objective *value* — but only when prefixes are drawn from $q_theta$. When prefixes come from a fixed dataset (GKD's off-policy fraction, DistiLLM, torchtune, all pretraining KD) or from the teacher, the summed token-level KL is a *different* objective: an expected conditional divergence under a mismatched visitation distribution, which neither equals nor generally bounds the sequence-level KL. Independently, each token-level KL may be computed *exactly over the full vocabulary* or estimated *at sampled tokens*; and the whole construct may serve as *primary objective* (distillation), *regularizer added to the loss* (GRPO-style), or *penalty added to the reward* (PPO-style). A reward-channel penalty and a loss-channel term describe the same objective value, but their gradients differ by construction — the subject of the next subsection. Finally, note that minimizing sequence-level reverse KL is *identical* to entropy-regularized reward maximization with reward $log p$ (up to constants, $min_theta "KL"(q parallel p) = max_theta bb(E)_q [log p] + cal(H)(q)$) — which is why RL machinery attaches so naturally to reverse-KL distillation (MiniLLM, Tinker, MOPD).

== Derivations: values versus gradients <sec-deriv>

All derivations are for on-policy sampling, $y tilde.op q_theta$, with $p$ fixed; write $r_t = p(y_t|s_t) slash q_theta (y_t|s_t)$ for the teacher/student probability ratio at the sampled token, so the per-token reverse KL is $D_"KL" (q_theta (dot|s_t) parallel p(dot|s_t)) = bb(E)_(y_t tilde.op q_theta) [-log r_t]$. Schulman proposes three single-sample estimators of this *value*, each computable from the sampled token alone @schulman2020klapprox:
$ k 1_t = -log r_t = log q_theta (y_t|s_t) - log p(y_t|s_t), $
$ k 2_t = (log r_t)^2 slash 2 = (k 1_t)^2 slash 2, $
$ k 3_t = r_t - 1 - log r_t = (r_t - 1) + k 1_t. $
$k 1$ is the naive Monte Carlo estimate — the log-ratio itself: unbiased by construction, but it can be negative even though the KL never is, and its variance is large when $q_theta$ and $p$ are far apart. $k 2$ replaces the log-ratio by half its square: always nonnegative and low-variance, but biased for the reverse KL — the squared log-ratio matches the KL only to second order in $log r$, so the bias vanishes as $q_theta -> p$. $k 3$ adds to $k 1$ the control variate $r_t - 1$, which has expectation zero under $q_theta$ (since $bb(E)_q [r] = 1$): it remains unbiased, and is pointwise nonnegative because $log r <= r - 1$, which is what makes it low-variance. The blog is explicit that these are diagnostics for the value. Whether their *derivatives* estimate the reverse-KL gradient is a separate question @tang2025pitfalls @liu2025rethinkingkl @zhang2025rpg.

The master identity is, for any per-sample integrand $f_theta$,
$ nabla_theta bb(E)_(y tilde.op q_theta)[f_theta (y)] = bb(E)[nabla_theta f_theta (y)] + bb(E)[f_theta (y) nabla_theta log q_theta (y)], $ <eq-master>
a pathwise term plus a score-function term. With $f_theta = log q_theta - log p$ the pathwise term vanishes in expectation ($bb(E)[nabla log q] = 0$), leaving the score-function-only form $nabla "KL" = bb(E)[(log q - log p) nabla log q]$; per token, using $bb(E)[nabla log q]=0$ again,
$ nabla_theta D_"KL" (q_theta parallel p) = bb(E)_(y_t tilde.op q_theta (dot|s_t)) [ k 1_t dot nabla_theta log q_theta (y_t | s_t) ]. $ <eq-truegrad>

*Naive backprop is zero-mean noise.* The estimator "sample $y tilde.op q$, compute $log q_theta (y) - log p(y)$, backpropagate" gives an unbiased estimate of the *value* of the KL, but its gradient is $nabla log q_theta (y)$, whose expectation is $0$: the estimator's expected gradient contains none of the score-function weighting in @eq-truegrad. (a) correct for the value; (b) pure noise for the gradient. This is Tang and Munos's first pitfall:

#quote(block: true, attribution: [@tang2025pitfalls · #emph[On a Few Pitfalls in KL Divergence Gradient Estimation for RL]])[…such implementations are generally incorrect and do not produce the desired KL gradient.]



*The three k-estimators as losses.* Differentiating each estimator through $log q_theta$ at the sampled token:
$ nabla k 1 = nabla log q, quad bb(E)[nabla k 1] = 0; $
$ nabla k 2 = (log q - log p) nabla log q = k 1 nabla log q, quad bb(E)[nabla k 2] = nabla D_"KL" (q parallel p); $ <eq-k2>
$ nabla k 3 = (1 - r) nabla log q, quad bb(E)[nabla k 3] = -sum_v p(v) nabla log q(v) = nabla D_"KL" (p parallel q). $ <eq-k3>
The $k 3$ line uses $bb(E)_q [r nabla log q] = sum_v p(v) nabla log q(v) = -nabla H(p, q)$. So *$k 2$-as-loss is the on-policy-correct differentiable surrogate for reverse KL, while $k 3$-as-loss — the GRPO pattern — has an expected gradient pointing along the forward KL*:

#quote(block: true, attribution: [@tang2025pitfalls · #emph[On a Few Pitfalls in KL Divergence Gradient Estimation for RL]])[…the positions of the two distributions are reversed compared to the target KL divergence … the regularization effect of the variance reduced estimate is completely incidental.]

It is still a penalty pulling $q$ toward $p$ (which is why it "works" as a regularizer), but a mass-covering one, not the mode-seeking one on the label.

*Repairing k3 with a differentiable ratio.* Multiplying $k 3$ by the importance ratio $rho = q_theta slash "sg"[q_theta]$ (value $1$, gradient $nabla log q$) — the correction introduced by DeepSeek-V3.2 (their Eq. 7, an importance-sampling fix with $pi_"old"$ in the denominator) and implemented by TRL's `use_bias_correction_kl` and NeMo-RL's surrogate — restores the reverse direction: $nabla(rho k 3) = k 3 nabla log q + nabla k 3$, and since $bb(E)[k 3 nabla log q] = bb(E)[(r - 1) nabla log q] + bb(E)[k 1 nabla log q] = -nabla D_"KL" (p parallel q) + nabla D_"KL" (q parallel p)$, the totals cancel to
$ bb(E)[nabla (rho k 3)] = nabla D_"KL" (q parallel p). $ <eq-rhok3>

DeepSeek-V3.2 motivates the fix by the uncorrected estimator's behavior when $pi_theta lt.double pi_"ref"$:

#quote(block: true, attribution: [@deepseek2025v32 · #emph[DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models]])[In such cases, the gradient of the K3 estimator assigns disproportionately large, unbounded weights to maximize the likelihood of these tokens, resulting in noisy gradient updates that accumulate to degrade sample quality in subsequent iterations and lead to unstable training dynamics.]

*KL in the reward channel.* Subtracting $beta k 1_t$ from the per-token reward and applying an idealized (critic-free, reward-to-go, unclipped) policy gradient contributes $-beta bb(E)[sum_t nabla log q(y_t|s_t) sum_(t' >= t) k 1_(t')]$, which is exactly the score-function form of $-beta nabla D_"KL"$ for the *sequence-level* KL, including the effect of $theta$ on prefix visitation. #cite(<liu2025rethinkingkl>, form: "prose") state that on-policy, "the 'k2 as loss' formulation is, in fact, gradient-equivalent to 'k1 in reward'"; by the derivations here the equivalence is exact for each token's *own* penalty term ($k 1_t nabla log q_t$, cf. @eq-k2), while the reward-to-go channel *additionally* propagates each $k 1_(t')$ into the score functions of earlier tokens $t < t'$ — precisely the visitation cross-terms that separate the sequence-level gradient from the frozen-visitation one, and which loss-channel $k 2$ does not carry. Off-policy, every variant needs importance-weight corrections, which RPG derives systematically, including "an off-policy importance-weighting mismatch in GRPO's KL term" @zhang2025rpg. In deployed PPO the idealization is broken by GAE, critics, whitening, and clipping — the objective family is right, the exact gradient is not.

*Token-level versus sequence-level: the visitation term.* For fixed $x$, let $d_(theta,t)(s_t) = q_theta (y_(<t)|x)$ be the distribution of the prefix $s_t$, and abbreviate the conditional divergence by
$ D_t (s_t; theta) = D_"KL" (q_theta (dot|s_t) parallel p(dot|s_t)). $
Then @eq-chain is the objective
$ cal(L)_"seq"(theta) = sum_t bb(E)_(s_t tilde.op d_(theta,t)) [D_t (s_t; theta)]. $
Applying @eq-master to the outer expectation gives the exact split
$ nabla_theta cal(L)_"seq" = underbrace(sum_t bb(E)_(s_t tilde.op d_(theta,t))[nabla_theta D_t (s_t; theta)])_(G_"local") + underbrace(sum_t bb(E)_(s_t tilde.op d_(theta,t))[D_t (s_t; theta) nabla_theta log d_(theta,t)(s_t)])_(G_"visit"). $
The first term is the gradient obtained with prefix visitation frozen. More formally, if $bar(theta)$ denotes the fixed parameters used to sample prefixes, then
$ cal(L)_"frz" (theta; bar(theta)) = sum_t bb(E)_(s_t tilde.op d_(bar(theta),t))[D_t (s_t; theta)], quad G_"local" = nabla_theta cal(L)_"frz" (theta; bar(theta))|_(bar(theta)=theta). $
The prefix score expands over earlier tokens:
$ nabla_theta log d_(theta,t)(s_t) = sum_(u<t) nabla_theta log q_theta (y_u|s_u). $
Since $bb(E)[k 1_t | s_t] = D_t (s_t;theta)$ and @eq-truegrad holds conditional on $s_t$, both terms have sampled-token forms:
$ G_"local" = bb(E)_(y tilde.op q_theta) [sum_t k 1_t nabla_theta log q_theta (y_t|s_t)], $
$ G_"visit" = bb(E)_(y tilde.op q_theta) [sum_t k 1_t sum_(u<t) nabla_theta log q_theta (y_u|s_u)]. $
Combining them and exchanging the two sums yields the full sequence gradient:
$ nabla_theta cal(L)_"seq" = bb(E)_(y tilde.op q_theta) [sum_u nabla_theta log q_theta (y_u|s_u) sum_(t >= u) k 1_t]. $
Thus implementations that sample prefixes from the student but treat them as data (GKD's full-vocabulary loss; Tinker's and MOPD's per-token advantages without reward-to-go) compute $G_"local"$ only — "a partial gradient at best" in Tang and Munos's phrase, though for GKD this is the *stated* objective, not an error. Indeed, under the usual policy-gradient loss convention, the local advantage
$ A_t^"KL" = -"sg"[k 1_t], quad nabla_theta[-A_t^"KL" log q_theta (y_t|s_t)] = k 1_t nabla_theta log q_theta (y_t|s_t) $
is unbiased for $G_"local"$. Replacing it by the undiscounted return-to-go,
$ A_u^"KL" = -sum_(t >= u) "sg"[k 1_t], $
restores $G_"local" + G_"visit" = nabla_theta cal(L)_"seq"$. The shipped defaults (Tinker `kl_discount_factor = 0.0`) deliberately omit this visitation term. The omission is Tang and Munos's second pitfall, and their §4.1 spells out the sense in which the result is "partial":

#quote(block: true, attribution: [@tang2025pitfalls · #emph[On a Few Pitfalls in KL Divergence Gradient Estimation for RL]])[Secondly, we show that some implementations do not account for the sequential nature of the estimation problem and produce a partial gradient at best. … Importantly, the above gradient [the summed per-token gradient estimates] does not account for the impact that token $y_t$ has on the future tokens $y_s , s > t$ which impacts future token-level KL divergences. Henceforth, we call the above loss implementation the token-level losses, which constitute estimates to partial gradient of sequence-level KL divergence…]

Their Appendix D derives the same split as above, its "part i" and "part ii" being exactly $G_"local"$ and $G_"visit"$.

*The dual partial gradient: exact KL in the reward channel.* The complementary omission arises when the *exact* per-position KL $D_t (s_t; theta)$ — not its sampled-token estimate $k 1_t$ — is placed in the reward under stop-gradient. Because $D_t$ is a deterministic function of the prefix, it carries no dependence on the token sampled at $t$: the own-position score term vanishes, $bb(E)[D_t (s_t) nabla_theta log q_theta (y_t|s_t)] = bb(E)[D_t (s_t) dot bb(E)[nabla_theta log q_theta (y_t|s_t) | s_t]] = 0$, and the stop-gradient blocks the pathwise route through $q_theta (dot|s_t)$. What survives of the idealized reward-to-go policy gradient is $-beta G_"visit"$ alone: derandomizing the penalty destroys exactly the sampled-token dependence that lets the score function recover $G_"local"$ from $k 1$-in-reward — the same mechanism by which naive backprop of $k 1$ loses everything. Token-level losses keep $G_"local"$ and drop $G_"visit"$; exact-KL-in-reward keeps $G_"visit"$ and drops $G_"local"$. Tang and Munos do not treat this case; it matters here because trl-classic ships it as the opt-in `"full"` penalty (@sec-table1, @sec-audit-notes).

For GKD, by contrast, computing only $G_"local"$ is the announced design — the on-policy loss (their Eq. 4) is defined with the sampling distribution outside the gradient:

#quote(block: true, attribution: [@agarwal2024gkd · #emph[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]])[…where we do not backpropagate through the student's sampling distribution $p_S (dot|x)$, similar to on-policy imitation. Not backpropagating through the sampling makes the training stable and computationally efficient.]

and the paper defends the choice as an advantage over MiniLLM's policy-gradient treatment, which does carry the score-function machinery:

#quote(block: true, attribution: [@agarwal2024gkd · #emph[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes]])[…we argue that GKD is simpler and more stable, being closer to supervised training, since it does not backpropagate through the student's sampling process. Indeed, MiniLLM relies on a number of stabilizing tricks, to tackle high variance, reward hacking, and generation length bias.]

FRPO subsequently made exactly this correction its contribution:

#quote(block: true, attribution: [@yao2026futurekl · #emph[Future-KL Regularized GRPO: Process-Level Credit Assignment from f-Divergence Regularization]])[The reward term recovers the standardized GRPO advantage, while the regularizer term includes a causal future-regularization return-to-go omitted by local KL losses. For reverse KL, this yields a simple future KL correction: add a reverse cumulative sum of per-token log ratios after advantage construction.]

#[
#set page(width: 36cm, height: 28.5cm)

== Estimator audit table <sec-table1>
#set text(size: 7.3pt)
#set list(body-indent: 0.3em)
#show table.cell.where(x: 3): set list(spacing: 1em)

*Table 1 — Estimator audit.* $q$ student/policy (trained), $p$ teacher/reference; $k 1, k 2, k 3$ as in @sec-deriv; sg $=$ stop-gradient; seq $=$ sequence-level, tok $=$ token-level conditioned on prefixes; full-vocab $=$ summed over the full vocabulary, sampled-token $=$ estimated at the sampled token only. Classifications: *C* correct for the claimed objective; *S* correct for a stated surrogate; *B* biased/inconsistent with the claimed objective; *I* indeterminate. Justifications for non-trivial rows follow in @sec-audit-notes.

#table(
  columns: (0.8fr, 0.42fr, 1.05fr, 2.0fr, 0.68fr, 0.85fr, 1.3fr, 0.6fr),
  stroke: 0.4pt + gray,
  table.header([*Occurrence*], [*Stage*], [*Claimed objective & stated surrogate*], [*Expression computed*], [*Sampled from*], [*Support mods*], [*Resulting objective / $bb(E)[nabla]$*], [*Class.*]),
  [
    MiniLLM \
    @gu2024minillm
  ], [KD fine-tune],
  [
    - seq-level RKL
    - surrogate stated: PPO-style PG with three named stabilizing techniques#footnote[PPO-style: clipped importance-sampling policy gradient, rollouts from $q_"old"$ reused over several epochs. The three techniques — single-step decomposition, teacher-mixed sampling, length normalization — are detailed in @sec-audit-notes.]
  ],
  [
    $cal(L) = cal(L)_"PG" + cal(L)_"reg" + cal(L)_"PT"$

    $cal(L)_"PG" = -1/N sum_t min(rho_t A_t, "clip"(rho_t, 1-epsilon, 1+epsilon) A_t)$

    - *importance ratio* $rho_t = (q_theta (y_t|s_t)) / (q_"old" (y_t|s_t)) dot "sg"[(q_"old" (y_t|s_t)) / (p_"sample" (y_t|s_t))]$#footnote[$q_theta$: the live student (current parameters; the only gradient path); $q_"old"$: the frozen student snapshot that generated the rollouts — the PPO "old" policy, with each batch reused for four inner epochs, so $q_theta$ drifts from it; $p_"sample"$: the teacher-mixed behavior policy the rollouts are actually drawn from (generation only, never carries gradient).]
    - *advantage* $A_t = "sg"["whiten"((sum_(t'>=t) gamma^(t'-t) r_(t')) slash L_t)]$ \
      #sidenote[whitened return-to-go; no critic]
    - *per-token reward* $r_t = log p(y_t|s_t) - log q_"old" (y_t|s_t)$
    
    $cal(L)_"reg" = 1/N sum_t D_"KL" (q_theta (dot|s_t) parallel p(dot|s_t))$ \
    #sidenote[full-vocab, same rollout prefixes]
    
    $cal(L)_"PT" = bb(E)_"OpenWebText" [1/2 (-log q_theta (y_t|s_t)) + 1/2 (-sum_v p(v|s_t) log q_theta (v|s_t))]$ \
    #sidenote[2nd term $=$ FKL $+$ const.]
    - $nabla$ through live $q_theta$ in $rho_t$, $cal(L)_"reg"$, $cal(L)_"PT"$ only

    Where:
    - $N$ $=$ \# response tokens in batch
    - $L_t$ $=$ remaining response length
  ],
  [
    $y tilde.op p_"sample" (dot|s_t)
    \
    = (1-alpha) q_"old" (dot|s_t)
    \
    + alpha p(dot|s_t)$, $alpha = 0.2$
  ],
  [
    - reward scaling 0.5
    - reward clip $plus.minus 100$
    - PPO clip 0.2
    - $gamma = 0.95$
    - whitening
    - length norm
    - teacher-mix sampling
  ],
  [
    - seq-PG $+$ tok-level full-vocab: variance-reduced, reward-shaped PG surrogate of $nabla "KL"_"seq" (q parallel p)$ — visitation term carried by the reward-to-go ($gamma = 0.95$ discounts it)
    - local term double-counted: $A_t$ keeps the own-token reward *and* the exact single-step term is added (the paper's Alg. 1 assigns $R_(t+1)$ to the PG part; @sec-audit-notes)
    - deviations deliberate but bias the gradient
    - per-token-only IS correction for mixture prefixes is an extra approximation
  ],
  [
    - *B* (mild; med)
    - S for the stated PPO composite; B because the shipped estimator deviates unstated from the paper's own Alg. 1 ($R_(t+1)$ stated; $R_t$ $+$ exact term shipped)
  ],
  [
    GKD \
    @agarwal2024gkd

    *Frameworks:* \
    trl `GKDTrainer` ($beta = 1$)
  ], [KD fine-tune],
  [
    - tok-level generalized JSD: $"JSD"(beta) = beta D_"KL" (p parallel m) + (1-beta) D_"KL" (q parallel m)$, $m = beta p + (1-beta) q$
    - at $beta = 1$ (audited here) the trainer computes exact RKL $D_"KL" (q parallel p)$
    - no surrogate — exact computation intended
  ],
  [
    $cal(L) = 1/N sum_t D_"KL" (q(dot|s_t) parallel p(dot|s_t))$ \
    #sidenote[full-vocab, exact]

    - $nabla$ through live $q$ over the whole vocab only; teacher sg; none through prefix sampling

    Where: $N$ $=$ \# response tokens in batch
  ],
  [prefixes: dataset w.p. $1-lambda$, student rollouts w.p. $lambda$ (per-batch coin)],
  [
    - generation at $T=0.9$ but loss at $T=1$
    - top-k/p off
  ],
  [exact token-KL under frozen data distribution — GKD's stated objective],
  [*C* (high)],
  [
    DistiLLM \
    @ko2024distillm
  ], [KD fine-tune \ (only for SRKL)],
  [
    - tok-level skew reverse KL (SRKL), adaptive off-policy
    - no surrogate
  ],
  [
    - full-vocab $D_"KL" (q parallel (1-alpha) p + alpha q)$, $alpha = 0.1$, per position
    - $nabla$ through $q$ everywhere incl. inside the mixture; teacher sg
  ],
  [prefixes: dataset / student replay (adaptive)],
  [none],
  [exact skew-RKL (the claimed objective)],
  [*C* (high)],
  [
    DistiLLM-2 \
    @ko2025distillm2
  ], [KD fine-tune],
  [
    - tok-level SKL on teacher data $+$ SRKL on student data
    - the mixture detach is unstated in the paper
  ],
  [
    - SKL $D_"KL" (p parallel m_1)$, $m_1 = alpha p + (1-alpha) q$, on teacher responses ($q$ inside $m_1$ live)
    - SRKL $D_"KL" (q parallel m_2)$, $m_2 = alpha q + (1-alpha) p$, on student responses ($q$ inside $m_2$ detached)
    - outer $q$, $log q$ live; tok-level full-vocab, per-seq mean
  ],
  [responses generated offline (teacher/student, $T=0.8$)],
  [fixed $alpha=0.1$],
  [
    - value $=$ claimed
    - gradient omits a bounded $O(alpha)$ mixture term that vanishes at the optimum (derivation in @sec-audit-notes)
  ],
  [*B* (mild; med)],
  [
    Qwen3 on-policy distillation \
    @qwen2025qwen3
  ], [post-training],
  [
    - "minimize the KL divergence" by "aligning its logits with those of a teacher model" (Qwen3-32B or Qwen3-235B-A22B) on student-generated sequences (their §4.5, phase 2) — *direction unspecified*
    - no surrogate — no equation anywhere in the report
  ],
  [
    - none recoverable: the quoted sentence is the report's entire specification; no training code released
    - logit-level alignment suggests a full-vocab integrand; direction, token aggregation, and $nabla$ path all unstated
  ],
  [$y tilde.op q$ (`/think` or `/no think` mode)],
  [none stated],
  [
    - indeterminate: full-vocab RKL on student rollouts (the GKD-on-rollouts family) and FKL on the same prefixes are both consistent with the text
    - cited as an RKL-OPD precedent by later frontier reports (@sec-reports) despite never stating a direction
  ],
  [*I* (high — one sentence, no equation, no code)],
  [
    Tinker on-policy distillation \
    @lu2025onpolicydistillation
  ], [post-training],
  [
    - per-token RKL on student rollouts as a dense reward
    - IS loss form per official docs
  ],
  [
    $cal(L) = -sum_t rho_t A_t$ \
    #sidenote[sampled-token, in-loss via advantage]

    - *importance ratio* $rho_t = (q_theta (y_t|s_t))/(q_"sample" (y_t|s_t))$#footnote[$q_theta$: the live student; $q_"sample"$: the student's log-probs recorded at sampling time (the rollout snapshot) — so $rho_t$ is the current-vs-sampling ratio, $= 1$ on fresh rollouts.]
    - *advantage* $A_t = -"sg"[log q_"sample" (y_t|s_t) - log p(y_t|s_t)]$
    - $nabla$ through live $q_theta$ in $rho_t$ only
  ],
  [$y tilde.op q$, $T=1$],
  [
    - no clipping
    - teacher scores the sampled sequence
  ],
  [
    - unbiased for $sum_t bb(E)_(s_t) [nabla D_"KL,t"]$, the frozen-visitation gradient (@sec-deriv)
    - omits visitation term (discount $0$)
  ],
  [*S* (high)],
  [
    Nemotron 3 Ultra MOPD \
    @nvidia2026nemotron3ultra
  ], [MOPD],
  [
    - seq-level RKL (but token-level surrogate) — per-token RKLs summed along the student rollout — to domain teachers $p_(T_i)$, mixed with weights $lambda_i$ (their Eq. 1)
    - surrogate stated: async PPO-clipped update with IcePop masking (their Eqs. 2–3)
  ],
  [
    $cal(J)_"PPO" = sum_t m_t c_t min(rho_t hat(A)_t, "clip"(rho_t, 1 plus.minus epsilon) hat(A)_t)$ (maximized)

    - *advantage = reward* $hat(A)_t = "sg"[log p_(T_i) (y_t|s_t) - log q_"prox" (y_t|s_t)]$#footnote[$q_theta$: the live student; $q_"prox"$: the proximal student snapshot serving as the PPO "old" policy; $q_"behav"$: the stale snapshot that generated the async rollouts; $p_(T_i)$: domain teacher $i$.] \
      #sidenote[sampled-token RKL log-ratio, in-loss via advantage]
    - *importance ratio* $rho_t = q_theta (y_t|s_t) slash q_"prox" (y_t|s_t)$
    - *behav-to-prox IS weight* $c_t = "sg"[q_"prox" (y_t|s_t) slash q_"behav" (y_t|s_t)]$
    - $m_t$ $=$ IcePop token mask $in {0, 1}$ (see "Support mods")
    - $nabla$ through live $q_theta$ in $rho_t$ only
  ],
  [$y tilde.op q_"behav"$ (stale by one optimizer step; async pipelined, §3.6.1; max generation length 192K)],
  [
    - PPO clip
    - IcePop token masking#footnote[The report leaves $m_t$ undefined; the `ultra-v3` code defines it — zero out tokens whose IS weight falls outside bounds (reference $[0.5, 5]$). The shipped public recipe (`mopd.yaml`) instead configures a truncated-IS clamp ("tis", $[0.2, 5]$).]
    - sg behav-to-prox IS ratio
  ],
  [
    - same family as the Tinker row: sampled-token RKL advantage, frozen-visitation surrogate — a partial gradient of their own seq-level Eq. 1; the report's "equivalently, at each prefix" bridge is the value identity @eq-chain, not a gradient claim
    - code-confirmed: `OPDAdvantageEstimator` returns the detached per-token log-ratio; no return-to-go in estimator or loss (@sec-audit-notes)
  ],
  [*S* (med-high — public `ultra-v3` recipe traced; production config unconfirmed)],
  [
    Nemotron-Cascade 2 MOPD \
    @nvidia2026nemotroncascade2
  ], [MOPD],
  [
    - per-token RKL on the sampled token as advantage
    - surrogate stated: REINFORCE with truncated IS correction for train–inference mismatch (their Eqs. 2–4)
  ],
  [
    $cal(L)_"MOPD" = -1/(|cal(V)(y)|) sum_(t in cal(V)(y)) w_t "sg"[a_t] log q(y_t|s_t)$

    - *distillation advantage* $a_t = log p_("dom"_i) (y_t|s_t) - log q(y_t|s_t)$#footnote[$q$: the training-engine student ($pi^"train"$); $q_"inf"$: the inference-engine copy of the student that generated the rollouts (train–inference mismatch); $p_("dom"_i)$: the domain teacher selected for the training example; $cal(V)(y)$: the valid response tokens of rollout $y$ retained by the token mask.] \
      #sidenote[sampled-token RKL log-ratio, in-loss via advantage]
    - *truncated IS weight* $w_t = "sg"[r_t] dot bb(1)[epsilon_"low" <= r_t <= epsilon_"high"]$ (band values in "Support mods")
    - *train-to-inf importance ratio* $r_t = q(y_t|s_t) slash q_"inf" (y_t|s_t)$
    - $nabla$ through $q$ in $log q(y_t|s_t)$ only
  ],
  [$y tilde.op q_"inf"$ (separate inference engine; $T = 1.0$, top-p $1.0$ — Table 8)],
  [
    - truncated IS band $[0.5, 2]$
    - valid-token mask
    - per-seq valid-token mean
  ],
  [
    - same family; per-seq mean makes it a length-normalized variant
    - distillation-only stage (no task reward)
  ],
  [*S* (med — report eqs.; code path not traced)],
  [
    DeepSeek-V4 OPD \
    @deepseek2026v4
  ], [MOPD],
  [
    - seq-level RKL to $N$ domain experts $p_(E_i)$, mixed with importance weights $w_i$ (their Eq. 29): $cal(L)_"OPD" = sum_(i=1)^N w_i dot D_"KL" (q_theta parallel p_(E_i))$
    - no surrogate stated — exact full-vocab computation described in prose; the sampled-token sg-advantage reuse of RL frameworks (the Tinker/Nemotron rows) explicitly rejected ("high variance … training instability")
  ],
  [
    $cal(L) = sum_(i=1)^N w_i sum_t D_"KL" (q_theta (dot|s_t) parallel p_(E_i) (dot|s_t))$ \
    #sidenote[full-vocab, exact per position; teacher logits rebuilt from cached last-layer hidden states; TileLang KL kernel]

    - *expert weight* $w_i$ $=$ per-teacher importance weight #footnote[$p_(E_i)$: domain-expert teacher $i$ (an SFT$+$RL specialist). Each sample is scored by its own domain teacher — samples are dispatched ordered by teacher index, one teacher head in device memory at a time — so the $i$-sum is realized by data routing.]
    - token aggregation (sum vs. length-norm mean) unstated
    - $nabla$ path unstated: no PG/advantage route; presumed pathwise through live $q_theta$ over the vocab only, none through prefix sampling

    Where: $N$ $=$ \# domain teachers ($> 10$)
  ],
  [$y tilde.op q_theta$ (rollout service; per-domain prompts)],
  [none stated (no clip, IS, or top-K; $|V| > 100"k"$)],
  [
    - value unbiased for their Eq. 29 (@eq-chain)
    - presumed-pathwise $bb(E)[nabla] = sum_t bb(E)_(s_t) [nabla D_"KL,t"]$ — the Tinker/Nemotron frozen-visitation gradient reached pathwise (GKD-style): sampled-token variance removed, visitation term still omitted
  ],
  [*S* (low-med — prose-level only: $nabla$ path, aggregation, IS unstated; no code; @sec-audit-notes)],
  [
    *Frameworks:* \
    torchtune `ReverseKLLoss`
  ], [KD recipe],
  [
    - tok-level RKL on dataset prefixes (class docstring)
    - no surrogate
  ],
  [
    - full-vocab $D_"KL" (q parallel p)$ per position
    - $nabla$ through $q$; teacher sg
  ],
  [dataset prefixes],
  [$-100$ label masking],
  [exact token-RKL on dataset visitation],
  [*C*, exists-unused (high)],
  [
    *Frameworks:* \
    NeMo-Aligner `bwd_kl`
  ], [offline teacher-top-K KD],
  [
    - tok-level RKL on teacher top-K support (`bwd_kl` option)
    - truncation inherent to the stored-logits API
  ],
  [
    - $sum_(k in "topK") tilde(q)_k (log tilde(q)_k - log tilde(p)_k)$, both renormalized over teacher top-K ids
    - hand-written analytic backward at the K ids, zero elsewhere
    - teacher $=$ dataset constants
  ],
  [dataset tokens (teacher-forced)],
  [teacher-top-K restriction $+$ renormalization *changes which KL*],
  [exact RKL of the K-renormalized conditionals ($eq.not$ full RKL)],
  [*C* for restricted objective, exists-unused (high)],
  [
    k1-in-reward \
    @ziegler2019finetuning @ouyang2022training

    *Frameworks:* \
    trl PPO (default) \
    OpenRLHF (default) \
    verl (opt-in) \
    Tülu-3-era PPO @lambert2024tulu3 \
    NeMo-Aligner PPO ($|k 1|$ variant)
  ], [RLHF/RLVR regularizer],
  [
    - seq-level RKL-penalized reward maximization
    - $k 1$ in reward is the stated recipe#footnote[Sequence-level as written in @ziegler2019finetuning: $R(x,y) = r(x,y) - beta log[pi(y|x) slash rho(y|x)]$; the per-token placement is stated in @ouyang2022training: "a per-token KL penalty from the SFT model at each token".]
  ],
  [
    $cal(L) = -sum_t min(rho_t A_t, "clip"(rho_t, 1 plus.minus epsilon) A_t) + c_V cal(L)_V$

    - *importance ratio* $rho_t = q_theta (y_t|s_t) slash q_"old" (y_t|s_t)$ ($q_"old"$: rollout-snapshot policy)
    - *advantage* (GAE over rewards $R$ and critic values $V$) $A_t = "sg"[sum_(k>=0) (gamma lambda)^k delta_(t+k)]$, $quad delta_t = R_t + gamma V(s_(t+1)) - V(s_t)$
    - *per-token reward* $R_t = R_t^"task" - beta k 1_t$, $quad k 1_t = log q_"old" (y_t|s_t) - log p(y_t|s_t)$
    - $R_t^"task"$ $=$ task/RM reward, typically nonzero only at the final token; $V$ $=$ learned critic (rollout-time values), trained by the value loss $cal(L)_V$
    - $nabla$ through live $q_theta$ in $rho_t$ only (and through $V$ in $cal(L)_V$)

    Where: $epsilon$ $=$ PPO clip; $gamma, lambda$ $=$ GAE discount and decay (values: footnote in "Resulting objective")
  ],
  [$y tilde.op q$ (rollout)],
  [
    - clamps ($plus.minus 10$ OpenRLHF)
    - adaptive coefs
    - temperature-scaled logprobs
    - NeMo-Aligner uses $|k 1|$ — a different penalty#footnote[$bb(E)|log(q slash p)|$ is not a KL: it upper-bounds the KL and is symmetric in sign (@sec-audit-notes).]
  ],
  [
    - idealized: unbiased $-beta nabla "KL"_"seq"$ incl. visitation#footnote[GAE $gamma = 1.0$ verified at every pinned commit in this row's scope ($lambda$ between 0.95 and 1.0), so the reward-to-go carries the visitation term undiscounted.]
    - in practice GAE/critic/clip surrogate
  ],
  [
    - *C* (high)
    - NeMo-Aligner $|k 1|$: *B* as named, C for what it implements (high)
  ],
  [
    k3-as-loss, plain \
    DeepSeekMath GRPO (as written) \
    @shao2024deepseekmath

    *Frameworks:* \
    verl `low_var_kl` (flagship GRPO recipes) \
    open-instruct (default) \
    trl GRPO (correction off)
  ], [RLVR regularizer],
  [
    - tok-level RKL regularizer
    - $k 3$ stated as an unbiased low-variance *value* estimator, then differentiated as a loss#footnote[DeepSeekMath Eq. 4, citing @schulman2020klapprox — where the single-sample estimators are proposed for the KL *value*, "if KL is just being used as a diagnostic".]
  ],
  [
    - $beta (e^(log p - log q) - (log p - log q) - 1)$ per token, in loss (sampled-token)
    - ref sg, $nabla$ through $log q$: $nabla k 3 = (1-r) nabla log q$
  ],
  [$y tilde.op q$],
  [
    - clamps: verl/nemo-rl $plus.minus 20$ on $log r$, $plus.minus 10$ on $k 3$
    - open-instruct $plus.minus 40$
  ],
  [
    - *$bb(E)[nabla] = nabla D_"KL" (p parallel q)$ — forward-KL gradient* (@eq-k3)
    - value estimate remains unbiased for reverse KL
  ],
  [*B* (high; assumes on-policy, clamps not binding)],
  [
    k3 $times$ differentiable ratio \
    @tang2025pitfalls

    *Frameworks:* \
    trl GRPO (default) \
    nemo-rl $exp(x - "sg"[x])$ (default) \
    open-instruct estimator 3 ($=$ ratio·k1)
  ], [RLVR regularizer],
  [
    - tok-level RKL regularizer
    - surrogate stated: bias-corrected $k 3$#footnote[nemo-rl cites Tang and Munos for the correction factor; trl's `use_bias_correction_kl` (default on) was added after DeepSeek-V3.2.]
  ],
  [
    - $rho dot k 3$ with $rho$ carrying $nabla log q$, per token, in loss (sampled-token)
    - $nabla$ through both $rho$ and $k 3$/$k 1$
  ],
  [$y tilde.op q$],
  [as the k3-as-loss row],
  [
    - on-policy $bb(E)[nabla] = nabla D_"KL" (q parallel p)$ (@eq-rhok3)
    - off-policy needs RPG-style weighting; still frozen-visitation per token
  ],
  [*C* on-policy (med-high)],
  [
    k2-as-loss

    *Frameworks:* \
    verl "mse"/k2 \
    OpenRLHF k2 \
    verl "k3+" (straight-through)
  ], [RLHF/RLVR regularizer],
  [
    - tok-level RKL regularizer
    - $k 2$ stated as the low-variance, biased-for-the-value option @schulman2020klapprox
  ],
  [
    - $(log q - log p)^2 slash 2$ per token, in loss (sampled-token)#footnote[verl's "k3+" straight-through option returns $k 3$'s value with $k 2$'s gradient.]
    - $nabla k 2 = k 1 nabla log q$
  ],
  [$y tilde.op q$],
  [clamps],
  [$bb(E)[nabla] = nabla D_"KL" (q parallel p)$ on-policy (@eq-k2)],
  [*C* (high)],
  [
    *Frameworks:* \
    trl-classic `"full"`
  ], [RLHF regularizer],
  [
    - seq-level RKL penalty in reward (opt-in `"full"` option)
    - no estimator — exact per-position KL
  ],
  [
    $cal(L) = -sum_t min(rho_t A_t, "clip"(rho_t, 1 plus.minus epsilon) A_t) + c_V cal(L)_V$ — the k1-in-reward row's PPO pipeline ($rho_t$, GAE $A_t$, $cal(L)_V$ as defined there), with the penalty derandomized:

    - *per-token reward* $R_t = R_t^"task" - beta dot D_"KL" (q_"old" (dot|s_t) parallel p(dot|s_t))$ \
      #sidenote[exact full-vocab KL, from rollout-snapshot log-softmaxes]
    - $R_t^"task"$ $=$ RM reward at the final token
    - $nabla$ through live $q_theta$ in $rho_t$ only
  ],
  [$y tilde.op q$ (prefix visitation)],
  [none],
  [
    - value exact per position, but $bb(E)[nabla] = -beta G_"visit"$ *only*: the penalty is prefix-measurable, so the own-position score term is zero-mean and `no_grad` blocks the pathwise route (@sec-deriv)
    - dual of the token-level pitfall: $G_"local"$ dropped, visitation kept (@sec-audit-notes)
  ],
  [*B*, opt-in (high; value exact)],
)
]

#[
#set page(flipped: true)

== Source table <sec-table2>
#set text(size: 7.3pt)

*Table 2 — Sources.* One row per distinct training stage / implementation. "Used?" = whether that code path was exercised in the reported experiments (evidence: run scripts, configs, docs). Commits: LMOps `4f2a9de`, trl `24db2a8` (main) / `1f8ba92` (v0.11.4) / `04fd120` (v0.24.0), distillm `d47e77f`, distillm-2 `fe4cf9b`, tinker-cookbook `3d3e976`, torchtune `bd2a0fc`, modelopt `9220fac`, NeMo-Aligner `af2b3bd`, verl `b75f0d0`, OpenRLHF `bc71bb1`, open-instruct `05ad0a2` (and `9502f16` for the deleted Tülu-3-era PPO), nemo-rl `037fb36`, NVIDIA-NeMo/RL branch `ultra-v3` `7314528`.

#table(
  columns: (1.0fr, 0.5fr, 1.05fr, 2.1fr, 1.25fr, 1.25fr),
  stroke: 0.4pt + gray,
  table.header([*Report / model*], [*Stage*], [*Claimed objective*], [*Implementation \@ commit*], [*Used?*], [*Evidence*]),
  [MiniLLM @gu2024minillm], [KD fine-tune], [sequence-level RKL], [microsoft/LMOps `minillm` \@ `4f2a9de`: #link("https://github.com/microsoft/LMOps/blob/4f2a9deb5f08e459fd44c2e4792344d78ca89fc3/minillm/minillm/losses.py#L58-L196")[`losses.py` L58–196], `sampler.py` L68–131, `reward.py` L38–83; patched sampling in t1101675/transformers \@ `be0435e`], [*used*], [#link("https://github.com/microsoft/LMOps/blob/4f2a9deb5f08e459fd44c2e4792344d78ca89fc3/minillm/scripts/gpt2/minillm/train_base_xl.sh#L69-L84")[run scripts] set `--type minillm --single-step-reg --teacher-mixed-alpha 0.2` etc.],
  [GKD @agarwal2024gkd], [KD fine-tune], [token-level JSD($beta$); RKL at $beta=1$], [huggingface/trl \@ `24db2a8`: #link("https://github.com/huggingface/trl/blob/24db2a825029f7e0e60f6d3c160485565a784503/trl/experimental/gkd/gkd_trainer.py#L238-L323")[`experimental/gkd/gkd_trainer.py` L238–323]], [reference impl.; *paper code not public* (Google-internal, T5)], [docs claim fidelity to paper Eq. 1; no official GKD release found in google-research],
  [DistiLLM @ko2024distillm], [KD fine-tune], [skew SKL / SRKL, adaptive off-policy], [jongwooko/distillm \@ `d47e77f`: #link("https://github.com/jongwooko/distillm/blob/d47e77ff9d27721783b32213be38c1204230cc0a/distillm/losses.py#L66-L95")[`losses.py` L66–95], `finetune.py` L171–193, L290–325], [*used*], [scripts set `adaptive-sfkl`/`adaptive-srkl`, `--student-gen`, `kd-ratio 1.0`; one openllama2 script's `srkl-aesop` type matches no code branch (static read)],
  [DistiLLM-2 @ko2025distillm2], [KD fine-tune], [SKL on teacher data $+$ SRKL on student data], [jongwooko/distillm-2 \@ `fe4cf9b`: #link("https://github.com/jongwooko/distillm-2/blob/fe4cf9bfbb4f83219dd1d69219800164b59685fa/src/distillm_trainer.py#L1036-L1099")[`distillm_trainer.py` L1036–1242]], [*used*], [`distillm_v2` config; adaptive-$alpha$ and gradual-$beta$ code present but hard-disabled (L507–509)],
  [TML on-policy distillation @lu2025onpolicydistillation], [post-training], [per-token RKL on student rollouts], [thinking-machines-lab/tinker-cookbook \@ `3d3e976`: #link("https://github.com/thinking-machines-lab/tinker-cookbook/blob/3d3e9767076d6f44825fb46c094a4e0fce39bba1/tinker_cookbook/distillation/train_on_policy.py#L54-L130")[`train_on_policy.py` L54–130]; loss kernel server-side (docs-verbatim)], [*used* (public code for blog); kernel closed-source], [recipe defaults `kl_penalty_coef=1.0`, discount `0.0`, `importance_sampling` loss],
  [Nemotron 3 Ultra / Cascade 2 @nvidia2026nemotron3ultra @nvidia2026nemotroncascade2], [MOPD], [per-token RKL advantage (Ultra: PPO-clipped $+$ IcePop; Cascade 2: truncated-IS REINFORCE)], [NVIDIA-NeMo/RL `ultra-v3` \@ `7314528`: #link("https://github.com/NVIDIA-NeMo/RL/blob/7314528b73bd3449f4fd83e6101de8d5311b42ac/nemo_rl/algorithms/advantage_estimator.py#L149-L240")[`advantage_estimator.py` L149–240] (`OPDAdvantageEstimator`), `loss_functions.py` (tis/IcePop modes), #link("https://github.com/NVIDIA-NeMo/RL/blob/7314528b73bd3449f4fd83e6101de8d5311b42ac/examples/configs/ultra/mopd.yaml")[`examples/configs/ultra/mopd.yaml`]], [public recipe ("Nemotron 3 Ultra — MOPD Stage 1"); production run unconfirmed], [advantage `= (teacher_logprobs - prev_logprobs).detach()`, no return-to-go in estimator or loss; yaml: `adv_estimator: opd`, "tis" clamp $[0.2, 5]$, `reference_policy_kl_penalty: 0.0`; docstring cites MiMo-V2-Flash Eq. 8 @xiaomi2026mimov2],
  [Qwen3 @qwen2025qwen3], [on-policy distillation], ["minimize the KL divergence" on logits (direction unspecified)], [none released], [—], [report text only],
  [torchtune], [KD recipe], [forward KL (RKL class shipped)], [pytorch/torchtune \@ `bd2a0fc`: #link("https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/loss/kd_losses.py#L11-L114")[`kd_losses.py` L11–114]], [FKL *used* in all shipped configs; `ReverseKLLoss` *exists-unused*], [grep over `recipes/configs/`: only `ForwardKLWithChunkedOutputLoss` referenced],
  [NVIDIA modelopt], [Minitron-style KD], [forward KL], [TensorRT-Model-Optimizer \@ `9220fac`: #link("https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/9220fac0535e4afd7e9929822eb332e09ea8a4f8/modelopt/torch/distill/plugins/megatron.py#L292-L370")[`plugins/megatron.py` L292–370]], [forward *used*; `reverse=True` flag *exists-unused* (unreachable from `DistillationConfig`)], [config instantiates `LogitsKLLoss` without exposing `reverse`],
  [NeMo-Aligner KD], [offline teacher-top-K KD], [`fwd_kl` default; `bwd_kl` = teacher-top-K-renormalized RKL], [NeMo-Aligner \@ `af2b3bd`: #link("https://github.com/NVIDIA/NeMo-Aligner/blob/af2b3bd96ad78190f6ebfb552b9e599e313ce027/nemo_aligner/utils/distributed.py#L510-L760")[`distributed.py` L510–760] (hand-written backward)], [fwd *used* (tutorial); *`bwd_kl` exists-unused*; which path Minitron production used: *indeterminate*], [docs/tutorial set `fwd_kl`; no shipped config invokes `bwd_kl`],
  [RLHF classic (trl $<=$0.11 PPO)], [RLHF regularizer], [RKL penalty], [trl \@ `1f8ba92`: #link("https://github.com/huggingface/trl/blob/1f8ba929fbb6962249386f7dce20155da8a26cda/trl/trainer/ppo_trainer.py#L1150-L1165")[`ppo_trainer.py` L1150–1165]], [k1-in-reward *default*; "full" (full-vocab RKL) opt-in], [config defaults `kl_penalty="kl"`, adaptive coef 0.2],
  [trl $<=$0.24 PPO], [RLHF regularizer], [RKL penalty], [trl \@ `04fd120`: `ppo_trainer.py` L558–563], [k1-in-reward *default*; k3 opt-in], [`kl_coef 0.05`; both sides temperature-scaled (default 0.7)],
  [trl GRPOTrainer (main)], [RLVR regularizer], [RKL penalty (k3)], [trl \@ `24db2a8`: #link("https://github.com/huggingface/trl/blob/24db2a825029f7e0e60f6d3c160485565a784503/trl/trainer/grpo_trainer.py#L3094-L3101")[`grpo_trainer.py` L3094–3146]], [$beta=0$ default (*opt-in*); when on: k3·IS-ratio in-loss (`use_bias_correction_kl=True`)], [config cites DeepSeek-R1; bias correction added after DeepSeek-V3.2],
  [verl], [RLVR regularizer], [RKL penalty], [verl \@ `b75f0d0`: #link("https://github.com/volcengine/verl/blob/b75f0d06a72b74e471cd4d3402bcbb0ceaa2e0c8/verl/trainer/ppo/core_algos.py#L2187-L2251")[`core_algos.py` L2187–2251], `losses.py` L131–142], [in-loss k3 (`low_var_kl`) set in flagship GRPO recipes; in-reward k1 opt-in; `"full"` raises `NotImplementedError`; `"k3+"` straight-through variant opt-in], [`run_qwen3_8b_fsdp.sh` L153–155],
  [OpenRLHF], [RLHF regularizer], [RKL penalty], [OpenRLHF \@ `bc71bb1`: #link("https://github.com/OpenRLHF/OpenRLHF/blob/bc71bb19464aca306b33080b2d2bb45d154e2f49/openrlhf/models/utils.py#L64-L95")[`models/utils.py` L64–95]], [k1-in-reward *default* (coef 0.01); in-loss k1/k2/k3 opt-in], [historical flag `--use_kl_estimator_k3` (v0.5.7)],
  [open-instruct (Tülu 3 / OLMo)], [RLVR regularizer], [RKL penalty], [\@ `05ad0a2`: #link("https://github.com/allenai/open-instruct/blob/05ad0a239663bfbeab8b72d949c9534504d49ca5/open_instruct/model_utils.py#L793-L818")[`model_utils.py` L793–818], `grpo_utils.py` L504–538; Tülu-3-era PPO \@ `9502f16`], [grpo_fast in-loss k3 *default* (Tülu 3 recipe $beta=0.01$); paper-era PPO used k1-in-reward], [recipe `grpo_fast_8b.sh`; deleted PPO file inspected at pinned commit],
  [NeMo-Aligner PPO], [RLHF regularizer], [RKL penalty], [\@ `af2b3bd`: `ppo_utils.py` L63–82], [$|k 1|$-in-reward *default* (`use_absolute_kl: True`)], [yaml defaults, coef 0.01],
  [NVIDIA-NeMo/RL (nemo-rl)], [RLVR regularizer], [RKL penalty], [\@ `037fb36`: #link("https://github.com/NVIDIA-NeMo/RL/blob/037fb36bdad40f6336edcdfd39f7c5150b152324/nemo_rl/algorithms/utils.py#L42-L100")[`utils.py` L42–100], `loss_functions.py` L466–521], [in-loss k3·$exp(x - "sg"[x])$ *default* (cites Tang & Munos)], [`grpo_math_1B.yaml`: k3, coef 0.01],
  [DeepSeekMath GRPO @shao2024deepseekmath], [RLVR regularizer], [RKL penalty (k3 in loss, Eq. 4)], [no official training code], [—], [paper equations; open frameworks above are the de-facto implementations],
)
]

== Audit notes and derivations for the non-trivial rows <sec-audit-notes>

*MiniLLM (S, medium).* The claimed objective is sequence-level reverse KL; the shipped estimator is a PPO surrogate whose principal deviations are stated in the paper as deliberate techniques: the exact single-step term (a full-vocabulary per-position RKL, intended in the paper as a variance reduction that *replaces* the sampled own-token term), teacher-mixed sampling (rollouts from $p_"sample" = 0.8 q_"old" + 0.2 p$ against reward hacking), and length normalization (against short-response bias — note this *changes* the objective to a length-normalized variant, which is the point). Unstated deviations: discount $gamma = 0.95$, advantage whitening, PPO clipping with four reuse epochs, reward scaling and clipping; the retained own-token reward — the paper's decomposition assigns $R_(t+1)$ (excluding $r_t$) to the policy-gradient part, but `losses.py` builds $A_t$ from the full $R_t$ (`last_rw = rewards[:, t] + gamma * last_rw`) while `--single-step-reg` adds the exact term, so the local gradient component enters twice (once sampled, whitened, and length-normalized; once exact); and — the subtle one — mixture sampling is corrected only by a *per-token* importance weight $w_t = q_"old" (y_t) slash p_"sample" (y_t)$ (#link("https://github.com/microsoft/LMOps/blob/4f2a9deb5f08e459fd44c2e4792344d78ca89fc3/minillm/minillm/sampler.py#L110-L115")[`sampler.py` L110–115]), which reweights the action at each step but not the visitation distribution of prefixes (that would require the product of ratios along the prefix). Every listed mechanism is visible in the pinned code. Classification *B* (mild) rather than S: for the PPO composite alone, "correct for a stated surrogate" would apply (its relation to $nabla "KL"_"seq"$ is not characterized anywhere, only its ingredients), but the single-step handling deviates from the paper's *own stated algorithm* — Algorithm 1 updates with $(nabla cal(L))_"Single" + (nabla cal(L))_"LongNorm" + nabla cal(L)_"PT"$, the long-term part defined with $R_(t+1)$, i.e. the exact term *replaces* the sampled own-token term, whereas the code retains the full $R_t$ in the advantage and adds the exact term ("single-step regularization" appears nowhere in the paper; the added-regularizer reading exists only in the code's flag name). Unstated deviations from a source's own equations are classified B in this audit (the DistiLLM-2 pattern); mild because the extra term is a second copy of a legitimate gradient component — it vanishes at the optimum and reweights local versus visitation pressure rather than pointing toward a different divergence. Medium confidence.

*DistiLLM-2 (B mild, medium).* The claimed loss on student data is $D_"SRKL" = sum_v q_v (log q_v - log m_v)$ with $m = (1-alpha) p + alpha q$. Exact differentiation gives $partial D slash partial q_v = (log q_v - log m_v) + 1 - alpha q_v slash m_v$. The code (#link("https://github.com/jongwooko/distillm-2/blob/fe4cf9bfbb4f83219dd1d69219800164b59685fa/src/distillm_trainer.py#L1194-L1198")[`distillm_trainer.py` L1194–1198]) detaches $q$ inside the mixture, computing $(log q_v - log m_v) + 1$ and omitting $-alpha q_v slash m_v$. Since $m_v >= alpha q_v$, the omitted term is bounded in $(-1, 0]$; at the optimum $q = p$ it equals $-alpha$ *uniformly* across the vocabulary, and uniform shifts are annihilated by the softmax Jacobian — so the gradient bias vanishes at the optimum and is bounded elsewhere. The detach appears deliberate (it is asymmetric with the SKL branch, where $q$ inside the mixture stays live) but is not stated in the paper, hence "biased (mildly)" rather than "stated surrogate"; medium confidence.

*Tinker (S, high).* The blog states the method as per-token reverse KL; the code computes the advantage $A_t = -"sg"[log q_"sample" (y_t) - log p(y_t)]$ and feeds it to a server-side importance-sampling loss $-sum_t exp(log q_theta - log q_"sample") A_t$ (kernel formula from official docs; the kernel itself is closed-source — the one indeterminacy). By the derivation in @sec-deriv this is unbiased for the frozen-visitation gradient $sum_t bb(E)_(s_t)[nabla D_"KL,t"]$ — the same objective GKD computes pathwise over the full vocabulary, estimated here from one sampled token via the score function. It is *not* the full sequence-KL gradient (visitation term omitted; restorable via `kl_discount_factor = 1`, shipped default $0$). High confidence: the mechanics are simple and the docs kernel matches the observed inputs.

*Nemotron 3 Ultra MOPD — code trace (S, med-high).* The `ultra-v3` branch of NVIDIA-NeMo/RL (\@ `7314528`) makes the public path traceable: `examples/configs/ultra/mopd.yaml` ("Nemotron 3 Ultra — MOPD Stage 1") selects `adv_estimator: opd`, and `OPDAdvantageEstimator.compute_advantage` returns `(teacher_logprobs - prev_logprobs).detach()` — the per-token advantage of the report's Eq. 2, with no return-to-go anywhere in the estimator or loss (no reverse accumulation exists in either file). This confirms at code level that the update is $G_"local"$ under the behavior policy's visitation: a partial gradient of the report's own sequence-level Eq. 1 in exactly Tang and Munos's sense, nowhere acknowledged as such — the report's bridge sentence "Equivalently, at each prefix $s_t$ …" is the value identity @eq-chain, which holds for the objective value while the gradients differ by $G_"visit"$. Report–code discrepancies: the shipped recipe configures a truncated-IS clamp ("tis", $[0.2, 5]$) rather than IcePop masking (implemented as an option — zero out tokens with IS weight outside bounds, reference $[0.5, 5]$), and `reference_policy_kl_penalty: 0.0` confirms the advantage is the sole training signal. The estimator's docstring attributes the advantage to "Equation 8 from the MOPD paper" — Xiaomi's MiMo-V2-Flash @xiaomi2026mimov2, which introduced the MOPD paradigm (explicit reverse-KL loss, gated sampled-token advantage, length-normalized REINFORCE-style update, optional ORM-advantage blending; its Eqs. 5–9). The public recipe is not guaranteed identical to the production run, hence med-high rather than high.

*DeepSeek-V4 OPD (S, low-med).* The report states the objective at sequence level — Eq. 29, a $w_i$-weighted sum of policy-level reverse KLs to more than ten domain experts — and the computation at the *integrand* level only: "full-vocabulary logit distillation", with "the exact KL divergences between teacher and student logits … computed using a specialized TileLang kernel" on trajectories sampled from the student (§5.1.2, §5.2.2). No loss-level equations or training code exist, so three estimator-defining facts are unstated: the token aggregation, the gradient path, and any train–inference correction for the rollout service (§5.2.3; sampling temperature and $w_i$ values are also unspecified). The classification presumes the natural reading — pathwise differentiation of the per-position exact KL, no gradient through prefix sampling — under which the value is unbiased for Eq. 29 by @eq-chain and the expected gradient is the frozen-visitation surrogate $sum_t bb(E)_(s_t) [nabla D_"KL,t"]$: the same partial gradient as the Tinker and Nemotron rows, reached by the pathwise route instead of the score function. The report's own critique of that family ("high variance in gradient estimation and often causes training instability") concerns the sampled-token *integrand*, which the full-vocabulary computation indeed removes; the omitted visitation term is shared. Alternatives that would change the class: a score-function term restoring $G_"visit"$ (would be C; nothing in the report suggests one), or the advantage-based RL-framework reuse (explicitly rejected). Low-med confidence: prose-level evidence only.

*k1-in-reward caveat, NeMo-Aligner PPO.* With `use_absolute_kl: True` (the default) the "KL" penalty is $|log q - log p|$, whose expectation is *not* the KL divergence (it upper-bounds it and is symmetric in sign); as a named reverse-KL penalty this is inconsistent, as an absolute-log-ratio proximity penalty it is correctly implemented. Flagged because downstream reports citing "a KL penalty" in NeMo-Aligner-era NVIDIA RLHF inherit this substitution silently.

*k3-as-loss, the GRPO k3 pattern (B, high).* DeepSeekMath's Eq. 4 introduces $k 3$ as an "unbiased estimator" — true for the *value* — and then differentiates it as a loss. By @eq-k3 the expected gradient is $nabla D_"KL" (p parallel q)$: deployed "reverse-KL-regularized" GRPO in verl (`low_var_kl`) and open-instruct (`kl_estimator 2`) applies, in expectation, *forward*-KL pressure toward the reference. Three independent sources reach the same conclusion @tang2025pitfalls @liu2025rethinkingkl @zhang2025rpg, and two frameworks have since shipped the k3 $times$ differentiable-ratio repair (TRL's `use_bias_correction_kl`, default on; nemo-rl's surrogate factor, citing Tang and Munos). Confidence high; assumptions: on-policy updates, clamps not binding (verl clamps $log r$ to $plus.minus 20$ and $k 3$ to $plus.minus 10$, open-instruct to $plus.minus 40$ — the clamps censor exactly the tail events where the two directions differ most).

*trl-classic `"full"` (B, high).* The opt-in `kl_penalty="full"` computes the exact per-position $D_"KL" (q parallel p)$ over the full vocabulary — `F.kl_div(ref_logprob, logprob, log_target=True).sum(-1)` — and places it in the reward under `no_grad`. Unlike $k 1_t$, the penalty does not depend on the sampled token, so the own-position score term has zero expectation and the stop-gradient removes the pathwise term; by the derivation in @sec-deriv, the idealized expected gradient of the penalty is $-beta G_"visit"$ alone. The option steers rollouts away from high-KL prefixes but exerts, in expectation, no direct pressure on any conditional $q_theta (dot|s_t)$ — the mirror image of the token-level pitfall, and a second instance of value-exactness mistaken for gradient-correctness. Verified at the pinned commit: full-vocab branch, `no_grad` reward construction, per-token placement, GAE with $gamma = 1.0$ (`lam` 0.95, `whiten_rewards` off by default), and the `[:, :-1]` logit slice aligning each penalty with the action drawn from the same prefix. Not covered by Tang and Munos; derived from @eq-master.

== Support and decoding modifications that change which KL is estimated <sec-support>

Four recurring modifications alter the estimated divergence, usually silently.

- *Temperature.* trl's newer PPO computes both $log q$ and $log p$ at the sampling temperature (default $0.7$) — the regularizer is the reverse KL between *temperature-scaled* models, not the $T=1$ models; GKD's trainer conversely generates at $T=0.9$ but computes the loss at $T=1$ (legitimate under its any-data-distribution objective, but worth knowing). trl-classic ($<=$0.11) allowed user generation kwargs to diverge silently from its $T=1$ scored logprobs.

- *Support truncation and renormalization.* NeMo-Aligner's KD kernel and modelopt's `TopKLogitsKLLoss` renormalize both distributions over the teacher's top-K ids — the result is the exact KL of *different* (conditional) distributions; Gemma 3's pretraining KD does the same with 256 sampled logits ("set to zero probability for non-sampled logits, and renormalized" @gemmateam2025gemma3); nemo-rl's on-policy distillation loss (also teacher top-k: ids from the teacher's `get_topk_logits`, student logprobs gathered at those ids) adds an explicit tail-entropy correction, and its RL path is the only one audited that deliberately computes the KL on *unfiltered* logprobs when rollout top-k/top-p filtering is active.

- *Clamps and clips.* log-ratio and estimator clamps (verl $plus.minus 20 slash plus.minus 10$, OpenRLHF $plus.minus 10$, open-instruct $plus.minus 40$, MiniLLM's reward clip $plus.minus 100$) censor the tails — precisely the region where $q > 0, p approx 0$ events give reverse KL its zero-forcing character; they also hide the infinities that arise on zero-probability support.

- *Sampling-distribution substitution.* MiniLLM's rollouts come from a teacher-mix, not $q$, with only a per-token correction (@sec-table1).

= Reported empirical effects <sec-empirical>

*Diversity.* The most careful controlled study, #cite(<kirk2024understanding>, form: "prose"), finds RLHF "substantially decreases the diversity of outputs sampled for a given input compared to SFT" with smaller across-input effects, and — crucially for the direction question — that the reverse-KL knob does not buy the diversity back:

#quote(block: true, attribution: [@kirk2024understanding · #emph[Understanding the Effects of RLHF on LLM Generalisation and Diversity]])[The results show that this does not work – increasing the KL penalty coefficient leads to a drop in performance as expected, but also to a drop in per-input diversity, rather than a gain.]

In distribution-matching settings the direction is directly implicated: RKL-DPG shows "low diversity but high quality … suggesting it captures a subset of the target distribution ('mode collapse')" @go2023aligning; f-DPO measures lowest predictive entropy for RKL, highest for FKL, JSD between @wang2024beyond; BOND reports "the backward KL can lead to policy and entropy collapses" @sessa2024bond. In KD, f-DISTILL quantifies RKL's coverage deficit @wen2023fdistill, while MiniLLM reports approximately preserved distinct-4 @gu2024minillm and AKL improves quality *and* diversity simultaneously @wu2025rethinking — evidence the trade-off is not intrinsic. The 2026 entropy-aware OPD results localize the effect: diversity loss and signal instability concentrate at positions where the teacher distribution has high entropy @jin2026entropy.

*Mode collapse.* #cite(<kirk2024understanding>, form: "prose") claim the first rigorous demonstration of across-input mode collapse from RLHF specifically; f-DPG's toy-scale observation adds an important qualifier — the entropy crash is transient when the model class can represent the target and severe under misspecification @go2023aligning, consistent with the optimization-centric account of @sec-folklore-crit.

*Calibration.* GPT-4's post-training "hurts calibration significantly" (Fig. 8), attributed to post-training as a whole rather than the KL direction @openai2023gpt4. f-DPO is the source that ties calibration to the constraint choice: "divergence constraints directly influence expected calibration error" @wang2024beyond. Against the folklore's expectation, MiniLLM's reverse-KL students were *better* calibrated than its KD baselines (ECE 0.099 on SST-2 vs SeqKD's 0.243) @gu2024minillm.

*Reward and convergence.* RL reward grows roughly linearly in $sqrt(D_"KL" (pi parallel pi_"init"))$ @bai2022training @gao2023scaling. As an RL substitute, on-policy reverse-KL distillation converges dramatically faster than RLVR: TML's 7–10$times$ step efficiency and 9–30$times$ cost reduction @lu2025onpolicydistillation, Qwen3's "approximately only 1/10 of the GPU hours" with improved pass\@64 @qwen2025qwen3, and Nemotron-Cascade 2's MOPD converging "in 40–50 steps" while beating GRPO's training efficiency on AIME25 @nvidia2026nemotroncascade2. Frontier adoption followed: DeepSeek-V4 replaced its mixed-RL consolidation stage with pure multi-teacher reverse-KL distillation @deepseek2026v4, and Kimi K3 and GLM-5 ship the same sampled-token recipe @kimi2026k3 @glm2026glm5. On stability, the estimator-level record is mixed: MiniLLM's PG needed three patches; DistiLLM attributes instability to unbounded gradient ratios and fixes it by skewing @ko2024distillm; SKD documents garbage teacher feedback on early student samples @xu2025skd; supervision-fidelity decay fades the teacher's corrective signal on long student prefixes @liu2026teacher; and thinking-pattern incompatibility can make a stronger teacher useless where a weaker one helps @li2026rethinking.

*Downstream quality.* Reverse-KL-family methods win on open-ended instruction following (MiniLLM, GKD's RKL/JSD variants, DistiLLM's SRKL), while the direction is task-dependent elsewhere (GKD; f-DISTILL's MT exception where "mode averaging does not occur") and irrelevant at convergence for representable students (Wu et al.). In production, MOPD gains concentrate exactly where the folklore predicts reverse KL should shine — recovering teacher behaviors on the student's own trajectories (Terminal Bench $+16.3$; restoration of regressed benchmarks) — with the reported weakness on out-of-support prompts (HLE) attributed by the authors to "a limitation of the on-policy distillation setting rather than a failure of the teacher" @nvidia2026nemotron3ultra. In RLVR, ProRL uses the reverse-KL penalty *for* entropy:

#quote(block: true, attribution: [@liu2025prorl · #emph[ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models]])[While DAPO and temperature adjustment help slow entropy collapse, we find that explicit regularization via a KL divergence penalty provides a stronger and more stable solution.]

The same term that reduces diversity relative to distribution matching preserves it relative to unregularized reward maximization; the two claims are consistent — they differ in baseline.

= Reproducing the Tinker recipe <sec-repro>

The Tinker recipe @lu2025onpolicydistillation has become the de-facto shared baseline of the 2026 on-policy-distillation literature, but no two papers run it identically. Table 3 collects, per experimental *setting*, the setup needed to reproduce each training run: one column per setting, with closely related settings of the same work sharing a column and distinguished by color (legend in each column header; black values are shared by all of a column's settings). Plain values are stated in the primary source (paper or blog); #text(size: 0.85em, fill: luma(120))[[code]] marks values recoverable only from released code, configs, or SDK defaults; #text(size: 0.85em, fill: luma(120))[[inf.]] marks inferences (arithmetic, framework defaults); "NR" = not reported anywhere examined (paper including appendices, released code, model cards). Quote-level provenance for every cell: `lit_review/notes/repro-settings-dossier.md`.

Two findings deserve flagging before the table. First, the blog's headline "Qwen3-32B → Qwen3-8B-Base" math run carries a footnote reversing its own teacher identity: "We actually use Qwen3-8B as a teacher, as it performs slightly better. However, for the purpose of comparing compute, we can still measure the FLOPs of the 32B model" — the 32B teacher exists only in the FLOPs accounting, and the one paper that faithfully ran a true 32B→8B pipeline (AOPD @jia2026aopd) is reproducing a configuration the blog itself did not run. Second, the evaluation protocol behind the blog's 60%/70% AIME'24 numbers was never published: no sample count, metric, temperature, or grader at blog time, with only a partial disclosure in the June 2026 cookbook README (for the Qwen3.5 rerun: temperature 1.0, top-p 1.0, top-k −1, max_tokens 64000 — samples per problem still unstated). Of the eight works tabulated, four (TML, OPD+ @zhao2026opdplus, AOPD, Geometry @shen2026geometry) report no complete evaluation protocol, and Geometry reports no numeric accuracy at all (training curves only).

#[
#set page(width: 88cm, height: auto, margin: 1.2cm)
#set text(size: 6.8pt)
#show table: set text(size: 6.8pt)

#let cA(body) = text(fill: rgb("#0f62ac"), body)
#let cB(body) = text(fill: rgb("#c0392b"), body)
#let cC(body) = text(fill: rgb("#1e7d32"), body)
#let cD(body) = text(fill: rgb("#b9770e"), body)
#let cE(body) = text(fill: rgb("#7d3c98"), body)
#let nr = text(fill: luma(135), [NR])
#let tg(t) = text(size: 0.8em, fill: luma(120), t)
#let grp(t) = table.cell(colspan: 10, fill: luma(235), align: left, text(weight: "bold", t))

*Table 3 — Reproduction setups for Tinker-recipe-family settings.* Colors identify settings within a column (header legend). NR = not reported; #tg[[code]] = from released code/configs/SDK defaults, not the paper; #tg[[inf.]] = inferred. Sub-variants (SFT-warm-up length, teacher sweeps) noted inline. "→" separates result progressions (baseline → method arm).

#table(
  columns: (3.0cm, 7.4cm, 7.4cm, 7.8cm, 8.2cm, 8.6cm, 8.8cm, 8.6cm, 8.4cm, 12.6cm),
  stroke: 0.4pt + gray,
  table.header(
    [],
    [*TML T1 — blog headline (full FT)* \ @lu2025onpolicydistillation],
    [*TML T2 — cookbook (LoRA-128)* \ @lu2025onpolicydistillation],
    [*OPD+* @zhao2026opdplus \ #cA[B1 math] · #cB[B2 tool-use]],
    [*AOPD* @jia2026aopd \ #cA[A1 32B→8B] · #cB[A2 8B→4B]],
    [*Geometry of OPD* @shen2026geometry \ #cA[G1 8B←32B] · #cB[G3 4B, teacher sweep] · #cC[G4 14B]],
    [*SCOPE* @zheng2026scope \ #cA[S1 1.5B math] · #cB[S2 Qwen3-1.7B math]],
    [*EOPD* @jin2026entropy \ #cA[E1 0.6B] · #cB[E2 1.7B] · #cC[E3 4B]],
    [*LGR* @liu2026teacher \ #cA[L1 math 1.5B] · #cB[L3 math 7B]],
    [*Rethinking OPD* @li2026rethinking \ #cA[R1–R2] · #cB[R3 cold-start] · #cC[R4–R5] · #cD[R6–R8] · #cE[R9–R11]],
  ),
  grp[Identity],
  [teacher],
  [Qwen3-32B #tg[(as presented)] \ #sidenote[blog footnote: "We actually use Qwen3-8B as a teacher, as it performs slightly better" — 32B enters only the FLOPs accounting]],
  [Qwen3-8B #tg[(script default; README passes none)]],
  [Qwen3-8B],
  [#cA[Qwen3-32B] \ #cB[Qwen3-8B]],
  [#cA[Qwen3-32B; sub-var.: Qwen3-30B-A3B] \ #cB[Qwen3-8B / 14B / 32B] \ #cC[Qwen3-32B]],
  [#cA[Skywork-OR1-7B] \ #cB["Qwen3-8B-Instruct" as printed — no such release; repo uses Qwen3-8B #tg[[code]]; think-mode #nr]],
  [Qwen3-8B, non-thinking (chat template pre-fills empty `<think>` block)],
  [#cA[Skywork-OR1-Math-7B] \ #cB[R1-Distill-Qwen-32B]],
  [#cA[R1: Qwen3-4B (NT) · R2: Qwen3-4B-Base-GRPO (author-trained, HF)] \ #cB[R3: Qwen3-4B (NT)] \ #cC[R4: Qwen3-4B (NT) · R5: Qwen3-4B-NT-RL-Math (external, arXiv 2602.12125)] \ #cD[R6: R1-D-7B · R7: Skywork-OR1-Math-7B · R8: JustRL-1.5B] \ #cE[R9: R1-D-1.5B · R10: R1-D-7B · R11: Skywork-OR1-Math-7B / R1-D-14B]],
  [student],
  [Qwen3-8B-Base],
  [Qwen3-8B-Base (LoRA rank 128)],
  [Qwen3-8B-Base],
  [#cA[Qwen3-8B-Base] \ #cB[Qwen3-4B-Base]],
  [#cA[Qwen3-8B] \ #cB[Qwen3-4B] \ #cC[Qwen3-14B]],
  [#cA[R1-Distill-Qwen-1.5B] \ #cB[Qwen3-1.7B-Base]],
  [#cA[Qwen3-0.6B-Base] \ #cB[Qwen3-1.7B-Base] \ #cC[Qwen3-4B-Base]],
  [#cA[R1-Distill-Qwen-1.5B] \ #cB[R1-Distill-Qwen-7B]],
  [#cA[Qwen3-1.7B-Base] \ #cB[Qwen3-1.7B-SFT (control: -Base)] \ #cC[Qwen3-1.7B (NT, instruct)] \ #cD[R1-Distill-Qwen-1.5B] \ #cE[JustRL-1.5B (R9–10); R1-Distill-Qwen-7B (R11)]],
  grp[Meta],
  [code available],
  [partial — tinker-cookbook \@ `330b73d8` (LoRA-on-Tinker path only; full-FT run was internal)],
  [yes — tinker-cookbook \@ `330b73d8` (README commands)],
  [no ("one line change" never linked)],
  [no],
  [no],
  [yes — `machine981/SCOPE` (verl fork) \ #sidenote[⚠ shipped scripts deviate from the paper's equations — see loss row]],
  [yes — `WLS04/EOPD` (verl snapshot; referenced run script missing)],
  [partial — `zui-jiang/LGR` (SLIME fork; only the 1.5B LGR-topk script; drifts: student∪teacher topk 20∪20 vs 10∪10, step cap 3000 vs 300)],
  [yes — `thunlp/OPD` (verl v0.7.0; SFT via LLaMA-Factory v0.9.5)],
  grp[Evaluation on AIME'24],
  [n_seeds],
  [#nr], [#nr], [#nr #tg[(1 run/arm inf.)]], [#nr], [#nr], [#nr], [#nr], [1 run/arm (checklist: no error bars)], [#nr #tg[(eval seed = sample index [code])]],
  [n_samples / metric],
  [#nr], [#nr],
  [avg\@32, best checkpoint],
  [Pass\@1 & Pass\@4; 16 samples/problem #tg[[inf. from score grid]]],
  [avg\@8 \ #sidenote[⚠ no numeric accuracy anywhere in the paper — curves only]],
  [Avg\@32 + Pass\@32 (repo Pass = bootstrap, 1000 resamples #tg[[code]])],
  [Avg\@8 + Pass\@8 (8 samples; Pass\@k curves to k=128)],
  [mean\@8 + pass\@8],
  [avg\@16],
  [prompting],
  [#nr], [#nr], [#nr], [#nr], [#nr],
  [question + boxed tip, single user turn, no system prompt #tg[[code]] (literal `\boxed{{}}` bug)],
  [zero-shot; "Please reason step by step, and put your final answer within `\boxed{}`" (Qwen2.5-Math pipeline)],
  [#nr #tg[(chat template [code])]],
  [JustRL-style boxed prompt #tg[[code]] — differs from training template],
  [answer extraction & grading],
  [#nr], [#nr], [#nr], [#nr], [#nr],
  [math_robust / math_verify / prime_math per benchmark #tg[[code]]],
  [Qwen2.5-Math pipeline (boxed extraction + rule grader)],
  [DeepScaleR-style boxed verify #tg[[code]]],
  [math-verify (JustRL pipeline) #tg[[code]]],
  [stop condition],
  [#nr], [#nr], [#nr], [#nr], [#nr], [EOS / max #tg[[code]]], [#nr #tg[(pipeline default)]], [#nr], [#nr],
  [temperature],
  [#nr #tg[(June-2026 rerun README: 1.0)]], [#nr #tg[(rerun: 1.0)]], [#nr], [#nr], [#nr], [0.6], [1.0], [1], [0.7],
  [top_p],
  [#nr #tg[(rerun: 1.0)]], [#nr #tg[(rerun: 1.0)]], [#nr], [#nr], [#nr], [0.95], [0.8], [1 #tg[[code]]], [0.95],
  [top_k],
  [#nr #tg[(rerun: −1)]], [#nr #tg[(rerun: −1)]], [#nr], [#nr], [#nr], [20], [#nr], [#nr], [#nr],
  [max_tokens],
  [#nr #tg[(rerun: 64000)]], [#nr #tg[(rerun: 64000)]], [#nr], [#nr], [#nr], [32768], [8192], [39k (39,936 #tg[[code]])], [31,744],
  [other benchmarks (same protocol)],
  [GPQA-D in blog table = Qwen3-report values, not TML measurements],
  [—],
  [AIME'25, HMMT'25; #cB[ToolAlpaca 68-problem split]],
  [AIME'25, HMMT'25-Feb; ToolAlpaca success rate (continual stage)],
  ["five reasoning benchmarks" named only in figure images; LCB v5 (code track)],
  [MATH500, AIME'25, AMC23, Minerva, OlympiadBench; #cB[repetition penalty 1.08 (eval-time; "severe repetition issues")] #cA[1.0]],
  [MATH500, AIME'25, AMC23, Minerva, OlympiadBench; OOD (#cB[E2]): GPQA-D, MMLU-Pro, AlpacaEval 2.0],
  [AIME-25/26, HMMT-25/26],
  [AIME'25, AMC'23],
  grp[SFT stage],
  [student (init)],
  [Qwen3-8B-Base],
  [Qwen3-8B-Base + LoRA-128],
  [Qwen3-8B-Base ("initialized by SFT")],
  [#cA[Qwen3-8B-Base] #cB[Qwen3-4B-Base]],
  [#cA[Qwen3-8B-Base (authors' anchor)] \ #cB[#cC["Qwen3-4B/14B-SFT" — provenance #nr]]],
  [— no SFT stage (#cA[released ckpt] / #cB[raw base])],
  [— no SFT stage (raw base students)],
  [— no SFT stage (released R1-Distill ckpts)],
  [#cB[R3 only: Qwen3-1.7B-Base]],
  [dataset],
  [OpenThoughts3-1.2M, "400k prompts" (= 128·3000 = 384k #tg[[code]])],
  [OpenThoughts3-1.2M, 384k prompts, shuffle buffer 384k #tg[[code]]],
  [#nr (never named)],
  ["OpenThoughts" (version/size/provenance #nr)],
  [#cA[Dolci-Think SFT]],
  [—], [—], [—],
  [#cB[200K Qwen3-4B (NT) rollouts on OpenThoughts3 math prompts (T 0.7, top-p 0.95, max 12,288; truncated/degenerate filtered); released on HF]],
  [num_steps],
  [3000 #tg[[inf.]]], [3000],
  [#nr],
  [3000 ("strong") / 1000 ("weak") sub-variants],
  [#cA[5 epochs ≈ 5375 steps #tg[[inf. from ckpt name]]]],
  [—], [—], [—],
  [#cB[1 epoch (≈3.1k steps #tg[[inf.]])]],
  [batch_size],
  [128 #tg[[code]]], [128], [#nr], [128], [#cA[256]], [—], [—], [—], [#cB[64 (8 GPU × 8)]],
  [lora_rank],
  [false (full FT)], [128], [#nr #tg[(Tinker ⇒ LoRA implied; rank unstated)]], [#nr #tg[(full FT inf.)]], [#cA[false #tg[[inf.]]]], [—], [—], [—], [#cB[false (full-param stated)]],
  [max_length],
  [16384 #tg[[code]]], [16384 #tg[[code]]], [#nr], [#nr], [#cA[#nr]], [—], [—], [—], [#cB[14,336 (cutoff)]],
  [loss mask],
  [all assistant messages #tg[[code]]], [all assistant messages #tg[[code]]], [#nr], [#nr], [#cA["Qwen3 loss mask" (entire statement)]], [—], [—], [—], [#cB[#nr]],
  [renderer / template],
  [`qwen3` #tg[[code]]], [`qwen3`], [#nr], [#nr], [#cA[#nr]], [—], [—], [—], [#cB[`qwen3` (LLaMA-Factory)]],
  [learning_rate],
  [1e-4 (README), linear → 0], [1e-3, linear → 0], [#nr], [#nr], [#cA[1e-5 cosine → 1e-6, warmup 10%]], [—], [—], [—], [#cB[1e-5 cosine, warmup 0.05]],
  [optimizer & settings],
  [Adam($beta$ 0.9/0.95, $epsilon$ 1e-8); no weight decay or grad clip (absent from blog-era SDK); 1 epoch #tg[[code]]],
  [same as T1],
  [#nr], [#nr],
  [#cA[Adam, wd 0.1, $beta$ (0.9, 0.95)]],
  [—], [—], [—],
  [#cB[BF16; ZeRO-2, FlashAttention-2, Liger #tg[[code]]]],
  [post-SFT ckpts available],
  [no],
  [yes — Tinker-hosted (Dec 2025), ranks 8/32/128, `weights/final`],
  [no], [no], [no], [—], [—], [—],
  [#cB[yes — HF `lllyx/Qwen3-1.7B-SFT`]],
  [post-SFT AIME'24],
  [60%],
  [≈55% (after 3000 steps)],
  [50.94 (avg\@32)],
  [P\@1: #cA[58.96 (3K) / 45.21 (1K)] · #cB[52.29 (3K) / 36.25 (1K)]],
  [#cA[#nr (figure only)]],
  [—], [—], [—],
  [#cB[figure-only (Fig. 8 step-0)]],
  [other post-SFT results],
  [rank-32 LoRA trails full FT by 13% after SFT (blog)],
  [—],
  [AIME'25 38.61, HMMT'25 25.52, ToolUse 24.63],
  [AIME'25/HMMT'25 per Table 1/3],
  [—], [—], [—], [—], [—],
  grp[OPD stage],
  [teacher],
  [see Identity (footnote: actual Qwen3-8B)],
  [Qwen3-8B],
  [Qwen3-8B],
  [#cA[Qwen3-32B] #cB[Qwen3-8B]; continual tool stage: #nr],
  [#cA[Qwen3-32B (sub-var. 30B-A3B)] #cB[8B/14B/32B] #cC[32B]],
  [#cA[Skywork-OR1-7B] #cB[Qwen3-8B]],
  [Qwen3-8B (non-thinking)],
  [#cA[Skywork-OR1-Math-7B] #cB[R1-D-32B]],
  [see Identity],
  [student init],
  [400k full-FT SFT checkpoint],
  [rank-128 SFT checkpoint (`load_checkpoint_path`)],
  [SFT checkpoint (data unnamed)],
  [SFT warm-up checkpoint (3K or 1K)],
  [#cA[SFT anchor `iter_0005375`] #cB[#cC["Qwen3-4B/14B-SFT"]]],
  [#cA[released ckpt] #cB[raw base]],
  [raw base],
  [released ckpts],
  [#cA[base] #cB[SFT ckpt (control: base)] #cC[instruct] #cD[#cE[released ckpts]]],
  [on-policy sync],
  [fully synchronous; sampler refreshed after *every* optimizer step (1-step lag corrected by IS ratio); `num_substeps`= 1],
  [same as T1],
  [#nr #tg[(Tinker loop implied)]],
  [#nr (on-policy implied)],
  [on-policy, student-generated; refresh cadence #nr],
  [1 update per rollout batch (256 = 256 mini), $rho equiv 1$; old logprobs recomputed on the trainer],
  [PPO-style: B=128 prompts, mini 32 ⇒ 4 grad steps per rollout iteration],
  [1 step per fresh rollout batch, staleness 0 #tg[[code]]],
  [1 update/batch, ratio $equiv$ 1 #tg[[code]]],
  [dataset],
  [DeepMath-103K (README; blog never names it)],
  [DeepMath-103K, `question` field],
  [#cA[DeepMath] · #cB[ToolAlpaca] (subset/size #nr)],
  [DeepMath (split/filtering #nr)],
  [dapo-math-17k],
  [DeepMath (split #nr; repo: `sample_deepmath_train.json`, ≤16,384-char filter)],
  [#cA[#cB[MATH (repo: MATH-lighteval train, 7.5k #tg[[inf.]])]] \ #cC["DAPO-Math-14k" = DAPO-Math-17k-Processed "en", 14,116 rows #tg[[code]]]],
  [Polaris (repo pins 53K version)],
  [DAPO-Math-17K; #cB[R3: ≈30K held-out OpenThoughts3 math prompts]],
  [num_steps],
  [≈150 (≈77K prompts)],
  [100 (README score point; full epoch ≈202 #tg[[inf.]])],
  [#cA[100] · #cB[80]],
  [90 (+40 ToolAlpaca continual, 3K-arms only)],
  [299 (endpoint at eval plateau)],
  [#nr (repo: 10 epochs)],
  [#cA[#cB[3 epochs ≈176 iters #tg[[inf.]]]] #cC[2 epochs ≈220 #tg[[inf.]]]],
  [300],
  [1 epoch ≈270 #tg[[inf.]] (length-sweep arms: 200)],
  [data ordering],
  [sequential, no shuffle #tg[[code]]],
  [sequential, no shuffle #tg[[code]]],
  [#nr], [#nr], [#nr],
  [#nr #tg[(verl shuffle default inf.)]],
  [#nr #tg[(repo shuffle: true)]],
  [shuffled #tg[[code]]],
  [sequential (`shuffle=False`) #tg[[code]]],
  [rollouts_per_group],
  [4], [4],
  [#nr (#cB[256 rollouts/step total])],
  [#nr],
  [8],
  [1 (OPD baseline; SCOPE arm: 8)],
  [1 (GRPO arm: 8)],
  [1],
  [4],
  [groups_per_batch],
  [≈512 #tg[[arith.: 77K/150; README: 512]]],
  [512],
  [#cA[64 "groups"] · #cB[#nr]],
  [512 ("batch size 512"; prompts vs sequences ambiguous)],
  [300 prompts/step; "global batch size 64" (tension unexplained)],
  [256],
  [128 (mini 32)],
  [128],
  [64],
  [prompt construction],
  [no system prompt; raw question; truncated to 1024 tok #tg[[code]]],
  [same as T1],
  [#nr], [#nr], [#nr],
  [question + boxed tip, no system prompt #tg[[code]]],
  [single user msg + boxed instruction #tg[[code]]; teacher template pre-fills empty `<think>`],
  [chat template; max prompt 1024 #tg[[code]]],
  [DAPO template ("Solve the following math problem step by step…Answer:"); max prompt 1024; NT students need `enable_thinking=False`],
  [rollout temperature],
  [1.0 #tg[[SDK default — nothing plumbed]]],
  [1.0 #tg[[SDK default]]],
  [#nr], [#nr],
  [1.0],
  [0.6],
  [1.0],
  [1.0 (student *and* teacher)],
  [1.0],
  [rollout top_p],
  [1.0 #tg[[SDK]]], [1.0 #tg[[SDK]]], [#nr], [#nr], [#nr], [#nr #tg[(verl 1.0 inf.)]], [1.0], [#nr #tg[(1.0 inf.)]], [1.0],
  [rollout top_k],
  [−1 #tg[[SDK]]], [−1 #tg[[SDK]]], [#nr], [#nr], [#nr], [#nr #tg[(−1 inf.)]], [#nr #tg[(−1 inf.)]], [#nr], [#nr #tg[(−1 inf.)]],
  [rollout max_tokens],
  [4096 #tg[[code]]], [4096 #tg[[code]]], [#nr], [#nr], [#nr],
  [12,288 (prompt cap 4096; ⚠ repo scripts 2048)],
  [4096],
  [9k main arm (sweep 3k/9k/16k/39k)],
  [7168],
  [rollout stop condition],
  [renderer stop sequence \ #sidenote[⚠ blog-day default resolves Qwen3-8B-Base → `role_colon` ("\n\nUser:"); intended `qwen3` (`<|im_end|>`), fixed later on main]],
  [same caveat as T1],
  [#nr], [#nr], [#nr], [EOS/length #tg[[inf.]]], [#nr], [#nr], [EOS #tg[[inf.]]],
  [framework],
  [Tinker (server-side loss kernel)],
  [Tinker],
  [Tinker #tg[[implied, not stated]]],
  [#nr],
  [#nr #tg[(Megatron-style ckpt names; sglang teacher serving)]],
  [verl (FSDP + vLLM) #tg[[code]]],
  [verl (KD arm: DistillKit)],
  [SLIME + SGLang (Megatron backend)],
  [verl v0.7.0 #tg[[code]]],
  [learning_rate],
  [5e-5, constant (README full-FT)],
  [1e-4, constant],
  [#nr],
  [1e-5],
  [1e-6, constant],
  [5e-5, constant; wd 0.01],
  [3e-6, cosine],
  [#cA[1e-5] · #cB[3e-6]; warmup 0],
  [1e-6, constant, no warmup #tg[[code]]],
  [lora_rank],
  [false], [128],
  [#nr #tg[(LoRA implied via Tinker)]],
  [#nr #tg[(full inf.)]],
  [false #tg[[inf.]]],
  [false (FSDP full FT)],
  [false #tg[[inf.]]],
  [false #tg[[inf.]]],
  [false (full-param FSDP)],
  [advantage],
  [$A_t = -"sg"[log q_"samp" (y_t) - log p_T (y_t)]$; coef 1.0; discount 0 \ #sidenote[env rewards ≡ 0; group-centering annihilates them]],
  [same as T1],
  [$A_t = "sg"[ln u_t]$, $u_t = q_T (y_t) slash p_theta (y_t)$; $gamma = 0$ \ #sidenote[OPD+ arm: $ln u_t - 1$ (Table 1; §3.2 prose sign-flipped)]],
  [$A_t = "sg"[log p_T (y_t) - log p_S (y_t)]$],
  [$A_t = -(log pi_theta (y_t) - log pi_T (y_t))$, sampled-token \ #sidenote[Table 4 says "GRPO estimator"; App. G.2 formula is the raw gap, normalization ambiguous]],
  [$"sg"[log pi_theta (y_t) - log pi_T (y_t)]$ as negative advantage, wrong-answer branch only \ #sidenote[⚠ repo instead ships −k3: $-"clamp"(e^Delta - Delta - 1, plus.minus 10)$, $Delta = log pi_T - log pi_"old"$]],
  [$hat(A)_t = log pi_"te" (x_t) - log pi_(theta_"old") (x_t)$],
  [$A_t = 1 + log pi_theta (x_t) - log pi_T (x_t)$ \ #sidenote[note the +1]],
  [per $(t, v in "student top-16")$: $-"sg"[log p_t (v) - log q_t (v)] dot macron(p)_t (v)$ #tg[[code]]],
  [loss],
  [$L = -sum_t exp(log q_theta (y_t) - log q_"samp" (y_t)) dot A_t$ — unclipped IS, server-side kernel],
  [same as T1],
  [PPO-like IS surrogate $-bb(E) sum_n rho_n (theta) A_n$, $rho_n = p_theta slash p_(theta_0)$; no clipping],
  [$L = -bb(E)[(1 slash |y|) sum_t A_t log p_S (y_t)]$ — length-normalized; no IS ratio, no clipping \ #sidenote[AOPD arm: tokens with $A_t <= 0$ get truncated FKL on teacher top-K = 32]],
  [policy gradient on $A_t$; clip $epsilon$ #nr (off-policy arm: $epsilon$=1, inactive)],
  [$sum_t rho_(i,t) (log pi_macron(theta) - log pi_T)$ (wrong) + PPL-weighted $-sum_t rho_(i,t)$ MLE (correct); no clipping in paper],
  [PPO-clipped: $max(-r_t hat(A)_t, -"clip"(r_t, 1 plus.minus epsilon) hat(A)_t)$; $epsilon$ #nr #tg[(verl 0.2 inf.)] \ #sidenote[EOPD arm: + $alpha bb(1)[H^"te" > 0.8] dot$ FKL on teacher top-16, $alpha$=1]],
  [$nabla = sum_t nabla log pi_theta dot A_t$ (stop-grad RKL) \ #sidenote[topk arm: student ∪ teacher top-10, renormalized (⚠ repo: 20/20 + PPO clip)]],
  [student-top-16 renormalized RKL: $bb(E) sum_t D_"KL" (macron(p)_t^(S_t) parallel macron(q)_t^(S_t))$, realized as PG with ratio ≡ 1 #tg[[code]] \ #sidenote[sampled-token TML estimator = §6.3 ablation arm only]],
  [loss mask],
  [sampled completion tokens],
  [sampled completion tokens],
  [#nr], [#nr],
  [response tokens #tg[[inf.]]],
  [response tokens; token-mean #tg[[code]]],
  [response tokens; token-mean],
  [#nr #tg[(response inf.)]],
  [response tokens; token-mean],
  [optimizer & settings],
  [Adam($beta$ 0.9/0.95, $epsilon$ 1e-8); no wd, no clip #tg[[code+SDK]]],
  [same as T1],
  [#nr], [#nr],
  [Adam, wd 0.1, $beta$ (0.9, 0.98); $epsilon$ #nr],
  [#nr #tg[(repo AdamW (0.9, 0.999), warmup 0)]],
  [AdamW ($beta$/wd #nr)],
  [AdamW $beta$ (0.9, 0.98), wd 0.1, warmup 0],
  [AdamW #tg[(verl defaults inf.)]],
  [post-OPD ckpts available],
  [no],
  [yes — Tinker-hosted (Dec 2025), ranks 8/32/128, `sampler_weights/000080` (step 80)],
  [no], [no], [no],
  [SCOPE arms only (HF); no OPD/GRPO/KD arms],
  [no],
  [no],
  [no student ckpts (teacher GRPO ckpt + SFT corpus released)],
  [post-OPD AIME'24],
  [70% (≈150 steps)],
  [≈65% (100 steps)],
  [#cA[OPD 64.27 → OPD+ 64.69 (avg\@32)]],
  [OPD: #cA[61.46 (3K) / 43.13 (1K, *below* SFT)] · #cB[57.08 / 46.25]; AOPD: #cA[66.87 / 53.75] · #cB[61.04 / 51.25] (P\@1)],
  [#nr — figures only],
  [(Avg\@32) #cA[base 29.4 → OPD 40.2 → SCOPE 42.7] \ #cB[base 3.7 → OPD 12.2 → SCOPE 13.3]],
  [(Avg\@8) OPD: #cA[2.50] #cB[8.33] #cC[18.33]; EOPD: #cA[4.17] #cB[10.42] #cC[17.92]],
  [(mean\@8) #cA[base 24.17 → OPD(9k) 45.83 → LGR 46.67] \ #cB[base 52.08 → OPD(9k) 62.08 → LGR 61.67]],
  [figure-only. Text: #cD[R8 recovers >80% of teacher gap; R6 "no improvement"]; #cE[R9/R10 regress to pre-RL level]],
  [other post-OPD results],
  [9–30× compute vs extrapolated SFT; ≈18× GPU-hours (blog)],
  [—],
  [#cB[ToolUse: OPD 59.97 → OPD+ 61.44; FKL/JSD OPD collapse to 0.00]],
  [tool-use success after continual stage: OPD 65.0, AOPD 75.0, GKD 76.0 (#cA[3K arms])],
  [bf16 update sparsity (Table 1): #cA[51.6% (MoE-teacher 48.6%)] #cB[50.3/51.1/51.7%] #cC[56.6%]],
  [6-bench avg: #cA[OPD 52.3 → SCOPE 55.2] #cB[OPD 30.6 → SCOPE 32.5]],
  [6-bench deltas EOPD−OPD: +1.16/+0.99/+1.80 Avg\@8],
  [AIME-25/26, HMMT-25/26 rows in Table 1; length sweep Table 2],
  [teacher-continuation probe: teacher advantage +0.37 \@1K prefix → +0.02 \@16K],
)
]

Reading the table columnwise, the recipe's reproducibility rests on three uneven legs. The *algorithm* is the well-specified part: every column implements a sampled-token student−teacher log-ratio as a detached per-token signal, and the differences (the +1 baseline in LGR, OPD+'s $ln u - 1$ correction, EOPD's PPO clip, Rethinking's student-top-16 support, AOPD's length normalization, the presence or absence of an importance ratio) are all visible in the loss row. The *pipeline* is bimodal: TML, OPD+, AOPD, Geometry (G1), and Rethinking's R3 run a true SFT→OPD staging, while every other setting outsources or skips the SFT stage — and where the SFT stage exists, it is the worst-documented part (OPD+ never names its SFT data; Geometry's 4B/14B anchors have no stated provenance). The *evaluation* is the weakest leg: only SCOPE, EOPD, LGR, and Rethinking publish a complete sampling protocol, no work reports eval seeds, and the two Tinker-defaults reruns (T2, OPD+ B1) land at 64–65% AIME'24 while the blog-day headline's 70% rests on an unpublished protocol and a teacher that, per the blog's own footnote, was not the 32B model the headline advertises.

= Criticism and alternatives <sec-criticism>

*Criticism of the divergence itself.* (i) Mode/coverage loss: acknowledged by MiniLLM, quantified by f-DISTILL and GKD, measured in alignment by Kirk, f-DPG, f-DPO, BOND (@sec-empirical), and localized to high-entropy teacher positions at 2026 scale @jin2026entropy. (ii) The mode-seeking justification may be vacuous at LLM scale: softmax students share the FKL/RKL optimum; observed differences are early-training head-versus-tail dynamics @wu2025rethinking. (iii) Over-concentration: ABKD frames RKL as excessive "confidence-concentration" @wang2025abkd, and DRKL reports RKL's non-target gradients "push the target logit upward even when the student already matches the teacher" @luong2026drkl. (iv) In its regularizer role, the penalty does not improve the gold-reward frontier @gao2023scaling and its coefficient does not trade back diversity @kirk2024understanding.

*Criticism of estimators and implementations — a separate matter.* The naive sequence-level objective admits degenerate optima requiring patches (MiniLLM's three techniques); pure on-policy data yields unreliable teacher signals early (SKD) and off-policy reuse breaks vanilla directions (DistiLLM); the k3-as-loss pattern optimizes the wrong direction in expectation (@sec-audit-notes); token-level losses drop the visitation term (Tang and Munos) while exact-KL-in-reward drops the local term (@sec-audit-notes); off-policy updates need importance-weight corrections that GRPO's KL term lacks (RPG); clamps censor exactly the zero-forcing tail; the 2026 OPD studies add data-level failure modes — the teacher's corrective signal fades on long on-policy prefixes @liu2026teacher, teacher–student thinking-pattern incompatibility bounds what distillation can transfer @li2026rethinking, and the agreement signal itself is gameable by student-manufactured contexts: repetition-induced degenerate agreement @yu2026mismatch and low-KL agreement traps @xin2026agreementtrap (@sec-folklore-crit). Nemotron 3 Ultra's failure report (§@sec-reports) belongs here, not with the divergence-level criticism: the reverse KL stayed, the full-distribution estimator went.

*Divergences introduced to fix FKL/RKL shortcomings.* The interpolation move recurs at every scale:

- *Minka's $alpha$-family* — "The divergences with $0 < alpha < 1$ are a blend of these extremes" @minka2005divergence.
- *Generalized JS* — Huszár's $"JS"_pi$ interpolation @huszar2015how.
- *JSD($beta$)* — GKD's forward-to-reverse knob @agarwal2024gkd.
- *JS and TVD* — f-DISTILL's symmetric losses ("symmetric … consistently better" @wen2023fdistill).
- *Skew KL/RKL* — DistiLLM's mixtures, for gradient stability rather than mode behavior @ko2024distillm.
- *AKL* — adaptive head/tail weighting @wu2025rethinking.
- *$alpha$-$beta$ family* — ABKD @wang2025abkd.
- *Jeffreys divergence* — BOND, chosen with the both-failure-modes argument quoted in @sec-folklore-llm ("the Jeffreys divergence inherits the best of both divergences" @sessa2024bond).
- *f-divergence constraint families* — f-DPO and f-PO @wang2024beyond @han2025fpo.
- *Forward-KL distribution matching* — the mirror-image fix (GDC/DPG @khalifa2021distributional, f-DPG's KL-DPG — itself penalized by high-variance importance weights @go2023aligning). A unified theory of f-divergence-regularized RLHF is emerging @wu2026fdivrlhf. Notably, DistiLLM-2's result that *same-data* interpolation does not help — only direction-matched-to-data pairing does @ko2025distillm2 — suggests the useful axis is often data-conditional rather than a global $beta$.

= Synthesis <sec-synthesis>

*On the folklore (Q1–Q2).* The when-to-use-which story is theoretically grounded but rests on capacity limitation and global optimization, assumptions that the LLM setting often violates; its standard figures all depict a unimodal family fit to a mixture. Empirical support at LLM scale is real but narrower than the folklore: reverse-KL-family objectives win on open-ended generation quality under sampling, with measurable diversity costs; the differences shrink or vanish with student capacity, greedy decoding, or training time. The strongest criticisms relocate mode collapse from the objective to optimization dynamics and misspecification.

*On usage (Q3, Q5).* Reverse KL earns its keep in exactly two production roles: the universal-but-small RLHF/RLVR regularizer (where it is the *tractable* direction, and where the variational view makes it principled), and on-policy post-training distillation (where it doubles as a dense, unhackable reward and an order-of-magnitude-cheaper RL substitute — by 2026 the standard frontier consolidation stage: DeepSeek-V4, Kimi K3, GLM-5, Nemotron). The regularizer role, meanwhile, has become contested: one prominent RLVR line removes the term outright while another keeps and repairs its estimator (@sec-rl). No production pretraining-scale distillation uses it, and the two clean production negatives bracket the regime: full-vocabulary reverse KL failed both off-policy at scale (Minitron) and on-policy at the estimator level (Nemotron 3 Ultra §3.3.5).

*On implementations (Q6).* The audit's central finding is that the field's stated objectives and computed gradients frequently diverge, in patterned ways: value-unbiasedness is routinely mistaken for gradient-correctness (k3-as-loss, expected gradient in the *forward* direction — now being repaired in TRL and nemo-rl; trl-classic's exact-KL-in-reward `"full"` option, expected gradient $G_"visit"$-only); token-level estimators silently optimize frozen-visitation objectives; support truncations, renormalizations, and clamps change which divergence is estimated; and several reverse-KL code paths (torchtune, NeMo-Aligner `bwd_kl`, modelopt `reverse`) exist without any shipped experiment using them. The successful production estimator — sampled-token log-ratio as a per-token advantage — is best understood not as "reverse KL, approximated" but as a deliberately partial, per-token surrogate whose robustness (only realized actions are supervised; off-support teacher noise is never queried) is plausibly *why* it beat the "more correct" full-distribution estimator.

*Open questions.* When does the visitation-gradient term matter in practice (discount $0$ versus $1$ in on-policy distillation)? Is the Nemotron full-distribution failure about teacher miscalibration off-support (their hypothesis) or about the gradient structure of full-vocabulary RKL on student prefixes? And does the direction question survive controlled comparisons at production scale — on-policy *forward* KL at sampled tokens being the natural missing baseline in nearly every comparison reviewed here?

#v(4mm)
#line(length: 100%, stroke: 0.5pt + gray)
#text(size: 8.5pt)[Raw dossiers with full verbatim quotes, per-source BibTeX, and additional audited code paths: `lit_review/notes/A–F.md`, plus per-paper notes (`notes/per-paper/`) and the unincorporated-papers dossier (`G-unincorporated.md`). Quotes were verified against primary sources; the Schulman blog was reachable only via a rendering proxy and its formulas were cross-checked against secondary sources. MiniLLM quotes follow the current arXiv v6 and may differ slightly from the ICLR 2024 camera-ready. The original figures reproduced in @sec-figures are © their respective rightsholders and are included in this unpublished, personal-use document for private study and commentary only.]

#bibliography("references.bib", style: "harvard-cite-them-right")
