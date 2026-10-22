# Agent A result: forward vs reverse KL — theory and folklore

**Convention: reverse KL = D_KL(q‖p), q = trained model, p = target/teacher. Forward KL = D_KL(p‖q).** [PDF] = verified verbatim from locally extracted PDFs; [fetch] = via page fetch. Local copies for figure recreation in scratchpad root: `minka2005.txt`, `prml.pdf`, `book1.pdf` (PML1), `book2.pdf` (PML2), `mackay.pdf`, `turner_sahani.pdf`.

## 1. Canonical theory sources

### Minka 2005 — Divergence measures and message passing
```bibtex
@techreport{minka2005divergence,
  author = {Thomas Minka},
  title = {Divergence Measures and Message Passing},
  institution = {Microsoft Research}, number = {MSR-TR-2005-173}, year = {2005}}
```
Link: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2005-173.pdf

Canonical zero-forcing/zero-avoiding source. α-divergence family (Zhu–Rohwer convention, eq. 2): **lim α→0 = KL(q‖p), lim α→1 = KL(p‖q)** (eqs. 4, 6). Theoretical (global minima over restricted family).
Quotes [PDF]:
- "These divergences are zero-forcing because p(x) = 0 forces q(x) = 0. In other words, they avoid 'false positives'… Zero-forcing emphasizes modeling the tails, rather than the bulk of the distribution, which tends to underestimate the variance of p." (§2, α ≤ 0)
- "Following the terminology of Frey et al. (2000), these divergences are inclusive (α < 1 are exclusive). Inclusive divergences require q > 0 whenever p > 0, thus avoiding 'false negatives.'" (§2)
- "When α is a large negative number, the best approximation represents only one mode, the one with largest mass (not the mode which is highest)." (§2) — mode-seeking ≠ seeking the highest mode.
- Figure 1 caption: "The Gaussian q which minimizes α-divergence to p (a mixture of two Gaussians), for varying α. α → −∞ prefers matching one mode, while α → ∞ prefers covering the entire distribution." (α=0 reverse KL, α=1 forward KL). Figure 3: "The structure of α-divergences." Mean-field = α=0, BP/EP = α=1.
- Capacity nuance [PDF]: "if p has 100 identical Gaussians in a row, forming a plateau, the optimal q is only as wide as one of them."
Flag: exclusive = our reverse, inclusive = our forward; his α convention differs from Bishop's.

### Bishop 2006 — PRML §10.1.2
```bibtex
@book{bishop2006prml,
  author = {Christopher M. Bishop},
  title = {Pattern Recognition and Machine Learning}, publisher = {Springer}, year = {2006}}
```
Link (free official PDF): https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf

**Figure 10.3** (p. 469): (a) forward KL covering a bimodal p with single Gaussian; (b),(c) reverse KL locking onto one mode — **two distinct local minima** shown. **Figure 10.2** (p. 468): factorized q on correlated 2-D Gaussian, KL(q‖p) too-compact vs KL(p‖q) too-broad.
Quotes [PDF], §10.1.2:
- "there is a large positive contribution to the Kullback-Leibler divergence … from regions of Z space in which p(Z) is near zero unless q(Z) is also close to zero. Thus minimizing this form of KL divergence leads to distributions q(Z) that avoid regions in which p(Z) is small."
- "a variational treatment based on the minimization of KL(q∥p) will tend to find one of these modes. By contrast, if we were to minimize KL(p∥q), the resulting approximations would average across all of the modes and, in the context of the mixture model, would lead to poor predictive distributions (because the average of two good parameter values is typically itself not a good parameter value)."
- α family (eq. 10.19): "for α ⩽ −1 the divergence is zero forcing … will tend to seek the mode with the largest mass. Conversely for α ⩾ 1 the divergence is zero-avoiding … will over-estimate the support of p(x)."
**TERMINOLOGY FLAG: Bishop calls KL(p‖q) "the reverse Kullback-Leibler divergence"** — opposite of our convention (his default is the VI objective KL(q‖p)): "suppose instead that we had been minimizing the reverse Kullback-Leibler divergence KL(p∥q) … used in … expectation propagation" (p. 467) [PDF]. His α: KL(p‖q) at α→+1, KL(q‖p) at α→−1 (differs from Minka's 0/1).

### Murphy 2022 — PML: An Introduction, §6.2.6
```bibtex
@book{murphy2022pml1,
  author = {Kevin P. Murphy},
  title = {Probabilistic Machine Learning: An Introduction}, publisher = {MIT Press}, year = {2022}}
```
Link: https://probml.github.io/pml-book/book1.html
§6.2.6 "Forward vs reverse KL" (pp. 216–217 in 2025-04-18 online version). Matches our convention: forward = inclusive = M-projection; reverse = exclusive = I-projection. Fig 6.3 = redrawn Bishop 10.3 ("KLfwdReverseMixGauss.ipynb"). §6.2.5: MLE = forward KL to empirical distribution. Exercise 6.4 (from MacKay Ex. 33.7): reverse KL with **three distinct minima**.
Quotes [PDF]: "minimizing the KL will force q to include all the areas of space for which p has non-zero probability. Put another way, q will be zero-avoiding or mode-covering, and will typically over-estimate the support of p"; "One way to do this is for q to put probability mass in very few parts of space; this is called zero-forcing or mode-seeking behavior. In this case, q will typically under-estimate the support of p".

### Murphy 2023 — PML: Advanced Topics, §5.1.4
```bibtex
@book{murphy2023pml2,
  author = {Kevin P. Murphy},
  title = {Probabilistic Machine Learning: Advanced Topics}, publisher = {MIT Press}, year = {2023}}
```
Link: https://probml.github.io/pml-book/book2.html
§5.1.4.1 "Forwards vs reverse KL" (pp. 225–227 online 2025-12-10); §5.1.4.2 moment projection proof (exp-family q); §5.1.4.3 information projection (diag-Gaussian gets m=µ, v_i=Λ_ii^{-1}, overconfident). Fig 5.1 (1-D bimodal, both fits), Fig 5.2 (Bishop 10.2 redrawn). **Carries the folklore caveat [PDF]: "Note, however, that minimizing the reverse KL does not always result in an overly compact approximation, as explained in [Tur+08]."** ([Tur+08] = Turner, Berkes, Sahani, MacKay counter-examples report.) Links Ari Seff's animation (https://twitter.com/ari_seff/status/1303741288911638530).
Quote [PDF]: "q tends to cover both modes as it must be nonvanishing everywhere p is; this is called mode-covering or zero-avoiding behavior … to prevent DKL(q∥p) from becoming infinite, we must have q = 0 whenever p = 0, which creates mode-seeking or zero-forcing behavior" (§5.1.4.1).

### MacKay 2003 — ITILA, Ch. 33
```bibtex
@book{mackay2003itila,
  author = {David J. C. MacKay},
  title = {Information Theory, Inference, and Learning Algorithms},
  publisher = {Cambridge University Press}, year = {2003}}
```
Link: https://www.inference.org.uk/itprnn/book.pdf
§33.1: variational free energy minimization = minimizing D_KL(Q‖P) (eq. 33.9: "βF̃(θ) = D_KL(Q||P) + βF") — **why VI uses reverse KL** (needs expectations only under Q). Ex. 33.5 (p. 434): "This exercise explores the assertion, made above, that the approximations given by variational free energy minimization always tend to be more compact than the true distribution." And: "[Note that in general it is not possible to evaluate the objective function G, because integrals under the true distribution P(x) are usually intractable.]" — tractability reason stated inline. Ex. 33.7 = three-reverse-KL-minima example (reused by Murphy).
Flag: MacKay's D_KL(Q‖P), Q = approximation — same direction as our reverse.

### Blei, Kucukelbir, McAuliffe 2017 — VI review
```bibtex
@article{blei2017variational,
  author = {David M. Blei and Alp Kucukelbir and Jon D. McAuliffe},
  title = {Variational Inference: A Review for Statisticians},
  journal = {Journal of the American Statistical Association},
  volume = {112}, number = {518}, pages = {859--877}, year = {2017},
  eprint = {1601.00670}, archivePrefix = {arXiv}}
```
KL(q‖p) chosen for tractability (ELBO needs expectations under q only). Quote [fetch]: "variational inference generally underestimates the variance of the posterior density; this is a consequence of its objective function." Fig 1 = mean-field Gaussian on correlated 2-D Gaussian. Convention matches ours.

### Turner & Sahani 2011 — "folk-theorem" language and limits
```bibtex
@incollection{turner2011two,
  author = {Richard E. Turner and Maneesh Sahani},
  title = {Two Problems with Variational Expectation Maximisation for Time-Series Models},
  booktitle = {Bayesian Time Series Models}, editor = {David Barber and A. Taylan Cemgil and Silvia Chiappa},
  publisher = {Cambridge University Press}, pages = {104--124}, year = {2011}}
```
Link: https://www.gatsby.ucl.ac.uk/~maneesh/papers/turner-sahani-2010-ildn.pdf
Quotes [PDF]: "It is well known that variational approximations, like those derived in the vEM E-Step, tend to be compact (MacKay, 2003). In other words, the variational approximation has a tendency to have a smaller entropy than the true distribution. The evidence for this folk-theorem is reviewed in the next section" (§1.2–1.3); "the uncertainty may appear to be smallest precisely when the approximation is poorest." (§1.1). Bibliography cites: Turner, Berkes, Sahani, MacKay (2008), "Counter-examples to variational free-energy compactness folk theorems," UCL tech report — direct anti-folklore reference (not located online).

## 2. Folklore articulations for generative models

### Huszár 2015
```bibtex
@article{huszar2015how,
  author = {Ferenc Husz{\'a}r},
  title = {How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?},
  journal = {arXiv preprint}, year = {2015}, eprint = {1511.05101}, archivePrefix = {arXiv}}
```
P = data, Q = model (matches ours). MLE (= forward KL) wrong for realistic samples; ideal perceptual objective closer to reverse KL; generalized JS_π interpolates; adversarial ≈ π=0.5.
Quotes [fetch]: "Minimising KL[P∥Q] often leads to models that overgeneralise, and sometimes produce samples that are very unlikely under P" (§5.0.2); "If we train a generative model by minimising KL[Q∥P], the model will very conservatively try to avoid any behaviour that is unlikely under P" (§5.0.2); "minimising JSπ divergence for a range of π∈(0,1) allows us to interpolate between the behaviour of KL[P∥Q] and KL[Q∥P]" (§6). Figure 1: JS_π sweep π ∈ {0.1, 0.5, 0.99}.
(Cite sections by title — "The Diagnosis" / "Generalised Adversarial Training" — ar5iv auto-numbering.)

### Theis, van den Oord, Bethge 2016
```bibtex
@inproceedings{theis2016note,
  author = {Lucas Theis and A{\"a}ron van den Oord and Matthias Bethge},
  title = {A Note on the Evaluation of Generative Models},
  booktitle = {ICLR}, year = {2016}, eprint = {1511.01844}, archivePrefix = {arXiv}}
```
Standard "likelihood and sample quality decouple" citation. Quotes [fetch]: Fig 1 caption: "An isotropic Gaussian distribution was fit to data drawn from a mixture of Gaussians by either minimizing Kullback-Leibler divergence (KLD), maximum mean discrepancy (MMD), or Jensen-Shannon divergence (JSD)."; "Minimizing MMD or JSD yields a Gaussian which fits one mode well, but which ignores other parts of the data. On the other hand, maximizing average log-likelihood or equivalently minimizing Kullback-Leibler divergence (KLD) avoids assigning extremely small probability to any data point but assigns a lot of probability mass to non-data regions." (§2); "Log-likelihood and visual appearance of samples are therefore largely independent." (§3.2)
Flag: "KLD" = forward direction (MLE).

### Goodfellow 2016 — NIPS GAN tutorial (folklore + pushback)
```bibtex
@article{goodfellow2016nips,
  author = {Ian Goodfellow},
  title = {NIPS 2016 Tutorial: Generative Adversarial Networks},
  journal = {arXiv preprint}, year = {2017}, eprint = {1701.00160}, archivePrefix = {arXiv}}
```
§3.2.5 states the folklore ("many people (including the author) believed that GANs produced sharp, realistic samples because they minimize the Jensen-Shannon divergence while VAEs produce blurry samples because they minimize the KL divergence" [fetch]) then knocks it down. Fig 14 = two-direction KL on 1-D GMM ("The two directions of the KL divergence are not equivalent. The differences are most obvious when the model has too little capacity to fit the data distribution…"). Fig 15: max-likelihood f-GANs still sharp/few-mode.
Key quote [fetch]: "GANs often choose to generate from very few modes; fewer than the limitation imposed by the model capacity. The reverse KL prefers to generate from as many modes of the data distribution as the model is able to; it does not prefer fewer modes in general. This suggests that mode collapse is driven by a factor other than the choice of divergence." (§3.2.5); "…we can conclude that the use of the Jensen-Shannon divergence is not a particular important distinguishing characteristic of GANs."

### Arjovsky & Bottou 2017
```bibtex
@inproceedings{arjovsky2017towards,
  author = {Martin Arjovsky and L{\'e}on Bottou},
  title = {Towards Principled Methods for Training Generative Adversarial Networks},
  booktitle = {ICLR}, year = {2017}, eprint = {1701.04862}, archivePrefix = {arXiv}}
```
Theorem 2.5: −log D generator gradient = ∇[KL(P_g‖P_r) − 2 JSD(P_g‖P_r)] at optimal discriminator — the popular GAN loss secretly reverse-KL-dominated. Quote [fetch]: "this KL assigns an extremely high cost to generating fake looking samples, and an extremely low cost on mode dropping" (§2.2.2). KL(P_g‖P_r), model first — matches our reverse.

### Blog posts (folklore canon)
- **Dibya Ghosh, "KL Divergence for Machine Learning"** — https://dibyaghosh.com/blog/probability/kldivergence/ (~2018, undated). Forward D_KL(P‖Q_θ) "mean-seeking", reverse D_KL(Q_θ‖P) "mode-seeking"; choice framed by access: forward needs "a dataset of samples from the true model P(X)", reverse needs ability to "evaluate probabilities of data-points under the true model P(X)" [fetch]. Matches our convention.
- **Agustinus Kristiadi, "KL Divergence: Forward vs Reverse?"** (2016-12-21) — https://agustinus.kristia.de/blog/forward-reverse-kl/ . "Forward KL is known as zero avoiding, as it is avoiding whenever p(x)=0"; reverse "is know as zero forcing, as it forces q to be on some areas" [fetch, sic]. Matches ours.
- **Eric Jang, "Tips for Training Likelihood Models"** (2019-07-05) — https://blog.evjang.com/2019/07/likelihood-model-tips.html . "minimizing forward KL D_KL(p||q) will learn mode-covering behavior while minimizing reverse KL D_KL(q||p) results in mode-dropping behavior" [fetch]. Matches ours. (No dedicated Jang KL post; defers to Ghosh.)
- **Andy Jones, "KL(q‖p) is mode-seeking"** — https://andrewcharlesjones.github.io/journal/klqp.html . Clean folklore statement + 2-D GMM animation; **no** capacity/local-optima caveats [fetch]. Matches ours.
- **John Schulman, "Approximating KL Divergence"** (2020-03-07) — http://joschu.net/blog/kl-approx.html . k1/k2/k3 for KL[q,p] with x ~ q, r = p(x)/q(x): **k1 = −log r; k2 = ½(log r)²; k3 = (r − 1) − log r**. k1 unbiased, "has high-variance, as it's negative for half of the samples, whereas KL is always positive"; k2 low-variance biased (~0.2% bias at KL=0.005); k3 unbiased, lower variance via control variate r−1 [fetch via proxy — joschu.net refused direct connections; re-fetch before verbatim quoting]. Flag: "KL[q, p]" first argument = sampled distribution.
- **janus, "Mysteries of mode collapse"** (LessWrong, 2022-11-08) — https://www.lesswrong.com/posts/t9svvNPNmFf5Qa3TA/mysteries-of-mode-collapse . LLM mode collapse = "drastic biases toward particular completions and patterns", term borrowed "after a common problem that plagues GANs"; extreme token confidence ("frequently above 99%", the " 97" attractor) [fetch]. **Carries in-post correction: "I have received evidence from multiple credible sources that text-davinci-002 was not trained with RLHF"** (it was FeedME SFT) — post's own RLHF attribution was wrong; main text does not attribute the phenomenon to a KL objective. (Original title "Mysteries of mode collapse due to RLHF"; retitled.)

## 3. LLM-distillation illustrations (theory-level)
- **MiniLLM Fig 2** caption: "The toy experiment. We fit a Gaussian mixture with a Gaussian distribution using forward KLD and reverse KLD." [fetch] Quotes: "minimizing KL[q_θ||p] causes q_θ to seek the major modes of p, and assign low probabilities to p's void regions" (§2.1).
- **GKD Fig A.16** (bimodal P vs unimodal Q_θ, both directions); generalized JSD(β) as forward↔reverse interpolation. Quotes [fetch]: forward KL "requires the student to cover the entire support of the teacher"; "mode-seeking divergences, such as reverse KL, prioritize the tokens where the teacher assigns high probability … at the expense of less diverse generations" (§3.1); "reverse KL performs much better than forward KL" on instruction tuning (§4.4) but divergence choice is task/sampling dependent.

## 4. Criticisms and complications

### Wu et al. — Rethinking KL in KD (strongest direct pushback)
```bibtex
@inproceedings{wu2025rethinking,
  author = {Taiqiang Wu and Chaofan Tao and Jiahao Wang and Runming Yang and Zhe Zhao and Ngai Wong},
  title = {Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models},
  booktitle = {COLING}, year = {2025}, eprint = {2404.02657}, archivePrefix = {arXiv}}
```
"neither mode-seeking nor mean-seeking properties manifest in KD for LLMs" [fetch, §3.2]: folklore assumptions (Gaussian/continuous q, limited capacity) violated for softmax discrete distributions; FKL and RKL share global optimum; transient difference: "RKL focuses on the tail part of the distributions, while FKL focuses on the head part at the beginning epochs" [fetch]. Proposes AKL. Notation matches ours.

### f-GAN (Nowozin, Cseke, Tomioka 2016)
```bibtex
@inproceedings{nowozin2016fgan,
  author = {Sebastian Nowozin and Botond Cseke and Ryota Tomioka},
  title = {f-{GAN}: Training Generative Neural Samplers using Variational Divergence Minimization},
  booktitle = {NeurIPS}, year = {2016}, eprint = {1606.00709}, archivePrefix = {arXiv}}
```
§2.5: single Gaussian fit to GMM under KL, reverse KL, JS, Hellinger, Pearson/Neyman χ², Jeffreys, TV (Table 3: KL μ=1.03 σ=1.82 vs reverse KL μ=1.56 σ=1.64). Quote [fetch]: "when the generative model is misspecified and does not contain the true distribution, the divergence function used for estimation has a strong influence on which model is learned." Implicitly: well-specified ⇒ all f-divergences share the optimum.

### Li & Farnia 2023 — rigorous mode-seeking definition
```bibtex
@inproceedings{li2023modeseeking,
  author = {Cheuk Ting Li and Farzan Farnia},
  title = {Mode-Seeking Divergences: Theory and Applications to GANs},
  booktitle = {AISTATS}, series = {PMLR}, volume = {206}, pages = {8321--8350}, year = {2023}}
```
Abstract [fetch]: "Our theoretical results show the mode-seeking nature of the Jensen-Shannon (JS) divergence over standard KL-divergence and Wasserstein distance measures"; numerics "highlight… the mode-covering properties of KL-divergence and Wasserstein distance measures." Flag: "standard KL" = forward/MLE direction in GAN estimation setting. Point: labels need re-deriving per divergence under a precise definition.

### Soletskyi, Gabrié, Loureiro 2024 — collapse from optimization, not expressivity
```bibtex
@article{soletskyi2024theoretical,
  author = {Roman Soletskyi and Marylou Gabri{\'e} and Bruno Loureiro},
  title = {A Theoretical Perspective on Mode Collapse in Variational Inference},
  journal = {arXiv preprint}, year = {2024}, eprint = {2410.13300}, archivePrefix = {arXiv}}
```
Even when "the variational family is statistically well-specified" (GMM q that CAN represent bimodal target), gradient-flow VI collapses; "mode collapse is not a by-product of insufficient model expressivity, but a consequence of the optimization procedure" [fetch]. Mean-alignment and vanishing-weight mechanisms; "quasi-mode collapse"; RealNVP validation. Strongest "dynamics, not divergence" source.

### Chan et al. 2022 — RL greedification
```bibtex
@article{chan2022greedification,
  author = {Alan Chan and Hugo Silva and Sungsu Lim and Tadashi Kozuno and A. Rupam Mahmood and Martha White},
  title = {Greedification Operators for Policy Optimization: Investigating Forward and Reverse KL Divergences},
  journal = {JMLR}, volume = {23}, number = {253}, pages = {1--79}, year = {2022},
  eprint = {2107.08285}, archivePrefix = {arXiv}}
```
Policy optimization toward Boltzmann targets: reverse KL stronger policy-improvement guarantees; reducing forward KL can yield worse policy; empirically forward KL more exploration but more suboptimal policies; differences often modest [abstract/search — no line-level quotes verified].

### Additional complications
- Minka + Bishop: reverse KL seeks the mode "with the largest mass (not the mode which is highest)".
- Bishop Fig 10.3(b)/(c): reverse-KL solutions are local minima (initialization artifact); MacKay Ex. 33.7 / Murphy Ex. 6.4: three distinct reverse-KL minima.
- Turner et al. 2008 "Counter-examples to variational free-energy compactness folk theorems" (UCL TR; cited in Turner & Sahani 2011 and Murphy PML2 as [Tur+08]) — not located online; cite via the 2011 chapter or PML2.
- ICLR 2025 blogpost "On LLM Knowledge Distillation — A Comparison between Forward KL and Reverse KL" (https://iclr.cc/virtual/2025/poster/31335): "with a sufficiently powerful student model and enough training steps, forward KL and reverse KL are likely to exhibit similar performance"; token-level Qwen2.5 experiments found "not much difference" [fetch]. Authors hidden on rendered page ("PLACEHOLDER FOR ACADEMIC ATTRIBUTION") — get from the ICLR page before citing.

## 5. Divergences introduced to fix forward/reverse shortcomings (theory-level)
- Minka 2005 [PDF]: "The divergences with 0 < α < 1 are a blend of these extremes. They are not zero-forcing, so they try to represent multiple modes, but will ignore modes that are far away from the main mass (how far depends on α)." Maps algorithms to α (mean-field α=0, BP/EP α=1, fractional BP/power EP any α).
- Bishop PRML §10.1.2 [PDF]: α∈{−1,+1} limits are the two KLs; α=0 is Hellinger.
- Huszár 2015 [fetch]: generalized JS_π interpolation; adversarial ≈ π=0.5.
- GKD [fetch]: JSD(β) for LLM KD (eq. 1).
- f-GAN [fetch]: unified f-divergence GAN training incl. Jeffreys and both KLs.
- Li & Turner 2016, Rényi Divergence Variational Inference (NeurIPS 2016, arXiv 1602.02311): α-family VI "enables a smooth interpolation from the evidence lower-bound to the log (marginal) likelihood" [fetch]; mass-covering-for-negative-α not verified.
- Naesseth, Lindsten, Blei 2020, "Markovian Score Climbing: Variational Inference with KL(p‖q)" (NeurIPS 2020, arXiv 2003.10374): forward/inclusive direction because it "has good statistical properties that makes it more appropriate for certain inference problems" [fetch abstract].

## Direct answers

**(a) The folklore, with evidentiary status.** Forward KL (= MLE direction) for coverage/calibrated support: zero-avoiding/inclusive/mass-covering/mean-seeking; "will typically over-estimate the support of p" (Murphy); "often leads to models that overgeneralise, and sometimes produce samples that are very unlikely under P" (Huszár). Reverse KL for sharp, conservative, high-precision outputs tolerating dropped modes: zero-forcing/exclusive/mode-seeking; "under-estimate[s] the support of p" (Murphy); underestimates variance (Blei, Minka); "assigns an extremely high cost to generating fake looking samples, and an extremely low cost on mode dropping" (Arjovsky–Bottou). VI uses reverse KL for tractability, not preference (MacKay eq. 33.9; Blei). Core arguments are THEORETICAL with two rarely-stated assumptions: (i) q capacity-limited relative to p (Bishop Fig 10.3, Minka Fig 1, Goodfellow Fig 14: "most obvious when the model has too little capacity"); (ii) global minimizer, not what SGD finds. EMPIRICAL support thinner and mixed: GAN sharpness folklore undercut by Goodfellow's f-GAN evidence; in LLM KD reverse KL helps on some tasks (MiniLLM, GKD instruction-tuning) but differences wash out with capacity and training time (Wu et al., ICLR'25 blog); janus's RLHF folklore was corrected in-post.

**(b) Standard motivating figures.** Bishop PRML Fig 10.2, 10.3 (the ur-figures; 10.3(b)/(c) are distinct reverse-KL local minima); Minka 2005 Fig 1 and Fig 3; Murphy PML1 Fig 6.3, PML2 Figs 5.1–5.2 (notebooks: KLfwdReverseMixGauss.ipynb, minimize_kl_divergence.ipynb, kl_pq_gauss.ipynb); Goodfellow Fig 14 & 15; Theis et al. Fig 1; Huszár Fig 1 (JS_π sweep); MiniLLM Fig 2; GKD Fig A.16; Ari Seff animation.

**(c) Strongest criticisms/complications.** (1) Capacity: well-specified q ⇒ same global optimum for every f-divergence (f-GAN; Wu et al. for softmax LLM students — differences are early-training head-vs-tail dynamics). (2) Mode-seeking ≠ finds a mode: optimum tracks largest-mass region not highest peak (Minka, Bishop); Goodfellow: "mode collapse is driven by a factor other than the choice of divergence." (3) Optimization, not objective: reverse-KL collapse occurs even with well-specified families as gradient-flow phenomenon (Soletskyi et al.); which mode is a local-minimum accident (Bishop 10.3b/c; MacKay Ex 33.7). (4) Compactness itself is a "folk-theorem" with published counterexamples (Turner & Sahani 2011; Turner et al. 2008; caveat in Murphy PML2 §5.1.4.3). (5) Under rigorous definition, JS is mode-seeking and "standard KL" is not (Li & Farnia 2023). (6) RL greedification: quality/coverage story maps onto guarantees only partially (Chan et al. 2022).

## Leads not verified
- Turner et al. 2008 UCL tech report — not located online; cite via 2011 chapter or Murphy PML2.
- Schulman blog: retrieved via r.jina.ai proxy (joschu.net ECONNRESET) — re-fetch before verbatim quoting.
- Bishop figure page numbers from free PDF (pp. 467–469) — spot-check print edition.
- Li & Turner 2016: "mass-covering for negative α" not confirmed verbatim.
- Chan et al. 2022: from abstract/summaries; no section-level quotes.
- ICLR 2025 blogpost authors: hidden on rendered page; get from https://iclr.cc/virtual/2025/poster/31335 .
- Frey et al. 2000 (origin of inclusive/exclusive per Minka): not fetched.
- Dibya Ghosh post date: undated; cite access date.
- MLaPP (Murphy 2012) §21.2.2: superseded by PML1/PML2; not checked.
