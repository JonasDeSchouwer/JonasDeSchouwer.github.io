# G — Relevant papers not yet incorporated

Findings of four parallel literature sweeps (2026-08-17/18) run against the paper's thesis (cross-context coupling; FKL→arithmetic vs RKL→geometric mixture; compromise latching) and the lit review. **Verification status: abstract/body fetched by search agents; none has a per-paper note yet — write one from the primary source before citing (per convention).** Papers since incorporated: gxchen2025modecollapse, xin2026agreementtrap, yu2026mismatch, peng2026thinkingcollapse, zhang2026illusion (see their notes). Tiers: [T1] read before writing further sections; [T2] read soon; [T3] skim/cite-in-passing.

**Overall verdicts from the sweeps:**
- The arithmetic-vs-geometric lemma is classical (≥4 independent statements: Amari 2007; Abbas 2009; Banerjee 2005 + Nielsen–Nock 2009; Grosse 2013). Cite; claim only the conditional/coupled-LM instantiation.
- Confirmed open ground: cross-context coupling as the KD bottleneck; RKL latching onto a shared low-probability *hedge* token; the constrained amortized optimum (Margossian–Blei stop short of it); distribution-level analysis of shared predictors under conflicting tasks.
- "FKL fixes OPD" is no longer novel per se (§3 below); the cross-context *justification* is.

## 1. Classic math prior art: sided KL barycenters, pooling, PoE

- [T1] **Amari 2007**, "Integration of Stochastic Models by Minimizing α-Divergence", Neural Comput. 19(10):2780 — α-mixtures minimize weighted α-divergences; arithmetic (m-) and normalized-geometric (e-) mixtures are exactly the two KL endpoints. The canonical general statement of the lemma. (Deep structure: Amari–Nagaoka, *Methods of Information Geometry*.)
- [T1] **Abbas 2009**, "A KL View of Linear and Log-Linear Pools", Decision Analysis 6(1):25 — linear pool = argmin Σw_i KL(p_i‖q), log-linear pool = argmin Σw_i KL(q‖p_i), on the simplex with normalization: the lemma verbatim in expert-aggregation form.
- [T1] **Banerjee, Merugu, Dhillon, Ghosh 2005**, "Clustering with Bregman Divergences", JMLR 6:1705 — Prop. 1: expected-divergence minimizer in the second argument is the arithmetic mean for ANY Bregman divergence (+ converse: only Bregman divergences have this property).
- [T1] **Nielsen & Nock 2009**, "Sided and Symmetrized Bregman Centroids", IEEE IT 55(6):2882 (arXiv 0711.3242) — Thms 2.1/2.2/2.4: both sided centroids in closed form; KL row: geometric mean via ∇F = log. Caveat: extended (unnormalized) KL; normalized version in Abbas/Nielsen 2013.
- [T2] **Grosse, Maddison, Salakhutdinov 2013**, "Annealing Between Distributions by Averaging Moments", NIPS — eqs. (15)–(16): geometric path = argmin of convex combo of reverse KLs ("mass only in the intersection" = the veto mechanism); moment path = forward side. Mainstream-ML statement. (Generalizations: q-paths, Masrani et al. UAI 2021, arXiv 2107.00745.)
- [T2] **Dietrich & List 2016**, "Probabilistic Opinion Pooling" (Oxford Handbook) — fn. 7: geometric pool zero-veto ("some individual assigns a probability of zero … geometric average … zero"); zero-preservation axioms (Aczél–Wagner 1980; McConway 1981). Cleanest citable veto statement — the mechanism behind compromise latching.
- [T2] **Hinton 1999/2002**, Products of Experts / contrastive divergence, Neural Comput. 14(8):1771 — geometric combination as a modeling primitive; sharpening + per-expert veto framed as a *feature* (our pathology).
- [T3] **Genest & Zidek 1986**, "Combining Probability Distributions", Statist. Sci. 1(1):114 — canonical linear-vs-log pooling survey (axiomatics; externally Bayesian). **Heskes NIPS 1997** — log pools in ensemble learning.
- [T3] **Veldhuis 2002** (IEEE SPL 9(3)) — earliest categorical arithmetic/normalized-geometric sided-KL pair; **Nielsen 2013** (IEEE SPL 20(7), arXiv 1303.7286) — Jeffreys centroid via Lambert W between the two means.
- [T3] **Topsøe 2000** (IEEE IT 46(4)) compensation identity Σw_i KL(p_i‖q) = Σw_i KL(p_i‖p̄) + KL(p̄‖q); **Lin 1991** (IEEE IT 37) generalized JSD — the algebra behind the FKL decomposition (forward half is latent in tishby1999ib Eqs. 18/27, already in bib).
- [T2] **AMiD** (Shin et al., ICLR 2026, arXiv 2510.15982) — Amari α-mixtures as teacher–student assistant distributions in LLM KD, explicitly citing the sided-minimizer theorems. Closest LLM paper to our math; distinguish: their mixtures are teacher–student pairs, ours are per-context target families.
- [T3] **MOD** (Shi et al., NeurIPS 2024, arXiv 2406.18853) — multi-objective decoding optimum = weighted geometric mean of policies under RKL-regularized alignment. **Razafindralambo et al. 2026** (arXiv 2603.04204) power-mean pooling; **arXiv 2606.00432** KL-barycenter decoding for RAG.
- [T3] **hinton2015distilling** and **Kim & Rush 2016 SeqKD** — foundational KD cites absent from the bib.

## 2. Amortized inference / shared-encoder analogs

- [T1] **Margossian & Blei**, "Amortized Variational Inference: When and Why?", UAI 2024 (arXiv 2307.11018) — Def. 2.2 is exactly the coupling condition (identical inputs ⇒ shared posterior); Thm 3.4 characterizes when A-VI can close the gap; does NOT characterize the constrained optimum — our theorem is the natural continuation.
- [T2] **Cremer, Li, Duvenaud**, ICML 2018 (arXiv 1801.03558) — amortization gap often dominates the approximation gap: "coupling, not family capacity, binds" in VAEs.
- [T2] **Le et al.**, "Revisiting Reweighted Wake-Sleep", UAI 2019 (arXiv 1805.10469) — most explicit forward-vs-reverse contrast for an amortized encoder (RKL/ELBO underestimates variance per-context; wake/FKL covers).
- [T3] **Hinton et al. 1995** wake-sleep (Science 268); **Bornschein & Bengio** RWS (ICLR 2015); **Naesseth et al.** Markovian Score Climbing (NeurIPS 2020, arXiv 2003.10374) — RWS gradient bias for inclusive KL; **McNamara et al.** SMC-Wake (AISTATS 2024, arXiv 2403.10610) — "circular pathology," concentrated fixed points; **Ikeda, Amari, Nakahara** NIPS 1998 — classical wake/sleep fixed-point analysis.
- [T3] **Hoffman & Johnson 2016** ELBO surgery — optimal shared prior = aggregate (arithmetic-mixture) posterior; VampPrior (Tomczak & Welling 2018). Prior-space precedent for "shared object = mixture of per-datum optima."
- [T3] Posterior-collapse attribution: **He et al.** ICLR 2019 (lagging encoder), vs **Lucas et al.** NeurIPS 2019 and **Dai & Wipf** ICML 2020 (landscape, anti-KL-story) — engage if the paper discusses collapse mechanisms.
- [T2] **Galashov et al.**, "Information Asymmetry in KL-regularized RL", ICLR 2019 (arXiv 1905.01240) — default policy denied context info, trained by *forward* KL ⇒ marginalization/arithmetic mixture. Closest RL instance of the information-restricted-q setup (forward side only). Also Distral (Teh et al. 2017).
- [T3] **Shu et al.**, Amortized Inference Regularization, NeurIPS 2018 — the flip side: coupling as beneficial regularizer.

## 3. 2026 OPD literature (field snapshot: ~250 papers since Dec 2025; coverage check: github.com/chrisliu298/awesome-on-policy-distillation)

Forward-KL / mixed-direction fixes (kills "FKL fixes OPD" as a standalone novelty):
- [T2] **Zhao et al.**, "Decoupling KL and Trajectories" (arXiv 2605.16826) — 2×2 of prefix source × KL direction unifying SFT/DAgger/offline-RL/OPD; KL mixing + entropy-gated curricula. Overlaps our Exp. 2/3 design space.
- [T2] **Trust Region OPD** (arXiv 2606.01249) — RKL in trust region, FKL for off-policy/outlier regions. (Distinct from arXiv 2607.04751.)
- [T2] **Xu et al.** (arXiv 2605.12483) — forward-KL warmup stage is load-bearing (−1.7 MATH if removed) in a sparse-to-dense recipe.
- [T3] **Veto** (Jang et al., arXiv 2601.07155) — both directions pathological on low-confidence tokens; adaptive logit-space target + "decisiveness knob". **ToDi** (EMNLP 2025, arXiv 2505.16297) — per-token sigmoid FKL/RKL interpolation. **ARKD** (arXiv 2606.29869), **f-OPD** (2605.17862), **AOPD** (2605.06387), **SG-OPD** (2606.09304), **HPD** (2604.20244), **Uni-OPD** (2605.03677).

Theory of OPD:
- [T2] **G-OPD/ExOPD** (Yang et al., arXiv 2602.12125) — OPD = dense KL-constrained RL with teacher log-ratio as implicit reward; reward-scaling lets student surpass teacher. Also candidate for lit_review @sec-estimators.
- [T2] **OPD+** (Zhao et al., Columbia, arXiv 2606.01039) — the ubiquitous stop-gradient advantage yields biased gradient estimates for general divergences; extends tang2025pitfalls. Candidate for the lit_review estimator audit.
- [T2] **Ge et al.** (arXiv 2608.11829) — "illusory distillation": OPD gains are sampling-efficiency; pass@K crosses over to base model; supports "RKL-OPD sharpens rather than transfers."
- [T3] **States-not-tokens** position (arXiv 2605.22731); **token teachability** (2605.26844); **Apple per-token gradient-alignment diagnostic** (2605.10889); **OPD geometry/subspace locking** (2606.07082); **many-faces incl. biased TopK-RKL gradients** (2605.11182); **rock tokens** (2605.09253); **empirical failure modes + fixes** (2603.25562); **counterfactual recoverability** (2608.04408); **RG-OPD** (2607.04037); **TIP** (2604.14084); **vOPD/PowerOPD/REOPD** (2605.07865/2606.17199/2608.11698); **GAD black-box OPD** (2511.10643); **second survey** (Zhang, 2606.22793 — "vocabulary-level probability routing" is adjacent to compromise latching).

Pathologies (supporting evidence for the paper):
- [T2] **Length inflation** (Luo et al., arXiv 2604.08527 — OPD "implicitly favors long and repetitive rollouts"; StableOPD) and **length exploitation** (Wang et al., arXiv 2607.13399 — truncation/padding gaming of the token-level objective).
- [T2] **OPSD diversity loss** (Nicolicioiu et al., arXiv 2606.26091) — blames demonstration-conditioning CMI, NOT KL direction: competing mechanism to distinguish.
- [T3] **AMR-SD** (2605.18529) — reflection-marker collapse counts; **Denser ≠ Better** (2607.01763) — OPSD drift/collapse warning; **SDFT** (2601.19897) and **MixSD** (2605.16865) — distillation-vs-SFT knowledge injection (MixSD: up to 100% retention vs 1% for SFT).

Bib-collision checks before adding: 2604.13016 vs `li2026rethinking`; 2603.07079 vs `jin2026entropy`; 2604.00626 vs `song2026survey`; 2604.00223 vs `luong2026drkl`.

## 4. KD theory, capacity gap, facts, soft-label harm

- [T1] **TAID** (Shing et al., ICLR 2025 Spotlight, arXiv 2501.16937) — temporally adaptive teacher–student interpolation with mode-collapse-prevention theory under capacity gap; uncited competitor whose engineered-around failure our framework explains.
- [T1] **Snell, Klein, Zhong 2022**, "Learning by Distilling Context" (arXiv 2209.15189) + **Padmanabhan et al.** NeurIPS 2023 (2306.09306) + **Kujanpää et al.** 2024 (2412.14964) + **Caccia et al.** 2025 (2503.08727) — the open-book-teacher → closed-book-student lineage; direct precedent for Exp. 1; reviewers will ask.
- [T1] **Chang & McCallum**, ACL 2022 (softmax bottleneck in ambiguous contexts) + **Yang et al.** ICLR 2018 (arXiv 1711.03953) — the canonical *output-space* capacity account the thesis displaces.
- [T2] **Nagarajan et al.**, NeurIPS 2023 (arXiv 2301.12923) — students *exaggerate* teacher confidence patterns; nearest formal ancestor of hedge amplification.
- [T2] **Tiapkin et al.**, "On Teacher Hacking in LM Distillation" (arXiv 2502.02671, CoLLAs 2025) — students over-optimize against imperfect teachers on fixed data; latching-onto-hedge-mass is a mechanistic instance.
- [T2] **Müller, Kornblith, Hinton 2019** (label-smoothed teachers distill worse — uniform hedge mass damages students); **Zhou et al.** ICLR 2021 (bias-variance of soft labels; "regularization samples" where soft labels hurt); **Yuan et al.** CVPR 2020 (KD ≈ label-smoothing regularization).
- [T2] **Allen-Zhu & Li**, Physics of LMs 3.3 (arXiv 2404.05405; + 3.1 2309.14316, 3.2 2309.14402) — synthetic-fact capacity methodology mirrored by Exp. 1; the capacity-story foil.
- [T3] **Menon et al.** ICML 2021 (Bayes-teacher, bias-variance); **Phuong & Lampert** ICML 2019 (student = projection of teacher); **Dao et al.** ICLR 2021 (semiparametric); **Stanton et al.** NeurIPS 2021 (fidelity gap is optimization — the alternative explanation to distinguish); **Mobahi et al.** 2020 (self-distillation = amplified regularization); **Safaryan et al.** 2023 (partial variance reduction).
- [T3] Capacity gap: **Mirzadeh et al.** AAAI 2020 (TAKD); **Cho & Hariharan** ICCV 2019; **Zhang et al.** ACL 2025 "Law of Capacity Gap" (arXiv 2311.07052); **Busbridge et al.** "Distillation Scaling Laws" (arXiv 2502.08606); **ATKD** ACL 2024 (2402.11890).
- [T3] Tail handling: **BiLD** (2406.13555 — long-tail logit noise harmful); **Peng et al.** ACL 2025 (2410.16215 — top-k truncation *improves* pretraining KD); **Distillation Traps and Guards** (2604.18963 — teacher calibration controls distillability); **Hooker et al.** 2019 (compression drops the tail first); cross-tokenizer trio (2402.12030, 2406.17328, 2503.20083); **SelecTKD** (2510.24021), token-adaptive KD (2510.11615).
- [T3] **Ovadia et al.** EMNLP 2024 (2312.05934 — FT struggles to add facts; RAG wins); **Zhu et al.** 2020 (2012.00363 — fact editing); **BabyLlama RKL** (2410.22081); ICLR 2025 blogpost "FKL vs RKL in LLM KD" (openreview jGVCs8gomF).

## Ready-to-paste BibTeX (Tier 1 + key Tier 2; add to references.bib only upon incorporation, after writing a per-paper note)

```bibtex
@article{amari2007alpha,
  author  = {Shun-ichi Amari},
  title   = {Integration of Stochastic Models by Minimizing $\alpha$-Divergence},
  journal = {Neural Computation},
  volume  = {19},
  number  = {10},
  pages   = {2780--2796},
  year    = {2007}
}

@article{abbas2009klpools,
  author  = {Ali E. Abbas},
  title   = {A {K}ullback-{L}eibler View of Linear and Log-Linear Pools},
  journal = {Decision Analysis},
  volume  = {6},
  number  = {1},
  pages   = {25--37},
  year    = {2009}
}

@article{banerjee2005bregman,
  author  = {Arindam Banerjee and Srujana Merugu and Inderjit S. Dhillon and Joydeep Ghosh},
  title   = {Clustering with {B}regman Divergences},
  journal = {Journal of Machine Learning Research},
  volume  = {6},
  pages   = {1705--1749},
  year    = {2005}
}

@article{nielsen2009sided,
  author  = {Frank Nielsen and Richard Nock},
  title   = {Sided and Symmetrized {B}regman Centroids},
  journal = {IEEE Transactions on Information Theory},
  volume  = {55},
  number  = {6},
  pages   = {2882--2904},
  year    = {2009}
}

@inproceedings{grosse2013moments,
  author    = {Roger B. Grosse and Chris J. Maddison and Ruslan Salakhutdinov},
  title     = {Annealing Between Distributions by Averaging Moments},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2013}
}

@inproceedings{margossian2024amortized,
  author    = {Charles C. Margossian and David M. Blei},
  title     = {Amortized Variational Inference: When and Why?},
  booktitle = {Uncertainty in Artificial Intelligence (UAI)},
  year      = {2024},
  note      = {arXiv:2307.11018}
}

@inproceedings{cremer2018amortization,
  author    = {Chris Cremer and Xuechen Li and David Duvenaud},
  title     = {Inference Suboptimality in Variational Autoencoders},
  booktitle = {International Conference on Machine Learning},
  year      = {2018}
}

@inproceedings{shing2025taid,
  author    = {Makoto Shing and Kou Misaki and Han Bao and Sho Yokoi and Takuya Akiba},
  title     = {{TAID}: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  note      = {arXiv:2501.16937}
}

@article{snell2022context,
  author  = {Charlie Snell and Dan Klein and Ruiqi Zhong},
  title   = {Learning by Distilling Context},
  journal = {arXiv preprint arXiv:2209.15189},
  year    = {2022}
}

@inproceedings{padmanabhan2023propagating,
  author    = {Shankar Padmanabhan and Yasumasa Onoe and Michael J.Q. Zhang and Greg Durrett and Eunsol Choi},
  title     = {Propagating Knowledge Updates to {LM}s Through Distillation},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  note      = {arXiv:2306.09306}
}

@inproceedings{chang2022softmax,
  author    = {Haw-Shiuan Chang and Andrew McCallum},
  title     = {Softmax Bottleneck Makes Language Models Unable to Represent Multi-mode Word Distributions},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year      = {2022}
}

@inproceedings{yang2018softmaxbottleneck,
  author    = {Zhilin Yang and Zihang Dai and Ruslan Salakhutdinov and William W. Cohen},
  title     = {Breaking the Softmax Bottleneck: A High-Rank {RNN} Language Model},
  booktitle = {International Conference on Learning Representations},
  year      = {2018}
}

@inproceedings{nagarajan2023disobey,
  author    = {Vaishnavh Nagarajan and Aditya Krishna Menon and Srinadh Bhojanapalli and Hossein Mobahi and Sanjiv Kumar},
  title     = {On Student-Teacher Deviations in Distillation: Does it Pay to Disobey?},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  note      = {arXiv:2301.12923}
}

@article{tiapkin2025teacherhacking,
  author  = {Daniil Tiapkin and Daniele Calandriello and Johan Ferret and others},
  title   = {On Teacher Hacking in Language Model Distillation},
  journal = {arXiv preprint arXiv:2502.02671},
  year    = {2025}
}

@article{zhao2026decouplingkl,
  author  = {Anhao Zhao and Haoran Xin and Yingqi Fan and Junlong Tong and Wenjie Li and Xiaoyu Shen},
  title   = {Decoupling {KL} and Trajectories: A Unified Perspective for {SFT}, {DAgger}, Offline {RL}, and {OPD} in {LLM} Distillation},
  journal = {arXiv preprint arXiv:2605.16826},
  year    = {2026}
}

@article{hinton2015distilling,
  author  = {Geoffrey Hinton and Oriol Vinyals and Jeff Dean},
  title   = {Distilling the Knowledge in a Neural Network},
  journal = {arXiv preprint arXiv:1503.02531},
  year    = {2015}
}

@inproceedings{kim2016seqkd,
  author    = {Yoon Kim and Alexander M. Rush},
  title     = {Sequence-Level Knowledge Distillation},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  year      = {2016}
}
```

Author-list caveat: names transcribed from the sweep agents' primary-source fetches; verify each against the paper's own author block when incorporating (e.g., the Tiapkin et al. "and others" needs the full list).
