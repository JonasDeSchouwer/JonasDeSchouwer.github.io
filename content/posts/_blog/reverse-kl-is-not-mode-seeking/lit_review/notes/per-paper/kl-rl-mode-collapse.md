# GX-Chen et al. 2025 — KL-Regularized RL is Designed to Mode Collapse

**arXiv 2510.20817** (Oct 2025). Anthony GX-Chen, Jatin Prakash, Jeff Guo, Rob Fergus, Rajesh Ranganath. Content from arXiv abs + HTML via WebFetch [FETCH 2026-08-18]. Incorporated: lit_review @sec-folklore-crit (=== entry).

```bibtex
@article{gxchen2025modecollapse,
  author  = {Anthony GX-Chen and Jatin Prakash and Jeff Guo and Rob Fergus and Rajesh Ranganath},
  title   = {{KL}-Regularized Reinforcement Learning is Designed to Mode Collapse},
  journal = {arXiv preprint arXiv:2510.20817},
  year    = {2025}
}
```

## Core claims

- Abstract [FETCH]: "It is commonly believed that optimizing the reverse KL divergence results in 'mode seeking', while optimizing forward KL results in 'mass covering', with the latter being preferred if the goal is to sample from multiple diverse modes. We show -- mathematically and empirically -- that this intuition does not necessarily transfer well to doing reinforcement learning with reverse/forward KL regularization."
- "Mode coverage depends primarily on other factors, such as regularization strength, and relative scales between rewards and reference probabilities."
- "commonly used settings such as low regularization strength and equal verifiable rewards tend to specify unimodal target distributions, meaning the optimization objective is, by construction, non-diverse."
- "the choice of reverse/forward KL determines the family of optimal target distributions, parameterized by the regularization coefficient."

## Equations [FETCH]

- Eq. 1 (reverse): J_β(π_θ) = E_{π_θ}[R(y)] − β D_KL(π_θ‖π_ref); Eq. 2 optimum: G_β(y) = (1/ζ) π_ref(y) exp(R(y)/β).
- Eq. 4 (forward): J_fwd(π_θ) = E_{π_θ}[R(y)] − β D_KL(π_ref‖π_θ); Eq. 5 optimum: G_fwd(y) = β π_ref(y)/(Λ − R(y)), Λ > max_y R(y).
- Body: "by construction, KL-regularized RL with equal rewards never promotes a low support answer"; "the solution distribution is often by definition unimodal"; "diversity collapse a natural consequence of correctly solving the RL problem."

## Fix

MARA (Mode Anchored Reward Augmentation): for high-reward samples, r̄_i = R(z) + β(log π_ref(z) − log π_ref(y_i)) — anchors ratios to a reference high-reward trajectory so the induced target is ~uniform over high-reward modes; works with either KL direction.

## Relevance

Third mechanism relocating folklore-attributed behavior away from the divergence direction (alongside wu2025rethinking = capacity, soletskyi2024theoretical = optimization): here, *target design*. Competing explanation to the paper's cross-context-coupling account — must be engaged in paper.typ related work (currently only in lit_review).
