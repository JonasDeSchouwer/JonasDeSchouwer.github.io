# Peng et al. 2026 — Diagnosing and Mitigating Thinking Collapse in On-Policy Self-Distillation (AD-OPSD)

**arXiv 2607.10805** (Jul 2026). Keqin Peng, Chen Li, Yuanxin Ouyang, Yancheng Yuan, Liang Ding. Content from arXiv abs + HTML via WebFetch [FETCH 2026-08-18]. Incorporated: paper.typ appendix "Additional Related Work".

```bibtex
@article{peng2026thinkingcollapse,
  author  = {Keqin Peng and Chen Li and Yuanxin Ouyang and Yancheng Yuan and Liang Ding},
  title   = {Diagnosing and Mitigating Thinking Collapse in On-Policy Self-Distillation},
  journal = {arXiv preprint arXiv:2607.10805},
  year    = {2026}
}
```

## Setting

On-policy *self*-distillation (OPSD): teacher = model conditioned on the reference solution. Intro [FETCH]: "By distilling ground-truth-conditioned, on-policy target distributions of a teacher model into the student policy over its own sampled rollouts, OPSD avoids the exposure bias of off-policy imitation." Privileged-context teacher.

## Thinking collapse

- Metric: epistemic-token density, ET per 1k = (count of tokens in T / sequence length) × 1000, T = {wait, hmm, perhaps, maybe, actually, alternatively, seems, might, likely, check} — "planning, uncertainty externalization, and backtracking" [FETCH].
- Collapse: 10.3 → 7.9 ET/1k under standard OPSD (Qwen3-1.7B, AIME25).
- Localization [FETCH]: "aggressive teacher gradients at high-student-entropy decision forks, where student epistemic tokens are frequently suppressed"; "Within these identical high-entropy regions, there is a substantial target discrepancy where the student's top-1 predicted token is an ET, but the teacher's corresponding top-1 predicted token is a Non-ET" — generating "strong negative gradients on the student's exploratory paths."

## Fix (AD-OPSD)

Asymmetric divergence gate (Eq. 5) [FETCH]: U_i = σ(P_θ(y_i|·)(log P_θ(y_i|·) − log P_t(y_i|·))/τ) if u_i > 0, else 0 — suppresses only student-overconfident tokens, protects exploratory ones. ET partially restored (8.6/1k); +6.7 AIME25 (Qwen-1.7B), +4.1 average.

## Relevance

OPPOSITE-signed hedge-token effect vs. compromise latching: privileged (answer-conditioned) teacher is context-certain, so distillation deletes the *student's* hedges. Reconciliation used in paper.typ: the direction of hedge-mass transfer is set by which side hedges — a hedging teacher + context-confused student (our setting) amplifies hedges via geometric pooling; a certainty-injected teacher removes them. Delimits the compromise-latching prediction to teachers whose conditionals carry hedge mass.
