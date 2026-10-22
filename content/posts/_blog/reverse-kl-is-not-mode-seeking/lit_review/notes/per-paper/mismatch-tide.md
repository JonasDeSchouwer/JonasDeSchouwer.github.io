# Yu et al. 2026 — Mismatch Matters: On-Policy Distillation Beyond Token Agreement (TIDE)

**arXiv 2608.09836** (Aug 2026). Zichao Yu, Chengzhi Yu, Shengze Xu, Yujin Han, Bingqing Jiang, Xu Wang, Difan Zou. Content from arXiv abs + HTML via WebFetch [FETCH 2026-08-18]. Incorporated: lit_review @sec-folklore-crit (=== entry) + estimator-criticism clause in @sec-criticism.

```bibtex
@article{yu2026mismatch,
  author  = {Zichao Yu and Chengzhi Yu and Shengze Xu and Yujin Han and Bingqing Jiang and Xu Wang and Difan Zou},
  title   = {Mismatch Matters: On-Policy Distillation Beyond Token Agreement},
  journal = {arXiv preprint arXiv:2608.09836},
  year    = {2026}
}
```

## Setting

Eq. 1 [FETCH]: ℒ_OPD(θ) = E_{x∼D}[D_KL(q_θ(·|x)‖p(·|x))] = E_{x, o∼q_θ}[Σ_t log(q_θ(o_t|s_t)/p(o_t|s_t))] — sampled-token reverse KL along student rollouts (the estimator family of lit_review Table 1).

## Failure mode: degenerate agreement / student-induced teacher hacking

- Abstract [FETCH]: "degenerate agreement, where students exploit repetitive loops to achieve near-perfect token agreement with the teacher despite globally flawed responses."
- Body (§2.2) [FETCH]: "the student can reduce the measured teacher–student divergence by generating contexts that make the teacher's outputs more aligned with its own" — repetition conditions the teacher to favor the same continuation; median token KL 0.0004 nats on globally degenerate responses (their Fig. 2).
- Mechanism = conditional acceptance of student-manufactured contexts; the agreement objective is gameable because the student controls visitation. Same pathology MiniLLM's teacher-mixed sampling patched in 2023 (reward hacking via repetition), now with a mechanism.

## Fix (TIDE) [FETCH]

- Excess branch (§3.1): g_exc = −E[Σ_t m_t^− sg[h(a_t)] log q_θ(o_t|s_t)], h(a) = 2(e^{a/2} − 1) — bounded Hellinger-shaped suppression of student-excess tokens.
- Deficit branch (Eq. 4, §3.2): ℒ_def = −E[Σ_t m_t^+ Σ_{v∈K_t} p̄_t(v) log q_θ(v|s_t)] — analytic teacher top-K cross-entropy injection (no sampling of deficit tokens).
- Strong mismatch (Qwen3-8B → Qwen3-1.7B-Base, Table 1): Avg@8 6.9% → 20.3%; length 22,395 → 7,294 tokens (3.6×); missing-boxed 65.5% → 5.4%.

## Relevance

Closest published mechanism to compromise latching ("context-robust continuations the teacher conditionally accepts") — but via repetition loops, not hedge/filler tokens. The specific universally-acceptable-hedge-token claim remains open ground; paper.typ must cite and differentiate.
