# Qwen Team 2025 — Qwen3 Technical Report

**arXiv 2505.09388v1** (14 May 2025; report dated 2025-05-15). Qwen Team (Alibaba). Sources: PDF `https://arxiv.org/pdf/2505.09388` (curl + pypdf text, 2026-09-14). Read in full for §4.2 (Reasoning RL), §4.5 (Strong-to-Weak Distillation) and §5.3 (ablation / Table 21).

```bibtex
@article{qwen2025qwen3,
  author  = {{Qwen Team}},
  title   = {Qwen3 Technical Report},
  journal = {arXiv preprint arXiv:2505.09388},
  year    = {2025}
}
```

## Tier

- **D (partial)** — one quotable statement that off-policy training (rollout reuse) improves sample efficiency, but it is made about the **GRPO reasoning-RL stage (§4.2)**, not about the on-policy distillation stage (§4.5). Quoting it as evidence about OPD would misattribute it.
- **E (for OPD, not for async OPD)** — this *is* the production tech report that made on-policy distillation credible at scale (Table 21: 74.4 AIME'24 at 1,800 GPU-hours vs 67.6 at 17,920 for RL), and it is the result the Thinking Machines blog set out to replicate. But nothing in §4.5 says the OPD stage was run asynchronously, so it does not establish async OPD in production.
- Not **A**/**B**/**C** — no staleness analysis, no async-vs-sync numbers, no theory.
- Not **F**/**G** — async OPD is neither demonstrated nor stated to be used.

## Setting

- Students (§4.5): "5 dense models (Qwen3-0.6B, 1.7B, 4B, 8B, and 14B) and one MoE model (Qwen3-30B-A3B)". Teachers: "Qwen3-32B or Qwen3-235B-A22B" (§4.5 phase 2).
- Teacher fixed: **yes** (the teachers are the already-post-trained flagship models; the report describes no teacher update during distillation).
- Domain/benchmarks: general post-training; the §5.3 ablation is "math and code-related queries" only, evaluated on AIME'24, AIME'25 (with pass@64), MATH500, LiveCodeBench v5, MMLU-Redux, GPQA-Diamond.
- Framework/code: not disclosed.

## Async mechanism

**Not stated for the distillation stage.** §4.5 describes two phases and neither mentions staleness, buffers, pipelining, rollout reuse, batch size, or number of gradient steps per rollout batch:

> "(1) **Off-policy Distillation**: At this initial phase, we combine the outputs of teacher models generated with both /think and /no think modes for response distillation. This helps lightweight student models develop basic reasoning skills and the ability to switch between different modes of thinking, laying a solid foundation for the next on-policy training phase."

> "(2) **On-policy Distillation**: In this phase, the student model generates on-policy sequences for fine-tuning. Specifically, prompts are sampled, and the student model produces responses in either /think or /no think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence."

**Terminology warning — "off-policy" means two different things in this report.** In §4.5 it means *teacher-sampled data used for SFT* (out of scope for the async sweep: the behaviour policy is the teacher, not an earlier student). In §4.2 it means *reusing a rollout batch for more than one gradient step*, which is exactly the staleness notion in scope. Any citation must say which.

The §4.2 usage, verbatim (Reasoning RL stage, GRPO on 3,995 query-verifier pairs):

> "We observe that using a large batch size and a high number of rollouts per query, along with off-policy training to improve sample efficiency, is beneficial to the training process."

No lag bound, no correction, no numbers, and no indication that this carries over to §4.5.

## Objective

Under-specified. §4.5: "fine-tuned by aligning its logits with those of a teacher model ... to minimize the KL divergence" — full-vocabulary logit-level KL, **direction not stated**. §5.3 confirms it is logit-based, not sampled-token: "distillation from teacher logits enables the student model to expand its exploration space and enhance its reasoning potential". §4 preamble: "directly distilling the output logits from teacher models into lightweight student models can effectively enhance their performance while maintaining fine-grained control over their reasoning processes."

So in our vocabulary this is a full-vocab distribution-level KL of unstated direction (neither `rkl` nor `fkl` can be asserted from the source), on student-sampled sequences. No equation is given anywhere in the report.

## Off-policy correction

**None stated** for the distillation stage. The §4.2 RL stage says GRPO was used (which carries a clipped importance ratio by construction) but gives no clip range, no mask, no equation, and no statement about which stage the "off-policy training" applies to beyond §4.2.

## Results: async vs sync

**None.** Table 21 is on-policy distillation vs RL, both starting from the same off-policy-distilled checkpoint — not sync vs async. Reproduced for reference (numbers in parentheses are pass@64):

| Method | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU Hours |
|---|---|---|---|---|---|---|---|
| Off-policy Distillation | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | – |
| + Reinforcement Learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
| + On-policy Distillation | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

(Table 21, §5.3. Qwen3-8B.)

## Explanation offered

On why OPD beats RL (§5.3) — a distribution-level argument, which is the part that matters for whether the objective should tolerate stale data:

> "The results, summarized in Table 21, show that distillation achieves significantly better performance than reinforcement learning while requiring approximately only 1/10 of the GPU hours. Furthermore, distillation from teacher logits enables the student model to expand its exploration space and enhance its reasoning potential, as evidenced by the improved pass@64 scores on the AIME'24 and AIME'25 benchmarks after distillation, compared to the initial checkpoint. In contrast, reinforcement learning does not lead to any improvement in pass@64 scores. These observations highlight the advantages of leveraging a stronger teacher model in guiding student model learning."

§4 preamble, on cost:

> "It leads to better immediate performance, as indicated by higher Pass@1 scores, and also improves the model's ability of exploration, as reflected in improved Pass@64 results. In addition, it achieves these gains with much greater training efficiency, requiring only 1/10 of the GPU hours compared to the four-stage training method."

No explanation is offered for the §4.2 off-policy remark.

## Quotes

- §4.2 (Reasoning RL, GRPO): "We observe that using a large batch size and a high number of rollouts per query, along with off-policy training to improve sample efficiency, is beneficial to the training process." — **the only statement in the report about rollout reuse / staleness, and it is about RL, not distillation.**
- §4.5(1): "Off-policy Distillation: At this initial phase, we combine the outputs of teacher models generated with both /think and /no think modes for response distillation."
- §4.5(2): "On-policy Distillation: In this phase, the student model generates on-policy sequences for fine-tuning. Specifically, prompts are sampled, and the student model produces responses in either /think or /no think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence."
- Abstract/§1: "smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge transfer from larger models to enhance their capabilities."
- §4 preamble: "Preliminary experiments suggest that directly distilling the output logits from teacher models into lightweight student models can effectively enhance their performance while maintaining fine-grained control over their reasoning processes. This approach eliminates the necessity of performing an exhaustive four-stage training process individually for every small-scale model."

## Not in the source

- **Nothing on staleness in the on-policy distillation stage.** No mention of asynchronous execution, replay, pipelining, rollout reuse, gradient steps per rollout batch, batch size, or rollout count for §4.5.
- No KL direction, no equation, no temperature, no learning rate, no truncation/top-k for the distillation loss.
- No off-policy correction (importance ratio, clipping, masking) stated for distillation.
- No sync baseline for the §4.2 off-policy-RL remark, and no quantification of how off-policy ("how many steps behind") that training was.
- No code, no infrastructure description — so the claim sometimes made that "Qwen3 ran async OPD" **cannot be sourced from this report**.
