# Liu, Ramachandran, Gangwani, Fu & Sethy 2025 — Knowledge Distillation with Training Wheels

**arXiv 2502.17717v1** (24 Feb 2025). Guanlin Liu, Anand Ramachandran, Tanmay Gangwani, Yan Fu, Abhinav Sethy (Amazon Alexa AI). Sources: full PDF (`curl https://arxiv.org/pdf/2502.17717`, pypdf extraction, 11 pages incl. appendix), fetched 2026-09-14. Async-OPD sweep 2026-09-14.

```bibtex
@article{liu2025trainingwheels,
  title         = {Knowledge Distillation with Training Wheels},
  author        = {Guanlin Liu and Anand Ramachandran and Tanmay Gangwani and Yan Fu and Abhinav Sethy},
  journal       = {arXiv preprint arXiv:2502.17717},
  year          = {2025},
  eprint        = {2502.17717},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2502.17717}
}
```

## Tier

- **G (weak)** — the student's reverse-KL/PCL updates are taken on a data mix that includes a replay buffer of earlier-batch trajectories, and four gradient updates are taken per batch of generated episodes (§3.3, p.7). Both are in-scope staleness mechanisms, but they are stated in one sentence with no measurement, no ablation and no discussion.
- **D (weak)** — one quotable claim that the PCL form of the reverse-KL objective needs no correction for off-policy trajectories (§2.1, p.3). The claim is made *about teacher demonstrations*, not about a stale student, so it only transfers to our question by analogy.
- **Careful**: the paper's own "off-policy" label means **teacher/oracle demonstrations**, not stale student samples — see `## Quotes`. On the primary screening question ("is the off-policy data teacher demonstrations or stale student samples?") the answer is *teacher demonstrations*, i.e. out of scope; the note exists only because of the replay buffer and the 4-updates-per-rollout-batch detail.
- Not A (no argument about how staleness interacts with the estimator; the paper never varies the behavior policy), not B/C (no async-vs-sync numbers anywhere), not E/F.

## Setting

- Student: FLAN-T5-SMALL. Teacher: FLAN-T5-XL (§3.2, p.7). Teacher **fixed** (it is treated as part of the environment: "Note that the teacher, here, is part of the environment and not part of the policy", §2.3, p.4).
- Tasks: WMT-19 De→En translation and XSum summarization; metrics ROUGE-2 vs latency on a single NVIDIA T4 (§3.2–§3.4, pp.7).
- Baselines: DistillSpec-style lossy speculative decoding with a GKD-trained student (§3.1–§3.3).
- No code release stated.
- The paper's actual contribution is orthogonal to our review: the student's action space gains a `<τ>` "ask the teacher" token and is trained under a constrained-RL teacher-use budget for test-time speculative decoding.

## Async mechanism

- Stale element 1 — multiple updates per rollout batch, **for both the proposed method and the GKD baseline**: "We performed four gradient updates per batch of episode generation" (§3.3, p.7); for DistillSpec/GKD, "For each batch of trajectories sampled, we performed four gradient updates" (§3.3, p.7). So on updates 2–4 the behavior student is the student from up to 3 updates earlier.
- Stale element 2 — replay buffer: "with a data mix that contains samples from a replay buffer, as well as oracle trajectories and on-policy trajectories that were generated in the current batch" (§3.3, p.7). The buffer is defined only by contrast with "generated in the current batch", so it holds earlier-batch oracle **and** earlier-batch student trajectories.
- How staleness is measured: **not at all**. No buffer size, no age distribution, no mixing ratio, no ratio/drift statistic.
- Range studied: a single point (4 updates/batch, unspecified buffer). No 1-update control, no buffer-size sweep.

## Objective

- Reverse KL, `KL(student ‖ teacher)`, recast as entropy-regularized value optimization and optimized by Path Consistency Learning rather than by a score-function policy gradient. The value identity (§2.1, p.2, Eq. 1) is
  `E_{i:T} = E_{x_i∼π(.|X_{<i})}[ log (π(x_i|X_{<i}) / p(x_i|X_{<i})) + E_{i+1:T} ]`
  with `V^π(X_{<i}) = −E_{i:T}`, giving the Bellman form (Eq. 2, §2.1, p.2)
  `V^π(X_{<i}) = Σ_{x_i} π(x_i|X_{<i}) [ log p(x_i|X_{<i}) − τ log π(x_i|X_{<i}) + γ V^π(X_{<i+1}) ]`
  and, with the teacher-use constraint, Eq. 9 (§2.2, p.4)
  `V^π_C(X_{<i},Y) = Σ_{x_i} π(x_i|X_{<i},Y) [ log p(x_i|X_{<i},Y) − λ r_C(X_{≤i},Y) − τ log π(x_i|X_{<i},Y) + γ V^π_C(X_{≤i},Y) ]`.
  Training minimizes the Path Consistency Loss on these (§2.2, p.4, bullet after Eq. 10). In our vocabulary this is a *distinct* fourth family: neither `rkl` (full-vocab, direct) nor `rkl_pg` — the reward is the teacher log-likelihood **of the sampled token only** (`log p(x_i|X_{<i})`, i.e. sampled-token teacher signal like `rkl_pg`) but it is fitted through a soft-Bellman residual with a learned value head instead of a score-function estimator.
- Phase-1 behavior cloning uses a *forward*-direction top-KL mask on teacher trajectories, `KL_i = D_KL(p(.|C^{(S)}_{<i}) ‖ π(.|C^{(T)}_{<i}))` (Eq. 16, §2.4.1, p.5) — note the letters: `p` = teacher first, so this term is `KL(teacher‖student)` on teacher-forced trajectories.
- The GKD baseline is trained with "reverse KL Divergence as the loss for the sampled trajectories" over one student and one teacher trajectory per task (§5.2, p.9).

## Off-policy correction

- **None, by design.** No importance ratio, no clipping, no mask. The paper's argument is that PCL does not need one: "In PCL, path consistency must be obeyed for trajectories from off-policy distributions as well without any modifications to the loss." (§2.1, p.3). Contrasted explicitly with the score-function route: "in policy-gradient optimization, several techniques need to be used to stabilize training such as mixing in teacher tokens and weighting them by importance weights and so forth" (§2.1, p.3).

## Results: async vs sync

None. Figures 1–2 and Table 1/§5.1 report only latency-vs-ROUGE-2 operating points for our-approach vs DistillSpec vs student/teacher endpoints. Staleness is never a variable.

## Explanation offered

Only the PCL-tolerates-off-policy-data argument, and only for teacher data:

- "The use of PCL allows us to natively utilize off-policy demonstrations from the teacher model, supplementing on-policy ones, to obtain the necessary training stability and convergence without any additional techniques." (§1, p.2)
- "This means that reverse KLD knowledge distillation can be optimized through minimizing the PCL loss over on- and off-policy demonstrations." (§2.1, p.2)
- "In PCL, path consistency must be obeyed for trajectories from off-policy distributions as well without any modifications to the loss. Hence, this offers us a natural means to incorporate teacher demonstrations in the training loop, without any changes to the formulation." (§2.1, p.3)

Mechanism, as far as it is given: path consistency is a *state-action-wise* identity satisfied by the optimal policy/value pair, so its residual can be evaluated on any trajectory distribution with full support, whereas the score-function estimator of the same reverse KL is an expectation under the current policy and needs correction when the behavior policy differs. The paper asserts this; it does not prove or measure it, and it never applies the argument to a stale student.

## Quotes

- Deciding sentence for the screen: "During RL finetuning, two trajectories are sampled for each task, one an off-policy trajectory according to the oracle policy, and one an on-policy trajectory." (§3.3, p.7) — "off-policy" = the oracle/teacher policy.
- "The use of PCL allows us to natively utilize off-policy demonstrations from the teacher model, supplementing on-policy ones …" (§1, p.2)
- "In PCL, path consistency must be obeyed for trajectories from off-policy distributions as well without any modifications to the loss." (§2.1, p.3)
- "We performed four gradient updates per batch of episode generation, with a data mix that contains samples from a replay buffer, as well as oracle trajectories and on-policy trajectories that were generated in the current batch." (§3.3, p.7)
- "For each batch of trajectories sampled, we performed four gradient updates." (§3.3, p.7, on the GKD/DistillSpec baseline)
- "Phase 1 training … is not on-policy, as the model learns based on teacher trajectories. Since the student model is of substantially lower capacity than the teacher model, the trajectories produced from it are unlikely to match those from the teacher model. This can cause downstream modeling fidelity issues such as due to exposure bias" (§2.4.2, p.5) — the paper's only on/off-policy failure argument, and it is the standard exposure-bias one.
- "This formulation has some practical advantages brought forth by the ability to seamlessly use off-policy and on-policy trajectories for model optimization." (§4, p.8)

## Not in the source

- No definition of the replay buffer (capacity, sampling policy, age, share of the batch) and no ablation with/without it.
- No control at one gradient update per rollout batch, so the 4-updates choice is unvalidated.
- No importance ratio or any other off-policy correction, and no measurement of the old-vs-current student mismatch it would correct.
- No claim, anywhere, that the PCL/off-policy tolerance extends to *stale student* trajectories; the argument is only ever made for teacher demonstrations.
- No LLM-scale or reasoning-benchmark evidence (FLAN-T5-SMALL/XL, translation + summarization only).
