# Lu & Thinking Machines Lab 2025 — On-Policy Distillation

**Blog post** (27 Oct 2025; carries an in-line "Update (June 2026)" on retired Tinker models, so the page read here is the June-2026 revision). Kevin Lu "in collaboration with others at Thinking Machines" (Thinking Machines Lab). DOI 10.64434/tml.20251026. Sources: the blog page `https://thinkingmachines.ai/blog/on-policy-distillation/` (HTML fetched 2026-09-14, full text; section anchors below are the page's own `[[...]]` heading ids); the Tinker cookbook `https://github.com/thinking-machines-lab/tinker-cookbook` at commit `485726f` (shallow clone 2026-09-14), files `tinker_cookbook/recipes/distillation/*` and `tinker_cookbook/distillation/*`.

```bibtex
@misc{lu2025onpolicydistillation,
  author       = {Kevin Lu and {Thinking Machines Lab}},
  title        = {On-Policy Distillation},
  howpublished = {Thinking Machines Lab: Connectionism},
  url          = {https://thinkingmachines.ai/blog/on-policy-distillation/},
  year         = {2025}
}
```

## Tier

- **D** — quotable statements about on-policy vs off-policy/stale data, including an explicit mechanism for why training on one's *own earlier* samples degrades over time (§`[[on-policy-learning-as-a-tool-for-continual-learning]]`). This is the single most-cited framing of OPD and the one our trainer follows, so its position on staleness matters even though it runs no async experiment.
- Not **A** — the finite-batch-drift argument is about *SFT / forward KL on one's own samples*, not about whether distribution-level reverse KL tolerates staleness better than sampled-token `rkl_pg`. It never contrasts the two estimator families.
- Not **B**/**C** — no async-vs-sync numbers anywhere; every experiment is fully synchronous.
- Not **E**/**F**/**G** — async OPD is neither used nor demonstrated. The shipped recipe is sync-only (see "Async mechanism").

## Setting

- Reasoning: student Qwen3-8B-Base (SFT'd on 400k OpenThoughts-3 prompts), teacher Qwen3-32B for the FLOPs accounting but **actually Qwen3-8B**: "We actually use Qwen3-8B as a teacher, as it performs slightly better. However, for the purpose of comparing compute, we can still measure the FLOPs of the 32B model." (§`[[on-policy-distillation]]`, footnote). Benchmark AIME'24. The June-2026 update moves the shipped recipe to student Qwen3.5-9B-Base ← teacher Qwen3.5-9B.
- Self-distillation / dense-supervision study (§`[[dense-supervision-greatly-improves-compute-efficiency]]`): student Qwen3-8B-Base, teacher = the *same* base model after LoRA-128 RL on DeepMath. Same-checkpoint RL teacher, as in the one-shot-OPD recipe.
- Personalization: student Qwen3-8B after a document mid-train, teacher Qwen3-8B (the pre-mid-train checkpoint), Tulu3 prompts. Evals: internal QA + IF-eval.
- Teacher fixed: **yes** in all experiments (a frozen sampling client; see code below).
- Framework/code: Tinker + tinker-cookbook. Recipe `tinker_cookbook/recipes/distillation/on_policy_distillation.py`, trainer `tinker_cookbook/distillation/train_on_policy.py`.

## Async mechanism

**None — the method as described and as shipped is strictly synchronous.** Concretely, from the cookbook at `485726f`:

- The distillation trainer has exactly one execution mode: `async def do_sync_training(...)` with the docstring `"""Implements fully synchronous on-policy training"""` (`tinker_cookbook/distillation/train_on_policy.py:283`), and it is the only one called (`:490`). Each iteration samples rollouts from the current sampling client, grades them, takes the optimizer step, then publishes a new sampling client — no overlap.
- **One optimizer step per rollout batch by default.** `num_substeps: int = 1` (`recipes/distillation/on_policy_distillation.py:77`; `distillation/train_on_policy.py:156`, commented "Number of optimizer steps per training iteration. Useful for very large batch sizes."). Crucially `num_substeps` **splits** the batch rather than re-using it: `batches = split_list(data_D, min(num_substeps, len(data_D)))` and then one `forward_backward` + `optim_step` per chunk (`tinker_cookbook/rl/train.py:329,338-341,346-351`). So each sampled token is consumed by exactly one gradient step; there is no multi-epoch reuse. The *only* staleness in the shipped recipe is that with `num_substeps = S > 1`, chunk *j* is trained against weights already updated *j−1* times since sampling — i.e. lag ∈ {0,…,S−1} optimizer steps, and lag ≡ 0 at the default S = 1.
- The sibling **RL** trainer *does* ship an async off-policy mode that the distillation trainer never imports: `AsyncConfig` (`rl/train.py:395-415`) — "In async mode, sampling and training run concurrently. Trajectory groups generated from a sampler that is too many steps behind the current training step are discarded (or requeued) to limit off-policy staleness." with `max_steps_off_policy` = "Maximum number of training steps a sample can lag behind the current step before being considered stale". Staleness is measured in optimizer steps via `WrappedTrajectoryGroup.sampling_client_step` ("The training step at which the sampling client was created for this rollout", `rl/train.py:876-888`) and enforced by `filter_stale_trajectory_group` (`rl/train.py:1088-1118`): `if i_batch - wrapped_trajectory_group.sampling_client_step > config.async_config.max_steps_off_policy:` → requeue or discard. **No default value for `max_steps_off_policy` is given** (it is a required field), and no experiment in the blog uses it.
- Data *reuse* in the blog is **prompt reuse, not rollout reuse**: "We train on this prompt for 20 consecutive steps, each with a batch of 256 rollouts" (§`[[distillation-can-effectively-reuse-training-data-for-data-efficiency]]`) — 256 *freshly sampled* rollouts per step from the same prompt. This is not staleness.
- Partial rollouts are allowed but not stale: "Since it doesn't require a rollout to finish sampling to calculate the reward, we can use shorter or partial rollouts for training." (§`[[loss-function-reverse-kl]]`).

## Objective

`rkl_pg` — sampled-token score-function estimator of per-token reverse KL, discount 0. Verbatim (§`[[loss-function-reverse-kl]]`; the page renders it without LaTeX):

```
KL(πθ​∣∣πteacher​) = E x∼πθ​ [ logπθ​(xt+1​∣x1..t​) − logπteacher​(xt+1​∣x1..t​) ]
```

"we choose the per-token reverse KL — the divergence between the student's (πθ) and teacher's (πteacher) distribution for each token conditioned on the same prior trajectory". Direction matches our convention (reverse KL = KL(student ‖ teacher)).

Discount: "For simplicity, we use a discount factor of zero: at any given timestep, the student only optimizes the immediate next token" with footnote "Although more mathematically correct, we do not find discount factors > 0 to improve performance in practice, so we choose zero for simplicity."

Explicitly **not** truncated-support: "In this post, we do not consider logit (top-k) distillation in any of our experiments, which could be used to further improve compute efficiency." (§`[[pseudocode]]`).

Implementation (§`[[pseudocode]]`, verbatim):

```python
teacher_logprobs = teacher_client.compute_logprobs(trajectories)
reverse_kl = sampled_logprobs - teacher_logprobs
trajectories["advantages"] = -reverse_kl
training_client.forward_backward(trajectories, loss_fn="importance_sampling")
```

matching `incorporate_kl_penalty` in `train_on_policy.py:54-108`: `reverse_kl = (sampled_logprobs - teacher_logprobs) * mask`, then `kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]` added into `advantages` (defaults `kl_penalty_coef = 1.0`, `kl_discount_factor = 0.0`).

## Off-policy correction

Per-token importance ratio, **unclipped**, against the logprobs recorded at sampling time — but it is a sampler/learner-mismatch correction, not a staleness correction, and at `num_substeps = 1` the ratio is ≈ 1 by construction.

- `loss_fn: LossFnType = "importance_sampling"` is the default (`on_policy_distillation.py:81`, `train_on_policy.py:151`). The blog: "During sampling, RL already computes the student's logprobs logπθ(x) for use as part of the importance-sampling loss." and "We set the per-token advantage to the negative reverse KL, and call the RL importance-sampling loss function to perform the training update on the student model." (§`[[pseudocode]]`).
- What the loss does, from the cookbook tutorials: "**`importance_sampling` loss** handles the off-policy correction between the sampling policy and the current training policy, using the logprobs recorded during sampling." (`tutorials/104_first_rl.py:359`). Properties table (`tutorials/202_loss_functions.py:308-310`): `importance_sampling` → "RL with on-policy or near-on-policy data | Corrects for sampler/learner mismatch; **unbounded ratio**"; `ppo` → "**RL with multiple gradient steps per rollout** | Clips the IS ratio to prevent large updates"; `cispo` → "Clips the ratio but applies it as a weight on log-prob; sometimes more stable".
- So the cookbook's own guidance is that the moment you take multiple gradient steps per rollout batch you should leave the OPD default (`importance_sampling`) for `ppo`/`cispo`. No clipping value, no IcePop-style mask, no truncation is used or suggested anywhere in the distillation recipes.
- No equation for the ratio is given: the loss is implemented service-side in Tinker, not in the open cookbook.

## Results: async vs sync

**None.** Every reported number is fully synchronous. The comparisons the blog does report (for context, all sync):

| Method | AIME'24 | GPQA-Diamond | GPU hours | Source |
|---|---|---|---|---|
| Off-policy distillation (SFT init) | 55.0% | 55.6% | unreported | §`[[reinforcement-learning]]`, from Qwen3 Table 21 |
| + Reinforcement learning | 67.6% | 61.3% | 17,920 | same |
| + On-policy distillation | 74.4% | 63.3% | 1,800 | same |

| Method | AIME'24 | Teacher FLOPs | Student FLOPs | CE vs SFT-2M | Source |
|---|---|---|---|---|---|
| Init: SFT-400K | 60% | 8.5 × 10²⁰ | 3.8 × 10²⁰ | – | §`[[on-policy-distillation]]` |
| SFT-2M (extrapolated) | ~70% | 3.4 × 10²¹ | 1.5 × 10²¹ | 1× | same |
| Reinforcement learning | 68% | – | – | ≈1× | same |
| On-policy distillation | 70% | 8.4 × 10¹⁹ | 8.2 × 10¹⁹ | 9–30× | same |

Personalization (§`[[on-policy-distillation-recovers-post-training-behavior]]`): Qwen3-8B 18% internal-QA / 85% IF-eval → + midtrain(100%) 43/45 → + midtrain(70%) 36/79 → + midtrain(70%) + distill 41/83.

Self-distillation (§`[[dense-supervision-greatly-improves-compute-efficiency]]`): "distillation reaches the teacher's level of performance approximately 7-10x faster than RL with matched model architecture (LoRA rank 128). The reverse KL decreases to near-zero and the AIME score is recovered in under 10 gradient steps, while RL took 70 steps to reach that level."

## Explanation offered

The blog's one mechanistic statement about training on one's own *earlier* samples — the closest thing in the source to an argument about staleness (§`[[on-policy-learning-as-a-tool-for-continual-learning]]`, about SFT on Qwen3-32B's own KL=0 samples):

> "A possible explanation for this is that while KL divergence is 0 in expectation, every finite batch will exhibit a slightly different distribution in practice. Training on these finite batches causes a non-zero gradient update, which then diverges the updated model's policy from that of its original state. This process turns training on one's own samples into off-policy training over time, which causes the same error accumulation and divergence over long sequences that happens with off-policy training."

> "On-policy distillation always stays on-policy, and since the teacher stays fixed, the student converges on the teacher's desirable behavior, without regressing in the self-distillation setting as SFT does. This makes on-policy distillation a very promising tool for continual learning."

Mechanism identified: **distribution shift / compounding error**, accumulated one gradient step at a time. Note the asymmetry the blog is drawing — it is contrasting *SFT (cross-entropy on stored samples)* with *OPD (a fresh teacher grade on fresh rollouts)*. The teacher being fixed is load-bearing in the second quote: what makes OPD self-correcting is that the target does not move, not that the data is fresh. The blog never separates those two factors, and never asks whether the OPD loss would survive if only the *data* went stale.

The general off-policy failure mode (§ intro, on SFT/off-policy distillation):

> "The drawback of off-policy training is that the student learns in contexts frequented by the teachers, not ones the student itself will often find itself in. This can cause compounding error: if the student makes an early mistake that the teacher never makes, it finds itself diverging ever farther from the states it observed in training. This problem becomes particularly acute when we care about the student's performance on long sequences. To avoid this divergence, the student must learn to recover from its own mistakes."

On why the objective tolerates prompt reuse (§`[[distillation-can-effectively-reuse-training-data-for-data-efficiency]]`) — relevant because it is a *distribution-level* argument about the target, not about the data:

> "With RL, training multiple epochs on the same prompt often leads to simple memorization of the final answer, especially with large models. In contrast, on-policy distillation learns to approximate the teacher's complete distribution by minimizing reverse KL, rather than memorizing a single answer. This allows us to train many samples from the same prompt."

On batch size and support (§`[[dense-supervision-greatly-improves-compute-efficiency]]`, footnote): "ie: when the teacher policy is within the support of the student policy. When this is not true, as in 'Distillation for reasoning', we require a significantly larger batch size."

And (§`[[pseudocode]]`, footnote) on support/direction, which bears on where staleness would bite: "SFT, using forwards KL, adds support for new tokens. Reverse-KL methods can then perform mode seeking within the initialization's support."

## Quotes

Every statement in the source bearing on on-policy vs off-policy/stale data:

- "On-policy training samples rollouts from the student model itself, and assigns them some reward. / Off-policy training relies on target outputs from some external source that the student learns to imitate." (§ intro). Note: the blog's "off-policy" = *teacher/external* data throughout, never "the student's own earlier rollouts" — except in the continual-learning passage below.
- "The strength of on-policy training is that by training on samples from itself, the student learns to avoid mistakes in a more direct way." (§ intro).
- "The core idea of on-policy distillation is to sample trajectories from the student model and use a high-performing teacher to grade each token of each trajectory." (§`[[on-policy-distillation--best-of-both-worlds]]`).
- Method table (§`[[on-policy-distillation--best-of-both-worlds]]`): SFT = off-policy sampling / dense reward; RL = on-policy / sparse; On-policy distillation = on-policy / dense.
- "Sample trajectories. We sample rollouts from the student exactly as we would in RL. During sampling, RL already computes the student's logprobs logπθ(x) for use as part of the importance-sampling loss." (§`[[pseudocode]]`).
- "Our implementation can actually be a one-line change on top of RL implementations which use KL regularization: we merely swap out the regularizer model." (§`[[pseudocode]]`, footnote).
- "Since it doesn't require a rollout to finish sampling to calculate the reward, we can use shorter or partial rollouts for training. Querying the teacher's log probabilities also requires just a single forward pass from the larger model, while the trajectories are generated by the smaller and cheaper student." (§`[[loss-function-reverse-kl]]`).
- "This process turns training on one's own samples into off-policy training over time, which causes the same error accumulation and divergence over long sequences that happens with off-policy training." (§`[[on-policy-learning-as-a-tool-for-continual-learning]]`).
- "On-policy distillation always stays on-policy, and since the teacher stays fixed, the student converges on the teacher's desirable behavior" (§`[[on-policy-learning-as-a-tool-for-continual-learning]]`).
- "Similarly to above, we build a dataset by taking Tulu3 prompts and sampling from Qwen3-32B at temperature = 1.0 and no further modifications. This dataset therefore has a KL of zero against Qwen3-32B." with footnote "The importance of 'truly on-policy' KL=0 data was also explored in our previous post, Defeating Nondeterminism in LLM Inference." (§`[[on-policy-learning-as-a-tool-for-continual-learning]]`).
- "We see that any practical learning rate that's greater than zero leads to a degradation of performance on the instruction-following evaluation!" (§`[[on-policy-learning-as-a-tool-for-continual-learning]]`).
- "This 'on-policy' background data sampled by Qwen3-8B acts as a forwards KL regularizer... We find sampling from Qwen3-8B is better than Qwen3-32B for preserving chat capabilities throughout mid-training, highlighting the sensitivity of the data source" (§`[[training-on-new-knowledge-degrades-learned-behavior]]`).
- "We train on the same prompt for multiple steps in sequential fashion, which normally leads to overfitting." (§`[[distillation-can-effectively-reuse-training-data-for-data-efficiency]]`) — prompts, not rollouts.
- "Our default configuration (also used in the personalization experiments) runs on-policy distillation with 64 prompts / batch, and 4 samples / prompt. All methods shown are trained with 256 samples / batch." (§`[[distillation-can-effectively-reuse-training-data-for-data-efficiency]]`, figure caption).
- Code, on the RL side only: "In async mode, sampling and training run concurrently. Trajectory groups generated from a sampler that is too many steps behind the current training step are discarded (or requeued) to limit off-policy staleness." (`rl/train.py:399-401`).

## Not in the source

- **No async/stale experiment of any kind.** No lag sweep, no replay buffer, no pipelined generator, no measurement of "off-policyness" in the distillation path. The `AsyncConfig`/`max_steps_off_policy` machinery exists in the cookbook's RL trainer and is simply not wired into `distillation/train_on_policy.py`.
- **No throughput/wall-clock argument for async.** All efficiency claims are FLOPs-based (9×/18×/30× vs SFT), and the blog explicitly notes FLOPs "overestimates the practical cost of computing log-probs" — i.e. the generator/teacher-forward pipeline bubble that motivates async is acknowledged as mis-measured but never addressed.
- **No statement that the reverse-KL loss is valid off-policy.** The `importance_sampling` loss makes it *formally* valid for a known behaviour policy, and the blog relies on that machinery, but the blog never claims staleness-tolerance and never separates "the teacher is fixed" from "the data is fresh" as the reason OPD does not drift.
- No clipping/truncation values, no guidance on `num_substeps > 1` for distillation, and no ratio equation (the loss is service-side and closed).
- No discussion of whether the blog's finite-batch-drift argument (which predicts that stale *self* data is harmful under cross-entropy) transfers to `rkl_pg`, where the target is a fixed teacher rather than the stored sample.
