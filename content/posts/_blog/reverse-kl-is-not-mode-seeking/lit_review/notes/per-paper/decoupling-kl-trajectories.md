# Zhao et al. 2026 — Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD in LLM Distillation

**arXiv 2605.16826v1** (16 May 2026). Anhao Zhao, Haoran Xin, Yingqi Fan, Junlong Tong, Wenjie Li, Xiaoyu Shen (Eastern Institute of Technology Ningbo; HK PolyU; SJTU; HKUST-Guangzhou). Sources: full v1 PDF (`curl https://arxiv.org/pdf/2605.16826`, pypdf extraction, 21 pages incl. Appendices A–H) and `https://arxiv.org/html/2605.16826` for equation LaTeX, fetched 2026-09-14. Code: `github.com/EIT-NLP/Decoupled-Distill`. Async-OPD sweep 2026-09-14.

```bibtex
@article{zhao2026decoupling,
  title         = {Decoupling {KL} and Trajectories: A Unified Perspective for {SFT}, {DAgger}, Offline {RL}, and {OPD} in {LLM} Distillation},
  author        = {Anhao Zhao and Haoran Xin and Yingqi Fan and Junlong Tong and Wenjie Li and Xiaoyu Shen},
  journal       = {arXiv preprint arXiv:2605.16826},
  year          = {2026},
  eprint        = {2605.16826},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.16826}
}
```

## Tier

- **A (partial, and the argument runs the *opposite* way from what the sweep is looking for).** Proposition 1 + Appendices E/F give an exact gradient identity for the **distribution-level** (full-vocabulary) token KL at a *fixed* prefix, in both directions, showing that **neither direction needs any importance-sampling correction for the prefix source**: the forward-KL gradient is an expectation under the fixed teacher `p_T`, the reverse-KL gradient is an expectation under the *current* student `q_theta`, and the prefix distribution enters only as the outer expectation. Verbatim (App. F): "Since the state distribution `d_T^t` is fixed with respect to theta, the gradient takes a policy-gradient form ... **No action-level importance sampling is required because the action expectation is already taken under `q_theta(·|s_t)` rather than under the teacher.**" The identity is stated for teacher prefixes, but nothing in it uses the fact that the prefix came from the teacher — it holds for any prefix distribution not depending on theta, which includes prefixes from an earlier student. So the paper argues **both** distribution-level directions tolerate off-policy prefixes; it never contrasts distribution-level RKL against sampled-token `rkl_pg`.
- **A (second strand): App. E formalizes stale-student-prefix aggregation for FORWARD KL.** The DAgger equivalence is stated as a sum over *all previous student policies* `theta_1 ... theta_k`: `\theta_{k+1}\in\arg\min_{\theta}\sum_{i=1}^{k}\sum_{t=1}^{T}\mathbb{E}_{s_{t}\sim d_{\theta_{i}}^{t}}\left[\mathrm{KL}\left(p_{T}(\cdot|s_{t})\|q_{\theta}(\cdot|s_{t})\right)\right].` This is literally "the student trains on prefixes sampled by earlier versions of itself", and the paper says it is exactly DAgger — i.e. a *justification* for replaying stale student prefixes under forward KL, inherited from DAgger's no-regret story. Crucially the paper then disclaims doing it (see Async mechanism).
- **D** — quotable statements defining off-policy prefixes, the offline-RL reading, and the aggregation caveat (see Quotes).
- **Not B / C** for staleness: the off-policy arm is **teacher** prefixes (SFT-style, explicitly out of scope by the sweep rule), not stale student prefixes. The teacher-vs-student prefix numbers are recorded below anyway because the caller asked about "KL direction under off-policy prefixes"; they show **no consistent direction ordering**.
- **Not E / F / G** — no async system, no replay buffer, no production report.
- Direction convention matches ours: reverse KL = KL(student `q_theta` ‖ teacher `p_T`) (Eq. 3, §2.1). Note the letter swap: **the student is `q_theta` and the teacher is `p_T`** here, the opposite of the usual p=model convention.

## Setting

- Student: Qwen3-0.6B-Base. Teachers: Qwen3-4B and Qwen3-8B ("We distill Qwen3-4B and Qwen3-8B into Qwen3-0.6B-Base [31], using the same model family to avoid cross-tokenizer artifacts in token-level KL computation.", §3; App. G.1 says Qwen3-4B-Base / Qwen3-8B-Base). Teacher **fixed** in all conditions.
- Domain: math reasoning; prompts from DeepScaleR. Benchmarks: GSM8K, MATH500, AMC23, AIME24, reported as Avg@k, Pass@k, mean response length; plus ARC-C/HellaSwag/MMLU/PIQA (Table 4). Held-out math eval every 30 steps, temp 0.6, 4096-token budget (App. G.5).
- Training (§3, App. G.3): bf16, lr 5e-7, batch size 32, constant LR with 5% warmup, 1000 steps, seed 42, two length regimes (128 and 4096 max generated tokens), colocated vLLM TP=1 for on-policy decoding. Single H100 94GB; "On-policy distillation took approximately 16 hours on average" (App. G.7). "We report training dynamics averaged over three runs." (§3).
- RL follow-up: GRPO, accuracy-only outcome reward, lr 1e-6, group 8, batch 32, 4096 max response, 1000 steps, beta=0 (no teacher KL penalty), one update per group (App. G.6).
- Framework: not named beyond vLLM for generation; code released.

## Async mechanism

**None is implemented.** The DAgger aggregation over earlier student policies is written down and then explicitly *not* used:

- What could be stale, in theory: prefixes from `d_{theta_i}^t` for `i < k`, i.e. rollouts from earlier student checkpoints, aggregated into the loss for `theta_{k+1}` (App. E, Ex68/Ex69).
- What the paper actually does (App. E, verbatim): "If the implementation does not aggregate all previous student-prefix data and instead trains only on the current student-induced distribution `d_{theta_k}`, then the method should be understood as an on-policy or DAgger-style variant rather than a literal implementation of the original DAgger algorithm." And App. G.2: "For student-prefix objectives, responses are instead sampled on-policy from the current student under the same prompt distribution."
- The *other* axis is off-policy in the SFT sense and IS precomputed once: "For teacher-prefix objectives, we generate offline teacher rollouts from this shared prompt set, compute the corresponding logits, and **reuse the resulting prefixes and cached logits throughout training**. In practice, these rollouts are generated with temperature 1.0, top-p 0.95, top-k 20, and batch size 128." (App. G.2). The teacher is fixed, so those cached logits never go stale — this is offline teacher data, not stale student data.
- Staleness measured: not at all. No buffer size, no lag in steps, no mixing ratio, no off-policyness metric. Range studied: a binary {teacher prefixes, student prefixes}, no interpolation and no partial-staleness condition.
- GRPO follow-up is strictly on-policy: "one policy update per generated group" (App. G.6) — so no multi-mini-batch reuse anywhere in the paper.

## Objective

**Full-vocabulary** both directions (this is the key estimator fact for us). App. G.4: "Our reverse-KL and forward-KL implementations compute the loss over the full teacher and student vocabulary distributions, rather than restricting matching to the teacher's top-k support. This is important because the behavior of KL-based objectives depends on the complete log-ratio geometry over the vocabulary."

- Sequence-KL decompositions that motivate the 2x2, Eq. (2) and Eq. (3) (§2.2):
  `\mathrm{KL}(p_{T}\|q_{\theta})(x)=\sum_{t=1}^{L}\mathbb{E}_{s_{t}\sim d_{T}^{t}}\!\left[\mathrm{KL}(p_{T}(\cdot\mid s_{t})\|q_{\theta}(\cdot\mid s_{t}))\right],`
  `\mathrm{KL}(q_{\theta}\|p_{T})(x)=\sum_{t=1}^{L}\mathbb{E}_{s_{t}\sim d_{S}^{t}}\!\left[\mathrm{KL}(q_{\theta}(\cdot\mid s_{t})\|p_{T}(\cdot\mid s_{t}))\right].`
- Proposition 1 (§2.3), the gradient identities that carry the whole argument:
  (i) `\nabla_{\theta}\mathrm{KL}\!\bigl(p_{T}(\cdot\mid s_{t})\,\|\,q_{\theta}(\cdot\mid s_{t})\bigr)=-\,\mathbb{E}_{y\sim p_{T}(\cdot\mid s_{t})}\!\left[\nabla_{\theta}\log q_{\theta}(y\mid s_{t})\right].`
  (ii) `\nabla_{\theta}\mathrm{KL}\!\bigl(q_{\theta}(\cdot\mid s_{t})\,\|\,p_{T}(\cdot\mid s_{t})\bigr)=\mathbb{E}_{y\sim q_{\theta}(\cdot\mid s_{t})}\!\left[\bigl(\log q_{\theta}(y\mid s_{t})-\log p_{T}(y\mid s_{t})\bigr)\nabla_{\theta}\log q_{\theta}(y\mid s_{t})\right].`
  with dense reward `r(s_{t},y)=\log p_{T}(y\mid s_{t})-\log q_{\theta}(y\mid s_{t}),` "treating r as scalar feedback, i.e., stopping gradients through r."
- In our vocabulary the four cells are: teacher-prefix FKL = off-policy SFT (`fkl`, full-vocab, teacher prefixes); student-prefix FKL = DAgger-style on-policy SFT (`fkl`, full-vocab, fresh student prefixes); teacher-prefix RKL = offline-RL-style (`rkl`, full-vocab, teacher prefixes); student-prefix RKL = OPD (`rkl`, full-vocab, fresh student prefixes). **None of the four is `rkl_pg`** — the sampled-token score-function estimator is never trained.
- Proposed method, KL mixing, Eq. (4) (§5.1):
  `\mathcal{L}_{\lambda}(s_{t})=\lambda\,\mathrm{KL}\!\left(q_{\theta}(\cdot\mid s_{t})\,\|\,p_{T}(\cdot\mid s_{t})\right)+(1-\lambda)\,\mathrm{KL}\!\left(p_{T}(\cdot\mid s_{t})\,\|\,q_{\theta}(\cdot\mid s_{t})\right),`
- **Ambiguity to flag:** App. D says "Our implementation uses the log-ratio reward form `r_{KL}(y,s_t)=\log p_T(y|s_t)-\log q_\theta(y|s_t)` ... Thus, our method directly optimizes the negative single-sample reverse-KL estimator as a reward. We do not use the squared-log approximation k2 or the non-negative unbiased estimator k3 in the reward." That reads as a k1/sampled-token statement, which contradicts App. G.4's full-vocabulary claim. The most consistent reading is that the *reward functional form* is the k1 log-ratio while the *expectation* is taken exactly over the vocabulary; the paper never resolves it.

## Off-policy correction

- **None, and the paper argues none is needed.** App. F, verbatim: "Thus, the teacher-prefix reverse-KL update can be interpreted as an offline policy-gradient update: states are drawn from the teacher-induced distribution, actions are sampled from the current student policy, and the reward is the log-ratio reward above. **No action-level importance sampling is required because the action expectation is already taken under `q_theta(·|s_t)` rather than under the teacher.**"
- The objective it licenses, App. F:
  `\mathcal{L}_{\mathrm{TP\text{-}RKL}}(\theta)=\sum_{t=1}^{T}\mathbb{E}_{s_{t}\sim d_{T}^{t}}\left[\mathrm{KL}\left(q_{\theta}(\cdot|s_{t})\|p_{T}(\cdot|s_{t})\right)\right].`
  `\mathcal{J}_{\mathrm{TP\text{-}RKL}}(\theta)=\sum_{t=1}^{T}\mathbb{E}_{s_{t}\sim d_{T}^{t}}\mathbb{E}_{y_{t}\sim q_{\theta}(\cdot|s_{t})}\left[\log p_{T}(y_{t}|s_{t})-\log q_{\theta}(y_{t}|s_{t})\right].`
- No ratio, no clipping, no truncation, no IcePop-style mask anywhere in the paper.
- **Our inference, not stated in the source** (flagged): the reason no correction is needed is that *both* cells here evaluate a **full-vocabulary** KL at the given prefix, so the only theta-dependent expectation is over the vocabulary and is computed exactly. A sampled-token `rkl_pg` estimator drawn from a *stale* student would break this, because the token would be drawn from `q_{theta_old}` while Prop. 1(ii) requires it under `q_theta`. The paper never makes this point and never tests it, but its identity is the cleanest statement of why the distribution-level form is prefix-provenance-agnostic and the sampled form is not.

## Results: async vs sync

**No stale-student condition, so none in the sweep's sense.** What exists is off-policy(teacher)-prefix vs on-policy(student)-prefix, crossed with KL direction. Endpoint Avg@k means over the four math benchmarks, computed from Tables 1 and 2 (arithmetic ours; the two averages below reproduce the paper's own "+1.80" and "+3.55" figures, §4.2/§4.3):

| Regime | Teacher | KL | Teacher prefixes (off-policy) | Student prefixes (on-policy) | Student − Teacher |
|---|---|---|---|---|---|
| 128 tok | Qwen3-4B | Forward | 30.87 | 29.04 | **−1.83** |
| 128 tok | Qwen3-4B | Reverse | 33.28 | 34.47 | **+1.19** |
| 128 tok | Qwen3-8B | Forward | 30.60 | 29.29 | **−1.31** |
| 128 tok | Qwen3-8B | Reverse | 32.30 | 34.48 | **+2.19** |
| 4096 tok | Qwen3-4B | Forward | 31.93 | 36.03 | **+4.10** |
| 4096 tok | Qwen3-4B | Reverse | 33.41 | 36.38 | **+2.97** |
| 4096 tok | Qwen3-8B | Forward | 31.54 | 35.48 | **+3.94** |
| 4096 tok | Qwen3-8B | Reverse | 33.42 | 36.60 | **+3.18** |

Reading: **no consistent ordering**. At 128 tokens, off-policy (teacher) prefixes are *better* for forward KL and *worse* for reverse KL; at 4096 tokens they are worse for both, and worse for forward KL by more. So this paper does not support "FKL tolerates off-policy prefixes better" as a general claim — at long horizons its own numbers point the other way. Paper-level statements: "student prefixes improve Avg@k by +1.80 points and Pass@k by +2.11 points on average; after 4096-token distillation, the gains increase to +3.55 Avg@k and +2.95 Pass@k" (§4.2).

Compute side-by-side (§4.2, Figure 6, 128-token reverse KL, Qwen3-4B teacher): "with cached logits, teacher prefixes reach roughly 38–40% MATH500 accuracy within 15–20k cumulative TFLOPs, while student prefixes require substantially more compute to reach the same range." FLOPs model (App. H.1): `C_off^cached ≈ 3F_s(L)` vs `C_on ≈ G_s(P,R) + F_t(L) + 3F_s(L)`; at L≈220, B=32, `C_off^cached ≈ 23.19 TFLOPs/step`. Wall-clock: on-policy distillation ~16 h, RL ~14 h on one H100 (App. G.7). No throughput numbers for a stale/async variant (none exists).

## Explanation offered

- Why the coupling exists at all (§1): "by the autoregressive chain rule, KL(pi_T ‖ pi_S) decomposes into token-level forward KL on teacher prefixes, while KL(pi_S ‖ pi_T) decomposes into token-level reverse KL on student prefixes. We argue that this coupling is not intrinsic at the token level: prefix source specifies where supervision is applied, while KL direction specifies how teacher and student token distributions are compared."
- Why prefix source is a *policy-regime* choice (§2.3): "The prefix-source axis determines the policy regime: teacher-induced prefixes are off-policy for the student, whereas student-induced prefixes are on-policy. The KL-direction axis determines the learning objective."
- Why off-policy prefixes need no correction: the App. F quote above. Mechanism: **student-weighting of the inner expectation**, i.e. the reverse-KL action expectation is already under the current student, so prefix provenance biases only the *state* distribution, never the estimator.
- Why stale student prefixes would be legitimate under forward KL (App. E): "Thus, with dataset aggregation, student-prefix forward KL exactly matches the supervised learning subproblem solved by DAgger, generalized from hard expert actions to soft teacher distributions." Caveat immediately after: "The original DAgger analysis is typically stated in terms of online no-regret learning with imitation losses ... Therefore, the correspondence above should be interpreted as an objective-level equivalence to the DAgger supervised learning subproblem, not as a direct reuse of the original DAgger performance bound."
- Why student prefixes help empirically (§4.2): "This supports the on-policy motivation: supervision is applied to the states the student actually visits, rather than only to teacher-induced states."
- Why KL direction dominates (§4.1): "This matches the mode-seeking geometry of reverse KL [11, 28], which concentrates probability mass on a narrower set of high-probability teacher continuations." and "reverse KL can produce a stronger pre-RL model, but its reduced entropy constrains exploration and can lead to accuracy degradation, whereas forward KL is a more reliable initialization for continued policy optimization."
- Decoupling claim that limits the transfer to our question (§4.2): "Unlike KL direction, however, prefix source has a much smaller effect on entropy and length dynamics, which are largely governed by the KL directions."

## Quotes

- Abstract: "We show that the prevailing paradigms, off-policy distillation and on-policy distillation (OPD), implicitly couple two orthogonal choices: prefix source and token-level KL direction."
- §1: "Prefix source determines the policy regime: teacher prefixes are off-policy for the student, as they are generated by another policy, whereas student prefixes are on-policy."
- §1: "teacher-prefix reverse KL realizes an offline-RL-style distillation objective [23]; and student-prefix reverse KL yields OPD, viewed as dense-reward on-policy RL [27]."
- §2.3: "The prefix-source axis determines the policy regime: teacher-induced prefixes are off-policy for the student, whereas student-induced prefixes are on-policy."
- §4.2: "Student prefixes improve quality but require online generation. Teacher-prefix training can instead reuse fixed rollouts and cached teacher logits."
- §4.2: "student prefixes improve the quality of the distilled initialization, but KL direction largely determines downstream RL trainability."
- App. E: "DAgger [32] is an iterative imitation-learning algorithm designed to address distribution shift caused by training only on expert-induced states. At iteration k, the learner policy pi_k is rolled out to collect states from the learner-induced state distribution. The expert policy pi* is then queried on these states to provide supervision. The collected state-action pairs are aggregated into a dataset, and the next learner policy is trained by minimizing a supervised imitation loss on the aggregated dataset."
- App. E: "where the sum over i corresponds to dataset aggregation across previous student rollouts."
- App. E: "If the implementation does not aggregate all previous student-prefix data and instead trains only on the current student-induced distribution d_{theta_k}, then the method should be understood as an on-policy or DAgger-style variant rather than a literal implementation of the original DAgger algorithm."
- App. F: "In standard offline RL, a behavior policy collects a fixed dataset, and a target policy is optimized using this fixed distribution without further environment interaction. In our setting, the teacher plays the role of the behavior policy: prefixes are sampled from the teacher-induced distribution d_T^t, while the student q_theta is the target policy optimized on these prefixes."
- App. F: "No action-level importance sampling is required because the action expectation is already taken under q_theta(·|s_t) rather than under the teacher."
- App. F: "This interpretation is only an objective-level analogy to offline RL. Unlike standard offline RL methods such as CQL [20] or IQL [19], this objective does not learn a value function, does not perform Bellman backups, and does not optimize the closed-loop trajectory distribution induced by the student."
- App. G.2: "For teacher-prefix objectives, we generate offline teacher rollouts from this shared prompt set, compute the corresponding logits, and reuse the resulting prefixes and cached logits throughout training."
- App. G.6: "one policy update per generated group."
- App. D: "This is Schulman's k1 estimator for the reverse KL under samples from q_theta. It is unbiased ... but it is not guaranteed to be non-negative for each individual sample."

## Not in the source

- No stale-student-prefix experiment. The DAgger aggregation (App. E, Ex68/Ex69) is written down as theory and explicitly not implemented; there is no replay buffer, no lag parameter, no mixed fresh/stale batch, no async generator.
- No sampled-token `rkl_pg` arm, so the paper cannot and does not compare distribution-level RKL against the sampled-token estimator under any prefix source. Its "no importance sampling required" argument is therefore silent on exactly the contrast the sweep is after.
- No analysis of *why* the prefix-source effect flips sign between the 128- and 4096-token regimes (the sign flip is visible in its own Tables 1–2 but is never discussed; the paper reports only pooled averages).
- The App. D k1 statement vs the App. G.4 full-vocabulary statement are never reconciled.
- No variance or bias analysis of any estimator under prefix shift; no gradient-norm or drift diagnostics.
- Training dynamics are "averaged over three runs" (§3) but endpoint Tables 1–4 carry no seeds, error bars, or run counts.
- AIME24 Avg@k is 0.00–2.67 across the board (30 problems, tiny student), so that column carries no signal.
