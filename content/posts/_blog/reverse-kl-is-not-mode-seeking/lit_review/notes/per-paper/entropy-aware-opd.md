# Jin et al. 2026 — Entropy-Aware On-Policy Distillation of Language Models (EOPD)

**arXiv 2603.07079v1** (7 Mar 2026; only version as of 2026-09-14). Woogyeol Jin, Taywon Min, Kimin Lee (KAIST); Yongjin Yang; Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo (IBM Research). Published at ICML 2026 ("Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026", p. 1 footer). Sources: PDF `https://arxiv.org/pdf/2603.07079` (curl + pypdf text, 18 pages, 2026-09-14); arXiv abs page (2026-09-14). No code URL in the paper.

```bibtex
@inproceedings{jin2026entropy,
  title     = {Entropy-Aware On-Policy Distillation of Language Models},
  author    = {Woogyeol Jin and Taywon Min and Yongjin Yang and others},
  booktitle = {International Conference on Machine Learning},
  year      = {2026},
  eprint    = {2603.07079},
  archivePrefix = {arXiv}
}
```

(key reused from `references.bib:710`, unchanged)

## Tier

- **D** — the paper states, in hyperparameters and in Algorithm 1, exactly how many gradient updates are taken per rollout batch and what corrects for the resulting off-policyness: "we use a batch size of `B = 128` and a mini-batch size of `B_mini = 32`, which results in four gradient update steps per training iteration" (App. A, "On-policy Training"), and §2.3 "We sample trajectories using a behavior policy `π_θold` instantiated from the student policy `π_θ`". Directly quotable for "how many updates per rollout batch does published OPD actually use".
- **G** — mini-batched / off-policy-corrected OPD is *used* (it is the paper's whole training loop, the `OPD` baseline included) but is never the object of study: the paper is about teacher entropy and the reverse-vs-forward KL mix. No async experiment, no staleness sweep.
- Not **A** — the paper contains the *structural* contrast we care about (sampled-token RKL carries an importance ratio + clip; distribution-level FKL carries neither, Eq. 8 vs Eq. 10) but never argues, or even remarks, that the distribution-level term is the one that tolerates the 3-step-stale mini-batches. The argument is there to be made; the paper does not make it. See "Explanation offered".
- Not **B**/**C** — no sync (1-update) vs mini-batched side-by-side. Every arm in every table uses the same `B/B_mini = 128/32`.
- Not **E** (academic, 4×A100, App. A), not **F** (async is not the point).

## Setting

- Students (§5.1): Qwen3-0.6B-Base, Qwen3-1.7B-Base, Qwen3-4B-Base. Teacher: **Qwen3-8B, non-thinking mode** — "We use Qwen3-8B as the teacher model, without enabling thinking mode."
- Teacher fixed: **yes**. Only the student is updated (Alg. 1 line 14 updates θ alone); the teacher is queried for log-probs, entropy and top-k (Alg. 1 line 9). In the toy study the fixity is explicit: "The teacher is fixed throughout training" (§3.2, toy setup).
- Data: MATH (Hendrycks et al. 2021) for the 0.6B/1.7B students; DAPO (Yu et al. 2025) for the 4B student (§5.1).
- Benchmarks (§5.1): MATH500, AIME24, AIME25, AMC23, Minerva, OlympiadBench; Avg@8 and Pass@8, 8 samples/question, temperature 1.0, top-p 0.8, max response 8192. OOD: GPQA-Diamond (App. C).
- Framework: **verl** — "On-policy distillation and GRPO are implemented using the verl (Sheng et al., 2024) framework" (§5.1 and App. A). The off-policy KD baseline uses DistillKit.
- Hardware/cost: "All experiments were conducted on a 4×A100 80GB GPU setup"; average step time 47.8 s, of which "most of the computation time is spent on student on-policy generation (∼37.7 seconds)" (App. A).
- Headline result (§2 contributions, Tables 2): EOPD over OPD, averaged over six math benchmarks — +1.16 Avg@8 / +1.37 Pass@8 (0.6B), +0.99 / +2.39 (1.7B), +1.80 / +5.05 (4B).

## Async mechanism

Not a pipelined trainer/generator and not a replay buffer across iterations — but **the student does train on rollouts sampled by an earlier version of itself, up to three optimizer steps stale, by construction of the inner loop.**

- What is stale: the **behaviour student** `π_θold`, snapshotted at the top of each training iteration and used for all generation in that iteration, versus the **current student** `π_θ` after 0, 1, 2 or 3 mini-batch updates. Alg. 1: line 2 `Set π_θold ← π_θ`; line 6 `Sample x = (x_1,…,x_|x|) ∼ π_θold(·|q)`; line 11 `for each mini-batch gradient step do`; line 14 `Update parameters: θ ← θ − η ∇_θ L(θ)`. The teacher quantities `Q_t` are also cached at rollout time (line 9–10), but the teacher is fixed so they do not go stale.
- The buffer is named but is intra-iteration only: "Rollout buffer `R ← ∅`" (Alg. 1 line 4), refilled every iteration; the mini-batches are drawn from it, "Sample a mini-batch of prompts `B_mini ⊂ R`" (Alg. 1 line 12).
- How staleness is measured: **not measured at all.** The paper reports no lag statistic, no `π_θ`/`π_θold` divergence, no clip-fraction, no off-policyness metric. The only handle is the ratio `r_t` itself (Eq. 8), and its clip rate is never plotted.
- Range studied: **a single point.** `B = 128`, `B_mini = 32` ⇒ 4 updates per rollout batch, i.e. staleness ∈ {0,1,2,3} optimizer steps within an iteration, for every run in the paper (Table 9; identical for OPD, EOPD and GRPO arms). No sweep over `B_mini`, and no `B_mini = B` (strictly on-policy) control.
- Note the *prompt* epochs are separate and larger: "Training epoch 3 (MATH), 2 (DAPO)" (Table 9) — prompts are revisited, but each visit draws fresh rollouts.

## Objective

Mixed: sampled-token reverse KL (`rkl_pg`, PPO-clipped) **plus** a teacher-top-k distribution-level forward KL (`fkl`, truncated), gated on teacher entropy.

Target objective, §2.2 Eq. (4)–(5), reverse KL = KL(student ‖ teacher) (matches our convention):

```
E_{q∼D} [ E_{x∼π_θ(·|q)} [ (1/|x|) Σ_{t=1}^{|x|} L^RKL_t(θ; c_t) ] ]        (4)
L^RKL_t(θ; c_t) = KL(π_θ(·|c_t) ‖ π_te(·|c_t))                             (5)
```

Per-token reward / advantage, §2.2 Eq. (6) and its `π_θold` substitution in §2.3:

```
A_t = log π_te(x_t|c_t) − log π_θ(x_t|c_t)                                  (6)
Â_t = log π_te(x_t|c_t) − log π_θold(x_t|c_t)          (§2.3, unnumbered)
```

— i.e. **the advantage is frozen at rollout time against the behaviour policy**, not recomputed against the current student at each of the four mini-batch steps. Verbatim: "define a per-token advantage `Â_t = log π_te(x_t|c_t) − log π_θold(x_t|c_t)`, substituting the behavior policy in (6)".

EOPD objective, §4 Eq. (9):

```
L^EOPD_t(θ; c_t) = L^OPD_t(θ; c_t) + α · I( H^te_t > τ ) · L^FKL_t(θ; c_t)   (9)
```

with `H^te_t = − Σ_{x∈V} π_te(x|c_t) log π_te(x|c_t)` over the **full** vocabulary, and τ = 0.8, α = 1.0 (§4). The forward-KL term, §4 Eq. (10), teacher-top-k renormalised, k = 16:

```
L^FKL_t(θ; c_t) ≈ Σ_{x∈S^k_t} π̃_te(x|c_t) log( π̃_te(x|c_t) / π_θ(x|c_t) )   (10)
π̃_te(x|c_t) = π_te(x|c_t) / Σ_{x'∈S^k_t} π_te(x'|c_t),  x ∈ S^k_t
```

Whose top-k: **teacher's**, stated explicitly — "an expectation computed over the teacher's top-k tokens `S^k_t`" and "We restrict to the teacher's top-k tokens to ensure that the student does not have to learn from the low-probability tails of the teacher" (§4). `k = 16` (Table 9; justified in App. F).

The two-estimator split is stated in one sentence (§4, after Eq. 9): "For the `L^OPD_t(θ; c_t)` term in (9), the expectation in (7) is approximated by a single-sample Monte Carlo estimate as discussed previously. In effect, the clipped reverse KL loss (8) is evaluated at the token `x_t` sampled during the rollouts. The forward KL `L^FKL_t(θ; c_t)` is approximated **not by sampling**, but as an expectation computed over the teacher's top-k tokens `S^k_t`." (emphasis in original by italics on "not by sampling").

FKL is applied to ~15–20% of tokens on average at τ = 0.8 (§5.7 footnote 2, citing Fig. 9).

## Off-policy correction

**Asymmetric — and this is the load-bearing detail for our question.**

- The **sampled-token reverse-KL term carries a per-token importance ratio and a PPO clip**, §2.3 Eq. (7)–(8):

```
L^OPD_t(θ; c_t) = E_{x_t∼π_θold(·|c_t)} [ Ã_t ]                              (7)
Ã_t = max( −r_t Â_t , −clip(r_t, 1−ε, 1+ε) Â_t )                            (8)
r_t = π_θ(x_t|c_t) / π_θold(x_t|c_t)
```

  verbatim gloss: "Here, `r_t = π_θ(x_t|c_t)/π_θold(x_t|c_t)` **corrects for sampling under `π_θold`**, while clipping limits overly large policy updates." (§2.3). Per-token, not per-sequence. Dual-clip / max-form (the `max(−r Â, −clip(r) Â)` shape).
- The **distribution-level forward-KL term carries no ratio, no clip, no mask**: Eq. (10) contains only `π̃_te` and `π_θ(x|c_t)`. It is evaluated at the same stale context `c_t` but needs no correction, because it is an expectation under the *teacher* at that context, not under the sampling student — the only thing `π_θold` contributed is the state `c_t`. The paper never remarks on this.
- **ε is never given a numerical value** — not in §2.3, not in Table 9, not in App. A. (verl's default is 0.2, but the paper does not say so.) Recorded in "Not in the source".
- No IcePop-style token mask, no truncated-IS (TIS), no sequence-level ratio, no KL penalty to a reference: "KL regularization" never appears; Table 9 lists no KL coefficient.

## Results: async vs sync

**None.** There is no 1-update-per-batch control anywhere in the paper; `B_mini = 32` is fixed across all methods and all tables, so the mini-batching is a constant, not a variable. The closest thing to an on-/off-policy data axis is §5.6, and its "off-policy" data is *teacher*-generated, not stale-student:

| §5.6 GKD sweep (Qwen3-1.7B-Base, Table 5) | λ=0.5 | λ=0.8 | λ=1.0 (fully on-policy) | EOPD |
|---|---|---|---|---|
| MATH500 Avg@8 | 64.53 | 65.28 | 67.20 | 68.73 |
| MATH500 Pass@8 | 84.40 | 84.20 | 85.00 | 87.60 |
| AMC23 Avg@8 | 35.62 | 37.50 | 38.44 | 41.88 |
| AIME24 Avg@8 | 5.83 | 6.67 | 9.17 | 10.42 |
| AIME25 Avg@8 | 4.58 | 3.75 | 5.00 | 5.83 |

"We observe that the fully on-policy setting (λ = 1.0) achieves the best performance among different λ values, consistent with GKD's observation that on-policy rollouts are particularly important for reasoning tasks." (§5.6). **Out of scope as async evidence** (λ mixes in *teacher* sequences, not earlier-student rollouts), but it is the only data-freshness knob the paper turns, and it turns the same way our question does.

Estimator comparison without any staleness axis, §5.7 Table 6 (Qwen3-1.7B-Base) — `OPD` = sampled-token RKL, `OPD-FKL` = "optimizes the policy using only forward KL":

| | OPD-FKL | OPD | EOPD |
|---|---|---|---|
| MATH500 Avg@8 | 67.20 | 67.76 | 68.73 |
| AMC23 Avg@8 | 38.44 | 39.06 | 41.88 |
| AIME24 Avg@8 | 9.17 | 8.33 | 10.42 |
| AIME25 Avg@8 | 5.00 | 6.25 | 5.83 |

"EOPD outperforms both OPD and OPD-FKL across most benchmarks, while OPD generally outperforms OPD-FKL." (Table 6 caption).

## Explanation offered

Nothing about staleness. The paper's mechanism story is about **teacher entropy**, and it is an argument about the *variance of the sampled-token reverse-KL estimator* — adjacent to, but not the same as, the staleness argument:

- §3.2 toy experiment (V = 80 categorical, five planted modes, student restricted to its own top-10, teacher fixed, reward `r(x) = log P_te(x) − log P_S10(x)` — i.e. a pure `rkl_pg` sampled-token update): "under the low-entropy teacher (Scenario A), the top-10 change rate decreases steadily and top-1 changes are rare. In contrast, under the high-entropy teacher (Scenario B), training exhibits persistent instability: the top-1 index changes frequently and the top-10 set fails to converge. **These results demonstrate that reverse-KL rewards provide unstable learning signals when the teacher distribution is uncertain. This motivates training objectives that more directly transmit the teacher's distributional structure to the student.**" (§3.2, Results). Top-1 change count 7.3±1.6 (low entropy) vs 84.0±16.7 (high entropy), Table 1.
- §2 contributions: "through a controlled toy experiment (§3.2), we demonstrate that the reverse KL objective provides unstable gradient signals when the teacher is uncertain, preventing proper convergence."
- §1/abstract: "the mode-seeking property of reverse KL reduces generation diversity and yields unstable learning signals when the teacher distribution has high entropy."

So the paper's own fix for an unstable sampled-token estimator **is to add a distribution-level term** ("more directly transmit the teacher's distributional structure"). That is the same move our staleness question asks about, arrived at from a different premise (teacher entropy rather than policy lag), and the resulting loss happens to be exactly the one whose distribution-level half needs no importance correction under the 4-step inner loop. **The paper does not connect these.**

## Quotes

Every statement in the paper bearing on off-policy / stale student data or updates-per-batch:

1. §2.3 (title): "**OPD with Clipped-Reverse KL**".
2. §2.3: "To stabilize training, the OPD objective can be implemented with standard PPO-style importance sampling and clipping (Schulman et al., 2017)."
3. §2.3: "**We sample trajectories using a behavior policy `π_θold` instantiated from the student policy `π_θ`**, query the teacher for log-probabilities of the sampled student tokens, and define a per-token advantage `Â_t = log π_te(x_t|c_t) − log π_θold(x_t|c_t)`, substituting the behavior policy in (6). We then update `π_θ` by minimizing the clipped PPO surrogate over generated tokens."
4. §2.3: "Here, `r_t = π_θ(x_t|c_t)/π_θold(x_t|c_t)` **corrects for sampling under `π_θold`**, while clipping limits overly large policy updates."
5. §4, Alg. 1 lines 2/6/11/12: "`Set π_θold ← π_θ`" … "`Sample x = (x_1,…,x_|x|) ∼ π_θold(·|q)`" … "`for each mini-batch gradient step do`" … "`Sample a mini-batch of prompts B_mini ⊂ R`".
6. §5: "The expectations over q and x are approximated by sampling batches B (line 3 in Algorithm 1) and **generating rollouts using the behavior policy `π_old`** (line 6)."
7. §5.1 (Implementation): "**We train with a batch size of `B = 128` and a mini-batch size of `B_mini = 32`, resulting in 4 gradient steps per training iteration.**"
8. App. A (On-policy Training): "As shown in Table 9, we use a batch size of `B = 128` and a mini-batch size of `B_mini = 32`, which **results in four gradient update steps per training iteration**. For on-policy distillation, we generate a single rollout per problem during training."
9. §4/§5 (estimator split): "For the `L^OPD_t(θ; c_t)` term in (9), the expectation in (7) is approximated by a single-sample Monte Carlo estimate as discussed previously. In effect, the clipped reverse KL loss (8) is evaluated at the token `x_t` sampled during the rollouts. The forward KL `L^FKL_t(θ; c_t)` is approximated *not by sampling*, but as an expectation computed over the teacher's top-k tokens `S^k_t`."
10. §2.2: "On-policy distillation (OPD) (Agarwal et al., 2024) is a post-training method in which a student model learns by matching a teacher's token-level probability distribution on its own generated sequences. By training on on-policy rollouts rather than teacher-generated trajectories, OPD enables precise credit assignment and mitigates compounding errors inherent in off-policy imitation."
11. §5.6: "We compare EOPD with GKD (Agarwal et al., 2024), which optimizes a divergence objective over a mixture of off-policy data and on-policy rollouts controlled by a coefficient λ. … the off-policy data is generated by the teacher model (Qwen3-8B), while the on-policy data is sampled from the student (Qwen3-1.7B-Base) during training. We evaluate λ ∈ {0.5, 0.8, 1.0}. **We observe that the fully on-policy setting (λ = 1.0) achieves the best performance among different λ values**, consistent with GKD's observation that on-policy rollouts are particularly important for reasoning tasks."
12. §3.2: "These results demonstrate that **reverse-KL rewards provide unstable learning signals when the teacher distribution is uncertain.** This motivates training objectives that more directly transmit the teacher's distributional structure to the student."
13. App. A: "Teacher Query. EOPD does not require additional teacher forward passes compared to OPD. Both methods query the teacher once for each student-generated token position."

Nowhere in the paper do the words *stale*, *staleness*, *asynchronous*, *replay*, *lag*, or *ppo_epochs* appear (full-text check, 2026-09-14). "off-policy" appears 11×, **always** meaning teacher-generated data (classic sequence-level KD / GKD's λ-mixture), never stale student rollouts.

## Not in the source

- **No sync control.** No `B_mini = B` (one update per rollout batch) arm, so the paper cannot say what its 4 inner updates cost or buy. Everything is measured at one staleness setting.
- **No clip value.** ε is a symbol in Eq. (8) and a `Require:` line in Alg. 1 and never a number, in the paper or its appendices.
- **No off-policyness diagnostics**: no clip fraction, no `KL(π_θ ‖ π_θold)`, no ratio histogram, no per-step drift. Nothing lets one check whether the 4th mini-batch is meaningfully off-policy.
- **No statement that the FKL term is correction-free**, and no discussion of why it needs no ratio while the RKL term does — the asymmetry in Eq. (8) vs Eq. (10) is presented purely as a compute/variance choice.
- **No ablation of the estimator under staleness**: Table 6 (OPD vs OPD-FKL vs EOPD) is run entirely at `B_mini = 32`, so it cannot separate "FKL is better" from "FKL survives 3-step-stale mini-batches better".
- No teacher-top-k vs student-top-k comparison for the FKL term (teacher-top-16 only; App. F varies only k).
- No wall-clock/throughput motivation for the mini-batching (the 4 inner steps are inherited from verl defaults, not justified).
