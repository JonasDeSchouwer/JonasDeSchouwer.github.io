# Kang, Galim et al. 2026 — AsyncOPD: How Stale Can On-Policy Distillation Be?

**arXiv 2606.24143v1** (23 Jun 2026; only version as of 2026-09-14). Wonjun Kang*, Kevin Galim*, Seunghyuk Oh, Minjun Kang, Sanghyun Park, Donghoon Kim, Minjae Lee, Minseo Kim, Rishabh Tiwari, Yuchen Zeng, Hyung Il Koo, Kangwook Lee (*equal contribution). Affiliations on the title page: **FuriosaAI** (1), Ajou University (2), UC Berkeley (3), Microsoft Research (4), KRAFTON (5), Ludo Robotics (6) — contact `{kangwj1995, kevin.galim}@furiosa.ai`. **This is a FuriosaAI-led paper, not a Microsoft paper** (only Y. Zeng is MSR). Code: `https://github.com/furiosa-ai/async-opd`.

Sources: PDF `https://arxiv.org/pdf/2606.24143` (curl + pypdf, whole paper incl. Appendices A–G, 2026-09-14); arXiv HTML `https://arxiv.org/html/2606.24143v1` for verbatim equation LaTeX; Fig. 1 read as an image (`teaser_cached_teacher.png`); Figs. 2–4, 6, 7, 10 read by parsing the published vector SVGs (see "Figure data provenance" below). Repo `furiosa-ai/async-opd` @ `e3f79e9` consulted **only** for two hyperparameters the paper omits — every such use is marked `[repo, not paper]`.

```bibtex
@article{kang2026asyncopd,
  title   = {AsyncOPD: How Stale Can On-Policy Distillation Be?},
  author  = {Wonjun Kang and Kevin Galim and Seunghyuk Oh and Minjun Kang and Sanghyun Park and Donghoon Kim and Minjae Lee and Minseo Kim and Rishabh Tiwari and Yuchen Zeng and Hyung Il Koo and Kangwook Lee},
  journal = {arXiv preprint arXiv:2606.24143},
  year    = {2026}
}
```

## Tier

- **A** — yes, but **the argument runs OPPOSITE to our hypothesis**. §6 argues that the *sampled-token* estimator (their MC = `rkl_pg`) is the one that tolerates staleness, because samples from `p_old` are importance-correctable in expectation (exact identity Eq. 5), whereas *distribution-level student-top-k* RKL is **not** correctable: the teacher-scored support is `S_old = TopK(p_old, k)` and reweighting inside it cannot manufacture teacher scores for actions in `S_θ = TopK(p_θ, k) \ S_old`. Verbatim: "Reweighting within `S_old` cannot recover these missing teacher scores, so stale sparse top-`k` remains a support-mismatched approximation, not an exact correction of the current top-`k` objective." (§6, "Sparse top-k: stale-support biased"). This is an **argument + an exact algebraic identity, not a theorem**: the paper contains no theorem/lemma/proposition/proof environment anywhere (grep: zero hits).
- **B** — full staleness sweeps `k ∈ {0,…,128}` side by side with explanations: Fig. 2 + Finding 1 (KL direction), Fig. 3/Table 1a + §5.2 (advantage & clipping), Fig. 4 (async-RL surrogates), Fig. 6 (MC vs stale top-k), Fig. 7/Table 1b (multi-sample MC), Fig. 10 (IS ablation). Plus true sync-vs-async *systems* comparison with throughput: Tables 2 and 7.
- **D** — dense in quotable async/stale OPD statements (§1, §4.1, §4.2, §5.1, §6, §8). See `## Quotes`.
- **F** — asynchronous OPD *is* the paper; the title is the research question.
- **Not C** (explanations are given), **not G** (async is the main point).
- **Not E** — a research preprint from an industry lab + academic collaborators on a single 8-GPU node; it is not a production model tech report. "Our experiments are also limited to a single 8-GPU node by available resources" (§8, Limitations).

## Setting

- **Student(s)**: Qwen3-4B-Base for *every* staleness/estimator experiment (§4.1 "Experimental setup", Table 3). The AsyncOPD scheduler section additionally uses Qwen3-{1.7B, 4B, 8B}-Base (Table 2) and thinking-disabled Qwen3-{1.7B, 4B, 8B} (Table 7; "we disable thinking at the tokenizer prompt-formatting level", App. G).
- **Teacher**: Qwen3-30B-A3B-Instruct-2507 in every run (Table 3). **Teacher fixed: yes** — it is only ever used for scoring ("Stage 2: Teacher scoring", §4.1); it is never trained and never updated. Single teacher, no multi-teacher.
- **Data / benchmarks**: DeepMath "filtered to 57,630 math problems with difficulty level at least 6" (§4.1). Eval = final-checkpoint Avg@32 on AIME24 (30 problems), AIME25 (30), AMC 2023 (40) (Table 4). `Avg@32(D) = 100 · (1/|D|) Σ_i c_i/32` (Eq. 14). Generation for eval: 32 samples/problem, temperature 1.0. AIME24 evaluated every 20 steps; AIME25/AMC only at the final checkpoint (App. C). The "Average" panel in every figure is the unweighted mean of the three benchmarks (verified numerically).
- **Training config** (Table 3): 2,048 prompt / 16,384 response tokens; 200 training iterations (100 for the scheduler runs, §7); batch 256, mini-batch 64, so **4 optimizer updates per train-batch step**; AdamW-style, lr `3e-6`, constant schedule, weight decay 0.01; PPO clip `ε = 0.2`; M2PO budget 0.01; rollout temperature 1.0, top-p 1.0, **no top-k truncation**.
- **Framework**: vLLM 0.16.0 for rollout *and* teacher scoring; PyTorch FSDP 2.9.1 learner (App. B, Table 5). Hardware: one 8×B200 node; one GPU reserved for the teacher, the other seven split 4 rollout / 3 trainer for the async schedulers, time-shared for strict sync (App. G).
- **No seeds, no error bars, no repeats anywhere** — grep for "seed" returns nothing. Every figure point is a single run.

## Async mechanism

- **What is stale: the behaviour student only.** Three-stage cached-data pipeline (§4.1): Stage 1 rollout actor samples from a stale student `p_old`, fixing the visited prefixes `s` and caching actions `C_old(s)` plus their rollout-time `log p_old`; Stage 2 the fixed teacher annotates a cache `C_score(s)`; Stage 3 the learner, now at `p_θ`, recomputes `log p_θ(a|s)` on `C_score(s)`. "Unlike synchronous OPD, these stages are separated in time, so the visited prefixes, action cache, teacher scores, and update policy may be tied to different student versions." (§4.1).
- **Two levels of staleness, only one studied.** "Asynchronous OPD has both prefix-level and action-level staleness. Once a rollout is generated, its visited prefixes are fixed, so an action-level estimator cannot change which states the learner sees. We therefore focus on the action-level staleness that estimator design can directly address." (§4, opening). So the paper makes **no attempt** to correct the prefix/state distribution shift; all correction is per-token at a fixed prefix.
- **Staleness unit: train-batch steps.** `Δ_batch = t_train − t_roll` (App. B), where "One train-batch step is one logical rollout batch consumed by the learner for a training iteration." "All examples in a logical batch share the same rollout snapshot and therefore share the same `Δ_batch`." A train-batch step = 4 optimizer updates here, and App. B is explicit that the *optimizer-step* conversion "is not the staleness axis used in the plots".
- **Controlled protocol** (App. B): "A value `k = 0` is synchronous: rollout, teacher scoring, training, and weight synchronization occur in strict sequence. For `k > 0`, the run first generates exactly `k` rollout batches with the initial student snapshot before the first learner update. Training then consumes the oldest available generated batch; after each learner update and weight synchronization, a new rollout batch is generated with the latest student snapshot whenever needed to restore the target cache depth." `Δ_batch` "ramps as 0,1,2,… while the initial cache is drained and then plateaus at `k`".
- **Range studied: `k ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128}`** train-batch steps (App. B), i.e. up to 128 batches ≈ 512 optimizer updates of lag, on a 200-iteration horizon. This is by far the widest staleness sweep in the async-OPD literature I have seen.
- **AsyncOPD (the system, §7 + App. G)** is *not* a fixed-lag scheme: queue depth `τ = 4` is "enforced as a capacity bound rather than as a learner-side drop rule"; a semaphore with `(τ+1)B` permits throttles the prompt feeder. "The queues themselves remain FIFO; items are not evicted for being stale." Realised staleness in these runs is therefore **not reported**.
- **Baseline async scheduler**: two-step-off, the VeRL [19] family — "a `k`-step-off run fixes rollout lag to `k` learner updates, but still waits for complete rollout batches" (§7). AsyncOPD instead streams: "workers pause only for weight sync, preserve in-flight prefixes, teacher scoring consumes completed items, and the learner updates once a scored batch is ready" (§7, Fig. 9).
- **Weight-sync keep-mode** (App. G): during sync rollout workers pause; "the already sampled token prefix is kept, the student weights are updated, and the running-request prefix cache is reset so the engine rebuilds the attention state for that prefix under the new weights before generation resumes. Thus, tokens before the synchronization point are reused rather than regenerated, while later tokens are sampled under the new student snapshot. Each completed sample records the token index at which the active weight version changes." (So a single sequence can be generated by two student versions.)

## Objective

Convention check: the paper's reverse KL is `KL(p_θ ‖ q)` = KL(student ‖ teacher) — **matches our convention**. Advantage `A = log q − log p_θ` (teacher minus student), so minimising `D_R` = maximising `E[A]`. `q` = teacher, `p_θ` = current student, `p_old` = rollout student, `s` = visited prefix, `a` = next token, `V` = vocabulary.

All four objective families appear; the sparse ones are **truncated-support**, and the supports differ by direction.

- **`fkl` (teacher-top-k, truncated)** — the forward-KL arm of Fig. 2. Dense reference, Eq. (1):
  `D_{F}(\theta;s)=\mathrm{KL}(q(\cdot\mid s)\,\|\,p_{\theta}(\cdot\mid s))=\sum_{a\in\mathcal{V}}q(a\mid s)(\log q(a\mid s)-\log p_{\theta}(a\mid s))`
  gradient `\nabla_{\theta}D_{F}(\theta;s)=-\sum_{a\in\mathcal{V}}q(a\mid s)\nabla_{\theta}\log p_{\theta}(a\mid s)`.
  The *implemented* sparse form is given **only in Fig. 1(b)** (read from the image): `Forward KL  \sum_{a\in S_q} q(\log q - \log p_\theta)`, with the annotations "Cache `q` logits (`q`'s top-3)", "Drop `q` logits for efficiency", "**No cache misses**". So the forward-KL support is the **teacher's own top-k**, selected by the teacher at scoring time — this is the "or be selected by the teacher at scoring time" branch of `C_score(s)` in §4.1 Stage 2. §4.2 calls it "sparse top-`k` forward KL [19]" (i.e. the VeRL recipe). **Note the inconsistency**: §3 says sparse implementations "evaluate the corresponding restricted KL after renormalizing teacher and student distributions on `S(s)`", but Fig. 1(b) writes the forward-KL sum with **no renormalizer**, and Appendix A.1 defines the renormalization **only for reverse KL**. The exact forward-KL loss actually used is never written down in the body.
- **`student-top-k` reverse KL (stale = rollout-student-top-k)** — Appendix A.1. Support `S_{\theta}(s)=\operatorname{TopK}(p_{\theta}(\cdot\mid s),k)` (Eq. 8); restricted normalizers and renormalized distributions
  `\tilde{p}^{S}_{\theta}(a\mid s)=\frac{p_{\theta}(a\mid s)\mathbf{1}[a\in S]}{Z^{S}_{p}(s)},\qquad \tilde{q}^{S}(a\mid s)=\frac{q(a\mid s)\mathbf{1}[a\in S]}{Z^{S}_{q}(s)}` (Eq. 9);
  `D^{S}_{R}(\theta;s)=\mathrm{KL}(\tilde{p}^{S}_{\theta}(\cdot\mid s)\,\|\,\tilde{q}^{S}(\cdot\mid s))=-\sum_{a\in S}\tilde{p}^{S}_{\theta}(a\mid s)(\log\tilde{q}^{S}(a\mid s)-\log\tilde{p}^{S}_{\theta}(a\mid s))` (Eq. 10). "In practice, when `S = S_θ(s)`, we treat the selected top-`k` support as fixed during the local update." Under staleness the cached/scored support is `S_{\mathrm{old}}(s)=\operatorname{TopK}(p_{\mathrm{old}}(\cdot\mid s),k)` (§6). **So: reverse KL = student-top-k, forward KL = teacher-top-k.** `[repo, not paper]` the loss mode is literally named `reverse_kl_rollout_student_topk` with `rollout_student_topk_k: 64` and teacher `n_logprobs: 64`; **the paper never states `k`**.
- **`rkl_pg` (their "MC")** — **yes, this is exactly our `rkl_pg`**, not a one-sample estimator of the KL *value* (no k3/Schulman-style estimator anywhere). Dense reverse KL, Eq. (2):
  `D_{R}(\theta;s)=\mathrm{KL}(p_{\theta}(\cdot\mid s)\,\|\,q(\cdot\mid s))=-\sum_{a\in\mathcal{V}}p_{\theta}(a\mid s)(\log q(a\mid s)-\log p_{\theta}(a\mid s))`;
  gradient Eq. (3), after killing the score-function term `E_{a\sim p_\theta}[\nabla_\theta\log p_\theta]=0`:
  `\nabla_{\theta}D_{R}(\theta;s)=\sum_{a\in\mathcal{V}}p_{\theta}(a\mid s)(\log p_{\theta}(a\mid s)-\log q(a\mid s)+1)\nabla_{\theta}\log p_{\theta}(a\mid s)=-\sum_{a\in\mathcal{V}}p_{\theta}(a\mid s)(\log q(a\mid s)-\log p_{\theta}(a\mid s))\nabla_{\theta}\log p_{\theta}(a\mid s)`;
  rewritten Eq. (11) `\nabla_{\theta}D_{R}(\theta;s)=-\mathbb{E}_{a\sim p_{\theta}(\cdot\mid s)}[A_{\theta}(a,s)\nabla_{\theta}\log p_{\theta}(a\mid s)]`, and the one-sample estimator Eq. (12)
  `\widehat{g}_{\mathrm{MC}}(s,a)=-A_{\theta}(a,s)\nabla_{\theta}\log p_{\theta}(a\mid s),\qquad a\sim p_{\theta}(\cdot\mid s)`,
  multi-sample Eq. (13) `\widehat{g}_{m}(s)=-\frac{1}{m}\sum_{i=1}^{m}A_{\theta}(a_{i},s)\nabla_{\theta}\log p_{\theta}(a_{i}\mid s)`.
  §3 names it explicitly: "for reverse KL, this yields the student-sampled policy-gradient estimator". So: per-token advantage `A = sg[log q − log p]` in a policy-gradient update = `rkl_pg`. ✔
  **One real difference from vanilla `rkl_pg`**: the `m` actions are **local next-token draws at the visited prefix**, not the realised rollout tokens — "at each decoding timestep of a student rollout, draws multiple local next-token samples from the behavior policy **without rolling them out into additional trajectories**" (§6.1); App. E: "local MC actions `a_{t,1},…,a_{t,m}` are sampled iid **with replacement** from `p_old(·|s_t)` (duplicates are allowed)". For `m = 1` this is distributionally identical to using the realised rollout token, so **MC1 ≡ standard `rkl_pg`**; `m > 1` is a per-position variance-reduction that has no RL analogue ("In RL for LLM post-training, branching a prefix into multiple actions is expensive because each branch typically requires a full continuation before the reward or advantage can be evaluated", §6.1).
- **`rkl` (full-vocab)** — defined as the reference (Eq. 2) but **never run asynchronously**. "We study sparse and Monte Carlo OPD estimators, not dense full-vocabulary KL in the asynchronous setting." (§8).

## Off-policy correction

- **Per-token importance ratio, at a fixed prefix.** `\rho_{\theta}(a,s)=p_{\theta}(a\mid s)/p_{\mathrm{beh}}(a\mid s)`, with `p_beh = p_old`; clipped variant `\bar{\rho}_{\theta}(a,s)=\operatorname{clip}(\rho_{\theta}(a,s),1-\epsilon,1+\epsilon)`, `ε = 0.2` (Table 3). **Not per-sequence** — App. E pins `ρ_θ(a,s)=p_θ(a|s)/p_old(a|s)` as a next-token ratio.
- **The exact identity that motivates the winner, Eq. (5)** — "assuming `p_old` has support wherever `p_θ` does":
  `D_{R}(\theta;s)=-\mathbb{E}_{a\sim p_{\theta}}[A_{\theta}(a,s)]=-\mathbb{E}_{a\sim p_{\mathrm{old}}}[\rho_{\theta}(a,s)A_{\theta}(a,s)]`
  with `A_{\theta}(a,s)=\log q(a\mid s)-\log p_{\theta}(a\mid s)`. "For the policy-gradient update, the advantage is used as a stop-gradient weight; the derivative of the omitted `A_θ` term cancels by the score-function identity, as in Eq. (3)." This is an **exact algebraic identity**, stated inline with its support condition; **no theorem, no proof environment**.
- **The four surrogates ablated (Eqs. 6–7)**, `sg(·)` = stop-gradient, all expectations under `a ~ p_old`:
  `L_{\mathrm{old}}^{\mathrm{clip}}(\theta)=-\mathbb{E}_{a\sim p_{\mathrm{old}}}[\min(\rho_{\theta}\operatorname{sg}(A_{\mathrm{old}}),\bar{\rho}_{\theta}\operatorname{sg}(A_{\mathrm{old}}))]`
  `L_{\mathrm{old}}^{\mathrm{noclip}}(\theta)=-\mathbb{E}_{a\sim p_{\mathrm{old}}}[\rho_{\theta}\operatorname{sg}(A_{\mathrm{old}})]`
  `L_{\theta}^{\mathrm{clip}}(\theta)=-\mathbb{E}_{a\sim p_{\mathrm{old}}}[\min(\rho_{\theta}\operatorname{sg}(A_{\theta}),\bar{\rho}_{\theta}\operatorname{sg}(A_{\theta}))]`
  `L_{\theta}^{\mathrm{noclip}}(\theta)=-\mathbb{E}_{a\sim p_{\mathrm{old}}}[\rho_{\theta}\operatorname{sg}(A_{\theta})]`
  with `A_{\mathrm{old}}(a,s)=\log q(a\mid s)-\log p_{\mathrm{old}}(a\mid s)` (frozen at rollout time) and `A_θ` recomputed at learner time. "Here `L_old^clip` is the PPO-style adaptation, while `L_θ^noclip` is the OPD/IS surrogate." Baseline reference PPO form, Eq. (4): `L_{\mathrm{PPO}}(\theta;A_{\mathrm{beh}})=-\mathbb{E}_{a\sim p_{\mathrm{beh}}}[\min(\rho_{\theta}\operatorname{sg}(A_{\mathrm{beh}}),\bar{\rho}_{\theta}\operatorname{sg}(A_{\mathrm{beh}}))]`.
- **What "advantage recomputed at learner time without clipping" is, exactly** — it is `L_θ^noclip` above. Two changes vs the mechanical PPO port, *both* required: (i) the per-token advantage is **`A_θ = sg[log q(a|s) − log p_θ(a|s)]` evaluated under the CURRENT student at learner time**, not the rollout-time `A_old = log q − log p_old`; the teacher term `log q` is the cached score and is identical in both, so the only thing recomputed is the student log-prob in the advantage; (ii) the ratio `ρ_θ = p_θ/p_old` is **kept but left unclipped**. It is *not* "no correction": the IS ratio is still there. Its gradient is `-E_{a~p_old}[ sg(A_θ) ρ_θ ∇_θ log p_θ ]`, which by Eq. (5) is an unbiased estimate of `∇_θ D_R(θ;s)`. Note `A_θ` appears *twice* in effect — as the stop-gradient weight and implicitly in `ρ_θ` — and the paper's justification for dropping the clip is empirical + mechanistic: "recomputing `A_θ` already reduces the high-percentile `ρ_θ` tail at staleness 64 (Fig. 5), so clipping removes useful signal" (§5.2).
- **Multi-sample MC surrogate (Eq. 15, App. E; also unnumbered in §6.1)**:
  `\widehat{L}_{m}^{\mathrm{MC}}(\theta;s_{t})=-\frac{1}{m}\sum_{i=1}^{m}\rho_{\theta}(a_{t,i},s_{t})\operatorname{sg}(A_{\theta}(a_{t,i},s_{t})),\qquad \rho_{\theta}(a,s)=\frac{p_{\theta}(a\mid s)}{p_{\mathrm{old}}(a\mid s)}`.
  "By linearity, the gradient has the same expectation as the one-sample MC estimator; averaging independent behavior-policy samples reduces the Monte Carlo variance." (§6.1) — elementary argument, then measured (Table 6).
- **No IcePop-style mask, no truncation of `ρ` other than PPO clip, no KL penalty, no reference-model anchor.** External async-RL correctors tried as baselines: **Decoupled PPO** (AReaL) and **M2PO** (budget 0.01). Both are only described by citation: "Decoupled PPO [4] stabilizes asynchronous RL training by separating the behavior policy for stale rollouts from the proximal policy that anchors PPO [16] updates. M2PO [33] stabilizes stale updates with second-moment importance-weight constraints" (§2). Their OPD instantiations are never written as equations.
- **Top-k + reweighting ("Top-k+RW")** is the correction attempt for the sparse arm: old-to-current reweighting *inside* `S_old`. It fails by construction (§6) and empirically (Fig. 6).

## Results: async vs sync

**Figure data provenance.** The paper publishes only *slopes* (Tables 1a/1b) for Figs. 2–4, 6, 7, 10. The per-point values below were recovered by parsing the published matplotlib SVGs (`arxiv.org/html/2606.24143v1/{mean,aime24,aime25,amc}_*_final.svg`): each series' path coordinates inverted through the axis tick calibration. **Validated**: re-fitting accuracy against `log2(staleness+1)` from these recovered values reproduces **every entry of Table 1a and Table 1b exactly** (e.g. AIME24 `A_old(clip) −1.44`, `A_old −3.99`, `A_θ(clip) −1.64`, `A_θ −0.69`, `A_old(M2PO) −0.66`, `A_θ(M2PO) −1.00`; MC1/4/16/64 `−0.69/−0.42/−0.46/−0.34`). Residual error ≤ ~0.1 accuracy points. Treat the digits as accurate to ±0.1.

### Fig. 2 — KL direction under staleness (Qwen3-4B-Base, Avg@32)

Staleness (train-batch steps) →

| Benchmark | Objective | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | slope |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Average | Reverse KL (PPO-style = `L_old^clip`) | 38.63 | 38.06 | 38.45 | 38.29 | 37.95 | 36.82 | 34.10 | 28.84 | **23.84** | −1.93 |
| Average | Forward KL (teacher-top-k) | 34.84 | 34.77 | 33.74 | 33.89 | 34.51 | 34.60 | 33.97 | 33.05 | **32.93** | −0.23 |
| AIME24 | Reverse KL | 24.69 | 23.13 | 24.79 | 25.21 | 24.37 | 22.08 | 21.98 | 17.19 | 13.44 | −1.44 |
| AIME24 | Forward KL | 22.08 | 21.77 | 21.35 | 22.50 | 21.56 | 21.88 | 20.94 | 21.04 | 20.83 | −0.17 |
| AIME25 | Reverse KL | 21.98 | 21.35 | 22.50 | 20.83 | 21.98 | 21.67 | 19.38 | 15.73 | 10.10 | −1.40 |
| AIME25 | Forward KL | 19.48 | 19.27 | 18.85 | 18.23 | 19.17 | 19.90 | 19.48 | 18.33 | 19.38 | −0.00 |
| AMC | Reverse KL | 69.22 | 69.69 | 68.05 | 68.83 | 67.50 | 66.72 | 60.94 | 53.59 | 47.97 | −2.97 |
| AMC | Forward KL | 62.97 | 63.28 | 61.02 | 60.94 | 62.81 | 62.03 | 61.48 | 59.77 | 58.59 | −0.51 |

Crossover on the Average panel is **between staleness 32 and 64**. At sync, RKL leads FKL by **+3.8** points (Avg); at staleness 128, FKL leads by **+9.1**.

**Critical caveat, verified numerically:** the "Reverse KL" curve in Fig. 2 is *bit-identical* to the `A_old, clip` curve in Fig. 3. So **Finding 1 is established with the frozen-advantage PPO-style surrogate** (`L_old^clip`), not with the estimator the paper itself goes on to recommend. That surrogate is second-best of the four in Fig. 3 (the paper calls it "a strong stale-surrogate baseline", §5.2) but it is **7.5 Avg points worse at staleness 128** than `A_θ`-no-clip (23.84 vs 31.37). §4.2 says so obliquely ("PPO-style reverse-KL surrogates [12, 25]") but never flags what it costs the headline.

### Fig. 3 / Table 1a — advantage × clipping (2×2), Average panel

| Surrogate | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | slope (Avg) |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_θ`, no clip (`L_θ^noclip`, = MC1) | 38.31 | 39.24 | 38.19 | 38.56 | 38.74 | 38.16 | 38.92 | 36.03 | **31.37** | −0.73 |
| `A_θ`, clip | 38.46 | 38.24 | 38.68 | 38.31 | 37.28 | 36.87 | 33.29 | 25.44 | 19.41 | −2.53 |
| `A_old`, no clip | 39.34 | 38.98 | 36.67 | 38.17 | 32.51 | **1.59** | 11.21 | 11.71 | 13.14 | −5.25 |
| `A_old`, clip (PPO-style) | 38.63 | 38.06 | 38.45 | 38.29 | 37.95 | 36.82 | 34.10 | 28.84 | 23.84 | −1.93 |

Per-benchmark slopes reproduce Table 1a exactly (AIME24 `−1.44 / −3.99 / −1.64 / −0.69`; AIME25 `−1.42 / −3.69 / −1.88 / −0.38`; AMC `−3.06 / −8.00 / −4.06 / −1.12`). `A_old, no clip` **collapses to 0.00 AIME24 / 0.00 AIME25 / 4.77 AMC at staleness 16** and only partly recovers — this is the run that makes clipping look necessary.
**Fig. 5** (p99 of `ρ_θ` vs train step, at staleness 64, read from the image): `A_old,no-clip` climbs to ≈4.1 around step 57, falls back to ≈1.7 by step 130, then blows up to ≈6.5 by step 175; `A_θ,no-clip` stays flat at ≈1.8–2.1; both clipped variants sit at ≈1.2–1.6.

### Fig. 4 / Table 1a — async-RL surrogates, Average panel

| Surrogate | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | slope (Avg) |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_θ`, no clip | 38.31 | 39.24 | 38.19 | 38.56 | 38.74 | 38.16 | 38.92 | 36.03 | **31.37** | −0.73 |
| `A_old`, M2PO | 39.05 | 38.04 | 38.49 | 39.06 | 38.68 | 38.23 | 37.51 | 34.52 | 28.88 | −1.09 |
| `A_θ`, M2PO | 38.45 | 38.32 | 39.28 | 38.49 | 38.41 | 38.36 | 37.27 | 33.63 | 28.78 | −1.14 |
| Decoupled PPO | 38.99 | 37.62 | 37.73 | 34.84 | — | — | — | — | — | n/a |

Decoupled PPO is off-scale from staleness 8 onward — "Decoupled PPO is clipped for readability because of low accuracy" (Fig. 4 caption). M2PO slopes on AIME24 (`A_old` −0.66, `A_θ` −1.00) are **flatter than** `A_θ`-no-clip (−0.69) on `A_old`, but M2PO's absolute accuracy at staleness 128 is ~2.5 points lower, hence "do not **consistently** improve".

### Fig. 6 — sampled MC vs stale sparse top-k, Average panel

| Estimator | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | slope (Avg) |
|---|---|---|---|---|---|---|---|---|---|---|
| MC1 (+IS) | 38.31 | 39.24 | 38.19 | 38.56 | 38.74 | 38.16 | 38.92 | 36.03 | **31.37** | −0.73 |
| MC1 noIS | 37.71 | 38.27 | 39.32 | 39.23 | 35.19 | **1.33** | 38.55 | 35.95 | 18.60 | −2.33 |
| Top-k + RW (reweight in `S_old`) | 34.96 | 35.23 | 36.03 | 37.52 | **0.76** | 12.03 | 31.26 | 36.20 | **4.53** | −2.88 |
| Top-k (stale student-top-k) | 36.12 | 36.34 | 36.25 | 36.27 | 38.78 | 36.80 | **1.85** | 32.83 | 32.37 | −1.81 |

Note how ragged the top-k rows are: near-zero collapses at staleness 8 (Top-k+RW) and 32 (Top-k) followed by full recovery at 64. With **no seeds**, these are single-run catastrophes, so the slope fits are dominated by which staleness level happened to blow up. The paper reports the ranking without commenting on the non-monotonicity.

### Fig. 7 / Table 1b — multi-sample MC, Average panel

| Estimator | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | slope (Avg) |
|---|---|---|---|---|---|---|---|---|---|---|
| MC1 | 38.31 | 39.24 | 38.19 | 38.56 | 38.74 | 38.16 | 38.92 | 36.03 | 31.37 | −0.73 |
| MC4 | 38.85 | 38.89 | 38.27 | 39.64 | 37.85 | 38.48 | 38.43 | 37.34 | 33.93 | −0.50 |
| MC16 | 38.48 | 39.12 | 38.39 | 39.04 | 38.89 | 38.72 | 38.24 | 36.87 | 34.39 | −0.48 |
| MC64 | 38.49 | 39.40 | 39.34 | 38.10 | 38.95 | 38.52 | 38.46 | 37.80 | **34.64** | −0.43 |

Table 1b per-benchmark slopes: AIME24 `−0.69 / −0.42 / −0.46 / −0.34`; AIME25 `−0.38 / −0.23 / −0.24 / −0.11`; AMC `−1.12 / −0.87 / −0.74 / −0.84`. "`m = 4` already gives a clear jump, while `m ∈ {4,16,64}` performs similarly" (§6.2) — the Average panel at staleness 128 is 31.37 → 33.93 → 34.39 → 34.64, so returns are strongly diminishing after `m = 4`.

### Fig. 10 (App. F) — IS ablation, Average panel

| Estimator | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| MC1 + IS | 38.31 | 39.24 | 38.19 | 38.56 | 38.74 | 38.16 | 38.92 | 36.03 | 31.37 |
| MC1 noIS | 37.71 | 38.27 | 39.32 | 39.23 | 35.19 | 1.33 | 38.55 | 35.95 | 18.60 |
| MC16 + IS | 38.48 | 39.12 | 38.39 | 39.04 | 38.89 | 38.72 | 38.24 | 36.87 | 34.39 |
| MC16 noIS | 39.11 | 37.73 | 38.51 | 38.84 | 40.90 | 18.15 | 30.64 | **2.30** | 24.28 |

"Increasing `m` changes the Monte Carlo variance of the estimator. It does not define the target distribution. Old-to-current IS is still the mechanism that turns behavior-policy samples into an estimator of the current reverse-KL local gradient." (App. F). MC16-noIS collapses to 2.30 Avg at staleness 64 (the lowest single point in the paper is Top-k+RW at staleness 8, 0.76 Avg, Fig. 6) — multi-sampling without IS does **not** substitute for the correction.

### Table 6 (App. E) — measured variance reduction at staleness 128, Qwen3-4B-Base

| `m` | fixed-prefix ratio `R_m^local` | sequence-level ratio `R_m^seq` |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 4 | 0.255 | 0.338 |
| 16 | 0.0588 | 0.152 |
| 64 | 0.0149 | 0.112 |

"Fixed-prefix ratios closely follow the `1/m` reference (`m = 64` leaves 1.49% of the one-sample local variance, versus `1/64 = 1.56%`). After timestep aggregation, `m = 64` still leaves only 11.2% of the one-sample sequence-level variance" (App. E).

### Table 2 — AsyncOPD scheduler, Qwen3-Base, 100 iterations, 8×B200, all runs `A_θ`+no-clip+IS

Train tok/s (×sync) / pipeline overlap / final AIME24 Avg@32:

| Student | Scheduler | MC64 tok/s | overlap | Avg@32 | MC1 tok/s | overlap | Avg@32 |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | Strict sync | 8.7k (1.00×) | 0.81 | 8.85 | 8.6k (1.00×) | 0.82 | 8.65 |
| | Two-step-off | 14.2k (1.64×) | 1.49 | 8.23 | 18.5k (2.15×) | 1.65 | 8.12 |
| | **AsyncOPD** | **23.4k (2.70×)** | 2.13 | **9.38** | **28.1k (3.28×)** | 2.31 | 8.44 |
| Qwen3-4B-Base | Strict sync | 9.5k (1.00×) | 0.81 | **25.00** | 8.1k (1.00×) | 0.82 | 23.33 |
| | Two-step-off | 12.4k (1.30×) | 1.64 | 23.85 | 12.9k (1.60×) | 1.65 | 23.54 |
| | **AsyncOPD** | **15.8k (1.66×)** | 2.19 | **25.00** | **16.4k (2.03×)** | 2.27 | **23.44** |
| Qwen3-8B-Base | Strict sync | 7.5k (1.00×) | 0.81 | 26.56 | 6.5k (1.00×) | 0.84 | 28.44 |
| | Two-step-off | 11.6k (1.55×) | 1.76 | 26.56 | 9.6k (1.47×) | 1.63 | **28.85** |
| | **AsyncOPD** | **14.5k (1.94×)** | 2.24 | **28.65** | **10.6k (1.63×)** | 2.17 | 27.50 |

### Table 7 (App. G) — same, Qwen3 instruct models with thinking disabled

| Student | Scheduler | MC64 tok/s | overlap | Avg@32 | MC1 tok/s | overlap | Avg@32 |
|---|---|---|---|---|---|---|---|
| 1.7B | Strict sync | 11.2k (1.00×) | 0.78 | **35.00** | 11.7k (1.00×) | 0.78 | **35.21** |
| | Two-step-off | 18.0k (1.61×) | 1.58 | 32.81 | 20.8k (1.78×) | 1.71 | 33.54 |
| | **AsyncOPD** | **24.2k (2.16×)** | 2.07 | 33.23 | **37.0k (3.17×)** | 2.56 | 34.79 |
| 4B | Strict sync | 4.5k (1.00×) | 0.87 | 54.90 | 7.4k (1.00×) | 0.83 | 56.15 |
| | Two-step-off | 11.5k (2.58×) | 1.54 | **56.77** | 13.0k (1.75×) | 1.63 | **58.02** |
| | **AsyncOPD** | **17.0k (3.82×)** | 2.15 | 54.69 | **18.5k (2.49×)** | 2.26 | 56.25 |
| 8B | Strict sync | 5.6k (1.00×) | 0.86 | 59.69 | 3.9k (1.00×) | 0.87 | 58.96 |
| | Two-step-off | 7.5k (1.34×) | 1.45 | 58.33 | 9.6k (2.45×) | 1.61 | **60.73** |
| | **AsyncOPD** | **10.1k (1.81×)** | 2.11 | **60.52** | **10.9k (2.78×)** | 2.17 | 58.65 |

Headline: "AsyncOPD improves training throughput by 1.6× to 3.8× over strict synchronous training while reaching comparable accuracy" (Abstract) — the 1.6× is Table 2 (8B-Base MC1, 1.63×) and the 3.8× is Table 7 (4B MC64, 3.82×). **Accuracy deltas AsyncOPD − strict sync** across all 12 matched cells: `+0.53, −0.21, 0.00, +0.11, +2.09, −0.94` (Table 2) and `−1.77, −0.42, −0.21, +0.10, +0.83, −0.31` (Table 7) — median −0.10, i.e. genuinely a wash. **AsyncOPD loses to two-step-off on final accuracy in 5 of 12 cells.** Metrics: `throughput = Σ_{j=6}^{J} n_j / (t_J − t_5)` discarding five warmup updates; `overlap = Σ_{s∈S} T_s / T_wall` over `S = {rollout, teacher, train}`, "A mostly serial schedule has overlap near 1, and the maximum remains 3" (App. G).

### Cross-figure comparison the paper does not make (my derivation)

All staleness runs "share the common model, data, batch-size, generation, and evaluation settings in Table 3" (App. B), so Fig. 2's forward-KL row is directly comparable to Fig. 7's MC rows. Average panel:

| Staleness | 0 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Forward KL (teacher-top-k) | 34.84 | 34.51 | 33.97 | 33.05 | 32.93 |
| Reverse KL, MC64 (`A_θ`, no clip, IS) | 38.49 | 38.95 | 38.46 | 37.80 | 34.64 |

**The paper's own best reverse-KL estimator beats its forward-KL baseline at every staleness level tested, including 128.** Forward KL is *flatter* (slope −0.23 vs −0.43) but starts ~3.7 points lower and never catches up. Finding 1's "forward KL is robust, reverse KL is vulnerable" is therefore a statement about the **PPO-style frozen-`A_old` surrogate only**, and the paper never retracts or qualifies it after Sections 5–6 fix the surrogate. This is the single most important thing to carry into our review.

## Explanation offered

Mechanism: **support ownership under a finite teacher-score cache**, not variance, not teacher inconsistency, not prefix distribution shift (the last is explicitly set aside).

- Why the KL direction matters (§4.2): "The KL direction fixes the action weighting: forward KL weights actions by the teacher `q`, whereas reverse KL weights them by the student `p_θ`. With cached teacher scores, this weighting difference becomes a **support-ownership difference** (Fig. 1). Under a scored-cache restriction, this makes forward KL less exposed to stale student action choices: **it does not need to convert stale student-sampled actions into a current-student expectation.** Reverse KL instead depends on student-weighted action terms, so the same asynchronous cache creates a different action-level staleness problem."
- Fig. 1(b), read from the image: forward KL caches `q`'s own top-k and gets "**No cache misses**"; reverse KL caches `p_old`'s top-3 `{a1,a2,a4}`, the current student's top-3 is `{a1,a2,a5}`, and `a5` is marked "**Cache miss**" with a red ✗.
- **Finding 1** (boxed, §4.2): "Forward KL is teacher-weighted and robust to rollout staleness, whereas reverse KL is student-weighted and vulnerable to rollout staleness."
- Hedged as a hypothesis, not proven (§4.2, "Two axes"): "The cache analysis above **suggests a possible mechanism** for this gap: because reverse KL is weighted by the current student, stale teacher-scored caches may fail to cover actions needed by the current reverse-KL objective."
- Why `A_θ` beats `A_old` (§5.1, §5.2): the IS identity Eq. (5) "points to the opposite surrogate choice from the mechanical PPO adaptation: recompute `A_θ` under the current student and use the old-to-current ratio without clipping, with `A_θ` treated as a stop-gradient advantage." And empirically: "Clipping helps `A_old` by limiting stale, large-ratio updates, but hurts `A_θ`: recomputing `A_θ` already reduces the high-percentile `ρ_θ` tail at staleness 64 (Fig. 5), so clipping removes useful signal."
- **Finding 2**: "The most effective reverse-KL correction is to recompute `A_θ` at learner time without clipping; advanced asynchronous RL surrogates such as decoupled PPO and M2PO do not improve over it."
- Why stale top-k cannot be fixed (§6): "Reweighting within `S_old` cannot recover these missing teacher scores, so stale sparse top-`k` remains a **support-mismatched approximation, not an exact correction** of the current top-`k` objective."
- Why MC can be fixed (§6): "When the behavior policy covers the current policy support, exact old-to-current IS gives an **unbiased estimator** of the current reverse-KL fixed-prefix gradient. Thus one-sample MC is action-level correctable in expectation, but the resulting IS estimator can have high variance. **This proposal-sampling structure is the key contrast with stale top-`k`, whose actions come from a deterministic stale support.**"
- Why multi-sample (§6.1): "In RL for LLM post-training, branching a prefix into multiple actions is expensive because each branch typically requires a full continuation before the reward or advantage can be evaluated. In synchronous OPD, sparse top-`k` already provides a low-variance approximation and one-sample MC provides an unbiased sampled gradient estimator, so there is little motivation to cache multiple sampled actions per prefix. Under asynchronous OPD, this tradeoff changes."
- **Finding 3**: "One-sample MC is more effective than stale sparse top-`k`; multi-sample MC further improves this estimator by reducing one-sample variance while preserving MC correctability."

**Proof vs empirical, itemised:**

| Claim | Status |
|---|---|
| Eq. (3) reverse-KL gradient | exact derivation (score-function identity), inline |
| Eq. (5) old-to-current IS identity | **exact algebraic identity**, inline, with stated support condition `supp(p_θ) ⊆ supp(p_old)`; no theorem/proof environment |
| one-sample MC + IS is unbiased for the fixed-prefix gradient | corollary of Eq. (5)/(11), asserted in prose |
| stale student-top-k is support-mismatched / not correctable | **argument** from the cache structure (§6, Fig. 1b); no formal non-identifiability statement |
| `1/m` variance reduction | "By linearity… averaging independent behavior-policy samples reduces the Monte Carlo variance" + measured (Table 6) |
| Finding 1 (FKL robust, RKL vulnerable) | **purely empirical** (Fig. 2, n=1 per point) + informal support-ownership argument, explicitly hedged ("suggests a possible mechanism") |
| Findings 2 and 3 | **purely empirical** (Figs. 3, 4, 6, 7, 10; single runs) |

No theorem, lemma, proposition, corollary or proof environment appears anywhere in the paper.

## Quotes

- Abstract: "We first show that **KL direction changes the stale-data problem: teacher-weighted forward KL is more robust to stale rollouts, whereas student-weighted reverse KL is vulnerable.**"
- Abstract: "Second, for this vulnerable reverse-KL case, we study whether methods designed to stabilize asynchronous RL can mitigate OPD staleness. In our experiments, **they do not improve over a simpler OPD-specific surrogate: recomputing the reverse-KL signal under the current student at learner time.**"
- Abstract: "Third, we analyze how finite teacher-score caches create a bias-variance tradeoff for sparse and sampled reverse-KL OPD estimators. This motivates multi-sample Monte Carlo (MC), which preserves MC correctability while reducing one-sample variance."
- Abstract: "Experiments show that AsyncOPD improves training throughput by 1.6× to 3.8× over strict synchronous training while reaching comparable accuracy."
- §1: "OPD and RL inherit an on-policy systems bottleneck: each learner update must wait for fresh rollouts from the model being trained. For reasoning tasks, these rollouts are long and expensive, so synchronous training often waits on generation rather than updating the model, leaving learners underutilized."
- §1: "However, asynchronous execution introduces stale-policy data, and learning from such data can degrade model quality. This creates a trade-off: more aggressive asynchrony improves training throughput, but it also increases the policy lag between rollout and learning."
- §1: "it remains underexplored whether these ideas and stale-data solutions transfer to OPD, because practical implementations of OPD expose a different feedback interface. Teacher feedback is often implemented through local KL losses, which require teacher scores over actions at student-visited prefixes. Since full-vocabulary teacher logits are expensive to store or transfer, especially in an asynchronous pipeline, teacher scores are usually cached only on a finite set of actions (Fig. 1)."
- §1: "Once the learner receives a teacher-scored cache, it can recompute current-student log probabilities on cached actions, but **it cannot recover teacher scores for actions that were never scored.**"
- §1: "This comparison is natural because **reverse KL in OPD admits an RL-style policy-gradient surrogate, where the teacher-student log-ratio acts as a token-level advantage.**"
- §1 (contributions): "We show that **stale student top-`k` supports are support-mismatched, while one-sample MC remains correctable but high-variance**; this motivates multi-sample MC (Section 6)."
- §1: "We provide **the first systematic study of staleness in asynchronous OPD** through the lens of an OPD-specific teacher-cache constraint."
- §2: "VeRL [19] implements step-off OPD schedulers that overlap student rollout, teacher scoring, and learner update by fixing rollout lag to one or two learner steps. **These schedulers establish the practical feasibility of asynchronous OPD, but leave open how OPD estimators behave under stale teacher-scored caches.**"
- §2: "KDFlow [29] improves systems efficiency for LLM distillation by decoupling teacher inference from learner training and transmitting teacher hidden states, but **targets synchronous OPD and leaves asynchronous execution as future work.**"
- §2: "These works clarify OPD as an effective post-training objective, but **assume rollouts, teacher scoring, and learner updates stay synchronized.**"
- §3: "Viewing Eq. (3) as a policy-gradient estimator and `A = log q(a|s) − log p(a|s)` as the advantage term connects reverse-KL OPD to standard RL training machinery, and practical implementations typically use PPO-style surrogates."
- §3: "**Monte Carlo (MC) implementations** draw actions from a proposal distribution and estimate the corresponding local gradient; **for reverse KL, this yields the student-sampled policy-gradient estimator.**"
- §4: "Asynchronous OPD has both prefix-level and action-level staleness. Once a rollout is generated, its visited prefixes are fixed, so an action-level estimator cannot change which states the learner sees. **We therefore focus on the action-level staleness that estimator design can directly address.**"
- §4.1: "Full-vocabulary teacher logits allow dense KL computation, but **caching and transferring them is prohibitively expensive, especially in an asynchronous pipeline.**"
- §4.1 (Stage 3): "the current local OPD objective may place mass on actions outside this teacher-scored cache, such as the current student top-`k` support or current student-sampled actions. Thus the learner can update current student probabilities on cached actions, but **cannot recover teacher signals for missing actions without additional teacher access.**"
- §4.2: "Reverse KL starts higher at zero staleness, but as staleness increases **it drops faster and is eventually overtaken by forward KL.**"
- Fig. 2 caption: "Reverse KL starts higher at zero staleness but degrades faster as staleness grows; forward KL is flatter across the sweep."
- §5.1: "In stale reverse-KL OPD, the behavior policy is the rollout student, so a mechanical PPO-style adaptation sets `p_beh = p_old` and uses the rollout-time reverse-KL advantage … together with the clipped old-to-current ratio."
- §5.2: "The best variant is the OPD/IS choice: `A_θ` without clipping. **The PPO-style baseline, `A_old` with clipping, remains a strong stale-surrogate baseline.**"
- Fig. 3 caption: "Recomputing `A_θ` at learner time and avoiding clipping gives the most stable performance across the sweep, while clipping mainly helps the frozen `A_old` baseline."
- Fig. 4 caption: "Decoupled PPO [4] and M2PO [33] **do not consistently improve** over the simpler OPD/IS surrogate that recomputes `A_θ` without clipping; Decoupled PPO is clipped for readability because of low accuracy."
- §6: "This cached-support axis is specific to OPD because **teacher scoring is local and expensive**: the teacher cache determines which actions have teacher scores available to the learner."
- §6.2: "For one-sample MC, IS substantially improves robustness as staleness increases. **For sparse top-`k`, the same reweighting does not improve performance, since it cannot recover missing current-support actions.**"
- Fig. 6 caption: "Old-to-current IS corrects sampled MC in expectation, **whereas reweighting cannot repair the missing teacher scores induced by stale top-`k` supports.**"
- §7: "The step-off scheduler family was originally implemented in VeRL [19]: a `k`-step-off run fixes rollout lag to `k` learner updates, **but still waits for complete rollout batches.** AsyncOPD streams examples instead."
- App. G: "Two-step-off fixes a two-update offset between rollout and the learner update that consumes it, so **stale rollout reuse is static and controlled rather than produced by queue timing.** … **Larger offsets only make the consumed data older; they do not create another OPD stage to overlap or remove the step-off batch barrier.**"
- App. G: "The queue-depth parameter `τ` is enforced as a capacity bound rather than as a learner-side drop rule. … **The queues themselves remain FIFO; items are not evicted for being stale.**"
- §8: "We study sparse and Monte Carlo OPD estimators, **not dense full-vocabulary KL in the asynchronous setting.** Although dense KL avoids cached-support mismatch, it is difficult to implement efficiently when rollout, teacher scoring, and learner updates are decoupled."

## Not in the source

- **The top-k size `k` is never stated** — not in §3, §6, Appendix A, or Table 3. Both the forward-KL and reverse-KL top-k arms of Figs. 2 and 6 depend on it entirely. `[repo, not paper]` `configs/examples/advanced/opd_deepmath_qwen3_4b_reverse_kl_topk_8gpu.yaml` sets `rollout_student_topk_k: 64` with teacher `vllm.n_logprobs: 64`; the forward-KL configs use the same 64 teacher logprobs.
- **The forward-KL loss is never written as an equation in the body** — only as an unnumbered expression inside Fig. 1(b), and §3's renormalization convention contradicts it (Fig. 1b shows no renormalizer; App. A.1 renormalizes only the reverse-KL case).
- **No seeds, no error bars, no repeated runs.** Several series collapse to near-zero at one staleness value and fully recover at the next (Top-k+RW at 8, Top-k at 32, MC16-noIS at 64), which is exactly what single-run instability looks like; the paper fits slopes through these points without discussing it.
- **No forward-KL entry anywhere after Fig. 2.** Forward KL is never run with `A_θ`-style correction, never run with multi-sample caching, never run in the AsyncOPD scheduler tables, and never compared to the *best* reverse-KL estimator. Finding 1 is therefore never revisited.
- **No dense/full-vocab `rkl` or `fkl` async baseline** (acknowledged, §8).
- **No token-budget or cache-size matching.** MC64 caches 64 teacher-scored actions per position; top-k caches `k` (=64 `[repo]`); MC1 caches 1. Fig. 6's MC1-vs-top-k comparison is therefore 1 scored action vs 64, and MC1 still wins — worth stating, but the paper never accounts for the 64× difference in teacher-scoring cost, nor reports teacher-scoring throughput or cache bytes per token.
- **Realised staleness of the AsyncOPD runs is not measured.** `τ = 4` is a capacity bound; the actual `Δ_batch` distribution in Tables 2/7 is never reported, so those runs cannot be placed on the Fig. 2–7 staleness axis.
- **No sequence-level reverse KL** (MiniLLM-style) and **no sequence-level importance ratio**; everything is local/per-token at a fixed prefix.
- **Decoupled PPO and M2PO are never written as equations** in their OPD form; only the M2PO "budget 0.01" hyperparameter is given (Table 3), and no tuning is reported for either.
- **No ablation of `τ`** for AsyncOPD, and no throughput-vs-staleness Pareto curve — the two halves of the paper (estimator sweep at 200 iterations, systems comparison at 100 iterations) are never joined.
- **No non-math domain.** All experiments are DeepMath → AIME/AMC. `[repo, not paper]` a code config exists (`opd_code_qwen3_1.7b_pg_mc64_async_8gpu.yaml`) but no code results are in the paper.
- **No wall-clock or cost accounting for teacher scoring** as a function of `m` or `k`, which is what would decide MC64 vs MC1 in practice.
