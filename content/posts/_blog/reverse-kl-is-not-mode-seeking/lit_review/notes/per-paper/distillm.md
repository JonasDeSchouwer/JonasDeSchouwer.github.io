# Ko, Kim, Chen, Yun 2024 — DistiLLM: Towards Streamlined Distillation for Large Language Models

**arXiv 2402.03898v2** (3 Jul 2024; ICML 2024, PMLR 235). Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun (KAIST AI; Microsoft Redmond). Sources: PDF full text `https://arxiv.org/pdf/2402.03898` (pypdf, 2026-09-14) incl. Algorithm 1, §2–5, App. B/C/D/E; code `github.com/jongwooko/distillm` @ `d47e77f` ("Update README.md", cloned 2026-09-14) — `distillm/buffer.py`, `distillm/sampler.py`, `distillm/losses.py`, `finetune.py`, `arguments.py`, `scripts/`.

```bibtex
@inproceedings{ko2024distillm,
  author    = {Jongwoo Ko and Sungnyun Kim and Tianyi Chen and Se-Young Yun},
  title     = {Disti{LLM}: Towards Streamlined Distillation for Large Language Models},
  booktitle = {International Conference on Machine Learning},
  year      = {2024},
  eprint    = {2402.03898},
  archivePrefix = {arXiv}
}
```

## Tier

- **F** — staleness is a headline contribution, not incidental: one of the paper's two stated contributions is the "**Adaptive off-policy approach**" (§1 Contributions bullet 2; §3.2), the replay buffer supplies the last 1.5× of the claimed speedup (Fig. 7: on-policy ×5.22/5.45 → adaptive ×2.46/2.58 → adaptive+off-policy ×1.52/1.67).
- **B** — side-by-side sync vs stale numbers *with* a mechanism: Tab. 2 (`Adaptive` = fresh SGO every time it is used vs `+ Off-policy` = same schedule but buffer replay), Tab. 4 (on- vs off-policy for ImitKD / GKD / DistiLLM), Tab. 9 (three replay-ratio schedules), Tab. 10 (buffer capacity sweep), Fig. 7 (wall-clock). Explanation given in §3.2 ("high bias error … significant divergence between past and current policies"; "Off-policy approach's success stems from the fast convergence speed of S(R)KL").
- **C** — subsumed by B.
- **D** — many quotable statements about stale-vs-fresh student samples (see `## Quotes`).
- **A** — *no*. The paper's staleness-tolerance argument compares **objectives at the same (full-vocab, distribution-level) granularity** — skewed KL/RKL vs plain KLD vs JSD — not distribution-level RKL vs a sampled-token `rkl_pg` estimator. MiniLLM (the `rkl_pg` representative) is a baseline but is never run off-policy. So the paper is *adjacent* to A: it argues that *which divergence you use* determines staleness tolerance, via convergence speed, not via estimator variance.
- **E**, **G** — no (academic paper; async is a main point, so F not G).

## Setting

- **Students / teachers (all fixed teachers, all SFT'd first):** GPT-2 XL 1.5B → GPT-2 {0.1B, 0.3B (Medium), 0.7B (Large)}; OPT-2.7B → OPT {125M, 350M, 1.3B}; OpenLLaMA2-7B → OpenLLaMA2-3B (LoRA for the ≥1B pair); T5 {Small, Base} for summarization/MT (§4, App. D.2).
- **Teacher fixed?** Yes — `teacher_model` loaded once, `torch.no_grad()` forward, never updated (`finetune.py` L171–193).
- **Domain / benchmarks:** instruction-following on `databricks-dolly-15k` (14k train / 500 val / 500 test), evaluated on Dolly-eval, Self-Instruct, Vicuna-eval, Super-Natural Instructions (SNI), Unnatural Instructions (UNI) by ROUGE-L and GPT-4 feedback; plus SAMSum/XSum/CNN-DM summarization and IWSLT'17 En-De MT (§4, App. D.1). A LM loss on OpenWebText is added for all instruction-following runs.
- **Framework:** own DeepSpeed codebase forked from MiniLLM (`github.com/jongwooko/distillm`); 4×A100-40GB. Per-script `BATCH_SIZE=8` per GPU, `GRAD_ACC=1`, 20 epochs (≤1B students), lr 5e-4, `--kd-ratio 1.0` (pure distillation loss on the KD data).

## Async mechanism

**What is stale:** *token sequences only*. `ReplayBuffer` (`distillm/buffer.py`) stores, per sample, `(input_ids, attention_mask, position_ids, label, loss_mask)` — **no logits, no log-probs, no behavior-policy statistics**. At training time both the teacher and the *current* student are re-run on the replayed tokens, so the loss is a fresh full-vocab divergence evaluated on a **stale prefix/continuation** produced by an earlier student checkpoint. This is "stale prompts/rollouts, fresh distributions", not stale-logprob importance sampling.

**Schedule (Algorithm 1, §3.2, and `finetune.py` L290–325, L411–418):**
- `φ` (`--init-threshold`, code `adaptive_threshold`) = probability of using a student-generated output (SGO) at all, vs. a ground-truth batch from `D`. Starts at `φ₀ = 0.0`; at each validation, if `L̃_t > L̃_{t−1} + ε` then `φ ← min(φ + 1/N_φ, 1.0)`. `ε = 0.1` (`--loss-eps`), `N_φ = 10` → φ moves on a 0.1 grid (App. C).
- `λ_R := φ(1 − t/T)` = probability of *generating a fresh* SGO (which is both trained on and pushed into `D_R`); `ζ := (1 − t/T)` is called the **replay ratio**.
- Per step, draw `u ∼ Unif(0,1)`: `u < λ_R` → generate fresh + store; else `u < φ` → sample a **stale** minibatch from `D_R`; else → ground-truth batch from `D`.
- Code-only detail not in the paper: the fresh-generation branch also fires while the buffer is under-filled — `elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity))` (`finetune.py` L311). So **no replay happens at all until `D_R` reaches `--capacity`**; staleness switches on only after the warm-up.
- Eviction FIFO, sampling uniform i.i.d. without replacement: `deque(maxlen=args.capacity)` and `random.sample(self.replay_memory, k=self.bs)` (`buffer.py` L20, L31). Paper: "We also replace the oldest samples of `D_R` with new ones once it reaches its maximum capacity" (§3.2).
- Buffer is **per rank** (constructed inside the per-process `finetune`), `capacity` = 1000 sequences, per-rank batch 8.

**Range of staleness studied:**
- Buffer capacity `|D_R| ∈ {250, 500, 1000, 2000, 4000}` sequences (Tab. 10); default 1000 (§4).
- Replay-ratio schedule `ζ ∈ {constant 0.5, increasing t/T, decreasing 1−t/T}` (Tab. 9); default decreasing.
- Mixing probability φ: terminal values reached by the scheduler are 0.2–0.8 depending on pair/task (App. E.3; GPT-2 family 0.4/0.4/0.5, OPT 0.2/0.3/0.5, OpenLLaMA2-3B 0.6, T5 0.3–0.8); Tab. 5 sweeps manually fixed φ ∈ {0.3, …, 0.6}.
- **Staleness is never reported in optimizer steps or tokens.** Derived estimate (mine, not the paper's): 1000 sequences / 8 per-rank batch = the last 125 *generated* batches; fresh generation fires with prob `λ_R = φ(1−t/T)` which averages well under 0.2, so the oldest replayed sample is **O(10²–10³) optimizer steps old** near the end of training. Treat as an order-of-magnitude only.

## Objective

**`fkl`-skew and `rkl`-skew, both full-vocab, token-level, teacher-forced** — no policy gradient, no sampled-token estimator anywhere in DistiLLM's own loss.

Token-level full-vocab decomposition the paper actually optimizes (Eq. 3):

```latex
\frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}_X}\sum_{t}^{|y|}\sum_{y_t\in V} p(y_t|y_{<t},x)\log\frac{p(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}
\tag{3}
```

Skew KL / skew reverse KL (§3.1, unnumbered):

```latex
D^{(\alpha)}_{\mathrm{SKL}}(p, q_\theta) = D_{\mathrm{KL}}\!\left(p,\ \alpha p + (1-\alpha) q_\theta\right)
```
```latex
D^{(\alpha)}_{\mathrm{SRKL}}(p, q_\theta) = D_{\mathrm{KL}}\!\left(q_\theta,\ (1-\alpha)p + \alpha q_\theta\right)
```

with `α = 0.1` optimal (§3.1 end; Fig. 8; Tab. 8). In our vocabulary: **SRKL = full-vocab `rkl` against a teacher-dominated mixture** (`(1−α)p + αq` with α small ⇒ close to plain `rkl`), **SKL = full-vocab `fkl` against a student-dominated mixture**. Headline config is **SRKL** (§4: "We apply DISTILLM with SRKL and the off-policy approach with initial probability as zero and replay buffer size of 1000 as SRKL with α of 0.1").

Gradient coefficients that motivate the skew (Eq. 6–7):

```latex
\nabla_\theta D^{(\alpha)}_{\mathrm{SKL}}(p, q_\theta) = -\underbrace{(1-\alpha) r_{p,\tilde q_\theta}}_{\text{coefficient}} \nabla_\theta q_\theta(y|x), \tag{6}
```
```latex
\nabla_\theta D_{\mathrm{KL}}(q_\theta, p) = -(\log r_{q_\theta,p} + 1)\nabla_\theta q_\theta(y|x), \tag{7}
```
```latex
\nabla_\theta D^{(\alpha)}_{\mathrm{SKL}}(q_\theta, p) = -\underbrace{(\log r_{q_\theta,\tilde p} + 1 - \alpha r_{q_\theta,\tilde p})}_{\text{coefficient}} \nabla_\theta q_\theta(y|x),
```

Code (`distillm/losses.py` L66–95) matches exactly: mixture formed in probability space, `.sum(dim=-1)` over the full vocabulary, mask `label != -100`, per-token mean; student `q` differentiable *inside and outside* the mixture (no detach); teacher under `no_grad`. Total loss `(1 − kd_ratio)·CE + kd_ratio·distil` with `kd_ratio = 1.0` in every DistiLLM script.

## Off-policy correction

**None.** No importance ratio, no per-token or per-sequence reweighting, no clipping, no truncation, no staleness mask — verified both ways:
- Paper: "importance" appears three times in the PDF and never in a sampling sense (§5.3 "importance of proper choice α", App. E.4, E.5). No IS equation exists.
- Code: `get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits)` is called **identically** whatever the provenance of `model_batch` (`finetune.py` L327–334); the buffer stores no behavior log-probs to correct against.

The paper's substitutes for a correction are (i) the *schedule* `λ_R = φ(1 − t/T)` that keeps stale draws rare when the student is moving fast, and (ii) the *finite buffer* that bounds how old a sample can be. §3.2: "we can consistently maintain a small `λ_R` by applying a large `ζ` for relatively smaller `φ` values and vice versa, effectively balancing bias error reduction with sample efficiency."

Structurally this is possible *only because the objective is distribution-level*: with no stale logits stored, a full-vocab divergence on a stale prefix is an unbiased estimate of the divergence at that prefix — the staleness is entirely in *which prefixes are visited*, never in the per-token target. A sampled-token `rkl_pg` estimator over the same buffer would carry a stale behavior log-prob and would need IS.

## Results: async vs sync

**(a) Tab. 2 — GPT-2 XL 1.5B → GPT-2 0.1B, ROUGE-L, mean (std) over 5 seeds.** `Adaptive` = SGOs used with the adaptive probability but always *freshly generated*; `+ Off-policy` = same schedule, stale draws from `D_R` (capacity 1000). This is the paper's cleanest sync-vs-stale pair.

| Loss | Generation | Dolly | Self-Inst | Vicuna | SNI | UNI | Rel. time (Fig. 7) |
|---|---|---|---|---|---|---|---|
| SKL | none (fixed data) | 24.80 (0.12) | 12.86 (0.34) | 16.20 (0.57) | 26.26 (0.41) | 28.06 (0.08) | ×1.0 |
| SKL | On-policy (fresh every step) | 24.27 (0.46) | 13.13 (0.44) | 16.39 (0.21) | 25.87 (0.18) | 26.49 (0.09) | ×5.22 |
| SKL | Mixed (p=0.5 fresh) | 25.27 (0.35) | 12.24 (0.69) | 17.19 (0.29) | 25.30 (0.33) | 26.51 (0.11) | ×3.73 |
| SKL | **Adaptive (sync)** | **25.90 (0.20)** | **13.24 (0.30)** | **17.59 (0.44)** | **27.62 (0.05)** | **28.30 (0.11)** | ×2.46 |
| SKL | **Adaptive + Off-policy (stale)** | 25.79 (0.28) | 13.03 (0.29) | 17.41 (0.15) | 27.32 (0.09) | 28.13 (0.21) | **×1.52** |
| SRKL | none (fixed data) | 25.21 (0.27) | 12.98 (0.24) | 15.77 (0.39) | 25.83 (0.15) | 28.62 (0.10) | ×1.0 |
| SRKL | On-policy (fresh every step) | 26.04 (0.33) | 12.93 (0.54) | 17.45 (0.37) | 27.29 (0.12) | 28.72 (0.10) | ×5.45 |
| SRKL | Mixed (p=0.5 fresh) | 26.01 (0.61) | 12.24 (0.69) | 17.19 (0.29) | 26.40 (0.34) | 29.02 (0.14) | ×3.64 |
| SRKL | **Adaptive (sync)** | **26.37 (0.21)** | 13.14 (0.37) | 18.32 (0.17) | **28.24 (0.22)** | **30.11 (0.04)** | ×2.58 |
| SRKL | **Adaptive + Off-policy (stale)** | 26.11 (0.68) | 13.14 (0.69) | **18.46 (0.53)** | 27.51 (0.03) | 29.35 (0.07) | **×1.67** |

Stale-vs-fresh delta for SRKL: −0.26 / ±0.00 / **+0.14** / −0.73 / −0.76 ROUGE-L, at **1.55× less wall-clock** (2.58 → 1.67). Every delta is within ~1–2 pooled std. §5.2: "the off-policy approach leads to a minimal performance drop while significantly improving computational efficiency. It achieves 2.2× to 3.4× faster training speed compared to the on-policy or mixed strategy."

**(b) Tab. 4 — replay applied to other objectives (same student/teacher, ROUGE-L).** This is the load-bearing result for "which objective tolerates staleness".

| Method (objective) | Dolly on- | Dolly off- | Δ | Self-Inst on- | Self-Inst off- | Δ | SNI on- | SNI off- | Δ |
|---|---|---|---|---|---|---|---|---|---|
| ImitKD (full-vocab `fkl`) | 21.63 | 20.62 | **−1.01** | 10.85 | 10.09 | **−0.76** | 19.94 | 18.04 | **−1.90** |
| GKD (generalized JSD, β=0.9) | 23.75 | 22.89 | **−0.86** | 12.73 | 12.78 | +0.05 | 26.05 | 24.97 | **−1.08** |
| DistiLLM (SRKL, α=0.1) | 26.37 | 26.12 | −0.25 | 13.14 | 13.16 | +0.02 | 28.24 | 28.20 | −0.04 |

Caption: "Off-policy significantly reduces the performance of ImitKD and GKD, as opposed to our proposed DISTILLM."

**(c) Tab. 9 — replay-ratio schedule `ζ` (SRKL, GPT-2 0.1B).**

| ζ | Rel. time | Dolly | Self | Vicuna | SNI | UNI |
|---|---|---|---|---|---|---|
| Constant (0.5) | ×2.13 | 26.32 | 12.99 | 17.31 | 27.45 | 28.64 |
| Increasing (t/T) — least stale late | ×2.39 | **26.44** | 12.42 | 17.60 | 26.80 | **29.90** |
| **Decreasing (1−t/T) — ours, most stale late** | **×1.67** | 26.11 | **13.14** | **18.46** | **27.51** | 29.35 |

App. E.5: "Despite the high bias error … of off-policy training, the performance of the constant or increasing ζ is similarly compared to our proposed decreasing manner thanks to our adaptive probability of using SGO."

**(d) Tab. 10 — buffer capacity (= staleness horizon), SRKL, GPT-2 0.1B.**

| Capacity `D_R` (sequences) | Dolly | Self | Vicuna | SNI | UNI |
|---|---|---|---|---|---|
| 250 | 24.71 | 11.95 | 16.79 | 24.88 | 26.73 |
| 500 | 25.32 | 12.46 | 17.64 | 25.69 | 27.00 |
| **1000** | 26.11 | **13.14** | **18.46** | **27.51** | **29.35** |
| 2000 | **26.48** | 12.88 | 17.49 | 26.51 | 28.61 |
| 4000 | 25.58 | 12.65 | 17.22 | 25.65 | 27.55 |

Non-monotone with an interior optimum — the paper's only direct staleness-horizon curve. App. E.8: "A capacity that is too small may lead to overfitting on a limited number of samples within `D_R`. Conversely, a capacity that is too large results in the inclusion of outdated SGO in the replay buffers, potentially introducing a high bias issue."

**(e) Wall clock (Fig. 5(d), §4.1):** "DISTILLM requires only 1.6× the training time of naïve KD, whereas other methods take 3 ∼ 7×." Abstract: "up to 4.3× speedup compared to recent KD methods." Fig. 2: SGO generation is "up to 80%" of total training time in the on-policy setting.

## Explanation offered

Three distinct mechanisms, all named in §3.2 / App. E.8:

1. **Policy-divergence bias (the cost of staleness).** "Off-policy reinforcement learning is prone to high bias error (Kumar et al., 2019; Lee et al., 2023), particularly when there is a significant divergence between past and current policies, using samples from the past policy becomes sub-optimal for updating the current policy." Mitigation is scheduling, not correction: "In the early training phase (i.e., when adaptive probability `φ` is small), where student model parameters rapidly evolve, we focus on using current SGOs with a high replay ratio to minimize bias error. • In the late stages of training (i.e., when `φ` is larger), as the student model nears convergence, we predominantly reuse stored SGOs from `D_R` with a small replay ratio."
   ⇒ Mechanism = **distribution shift between behavior and current student**, controlled by *how fast the student is still moving*.

2. **Convergence speed of the objective decides staleness tolerance.** "Synergy with SKL. Off-policy approach's success stems from the fast convergence speed of S(R)KL while other loss functions cannot be achieved. As Fig. 6 shows, both SKL and SRKL have a significant early-stage improvement, effectively leveraging the off-policy approach without high bias issues. This advantage is also evident in Tab. 4, where, unlike other baselines (Lin et al., 2020; Agarwal et al., 2024) that suffer performance drops when switching from on-policy to off-policy, our method maintains its efficacy." And §5.3: "Given that the JSD and KLD exhibit slower training speeds compared to our SRKL … we observe that the effectiveness of the off-policy approach is significantly lower than that of our proposed DISTILLM."
   ⇒ Mechanism = a **fast-converging objective front-loads its learning, so the samples that are replayed late carry little residual gradient signal** — staleness hurts least when the loss has already mostly converged. Note this is an argument about *the divergence*, and it is asserted + supported by Tab. 4, not proved.

3. **Buffer capacity trades overfitting against outdatedness** (App. E.8, quoted above) — an interior optimum at 1000, the only place the paper treats staleness as a tunable dial.

Separately, the *reason for using SGOs at all* is training–inference mismatch, and the *reason to use few of them* is teacher misgrounding: §2.2 "teacher models may experience a distribution mismatch between their training data and unfamiliar or inaccurate SGOs, potentially leading to misguidance on `q_θ`. As depicted in Fig. 1, such a mismatch can result in the teacher model assigning low validation loss to incorrect but shorter generations and high validation loss to longer but correct ones." (Support-mismatch on the *teacher* side, which is orthogonal to staleness but is why φ starts at 0.)

## Quotes

All §-located, verbatim:

- Abstract: "an adaptive off-policy approach designed to enhance the efficiency in utilizing student-generated outputs."
- §1: "Agarwal et al. (2024) employed on-policy distillation with SGOs, but it faces low sample efficiency and high generation time as it constantly prompts the student to produce new training sequences."
- §1: "Gu et al. (2024) introduced a policy optimization method to minimize reverse KLD between student and teacher distributions, yet this also compromises training efficiency by requiring generation from both models in every iteration."
- §1 (Contributions): "While using SGOs in KD is generally effective in improving performance, this approach significantly increases training time (Fig. 2) and makes it challenging to find the optimal proportion for using SGOs."
- §2.2: "as shown in Fig. 2, generating SGOs for every iteration proves computationally inefficient. Across all experiments, irrespective of the maximum sequence length (ranging from 64 to 256) of SGOs or the model size (from GPT-2 to OpenLLaMA2-3B), the SGO generation accounts for a considerable portion of the total training time, reaching up to 80%."
- §3.2: "To enhance efficiency, we replace the recently adopted on-policy approach (Gu et al., 2024; Agarwal et al., 2024) with an off-policy approach, employing a replay buffer (Mnih et al., 2015; Fedus et al., 2020), as illustrated in Fig. 4(c). In this buffer, we store SGOs from student models at a probability of `λ_R`, as indicated by the frequency of the dashed blue arrows. Subsequently, we randomly draw samples from this pool. We also replace the oldest samples of `D_R` with new ones once it reaches its maximum capacity. This off-policy strategy significantly improves the sample efficiency of KD with SGO, saving more resources than on-policy which constantly requires new data."
- §3.2: "Off-policy reinforcement learning is prone to high bias error (Kumar et al., 2019; Lee et al., 2023), particularly when there is a significant divergence between past and current policies, using samples from the past policy becomes sub-optimal for updating the current policy."
- §3.2: "In the early training phase (i.e., when adaptive probability `φ` is small), where student model parameters rapidly evolve, we focus on using current SGOs with a high replay ratio to minimize bias error."
- §3.2: "In the late stages of training (i.e., when `φ` is larger), as the student model nears convergence, we predominantly reuse stored SGOs from `D_R` with a small replay ratio."
- §3.2: "Hence, we can consistently maintain a small `λ_R` by applying a large `ζ` for relatively smaller `φ` values and vice versa, effectively balancing bias error reduction with sample efficiency."
- §3.2: "Off-policy approach's success stems from the fast convergence speed of S(R)KL while other loss functions cannot be achieved."
- §3.2: "we verify that our off-policy approach significantly improves the training efficiency with a negligible performance drop as depicted in Tab. 2 and Fig. 7."
- §5.2: "the off-policy approach leads to a minimal performance drop while significantly improving computational efficiency. It achieves 2.2× to 3.4× faster training speed compared to the on-policy or mixed strategy."
- §5.3: "Given that the JSD and KLD exhibit slower training speeds compared to our SRKL, as shown in Fig. 6, we observe that the effectiveness of the off-policy approach is significantly lower than that of our proposed DISTILLM."
- Fig. 4 caption: "This advantage is attributed to introducing a replay buffer and progressively decreasing a replay ratio `ζ := (1 − t/T)`, which consequently maintains small SGO generation frequency `λ_R := φ(1 − t/T)` during the entire training phase."
- Tab. 4 caption: "Off-policy significantly reduces the performance of ImitKD and GKD, as opposed to our proposed DISTILLM."
- App. E.5: "Despite the high bias error (Lee et al., 2023) of off-policy training, the performance of the constant or increasing `ζ` is similarly compared to our proposed decreasing manner thanks to our adaptive probability of using SGO."
- App. E.8: "A capacity that is too small may lead to overfitting on a limited number of samples within `D_R`. Conversely, a capacity that is too large results in the inclusion of outdated SGO in the replay buffers, potentially introducing a high bias issue as noted by Lee (2023) (Lee et al., 2023). In our experiments, a capacity value of 1000 demonstrates the most balanced performance overall."

## Not in the source

- **No staleness measured in optimizer steps, tokens, or "off-policyness".** Capacity is in *sequences* and the mixing rate in *probability*; the paper never reports how many gradient steps old a replayed sample is, nor any divergence between behavior and current student.
- **No importance correction and no discussion of whether one is needed** — the off-policy-RL citations (Kumar 2019, Fedus 2020, Lee 2023) are invoked for the *bias* framing but the standard IS remedy is never mentioned or ablated.
- **No `rkl_pg` under staleness.** MiniLLM (the only sampled-token policy-gradient RKL baseline) is compared on-policy only; Tab. 4's off-policy column covers ImitKD (`fkl`) and GKD (JSD), so the paper says nothing about whether a score-function estimator survives replay. This is exactly the gap our sweep is about.
- **No single-factor ablation of the two "adaptive off-policy" components at fixed compute.** Tab. 2's `Adaptive` vs `+Off-policy` rows differ in wall-clock by 1.55×; there is no "off-policy with the extra compute given back" control, so the reported −0.0 to −0.8 drop is a drop at *lower* cost, not a matched comparison.
- **No seed statistics on the staleness ablations.** Tab. 2 has 5 seeds; Tab. 4, 9, 10 report single numbers, and several of the interesting deltas (±0.05–0.3) are smaller than Tab. 2's own std.
- **No large-model staleness ablation** — Tab. 2/4/9/10 are all GPT-2 XL → GPT-2 0.1B; the OpenLLaMA2-7B→3B pair only appears with the full method.
- **Terminology:** reverse KL is used in our sense throughout (`D_RKL(p, q_θ) := D_KL(q_θ, p)`, §2.2) — **no direction flag needed**.

## Code-audit addenda (not stated in the paper)

- The buffer stores tokens only (`buffer.py` field names) — so no stale-logprob problem exists by construction; see `## Off-policy correction`.
- Replay is gated behind buffer warm-up (`finetune.py` L311) — `... or (r < adaptive_threshold and len(replay_buffer) < args.capacity)`. Until `|D_R| = capacity`, every SGO draw is freshly generated.
- The released `mixed` branch (used for the ImitKD / GKD baselines and the `Mixed` row of Tab. 2) generates a fresh batch, pushes it to the buffer, and then **trains on a batch sampled from the buffer**, not on the fresh one (`finetune.py` L303–309). At commit `d47e77f` there is no code path that trains on a freshly generated batch without buffer round-tripping *except* the adaptive fresh branch. Taken literally, the released "on-policy" baselines are partly stale; the Tab. 2 `Mixed` / Tab. 4 `on-` numbers may therefore not be reproducible from this commit as pure on-policy.
