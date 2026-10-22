# Ko, Chen, Kim, Ding, Liang, Zharkov, Yun 2025 — DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs

**arXiv 2503.07067v2** (30 May 2025; ICML 2025, PMLR 267; **Oral**, "top 1%" per the repo README, 2025-06-09). Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, Se-Young Yun (KAIST AI; Microsoft — Ko as Microsoft research intern). Sources: PDF full text `https://arxiv.org/pdf/2503.07067` (pypdf, 2026-09-14); arXiv HTML v2 (curl, 2026-09-14) for Table 12 and Algorithm 1 verbatim; code `github.com/jongwooko/distillm-2` @ `fe4cf9b` (cloned 2026-09-14) — `src/distillm_trainer.py`, `generate/generate_vllm.py`, `generate/reformat.py`, `training_configs/`, `README.md`.

```bibtex
@inproceedings{ko2025distillm2,
  author    = {Jongwoo Ko and Tianyi Chen and Sungnyun Kim and Tianyu Ding and Luming Liang and Ilya Zharkov and Se-Young Yun},
  title     = {Disti{LLM}-2: A Contrastive Approach Boosts the Distillation of {LLM}s},
  booktitle = {International Conference on Machine Learning},
  year      = {2025},
  eprint    = {2503.07067},
  archivePrefix = {arXiv}
}
```

## Tier

- **G** — stale student samples are used throughout (every student response is generated once per epoch from the *previous* epoch's checkpoint and replayed for the whole epoch), but the paper's main point is the contrastive SKL/SRKL-by-data-type loss, not the staleness. Data curation is §3.2 + App. D.1, one of three sub-components.
- **C** — a clean three-way side-by-side (`on-policy` / `batched on-policy` / `off-policy`) across six methods, Tab. 12, App. D.1 — **but with no explanation of its own**; the mechanism is inherited by citation ("shares the same takeaway as the adaptive off-policy approach in Ko et al. (2024)"). Not B.
- **D** — several quotable statements about replacing per-iteration on-policy sampling with batched generation (see `## Quotes`).
- **A, B, E, F** — no. No estimator-granularity argument; no mechanism of its own; academic paper; async is not the headline.

**Headline answer to the sweep question:** DistiLLM-2 **drops the replay buffer and the adaptive off-policy probability schedule entirely** and replaces them with **"batched on-policy"** generation — the student's responses for epoch *e* are all sampled once from `q_{θ_{e-1}}` before the epoch begins, then reused for every iteration in it. It is still async in our sense (the behavior policy is an earlier student checkpoint), just with a *different, coarser* staleness structure: one fixed lag per epoch instead of a mixed-age buffer.

## Setting

- **Students / teachers (teacher fixed in every experiment):**
  - Instruction-following: Qwen2-7B-Inst → Qwen2-1.5B; Mistral-7B-Inst → Danube2-1.8B; Gemma-2-9B-Inst → Gemma-2-2B (§4.1, Tab. 2). App. D.2 additionally: Gemma-2-9B-SimPO → Gemma-2-2B-it.
  - Math: Qwen2-Math-7B-Inst → Qwen2-Math-1.5B; Qwen2.5-Math-7B-Inst → Qwen2.5-Math-1.5B (§4.2, Tab. 3).
  - Code: DS-Coder-6.9B-Inst → DS-Coder-1.3B; Qwen2.5-Coder-7B-Inst → Qwen2.5-Coder-1.5B (§4.3, Tab. 4).
  - App. D.1 reproduction setup: GPT-2 XL → GPT-2 on `databricks-dolly-15k`, "in the code-base of DistiLLM", "All hyper-parameter setups are from (Ko et al., 2024)".
- **Teacher fixed?** Yes — teacher is the `ref_model` of a DPO-style trainer, `torch.no_grad()`, never updated (`distillm_trainer.py`); it also generates its responses once, offline.
- **Data / benchmarks:** 50k prompts sampled from UltraChat200k (§4.1) with teacher and student responses generated for each; AlpacaEval / Evol-Instruct / UltraFeedback with LLM-as-a-Judge (GPT-4o, GPT-4o-mini); GSM8K + MATH; HumanEval + MBPP (EvalPlus). Students are SFT'd on ground-truth responses first (App. C.2).
- **Framework:** HuggingFace `alignment-handbook` fork + DeepSpeed ZeRO-3; **vLLM** for generation; LoRA r=16 on all attention/MLP linear layers, lr 5e-5, effective batch **128**, 4×A100-80GB; **3 epochs** instruction-following, 2 epochs math/code (Tab. 11). "Unlike the previous version (Ko et al., 2024), we do not use language modeling loss on pre-training corpus for all experiments" (App. C.2).

## Async mechanism

**Replay buffer: dropped.** No `ReplayBuffer`, no `--capacity`, no `--replay-ratio`, no adaptive SGO probability `φ` anywhere in the paper or in `jongwooko/distillm-2` @ `fe4cf9b` (grep for replay/buffer/capacity returns only unrelated rate-limiter and mmap code).

**What replaces it — "batched on-policy" (§2 Summary & Connection, Algorithm 1 lines 3–9):** one generation pass per epoch.

```
3: for epoch e = 1, 2, ..., E do
4:   /* Sample batched on-policy responses */
5:   Sample responses y_t, y_s from teacher p(·|x) and student q_{θ_{e-1}}(·|x) for given prompt x
6:   Construct D_t = {(x, y_t, y_s)} for training dataset for training epoch e.
7:   Initialize θ_e ← θ_{e-1}
8:   for iteration τ = 1, 2, ..., T do
9:     Sample mini-batch B = {(x^(i), y_t^(i), y_s^(i))}_{i=1}^{|B|} from D_t
```

- **What is stale:** the student responses `y_s` — *token sequences only*, sampled from `q_{θ_{e-1}}`, i.e. the checkpoint frozen at the start of the epoch, then held fixed for all `T` iterations. Teacher responses `y_t` are also generated once but the teacher is fixed, so they are not stale in our sense.
- **No stale logits / log-probs are stored.** `generate/reformat.py` writes `{"prompt", "chosen": <teacher text>, "rejected": <student text>}` — text only. Both `p` and the *current* `q_θ` are re-evaluated on those tokens at training time (`distillm_trainer.py` `get_batch_logps`). As in DistiLLM, the staleness lives entirely in *which sequences are visited*, never in the per-token target.
- **Staleness range:** 0 → `T` optimizer steps, reset at every epoch boundary. With 50k prompts and effective batch 128, `T ≈ 390` steps, so the oldest sample in an epoch is ~390 steps stale; 3 epochs ⇒ 3 regeneration rounds (2 for math/code). The paper does not state `T` or any staleness metric — this is derived from §4.1 (50k prompts) + Tab. 11 (effective batch 128, 3 epochs).
- **In the released code the lag is even coarser:** generation is a *separate offline job* (`python generate/generate_vllm.py --model $MODEL_DIR ... --iter <n>`, temperature 0.8, top-p 0.95, one response per prompt), then `reformat.py`, then training; every `training_configs/*-distillm2.yaml` sets `num_train_epochs: 1`. So one shipped "round" = generate offline from the current checkpoint, train one full epoch on it, repeat. Nothing regenerates mid-run.
- **Stated motivation for the change (§2):** vLLM compatibility. Per-iteration `model.generate()` (DistiLLM's path) cannot use a paged-attention server; a once-per-epoch batch can.

## Objective

**Data-matched skew divergences, full-vocab, per-token — `fkl`-skew on teacher text, `rkl`-skew on student text.** No policy gradient, no sampled-token estimator.

Main loss (Eq. 2):

```latex
\mathcal{L}_{\textsc{DistiLLM-2}} := \frac{1}{2|\mathcal{D}|}\sum_{({\bm{x}},{\bm{y}}_t,{\bm{y}}_s)\sim\mathcal{D}}
\Big[(1-\beta)\,D^{(\alpha_t)}_{\mathrm{SKL}}({\bm{x}},{\bm{y}}_t) + \beta\, D^{(\alpha_s)}_{\mathrm{SRKL}}({\bm{x}},{\bm{y}}_s)\Big]
\tag{2}
```

Its un-curriculum'd form (Eq. 5, the "CALD" loss):

```latex
\mathcal{L}_{\mathrm{CALD}} = \frac{1}{2|\mathcal{D}|}\sum_{({\bm{x}},{\bm{y}}_t,{\bm{y}}_s)\sim\mathcal{D}} D^{(\alpha)}_{\mathrm{SKL}}({\bm{x}},{\bm{y}}_t) + D^{(\alpha)}_{\mathrm{SRKL}}({\bm{x}},{\bm{y}}_s).
\tag{5}
```

`SKL`/`SRKL` are DistiLLM's (@ko2024distillm §3.1): `D^{(α)}_SKL(p,q) = KL(p ‖ αp+(1−α)q)`, `D^{(α)}_SRKL(p,q) = KL(q ‖ (1−α)p+αq)`. **The stale data carries the reverse-KL term**: `SRKL` is applied to `y_s` precisely because `p(y_s|x) ≃ 0` there. §3.1: "we design the loss function for CALD, using SKL for teacher responses (i.e., `y_t`) where most of `p(y_t|x) ≫ 0` and using SRKL for student responses, `y_s`, where the most of `p(y_s|x) ≃ 0`."

Curriculum on `α` (Alg. 1 line 11), from the first-order approximation Eq. 7:

```latex
\alpha_t \leftarrow 1-(1-\alpha_0)\cdot\frac{m}{p({\bm{y}}_s|{\bm{x}})-q_\theta({\bm{y}}_s|{\bm{x}})},\qquad
\alpha_s \leftarrow 1-(1-\alpha_0)\cdot\frac{m}{p({\bm{y}}_t|{\bm{x}})-q_\theta({\bm{y}}_t|{\bm{x}})}
```
```latex
\log \frac{p({\bm{y}}|{\bm{x}})}{\tilde q^{(\alpha)}_\theta({\bm{y}}|{\bm{x}})} \simeq (1-\alpha)\cdot\big(p({\bm{y}}|{\bm{x}})-q_\theta({\bm{y}}|{\bm{x}})\big),
\qquad \tilde q^{(\alpha)}_\theta = \alpha p + (1-\alpha) q_\theta
\tag{7}
```

Linear ramp on the SRKL coefficient (Alg. 1 line 13): `β ← clip(e/E + τ/T, β₀, 1)`, `β₀ = 0.5` (Tab. 11); `α₀ = 0.1`, "we do not use curriculum-based update in 1st epoch" (Tab. 11).

**Code (`distillm_trainer.py` L1130–1235, L1070–1078):** full-vocab `.sum(-1)` over the mixture in log-space via `logsumexp`; `average_log_prob=True` ⇒ per-sequence token-mean then batch mean; teacher under `no_grad`. Two deviations from the paper worth recording: (i) the SRKL mixture **detaches the student inside the mixture** — `log_alpha_2 + vocab_logps.detach()` — while the outer `q·log q` stays live (no such detach in DistiLLM-1); (ii) at this commit `self.update_alpha = False` and `self.gradual_beta = False` are hardcoded (L507–509), so the released `distillm_v2` path runs **fixed `α_t = α_s = 0.1`, `β = 1.0`**, i.e. plain `SKL(teacher data) + SRKL(student data)` without the curriculum or the ramp.

## Off-policy correction

**None** — same as DistiLLM. No importance ratio, no clipping, no truncation, no staleness mask, no IcePop-style gate. Neither paper nor code contains any reweighting of the epoch-old student samples; the loss is applied identically regardless of how old `y_s` is. The only "correction" is the *structure*: nothing stale is stored except tokens, so a full-vocab divergence recomputed on them is exact at those prefixes.

(The `vocab_logps.detach()` inside the SRKL mixture is a gradient-path choice, not an off-policy correction.)

## Results: async vs sync

**Tab. 12 (App. D.1) — the only direct comparison. GPT-2 XL → GPT-2 on `databricks-dolly-15k`, ROUGE-L, DistiLLM code-base, DistiLLM hyperparameters.** Rows = data-staleness regime; columns = method. Same setup as Fig. 1(c).

| Setup | MiniLLM | GKD | DPO | DPKD | DistiLLM | DistiLLM-2 |
|---|---|---|---|---|---|---|
| on-policy (fresh every iteration) | 23.84 | 23.75 | – | 6.85 | 26.12 | **26.37** |
| **batched on-policy (ours; stale ≤1 epoch)** | 23.75 | 23.21 | 23.42 | 6.43 | 26.11 | 26.20 |
| off-policy (DistiLLM replay buffer) | 23.41 | 22.89 | 22.78 | 6.43 | 26.12 | 26.13 |

Deltas vs fully on-policy: DistiLLM-2 −0.17 (batched) / −0.24 (buffer); DistiLLM −0.01 / **+0.00**; MiniLLM −0.09 / −0.43; GKD −0.54 / −0.86. Ordering `on-policy ≥ batched ≥ off-policy` holds for every method except DistiLLM (flat within noise). No error bars, no throughput numbers, no wall-clock. Everything else in the paper (Tab. 2–5, Fig. 3) is run in the batched regime only, so there is **no sync baseline for the main results**.

Efficiency claim is qualitative only — §2: "this greatly reduces the computational cost of gathering training samples with minimal impact on student performance"; App. D.1: "does not suffer from severe performance degradation despite its significant efficiency."

For context, the main-table headline (Tab. 2, avg. win rate): DistiLLM-2 vs DistiLLM = 58.69 vs 56.35 (Qwen2-1.5B), 56.45 vs 52.37 (Danube2-1.8B), 74.83 vs 70.30 (Gemma-2-2B); §4.1 reports "+2.34%, +1.95%, and +4.53% on average" over the *second-best* method (which is Speculative KD, not DistiLLM, for Danube2).

## Explanation offered

**No mechanism of its own.** The paper asserts the result and points at DistiLLM:

- App. D.1: "Table 12 show that our 'batched' on-policy setup, which shares the same takeaway as the adaptive off-policy approach in Ko et al. (2024), does not suffer from severe performance degradation despite its significant efficiency."
- §2: "we adopt a batch approach (Rosset et al., 2024) that collects SGO ahead of every training epoch in our setup, rather than on-policy approach, which samples at every training iteration. This also ensures compatibility with advanced LLM inference techniques, such as vLLM (Kwon et al., 2023), thereby increasing generation efficiency and preserving the core philosophy of the adaptive off-policy approach."

So the stated mechanism is **inherited**: DistiLLM's "fast convergence of S(R)KL ⇒ low bias from stale samples" + "small `λ_R` keeps the behavior policy close" (@ko2024distillm §3.2). The credited precedent for once-per-epoch batching is Rosset et al. 2024 (Direct Nash Optimization), i.e. the iterative-DPO pattern, not an RL staleness analysis.

The paper's *own* mechanism work is about **which divergence goes on which data**, which is staleness-adjacent and is the interesting transferable claim: §3.1/§3.2 argue SRKL's "pushing-down effect is maximized at the tail of `p(·|x)` (i.e., `y_s`)" and that *low-quality* student samples are the right target for it — "using low-quality SGO samples on the SRKL side may be beneficial for our contrastive approach. The effectiveness of reduced training-inference mismatch via SGOs can be attributed to this edge of alignments" (§3.2). A stale `y_s` is, by construction, a *worse* sample than a fresh one; this argument says that is not obviously a problem for the reverse-KL term. The paper never draws that connection explicitly.

Speculative-decoding probe (§3.2, Fig. 2a), interpolating `y_s` toward `y_t` via the acceptance threshold `ε`, is the closest thing to a controlled study of "how student-like must the student data be": the SRKL term is **best at `ε = 0.0`, i.e. exactly `y_s`** — "the distilled model achieves the highest performance with `y_spec` at `ε = 0.0` (i.e., identical to `y_s`), as shown with green bars, although these responses are of the lowest quality."

## Quotes

- §2 (Data curation of KD in LLMs): "Ko et al. (2024) introduced an adaptive off-policy approach, which bridges offline and purely on-policy setups, striking a balance between the efficiency and efficacy of KD. This balanced strategy reuses SGO by introducing replay buffer, significantly improving computational efficiency while preserving the effectiveness of on-policy distillation. This approach has proven effective in subsequent works on preference alignment of LLMs (Rosset et al., 2024) as in more generalized version."
- §2 (Summary & Connection to our work): "On the data curation side, we adopt a batch approach (Rosset et al., 2024) that collects SGO ahead of every training epoch in our setup, rather than on-policy approach, which samples at every training iteration."
- §2: "This also ensures compatibility with advanced LLM inference techniques, such as vLLM (Kwon et al., 2023), thereby increasing generation efficiency and preserving the core philosophy of the adaptive off-policy approach. As shown in our preliminary results in Appendix D.1, this greatly reduces the computational cost of gathering training samples with minimal impact on student performance."
- §2: "To address the training inefficiency and low quality of SGO, which can lead to inaccurate teacher feedback in on-policy approaches (Lin et al., 2020; Agarwal et al., 2024) …"
- Algorithm 1, line 4: "/* Sample **batched on-policy** responses */"; line 5: "Sample responses `y_t, y_s` from teacher `p(·|x)` and student `q_{θ_{e−1}}(·|x)` for given prompt `x`"; line 6: "Construct `D_t = {(x, y_t, y_s)}` for training dataset for training epoch `e`."
- §3.1: "we design the loss function for CALD, using SKL for teacher responses (i.e., `y_t`) where most of `p(y_t|x) ≫ 0` and using SRKL for student responses, `y_s`, where the most of `p(y_s|x) ≃ 0`."
- §3.2: "Previous works (Agarwal et al., 2024) have discussed that while teacher responses provide useful information, they can cause training-inference mismatches. In contrast, SGOs, though lower in quality, effectively reduce such mismatches, leading to higher efficacy."
- §3.2: "This implies that using low-quality SGO samples on the SRKL side may be beneficial for our contrastive approach. The effectiveness of reduced training-inference mismatch via SGOs can be attributed to this edge of alignments."
- §3.2: "This implies that on the SKL side, mitigating the training-inference mismatch via SGOs does not always lead to performance improvement. Rather, strong guidance from the teacher response is highly related to the distillation performance."
- §3.3: "This improvement is achieved by compromising the imitation of the teacher's behavior, which is relatively hard to achieve, while focusing on directly obtaining feedback from SGOs, thereby effectively reducing the training–inference mismatch."
- App. D.1 (Setup): "We follow the experimental setup of Ko et al. (2024), which trained GPT-2 (Radford et al., 2019) on databricks-dolly-15k (Conover et al., 2023). All hyper-parameter setups are from (Ko et al., 2024). Note that we apply same experimental setup for the result in Figure 1(c)."
- App. D.1 (Results): "Table 12 show that our 'batched' on-policy setup, which shares the same takeaway as the adaptive off-policy approach in Ko et al. (2024), does not suffer from severe performance degradation despite its significant efficiency."
- Tab. 12 caption: "Application of 'batched' on-policy setup compared to fully on-policy and off-policy setup. We evaluate the student models in databricks-dolly-15k test set with ROUGE-L following Ko et al. (2024)."
- App. C.2: "Unlike the previous version (Ko et al., 2024), we do not use language modeling loss on pre-training corpus for all experiments."

## Not in the source

- **No explanation for why batched-stale works** — only a pointer to DistiLLM. Tier C, not B.
- **No throughput / wall-clock numbers at all** for batched vs on-policy vs off-policy. "significant efficiency" and "greatly reduces the computational cost" are never quantified, in contrast to DistiLLM's Fig. 7 / Tab. 9.
- **No staleness metric.** Never states `T` (iterations per epoch), token counts, a behavior-vs-current divergence, or how performance varies with the regeneration interval. Only the epoch boundary is a knob, and it is never swept.
- **No sync baseline for any main result.** Tab. 2/3/4/5 and Fig. 3 are all batched-regime; only App. D.1's GPT-2 reproduction has the three-way comparison.
- **Tab. 12 has no seeds/error bars** and its deltas (0.1–0.9 ROUGE-L) are comparable to DistiLLM's own Tab. 2 seed std (0.03–0.69).
- **No `rkl_pg` under staleness.** MiniLLM appears in Tab. 12 as a row but is the only policy-gradient method there, is run at GPT-2 scale only, and no estimator-level analysis is given. Same gap as DistiLLM-1.
- **No intermediate regeneration frequency** between "every iteration" and "every epoch" — the interesting middle (regenerate every *k* steps) is untested.
- **Reverse KL direction matches our convention** — `SRKL(p, q_θ) = KL(q_θ ‖ (1−α)p + αq_θ)`, student first. No flag needed.
- **Released-code gaps:** the curriculum `α` and the `β` ramp are hardcoded off at `fe4cf9b` (`update_alpha = False`, `gradual_beta = False`, L507–509), and the SRKL mixture detaches the student — neither is stated in the paper.
