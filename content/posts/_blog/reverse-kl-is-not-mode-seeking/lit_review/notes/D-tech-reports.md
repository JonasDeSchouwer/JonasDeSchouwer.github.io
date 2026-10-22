# Agent D result: Reverse-KL objectives in production/open LLM technical reports

All quotes verified against the primary source during the agent session (PDF downloaded and grepped, unless marked "via WebFetch"). Extracted full-text corpus for re-checking: scratchpad `txt/` dir.
Convention: reverse KL = D_KL(q‖p), q = student/policy, p = teacher/reference.

---

## PRIORITY: NVIDIA Nemotron family

### 1. Nemotron 3 Ultra — THE report with the failure quote

```bibtex
@misc{nvidia2026nemotron3ultra,
  title={Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning},
  author={{NVIDIA}}, year={2026}, eprint={2606.15007}, archivePrefix={arXiv},
  note={Also: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf}}
```
Link: https://arxiv.org/abs/2606.15007 (v1 submitted 2026-06-12; report dateline 2026-06-16)

**Stage: Multi-teacher On-Policy Distillation (MOPD), post-training (§3.3)** — explicit, verbatim **reverse-KL** production stage:

> "MOPD trains the student to match the corresponding teacher on states induced by the student itself. In the fully on-policy case, this corresponds to maximizing the negative reverse-KL objective 𝒥MOPD(𝜃) = Σᵢ 𝜆ᵢ E_{𝑞∼𝒟ᵢ, 𝑦∼𝜋_𝜃(·|𝑞)} [Σₜ log 𝜋_{𝑇ᵢ}(𝑦ₜ|𝑠ₜ) − log 𝜋_𝜃(𝑦ₜ|𝑠ₜ)] … Equivalently, at each prefix 𝑠ₜ, the student minimizes 𝐷KL(𝜋_𝜃(·|𝑠ₜ)‖𝜋_{𝑇ᵢ}(·|𝑠ₜ))." (§3.3.1, Eq. 1)

> "The dense distillation advantage is the sampled negative reverse-KL estimate with respect to the proximal policy: 𝐴̂ₜ = sg[ℓ^{𝑇ᵢ}ₜ − ℓ^{prox}ₜ]" (Eq. 2), optimized with PPO-style clipping and IcePop token masking (Eq. 3).

- **Eq. 3 surrogate detail (via WebFetch, 2026-08-17):** async-MOPD update maximizes 𝔼[Σₜ mₜ cₜ min(rₜ(θ)𝐴̂ₜ, clip(rₜ(θ), 1−ϵ, 1+ϵ)𝐴̂ₜ)] with rₜ(θ) = π_θ(yₜ|sₜ)/π_prox(yₜ|sₜ); mₜ = IcePop token mask (no explicit definition in report); cₜ = sg[π_prox(yₜ|sₜ)/π_behav(yₜ|sₜ)] — a second, stop-gradiented behavior→proximal IS correction. Rollouts come from a stale behavior policy π_behav; "student rollout generation, teacher scoring, and student optimization fully pipelined" (async); batches of 1,024 prompts × 1 rollout. No length normalization, temperature, or clamps specified for MOPD. Advantage used alone — dense distillation signal, not mixed with task/RL reward.

- Classification: **reverse KL**, matches our convention. Sampled from **student** (one-step off-policy asynchronous pipeline, §3.6.1; 192K is the *maximum generation length*, not a staleness figure — corrected 2026-09-14, see per-paper/nemotron-3-ultra.md), token-level signal, **single sampled token** per position (not full vocab).
- Empirical: Table 5 — large gains over RLVR student on agentic benchmarks (Terminal Bench 2.0: 44.5 RLVR → 50.8 MOPD1 → 54.0 MOPD2, i.e. +9.5 over RLVR (the +16.3 previously recorded here is MOPD1 minus SFT), recovery rate 172.7%; BrowseComp, SWE-Bench, TauBench, IFBench); smaller HLE gains attributed to "a limitation of the on-policy distillation setting rather than a failure of the teacher" (student rollouts out-of-support for teacher).

**THE FAILURE QUOTE — §3.3.5 "Limitations and Open Problems", bullet "Logit matching":**

> "A natural alternative to the sampled-token objective is distribution-level distillation, where the student is trained to match the teacher's predictive distribution over the top-𝑘 tokens or the full vocabulary at each prefix. In our preliminary experiments, these objectives did not improve MOPD performance and consistently underperformed the sampled-token objective on some agentic benchmarks such as Terminal Bench. We hypothesize that full-distribution matching may impose an overly strong local constraint on prefixes sampled from the student policy, particularly when those prefixes have limited support under the teacher distribution. In this regime, teacher logits can become poorly calibrated or less informative, and matching the full distribution may amplify noise from off-support states. By contrast, the sampled-token objective applies supervision only to realized actions and may therefore provide a more stable on-policy learning signal. Characterizing when broader distributional supervision is beneficial remains an open problem."

Framing: shipped objective **is** reverse KL — the failure is the **full-distribution / top-k estimator** on student-sampled prefixes vs the single-sampled-token estimator.

**Second stage, MTP Boosting (§3.4): explicit forward KL.** "Loss. We use a temperature-scaled forward-KL loss against the backbone's logits. The standard cross-entropy term against the gold token is disabled, so the head matches the backbone's full distribution rather than the one-hot label. … ℒMTP(𝜃) = (𝑇²/(𝑁mtp|𝒜|)) Σₖ Σₜ 𝐷KL(𝜎(𝑧ₜ₊ₖ/𝑇) ‖ 𝜎(𝑧^{mtpₖ}ₜ₊ₖ/𝑇))" T=2, teacher first.

- Codebase: NVIDIA-NeMo/RL — https://github.com/NVIDIA-NeMo/RL/blob/ultra-v3/docs/guides/nemotron-3-ultra.md (MOPD training guide).
- **Code trace (2026-08-17, firsthand at `ultra-v3` HEAD `7314528b73bd3449f4fd83e6101de8d5311b42ac`) — supersedes "code path not fully traceable":** `examples/configs/ultra/mopd.yaml` is headed "Nemotron 3 Ultra — MOPD Stage 1" and selects `adv_estimator: opd`; `nemo_rl/algorithms/advantage_estimator.py` `OPDAdvantageEstimator.compute_advantage` (L149–240) returns `(teacher_logprobs - prev_logprobs).detach()` — per-token, no return-to-go (grep for cumsum/reversed/flip over estimator + `loss_functions.py`: empty). Optional ORM blending `+ orm_advantage_weight * orm_advantages.detach()`. Report–code discrepancies: yaml configures `truncated_importance_sampling_type: tis` (clamp, min 0.2 / max 5) with `force_on_policy_ratio: true`, not the report's IcePop; IcePop exists as an option in `ClippedPGLoss` ("zero out tokens whose IS weight ∉ [min, max] (ref bounds: 0.5–5)"). `reference_policy_kl_penalty: 0.0` (advantage is the sole signal); `grpo.num_generations_per_prompt: 1`. `nemo_rl/algorithms/opd.py` handles teacher routing only ("Advantage computation lives in advantage_estimator.OPDAdvantageEstimator. IS truncation lives in loss_functions.ClippedPGLoss (ICE-POP mode)").
- **Provenance:** the `OPDAdvantageEstimator` docstring cites "the MOPD paper, arXiv:2601.02780" (its Eq. 8 = the advantage, Eq. 9 = ORM blending) — that is **Xiaomi's MiMo-V2-Flash technical report** (see `per-paper/mimo-v2-flash.md`), which introduces the MOPD paradigm (Jan 2026, predating both Nemotron reports).
- **Upstreamed (checked 2026-08-19):** the ultra-v3 code drop (single commit "Nemotron 3 Ultra", 2026-06-04) was merged to NVIDIA-NeMo/RL **main** as PR #2780 "feat: Multi-Teacher On-Policy Distillation (MOPD)" (2026-06-25) and is maintained there (#3340 teacher-cluster fix 2026-07-29; pydantic refactor #2518) — MOPD code is now fully public on main, no branch pin needed. Lineage: opd.py cites MiMo, **not** Cascade 2; Cascade 2 released no MOPD code (see §2).

### 2. Nemotron-Cascade 2 (arXiv 2603.19220)
```bibtex
@misc{nvidia2026nemotroncascade2, title={Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation},
  author={{NVIDIA}}, year={2026}, eprint={2603.19220}, archivePrefix={arXiv}}
```
- Full verbatim Eqs. 2–4 + surrounding prose: `per-paper/nemotron-cascade-2.md` (WebFetch 2026-08-19).
- MOPD inside Cascade RL (§4.4). Verbatim: "we define the token-level distillation advantage using reverse-KL as 𝑎ₜ^MOPD = log 𝜋_{domainᵢ}(𝑦ₜ|𝑠ₜ) − log 𝜋_train(𝑦ₜ|𝑠ₜ). … The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary." Truncated importance weighting 𝑤ₜ = sg[𝑟ₜ]1[𝜖_low ≤ 𝑟ₜ ≤ 𝜖_high] (Eqs. 2–4).
- **Eqs. 2–4 detail (via WebFetch, 2026-08-17):** 𝑟ₜ = 𝜋_train(𝑦ₜ|𝑠ₜ)/𝜋_inf(𝑦ₜ|𝑠ₜ), 𝜖_low = 0.5, 𝜖_high = 2.0 — "truncated importance weighting to account for train–inference mismatch" (rollouts from a separate inference engine 𝜋_inf, not a stale learner snapshot). Loss ℒ_MOPD = −𝔼[(1/|𝒱(y)|) Σₜ 𝑤ₜ · sg[𝑎ₜ^MOPD] · log 𝜋_train(𝑦ₜ|𝑠ₜ)], 𝒱(y) = "the set of valid response tokens retained by the token mask" — REINFORCE-style score-function surrogate (no PPO clipping, no IcePop), with per-sequence valid-token mean = length normalization. **Sampling temperature NOT stated for MOPD (correction 2026-08-19: exhaustive temperature check — every RL stage states T=1.0/top-p 1.0, MOPD states none, no global statement; the 2026-08-17 "temperature 1.0" claim here was wrong).** Distillation-only stage: advantage not mixed with task/RL reward. Contrast with Ultra's MOPD: PPO-clipped update vs π_prox + IcePop mask + sg[π_prox/π_behav] factor vs plain truncated-IS REINFORCE; advantage baseline π_prox vs π_train; async stale π_behav vs train–inference-mismatched π_inf.
- **Reverse KL**, student-sampled, token-level, sampled-token estimator.
- **No full-vocab failure comment (checked 2026-08-19, via WebFetch of arXiv HTML):** unlike Ultra §3.3.5, Cascade 2 has no ablation, limitation, or justification for the sampled-token choice — the only statement is the design fact after Eq. 2: "The log-probability difference is computed only on the student-sampled token rather than over the full vocabulary." No limitations section; estimator choice presented without comparison. Ultra §3.3.5 remains the sole Nemotron-family primary-source failure evidence for full-distribution RKL on student prefixes.
- Empirical: "Multi-domain On-policy Distillation (§4.4) serves as a critical stabilization point … effective at recovering benchmark performance that may have regressed during earlier, more specialized stages"; Fig. 3: reverse-KL curve converges in 40–50 steps; MOPD beats GRPO in training efficiency on AIME25.
- Their GRPO **removes** the KL term: "we remove KL divergence term entirely, which simplifies the GRPO objective to the standard REINFORCE objective" (§4.1.2); "set both the entropy loss coefficient and KL loss coefficient to 0" (§4.2.2).
- **Reproducibility artifacts (verified on HF, 2026-08-19):** final model `nvidia/Nemotron-Cascade-2-30B-A3B` (ungated), `nvidia/Nemotron-Cascade-2-SFT-Data`, `nvidia/Nemotron-Cascade-2-RL-data`; base = `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`. Paper: "We fully open source the model weights, training data, and methodological details" — training code and intermediate checkpoints excluded; no GPU-hours/cluster info anywhere. Stage order (Fig. 2): SFT → IF-RL → multi-domain RL → **MOPD** → RLHF → long-context RL → code RL → SWE RL. MOPD teachers (all self-derived, not released): math = initial SFT ckpt; RLHF teacher = separate RLHF-from-SFT branch; multi-domain = post-IF-RL+MDRL ckpt. Judges/RMs all open-weights: Qwen3-235B-A22B-Thinking-2507 (RLHF GenRM), Qwen3-235B-A22B-Instruct-2507 (long-context judge), GPT-OSS-120B (agentless SWE RM). Per-stage hparams in paper (RL stages: batch 128 × 16 rollouts, lr 3e-6; MOPD: batch 512, lr 2e-6→2e-7, 40–50 steps; contexts 16K–256K by stage).
- Codebase: Nemo-RL. **But no MOPD training code released (checked 2026-08-19):** the only Cascade 2 footprint in NVIDIA-NeMo/RL is `nemo_rl/data/datasets/response_datasets/nemotron_cascade2_sft.py` (SFT dataset loader); its REINFORCE + indicator-truncation MOPD variant matches neither public path (`distillation.py` top-k logit matching nor `opd.py` PPO-clipped OPD, see notes/E §6c). Ultra's ultra-v3/main OPD code is **not** built on Cascade 2 code — common ancestor is the MiMo-V2-Flash paradigm.

### 2b. Nemotron-Cascade 1 (arXiv 2512.13607, 2025-12-08) — pre-MOPD ancestor
```bibtex
@misc{nvidia2025nemotroncascade, title={Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models},
  author={{NVIDIA}}, year={2025}, eprint={2512.13607}, archivePrefix={arXiv}}
```
- Bases: **Qwen3-8B/14B-Base** (recipe branding, not Nemotron bases). Pipeline: 2-stage SFT (16K→32K) → Cascade RL: RLHF → IF-RL → Math RL → Code RL → SWE RL. **No weight-space distillation stage** (via WebFetch 2026-08-19) — teacher signal is data-level only (SFT on DeepSeek-R1(-0528) / Qwen3-235B-A22B samples). MOPD enters the Cascade line only in v2, post-MiMo. GRPO on-policy with **no KL divergence term** (consistent with Cascade 2's KL removal). RLHF RM = Qwen2.5-72B-Instruct BT.
- **Reproducibility contrast with Cascade 2 (HF-verified 2026-08-19):** v1 released **intermediate checkpoints** — `nvidia/Nemotron-Cascade-8B-Intermediate-ckpts` holds post-SFT/RLHF/IFRL/MathRL/CodeRL checkpoints — plus final models (`Nemotron-Cascade-8B` unified, `-8B/-14B-Thinking`), the RM (`nvidia/Qwen2.5-CascadeRL-RM-72B`), and per-stage datasets (SFT-Stage-1/2, RL-RLHF, RL-Math, RL-SWE, SFT-SWE, RM-Training). Ready-made same-family specialist checkpoints at 8B → usable off-the-shelf as MOPD-style teachers without retraining.

### 3. Minitron (arXiv 2407.14679) — classic anti-reverse-KL data point
```bibtex
@misc{muralidharan2024minitron, title={Compact Language Models via Pruning and Knowledge Distillation},
  author={Muralidharan, Saurav and Sreenivas, Sharath Turuvekere and Joshi, Raviraj and others},
  year={2024}, eprint={2407.14679}, archivePrefix={arXiv}}
```
- Prune-retrain KD (base-model compression). L_logits = (1/l) Σₖ Loss(pₖᵗ(x,τ), pₖˢ(x,τ)); total L = L_CLM + L_logits + α·L_is. Best practice #5: "Retrain exclusively with distillation loss using KLD instead of conventional training."
- **Forward confirmed**: Table 15 caption: "The loss component Llogits alone with **forward KLD loss** outperforms the rest."
- **Reverse-KL negative result, verbatim (§4.3)**: "Choice of Loss Function (Best Practice #5): we experiment with Kullback-Leibler divergence (KLD), MSE, cosine similarity and reverse KLD (R-KLD) to compute Llogits. Recent work has shown R-KLD [15, 27] to be a better fit than KLD in the SFT/instruction-following setting, and Agarwal et al. [2] claim the choice of loss is task-dependent. We observe from Table 15 and 16 that KLD is the best choice for pruned base model training."
- App. A.3: "we see that KLD consistently outperforms R-KLD, cosine and MSE. WSL-KD [58] also performs inferior to KLD." Numbers: Table 15 — KLD LM loss 2.117 vs RKLD 2.140; Table 16 (8B depth-pruned) — KLD 2.155 vs **RKLD 2.665**.
- Dataset-sampled (teacher-forced), token-level, full vocab. Code pointers: https://github.com/NVlabs/Minitron (Megatron-LM/NeMo & TensorRT-Model-Optimizer KD).

### 4. Minitron-in-practice (arXiv 2408.11796)
```bibtex
@misc{sreenivas2024llmpruning, title={LLM Pruning and Distillation in Practice: The Minitron Approach},
  author={Sreenivas, Sharath Turuvekere and Muralidharan, Saurav and others}, year={2024}, eprint={2408.11796}, archivePrefix={arXiv}}
```
- "Distillation is then performed by minimizing KL divergence on the logits of the teacher and the pruned student model." (Fig. 2 caption) — **UNSPECIFIED** in this paper (forward per companion Minitron paper). Adds "teacher correction" (~127B tokens).

### 5. Nemotron-H (2504.03624) and Nemotron Nano 2 (2508.14444)
```bibtex
@misc{nvidia2025nemotronh, title={Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models},
  author={{NVIDIA}}, year={2025}, eprint={2504.03624}, archivePrefix={arXiv}}
@misc{nvidia2025nemotronnano2, title={NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model},
  author={{NVIDIA}}, year={2025}, eprint={2508.14444}, archivePrefix={arXiv}}
```
- Both (§3.4 / §4.3 "Retraining with Distillation"): "we thus adopt logit-based distillation for continued training, employing **forward KL divergence loss** exclusively during the accuracy recovery phase (see §3 of the Minitron paper … for the distillation loss formulation)."
- **Forward KL**, dataset-sampled, token-level, full vocab.

### 6. Llama-Nemotron (2505.00949)
```bibtex
@misc{nvidia2025llamanemotron, title={Llama-Nemotron: Efficient Reasoning Models},
  author={{NVIDIA}}, year={2025}, eprint={2505.00949}, archivePrefix={arXiv}}
```
- Post-NAS "Knowledge Distillation and Continued Pretraining" (§2.2): LN-Super 40B tokens KD on Distillation Mix; LN-Ultra 65B KD + 88B CPT. Direction **UNSPECIFIED** (block-wise local distillation per Puzzle paper).
- GRPO with "KL penalty 𝛽 of 1e-3" (no estimator specified).
- RL-vs-distillation (§5): "Consistent with the findings of DeepSeek-AI et al. (2025), our preliminary experiments indicate that reinforcement learning (RL) yields suboptimal results for smaller models compared to distillation."

### 7. Nemotron 3 Nano (2512.20848), Super (2604.12374) and family overview (2512.20856)
- Nano 3: **no weight-space distillation stage** (pretraining + SFT + RLVR/RLHF). Only data-level: "We distill trajectories from three open-source agent harnesses … using Qwen3-Coder-480B-A35B-Instruct … as the teacher model" (SFT-on-teacher-samples). Family overview: no KL/distillation mentions (checked).
- Nemotron 3 Super (2604.12374v1, 15 Apr 2026): **no distillation stage of any kind** — the string "KL" does not occur in the report. Post-training = SFT (two stages) + RLVR/SWE-RL/RLHF + MTP healing. Teachers are data generators only: "We distill trajectories from the OpenHands agent harness using Qwen3-Coder-480B-A35B-Instruct as the teacher model" (§3.1.1); datasets "refresh[ed] … with new teacher models (DeepSeek v3.2, Kimi K2)" (§3.1.1). Self-generated stage without a teacher: "After the main SFT stage, we add a short semi-on-policy SFT stage of 350 steps for inference-time budget control, where we collect roll-outs from the model and truncate 12% of reasoning traces to random reasoning budgets" (§3.1.3). Async infra is RL-only (§3.2.4–§3.2.5, one-step-off-policy async GRPO + masked importance sampling + in-flight weight updates) — see notes/H-async.md.
- NVFP4 QAD report (arXiv 2601.20088, applied to Nemotron 3 Nano 30B-A3B): "ℒQAD = DKL(p_teacher‖p_student)" — **forward KL**, BF16 model as frozen teacher (via WebFetch).

---

## The rest of the sweep

### Qwen3 (2505.09388)
```bibtex
@misc{qwen2025qwen3, title={Qwen3 Technical Report}, author={{Qwen Team}}, year={2025}, eprint={2505.09388}, archivePrefix={arXiv}}
```
- "Strong-to-Weak Distillation" (§4.5), two phases, verbatim:
  - "(1) Off-policy Distillation: … we combine the outputs of teacher models generated with both /think and /no think modes for response distillation." → **SFT-on-teacher-samples**.
  - "(2) On-policy Distillation: In this phase, the student model generates on-policy sequences for fine-tuning. Specifically, prompts are sampled, and the student model produces responses in either /think or /no think mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence." → student-sampled, logit alignment; **direction UNSPECIFIED** (no equation anywhere; this is the only KL sentence).
- Compute quotes: "it achieves these gains with much greater training efficiency, requiring only 1/10 of the GPU hours compared to the four-stage training method"; "distillation achieves significantly better performance than reinforcement learning while requiring approximately only 1/10 of the GPU hours. Furthermore, distillation from teacher logits enables the student model to expand its exploration space and enhance its reasoning potential, as evidenced by the improved pass@64 scores" (Discussion, Table 21).
- No training code released for this stage. Re-verified 2026-08-19 (via WebFetch of github.com/QwenLM/Qwen3): the official repo is inference/deployment-only — docs, examples, quantization, eval; for training it explicitly defers to third-party frameworks ("We advise you to use training frameworks, including Axolotl, UnSloth, Swift, Llama-Factory, etc., to finetune your models with SFT, DPO, GRPO, etc."). No distillation or post-training pipeline code anywhere in the org's Qwen3 repos.

### Gemma 2 (2408.00118)
```bibtex
@misc{gemmateam2024gemma2, title={Gemma 2: Improving Open Language Models at a Practical Size},
  author={{Gemma Team}}, year={2024}, eprint={2408.00118}, archivePrefix={arXiv}}
```
- Pretraining KD (§3.2), verbatim: "we learn smaller models by distilling from the probability given by the teacher of each token 𝑥 given its context 𝑥_𝑐, i.e., 𝑃_𝑇(𝑥|𝑥_𝑐). More precisely, we minimize the negative log-likelihood between the probabilities from the teacher and the student: min_{𝑃_𝑆} Σ_𝑥 −𝑃_𝑇(𝑥|𝑥_𝑐) log 𝑃_𝑆(𝑥|𝑥_𝑐)". "replaces the one-hot vector seen at each token with the distribution of potential next tokens computed from a large model"; "train for large quantities of tokens with distillation in order to simulate training beyond the number of available tokens" (2B/9B >50x compute-optimal).
- **Forward KL** (CE to teacher soft targets). Dataset contexts, token-level, full vocab.
- Post-training SFT: "We also run distillation from the teacher on the student's distribution (Agarwal et al., 2024; Gu et al., 2024)." → **on-policy distillation on student samples, direction UNSPECIFIED** (citations = GKD, MiniLLM).
- Tables 6–7: distilled 2B/9B beat from-scratch. No codebase.

### Gemma 3 (2503.19786)
```bibtex
@misc{gemmateam2025gemma3, title={Gemma 3 Technical Report}, author={{Gemma Team}}, year={2025}, eprint={2503.19786}, archivePrefix={arXiv}}
```
- Pretraining (§2.2), verbatim: "Distillation. We sample 256 logits per token, weighted by teacher probabilities. The student learns the teacher's distribution within these samples via cross-entropy loss. The teacher's target distribution is set to zero probability for non-sampled logits, and renormalized."
- **Forward KL/CE**, dataset contexts, **sampled 256-logit support**. Post-training: "an improved version of knowledge distillation" + RL — **UNSPECIFIED**.
- Ablation (§5.4): "A common finding is that, to train a small model, it is preferable to distill from a smaller teacher. … Smaller numbers means distilling from a larger teacher is better" — large teacher wins at longer horizons.

### Llama 3.2 1B/3B (Meta blog; no arXiv)
```bibtex
@misc{meta2024llama32, title={Llama 3.2: Revolutionizing edge AI and vision with open, customizable models},
  author={{Meta AI}}, year={2024}, howpublished={\url{https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/}}}
```
- Verbatim (via WebFetch): "we incorporated logits from the Llama 3.1 8B and 70B models into the pre-training stage of the model development, where outputs (logits) from these larger models were used as token-level targets"; "structured pruning in a single shot manner from the Llama 3.1 8B"; "Knowledge distillation was used after pruning to recover performance."
- Token-level logit distillation, direction **UNSPECIFIED** (presumably forward). SmolLM2 paper corroborates: "Llama3.2-1B … trained using distillation on 9 trillion tokens". Blog-level evidence.

### DeepSeek-R1 (2501.12948; current post-Nature-revision arXiv version)
```bibtex
@misc{deepseekai2025r1, title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={{DeepSeek-AI}}, year={2025}, eprint={2501.12948}, archivePrefix={arXiv}}
```
- Distillation into Qwen/Llama students = pure SFT, verbatim (App. F): "we fine-tune open-source foundation models such as Qwen … and LLaMA … using a curated dataset comprising 800,000 samples generated with DeepSeek-R1. … For distilled models, we apply only SFT and do not include an RL stage, even though incorporating RL could substantially boost model performance." (App. B.4.3: 2–3 epochs.)
- **SFT-on-teacher-samples**. No KL/KD loss anywhere.
- RL-vs-distillation (App., near Table 16): "First, distilling more powerful models into smaller ones yields excellent results, whereas smaller models relying on the large-scale RL mentioned in this paper require enormous computational power and may not even achieve the performance of distillation. Second, while distillation strategies are both economical and effective, advancing beyond the boundaries of human intelligence may still require more powerful base models and larger-scale reinforcement learning."

### Apple Intelligence Foundation Models 2024 (2407.21075) and 2025 (2507.13575)
```bibtex
@misc{apple2024afm, title={Apple Intelligence Foundation Language Models}, author={{Apple}}, year={2024}, eprint={2407.21075}, archivePrefix={arXiv}}
@misc{apple2025afm, title={Apple Intelligence Foundation Language Models: Tech Report 2025}, author={{Apple}}, year={2025}, eprint={2507.13575}, archivePrefix={arXiv}}
```
- 2024 core pretraining AFM-on-device (§3.2.1), verbatim: "a distillation loss is used by replacing the target labels with a convex combination of the true labels and the teacher model's top-1 predictions, (with 0.9 weight assigned to the teacher's labels), training for a full 6.3T tokens. … adding distillation boosts MMLU and GSM8K by about 5% and 3% respectively." → **top-1 hard-label distillation** (degenerate forward-CE).
- 2024 continued pretraining (§3.2.2), mild negative: "We did not find a distillation loss to be helpful here for AFM-on-device, unlike in core pre-training."
- 2024 RLHF: committee rejection-sampling distillation (>1M responses, SFT-on-teacher-samples); MDLOO Eq. 1 = "r_ϕ(x,y) − βD_KL(π_θ(·|x)‖π_ref(·|x))" + mirror-descent "−γD_KL(π_θ(·|x)‖π_θk(·|x))" — rare production report writing out the reverse-KL direction in RLHF.
- 2025 (§4.3): dense on-device "retrained … for the last 10% of tokens (about 1.4T) using a distillation loss from the MoE teacher. This new pipeline not only reduced the cost of training the large distillation teacher by 90% but also greatly improved the efficiency of producing teacher logits." Direction **UNSPECIFIED**.

### Others (checked, full text)
- **MiniCPM (2404.06395)**: no KD/KL stage (SFT + DPO).
- **Hunyuan-Large (2411.02265)**: zero KD/KL mentions. **Hunyuan-TurboS (2505.15431)**: no KD; GRPO stage II: "the KL divergence penalty coefficient is increased to mitigate issues like catastrophic forgetting or performance collapse" (no direction/estimator).
- **GLM-4 (2406.12793)**: no KD; alignment deferred to ChatGLM-RLHF paper.
- **Phi-4 (2412.08905)**: no logit KD; "previous models in the Phi family largely distill the capabilities of a teacher model (specifically GPT-4)" (synthetic-data SFT); "phi-4 substantially surpasses its teacher model on STEM-focused QA … our data-generation and post-training techniques go beyond distillation."
- **SmolLM2 (2502.02737)**: no distillation stage for itself.
- **Falcon 3** (HF blog): "we developed Falcon3-1B-Base and Falcon3-3B-Base by leveraging pruning and knowledge distillation techniques, using less than 100GT of curated high-quality data" — **UNSPECIFIED**, blog-level.
- **InternLM2 (2403.17297)**: no KD; COOL-RLHF PPO "KL divergence coefficient to 0.01". InternLM2.5: no report.
- **Yi (2403.04652)**: no distillation stage.
- **OLMo 2 (2501.00656)**: no KD ("Better performance could also be achieved by distilling from a more powerful model, a strategy used by the smaller Gemma 2 models"); RLVR "with a KL beta of 0.1" (Tülu 3 objective). Code: allenai/open-instruct.
- **Kimi K2 (2507.20534)**: no KD. RL (§3.2.3, OPMD): "L_RL(θ) = E[ (1/K) Σᵢ ( r(x,yᵢ) − r̄(x) − τ log(π_θ(yᵢ|x)/π_old(yᵢ|x)) )² ]" — squared-loss log-ratio regularizer to previous policy, **other** (not explicit KL).
- **Tülu 3 (2411.15124)**: RLHF/RLVR Eq. 4 "r_ϕ(x,y) − βKL[π_θ(y|x)‖π_ref(y|x)]" — **reverse-KL penalty** (policy first), boilerplate; no estimator named. Code: allenai/open-instruct.
- **Llama 3 (2407.21783)**: no distillation ("An alternative path is to distill larger models into smaller ones, as in Phi" — what they did NOT do); SFT + DPO.
- **ERNIE 4.5**: only "self-distillation and multimodal transformation" as data synthesis; no KD/KL objective.
- **Seed-OSS**: no report; card describes no KD/KL stage.
- **Mistral/Ministral**: no training methodology disclosed.
- **Grok (xAI)**: model cards contain zero distillation/KL/teacher mentions (Grok 4 Fast card grep-verified).
- **GPT-4o mini**: no report; API "Model Distillation" product = SFT on frontier outputs.
- **Gemini 1.5 Flash (2403.05530)**: verbatim: "is also online distilled (Agarwal et al., 2024b; Anil et al., 2018; Beyer et al., 2021; Bucila et al., 2006; Hinton et al., 2015) from the much larger Gemini 1.5 Pro model." Direction **UNSPECIFIED**; "online distilled" + GKD citation hints on-policy. Nothing for Gemini 2.x Flash.

---

## Summary table

| Report | Stage | Direction (our convention) | Sampled from | Evidence |
|---|---|---|---|---|
| Nemotron 3 Ultra (2606.15007) | MOPD post-training | **reverse KL** (sampled-token) | student rollouts, token-level, single token | verbatim eq., primary |
| Nemotron 3 Ultra | MTP Boosting | **forward KL** (T=2) | own rollouts vs frozen backbone, full vocab | verbatim eq., primary |
| Nemotron 3 Ultra §3.3.5 | (ablation) full-vocab/top-k logit matching | full-distribution reverse-KL variant — **failed** | student prefixes | verbatim, primary |
| Nemotron-Cascade 2 (2603.19220) | MOPD | **reverse KL** (sampled-token) | student, token-level | verbatim eq., primary |
| Minitron (2407.14679) | prune-retrain KD | **forward KL** (R-KLD tried, worse) | dataset, full vocab | verbatim, primary |
| Nemotron-H / Nano 2 | prune-retrain KD | **forward KL** (explicit) | dataset, full vocab | verbatim, primary |
| Llama-Nemotron (2505.00949) | post-NAS KD | UNSPECIFIED | dataset | verbatim, primary |
| NVFP4 QAD (2601.20088) | quantization recovery | **forward KL** | data, full vocab | via WebFetch |
| Qwen3 (2505.09388) | off-policy distill | SFT-on-teacher-samples | teacher, sequence | verbatim, primary |
| Qwen3 | on-policy distill | **UNSPECIFIED KL** on logits | student rollouts | verbatim, primary |
| Gemma 2 (2408.00118) | pretraining KD | **forward KL** (CE to soft targets) | dataset, full vocab | verbatim eq., primary |
| Gemma 2 | post-training SFT | UNSPECIFIED "on the student's distribution" (cites GKD, MiniLLM) | student | verbatim, primary |
| Gemma 3 (2503.19786) | pretraining KD | **forward KL/CE**, 256 sampled logits | dataset, sampled vocab | verbatim, primary |
| Llama 3.2 1B/3B (blog) | pretraining KD after pruning | UNSPECIFIED (logit targets) | dataset | blog-level |
| DeepSeek-R1 | distill to small models | SFT-on-teacher-samples | teacher, 800k seqs | verbatim, primary |
| Apple 2024 | core pretrain (on-device) | teacher top-1 label mix | dataset | verbatim, primary |
| Apple 2024 | RLHF (MDLOO) | reverse-KL penalty, written out | student | verbatim eq., primary |
| Apple 2025 | last-10%-tokens retrain | UNSPECIFIED (MoE teacher logits) | dataset | verbatim, primary |
| Gemini 1.5 Flash | "online distilled" from Pro | UNSPECIFIED | unspecified | verbatim, primary |
| Falcon 3 (blog) | pruning+KD 1B/3B | UNSPECIFIED | unspecified | blog-level |
| Phi-4, MiniCPM, Hunyuan, GLM-4, SmolLM2, InternLM2/2.5, Yi, OLMo 2, Kimi K2, Llama 3, ERNIE 4.5, Seed-OSS, Ministral, Grok, GPT-4o mini | — | no KD/KL distillation stage described (checked) | — | primary (except last four: card/blog-level) |

## Direct answers

**(a) When are production models trained with reverse KL?**
1. **On-policy distillation as a post-training stage** — the only production setting where reverse KL is explicit and central: Nemotron 3 Ultra MOPD, Nemotron-Cascade 2 MOPD (both write "reverse-KL", use single-sampled-token estimator as dense advantage in PPO/REINFORCE-style update on student rollouts, never full-vocab integrand). Gemma 2's post-training "distillation … on the student's distribution" and Gemini 1.5 Flash "online distilled" plausibly in family but unstated. Qwen3 on-policy phase: unspecified-direction KL on student-sampled sequences.
2. **As the RLHF regularizer** — universal boilerplate: βD_KL(π_θ‖π_ref) reverse by construction (Apple 2024 MDLOO and Tülu 3 write direction; InternLM2, Llama-Nemotron, Hunyuan-TurboS, OLMo 2 only coefficients; Nemotron-Cascade 2 removes KL; none name an estimator).
3. **Never in pretraining-scale distillation**: every full-distribution teacher-forced production KD stage (Gemma 2/3, Minitron, Nemotron-H, Nano 2, QAD, Apple, Llama 3.2) is forward or unspecified.

**(b) Failure/negative quotes (verbatim):**
1. Nemotron 3 Ultra §3.3.5 (full quote above) — full-distribution estimator of reverse KL failed; sampled-token reverse KL is the shipped method.
2. Minitron §4.3/A.3 — "KLD is the best choice for pruned base model training"; Table 16 KLD 2.155 vs RKLD 2.665.
3. Apple 2024 §3.2.2 — "We did not find a distillation loss to be helpful here for AFM-on-device, unlike in core pre-training." (distillation generally, not direction-specific)
4. Research corroboration (via WebFetch, unverified against full papers): Entropy-Aware On-Policy Distillation (2603.07079): RKL's "mode-seeking property" "reduces generation diversity and yields unstable learning signals when the teacher distribution has high entropy." "Your Teacher Can't Help You Here" (2605.30833): "the teacher-dependent corrective signal in reverse-KL distillation weakens, causing student drift to compound across long reasoning chains." Rethinking OPD (2604.13016): OPD fails unless "the student and teacher … share compatible thinking patterns."
5. Distillation-vs-RL (not direction): DeepSeek-R1, Llama-Nemotron quotes above; Qwen3 distillation beats RL at 1/10 GPU-hours.

## Terminology flags
- No surveyed report uses "reverse KL" contrary to our convention. Minitron's bare "KLD" = forward (explicit in Table 15 caption); QAD writes D_KL(p_teacher‖p_student) but calls it "KL divergence"; Qwen3/Minitron-practice direction genuinely unrecoverable from text.
- Nemotron MOPD frames reverse KL as a negative reward/advantage (log π_T − log π_θ, stop-gradient) inside a policy-gradient surrogate, not a differentiated divergence loss — the estimator (sampled-token vs full-distribution) was the decisive design choice.

## Leads not verified
- TML blog cited as prior work by both NVIDIA MOPD papers (citation verified in Ultra reference list).
- Standalone "Nemotron 3 Super" report — not located.
- DRKL (2604.00223), OPD survey (2604.00626) — snippet evidence only.
- Puzzle (2411.19146) for Llama-Nemotron's block-wise loss direction.
- Grok/OpenAI distillation testimony — press only; ERNIE 5.0 / GLM-5 / Kimi-K3-era not fully swept. DeepSeek-V4 now has a per-paper dossier ([[deepseek-v4]], arXiv HTML, 2026-08-19).
