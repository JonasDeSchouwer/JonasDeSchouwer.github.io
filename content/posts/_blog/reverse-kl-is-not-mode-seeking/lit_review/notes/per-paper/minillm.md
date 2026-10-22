# Gu, Dong, Wei & Huang 2024 — MiniLLM: Knowledge Distillation of Large Language Models

**arXiv 2306.08543v6** (31 Jan 2026; ICLR 2024; v6 retitled "MiniLLM: On-Policy Distillation of Large Language Models"). Yuxian Gu, Li Dong, Furu Wei, Minlie Huang (Tsinghua CoAI; Microsoft Research). Sources: PDF `https://arxiv.org/pdf/2306.08543` (curl + pypdf text, 2026-09-14); released code `https://github.com/microsoft/LMOps` under `minillm/` (shallow clone 2026-09-14).

```bibtex
@inproceedings{gu2024minillm,
  author    = {Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
  title     = {Mini{LLM}: Knowledge Distillation of Large Language Models},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  eprint    = {2306.08543},
  archivePrefix = {arXiv},
  note      = {arXiv v6 (2026) retitled ``MiniLLM: On-Policy Distillation of Large Language Models''}
}
```

## Async / off-policy (sweep 2026-09-14)

## Tier

- **G** — the student *does* train on rollouts from an earlier version of itself (4 PPO-style inner epochs over each rollout collection), and it *is* corrected (clipped importance ratio), but staleness is never the point: it is an unremarked hyperparameter inherited from InstructGPT, never varied, never ablated, never discussed.
- **D (weak)** — the only quotable statement is the hyperparameter sentence in App. B.1; there is no prose anywhere about staleness, lag, or why reuse is safe.
- Not **A**/**B**/**C** — no sync baseline, no lag sweep, no theory about staleness.
- Not **E**/**F** — academic paper, and async is not a contribution.
- **Useful as the counter-example to the "OPD must be exactly on-policy" framing**: the canonical reverse-KL OPD paper has been doing ~16 gradient steps per rollout batch since 2023, under a clipped ratio, without anyone calling it async.

## Setting

- Students: GPT-2 120M/340M/760M, OPT 1.3B/2.7B/6.7B, LLaMA-7B (§3.1). Teachers: GPT-2-1.5B, OPT-13B, LLaMA-13B respectively (§3.1); a separate scaling study uses GPT-2-125M ← GPT-2 340M/760M/1.5B.
- Teacher fixed: **yes** (frozen; used only for logits and for the mix-in distribution).
- Domain/benchmarks: instruction following. ~12.5K training examples from databricks-dolly-15k; eval on Dolly, SelfInst, VicunaEval, S-NI, UnNI; Rouge-L and GPT-4 preference. `D_PT` = OpenWebText (GPT-2) / RoBERTa corpus, for the auxiliary LM loss.
- Framework/code: released as `microsoft/LMOps` `minillm/`; the trainer is a fork of a trlx-style PPO loop (`minillm/trainer.py`, `minillm/losses.py`, `minillm/sampler.py`).
- Compute: "We train the model for 5000 steps"; "Distilling LLaMA-7B from LLaMA-13B takes less than 10 [h]ours on 16 GPUs" (App. B.1), V100-32G.

## Async mechanism

**Yes — multiple gradient steps per rollout batch.** Two distinct sources of off-policyness, both folded into one ratio:

1. **Policy lag from inner epochs (the in-scope one).** App. B.1, Phase 2, verbatim:

   > "We continuously train the model from **Phase 1** as described in Algorithm 2.3 using a learning rate 5e-6, a mini-batch size 64 in all cases. ... Similar to [OWJ+22], we collect 256 sentences at once and adopt 4 inner epochs when doing the policy optimization. The clipping rate ϵ is set to 0.2, and the max length of the model is 512."

   256 sentences collected per sampling phase ÷ mini-batch 64 = 4 minibatches per epoch × 4 inner epochs = **16 optimizer steps per rollout collection**, so a sample is used at a lag of 0–15 optimizer steps behind the current student. (The arithmetic is mine; the paper states only the three numbers.)

   Confirmed in code: `self.n_updates_per_batch = self.args.ppo_epochs` (`minillm/trainer.py:410`), the nested loop `for ppo_epoch in range(self.n_updates_per_batch): / for it, batch in enumerate(self.train_dataloader):` (`minillm/trainer.py:243-244`), and fresh rollouts only at the end of each outer pass: `def post_epoch_callback(self, epoch): self.store.clear_history(); self.sampler.run_sample(self.args.num_rollouts_per_device, self.global_iter_count)  # Collect more rollouts for training` (`minillm/trainer.py:386-391`). Released scripts set `--ppo-epochs 4 --num-rollouts 256` and `--cliprange 0.2` (e.g. `scripts/gpt2/minillm/train_base_xl.sh:70-71`, `scripts/llama/minillm/train_7B_13B.sh:75-76`; `--num-rollouts` default 256 and `--cliprange` default 0.2 in `arguments.py:154,156`).

   Rollouts *are* regenerated — but once per 16 updates, not once per update. There is no buffer that survives a collection: `self.store.clear_history()` discards the old rollouts before resampling.

2. **Teacher-mixed sampling (not staleness, but off-policy).** Even at lag 0 the behaviour distribution is not the student. Eq. (4), §2.2:

   ```latex
   \tilde{p}(y_t|y_{<t},x) = \alpha\cdot p(y_t|y_{<t},x) + (1-\alpha)\cdot q_\theta(y_t|y_{<t},x)
   ```

   with "α = 0.2 throughout the experiments in Eq. 4" (§3.1). Motivation: "We observe reward hacking when training with Eq. 2 because q_θ sometimes produces degenerated sentences y that receive high scores from the teacher (e.g., repeated phrases) during sampling, especially for small student models. To create a better sampling distribution, we mix the teacher and the student distribution at each time step" (§2.2).

**Discrepancy to flag when citing.** Algorithm 1 in the main text (§2.3) reads as **one** update per collection — `repeat / Sample a mini-batch of prompts from D and collect responses from p̃ to get S={(x_m,y_m)}_{m=1}^{M} / Compute (∇L)_Single ... / Compute (∇L)^Norm_Long ... / Update model parameters: θ ← θ − η[...] / until converge` — with no inner loop and no mention of epochs. Only App. B.1 and the code reveal the 4 inner epochs. Anyone reading the main text alone would conclude MiniLLM is strictly on-policy; it is not. (The clipping in Algorithm 1's `(∇L)^Norm_Long` line is the giveaway: a clipped ratio is pointless at zero lag with α = 0 — here it is doing double duty for the mix-in and the lag.)

**Staleness is never measured** — no lag metric, no KL between behaviour and current student, no "off-policyness" diagnostic, no sweep over `ppo_epochs`.

## Objective

Sequence-level **reverse KL**, direction matching our convention. §2.1 objective (Eq. 1, referenced as `L(θ) = KL[q_θ‖p]` in App. A.2), with q_θ = student, p = teacher: "unlike sequence-level KD that minimizes *forward* KLD, MINILLM that minimizes *reverse* KLD does not force q_θ to fit all y sampled from the teacher distribution p."

The estimator is a **hybrid**, which matters for our taxonomy. Eq. (3), §2.2:

```latex
\nabla L(\theta) = \mathbb{E}_{x\sim p_x, y\sim q_\theta(\cdot|x)}\Big[-\sum_{t=1}^{T}\nabla\,\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]\Big] + \mathbb{E}_{x\sim p_x, y\sim q_\theta(\cdot|x)}\Big[-\sum_{t=1}^{T} R_{t+1}\nabla\log q_\theta(y_t|y_{<t},x)\Big] = (\nabla L)_{\mathrm{Single}} + (\nabla L)_{\mathrm{Long}}
```

- `(∇L)_Single` is **full-vocab `rkl`** at the current position: "Note that E_{y_t∼q_θ(t)}[r_t] can be computed directly by summing over the vocabulary instead of using Monte-Carlo sampling and is derivable with respect to θ." (§2.2). Algorithm 1 spells it out: `(∇L)_Single = −(1/M) Σ_{x,y∈S} Σ_t w_t ∇ Σ_{y_t∈V} q_θ(y_t|y_{<t},x) log[p(y_t|y_{<t},x)/q_θ(y_t|y_{<t},x)]`.
- `(∇L)_Long` is **sampled-token `rkl_pg`** for the future-reward remainder, with length normalization (Eq. 6): `R^Norm_{t+1}`.

Base policy gradient, Eq. (2), §2.2:

```latex
\nabla L(\theta) = -\,\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}(R_t-1)\nabla\log q_\theta(y_t|y_{<t},x),\quad R_t=\sum_{t'=t}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}
```

Plus an auxiliary LM loss `L_PT = −E_{d∼D_PT} log q_θ(d)`; total update `θ ← θ − η[(∇L)_Single + (∇L)^Norm_Long + ∇L_PT]` (Algorithm 1).

So MiniLLM runs the full-vocab single-step term and the sampled-token long-horizon term **together, under the same importance weight** — the closest published thing to a controlled `rkl` vs `rkl_pg` comparison under off-policy data, except that it is never run as a comparison.

## Off-policy correction

**Yes: per-token importance ratio with PPO clipping at ε = 0.2.** This is the only one of the four sources with a real staleness correction.

Paper, Eq. (5) §2.2 — importance sampling for the mix-in:

```latex
(\nabla L)_{\mathrm{Single}} = -\,\mathbb{E}_{x\sim p_x,\,y\sim\tilde{p}(\cdot|x)}\Big[\sum_{t=1}^{T} w_t\,\nabla\,\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]\Big],\qquad
(\nabla L)_{\mathrm{Long}} = -\,\mathbb{E}_{x\sim p_x,\,y\sim\tilde{p}(\cdot|x)}\Big[\sum_{t=1}^{T} w_t R_{t+1}\nabla\log q_\theta(y_t|y_{<t},x)\Big]
```

with `w_t = Π_{t'=1}^{t} q_θ(y_{t'}|y_{<t'},x)/p̃(y_{t'}|y_{<t'},x)`. Per-sequence products are then **truncated to a single token** — a variance/bias trade the paper states explicitly:

> "However, w_t brings high variance in practice because it requires multiplying per-token importance weight over multiple time steps, and thus the variance of each step accumulates. Therefore, we approximately set w_t ≈ q_θ(y_t|y_{<t},x)/p̃(y_t|y_{<t},x) to reduce the variance of the estimator in Eq. 5." (§2.2)

So the correction is **per-token, not per-sequence**, by design.

Clipping, Algorithm 1 (§2.3):

```latex
(\nabla L)^{\mathrm{Norm}}_{\mathrm{Long}} = -\frac{1}{|M|}\sum_{x,y\in S}\sum_{t=1}^{T} R^{\mathrm{Norm}}_{t+1}\nabla\min\big[\rho_t(\theta),\ \mathrm{clip}(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon)\big],\quad \rho_t(\theta)=\frac{q_\theta(y_t|y_{<t},x)}{\tilde{p}(y_t|y_{<t},x)}
```

ε = 0.2 (App. B.1; `--cliprange` default 0.2).

**The code shows the ratio absorbs both off-policy sources in one factor**, which is the detail worth carrying into our own design:
- At collection time, `w = exp(logprobs − mix_logprobs)` where `mix_probs = (1 − α)·exp(rollout_logprobs) + α·exp(t_rollout_logprobs)` (`minillm/sampler.py:110-115`) — i.e. `w = q_{θ_old}/p̃`, computed once and stored.
- At update time, `log_ratio = (logprobs − old_logprobs) * mask; ratio = torch.exp(log_ratio.float()); ratio = ratio * w` (`minillm/losses.py:72-74`, with `old_logprobs = batch.logprobs` from `losses.py:133`) — i.e. `ratio = (q_{θ_new}/q_{θ_old}) · (q_{θ_old}/p̃) = q_{θ_new}/p̃ = ρ_t(θ)`.
- Then `pg_loss1 = -advantages * ratio; pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange); pg_loss = sum(max(pg_loss1, pg_loss2) * mask) / n` (`minillm/losses.py:88-94`).

So the **same clipped ratio corrects the teacher mix-in *and* the 4-inner-epoch policy lag** — the staleness correction is free, a side effect of already needing a ratio for the mix-in. Note the clip is applied only to `(∇L)^Norm_Long` (the sampled-token term); the full-vocab `(∇L)_Single` term carries the unclipped `w_t` only.

## Results: async vs sync

**None.** `ppo_epochs` is never varied; there is no 1-inner-epoch baseline and no clipping ablation. The only optimization ablation (Table 4, §3.4; GPT-2-125M ← GPT-2-1.5B, Rouge-L):

| Variant | Valid. R-L | Dolly R-L |
|---|---|---|
| MINILLM | 27.4 | 24.6 |
| w/o Length Norm. | 17.4 | 14.7 |
| w/o Teacher-Mixed | 22.3 | 20.4 |
| w/o Single-Step | 27.0 | 23.7 |

None of these isolates staleness. "w/o Teacher-Mixed" removes the mix-in (and with it the `w` factor) but leaves the 4 inner epochs and the `q_new/q_old` ratio in place, so it is not a sync/async contrast either.

## Explanation offered

Nothing about staleness. The paper's variance discussion is about the *estimator*, not about lag, but it is the closest mechanistic material and it cuts the way we care about:

> "However, policy gradient suffers from high variance and reward hacking, despite some subsequent solutions. Besides, we notice that R_t favors short sentences, which causes the student model to output empty responses." (§2.2)

> "[CPO+19] has found that the single-step generation quality r_t is critical to the training variance because the error in the front tokens accumulates along the whole sentence. To pay more attention to r_t, we re-write ∇L(θ) to decompose r_t from R_t and directly compute the gradient of E_{y_t∼q_θ(t)}[r_t] ... This decomposition gives a more precise and efficient estimation of the single-step generation quality, which reduces the variance during training and accelerates convergence." (§2.2)

> "From Figure 8, we also observe that the Single-Step Decomposition effectively reduces the variance of the training process, which also results in higher scores on the validation and test sets." (§3.4)

Read against our question: MiniLLM's own finding is that replacing the sampled-token estimator with the **full-vocab** one at the current position reduces variance and accelerates convergence — the same direction an "A"-tier staleness argument would need, but established for on-policy variance, not for lag.

On why reverse KL at all (§2.1): "it encourages the student to generate samples preferred by the teacher within its own capacities, which is more possible to achieve." And on exposure bias (§3.3): the student "samples from student models during training and thus alleviates exposure bias."

## Quotes

- App. B.1 (Phase 2): "Similar to [OWJ+22], we collect 256 sentences at once and adopt 4 inner epochs when doing the policy optimization. The clipping rate ϵ is set to 0.2" — **the only statement in the paper about multiple gradient steps per rollout batch.**
- §2.3: "we compute the gradients (∇L)_Single and (∇L)^Norm_Long based on Eq. 5 and Eq. 6, with a clipping strategy [SWD+17] added to further improve stability."
- §2.3: "The whole on-policy training pipeline is similar to Reinforcement Learning from Human Feedback (RLHF; OWJ+22)."
- §2.2: "To create a better sampling distribution, we mix the teacher and the student distribution at each time step" — Eq. (4), α = 0.2.
- §2.2: "We re-write (∇L)_Single and (∇L)_Long with importance sampling to get to an unbiased estimator of the gradient."
- §2.2: "However, w_t brings high variance in practice because it requires multiplying per-token importance weight over multiple time steps, and thus the variance of each step accumulates. Therefore, we approximately set w_t ≈ q_θ(y_t|y_{<t},x)/p̃(y_t|y_{<t},x) to reduce the variance of the estimator in Eq. 5."
- §2.2 (Gradient Derivation): "the gradient of the objective function L(θ) in Equation (1) can be derived using the Policy Gradient Theorem for *on-policy* optimization" — note the paper calls the whole procedure on-policy while running 4 inner epochs.
- Code: `self.n_updates_per_batch = self.args.ppo_epochs` (`minillm/trainer.py:410`); `# Collect more rollouts for training` (`minillm/trainer.py:391`).

## Related "off-policy" baselines

MiniLLM's baselines are all *teacher-data* baselines, not stale-student baselines (§3.1 "Baselines"): SFT w/o KD (ground-truth responses), KD (word-level forward KL on ground-truth responses, a.k.a. ImitKD-style supervised KD), and **SeqKD** ("fine-tunes the student model on the data generated by the teacher"). None involves student rollouts at any lag, so none is off-policy in the sense of this sweep. GKD (concurrent) is discussed only as a comparison point in the other direction — see `gkd.md` §5 for GKD's critique of MiniLLM's "stabilizing tricks".

## Not in the source

- **No sweep over `ppo_epochs` / inner epochs, and no 1-step-per-batch baseline.** The single most relevant number in the paper (4) is stated once in an appendix and never studied. So MiniLLM shows that reverse-KL OPD *tolerates* ~16 steps of lag under a clipped ratio, but gives no evidence of how much lag is affordable or what happens without the clip.
- **No clipping ablation** — the value of the off-policy correction is never measured, and the ε = 0.2 value is inherited, not tuned.
- **No staleness metric** (no lag counter, no behaviour-vs-current KL, no ratio-distribution statistics logged in the paper).
- **No separation of the two off-policy sources**: the teacher mix-in (α = 0.2) and the policy lag are corrected by one ratio and never disentangled, so "w/o Teacher-Mixed" cannot be read as a staleness ablation.
- No wall-clock/throughput motivation for the reuse — it is presented as an RLHF convention, not as a pipelining decision, and there is no generator/trainer overlap anywhere (rollouts and updates strictly alternate: `post_epoch_callback` → `run_sample`).
- No statement that the reverse-KL objective is or is not valid off-policy.
