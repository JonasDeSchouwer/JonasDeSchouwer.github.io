# Kimi Team 2026 — Kimi K3: Open Frontier Intelligence

**arXiv 2607.24653v2** (v1 27 Jul 2026, v2 7 Aug 2026; v2 current as of 2026-09-14). Kimi Team (Moonshot AI). Sources: PDF `https://arxiv.org/pdf/2607.24653` and arXiv HTML `https://arxiv.org/html/2607.24653` (curl + local parse, 2026-09-14); Eq. 15 taken from the MathML `alttext` LaTeX; abs page for version history (2026-09-14). Section ids below are the LaTeXML ids of the HTML (e.g. `S4.SS1.SSS3.p1.1` = §4.1.3, first paragraph).

```bibtex
@article{kimi2026k3,
  title   = {Kimi {K3}: Open Frontier Intelligence},
  author  = {{Kimi Team}},
  journal = {arXiv preprint arXiv:2607.24653},
  year    = {2026}
}
```

## Tier

- **D** — the report makes an explicit, quotable connection between its OPD objective and stale-rollout infrastructure: "This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks." (§4.1.3, `S4.SS1.SSS3.p1.2`), and §4.1.2 supplies the staleness vocabulary it inherits ("data staleness", "extreme off-policy regime", "highly stale data").
- **E** — async/stale-rollout OPD in a production frontier tech report: MOPD is the shipped consolidation stage of Kimi K3 (§4.1 `S4.SS1.p1.1`), and it is placed inside the same partial-rollout RL framework used for the RL stage.
- **G** — the stale-rollout property is used, not studied; the paper's contributions are architecture (KDA), agentic RL and infrastructure, not async OPD.
- **Not A** — no argument anywhere that one KL estimator tolerates staleness better than another. The one estimator comparison the report does make (sampled-token vs "more fine-grained top-k") is about convergence speed and final score, not about staleness: "we observed no clear advantage in either convergence speed or final performance in our setting" (§4.1.3).
- **Not B / not C** — no side-by-side async-vs-sync MOPD numbers anywhere; no λ sweep, no lag sweep, no MOPD ablation table.
- **Not F** — async OPD is not a main point of the paper.
- **Tier caveat (important).** The report never writes a flat sentence of the form "MOPD is trained on stale rollouts". It writes that the MOPD reward *enables* partial rollout ("naturally enabling infrastructure-level optimizations such as partial rollout training"). The D/E/G grades rest on reading "enabling" as design intent realized, plus the fact that MOPD is expressed as a per-token reward inside the very RL framework whose rollout path is partial (§4.1.2). A stricter reader could downgrade to D-only. Flagged again under "Not in the source".

## Setting

- **Student:** the single unified Kimi K3 model (hybrid KDA + gated-MLA MoE, §2). "consolidating these domain-specific policies into a single model using Multi-Teacher On-Policy Distillation (MOPD)" (§4.1, `S4.SS1.p1.1`).
- **Teachers:** nine RL-trained experts = 3 domains × 3 reasoning-effort levels. "Crossing these three domain experts with three reasoning effort levels in $\{\text{low},\text{high},\text{max}\}$ yields a total of nine expert models." (§4.1.2, `S4.SS1.SSS2.p1.1`). Domains: "(i) general tasks … (ii) general agents … (iii) coding agents" (same paragraph). Teacher selection at train time: "for a given domain $d$ and a sampled reasoning effort level $e\in\{\text{low},\text{high},\text{max}\}$ , optimization is guided by the corresponding teacher model $\pi_{\text{teacher}}^{(d,e)}$ among the nine experts." (§4.1.3, `S4.SS1.SSS3.p1.1`).
- **Teacher fixed: yes.** The nine experts are finished RL checkpoints produced in the preceding stage (§4.1.2); nothing in §4.1.3 updates them. In scope for this sweep.
- **Pipeline:** SFT (§4.1.1) → per-domain, per-effort RL (§4.1.2) → MOPD (§4.1.3) → deployment-aware post-training / MXFP4 QAT (§4.1.4).
- **Framework / code:** in-house. Policy optimization "follows the algorithm in Kimi K2.5 [61]" (§4.1.2, `S4.SS1.SSS2.Px1.p1.1`); partial rollout is "extend[ed] … from our synchronous RL framework [119, 61]" ([119] = Kimi K1.5, [61] = Kimi K2.5). Co-located trainer+inference: "We adopt co-located RL training [58] … and use partial rollouts [119] to reduce tail latency from ultra-long trajectories." (§5.3.1, `S5.SS3.SSS1.p1.1`). Sandboxes: AgentENV, open-sourced at `https://github.com/kvcache-ai/AgentENV` (§5.3.2). Weights at `https://huggingface.co/moonshotai/Kimi-K3` (§1). Infrastructure side-releases MoonEP and minitriton are named in §5, but **no RL/OPD trainer, algorithm code or configs are released**.
- **Benchmarks:** §6 (public + in-house agentic suites: Swarm Bench, Deep Research Bench, Kimi Code Bench 2.0, Kimi Webdev Bench, …). None of them are reported per-MOPD-configuration.

## Async mechanism

- **Mechanism = partial rollout** (pipelined within the RL iteration loop; not a replay buffer and not offline precomputed rollouts). §4.1.2 "Algorithm" (`S4.SS1.SSS2.Px1.p1.1`), verbatim:
  > "To mitigate the long-tail latency that intensifies in long-horizon tasks, we extend the partial rollout scheme from our synchronous RL framework [119, 61]. During the rollout phase of each iteration, we sample $K$ completions for each of $N$ prompts, maintaining an active workload of $N\times K$ trajectories. Rather than waiting for all rollouts to terminate, the generation phase pauses as soon as a fraction $\lambda\in(0,1)$ of trajectories completes (i.e., $\lambda NK$ ), allowing policy optimization to proceed without execution stragglers. Paused rollouts are enqueued and prioritized for resumption at the start of the next iteration, powered by our sandbox infrastructure (§ 5.3.2). Once all $K$ responses for a prompt complete, they are immediately dispatched for policy optimization, which follows the algorithm in Kimi K2.5 [61]."
- **What is stale:** the *behaviour student* — the prefix of a long trajectory was decoded by an earlier student version than the one being updated. Verbatim (`S4.SS1.SSS2.Px1.p1.1`):
  > "Under our partial rollout scheme, an individual long-horizon trajectory naturally spans multiple iterations, introducing data staleness that threatens training stability."
  So a *single sequence* is mixed-version (early tokens older than late tokens), rather than whole batches being uniformly k steps old. The teacher is fixed, so all drift is student-side.
- **Link to MOPD** (§4.1.3, `S4.SS1.SSS3.p1.2`): "This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks."
- **How staleness is measured: not measured.** No steps-behind metric, no version tag per token, no "off-policyness" statistic, no discard threshold. The only handle is the qualitative "extreme off-policy regime" / "highly stale data" (§4.1.2).
- **Range of values studied: none reported.** $\lambda\in(0,1)$ is symbolic; no value, no sweep, no sensitivity. $N$, $K$, iteration count and the number of iterations a trajectory may span are all unstated.
- **Corroborating infra statements (§5.3.1, `S5.SS3.SSS1.Px1.p1.1`):** "Partial rollout exacerbates this issue at the beginning of each iteration, due to many unfinished long prefill requests from the previous iteration arriving at the same time." — confirms cross-iteration trajectory carry-over is real and routine at 1M context. Paused rollouts are held by pausable microVM sandboxes: "a sandbox can therefore be paused while the agent is waiting for the model's inference result, which can account for as much as 98% of the sandbox lifetime." (§5.3.2, `S5.I1.i2.p1.1`).

## Objective

- **Vocabulary: `rkl_pg`** — sampled-token score-function estimator of reverse KL, used as a per-token reward/advantage inside a policy-gradient update. Exactly the family the template names, and exactly the family DeepSeek-V4 §5.1.2 rejects.
- **Eq. 15** (§4.1.3, display `S4.EGx6`), verbatim LaTeX from the HTML `alttext`:
  ```latex
  r^{d}_{\mathrm{opd}}(y_{t}\mid e,x,y_{<t})=\mathrm{clip}\left(\mathrm{sg}\left(\log\frac{\pi_{\text{teacher}}^{(d,e)}(y_{t}\mid x,y_{<t})}{\pi_{\theta}(y_{t}\mid e,x,y_{<t})}\right),-R_{\max},R_{\max}\right)\,,
  ```
  Introduced as: "Given an input query $x$ and the prefix response $y_{<t}$ , the per-token OPD reward evaluated on $y_{t}$ between the teacher $\pi_{\text{teacher}}^{(d,e)}$ and the student $\pi_{\theta}$ is defined as:" (`S4.SS1.SSS3.p1.1`). And after: "where $\operatorname{sg}(\cdot)$ denotes the stop-gradient operator, and $R_{\max}>0$ is a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training." (`S4.SS1.SSS3.p1.2`).
- **Direction check (convention: reverse KL = KL(student ‖ teacher)).** Consistent with the sweep convention: maximizing $\mathbb{E}_{y\sim\pi_\theta}[\mathrm{sg}\log(\pi_{\text{teacher}}/\pi_\theta)]$ by score function is the sampled-token estimator of $-\mathrm{KL}(\pi_\theta\|\pi_{\text{teacher}})$. Note the effort argument is asymmetric in the equation as printed: the teacher is conditioned on $(x,y_{<t})$ while the student is conditioned on $(e,x,y_{<t})$ — i.e. the effort token is a student-side control, the teacher being already effort-specialized.
- **Loss / aggregation not given.** K3 never writes the surrogate that consumes $r^d_{\mathrm{opd}}$: no advantage normalization, no group size, no length normalization, no coefficient. It defers to "the algorithm in Kimi K2.5 [61]".
- **Truncated-support variants tried and dropped** (§4.1.3, `S4.SS1.SSS3.p1.2`): "While we also experimented with more fine-grained top- $k$ distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting." No numbers, no statement of whose top-k (teacher-top-k / student-top-k / union) — so this cannot be attributed in the review beyond "unspecified top-k".

## Off-policy correction

Two distinct, separately-stated devices; neither is an importance ratio.

1. **Reward-magnitude clipping, Eq. 15.** Symmetric, per-token, on the stop-gradient log-ratio itself: `clip(·, -R_max, R_max)`. This bounds the *advantage/reward*, not the policy ratio, so it is not an off-policy correction in the IS sense — it is variance control on the sampled-token RKL estimate. **$R_{\max}$ value is not given.**
2. **Per-token regularization in the policy loss (the actual staleness handle), §4.1.2 `S4.SS1.SSS2.Px1.p1.1`**, verbatim:
   > "Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization. By constraining policy updates within a localized neighborhood, this regularization enables the algorithm to robustly handle highly stale data and sustains training stability."
   **No equation for this is given in K3**; it is inherited from Kimi K2.5 [61] (arXiv 2602.02276), which is out of scope for this note. "per-token" + "localized neighborhood" reads as a trust-region / clip-or-mask family device (cf. GLM-5 Eq. 5, IcePop), but K3 does not name it, does not give its hyperparameters, and does not state that it is an importance ratio.
- **No IS ratio, no per-sequence ratio, no rollout-version tagging, no staleness-based sample dropping** appears anywhere in K3 (full-text check for "importance sampling", "importance ratio", "$\pi_{\text{rollout}}$", "replay" in a training sense — the only "replay" hits are CUDA-graph replay in a reward-hacking list (§4.2) and ReplaySSM, a KV/state caching trick (§5.4.2)).
- **Train–inference mismatch is handled separately and is explicitly eliminated** (§4.1.4, `S4.SS1.SSS4.Px1.p1.1`): "During RL, rollout and training share the same quantization scheme — eliminating the train–inference mismatch." Worth keeping distinct from staleness: K3 removes the *engine/precision* mismatch (MXFP4/MXFP8 QAT in both paths) while deliberately accepting the *version* mismatch.

## Results: async vs sync

**none.** No table, figure or sentence reports MOPD (or RL) scores at more than one staleness setting, and no synchronous-MOPD baseline exists in the report. Figure 8 plots scores vs RL FLOPs for the *RL* stage only, with no λ or staleness axis. The MOPD stage has no ablation of any kind.

## Explanation offered

Only one mechanism sentence, and it is about the RL stage's tolerance, inherited by MOPD by construction (§4.1.2, `S4.SS1.SSS2.Px1.p1.1`):

> "Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization. By constraining policy updates within a localized neighborhood, this regularization enables the algorithm to robustly handle highly stale data and sustains training stability."

Identified mechanism: **trust-region / locality constraint on the update**, i.e. staleness is tolerated by keeping $\pi_\theta$ close to the behaviour policy per token, not by reweighting the data and not by any property of the KL estimator. There is **no** analysis of support mismatch, student-weighting, MC variance, or teacher inconsistency. The variance question is acknowledged only obliquely, by the $R_{\max}$ clip "to constrain extreme advantage signals, thereby stabilizing RL training" (§4.1.3) — variance control at the reward level, not a staleness argument.

## Quotes

Every verbatim statement in K3 bearing on async / stale / off-policy / partial rollout, with section ids.

**On the OPD stage (§4.1.3):**
- `S4.SS1.SSS3.p1.2`: "This dense reward signal seamlessly integrates into our RL framework, naturally enabling infrastructure-level optimizations such as partial rollout training for long-horizon tasks." ← the single load-bearing sentence for this sweep.
- `S4.SS1.SSS3.p1.2`: "where $\operatorname{sg}(\cdot)$ denotes the stop-gradient operator, and $R_{\max}>0$ is a clipping threshold to constrain extreme advantage signals, thereby stabilizing RL training."
- `S4.SS1.SSS3.p1.2`: "While we also experimented with more fine-grained top- $k$ distillation objectives, we observed no clear advantage in either convergence speed or final performance in our setting."
- `S4.SS1.SSS3.p1.1`: "We adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate these domain-specialized capabilities across varying reasoning efforts into a unified model [75, 135, 29]." ([75] = Thinking Machines Lab OPD post; [135] = MiMo-v2-flash, arXiv 2601.02780; [29] = DeepSeek-V4, arXiv 2606.19348.)

**On the RL stage's staleness, which MOPD is said to plug into (§4.1.2, `S4.SS1.SSS2.Px1.p1.1`):**
- "To mitigate the long-tail latency that intensifies in long-horizon tasks, we extend the partial rollout scheme from our synchronous RL framework [119, 61]."
- "Rather than waiting for all rollouts to terminate, the generation phase pauses as soon as a fraction $\lambda\in(0,1)$ of trajectories completes (i.e., $\lambda NK$ ), allowing policy optimization to proceed without execution stragglers."
- "Paused rollouts are enqueued and prioritized for resumption at the start of the next iteration, powered by our sandbox infrastructure (§ 5.3.2)."
- "Under our partial rollout scheme, an individual long-horizon trajectory naturally spans multiple iterations, introducing data staleness that threatens training stability."
- "Our policy optimization algorithm inherently tolerates such an extreme off-policy regime through a per-token regularization. By constraining policy updates within a localized neighborhood, this regularization enables the algorithm to robustly handle highly stale data and sustains training stability."

**On the infrastructure (§5.3.1, §5.3.2):**
- `S5.SS3.SSS1.p1.1`: "We adopt co-located RL training [58] to keep each 1M-context Kimi K3 RL experiment within a few hundred GPUs, and use partial rollouts [119] to reduce tail latency from ultra-long trajectories."
- `S5.SS3.SSS1.Px1.p1.1`: "Partial rollout exacerbates this issue at the beginning of each iteration, due to many unfinished long prefill requests from the previous iteration arriving at the same time."
- `S5.I1.i2.p1.1`: "a sandbox can therefore be paused while the agent is waiting for the model's inference result, which can account for as much as 98% of the sandbox lifetime."

**On train–inference (not staleness) mismatch (§4.1.4, `S4.SS1.SSS4.Px1.p1.1`):**
- "During RL, rollout and training share the same quantization scheme — eliminating the train–inference mismatch."

**Ambiguous, flagged (§4.1.2, `S4.SS1.SSS2.Px2.p1.1`):**
- "Trajectories produced by the resulting experts at all reasoning levels are jointly collected for supervised fine-tuning and multi-teacher on-policy distillation." — read literally this says *teacher*-generated trajectories feed MOPD, which would make MOPD off-policy in the GKD "teacher-sampled" sense rather than the staleness sense. The lumping with SFT ("jointly collected for supervised fine-tuning and …") and Eq. 15's per-token-reward form inside an RL rollout loop both point to the intended reading being *prompts/tasks* harvested from expert trajectories, with MOPD itself sampling $y$ from $\pi_\theta$. The report never disambiguates. Do not cite K3 as evidence for teacher-sampled OPD.

## Not in the source

- **No value for $\lambda$** (the partial-rollout completion fraction), $N$, $K$, or the number of iterations a trajectory may span → the staleness range is entirely unquantified.
- **No value for $R_{\max}$**, no $w$/weighting across the nine teachers, no MOPD step count, learning rate, batch size, temperature, or data mixture.
- **No equation for the "per-token regularization"** that does the actual staleness tolerating — deferred to Kimi K2.5 [61]. So K3 alone does not tell us whether the correction is an IS clip, a mask, or a KL penalty.
- **No explicit sentence that MOPD was run with partial rollout enabled.** The report says the reward "naturally enabl[es]" it. Every tier grade above depends on that reading; do not quote K3 as saying MOPD *is* asynchronous.
- **No sync baseline for MOPD** and no staleness sweep → tiers B and C are unavailable, and K3 contributes no async-vs-sync data point to the review, only a production existence-proof plus vocabulary.
- **No numbers behind the top-k comparison** and **no attribution of whose top-k** (teacher-top-k vs student-top-k vs union) → per the top-k attribution rule, cite this only as "an unspecified top-k variant, no advantage observed, no numbers".
- **No discussion of the interaction** between the sampled-token estimator and staleness — i.e. K3 never asks whether `rkl_pg` is more or less staleness-tolerant than a distribution-level RKL, which is precisely our question. It is a counterparty data point (frontier lab ships `rkl_pg` on stale rollouts) rather than evidence either way.
- **No per-token version tagging, no staleness-based filtering, no measured off-policyness** — contrast GLM-5 §4.1.2, which does log rollout versions and drops samples with $w'-w_0>\tau$.
