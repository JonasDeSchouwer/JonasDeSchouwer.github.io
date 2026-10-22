# Zhang et al. 2026 — The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation (CaOPD)

**arXiv 2604.16830** (Apr 2026). Jiaxin Zhang, Xiangyu Peng, Qinglin Chen, Qinyuan Ye, Caiming Xiong, Chien-Sheng Wu (Salesforce per sweep; affiliation not shown on abs page). Content from arXiv abs + HTML via WebFetch [FETCH 2026-08-18]. Incorporated: paper.typ appendix "Additional Related Work".

```bibtex
@article{zhang2026illusion,
  author  = {Jiaxin Zhang and Xiangyu Peng and Qinglin Chen and Qinyuan Ye and Caiming Xiong and Chien-Sheng Wu},
  title   = {The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation},
  journal = {arXiv preprint arXiv:2604.16830},
  year    = {2026}
}
```

## Core claims

- "We identify a pervasive Scaling Law of Miscalibration: while OPD effectively improves task accuracy, it systematically traps models in severe overconfidence." Body: "whether utilizing massive proprietary APIs or scaling up open-weight models, increasing capability … does not resolve the optimism bias." (Empirical law; no closed-form equation.)
- Mechanism [FETCH]: "Teacher supervision is formed under privileged context available during training, whereas the deployed model must report confidence using only deployment-time information."; "the model π_θ(·|y_<t, X) is penalized for expressing the natural uncertainty H(A|X), forcing its internal logits to become artificially sharpened"; "teacher-conditioned success probability μ_T(X,Z) is not measurable with respect to X alone"; "Teacher-conditioned success is generally not a valid target for deployment-time confidence."

## Fix (CaOPD) [FETCH]

- Eq. 6: empirical confidence target μ̂(x) = (1/K)Σ_k R(x, a_k) from the student's own rollouts, replacing self-reported confidence.
- Eq. 7: distillation objective is context distillation — D_KL(π_θ(·|ỹ_<t, x) ‖ π_θ(·|ỹ_<t, x, z̃)) summed over answer-token and confidence-token index sets (student-without-context toward itself-with-privileged-context; note lineage to Snell et al. 2209.15189).
- Numbers: Science Q&A Qwen3-8B overconfidence gap +48.1% (SDFT) → −11.8% (CaOPD); ECE 0.486 → 0.266; accuracy 49.1 → 50.0 (Science), 67.6 → 70.6 (ToolAlpaca). Models Qwen3 0.6B–32B, OLMo-3-7B-Instruct.

## Relevance

Opposite-signed uncertainty pathology vs. compromise latching: the privileged teacher's conditionals are *sharper* than any deployment-measurable target, so distillation trains overconfidence in — mirror image of our setting where the teacher's conditionals legitimately carry hedge mass and reverse KL over-weights it. Both are distillation across an information asymmetry between teacher and student context variables (Z_T vs Z). Used in paper.typ to delimit the setting; also relevant to Experiment 1 design (our teacher hedges *by construction of its prompt*).
