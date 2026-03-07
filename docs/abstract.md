Abstract

We investigate the dynamical behavior of recursively updated language-model corpora under controlled mixture dynamics. Specifically, we analyze a system in which a mutable text pool evolves through iterative transformation while a fixed anchor subset is retained with weight parameter α. We treat this as a discrete-time dynamical system over semantic state space and study its macroscopic observables.

We introduce three orthogonal macroscopic metrics:
	1.	Freeze Occupancy (π_F) — the proportion of windows exhibiting structural convergence (low boundary density, large grain size).
	2.	Signature Entropy (H) — entropy of invariant-signature distributions over a sliding anchor-constrained pool.
	3.	Trajectory Invariance Metric (TIM) — robustness of semantic trajectory under time-axis distortions (rescaling, truncation).

Through parameter sweeps in α (anchor strength), mutation ratio r, and smoothing scale W, we identify regime-dependent structural behavior:
	•	At low anchor strength, the system exhibits entropy decay and variance amplification consistent with semantic collapse.
	•	At intermediate α, metastable coexistence emerges: grain segmentation alternates between frozen and mutable phases.
	•	Above a critical α*, freeze occupancy increases and structural persistence dominates.

Lag-correlation analysis reveals strong anticorrelation between π_F and H at coarse smoothing scales (|corr| ≈ 0.9–0.97). However, Granger-style nested regression tests demonstrate minimal one-step cross-predictive power (ΔR² ≪ 0.01 in most regimes), indicating that the observed anticorrelation arises from shared slow-manifold dynamics rather than direct linear forcing.

These findings suggest that freeze and entropy are co-manifestations of a latent regime variable rather than causal drivers of one another. Linear cross-coupling is negligible once autoregressive persistence is controlled, implying either nonlinear threshold dynamics or regime-switch behavior.

Adversarial stress tests—including seed sweeps, window variation, sampling perturbations, and minimal-model regression—support the robustness of regime classification across orthogonal metrics. The system exhibits metastability, hysteresis-like memory under quench conditions, and parameter-sensitive phase boundaries.

We propose a general phase-discovery protocol:
	1.	Introduce a tunable control parameter.
	2.	Measure orthogonal macroscopic observables.
	3.	Sweep and detect regime shifts.
	4.	Make falsifiable predictions.
	5.	Stress-test under adversarial perturbation.
	6.	Compress invariant structure into reusable formal schema.

This framework provides a principled method for identifying geometric phase structure in recursive generative systems. Our results indicate that sustained self-mixture without sufficient anchoring induces collapse, while controlled anchoring induces metastable confinement regimes.

The work reframes recursive language model dynamics as phase-structured systems governed by latent manifold geometry rather than simple autoregressive coupling.
