# PAM Research Notes

This document records noteworthy conceptual and experimental milestones during the development of PAM (Phase Analysis of Meaning).

The goal is to preserve the reasoning path, discoveries, and methodological decisions that shaped the framework.

---

# 1. Initial Motivation

The project began as an exploration of how **recursive language model outputs behave when repeatedly mixed with their own generations**.

Motivating questions included:

- Does recursive mixture inevitably lead to **semantic collapse**?
- Can structural stability be maintained through **controlled anchoring**?
- Can language dynamics exhibit **phase-like behavior** similar to physical systems?

Early intuition suggested that recursive text systems might exhibit **regime-dependent dynamics** rather than simple degradation.

---

# 2. Reframing the System as a Dynamical Process

A key conceptual shift occurred when the corpus evolution was framed as a **discrete-time dynamical system**:

Corpus_t → transformation → Corpus_{t+1}

Rather than studying individual texts, the focus shifted to **macroscopic observables of the evolving corpus**.

This reframing allowed the system to be analyzed using tools from:

- dynamical systems
- statistical physics
- phase transition analysis

---

# 3. Introduction of Invariant Signatures (TIP)

The first structural component introduced was the **Invariant Perceptron (TIP)**.

TIP detects invariant semantic features in text using heuristic scoring rules.

Each text sample is mapped to a binary invariant signature:

signature = {
reflective: True/False
coherent: True/False
playful_serious: True/False
geometric: True/False
}

This representation provides a **coarse semantic coordinate system** over the corpus.

---

# 4. Emergence of Macrostate Structure

Once invariant signatures were introduced, corpus-level distributions began to reveal **large-scale structural patterns**.

Two macroscopic regimes emerged:

- **Freeze regime** — invariant signatures become stable and homogeneous
- **Mixed regime** — signatures fluctuate across windows

This led to the definition of:

Freeze Occupancy (π_F)

which measures the proportion of windows exhibiting structural convergence.

---

# 5. Signature Entropy

To quantify structural diversity, the **Signature Entropy (H)** metric was introduced.

This measures entropy over invariant-signature distributions in the mutable pool.

Two formulations were explored:

- pooled marginal entropy
- joint signature entropy

Joint entropy ultimately provided the most informative signal.

---

# 6. Microstructure Metrics

Further analysis of corpus segmentation introduced **microstructure metrics**:

- boundary density
- grain size

These metrics enabled identification of frozen windows through structural segmentation.

This provided the basis for macrostate classification.

---

# 7. Trajectory Invariance Metric (TIM)

A third orthogonal observable was introduced:

Trajectory Invariance Metric (TIM)

TIM measures how robust a text’s semantic trajectory is under **time-axis distortions**, including:

- truncation
- rescaling
- windowing

The motivation was to capture **structural stability under perturbation**.

---

# 8. Parameterized Dynamics

The system dynamics were parameterized by two control variables:

r  = replacement fraction
α  = anchor injection probability

The anchor set acts as a stabilizing reference subset.

Mutation and resampling introduce stochastic drift.

Together these parameters control the balance between:

- exploration
- structural confinement

---

# 9. Discovery of Strong π_F – H Anticorrelation

Lag-correlation analysis revealed a striking empirical pattern:

corr(π_F, H) ≈ -0.90 to -0.97

This strong anticorrelation appeared consistently across smoothing scales and seed sweeps.

This suggested a deep relationship between **structural freezing and entropy collapse**.

---

# 10. Causal Analysis via Minimal Dynamical Models

To test whether one observable *caused* the other, minimal autoregressive models were introduced:

F_{t+1} = a + bF_t + cH_t
H_{t+1} = d + eH_t + fF_t

Granger-style nested regression tests measured cross-predictive power.

Results showed:

ΔR²_freeze ≈ 0.1
ΔR²_entropy ≈ 0.003

After controlling for autoregressive persistence, cross-coupling was minimal.

---

# 11. Interpretation: Shared Latent Regime Variable

These results suggested that freeze and entropy are **not causally driving each other**.

Instead they appear to be **co-manifestations of a latent regime variable**.

This implies the system evolves along a **slow manifold** in phase space.

Observed anticorrelation arises from **shared regime dynamics**, not direct forcing.

---

# 12. Stress Testing

Several adversarial tests were introduced:

- seed sweeps
- window size variation
- mutation perturbations
- injector perturbations
- minimal-model regression checks

The regime structure remained stable across these perturbations.

---

# 13. Emergent Phase Structure

Parameter sweeps across α revealed three regimes:

| Regime | Description |
|------|-------------|
| Entropy-dominated | uncontrolled drift |
| Mixed | metastable coexistence |
| Freeze | structural confinement |

Evidence of **metastability** and **hysteresis-like behavior** emerged in quench experiments.

---

# 14. Phase Discovery Protocol

The research eventually crystallized into a general experimental protocol:

1. Introduce a tunable control parameter
2. Measure orthogonal macroscopic observables
3. Perform parameter sweeps
4. Detect regime transitions
5. Test causal structure
6. Stress-test under perturbations
7. Compress invariant structure

This protocol can be applied beyond language systems.

---

# 15. Conceptual Interpretation

The project reframes recursive language dynamics as a **phase-structured system**.

Rather than simple autoregressive processes, recursive generative systems appear to evolve on a **latent geometric manifold** whose structure is revealed through macroscopic observables.

---

# Current Status

The PAM framework now functions as a **controlled experimental laboratory** for studying recursive generative dynamics.

Ongoing work focuses on:

- mapping phase diagrams across (r, α)
- analyzing metastability and regime boundaries
- extending trajectory invariance metrics
