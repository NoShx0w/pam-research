# PAM Research Notes

This document records the conceptual development of the PAM project.

PAM did not begin as a conventional software project. It emerged from an extended exploratory dialogue investigating how meaning evolves under recursive generation.

The goal of these notes is to preserve the key insights and turning points that led to the current experimental framework.

---

# Prelude

The research began with a simple intuition:

> Meaning might behave like a geometric object rather than a static property of text.

Instead of treating language purely as sequences of tokens, the working hypothesis was that semantic structures might exhibit **invariants**, **curvature**, and **phase-like behavior** when subjected to recursive transformation.

The early conversations explored ideas from:

- dynamical systems
- information geometry
- invariance detection
- recursive generative processes

These ideas gradually converged into a concrete experimental question:

**What happens when language systems recursively transform their own outputs?**

---

# Early Concept: Invariants

A central idea emerged early in the project:

Certain semantic properties appear to remain stable under paraphrasing or structural variation.

These properties were referred to as **semantic invariants**.

Examples include:

- reflective tone
- geometric metaphor usage
- coherent conceptual structure
- playful-serious rhetorical balance

To test this idea computationally, the **TIP (Text Invariant Perceptron)** was introduced.

TIP acts as a lightweight detector of invariant semantic signatures.

Each text can therefore be mapped to a **signature vector of invariants**.

---

# Corpus Dynamics

Once invariant signatures could be detected, the next question was how they behaved under recursive generation.

This led to the construction of the core experimental loop.

Each iteration:

1. A subset of the corpus is replaced.
2. Replacement samples are generated either by:
   - mutation of anchor texts
   - resampling from recent outputs.
3. A fixed anchor subset is periodically reintroduced.

This produced a simple but expressive control system defined by two parameters:

r  = replacement fraction
α  = anchor injection probability

The evolving corpus therefore becomes a **discrete-time dynamical system**.

---

# Discovery of Freeze States

While observing early runs, a striking structural pattern appeared.

The corpus did not drift randomly.

Instead it organized itself into regions of:

- **structural stability**
- **structural mutation**

These regions appeared and disappeared over time.

This led to the concept of **freeze occupancy (π_F)**:

The fraction of windows exhibiting structural convergence.

Freeze detection relied on microstructure analysis using:

- boundary density
- mean grain size

---

# Entropy Dynamics

To track diversity of semantic signatures, a second observable was introduced:

**Signature Entropy (H)**

Entropy measures the diversity of invariant signatures within the mutable corpus.

Two entropy formulations were explored:

- marginal entropy of individual invariants
- joint entropy of full signature patterns

The joint entropy proved to capture structural regime shifts more clearly.

---

# A Surprising Empirical Result

Once freeze occupancy and entropy were tracked together, a strong empirical relationship emerged:

corr(π_F, H) ≈ −0.90 to −0.97

Across many runs and smoothing scales, freeze and entropy exhibited strong anticorrelation.

At first glance this suggested a simple causal story:

Entropy collapse might cause structural freezing.

However further analysis revealed a more subtle picture.

---

# Minimal Dynamical Models

To test causal interaction, minimal autoregressive models were constructed.

Two coupled equations were fit:

F_{t+1} = a + bF_t + cH_t
H_{t+1} = d + eH_t + fF_t

The key test was whether adding the cross-variable significantly improved prediction.

Results consistently showed:

ΔR²_freeze ≈ 0.08–0.12
ΔR²_entropy ≈ 0.002–0.005

Despite strong correlations, direct one-step causal coupling appeared weak.

---

# Interpretation: A Latent Regime Variable

The most plausible explanation is that freeze and entropy are not driving each other directly.

Instead they appear to be **co-manifestations of an underlying latent regime variable**.

The system seems to evolve along a **slow manifold** in semantic state space.

Freeze and entropy are therefore projections of the same underlying system state.

---

# Phase Structure

Sweeps over parameter space reveal three regimes:

| Regime | Description |
|------|------|
| Entropy-dominated | rapid structural drift |
| Mixed | intermittent freeze and mutation |
| Freeze | stable invariant structure |

These regimes form a **phase diagram in (r, α) parameter space**.

---

# Trajectory Invariance

A third observable was introduced:

**TIM — Trajectory Invariance Metric**

TIM measures robustness of semantic trajectories under perturbations such as:

- sentence truncation
- time-axis distortion
- partial sampling

TIM helps detect whether texts preserve structural meaning under transformations.

---

# Robustness Testing

Several stress tests were introduced:

- seed sweeps
- smoothing window variation
- sampling perturbations
- nested regression tests

The freeze–entropy relationship remained stable across these perturbations.

---

# Toward a Phase Discovery Protocol

The project gradually converged toward a general methodology for studying recursive generative systems.

Proposed protocol:

1. Introduce a tunable control parameter
2. Measure orthogonal macroscopic observables
3. Sweep parameter space
4. Detect regime transitions
5. Test causal coupling
6. Stress-test robustness
7. Compress invariant structure

This protocol forms the methodological core of PAM.

---

# Meta Observation

An interesting property of the project is that the research process itself involved a recursive dialogue with a language model.

The research system therefore exhibits a form of **meta-recursion**:

The system used to study recursive language dynamics was itself developed through recursive language interaction.

---

# Current Status

The PAM framework now supports:

- corpus evolution experiments
- entropy and macrostate metrics
- lag correlation analysis
- minimal dynamical regression tests
- parameter sweeps and batch experiments
- structured logging of runs

Current work focuses on mapping the **phase geometry of the system**.

---

# Next Steps

Immediate research directions:

- generate high-resolution phase diagrams
- locate critical anchor threshold α*
- study hysteresis under quench conditions
- test larger corpora and mutation operators

Longer term:

- investigate nonlinear regime switching
- explore connections to model collapse dynamics
- extend invariant detectors

---

# Repository

Code and experiments:

https://github.com/NoShx0w/pam-research

---

These notes document the path taken to reach the current system. They are intended to make the conceptual development of PAM transparent and reproducible.

