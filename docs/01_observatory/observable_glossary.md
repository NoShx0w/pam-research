# Observable Glossary

This document defines the main observables used by the PAM Observatory.

These observables originate in recursive corpus trajectories and are summarized into run-level artifacts such as:

```text
outputs/index.csv
```

They form the empirical interface between experiment dynamics and the downstream geometry layer.

---

## Observable Layers

It is useful to distinguish three levels.

### 1. Trajectory-level quantities

These are time-series quantities measured along a single run.

Examples include:

- freeze-state sequence
- entropy sequence
- signature-count sequence
- lag / correlation traces

These are often stored in trajectory artifacts under:

```text
outputs/trajectories/
```

### 2. Run-level summary observables

These are scalar summaries derived from a single run and typically written into:

```text
outputs/index.csv
```

Examples include:

- `piF_mean`
- `piF_tail`
- `H_joint_mean`
- `var_H_joint`
- `corr0`
- `best_corr`
- `delta_r2_freeze`
- `delta_r2_entropy`
- `K_min`
- `K_max`

### 3. Geometry-input observables

These are the run-level observables selected to form the feature vector used by the geometry layer.

Conceptually:

```math
m(\theta)
```

This observable vector is what the Fisher-type metric is built from.

---

## Core Observable Families

### Freeze occupancy

Freeze-related observables measure structural convergence or persistence in the recursive system.

Common summaries include:

- `piF_mean` — mean freeze occupancy over the run
- `piF_tail` — freeze occupancy near the end of the run

Interpretation:

- higher values indicate stronger structural persistence
- lower values indicate more mutable or mixed dynamics

---

### Entropy

Entropy observables measure diversity of invariant-signature structure.

Common forms include:

- `H_joint` — joint signature entropy over the mutable pool
- `H_joint_mean` — average joint entropy across the run
- `var_H_joint` — variance of joint entropy across the run

Interpretation:

- higher entropy indicates greater structural diversity
- lower entropy indicates more concentrated or collapsed signature structure

---

### Signature-count / complexity observables

These observables track how many structurally distinct signature patterns are active.

Common forms include:

- `K` — signature-count-like or complexity-like quantity
- `K_min`
- `K_max`

Interpretation:

- larger values indicate richer active structure
- smaller values indicate reduced or more compressed structure

---

### Correlation observables

These observables describe temporal alignment between major run-level quantities.

Common forms include:

- `corr0` — zero-lag correlation between smoothed observables
- `best_corr` — strongest lag-correlation value over a lag window
- `best_lag` — lag at which strongest correlation occurs

Interpretation:

- these quantify temporal coupling structure
- they help detect whether observables co-evolve directly or through slower latent structure

---

### Regression / predictive observables

These observables summarize minimal predictive models fit to run-level series.

Common forms include:

- `delta_r2_freeze`
- `delta_r2_entropy`

Interpretation:

- they estimate whether one observable adds predictive power for another beyond autoregressive persistence
- small values suggest correlation without strong direct one-step forcing

---

## Smoothing-Related Quantities

Some observables are also represented in smoothed form during internal analysis.

Examples include:

- smoothed freeze occupancy
- smoothed joint entropy

These are useful for lag and dynamical analysis, but the main geometry pipeline typically consumes scalar run-level summaries rather than full smoothed traces.

---

## Role in the Geometry Layer

The observables define a feature vector over the control manifold:

```math
m(\theta)
```

where \(\theta = (r, \alpha)\).

From this vector, the observatory constructs a Fisher-type metric:

```math
G_{ij}(\theta) = \partial_i m(\theta)^T \Sigma^{-1} \partial_j m(\theta)
```

Interpretation:

- observables provide the measurable state
- the metric measures distinguishability of nearby parameter configurations
- geometry is therefore induced by observable behavior rather than imposed externally

---

## Summary

Observables are the empirical interface between recursive dynamics and manifold structure.

They:

- begin as trajectory measurements
- are summarized into run-level artifacts
- define the feature vector used by the geometry layer
- support downstream phase, operator, and topology analysis

In this sense, observables are the layer where behavior first becomes measurable structure.
