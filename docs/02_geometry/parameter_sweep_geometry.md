# Parameter Sweep Geometry

This document defines how the discrete parameter sweep over (r, α) gives rise to a structured geometric manifold in the PAM Observatory.

---

## Overview

The PAM Observatory constructs its manifold from a finite experimental sweep:

- parameters are sampled on a grid
- each grid point produces observable data
- geometry emerges from relationships between these points

This document explains how discrete sampling becomes continuous structure.

---

## Parameter Space

The system is defined over:

```math
\theta = (r, \alpha)
```

With:

- r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
- α ∈ linspace(0.03, 0.15, 15)
- seeds = 10

Total runs:

```math
5 \times 15 \times 10 = 750
```

Each configuration corresponds to a point in parameter space.

---

## Discrete Sampling

The sweep produces a lattice:

- nodes = (r, α) pairs
- repeated trials (seeds) reduce stochastic noise
- outputs aggregated into summary observables

Stored in:

- outputs/index.csv

---

## Observable Surface

Each observable defines a scalar field over the grid:

```math
f(\theta)
```

Examples:

- piF_tail(θ)
- H_joint_mean(θ)
- K_max(θ)

These form layered surfaces over parameter space.

---

## From Grid to Geometry

The key transformation:

> discrete samples → continuous geometric structure

Steps:

1. Estimate gradients across grid
2. Construct Fisher metric
3. Define local distances
4. Connect nodes into graph

---

## Local Neighborhood Structure

Each point interacts with neighbors:

- finite differences approximate derivatives
- local patches approximate tangent space
- anisotropy emerges from observable variation

Interpretation:

- flat region → uniform behavior
- steep region → sensitive transitions

---

## Metric Construction

From observable gradients:

```math
G_{ij}(\theta) = \partial_i m(\theta)^T \Sigma^{-1} \partial_j m(\theta)
```

This converts discrete variation into a continuous metric.

---

## Graph Approximation

The manifold is approximated as a weighted graph:

- nodes = parameter points
- edges = local neighbors
- weights = Fisher distances

This enables:

- geodesic computation
- global structure recovery

---

## Resolution and Limits

The manifold quality depends on:

- grid density
- observable smoothness
- noise in measurements

Limitations:

- coarse grids → aliasing of structure
- missing data → holes in manifold
- edge effects near parameter bounds

---

## Backfilling and Completion

Missing trajectories create gaps:

- incomplete observables
- unreliable gradients
- distorted geometry

Backfilling restores:

- continuity
- metric stability
- reliable curvature

---

## Relation to Geometry Pipeline

This document defines the **input layer** of the pipeline.

It feeds into:

- Fisher metric → local geometry
- distances → global structure
- embedding → visualization
- phase → behavioral segmentation

---

## Implementation Mapping

| Concept | File |
|--------|------|
| Parameter sweep | experiments/exp_batch.py |
| Data aggregation | outputs/index.csv |
| Trajectories | outputs/trajectories/ |
| Backfill tools | experiments/backfill_trajectories.py |

---

## Outputs

- outputs/index.csv
- outputs/trajectories/
- outputs/manifests/

---

## Summary

The parameter sweep defines the substrate of the PAM manifold.

- Grid → defines sampling
- Observables → define fields
- Gradients → define structure
- Metric → defines geometry

This is the layer where experimental data becomes geometric input.
# Parameter Sweep Geometry

This document explains how the PAM parameter sweep forms the foundation of the observatory.

Before constructing geometry, we must understand what the parameter space represents.

## 1. The Parameter Manifold

The system is defined over a two-dimensional parameter space:

θ = (r, α)

where:
- **r** — controls recursive strength
- **α** — controls update sensitivity / coupling

The parameter space is discretized as:

r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
α ∈ linspace(0.03, 0.15, 15)

Each point corresponds to one experimental configuration.

## 2. The Grid as a Sampling of a Continuous Space

Although the parameter space is evaluated on a grid, it is conceptually Θ ⊂ ℝ², a continuous domain.

The grid should therefore be interpreted as a discrete sampling of an underlying continuous system.

## 3. Each Point as a System

For each parameter configuration θ:
- a trajectory is generated
- the system evolves over time
- observables are extracted

Thus, each point in parameter space corresponds to:

θ → dynamical system → observable signature

## 4. From Grid to Field

The collection of observables defines a field x : Θ → ℝⁿ.

Each observable becomes a scalar field over the parameter domain.

## 5. Structure in Parameter Space

In PAM:
- observables vary nonlinearly
- distinct regions emerge
- transitions occur between regimes

This implies the parameter space contains latent structure.

## 6. From Parameter Space to Geometry

Instead of treating Θ as a flat grid, we construct:
- a metric (Fisher Information)
- a distance structure (geodesics)
- an embedding (manifold)

This transforms Θ → 𝓜, where 𝓜 is an intrinsic geometric representation.

## 7. Final Perspective

The role of the parameter sweep is to provide:
- coverage of the system
- resolution of structure
- input to geometry

The final object of interest is the manifold 𝓜 and its phase structure.
