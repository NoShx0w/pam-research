# Parameter Sweep Geometry

This document defines how the discrete parameter sweep over \((r, \alpha)\) gives rise to the sampled substrate from which the PAM manifold is constructed.

Before the observatory can recover intrinsic geometry, it must first populate parameter space with measured experiment outputs.

---

## Overview

The PAM Observatory constructs its manifold from a finite experimental sweep:

- parameters are sampled on a grid
- each grid point produces run-level observables
- repeated seeds reduce stochastic noise
- geometry emerges from relationships across the sampled grid

This document explains how discrete sampling becomes geometric input.

---

## Parameter Space

The system is defined over:

```math
\theta = (r, \alpha)
```

with a canonical sweep such as:

- \(r \in \{0.10, 0.15, 0.20, 0.25, 0.30\}\)
- \(\alpha \in \mathrm{linspace}(0.03, 0.15, 15)\)
- seeds = 10

Total runs:

```math
5 \times 15 \times 10 = 750
```

Each run corresponds to a specific experimental configuration in parameter space.

---

## Discrete Sampling

The sweep produces a lattice over the control manifold:

- nodes = parameter pairs \((r, \alpha)\)
- repeated trials over seeds reduce stochastic noise
- each run contributes summary observables and optional trajectory artifacts

Important active artifacts include:

- `outputs/index.csv`
- `outputs/trajectories/`

`outputs/index.csv` acts as the main run ledger from which downstream geometry inputs are assembled.

---

## Each Parameter Point as a Measured System

Each parameter configuration should be read as:

```text
(r, α, seed)
↓
recursive experiment
↓
trajectory
↓
run-level observables
```

So a point in parameter space is not just a coordinate.

It is a measured dynamical system instance.

---

## Observable Surfaces

From the run ledger, observables can be organized into scalar fields over the grid.

Conceptually:

```math
f(\theta)
```

Examples include:

- `piF_tail(\theta)`
- `H_joint_mean(\theta)`
- `K_max(\theta)`

These observables form layered surfaces over parameter space and become the empirical basis for geometry construction.

---

## From Grid to Geometry

The key transformation is:

> discrete samples → intrinsic geometric structure

In practice, this involves:

1. assembling observable vectors over parameter space  
2. estimating local variation across the grid  
3. constructing a Fisher-type metric  
4. building local distance structure  
5. recovering global manifold geometry through graph distances and embedding  

So the sweep is not the geometry itself.

It is the sampled substrate from which geometry is inferred.

---

## Local Neighborhood Structure

Each sampled point interacts with nearby parameter neighbors.

This supports:

- finite-difference derivative estimates
- local tangent-like approximation
- detection of anisotropy in observable variation

Interpretation:

- flat regions correspond to relatively uniform behavioral organization
- steep or anisotropic regions correspond to sensitive transition structure

---

## Metric Construction

From local observable variation, the observatory constructs a Fisher-type metric:

```math
G_{ij}(\theta) = \partial_i m(\theta)^T \Sigma^{-1} \partial_j m(\theta)
```

This converts discrete observable variation into an intrinsic local geometry.

---

## Graph Approximation

The manifold is then approximated as a weighted graph:

- nodes = sampled parameter points
- edges = local neighborhood relations
- weights = Fisher-derived local distances

This graph supports:

- shortest-path geodesic approximation
- global distance recovery
- embedding into low-dimensional manifold coordinates

---

## Resolution and Limits

The quality of the inferred manifold depends on:

- grid density
- observable smoothness
- seed coverage
- noise level in measurements

Important limitations include:

- coarse grids can alias structure
- missing data can distort local derivatives
- edge regions have weaker neighborhood support
- unstable observables can degrade metric quality

---

## Backfilling and Completion

Missing trajectories or missing run summaries create geometric problems:

- incomplete observable surfaces
- unreliable local derivatives
- distorted distance structure
- unstable curvature estimates

Backfilling restores:

- continuity of the sampled manifold
- metric stability
- more reliable downstream geometry and phase outputs

This is why sweep completion and trajectory validation matter scientifically, not just operationally.

---

## Relation to the Geometry Layer

This document describes the sampling substrate that feeds the canonical geometry layer.

It supports downstream construction of:

- Fisher metric
- geodesic distance graph
- manifold embedding
- curvature diagnostics

Those are implemented canonically under:

- `src/pam/geometry/`

and orchestrated through:

- `src/pam/pipeline/stages/geometry.py`
- `scripts/run_full_pipeline.sh`

---

## Implementation Mapping

### Experimental substrate

| Concept | File |
|--------|------|
| Parameter sweep | `experiments/exp_batch.py` |
| Run execution | `experiments/exp_quench.py` |
| Run ledger | `outputs/index.csv` |
| Trajectory artifacts | `outputs/trajectories/` |
| Missing-scan tooling | `experiments/scan_missing_trajectories.py` |
| Backfill tooling | `experiments/backfill_trajectories.py` |
| Validation tooling | `experiments/validate_trajectories.py` |

### Canonical downstream consumer

| Concept | Canonical module |
|--------|------|
| Geometry layer | `src/pam/geometry/` |
| Geometry pipeline stage | `src/pam/pipeline/stages/geometry.py` |

---

## Summary

The parameter sweep defines the sampled substrate of the PAM manifold.

- grid coverage defines where the system is measured
- observables define empirical fields over that grid
- local variation defines metric structure
- the geometry layer turns those measurements into an intrinsic manifold

This is the layer where recursive experiments first become geometric input.
