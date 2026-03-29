# Geometry Analysis Pipeline

This document defines how experimental observables are transformed into an intrinsic manifold geometry in the PAM Observatory.

The geometry layer is the canonical bridge from measured system behavior to manifold structure.

---

## Pipeline Overview

![Geometry Pipeline](../figures/geometry_pipeline.svg)

---

## Role of the Geometry Layer

The geometry layer takes observable summaries defined over the control manifold

```math
\theta = (r, \alpha)
```

and constructs an intrinsic geometric description of that manifold.

In the canonical instrument, geometry is responsible for:

- Fisher-type metric estimation
- geodesic distance graph construction
- manifold embedding
- curvature estimation
- geodesic tracing / extraction

These outputs form the substrate for the downstream phase layer.

---

## Parameter Manifold

The PAM Observatory studies a parameter manifold:

```math
\theta = (r, \alpha)
```

where:

- \(r\) controls recursion / coupling strength
- \(\alpha\) controls update / mixing dynamics

Each parameter point is associated with observable summaries derived from recursive experiments.

---

## Observable Representation

At each parameter point, the system is represented by an observable vector

```math
m(\theta)
```

constructed from experiment summaries such as:

- freeze statistics
- entropy statistics
- correlation structure
- regression-derived quantities
- other derived observables carried in `outputs/index.csv`

These observable summaries define the measurable state from which geometry is inferred.

---

## Fisher Information Metric

The Fisher-type metric defines the local geometry:

```math
G_{ij}(\theta) = \partial_i m(\theta)^T \Sigma^{-1} \partial_j m(\theta)
```

where:

- \(m(\theta)\) is the observable vector
- \(\Sigma\) is the empirical covariance of observable noise

Interpretation:

- the metric measures how strongly nearby parameter points are distinguishable through observables
- large metric values indicate sharp behavioral sensitivity
- anisotropy indicates directional structure in the manifold

This stage produces the local metric tensor and associated diagnostics.

---

## Geodesic Distance Graph

The metric defines local geometry, but not global distances directly.

To recover intrinsic global structure, the system constructs a graph over the parameter grid:

- nodes = parameter points
- edges = local Fisher-derived distances

Shortest-path distances on this graph approximate manifold geodesics.

Interpretation:

- converts local distinguishability into global manifold structure
- defines the intrinsic distance geometry used downstream
- supports path extraction across the manifold

---

## MDS Embedding

Once intrinsic distances are available, the manifold is embedded in low dimension using multidimensional scaling (MDS).

This produces coordinates such as:

- `mds1`
- `mds2`

Interpretation:

- preserves geodesic-distance structure as well as possible in 2D
- reveals large-scale manifold organization
- makes seams, folds, and basin-like organization visually inspectable

The embedding is not the geometry itself; it is a visualization-compatible coordinate system for the geometry.

---

## Curvature

Curvature is estimated from the metric field.

Interpretation:

- high curvature indicates regions where the observable geometry changes rapidly
- curvature ridges often align with phase transition structure
- curvature provides a local signal for downstream seam extraction

In the canonical pipeline, curvature is a geometry output that feeds the phase layer.

---

## Geometry to Phase Interface

The geometry layer prepares the substrate for phase extraction by producing:

- local metric structure
- intrinsic manifold distances
- low-dimensional embedding coordinates
- curvature diagnostics

These feed the downstream phase layer, where seam structure, seam distance, and signed phase are inferred.

So geometry answers:

> **How are observable states arranged intrinsically across parameter space?**

Phase then answers:

> **Where do qualitative regime boundaries emerge on that manifold?**

---

## Geodesics and the Connection

The canonical geometry layer includes geodesic structure in two forms:

### Discrete geodesics (implemented)

In the current canonical runtime, geodesics are handled primarily through the Fisher-distance graph:

- local edge costs are derived from the metric
- shortest paths approximate intrinsic geodesics
- these support geodesic extraction and operator probing

This is the implemented path used by the active instrument.

### Connection-based geodesics (conceptual / advanced)

A continuous Riemannian treatment introduces the Levi-Civita connection through the Christoffel symbols:

```math
\Gamma^k_{ij} = \frac{1}{2} G^{kl}
\left(
\partial_i G_{jl} +
\partial_j G_{il} -
\partial_l G_{ij}
\right)
```

Interpretation:

- the Christoffel symbols describe how tangent directions bend under the metric
- they define continuous geodesic flow
- they connect the metric field to continuous manifold dynamics

In the PAM Observatory, this object belongs conceptually to the **geometry layer**, not to phase or topology.

However, it is best understood as an **advanced geometric structure derived from the metric**, rather than as part of the minimal canonical runtime path.

So its place in the canonical picture is:

```text
observables
↓
metric
↓
connection / geodesic structure
↓
distance / embedding / curvature
↓
phase
```

Practically:

- the current pipeline implements discrete geodesic structure directly
- Christoffel-symbol analysis is a valid canonical extension of geometry
- it is especially relevant for continuous geodesic interpretation and visualization-oriented research arcs

For this reason, the Christoffel symbols are retained here as part of the geometry story, but treated as an advanced geometric layer rather than a required runtime stage.

---

## Canonical Implementation Mapping

### Canonical modules

| Concept | Canonical module |
|--------|------|
| Fisher metric | `src/pam/geometry/fisher_metric.py` |
| Distance graph | `src/pam/geometry/distance_graph.py` |
| MDS embedding | `src/pam/geometry/embedding.py` |
| Curvature | `src/pam/geometry/curvature.py` |
| Geodesics | `src/pam/geometry/geodesics.py` |

### Pipeline orchestration

| Concept | Pipeline module |
|--------|------|
| Geometry stage | `src/pam/pipeline/stages/geometry.py` |
| Full pipeline runner | `src/pam/pipeline/runner.py` |
| Canonical shell entrypoint | `scripts/run_full_pipeline.sh` |

### Compatibility wrappers

| Concept | Wrapper script |
|--------|------|
| Fisher metric | `experiments/fim.py` |
| Distance graph | `experiments/fim_distance.py` |
| MDS embedding | `experiments/fim_mds.py` |
| Curvature | `experiments/fim_curvature_scalar.py` |
| Geodesic path wrapper | `experiments/fim_geodesic.py` |
| Geodesic fan wrapper | `experiments/fim_geodesic_fan.py` |

---

## Active Geometry Outputs

The geometry layer writes active artifacts under:

- `outputs/fim/`
- `outputs/fim_distance/`
- `outputs/fim_mds/`
- `outputs/fim_curvature/`

These geometry outputs are then consumed by downstream phase, operator, and topology stages.

---

## Design Principles

### Layer ownership

Geometry is its own canonical layer under `src/pam/geometry/`.

### File-first interfaces

Geometry stages derive their state from observable artifacts and write explicit outputs to disk.

### Intrinsic over coordinate-first analysis

The goal is not merely to plot parameter space, but to recover the intrinsic structure induced by observable behavior.

### Compatibility-preserving evolution

Legacy wrappers remain available, but canonical ownership now lives in the geometry package and pipeline stage modules.

---

## Summary

The geometry pipeline converts observable behavior into an intrinsic manifold structure.

In the canonical PAM Observatory, geometry provides:

- a local metric
- intrinsic distances
- an embedding-compatible manifold representation
- curvature diagnostics
- geodesic structure

This forms the geometric backbone of the instrument and supplies the substrate on which phase, operators, and topology are built.
