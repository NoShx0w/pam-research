# Phase Geometry

This document defines how phase structure emerges on the PAM manifold and how phase boundaries are detected, interpreted, and analyzed.

---

## Overview

The PAM Observatory reveals that the parameter manifold is not uniform, but divided into distinct behavioral regimes.

These regimes are separated by a **phase boundary (seam)** embedded in the geometry.

---

## Parameter Manifold

The system is defined over:

```math
\theta = (r, \alpha)
```

Each point corresponds to a full experimental configuration whose observables define its position in the manifold.

---

## Observable Field

Each parameter point produces a vector of observables:

- piF_tail  
- H_joint_mean  
- corr0  
- delta_r2_freeze  
- delta_r2_entropy  
- K_max  

These observables define the local state of the system.

---

## Signed Phase Field

A scalar phase field is constructed over the manifold:

```math
\phi(\theta)
```

This field assigns each point to a phase regime.

Typical construction:
- normalization of key observables
- projection onto a dominant axis of variation
- sign assignment based on behavioral regime

---

## Phase Boundary (Seam)

The phase boundary is defined as:

```math
\phi(\theta) = 0
```

This is the set of points where the system transitions between regimes.

Interpretation:
- separates distinct dynamical behaviors
- represents an outcome-equivalence boundary
- defines a structural discontinuity in the system

---

## Phase Transition Behavior

Across the seam:

- observables change rapidly
- Fisher metric determinant increases
- curvature often peaks
- trajectories exhibit compression or instability

---

## Lazarus Regime

Near the seam, a pre-collapse region can emerge:

- high curvature
- strong gradients in phase
- trajectories slow or compress
- system has not yet transitioned

Interpretation:
- metastable region
- precursor to phase transition
- candidate predictive signal

---

## Phase on MDS Embedding

The phase field can be visualized in embedded coordinates:

- MDS embedding preserves geodesic structure
- phase coloring reveals global topology
- seam appears as a visible boundary in embedding space

---

## Phase Gradient

The gradient of the phase field:

```math
\nabla \phi(\theta)
```

Indicates:
- direction of strongest phase change
- local transition pressure
- alignment with curvature structures

---

## Relation to Geometry

Phase structure is not independent:

- aligns with curvature ridges
- correlates with Fisher determinant spikes
- influences geodesic paths

Interpretation:
- phase is a geometric feature, not an external label

---

## Interaction with Geodesics

Geodesic paths interact with the phase structure:

- may cross the seam
- may bend toward or away from it
- may dwell in the Lazarus region

This interaction defines:

- phase transition paths
- stability of trajectories
- accessibility of regimes

---

## Implementation Mapping

| Concept | File |
|--------|------|
| Phase computation | experiments/fim_signed_phase.py |
| Phase visualization | experiments/fim_phase_report.py |
| MDS projection | experiments/fim_mds.py |

---

## Outputs

- outputs/fim_phase/
- signed_phase_on_grid.png
- signed_phase_on_mds.png
- phase_boundary.png

---

## Summary

Phase geometry defines how qualitative system behavior is organized over the parameter manifold.

- Phase field → assigns regime
- Seam → defines boundary
- Gradient → defines transition direction
- Lazarus region → indicates imminent transition

This layer transforms the geometric manifold into a structured space of behaviors.
