# Phase Geometry

This document defines how phase structure is extracted from the intrinsic geometry of the PAM manifold.

In the canonical PAM Observatory, the phase layer sits downstream of geometry and converts geometric structure into regime structure.

---

## Overview

The PAM Observatory reveals that the parameter manifold is not behaviorally uniform.

Instead, the intrinsic manifold is organized into distinct regimes separated by an emergent **phase boundary**, or **seam**.

The phase layer is responsible for:

- extracting candidate seam structure from geometry
- representing the seam in manifold coordinates
- computing distance to seam
- constructing a signed phase coordinate

---

## Geometry to Phase Transition

The phase layer does not begin from raw trajectories.

It begins from geometry outputs produced by the canonical geometry layer:

- manifold embedding
- curvature diagnostics
- intrinsic distances

These are used to infer the regime structure of the manifold.

So the conceptual flow is:

```text
observables
↓
geometry
  - metric
  - distance graph
  - embedding
  - curvature
↓
phase
  - seam extraction
  - seam embedding / backprojection
  - distance to seam
  - signed phase
```

---

## Parameter Manifold

The phase layer is defined over the control manifold

```math
\theta = (r, \alpha)
```

Each point corresponds to a full experimental configuration whose observable summaries induce a position in the intrinsic manifold.

---

## Seam Extraction

A first phase boundary is inferred from geometric structure, especially curvature.

In the implemented workflow, the seam is extracted as a candidate boundary from curvature-derived ridge structure in parameter space.

Interpretation:

- the seam is not imposed externally
- it emerges from the intrinsic geometry
- it identifies where regime structure is most sharply reorganized

---

## Seam Representation in Manifold Coordinates

Once candidate boundary points are identified in parameter space, the seam is represented in manifold coordinates using the MDS embedding.

This allows:

- ordering seam points along the embedded manifold
- fitting a seam-like curve in manifold space
- backprojecting seam structure to the parameter grid if needed

This step makes the boundary visible in the intrinsic representation rather than only in raw \((r, \alpha)\) coordinates.

---

## Distance to Seam

For each manifold point, the phase layer computes a seam-relative distance.

This quantity measures how far a point lies from the extracted regime boundary in intrinsic geometry.

Interpretation:

- small distance indicates seam proximity
- larger distance indicates basin depth or regime interior
- seam distance becomes a key interface to operator and topology analyses

---

## Signed Phase Coordinate

A signed phase coordinate is then constructed over the manifold.

Conceptually:

```math
\phi(\theta)
```

where:

- the magnitude reflects seam-relative distance
- the sign reflects which side of the seam the point occupies

In the implemented workflow, the sign is inferred from seam orientation in embedding space, while magnitude is normalized from distance-to-seam structure.

Interpretation:

- signed phase assigns each point to a regime
- the seam corresponds to the regime boundary
- the phase coordinate organizes the manifold into opposing sides of behavior

---

## Phase Boundary (Seam)

The seam is the structural boundary between phase regimes.

Conceptually, it corresponds to the zero set of the signed phase field:

```math
\phi(\theta) = 0
```

Interpretation:

- separates distinct behavioral regimes
- defines an outcome-equivalence boundary
- marks where the manifold reorganizes qualitatively

---

## Phase Transition Behavior

Near the seam:

- observables reorganize rapidly
- curvature often increases
- seam-relative distance becomes small
- operator trajectories may compress, graze, cross, or flip

So the seam is not merely a plotting artifact; it is a structural transition object in the manifold.

---

## Lazarus Regime

Near the seam, a pre-collapse or boundary-activated region can emerge.

In the current observatory, the Lazarus regime is analyzed downstream using:

- curvature
- seam proximity
- signed phase centering

Interpretation:

- metastable boundary-adjacent structure
- precursor to phase transition
- candidate predictive signal for operator-path outcomes

The Lazarus regime is therefore not identical to the seam, but is strongly organized by seam-relative geometry.

---

## Phase on the MDS Embedding

The phase structure is especially visible in embedded manifold coordinates:

- MDS preserves intrinsic distance relationships approximately
- coloring by signed phase reveals global regime organization
- the seam becomes visible as a manifold boundary curve

This is one of the main ways the observatory makes regime structure inspectable.

---

## Phase Gradient

The phase field also supports local directional interpretation through its gradient:

```math
\nabla \phi(\theta)
```

Interpretation:

- indicates direction of strongest phase change
- reveals local transition direction
- supports downstream alignment analyses with Lazarus and response fields

In the current repository, gradient-style alignment is treated mainly in downstream topology and response analyses rather than as the core phase runtime itself.

---

## Relation to Geometry

Phase is not independent of geometry.

It is derived from geometry and remains constrained by it.

In practice, phase structure is closely related to:

- curvature ridges
- manifold distance organization
- embedding structure
- geodesic accessibility

Interpretation:

> phase is a geometric organization of behavior, not an externally attached label

---

## Interaction with Geodesics and Operators

Operator paths interact with phase structure in several ways:

- they may remain within one regime
- they may graze the seam
- they may cross the seam
- they may enter Lazarus-like boundary structure before flipping

This makes phase geometry central to operator analysis, because seam-relative structure determines accessibility, transition ordering, and regime stability.

---

## Canonical Implementation Mapping

### Canonical modules

| Concept | Canonical module |
|--------|------|
| Seam extraction | `src/pam/phase/seam.py` |
| Seam embedding / backprojection | `src/pam/phase/seam_embedding.py` |
| Distance to seam | `src/pam/phase/seam_distance.py` |
| Signed phase | `src/pam/phase/signed_phase.py` |

### Pipeline orchestration

| Concept | Pipeline module |
|--------|------|
| Phase stage | `src/pam/pipeline/stages/phase.py` |
| Full pipeline runner | `src/pam/pipeline/runner.py` |
| Canonical shell entrypoint | `scripts/run_full_pipeline.sh` |

### Compatibility wrappers

| Concept | Wrapper script |
|--------|------|
| Seam extraction | `experiments/fim_phase_boundary.py` |
| Seam embedding / backprojection | `experiments/fim_phase_boundary_mds.py` |
| Distance to seam | `experiments/fim_phase_distance.py` |
| Signed phase | `experiments/fim_signed_phase.py` |

---

## Active Phase Outputs

The phase layer writes active artifacts under:

- `outputs/fim_phase/`

Important outputs include:

- `phase_boundary_points.csv`
- `phase_boundary_mds_backprojected.csv`
- `phase_distance_to_seam.csv`
- `signed_phase_coords.csv`
- `signed_phase_on_grid.png`
- `signed_phase_on_mds.png`

These outputs are then consumed by downstream operator and topology analyses.

---

## Summary

Phase geometry converts intrinsic manifold structure into regime structure.

In the canonical PAM Observatory, the phase layer provides:

- seam extraction
- seam-relative geometry
- regime assignment through signed phase
- the structural interface between geometry and operator/topology analysis

This is the layer where manifold shape becomes behavioral organization.
