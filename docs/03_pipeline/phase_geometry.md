# Phase Geometry

This document formalizes the phase structure discovered by the PAM Observatory.

The phase is not defined a priori. It emerges from the intrinsic geometry of the system constructed via the Fisher Information Metric.

## 1. From Geometry to Phase

The PAM pipeline produces:

θ = (r, α)
→ observables x(θ)
→ Fisher metric G(θ)
→ geodesic distances
→ manifold embedding (MDS)
→ curvature field

This defines an intrinsic manifold 𝓜 ⊂ ℝ².

## 2. Phase as a Scalar Field

We define a scalar function:

φ : 𝓜 → [-1, 1]

called the **signed phase coordinate**.

- φ > 0 → one dynamical regime
- φ < 0 → another regime
- φ ≈ 0 → transition region

## 3. Construction of the Phase

### 3.1 Identify the Phase Seam
A set of points is identified where geometric distance between regions is minimal, curvature is elevated, and/or observable transitions are sharp.

These points define a discrete curve S ⊂ 𝓜 called the **phase seam**.

### 3.2 Define Distance to Seam
For each point p ∈ 𝓜, compute d(p, S), the geodesic distance to the seam.

### 3.3 Assign Orientation (Sign)
Assign positive sign to one side of the seam and negative sign to the other, yielding φ(p) = ± d(p, S), normalized to [-1, 1].

## 4. Phase Boundary

The **phase boundary** is defined as φ(p) = 0. This corresponds exactly to the seam S.

It is:
- curved, not linear
- data-driven, not imposed
- intrinsic to the geometry

## 5. Critical Points

Critical points are locations where curvature is high and/or the gradient of φ is large.

These points:
- lie near the phase boundary
- act as anchors of the seam
- mark regions of maximal structural change

## 6. Canonical Representation

The phase structure is visualized as:

Phase Flow on the PAM Manifold

This combines:
- point cloud → manifold
- color → φ (phase)
- curve → phase boundary
- stars → critical points

## 7. Core Insight

The parameter sweep is not a collection of independent experiments.

It is a single geometric object with curvature, structure, and phase.

Phase transitions are therefore **properties of geometry, not parameters**.
