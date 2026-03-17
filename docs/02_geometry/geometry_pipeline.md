# Geometry Pipeline

This document describes how experimental data is transformed into an intrinsic geometric structure.

The PAM Observatory does not treat experiments as independent results. Instead, it constructs a continuous manifold over parameter space using information geometry.

## Overview

The full pipeline is:

experiments → observables → Fisher metric → geodesic distances → embedding → curvature → phase

Each stage adds a layer of structure.

## 1. Experiments

The system is evaluated over a parameter manifold:

θ = (r, α)

For each configuration:
- a trajectory is generated
- multiple seeds are run
- observables are extracted

Output:

outputs/index.csv

## 2. Observables

Each parameter configuration produces a feature vector x(θ) ∈ ℝⁿ.

These observables capture entropy, coupling, correlation, and temporal structure.

## 3. Fisher Information Metric

From the observable field, we construct a metric:

G_ij(θ) = ∂_i x(θ)ᵀ Σ⁻¹ ∂_j x(θ)

This defines a **local geometry** on parameter space.

## 4. Geodesic Distances

Using the metric, we construct a graph over parameter points:
- nodes → parameter configurations
- edges → neighboring points
- weights → local metric distance

From this graph, we compute geodesic distance ≈ shortest path distance.

## 5. Manifold Embedding (MDS)

We embed the distance matrix into a low-dimensional space 𝓜 ⊂ ℝ² using Multidimensional Scaling (MDS).

## 6. Curvature

From the metric and/or embedding, we compute a curvature field.

Curvature measures how the geometry bends or concentrates.

## 7. Phase Extraction

The final step is the extraction of a phase structure from the geometry:
- identify a phase seam
- compute distance to the seam
- assign a signed phase coordinate

Result:

φ : 𝓜 → [-1, 1]

See: `../03_pipeline/phase_geometry.md`

## 8. Key Insight

The pipeline does not analyze parameters directly. Instead, it constructs geometry from how observables change, how configurations relate, and how structure emerges globally.

Phase transitions are therefore **properties of the manifold, not of individual parameters**.
