# How to Read the PAM Observatory

This document explains how to interpret the outputs of the PAM Observatory.

The observatory transforms a parameter sweep into a geometric object. Each visualization represents a different aspect of that geometry.

## 1. The Parameter Space

The system is defined over a 2D parameter manifold:

θ = (r, α)

- **r** — controls the recursive strength of the system
- **α** — controls the coupling / update sensitivity

Each point in this space corresponds to one experimental configuration.

## 2. Observables (index.csv)

Each experiment produces a trajectory, from which observable quantities are extracted.

These are stored in:

outputs/index.csv

Each row corresponds to one (r, α) configuration.

Observables include:
- entropy-like measures
- coupling measures
- correlation statistics

See: `observable_glossary.md` for definitions.

## 3. The Geometry Pipeline

The observatory constructs geometry in the following steps:

observables → Fisher metric → geodesic distances → embedding → curvature → phase

| Stage | Meaning |
|------|--------|
| Observables | raw measurements |
| Fisher Metric | local sensitivity / distinguishability |
| Distance Graph | global structure (geodesics) |
| MDS Embedding | visualizable manifold |
| Curvature | geometric instability |
| Phase | large-scale regime structure |

## 4. Reading the Plots

### 4.1 Parameter Heatmaps

Example:
- `log10 det(G)`
- `log10 |curvature|`
- `distance to seam`

These are plotted over the (r, α) grid.

### 4.2 MDS Manifold

The MDS plot shows the parameter space embedded into a low-dimensional geometry.

- Each point = one (r, α)
- Distance between points ≈ geodesic distance under the Fisher metric

### 4.3 Curvature

Curvature highlights where the geometry bends or concentrates.

- High curvature → sensitive / unstable regions
- Low curvature → stable regions

## 5. Phase Structure

The most important observable is the **phase field**.

### 5.1 Signed Phase

Each point is assigned a scalar:

φ ∈ [-1, 1]

- positive values → one regime
- negative values → another regime
- near zero → transition region

### 5.2 Phase Boundary (Seam)

The **phase seam** is the curve where:

φ ≈ 0

This is the boundary between regimes. It is not linear and not imposed; it emerges from the geometry.

### 5.3 Critical Points

Critical points are locations where:
- curvature is high, and/or
- phase transitions are sharp

They act as anchors of the phase boundary.

## 6. Canonical View

The most complete representation is:

Phase Flow on the PAM Manifold

This combines:
- color → phase
- curve → phase boundary
- stars → critical points

## 7. Key Insight

The PAM Observatory treats a parameter sweep as a **geometric object**.

Rather than asking “What happens at each parameter?” it asks “What is the shape of the system across parameters?”

Phase structure, curvature, and transitions are then properties of that shape.
