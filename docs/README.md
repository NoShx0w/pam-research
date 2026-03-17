# PAM Observatory Documentation

This repository contains the experimental and analytical infrastructure for the **Phase Analysis of Meaning (PAM) Observatory**.

The observatory studies the geometry of recursive language systems across a parameter manifold:

θ = (r, α)

using observable surfaces and Fisher Information geometry.

---

# Documentation Map

## 1. Observatory

- [How to Read the PAM Observatory](01_observatory/how_to_read.md)
- [Observable Glossary](01_observatory/observable_glossary.md)

## 2. Geometry

- [Parameter Sweep Geometry](02_geometry/parameter_sweep_geometry.md)
- [Geometry Pipeline](02_geometry/geometry_pipeline.md)

## 3. Analysis Pipeline

- [Phase Geometry](03_pipeline/phase_geometry.md)

## 4. Observatory Interface

- `04_interface/` reserved for TUI and observatory interface documentation

## 5. Project

- `05_project/` reserved for roadmap and reproducibility guides

---

The pipeline converts experimental observables into an intrinsic manifold geometry:

experiments → index.csv → Fisher metric → geodesic distances → MDS embedding → curvature → phase seam

Current experiment:

r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
α ∈ linspace(0.03, 0.15, 15)
seeds = 10

Total runs: 750 quenches
