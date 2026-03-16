# PAM Observatory Documentation

This repository contains the experimental and analytical infrastructure for the **Phase Analysis of Meaning (PAM) Observatory**.

The observatory studies the geometry of recursive language systems across a parameter manifold:

θ = (r, α)

using observable surfaces and Fisher Information geometry.

---

# Documentation Map

## 1. Observatory

Understanding the experimental instrument and its outputs.

- [How to Read the PAM Observatory](01_observatory/how_to_read_the_observatory.md)
- [Observable Glossary](01_observatory/observables_glossary.md)

---

## 2. Geometry

Conceptual explanation of the parameter sweep and the information geometry.

- [Parameter Sweep Geometry](02_geometry/parameter_sweep_geometry.md)
- [Fisher Information Geometry](02_geometry/fisher_information_geometry.md)

---

## 3. Analysis Pipeline

How experimental data is transformed into geometric structure.

- [Geometry Analysis Pipeline](03_pipeline/geometry_pipeline.md)
- [Phase Boundary Detection](03_pipeline/phase_boundary_detection.md)

---

## 4. Observatory Interface

Visualization and exploration tools.

- [Observatory TUI](04_interface/observatory_tui.md)

---

## 5. Project

Project overview and reproducibility.

- [Project Roadmap](05_project/roadmap.md)
- [Reproducibility Guide](05_project/reproducibility.md)

---

# Geometry Pipeline Overview

![Geometry Pipeline](figures/geometry_pipeline.svg)

The pipeline converts experimental observables into an intrinsic manifold geometry:
```text
experiments → index.csv → Fisher metric → geodesic distances → MDS embedding → curvature → phase seam
```
---

# Current Experiment

Parameter sweep:
```text
750 quenches
```
---

# Repository

Main repository:

https://github.com/NoShx0w/pam-research
