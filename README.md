![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-observatory_active-green)
![Runs](https://img.shields.io/badge/quenches-750-orange)

# PAM Observatory
Phase Analysis of Meaning — Information Geometry of Recursive Language Systems

![PAM Observatory](docs/figures/phase_report_panel.png)

---

## Overview

The PAM Observatory studies the geometry of recursive language systems across a parameter manifold:

θ = (r, α)

Using observable surfaces extracted from experimental quenches, the project constructs an intrinsic manifold using the Fisher Information Metric.

---

## Geometry Pipeline

experiments → index.csv → Fisher metric → geodesic distances → MDS embedding → curvature → phase seam

---

## Parameter Sweep

r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}  
α ∈ linspace(0.03, 0.15, 15)  
seeds = 10  

Total runs: **750 quenches**

---

## Documentation

Full documentation for the PAM Observatory is available in:

→ [`docs/`](docs/README.md)

---

## Current Status

| Component | Status |
|-----------|-------|
| Parameter Sweep | running |
| Geometry Pipeline | operational |
| Trajectory Recovery | implemented |
| Observatory Interface | in development |
