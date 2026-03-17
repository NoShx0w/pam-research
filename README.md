![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-observatory_active-green)
![Runs](https://img.shields.io/badge/quenches-750-orange)

# PAM Observatory

**Phase Analysis of Meaning — Information Geometry of Recursive Language Systems**

The PAM Observatory studies the geometry of recursive language systems across a parameter manifold, revealing an emergent phase structure using Fisher information geometry.

---

## Phase Flow on the PAM Manifold

![Phase Flow on the PAM Manifold](docs/figures/phase_flow_on_manifold.png)

Each point represents a parameter configuration \((r, \alpha)\), embedded using Fisher–information geodesic distances.  
Color encodes a **signed phase coordinate**, revealing two distinct regimes separated by an emergent phase boundary (black curve).  
Critical points (stars) concentrate along this boundary, indicating regions of maximal structural change.

This provides a **purely data-driven phase diagram**, derived from the intrinsic geometry of the system.

---

## Overview

The observatory constructs an intrinsic manifold from observable surfaces extracted from experimental quenches.  
Using the Fisher Information Metric, the system is embedded into a geometric space where curvature and phase structure emerge.

---

## Geometry Pipeline

```text
experiments → index.csv → Fisher metric → geodesic distances → MDS embedding → curvature → phase seam
```
The pipeline converts experimental observables into an intrinsic manifold geometry.

---

## Parameter Sweep

```text
r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}  
α ∈ linspace(0.03, 0.15, 15)  
seeds = 10
```
**Total runs: 750 quenches**

---

## Documentation

The observatory is structured into five layers:

- **Observatory** — how to read outputs and observables  
- **Geometry** — parameter sweep and Fisher information geometry  
- **Pipeline** — transformation from data to manifold  
- **Interface** — visualization and TUI tools  
- **Project** — roadmap and reproducibility  

→ See [`docs/`](docs/)

---

## Current Status

| Component              | Status         |
|-----------------------|----------------|
| Parameter Sweep       | running        |
| Geometry Pipeline     | operational    |
| Trajectory Recovery   | implemented    |
| Observatory Interface | in development |

---

## Repository Structure

```text
experiments/     # parameter sweep + data generation
src/pam/         # core library (metrics, geometry)
tools/           # analysis tools (e.g. phase_movie)
tui/             # observatory interface (in development)
docs/            # full documentation
outputs/         # generated data + geometry artifacts
```
---

---

## Core Idea

The PAM Observatory treats a parameter sweep not as a collection of independent experiments, but as a **continuous geometric object**.

From observable surfaces, we construct:

- a **Fisher information metric**
- a **geodesic distance graph**
- a **low-dimensional manifold embedding**
- a **curvature field**
- and an emergent **phase structure**

The result is a system where **phase transitions are discovered, not imposed**.

---

## Reproducibility

All results are generated from:

```bash
python experiments/exp_batch.py
```
Followed by the geometric pipeline:
```bash
python experiments/fim.py
python experiments/fim_distance.py
python experiments/fim_mds.py
python experiments/fim_curvature.py
```
Phase extraction and visualization:
```bash
python experiments/fim_signed_phase.py
python experiments/fim_canonical_figure.py
```
---

## License

![MIT License](LICENSE)