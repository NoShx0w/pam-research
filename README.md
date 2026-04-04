# PAM Observatory

**Phase Analysis of Meaning (PAM)**

The PAM Observatory is a computational instrument for studying phase structure in recursive language systems.

It combines controlled corpus dynamics, information geometry, phase extraction, topology analysis, and operator-based probing to reveal how systems organize, transition, and stabilize across a parameter manifold.

![Python](https://img.shields.io/badge/python-3.14-blue)
![Status](https://img.shields.io/badge/status-observatory_active-green)
![Runs](https://img.shields.io/badge/quenches-750-orange)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19218700.svg)

---

## Preprint

Preprint on Zenodo: [Geometric Constraints on Transition Dynamics in Recursive Language Systems](https://zenodo.org/records/19218700)  
DOI: [10.5281/zenodo.19218700](https://doi.org/10.5281/zenodo.19218700)

---

## Status

Active research repository.  
The canonical instrument architecture is now implemented in layered package form under `src/pam/`, with a single full-pipeline entrypoint under `scripts/`.

---

## Canonical Entry Point

Run the full instrument with:

```bash
bash scripts/run_full_pipeline.sh
```

This executes the canonical file-first pipeline over the current `outputs/` artifact store.

---

## Phase Flow on the PAM Manifold

![Phase Flow on the PAM Manifold](docs/figures/phase_flow_on_manifold.png)

Each point represents a parameter configuration $(r, \alpha)$, embedded using Fisher-geodesic distances.  
Color encodes a **signed phase coordinate**, revealing two distinct regimes separated by an emergent phase boundary.  
Critical points concentrate near that boundary, indicating regions of maximal structural change.

This yields a **data-driven phase diagram** derived from the intrinsic geometry of the system.

---

## Overview

We study the control manifold

```math
\theta = (r, \alpha)
```

by running controlled recursive experiments and extracting structure at multiple levels:

- **engine** — how the corpus evolves
- **measurement** — how invariants are scored
- **observables** — what is measured from runs
- **geometry** — how states are arranged
- **phase** — how regimes are separated
- **operators** — how the manifold is actively probed
- **topology** — how the field is structurally organized

The repository has evolved from a script-heavy analysis workflow into a **canonical layered instrument**.

---

## Core Idea

> The goal is not to identify what a system *is*,  
> but how it *behaves under transformation*.

This leads to a central principle:

> **Topology is the relational identity of the field.**

Two runs are considered structurally equivalent if they preserve:

- critical organization
- seam relationships
- basin and transition structure

not if they merely look similar in raw coordinates.

---

## Canonical Pipeline

The instrument now runs through these stages:

```text
engine
↓
measurement
↓
observables
↓
geometry
  - Fisher metric
  - distance graph
  - embedding
  - curvature
↓
phase
  - seam extraction
  - seam distance
  - signed phase
↓
operators
  - geodesic extraction
  - probes
  - scaled probes
  - Lazarus regime
  - transition-rate analysis
↓
topology
  - field alignment
  - gradient alignment
  - critical structure
  - organizational summaries
```

The canonical orchestration lives in:

- `src/pam/pipeline/stages/`
- `src/pam/pipeline/runner.py`

and is exposed through:

- `scripts/run_full_pipeline.sh`

---

## Conceptual Layers

### Engine
Recursive corpus dynamics over the control parameters $(r, \alpha)$.

### Measurement
TIP and TIM instruments for scoring invariant and time/scale-stable structure.

### Observables
Entropy, macrostate, lag, regression, and related derived descriptors.

### Geometry
Fisher information metric, geodesic distance graph, manifold embedding, and curvature.

### Phase
Seam extraction, seam-relative geometry, and signed phase coordinates.

### Operators
Active probing of the manifold through geodesic extraction, canonical probes, scaled probes, and transition diagnostics.

### Topology
Critical structure, field alignment, organizational summaries, and phase-selection structure.

---

## Repository Structure

```text
src/pam/
  engine/         # runtime dynamics and injector logic
  measurement/    # TIP, TIM, and measurement builders
  observables/    # core and derived observables
  geometry/       # FIM, distances, embedding, curvature, geodesics
  phase/          # seam extraction, seam distance, signed phase
  topology/       # alignment, criticality, organization
  operators/      # probes, Lazarus, transition-rate analysis
  pipeline/       # PipelineState, stages, runner

observatory/
  corpora/        # externalized corpus payloads and registry
  runs/
  derived/
  reports/
  figures/

experiments/
  root wrappers and operational scripts
  figures/        # figure-generation scripts
  studies/        # active analytical studies
  toy/            # pedagogical / toy experiments
  archive/        # legacy and superseded material

scripts/
  canonical entrypoints and repository guards

outputs/
  active file-first artifact store
```

---

## Reproducibility

The canonical full run is:

```bash
bash scripts/run_full_pipeline.sh
```

The repository remains file-first:
- current derived outputs are written under `outputs/`
- corpus payloads are externalized under `observatory/corpora/`

Many legacy experiment wrappers are preserved under `experiments/` for compatibility and inspection, but the canonical runtime path is now the pipeline runner.

---

## Documentation

See:

- [`docs/README.md`](docs/README.md)

Useful repository anchors:

- `src/pam/pipeline/runner.py`
- `scripts/run_full_pipeline.sh`
- `observatory/corpora/README.md`

---

## Current State

- parameter sweep completed
- trajectory recovery completed
- canonical geometry layer implemented
- canonical phase layer implemented
- canonical operators layer implemented
- canonical topology layer implemented
- canonical pipeline stages and runner implemented
- corpora externalized into observatory data storage

---

## One-Line Summary

> The PAM Observatory is a layered instrument for extracting geometric, phase, topological, and operator structure from recursive language dynamics.

---

## License

[MIT License](LICENSE)
