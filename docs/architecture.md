# PAM Observatory Architecture

This document describes the canonical architecture of the PAM Observatory.

The repository is now organized as a **layered computational instrument** for studying phase structure in recursive language systems.

Its canonical structure is:

1. Engine  
2. Measurement  
3. Observables  
4. Geometry  
5. Phase  
6. Operators  
7. Topology  
8. Pipeline orchestration  

The system remains **file-first**: stages communicate through explicit artifacts written to the repository’s active output store.

---

## System Overview

The canonical entrypoint is:

```bash
bash scripts/run_full_pipeline.sh
```

which invokes the pipeline runner:

```text
src/pam/pipeline/runner.py
```

The canonical stage order is:

```text
engine
↓
measurement
↓
observables
↓
geometry
↓
phase
↓
operators
↓
topology
```

The pipeline is orchestrated through:

```text
src/pam/pipeline/stages/
src/pam/pipeline/runner.py
```

and uses:

```text
src/pam/pipeline/state.py
src/pam/io/paths.py
```

to address artifact families.

---

## 1. Engine

The engine governs recursive corpus dynamics over the control manifold.

Canonical package:

```text
src/pam/engine/
```

Primary responsibilities:

- corpus evolution over $(r, \alpha)$
- mixture/quench execution
- injector composition
- runtime dynamics wrappers

Key modules include:

- `src/pam/engine/core.py`
- `src/pam/engine/mixture.py`
- `src/pam/engine/injectors.py`

The engine is responsible for how the system changes over time, but not for higher-level interpretation of those changes.

---

## 2. Measurement

The measurement layer scores invariant structure in texts and rescaled views.

Canonical package:

```text
src/pam/measurement/
```

Primary responsibilities:

- invariant scoring through TIP
- time/scale-invariant diagnostics through TIM
- reusable builders for active measurement configuration

Key modules include:

- `src/pam/measurement/tip.py`
- `src/pam/measurement/tim.py`
- `src/pam/measurement/builders.py`

This layer turns raw text states into structured measurements.

---

## 3. Observables

The observables layer derives scalar and sequence-valued descriptors from runs.

Canonical package:

```text
src/pam/observables/
```

Primary responsibilities:

- entropy observables
- macrostate and microstructure observables
- lag/correlation analysis
- regression-derived summaries

Key modules include:

- `src/pam/observables/core.py`
- `src/pam/observables/derived.py`

This is the bridge between measurement and geometry.

---

## 4. Geometry

The geometry layer constructs the intrinsic manifold induced by observables.

Canonical package:

```text
src/pam/geometry/
```

Primary responsibilities:

- Fisher-type metric estimation
- geodesic distance graph construction
- manifold embedding
- scalar curvature estimation
- geodesic tracing and extraction

Key modules include:

- `src/pam/geometry/fisher_metric.py`
- `src/pam/geometry/distance_graph.py`
- `src/pam/geometry/embedding.py`
- `src/pam/geometry/curvature.py`
- `src/pam/geometry/geodesics.py`

This layer answers: **how are states arranged?**

---

## 5. Phase

The phase layer extracts regime structure from the geometry.

Canonical package:

```text
src/pam/phase/
```

Primary responsibilities:

- seam / boundary extraction
- seam embedding and backprojection
- distance to seam
- signed phase coordinates

This layer turns geometry into regime structure.

---

## 6. Operators

The operators layer actively probes the manifold.

Canonical package:

```text
src/pam/operators/
```

Primary responsibilities:

- geodesic extraction
- canonical probes
- scaled probe experiments
- Lazarus regime diagnostics
- transition-rate estimation

Key modules include:

- `src/pam/operators/geodesic_extraction.py`
- `src/pam/operators/probes.py`
- `src/pam/operators/scaled_probes.py`
- `src/pam/operators/lazarus.py`
- `src/pam/operators/transition_rate.py`

Operators answer: **how does the system behave under controlled traversal or perturbation?**

---

## 7. Topology

The topology layer analyzes structural organization of the phase field and operator response.

Canonical package:

```text
src/pam/topology/
```

Primary responsibilities:

- field alignment
- gradient alignment
- critical point / criticality summaries
- organizational topology
- phase-selection structure

This layer answers: **how is the field organized?**

---

## 8. Pipeline Orchestration

The orchestration layer composes the full instrument.

Canonical package:

```text
src/pam/pipeline/
```

Key components:

- `src/pam/pipeline/state.py`
- `src/pam/pipeline/stages/geometry.py`
- `src/pam/pipeline/stages/phase.py`
- `src/pam/pipeline/stages/operators.py`
- `src/pam/pipeline/stages/topology.py`
- `src/pam/pipeline/runner.py`

The runner uses a shared `PipelineState` to address current artifact families through the active file-first output store.

Canonical shell entrypoint:

```text
scripts/run_full_pipeline.sh
```

---

## Artifact Model

The repository remains explicitly file-first.

### Active output store

Current derived artifacts are written under:

```text
outputs/
```

Important active families include:

- `outputs/index.csv`
- `outputs/trajectories/`
- `outputs/fim/`
- `outputs/fim_distance/`
- `outputs/fim_mds/`
- `outputs/fim_curvature/`
- `outputs/fim_phase/`
- `outputs/fim_ops/`
- `outputs/fim_ops_scaled/`
- `outputs/fim_lazarus/`
- `outputs/fim_transition_rate/`
- `outputs/fim_field_alignment/`
- `outputs/fim_gradient_alignment/`
- `outputs/fim_critical/`
- `outputs/fim_initial_conditions/`

### Observatory data root

Corpora and longer-lived observatory data are externalized under:

```text
observatory/
```

especially:

```text
observatory/corpora/
```

This separation keeps corpus payloads out of canonical implementation code.

---

## Experiment Layer vs Canonical Layer

The repository still contains an `experiments/` directory, but it no longer defines the canonical architecture.

Instead:

- canonical code lives under `src/pam/`
- `experiments/` now contains:
  - thin compatibility wrappers
  - operational study scripts
  - figure-generation scripts
  - toy experiments
  - archived legacy material

Current organization:

```text
experiments/
  figures/
  studies/
  toy/
  archive/
```

This keeps exploratory and historical work available without confusing it with the canonical instrument.

---

## Design Principles

### Layered ownership

Each conceptual layer has a canonical package home.

---

## Pipeline scope note

PAM is a file-first observatory. The canonical runtime pipeline currently orchestrates derived-stage analysis over an existing artifact store. In its present form, `scripts/run_full_pipeline.sh` rebuilds the geometry → phase → operators → initial-conditions → topology layers from existing upstream outputs.

It should not be read as a single raw-corpus-to-final-result pipeline. Upstream corpus generation, measurement, and observable construction are part of the broader instrument, but they are not all regenerated by this entrypoint.

This distinction is intentional: intermediate artifacts are inspectable, reusable, and auditable. The file-first design makes each layer visible rather than hiding the full research state inside one monolithic run.

### Reproducibility

Corpora are externalized, outputs are inspectable, and the canonical runtime is explicit.

### Compatibility-preserving refactoring

Legacy wrappers and shims are retained where useful so the repository can evolve without abrupt breakage.

### Separation of canonical and exploratory work

Canonical instrument code lives under `src/pam/`; non-canonical studies and figures live under `experiments/`.

---

## Historical Note

Earlier versions of the repository centered more strongly on:

- flat experiment scripts
- terminal observatory interfaces
- visualization tooling
- `outputs/index.csv` as the main live interface

Those workflows remain historically important, but the architecture has since been consolidated into the layered instrument described above.

---

## Summary

The PAM Observatory is now best understood as a **canonical layered instrument** for extracting geometric, phase, topological, and operator structure from recursive language dynamics.

Its core logic lives in `src/pam/`, its corpora live in `observatory/corpora/`, and its canonical runtime is exposed through:

```bash
bash scripts/run_full_pipeline.sh
```
