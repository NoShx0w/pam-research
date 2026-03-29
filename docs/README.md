# PAM Observatory — Documentation

## Overview

This documentation describes the canonical architecture, runtime flow, and research context of the PAM Observatory.

The repository is now organized as a **layered computational instrument** for studying phase structure in recursive language systems across the control manifold

\[
\theta = (r, \alpha)
\]

through:

- recursive corpus dynamics
- invariant measurement
- observable extraction
- information geometry
- phase structure
- operator-driven probing
- topological organization

The canonical runtime is now exposed through:

```bash
bash scripts/run_full_pipeline.sh
```

---

## Documentation Index

### Core architecture

- [`architecture.md`](architecture.md)  
  Canonical repository architecture and layer ownership

- [`README.md`](../README.md)  
  Top-level repository overview and canonical runtime entrypoint

---

### Geometry

- [`02_geometry/geometry_pipeline.md`](02_geometry/geometry_pipeline.md)  
  Geometry pipeline from observables to Fisher manifold structure

- [`02_geometry/parameter_sweep_geometry.md`](02_geometry/parameter_sweep_geometry.md)  
  Parameter-space coverage and geometric interpretation of the \((r, \alpha)\) sweep

---

### Phase, topology, and operators

- [`03_pipeline/phase_geometry.md`](03_pipeline/phase_geometry.md)  
  Seam extraction, signed phase, and phase interpretation

- [`03_pipeline/field_topology.md`](03_pipeline/field_topology.md)  
  Field alignment, critical structure, and topological organization

- [`03_pipeline/operators.md`](03_pipeline/operators.md)  
  Geodesic extraction, probes, scaled probes, and operator-based analysis

---

### Observatory and interpretation

- [`01_observatory/how_to_read.md`](01_observatory/how_to_read.md)  
  How to read observatory outputs and manifold-derived artifacts

- [`01_observatory/observable_glossary.md`](01_observatory/observable_glossary.md)  
  Definitions of key observables and derived quantities

- [`observatory_philosophy.md`](observatory_philosophy.md)  
  Conceptual framing of the observatory

---

### Research context

- [`abstract.md`](abstract.md)  
  Project abstract

- [`research_log.md`](research_log.md)  
  Running research log

- [`conversation_excerpts.md`](conversation_excerpts.md)  
  Selected excerpts that informed framing and development

---

## Canonical Runtime

The repository now has a single canonical full-pipeline entrypoint:

```bash
bash scripts/run_full_pipeline.sh
```

This executes the orchestrated stage pipeline defined in:

- `src/pam/pipeline/stages/`
- `src/pam/pipeline/runner.py`

Stage order:

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

---

## Canonical Code Layout

The main package structure is:

```text
src/pam/
  engine/
  measurement/
  observables/
  geometry/
  phase/
  topology/
  operators/
  pipeline/
```

Supporting repository roots:

```text
observatory/
  corpora/
  runs/
  derived/
  reports/
  figures/

experiments/
  figures/
  studies/
  toy/
  archive/

outputs/
  active file-first artifact store
```

---

## Documentation Notes

A number of older documents reflect earlier stages of the repository, when the system was centered more strongly on:

- flat experiment scripts
- TUI-first monitoring
- visualization-first workflows
- `outputs/index.csv` as the main architectural interface

Those materials remain useful historically, but the canonical repository architecture is now the layered instrument described in:

- [`architecture.md`](architecture.md)

---

## Summary

The PAM Observatory documentation now supports a repository that is no longer just a script collection or visualization workflow.

It documents a canonical layered instrument for:

- evolving recursive systems
- measuring invariant structure
- extracting geometry and phase
- probing the manifold with operators
- analyzing topological organization
