# PAM Identity Transport / Holonomy — Canonical Stabilization Plan

## Purpose

Promote identity transport and holonomy from successful study outputs into canonical topology-layer infrastructure.

The objective is not to invent new formalism yet.  
It is to stabilize what has already been operationally defined, measured, and validated:

- identity distance
- identity field
- identity metric
- identity transport
- identity spin
- loop-based holonomy residual

This is the architectural step that turns identity transport from a research branch into a stable subsystem of the observatory.

---

## Core principle

> Stabilize before expanding.

At this point, the science is ahead of the architecture.

What has already been shown:

- identity magnitude is metric-adjacent
- identity spin is not strongly explained by local metric structure
- absolute loop residual aligns strongly with local spin magnitude
- identity spin is therefore best treated as an operational measure of connection curvature

The next step is to make this transport layer canonical.

---

## Canonical scope

### Keep canonical under `src/pam/topology/`

The following modules now belong to the canonical topology layer:

- `identity.py`
- `identity_field.py`
- `identity_metric.py`
- `identity_transport.py`

These together define the stabilized identity stack:

1. identity distance
2. identity field
3. identity metric
4. identity transport / holonomy
5. identity obstruction

### Keep experimental under `experiments/studies/`

These may remain exploratory / interpretive for now:

- alignment studies
- singularity overlay studies
- diagonal vs full metric comparisons
- extended figure builders

They are useful for interpretation, but they are not the core layer production path.

---

## Stabilized identity stack

### 1. Metric layer

Canonical objects:

- `identity_distance`
- local identity metric estimators
- identity magnitude

Role:

- local structural distinguishability
- local structural sensitivity
- metric-adjacent geometry of identity change

---

### 2. Transport layer

Canonical objects:

- local path composition
- loop residual / holonomy construction

Role:

- path-based structural identity propagation
- local transport inconsistency measurement

---

### 3. Obstruction layer

Canonical objects:

- `identity_spin`
- `holonomy_residual`
- `abs_holonomy_residual`

Role:

- local obstruction to path-independent transport
- operational connection curvature signal

---

## Canonical artifact families to stabilize

### Identity field artifacts

Keep stable under:

- `outputs/fim_identity/identity_field_nodes.csv`
- `outputs/fim_identity/identity_field_edges.csv`
- `outputs/fim_identity/identity_spin.csv`
- `outputs/fim_identity/identity_magnitude.png`
- `outputs/fim_identity/identity_field_quiver.png`
- `outputs/fim_identity/identity_spin.png`

### Identity transport / holonomy artifacts

Keep stable under:

- `outputs/fim_identity_holonomy/identity_holonomy_cells.csv`
- `outputs/fim_identity_holonomy/identity_holonomy_alignment.csv`
- `outputs/fim_identity_holonomy/identity_holonomy_on_grid.png`
- `outputs/fim_identity_holonomy/identity_abs_holonomy_on_grid.png`

These should now be treated as first-class topology outputs rather than disposable study artifacts.

---

## Required architectural change

## 1. Integrate identity transport into the topology stage

### Target file

- `src/pam/pipeline/stages/topology.py`

### Desired behavior

The topology stage should now be able to produce, in one canonical pass:

- criticality / existing topology summaries
- identity proxy graphs
- identity field artifacts
- identity transport / holonomy artifacts

### Suggested stage order

1. criticality / existing topology outputs
2. build local identity proxy graphs
3. compute identity field
4. compute identity transport / holonomy
5. write node / edge / cell artifact families

### Acceptance

- identity transport no longer depends on ad hoc manual sequencing
- topology stage becomes the canonical place where transport-layer observables are produced

---

## 2. Keep file-first orchestration

Identity transport must remain artifact-driven.

It should consume:
- node / patch metadata
- existing identity graphs
- grid selection / lattice relationships

It should write:
- node-level and cell-level outputs
- alignment summaries
- canonical plots

No hidden in-memory-only dependency chain should be introduced.

---

## 3. Add one consolidated comparison figure

### New study script

- `experiments/studies/fim_identity_transport_panel.py`

### Purpose

Render the canonical visual confirmation of the transport layer.

### Figure content

A side-by-side pair:

- `identity_spin_on_grid`
- `identity_abs_holonomy_on_grid`

### Why this figure

This is the clearest current visual confirmation that:

- spin is local obstruction
- holonomy is loop-level confirmation
- both belong to the same transport / obstruction layer

This figure should become the canonical explanatory panel for the transport result.

---

## Suggested implementation sequence

### PR 1 — topology-stage integration

Integrate identity field and holonomy production into:

- `src/pam/pipeline/stages/topology.py`

Acceptance:
- topology stage can produce stabilized identity transport artifacts

---

### PR 2 — transport comparison panel

Add:

- `experiments/studies/fim_identity_transport_panel.py`

Acceptance:
- side-by-side spin / holonomy figure is reproducible from canonical outputs

---

### PR 3 — docs and observatory stabilization

Update:

- `docs/research_log.md` with OBS-008
- topology / identity docs as needed
- optional pre-push validation if holonomy outputs become mandatory

Acceptance:
- transport layer is reflected in docs and artifact expectations

---

## Acceptance criteria for stabilization

Identity transport / holonomy is considered stabilized when:

- `identity_transport.py` remains canonical under `src/pam/topology/`
- topology stage produces holonomy outputs through a stable public path
- `identity_holonomy_cells.csv` is reproducible from canonical stage execution
- `identity_holonomy_alignment.csv` is reproducible
- spin / holonomy comparison figure exists
- docs explicitly recognize:
  - metric layer
  - transport layer
  - obstruction layer

---

## What should NOT happen yet

Do **not** immediately expand into:

- full discrete connection operators
- Christoffel-style formal connection machinery
- gauge-theoretic notation in code paths
- overly rich transport algebra
- new metric families beyond what has already been tested

The current result is already strong.  
The right move now is stabilization, not conceptual inflation.

---

## Next scientific refinement after stabilization

Only after the transport layer is canonical:

### 1. Oriented loop conventions
Define loop orientation explicitly and compare signed holonomy against signed spin.

### 2. Signed transport refinement
Move from absolute loop inconsistency to orientation-sensitive local transport analysis.

### 3. Richer discrete transport operators
Only if the stabilized scalar-path construction remains robust.

This is the correct OBS-009 direction, but it is downstream of stabilization.

---

## Summary

The next step is architectural, not exploratory.

Promote identity transport and holonomy from study outputs into canonical topology-layer infrastructure by:

- integrating transport production into the topology stage
- stabilizing holonomy artifacts
- preserving a canonical spin-vs-holonomy comparison panel
- delaying richer connection machinery until the present layer is fully stabilized

In short:

> Identity transport is now strong enough to become part of the instrument.
