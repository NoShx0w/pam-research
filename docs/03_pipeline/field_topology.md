# Field Topology

## Overview

Field topology is the layer where the PAM Observatory moves from manifold description to structural organization.

If geometry answers:

> how are states arranged?

and phase answers:

> where do regime boundaries emerge?

then topology answers:

> how is the manifold-organized field structurally put together?

In the canonical PAM Observatory, topology studies organization through:

- field alignment
- gradient alignment
- critical structure
- seam-relative organization
- phase-selection summaries

This is where the system becomes relational rather than merely geometric.

---

## From Geometry and Phase to Topology

The topology layer begins downstream of geometry and phase.

Its main inputs are:

- manifold coordinates
- signed phase structure
- seam-relative distance
- Lazarus-related fields
- operator-path outcomes

So the conceptual flow is:

```text
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

Topology does not replace geometry or phase. It organizes their relational consequences.

---

## Field View

A useful conceptual view is to think in terms of fields over the manifold.

Examples include:

- signed phase
- seam distance
- Lazarus intensity
- response-alignment structure

These fields allow the observatory to ask not only where points lie, but how structure is distributed and how local changes align with global organization.

In this sense, the manifold supports a field description even when the implementation remains partly discrete and artifact-driven.

---

## Gradient Structure

A central topological question is how different field directions align locally.

Examples:

- phase-gradient alignment with Lazarus-gradient structure
- seam-relative organization of high-response regions
- local directional structure near transition zones

This is why the canonical topology layer includes field and gradient alignment analysis.

Interpretation:

- aligned gradients indicate co-organized structure
- orthogonal or weakly aligned gradients indicate partially independent axes of organization
- strong boundary-localized gradients often mark transition structure

---

## Critical Structure

Topology also studies where organization becomes concentrated or unstable.

In the current observatory, this is expressed through criticality summaries rather than only through a classical continuous critical-point solver.

Examples include:

- high-curvature regions
- seam-adjacent concentration
- criticality scores combining determinant, curvature, and seam distance
- organizational concentration near transition zones

Interpretation:

- these structures indicate where the manifold reorganizes most strongly
- they function as topological anchors for phase transition analysis

---

## Seam as Structural Organizer

The seam is not merely a boundary curve.

Topologically, it acts as a structural organizer.

Observations in the observatory include:

- alignment structure changes near the seam
- Lazarus intensity concentrates near seam-relative regions
- operator-path outcomes are strongly organized by seam distance
- critical regions often cluster near seam-adjacent structure

So the seam is central not only to phase assignment, but to the relational organization of the manifold.

---

## Field Alignment

One topological task is to compare different fields over the manifold.

For example:

- Lazarus vs seam distance
- phase vs Lazarus temporal ordering
- boundary contact vs later phase transitions

This produces a structural picture of how different observables and derived fields cohere.

It lets the observatory distinguish:

- mere correlation
- local directional alignment
- boundary-driven structural coupling

---

## Organizational Topology

Topology in the current repository also includes organizational summaries over trajectories and outcomes.

This includes:

- initial-condition outcome maps
- dominant outcome regimes
- phase-selection structure
- operator-conditioned organization

In this form, topology becomes a ledger of how the manifold organizes possible behavior.

This is the point where the observatory moves from:

- where states are
to
- how possible behaviors are structurally arranged

---

## Topological Ledger

A useful summary object is the **topological ledger**.

This ledger may include:

- critical structure
- seam relations
- alignment summaries
- outcome organization
- basin-like or transition-like regions
- path-conditioned structural differences

This ledger is not just a visualization artifact.

It is the structural record of how the field is organized.

---

## Relational Identity

A key principle remains:

> **Topology is the relational identity of the field.**

Unlike point-wise measurements, topology encodes:

- how structures align
- how transition organization is arranged
- how boundaries constrain outcomes
- how different fields relate across the manifold

Two systems are structurally close when these relational patterns are preserved, not merely when their pictures look similar.

---

## Invariance

Topological interpretation is more stable than raw representation.

It is comparatively robust to:

- embedding distortions
- moderate plotting differences
- superficial coordinate changes

provided the underlying relational organization is preserved.

That is why topology is the strongest candidate for structural identity in the observatory.

---

## What This Enables

With topology extracted, the observatory can:

- identify structurally important transition zones
- compare manifold organization across runs
- relate phase structure to operator outcomes
- identify boundary-conditioned behavior
- summarize initial-condition selection structure
- move from local measurements to global organization

---

## Canonical Implementation Mapping

### Canonical modules

| Concept | Canonical module |
|--------|------|
| Field alignment | `src/pam/topology/field.py` |
| Gradient alignment | `src/pam/topology/flow.py` |
| Critical structure | `src/pam/topology/critical_points.py` |
| Organizational topology | `src/pam/topology/organization.py` |

### Pipeline orchestration

| Concept | Pipeline module |
|--------|------|
| Topology stage | `src/pam/pipeline/stages/topology.py` |
| Full pipeline runner | `src/pam/pipeline/runner.py` |
| Canonical shell entrypoint | `scripts/run_full_pipeline.sh` |

### Compatibility wrappers

| Concept | Wrapper script |
|--------|------|
| Field alignment | `experiments/fim_field_alignment.py` |
| Gradient alignment | `experiments/fim_gradient_alignment.py` |
| Critical structure | `experiments/fim_critical_points.py` |
| Organizational topology | `experiments/fim_phase_selection_diagram.py` |

---

## Active Topology Outputs

The topology layer writes active artifacts under directories such as:

- `outputs/fim_field_alignment/`
- `outputs/fim_gradient_alignment/`
- `outputs/fim_critical/`
- `outputs/fim_initial_conditions/`

These outputs summarize how the manifold is structurally organized beyond raw geometric position.

---

## Identity Geometry Stack

The identity layer now resolves three geometric levels:

1. **Metric layer**
   - identity distance
   - local identity metric
   - identity magnitude

2. **Transport layer**
   - local path composition
   - loop-based holonomy residual

3. **Obstruction layer**
   - identity spin as a local obstruction signal
   - loop-level confirmation via holonomy magnitude

This upgrades identity from a derived field into a structured geometric subsystem of the observatory.

### Identity transport artifacts

Canonical outputs include:

- `outputs/fim_identity/identity_field_nodes.csv`
- `outputs/fim_identity/identity_field_edges.csv`
- `outputs/fim_identity/identity_spin.csv`
- `outputs/fim_identity_holonomy/identity_holonomy_cells.csv`
- `outputs/fim_identity_holonomy/identity_holonomy_alignment.csv`
- `outputs/fim_identity_holonomy/identity_transport_panel.png`

---

## From Visualization to Structure

The critical shift is:

Before:
- embeddings
- colors
- seam plots
- local diagnostics

After:
- alignment structure
- critical organization
- outcome-conditioned topology
- relational summaries of the field

This is the move from seeing the manifold to understanding its organization.

---

## One-Line Summary

> Topology is the part of manifold organization that remains when representation changes but structural relations are preserved.

---

## Closing

Field topology is the layer where the PAM Observatory becomes structurally comparative.

It extracts:

- organization from geometry
- relation from fields
- transition structure from phase
- behavioral arrangement from operator outcomes

and turns the manifold into a space of structural identity.
