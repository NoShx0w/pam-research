# Operators

## Overview

The PAM Observatory no longer operates purely as a descriptive system.

With the operator layer, it becomes an **active experimental instrument**.

Operators act on paths through the intrinsic manifold, allowing the observatory to:

- probe structure under constraint
- measure boundary interaction
- compare path outcomes across regimes
- study transition behavior under controlled traversal

This marks the shift from:

> describing geometry → probing geometry

In the canonical PAM Observatory, the operators layer sits downstream of geometry and phase, and upstream of topology.

---

## From Geometry to Operators

Operators do not begin from raw experiment logs alone.e
They begin from the geometric and phase structure already extracted by the observatory:

- intrinsic manifold distances
- manifold embedding
- seam structure
- seam-relative distance
- signed phase organization

So the conceptual flow ie:

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

Operators make the manifold experimentally actionable.

---

## Definition

An operator acts on manifold paths or path endpoints under geometric constraints.

Conceptually:

```math
\theta(t) \xrightarrow{\mathcal{O}} \tilde{\theta}(t)
```

where:

- $\theta(t)$ is a path through the manifold
- $\mathcal{O}$ is an operator or probe procedure
- $\tilde{\theta}(t)$ is the resulting constrained or extracted path

Operators do not redefine the manifold itself.

They expose how trajectories behave **with respect to** the manifold’s intrinsic structure.

---

## Why Operators Matter

Before operators, the observatory could:

- estimate geometry
- extract seam structure
- identify phase organization
- summarize topological organization

With operators, the observatory can also:

- test how paths interact with the seam
- measure regime accessibility
- compare within-phase and cross-phase traversal
- estimate transition behavior under controlled path families
- identify boundary-activated or pre-collapse regimes

This introduces an experimental loop:

1. measure  
2. geometricize  
3. extract phase structure  
4. probe the manifold  
5. analyze outcomes  

---

## Canonical Operator Families

The current observatory includes several operator-level procedures.

### 1. Geodesic extraction

The geometry induces intrinsic path structure through Fisher-distance geodesics.

This supports:

- shortest-path extraction
- geodesic path visualization
- geodesic fan analysis

Interpretation:

- reveals how the manifold connects states intrinsically
- provides the path substrate for operator reasoning
- shows how shortest traversal interacts with phase structure

### 2. Canonical probes

The observatory defines canonical probe endpoint families such as:

- basin-to-basin
- seam-to-positive
- seam-to-negative
- same-phase controls

These expose how qualitatively different path families behave under the same intrinsic geometry.

### 3. Scaled probes

The observatory also scales probe analysis to many sampled endpoint pairs.

This supports:

- family-level comparison
- transition statistics
- seam-crossing rate estimation
- predictive grouping by path exposure variables

This is the main active operator analysis substrate in the current repository.

---

## Canonical Operator: Geodesic Extraction (S)

The first canonical operator family is built around **geodesic extraction**, often denoted \(S\).

### Description

Geodesic extraction uses the intrinsic geometry to trace manifold-respecting paths between selected endpoints.

Its role is to:

- align traversal with intrinsic geometry
- expose shortest-path structure under Fisher-derived costs
- reveal interaction with seams, basins, and transition regions

### Role

$(S)$ is the first active probe of the manifold.

It transforms the observatory from:

- passive mapping

into:

- controlled path experimentation

---

## Operator-Level Phenomena

Applying operators makes several structural behaviors observable.

### Seam graze

A path approaches the seam closely without crossing it.

Interpretation:
- near-boundary interaction
- regime pressure without full transition

### Seam crossing

A path crosses from one phase side to the other.

Interpretation:
- qualitative regime transition
- phase accessibility under intrinsic traversal

### Phase flip

A path’s signed-phase sequence changes sign along the path.

Interpretation:
- stronger evidence of regime transition than mere geometric proximity alone

### Boundary-activated / Lazarus structure

A path encounters high seam-relative pressure, curvature, or Lazarus intensity before transition.

Interpretation:
- metastable or pre-collapse structure
- candidate predictive regime preceding qualitative change

### Collapse-like transition

Some path classes exhibit strong transition signatures or degenerate outcome structure under constraint.

Interpretation:
- boundary-conditioned instability
- transition endpoint class rather than mere noise

---

## Lazarus as an Operator-Side Diagnostic

In the current observatory, Lazarus is analyzed as a boundary-adjacent regime derived from:

- curvature
- seam proximity
- phase centering

Within the operator layer, Lazarus becomes useful because it can be measured **along paths**.

This allows the observatory to study questions like:

- does Lazarus peak before seam contact?
- does Lazarus exposure predict transition within \(k\) steps?
- do high-Lazarus path states cross more often than low-Lazarus path states?

So Lazarus is not itself an operator, but a critical operator-side diagnostic field.

---

## Transition-Rate Analysis

The operator layer also supports explicit transition-rate estimation.

Using scaled probe paths, the observatory can estimate:

- transition probability within a short future window
- lag to next phase flip
- dependence of transition rate on Lazarus exposure
- comparison against baseline predictors such as seam distance or curvature

This is one of the most important ways the observatory moves from structural description to experimentally testable behavior.

---

## Constraint Surfaces

A major operator-layer insight is that the seam is not merely a visual boundary.

Under operator analysis, it behaves like a **constraint surface**.

Paths may:

- approach it
- compress near it
- dwell near it
- cross it
- fail to cross it

This turns static phase structure into experimentally accessible path behavior.

---

## Toward an Operator Algebra

The repository is not yet a full operator algebra in the strict formal sense.

However, the operator layer already introduces the beginnings of one:

- path families can be defined systematically
- operator outcomes can be grouped and compared
- endpoint classes induce recurring behavioral patterns
- transition structure can be studied across probe ensembles

Future work may include:

- operator composition
- equivalence classes over operator outcomes
- operator-induced invariants
- richer response-operator formalisms

This is best understood as a canonical direction of growth, not as a completed subsystem.

---

## Relation to Topology

The operator layer is one of the strongest bridges into topology.

Operators reveal:

- which regime boundaries are traversable
- where seam-relative structure concentrates
- how path outcomes depend on manifold organization
- which regions act as transition corridors versus stable interiors

In this sense, topology is not only inferred from static fields.

It is also revealed by how paths behave under structured probing.

---

## Canonical Implementation Mapping

### Canonical modules

| Concept | Canonical module |
|--------|------|
| Geodesic extraction | `src/pam/operators/geodesic_extraction.py` |
| Canonical probes | `src/pam/operators/probes.py` |
| Scaled probes | `src/pam/operators/scaled_probes.py` |
| Lazarus regime | `src/pam/operators/lazarus.py` |
| Transition-rate analysis | `src/pam/operators/transition_rate.py` |

### Pipeline orchestration

| Concept | Pipeline module |
|--------|------|
| Operators stage | `src/pam/pipeline/stages/operators.py` |
| Full pipeline runner | `src/pam/pipeline/runner.py` |
| Canonical shell entrypoint | `scripts/run_full_pipeline.sh` |

### Compatibility wrappers

| Concept | Wrapper script |
|--------|------|
| Geodesic path wrapper | `experiments/fim_geodesic.py` |
| Geodesic fan wrapper | `experiments/fim_geodesic_fan.py` |
| Canonical operator S | `experiments/fim_operator_S.py` |
| Canonical probes | `experiments/fim_operator_probes.py` |
| Scaled probes | `experiments/fim_operator_probe_scale.py` |
| Lazarus regime | `experiments/fim_lazarus.py` |
| Transition-rate analysis | `experiments/fim_transition_rate.py` |

---

## Active Operator Outputs

The operators layer writes active artifacts under directories such as:

- `outputs/fim_ops/`
- `outputs/fim_ops_scaled/`
- `outputs/fim_lazarus/`
- `outputs/fim_transition_rate/`

These outputs capture how manifold paths behave under controlled probing and constraint.

---

## Summary

Operators turn the PAM Observatory into an active instrument.

In the canonical repository, the operators layer provides:

- geodesic path extraction
- canonical and scaled probes
- path-level Lazarus diagnostics
- transition-rate estimation
- seam-relative experimental traversal

In short:

> Geometry tells you what exists.  
> Phase tells you where regimes divide.  
> Operators tell you how the manifold behaves under probing.
