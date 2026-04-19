# Canonical Family Layer — Current Status

## Purpose

This document records the current status of the canonical family/gateway layer in the PAM Observatory repository.

It is meant to answer, plainly:

- what now exists,
- what is implemented,
- what is validated,
- what remains provisional,
- and why the layer currently lives in `scripts/canonical/` rather than `src/pam/`.

This is a status note, not a theory paper and not a full implementation spec.

## What now exists

The repository now contains a first end-to-end canonical family/gateway layer.

This layer includes:

### 1. Canonical family-level summaries

Implemented outputs:

- `outputs/canonical/gateway_comparison.csv`
- `outputs/canonical/temporal_depth_summary.csv`
- `outputs/canonical/memory_compression_summary.csv`

These tables consolidate the family-level gateway, temporal-depth, and compression picture into stable canonical artifacts.

### 2. Canonical event-level normalization

Implemented output:

- `outputs/canonical/event_family_features.csv`

This table is the first canonical event-level row surface for the family/gateway layer.

It normalizes heterogeneous gateway-study rows into one assignment-ready event table.

### 3. Canonical event-level assignment

Implemented output:

- `outputs/canonical/event_family_assignment.csv`

This table formalizes event-family assignment from the normalized event layer.

The current assignment method is deliberately conservative and should be understood as an assignment surface with diagnostic confidence, not as a de novo learned classifier.

### 4. Assignment-derived family aggregation

Implemented output:

- `outputs/canonical/family_aggregation.csv`

This table rolls assigned events back up to one row per canonical family and functions as both:

- a family summary,
- and an end-to-end coherence check for the canonical layer.

### 5. Trajectory rollup scaffold

Implemented output:

- `outputs/canonical/trajectory_family_rollup.csv`

At present this table is intentionally empty, because the current canonical event layer does not yet contain justified non-null `trajectory_id` linkage.

This is not a bug. It is an explicit statement about current layer maturity.

### 6. Integrated validation

Implemented outputs:

- `outputs/canonical/validation/family_layer_validation.txt`
- `outputs/canonical/validation/family_layer_warnings.csv`

The canonical family layer is now validated as an integrated system rather than only table-by-table.

## Current implementation home

The canonical family/gateway layer currently lives in:

- `scripts/canonical/`

This is intentional.

The layer is now real enough to be official and useful, but it is still best understood as a downstream canonicalization and validation surface built over evolving study outputs.

For that reason, it has not yet been promoted into:

- `src/pam/`

## What is implemented

The following implementation pieces now exist:

### Canonicalization scripts

- `build_gateway_comparison.py`
- `build_temporal_depth_summary.py`
- `build_memory_compression_summary.py`
- `build_event_family_features.py`
- `build_event_family_assignment.py`
- `build_family_aggregation.py`
- `build_trajectory_family_rollup.py`

### Integrated validation

- `validate_family_layer.py`

Together, these scripts define the current downstream canonical family layer.

## What is validated

The current integrated validator checks:

- canonical file existence
- required columns
- unique keys where expected
- canonical family labels
- canonical event types where expected
- event-feature / assignment join integrity
- family aggregation count consistency
- version-field consistency across canonical layers
- scientific coherence checks, including:
  - forgetting ordering
  - temporal-depth ordering
  - strongest local gateway signal for stable seam corridor
  - event-type share consistency
- provenance sanity for mapped enrichment fields

At the current checkpoint, the integrated family-layer validator reports:

- zero errors
- zero warnings

This does **not** mean the science is finished. It means the implemented canonical layer is internally coherent on its own current terms.

## What remains provisional

The canonical family/gateway layer is implemented and validated, but it is still provisional in several important ways.

### 1. Some important values are family-mapped rather than native

Several event-level enrichment fields are currently attached from family-level canonical summaries rather than computed natively per event.

This is explicit and intentional in the current design, but it means the event layer is not yet fully local in provenance.

### 2. Event assignment is not a de novo classifier

The current event assignment layer formalizes observed normalized family labels and adds diagnostic confidence.

It should not yet be described as an independently learned or first-principles classifier.

### 3. Trajectory-native recomposition is not yet available

The trajectory rollup exists only as a scaffold.

At present, the canonical event layer does not yet carry justified trajectory identity, so the rollup is empty by design.

### 4. The layer has not yet been pressure-tested against broader multi-corpus use

The current canonical layer was built in a repository state still dominated by one corpus line of analysis.

This makes the layer useful and real, but still subject to future multi-corpus revision.

## Why the layer remains in `scripts/canonical/`

The reason is architectural honesty.

The current canonical layer is:

- scientifically meaningful,
- reproducible,
- validated,
- and useful,

but it is still best modeled as:

- downstream consolidation,
- cross-study normalization,
- assignment formalization,
- and validation.

Promotion into `src/pam/` should wait until at least the following are more stable:

- schema contracts
- provenance contracts
- trajectory linkage
- assignment logic maturity
- multi-corpus pressure testing

Until then, `scripts/canonical/` is the correct home.

## Conceptual role in observatory mechanics

In the current observatory mechanics, the canonical family layer sits **after** the main study-producing runtime.

Conceptually:

1. trajectories are computed upstream
2. observatory/study scripts produce family-relevant artifacts
3. the canonical layer consolidates those artifacts
4. validation checks the canonical layer as an integrated system

So the canonical family layer is not the source of the scientific observables.

It is the place where those observables are:

- stabilized,
- cross-walked,
- assigned,
- aggregated,
- and made legible as one coherent layer.

## Current interpretation of the trajectory rollup

The empty `trajectory_family_rollup.csv` is not a failed artifact.

It is a truthful one.

It means that the current canonical event layer is:

- event-native,
- family-native,
- but not yet trajectory-native.

That is an important current fact about the system.

## Recommended current reading of the layer

The safest current reading is:

The repository now has a credible canonical family/gateway layer that is implemented and validated as a downstream observatory surface, but still provisional in provenance depth, assignment maturity, and trajectory linkage.

That is already a major step forward.

## Near-term next directions

The most likely next directions are:

- documentation catch-up across root/docs indices
- later recovery of justified trajectory linkage
- refinement of confidence / ambiguity logic
- eventual promotion of stabilized pieces into `src/pam/`, if warranted
- future pressure testing under additional corpora

## Status summary

The canonical family/gateway layer is now:

- specified
- implemented
- artifactized
- validated
- still architecturally downstream
- still scientifically provisional in some important respects

## Distributed recoverability

Later recoverability analysis shows that route-family identity is not strongly pointwise recoverable in the canonical event-family dataset.

Using a strict feature ladder:

- pointwise local observables yield weak recoverability
- local neighborhood support adds only modest improvement
- short route context improves recoverability further
- but the overall result remains far from clean recovery

The family-level picture is asymmetric:

- stable_seam_corridor is relatively locally legible
- branch_exit improves under short-context enrichment
- reorganization_heavy remains poorly recoverable even after short-context enrichment

The canonical reading is therefore:

route-family identity is better understood as a distributed recoverable object rather than a simple local state label.

This result strengthens the broader family-layer interpretation:

- corridor is closest to a local seam-boundary law
- branch-exit is only partially local
- reorganization-heavy is the least reducible to local observables and remains the clearest path-context-dependent family
