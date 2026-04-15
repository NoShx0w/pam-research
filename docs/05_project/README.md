# Project

This section contains project-level documentation for the PAM Observatory.

It is the home for:

- stabilization plans
- canonicalization specs
- implementation roadmaps
- milestone and release notes
- repository maintenance procedures
- canonical layer status and architecture notes

## Current role

The repository has:

- a canonical implemented runtime for the downstream observatory pipeline
- a downstream canonical family/gateway layer implemented in `scripts/canonical/`
- validated canonical artifacts under `outputs/canonical/`

Primary runtime entrypoints:

- `scripts/run_full_pipeline.sh`
- `src/pam/pipeline/runner.py`

Primary architecture reference:

- [`../architecture.md`](../architecture.md)

## Canonical family layer

The repository now includes a first end-to-end canonical family/gateway layer.

Implemented canonical artifacts currently include:

- family-level summary tables
  - `gateway_comparison.csv`
  - `temporal_depth_summary.csv`
  - `memory_compression_summary.csv`
- event-level normalized and assigned tables
  - `event_family_features.csv`
  - `event_family_assignment.csv`
- assignment-derived family rollup
  - `family_aggregation.csv`
- trajectory rollup scaffold
  - `trajectory_family_rollup.csv`
- integrated validation outputs
  - `family_layer_validation.txt`
  - `family_layer_warnings.csv`

Current implementation home:

- `scripts/canonical/`

Current artifact root:

- `outputs/canonical/`

## Canonicalization and stabilization

The documents in this section define **provisional canonical contracts** for research structures that are scientifically mature enough to stabilize, but not yet ready to be treated as fully internalized runtime primitives.

These documents are intended to:

- make evolving observatory structures explicit
- provide stable implementation targets
- distinguish canonical direction from transitional study outputs
- improve public-facing transparency as the repository catches up to the science
- mark what is implemented, what is validated, and what remains provisional

## Current architectural status

The canonical family/gateway layer is now:

- specified
- implemented
- artifactized
- validated

But it remains architecturally downstream for now.

That is why it currently lives in:

- `scripts/canonical/`

rather than being promoted into:

- `src/pam/`

Promotion should wait until:

- schemas stabilize further
- provenance contracts settle
- event/trajectory linkage matures
- and multi-corpus pressure testing is further along

## Documents

Current project-level documents include:

- [`OBS-015.md`](./OBS-015.md)
- [`pam_identity_transport_holonomy_stabilization_plan.md`](./pam_identity_transport_holonomy_stabilization_plan.md)

Canonical family-layer documents include:

- `canonical_family_gateway_spec.md`
- `canonical_event_family_classification_spec.md`
- `canonical_event_normalization_contract.md`
- `canonicalization_implementation_plan.md`

A current status note for the implemented canonical layer should also live in this section.

## Guidance

For current runtime and high-level structure, prefer:

- the repository root documentation
- [`../architecture.md`](../architecture.md)

For project-level stabilization, canonicalization, implementation planning, and canonical-layer status, prefer the documents in this section.