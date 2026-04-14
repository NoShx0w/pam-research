# Canonicalization Implementation Plan

## Purpose

Define the implementation approach for the first canonicalization pass of the family/gateway layer.

This plan turns the provisional project-level specifications into an incremental build strategy that is:

- transparent in a public research repository,
- conservative about provenance and uncertainty,
- compatible with existing study outputs,
- and efficient enough to support iterative development.

This plan does **not** assume upstream rewrites. It treats current study outputs as source artifacts and builds a downstream canonical layer on top of them.

## Status

This plan is implementation-facing and provisional.

It is intended to:

- guide the first canonicalization PRs,
- define build order,
- identify the smallest useful milestones,
- and reduce the risk of mixed semantics or premature refactors.

It should be updated only when the canonical layer or its staging assumptions materially change.

## Guiding principles

Implementation should follow these principles:

1. **Consolidate first, refactor later**
   - do not rewrite upstream study scripts before the canonical layer is stable

2. **Family-level summaries before row-level normalization**
   - stabilize gateway, temporal-depth, and compression summaries first

3. **Event-level row unit before trajectory-level rollup**
   - the observed artifact surface is event-centered
   - trajectory rollups are optional later derivatives

4. **Prefer null over invented precision**
   - mapped or inherited values must never masquerade as native measurements

5. **Preserve provenance explicitly**
   - source semantics must remain visible after normalization

6. **Validate integrity and scientific coherence separately**
   - both kinds of validation are required

## Implementation stages

## Stage 1 — Freeze the contracts

Before implementation grows, freeze the project-level specifications that define the target layer:

- `canonical_family_gateway_spec.md`
- `canonical_event_family_classification_spec.md`
- `canonical_event_normalization_contract.md`

### Goal

Prevent redesign churn and ensure that all downstream scripts target the same canonical contracts.

### Deliverable

- stable docs in `docs/05_project/`

## Stage 2 — Build family-level canonical summaries

The family-level summary layer should be built before event normalization, because several row-level fields will initially be family-mapped.

### 2.1 Build `gateway_comparison`

#### Primary sources

- OBS-038 family-specific gateway summaries
- supporting gateway artifacts from OBS-034 through OBS-039 as needed

#### Goal

Produce one canonical family-level gateway table with:

- gateway crossing metrics
- local-vs-context predictive quantities
- routing-related family summaries
- version and provenance fields

#### Output

- `gateway_comparison`

### 2.2 Build `temporal_depth_summary`

#### Primary sources

- OBS-040 variable-horizon family summaries
- OBS-040b stress-test summaries
- OBS-042 family temporal-regime synthesis

#### Goal

Produce one canonical family-level temporal-depth table with:

- best horizon
- saturation behavior
- regime label
- predictive locus
- version and provenance fields

#### Output

- `temporal_depth_summary`

### 2.3 Build `memory_compression_summary`

#### Primary sources

- OBS-041 forgetting-node and compression summaries
- supporting OBS-040b stress-test compression context where needed

#### Goal

Produce one canonical family-level compression table with:

- forgetting share
- mean gain over suffix
- dominant structural compression state
- compression regime label
- padding contamination handling
- version and provenance fields

#### Output

- `memory_compression_summary`

### Stage 2 success condition

At the end of Stage 2, the repository should have a usable canonical family-level summary layer even before event normalization exists.

## Stage 3 — Define normalization mappings

Before building the event table, define explicit source-to-canonical mappings.

### Required mapping categories

- source file → canonical source row type
- source column names → canonical column names
- source event labels → canonical `event_type`
- source family labels → canonical `route_class` only where already present
- default provenance class by source and field

### Why this stage exists

Without explicit mapping rules, the event consolidation script will become difficult to audit and easy to silently over-merge.

### Deliverable

A lightweight normalization configuration implemented either as:

- code-level dictionaries/constants,
- or a small config module.

## Stage 4 — Build `event_family_features`

This is the primary row-level canonical table.

### Primary event sources

Use these as primary event-like sources:

- OBS-035c instance-level gateway rows
- OBS-036 refined gateway rows
- OBS-037 history-aware gateway rows
- OBS-037b pre-second-step gateway rows
- OBS-039 family-specific path-context rows

### Supporting source

Use as supporting linkage or seam-context source:

- OBS-029 step/path rows

### Goal

Produce one normalized event table with:

- deterministic `event_id`
- normalized `event_type`
- canonical `route_class`
- explicit source row type
- local/history/context features
- provenance fields
- conservative family-level enrichments where needed

### Output

- `event_family_features`

### Important implementation rule

Do not require every source to populate every field. Sparse-but-explicit rows are acceptable and preferred over artificial completeness.

## Stage 5 — Build `event_family_assignment`

Once normalized event rows exist, build the first assignment surface.

### Goal

Assign each normalized event to one canonical family using a conservative initial classifier.

### First implementation style

Use a transparent rule-based assignment first.

This is preferred because it is:

- easier to review,
- easier to debug,
- better aligned with current scientific maturity,
- and less likely to overstate precision.

### Output

- `event_family_assignment`

### Required assignment fields

- assigned family
- confidence
- ambiguity flag
- assignment version
- source feature version

## Stage 6 — Build `family_aggregation`

Once event assignments exist, aggregate them to family-level summaries.

### Goal

Create a canonical family reporting surface derived from event-level assignments and features.

### Output

- `family_aggregation`

### Role

This table is downstream of event normalization and assignment. It should not be the first canonicalization target.

## Stage 7 — Add validation

Validation should be implemented as its own stage and not treated as optional.

### 7.1 Integrity validation

Check:

- required columns present
- unique primary keys
- valid canonical family labels
- valid canonical event types
- no orphan joins
- provenance/version fields present and consistent

### 7.2 Scientific coherence validation

Check:

- `branch_exit` aggregates to highest `escape_internal_share`
- `stable_seam_corridor` aggregates to strongest local gateway signal
- `stable_seam_corridor` aggregates to strongest core-to-escape boundary identity
- `reorganization_heavy` aggregates to strongest path-context gain
- temporal ordering matches the stabilized picture
- compression ordering matches the stabilized picture

### Outputs

- human-readable validation report
- optional machine-readable warnings table

## Stage 8 — Optional trajectory rollup

Only after event-level normalization and assignment are stable should trajectory rollups be added.

### Goal

Provide optional one-row-per-trajectory summaries derived from event assignments.

### Output

- `trajectory_family_rollup`

### Important caution

Trajectory rollup is optional and downstream. It should not drive the first canonicalization pass.

## Recommended script set

The initial script set should remain small and stage-aligned.

### Family-level scripts

- `build_gateway_comparison.py`
- `build_temporal_depth_summary.py`
- `build_memory_compression_summary.py`

### Event-level scripts

- `build_event_family_features.py`
- `build_event_family_assignment.py`

### Aggregation and validation

- `build_family_aggregation.py`
- `validate_family_layer.py`

### Optional orchestration later

- `build_canonical_family_layer.py`

## Suggested module/helpers structure

A lightweight internal helper layer is recommended:

- `common.py`
- `schemas.py`
- `normalize.py`
- `provenance.py`

### `common.py`

Responsibilities:
- load CSVs
- assert required columns
- standardize writing and logging

### `schemas.py`

Responsibilities:
- canonical route-class constants
- canonical event-type constants
- expected output columns
- version identifiers

### `normalize.py`

Responsibilities:
- source row type mapping
- event-type normalization
- column rename maps
- structural compression-state cleanup

### `provenance.py`

Responsibilities:
- provenance classification helpers
- projection flag rules
- provisional flag policies

## Recommended build order

The build order should be:

1. `gateway_comparison`
2. `temporal_depth_summary`
3. `memory_compression_summary`
4. `event_family_features`
5. `event_family_assignment`
6. `family_aggregation`
7. validation

This order is preferred because row-level event normalization will initially depend on family-level enrichments.

## Earliest useful milestone

The first scientifically useful milestone is:

- `gateway_comparison`
- `temporal_depth_summary`
- `memory_compression_summary`
- `event_family_features`
- `event_family_assignment`
- validation report

This milestone is sufficient to expose a real canonical event-level layer without requiring trajectory rollups or upstream rewrites.

## Major implementation risks

The main risks are:

- mixing provenance classes without explicit flags
- silently treating family-level enrichments as event-native values
- over-joining event rows across studies without justified keys
- overcommitting to exact thresholds before the canonical surface is stable
- allowing older `path_family` vocabulary to leak into canonical `route_class`
- carrying padding-contaminated compression states forward as if they were structural
- using unstable older horizon readings instead of the stress-tested interpretation

## What not to do yet

The first canonicalization pass should avoid:

- rewriting upstream OBS scripts
- forcing trajectory-level rollup before event normalization is stable
- using learned assignment models before the rule-based surface is debugged
- silently coercing old path-family labels into canonical route-class labels
- pretending every desired feature already exists natively at event level

## Immediate next coding step

The recommended first implementation step is:

- build `gateway_comparison`

Reason:

- it is strongly supported by existing artifacts,
- comparatively small,
- easy to validate,
- and establishes naming and provenance discipline before event normalization begins.

After that, build:

- `temporal_depth_summary`
- `memory_compression_summary`

Only then move to:

- `event_family_features`

## Relationship to companion specs

This implementation plan assumes the contracts defined in:

- `canonical_family_gateway_spec.md`
- `canonical_event_family_classification_spec.md`
- `canonical_event_normalization_contract.md`

Together, these documents define:

- the scientific target,
- the canonical row unit,
- the normalization and provenance rules,
- and the sequence in which the canonical layer should be built.
