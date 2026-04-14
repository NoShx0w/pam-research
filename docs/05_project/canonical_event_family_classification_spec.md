# Provisional Canonical Spec — Event-Level Family Classification

## Purpose

Define a provisional canonical event-level family-classification layer for the PAM Observatory.

This specification uses **gateway-relevant events** as the primary row unit rather than whole trajectories. It exists to align the canonical layer with the actual OBS artifact surface, where gateway behavior is already operationalized at event, composition, and context-row level.

The goals are to:

- classify family structure at the level where gateway behavior is observed,
- preserve compatibility with existing event- and family-oriented study outputs,
- support later rollup to trajectories and family-level summaries,
- avoid forcing early trajectory-level coercion where event structure is already the cleaner substrate.

This is a consolidation interface, not a redesign of upstream analyses.

## Status

This document is **provisional but normative** for the current canonicalization effort.

That means:

- it defines the current target row unit and table structure,
- it allows explicit uncertainty where the artifact surface is still mixed,
- it should be treated as the implementation target for the first row-level canonical layer.

It does **not** claim that every required field already exists natively in current outputs.

## Canonical row unit

The canonical primary row unit is:

- **one gateway-relevant event**

A gateway-relevant event is any event or composition instance that can be labeled or summarized in terms of:

- `core_internal`
- `core_to_escape`

Related continuation behaviors may also appear in supporting tables:

- `escape_internal`
- `escape_to_core`

### Why event-level is canonical first

This row unit is the best current fit for the observed artifact surface because:

- gateway behavior is already studied as event types,
- local vs path-context prediction is evaluated on boundary events,
- family differences are sharpest at the event level,
- current OBS outputs already expose `route_class` on event-like rows.

### Uncertainty

The exact final event extractor is not yet fully frozen. So “gateway-relevant event” remains a provisional implementation contract rather than a theorem-level object.

## Canonical tables

The event-level canonical layer should consist of these tables:

1. `event_family_features`
   - primary canonical row table
   - one row per gateway-relevant event
   - stores classifier-relevant event features and provenance

2. `event_family_assignment`
   - one row per normalized event
   - stores assigned family, confidence, ambiguity, and assignment provenance

3. `family_aggregation`
   - one row per family
   - aggregates assigned event rows and event-level features

4. `gateway_comparison`
   - one row per family
   - stores family-level gateway and routing quantities

5. `temporal_depth_summary`
   - one row per family, or family × encoding if needed
   - stores family-specific temporal-depth summaries

6. `memory_compression_summary`
   - one row per family
   - stores family-specific forgetting and compression summaries

### Optional later companion table

- `trajectory_family_rollup`
  - one row per trajectory
  - derived from event-level assignments
  - not required for the minimal event-level layer

## Minimal event-level classifier contract

The smallest plausible set of event-level quantities needed to support canonical family assignment is:

- `event_id`
- `trajectory_id`
- `event_type`
- `seam_residency`
- `escape_internal_share`
- `core_to_escape_share`
- `local_gateway_strength`
- `path_context_gain`
- `effective_temporal_depth`
- `forgetting_share`

Strongly recommended:

- `dominant_compression_state`

### Interpretation of these fields at event level

- `event_id`
  - required in the canonical layer
  - may be synthesized deterministically during consolidation

- `trajectory_id`
  - required if recoverable from supporting sources
  - otherwise may remain null until stable linkage exists

- `event_type`
  - normalized event label
  - first minimal classifier should prioritize:
    - `core_internal`
    - `core_to_escape`

- `seam_residency`
  - event-local or event-anchored seam-nearness summary

- `escape_internal_share`
  - event-context routing tendency toward escape-internal continuation

- `core_to_escape_share`
  - event-context tendency toward genuine crossing

- `local_gateway_strength`
  - strength of local launch-side explanation for the event or mapped family context

- `path_context_gain`
  - incremental predictive contribution of broader context over local-only predictors

- `effective_temporal_depth`
  - event-mapped family memory depth or later event-native surrogate

- `forgetting_share`
  - event-mapped or family-mapped compression summary

### Minimum assignment outputs

- `event_id`
- `assigned_family`
- `assignment_confidence`
- `assignment_ambiguity_flag`

### Uncertainty

Several classifier-supporting quantities are currently most stable at family or per-study level, not fully native at event level.

## Provisional decision logic

The first canonical assignment surface should remain conservative and rule-based.

### Decision sequence

1. Assign `branch_exit` when evidence favors downstream/immediate behavior:
   - high `escape_internal_share`
   - low `core_to_escape_share`
   - immediate temporal regime
   - weak compression

2. Among remaining events, assign `stable_seam_corridor` when evidence favors local seam-boundary gateway behavior:
   - strongest seam residency
   - strongest `local_gateway_strength`
   - short/one-step `effective_temporal_depth`
   - rapid compression

3. Otherwise assign `reorganization_heavy` when evidence favors path-context and extended-memory behavior:
   - strongest `path_context_gain`
   - extended-memory regime
   - strongest `forgetting_share`

### Important caution

All thresholds, score margins, and exact weighting remain provisional.

The first implementation should prefer explicit ambiguity over premature precision.

## Provenance policy

Values entering canonical tables must be classified as one of:

- `event_native`
  - computed directly from that event and its local/history context

- `trajectory_mapped`
  - computed at trajectory level and attached to the event

- `family_mapped`
  - computed at family level and attached to all events currently assigned to that family

- `study_inherited`
  - copied from per-study artifacts without native event/trajectory resolution

### Precedence rule

Use this precedence order:

- `event_native`
- `trajectory_mapped`
- `family_mapped`
- `study_inherited`

### Rules

1. prefer `event_native`
2. if no event-native value exists, use `trajectory_mapped` if recoverable
3. if no event/trajectory-native value exists, use `family_mapped` only when the quantity is scientifically central and already stabilized at family level
4. use `study_inherited` conservatively and mark provisional by default
5. do not present mapped or inherited values as if they were event-native

### Minimum provenance fields

- `provenance_class`
- `source_table`
- `source_key`
- `projection_flag`
- `provisional_flag`

### Important note

Mixed provenance across columns within the same event row is expected during early implementation and must be represented explicitly.

## Ambiguity handling

Ambiguity must be represented explicitly at event level.

### Required

- `assignment_confidence`
- `assignment_ambiguity_flag`

### Recommended

- `runner_up_family`
- `score_margin`
- `manual_review_flag`

### Policy

- do not force low-margin or mixed-evidence events into one family without ambiguity marking
- if an event lacks support for key classifier variables, assignment may be:
  - null
  - ambiguous
  - marked for manual review

### Most likely ambiguous boundary

- `stable_seam_corridor` vs `reorganization_heavy`

Reason:

- both are seam-engaged and not purely downstream
- the key distinction is local-gateway versus path-context dependence, not simple seam contact alone

## Validation logic

The event-level layer is coherent only if all of the following hold.

### 1. Assignment consistency

- every row in `event_family_assignment` joins to exactly one row in `event_family_features`

### 2. Family aggregation consistency

- family counts equal counts of assigned events
- aggregated family means match event-level aggregates

### 3. Gateway consistency

- events assigned to `branch_exit` aggregate to highest `escape_internal_share`
- events assigned to `stable_seam_corridor` aggregate to strongest local gateway signal
- events assigned to `stable_seam_corridor` also aggregate to strongest core-to-escape boundary identity
- events assigned to `reorganization_heavy` aggregate to strongest `path_context_gain`

### 4. Temporal consistency

Event assignments aggregated by family should reproduce:

- `branch_exit` = immediate
- `stable_seam_corridor` = one-step / short-memory
- `reorganization_heavy` = extended-memory

### 5. Compression consistency

Event assignments aggregated by family should reproduce:

- weakest compression for `branch_exit`
- rapid/intermediate compression for `stable_seam_corridor`
- strongest compression for `reorganization_heavy`

### 6. Provenance consistency

- projected or inherited values are explicitly marked
- no family-level quantity is silently treated as event-native

### 7. Event-type consistency

- primary classifier rows use normalized gateway event types:
  - `core_internal`
  - `core_to_escape`
- continuation-only rows are either:
  - normalized separately
  - or excluded from the first minimal classifier surface

## Backward-compatible rollout

The rollout should use a downstream consolidation layer rather than upstream rewrites.

### Recommended rollout

1. keep existing study outputs unchanged
2. build family-level canonical summaries first:
   - `gateway_comparison`
   - `temporal_depth_summary`
   - `memory_compression_summary`
3. add an event feature consolidation script that reads boundary/gateway artifacts and emits `event_family_features`
4. populate event rows with:
   - event-native values where available
   - trajectory-mapped values where available
   - family-mapped values only where needed
5. build `event_family_assignment`
6. build `family_aggregation`
7. optionally add later:
   - `trajectory_family_rollup`
   - trajectory-level summaries
   - richer event-history features

This keeps current study outputs intact and adds only a downstream canonicalization layer.

## Known limitations

- the exact canonical event extractor is not fully specified yet
- some classifier-supporting quantities are still strongest at family level, not yet event-native
- `branch_exit` remains somewhat less secure due to lower effective support in some analyses
- `effective_temporal_depth` for `reorganization_heavy` should be treated categorically as extended-memory, not as a final precise event-level scalar
- `dominant_compression_state` can be contaminated by padded or non-structural states unless cleaned
- event-level family assignment may not map trivially to one stable family per whole trajectory if trajectories mix regimes across events
- `local_gateway_strength` and `path_context_gain` are currently easiest to support as family- or study-level quantities, not native event-level measurements
- current OBS datasets may use different row semantics and therefore require normalization before consolidation

## Immediate implementation priority

Build first:

1. `gateway_comparison`
2. `temporal_depth_summary`
3. `memory_compression_summary`
4. `event_family_features`
5. `event_family_assignment`

This is the earliest scientifically useful event-level implementation.

`family_aggregation` should follow immediately after assignments are stable.

### Optional later priority

- `trajectory_family_rollup`, derived from event assignments once event-level classification is functioning coherently

## Relationship to companion specs

This document is complemented by:

- `canonical_family_gateway_spec.md`
- `canonical_event_normalization_contract.md`
- `canonicalization_implementation_plan.md`

Together, these documents define:

- the canonical family/gateway target,
- the primary event-level classifier surface,
- the normalization contract for heterogeneous OBS rows,
- and the implementation sequence for canonicalization.
