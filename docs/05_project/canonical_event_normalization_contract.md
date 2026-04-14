# Provisional Canonical Spec — Event Normalization Contract

## Purpose

Define how heterogeneous OBS gateway-study rows are normalized into one canonical `event_family_features` table.

The current event-bearing outputs do not all share one row semantics. Some are:

- composition summaries
- instance-level gateway rows
- history-aware gateway rows
- family-specific path-context rows
- step/path rows adjacent to seam or escape behavior

The purpose of normalization is not to erase those differences. It is to:

- preserve them explicitly,
- align shared columns conservatively,
- and produce one consistent downstream event-level classifier substrate.

This is a downstream canonicalization contract, not an upstream rewrite.

## Status

This document is **provisional but normative** for the current canonicalization effort.

That means:

- it defines how event rows should be normalized into the canonical layer,
- it permits explicit uncertainty where source artifacts differ,
- it should be treated as the implementation contract for the first event-level consolidation pass.

It does **not** claim that one universal event key or one final event extractor already exists upstream.

## Scope

This contract governs only normalization into the canonical event-level layer.

It does not:

- redefine the scientific meaning of the gateway,
- replace or overwrite upstream study outputs,
- force one study’s row semantics onto another,
- or guarantee full event-to-trajectory linkage in the first implementation.

## Canonical event concept

A canonical event is a normalized row representing one gateway-relevant decision or boundary-local transition context.

The first minimal classifier surface should prioritize **gateway decision events**, especially rows that distinguish:

- `core_internal`
- `core_to_escape`

Continuation-oriented rows such as:

- `escape_internal`
- `escape_to_core`
- `other`

may be retained as secondary event types, but they are not required for the first minimal classifier build.

## Normalization principle

Normalization should preserve three things simultaneously:

1. **event meaning**
   - whether the row is a gateway decision event, a continuation event, or a supporting step/path row

2. **source semantics**
   - whether the row came from:
     - composition summary
     - instance-level event
     - history-aware event
     - family-specific context row
     - step/path row

3. **classifier usability**
   - whether the row contains enough information to enter the first canonical event-level classifier surface

The normalized table should therefore preserve row type and provenance explicitly. It should not flatten away source semantics.

## Source row classes

The current artifact surface suggests at least these source row classes.

### 1. Composition-summary rows

Examples:
- OBS-034
- OBS-035

Typical properties:
- aggregated over compositions
- may include `n_compositions`, `composition_share`
- may carry `composition_typed`
- often already labeled by `route_class` and `crossing_type`

Normalization role:
- valid source for coarse gateway-event rows
- useful when instance-level rows are absent
- must be marked as aggregated event rows, not instance-native rows

### 2. Instance-level gateway rows

Examples:
- OBS-035c
- OBS-036

Typical properties:
- closer to individual gateway event instances
- launch-side fields available
- may include refined launch and target state labels

Normalization role:
- preferred source for local gateway features
- strongest current candidate for event-native classifier rows

### 3. History-aware gateway rows

Examples:
- OBS-037
- OBS-037b

Typical properties:
- add previous-generator or prelaunch history context
- may include history words or prelaunch words

Normalization role:
- preferred source for event-native path-history features
- should extend instance-level rows where justified joins exist
- may also stand alone as normalized rows if no cleaner join exists

### 4. Family-specific path-context rows

Examples:
- OBS-039

Typical properties:
- richer cumulative and recent-context features
- may exist only for one family
- often already event-like but specialized

Normalization role:
- valid source for context-rich event rows
- should be included with explicit provenance
- missing columns for other families should remain null rather than imputed

### 5. Step/path rows

Examples:
- OBS-029

Typical properties:
- stepwise seam/escape movement
- may contain `path_id`
- may contain both `route_class` and older `path_family`
- support seam/escape behavior but are not always gateway decision rows directly

Normalization role:
- supporting source for:
  - `trajectory_id`
  - step-local seam anchoring
  - event-to-path linkage when recoverable
- should not automatically be treated as primary gateway decision rows

## Canonical normalized row types

The normalized table should include an explicit row-type field, such as:

- `gateway_composition`
- `gateway_instance`
- `gateway_history`
- `gateway_context`
- `supporting_step`

This field is required because different source rows are scientifically useful but not interchangeable.

## Minimal normalized columns

Every row in `event_family_features` should aim to contain these core columns.

### Required canonical identity fields

- `event_id`
- `source_row_type`
- `source_table`
- `source_key`
- `route_class`
- `event_type`

### Required classifier-supporting fields

- `y_cross`
- `seam_residency`
- `escape_internal_share`
- `core_to_escape_share`
- `local_gateway_strength`
- `path_context_gain`
- `effective_temporal_depth`
- `forgetting_share`

### Required provenance fields

- `provenance_class`
- `projection_flag`
- `provisional_flag`
- `normalization_version`

### Strongly recommended linkage fields

- `trajectory_id`
- `composition_typed`
- `launch_generator`
- `next_generator`
- `prev_generator`
- `launch_state`
- `launch_target`
- `prev_state`
- `prev_target`

Not all source rows will populate all of these fields. Sparse-but-explicit is preferred over invented completeness.

## Event ID policy

`event_id` is required in the canonical layer even if it does not exist upstream.

### Rule

If upstream data do not provide a stable event identifier, the consolidation layer must synthesize one.

### Acceptable basis for synthetic event IDs

A synthetic `event_id` may be created from:

- source table
- stable row index or deterministic row hash
- `route_class`
- `event_type`
- composition or generator signature where needed

### Constraints

Synthetic IDs must be:

- deterministic
- versioned with the normalization contract
- reproducible from the same source artifacts

## Event type normalization

The canonical `event_type` vocabulary should begin minimally with:

### Primary gateway event types

- `core_internal`
- `core_to_escape`

### Secondary continuation event types

- `escape_internal`
- `escape_to_core`
- `other`

### Rule

The first minimal classifier surface should prioritize rows normalized to:

- `core_internal`
- `core_to_escape`

Rows of other types may be:

- retained in the canonical table,
- excluded from the first classifier,
- or used only in auxiliary routing summaries.

## Route class normalization

The canonical family label vocabulary is:

- `branch_exit`
- `stable_seam_corridor`
- `reorganization_heavy`

### Rule

If a row already carries canonical `route_class`, preserve it.

If a row carries only older `path_family` labels:

- do not silently coerce them into canonical `route_class`
- retain them in auxiliary columns until a separately justified mapping exists

This prevents accidental contamination of the canonical family surface.

## Column inheritance rules

When populating normalized rows, use the following precedence:

1. `event_native`
2. `trajectory_mapped`
3. `family_mapped`
4. `study_inherited`

### Examples

- `y_cross` from OBS-035c / OBS-036 / OBS-037 event rows → `event_native`
- `path_context_gain` from family summary only → `family_mapped`
- `effective_temporal_depth` from family-level temporal synthesis → `family_mapped`
- `trajectory_id` recovered from OBS-029 linkage → `trajectory_mapped`

## Join policy across studies

Rows from different OBS datasets should only be merged at event level if there is a justified join key.

### Safe join keys

Potentially safe keys include:

- stable upstream event identifier if present
- deterministic synthetic `event_id`
- `composition_typed` plus route/event context when uniquely sufficient
- explicit path/step linkage if available

### Unsafe joins

Do not merge rows solely because they share:

- `route_class`
- `crossing_type`
- family-level interpretation

These are grouping variables, not event identifiers.

## First minimal normalization strategy

For the first usable implementation:

### Include directly as primary event sources

- OBS-035c instance-level gateway rows
- OBS-036 refined gateway rows
- OBS-037 history-aware gateway rows
- OBS-037b pre-second-step rows
- OBS-039 family-specific path-context rows

### Use as fallback or coarse event support

- OBS-034 and OBS-035 composition-summary rows

### Use as family-level enrichment only

- OBS-038 family-specific gateway summaries
- OBS-040 / OBS-040b temporal summaries
- OBS-041 compression summaries
- OBS-042 family temporal-regime synthesis

### Use as supporting linkage source

- OBS-029 step/path rows

This gives the cleanest event-first layer without forcing speculative cross-study row matching.

## Nullability rules during normalization

A normalized field must be null if:

- no event-native value exists,
- no justified mapped value exists,
- the join needed to recover the value is not trustworthy,
- the only available source is interpretive rather than operational.

### Conservative rule

Prefer null over speculative cross-study merge or pseudo-precise projection.

## Quality flags

The normalized table should emit quality signals such as:

- `missing_event_native_flag`
- `family_mapped_flag`
- `study_inherited_flag`
- `unsupported_join_flag`
- `non_gateway_event_flag`
- `ambiguous_event_type_flag`

These may exist either as explicit columns or be derivable from provenance metadata.

## Validation logic for normalization

Normalization is behaving sensibly only if all of the following hold.

### 1. Identity completeness

Every normalized row has:

- `event_id`
- `source_row_type`
- `source_table`
- `route_class`
- `event_type`

### 2. Event-type normalization

Primary classifier rows use normalized event types:

- `core_internal`
- `core_to_escape`

### 3. Event ID integrity

Synthetic IDs are:

- deterministic
- unique within a normalization version
- reproducible from the same source artifacts

### 4. Join discipline

No join is performed solely on family label or gateway label when event-level identity is required.

### 5. Provenance visibility

Mapped and inherited values are explicitly flagged.

### 6. Row-type preservation

Rows from different source row classes remain distinguishable after normalization.

## Known limitations

- current OBS datasets do not yet expose one universal event key
- some studies are aggregated and others are instance-level
- some features exist only for certain families or certain studies
- event-to-trajectory linkage may be partial in the first implementation
- some quantities central to classification remain family-level enrichments rather than event-native measurements

## Immediate implementation consequence

The first consolidation script for `event_family_features` should:

1. ingest OBS-035c, OBS-036, OBS-037, OBS-037b, and OBS-039 as primary event sources
2. normalize:
   - `event_type`
   - `route_class`
   - row type and source semantics
   - core local/history/context columns
3. synthesize deterministic `event_id` values
4. attach provenance and row-type labels
5. enrich rows conservatively with family-level values from OBS-038, OBS-040/040b, OBS-041, and OBS-042
6. leave uncertain fields null rather than over-joining

This is the safest first canonical event-normalization layer.

## Relationship to companion specs

This document is complemented by:

- `canonical_family_gateway_spec.md`
- `canonical_event_family_classification_spec.md`
- `canonicalization_implementation_plan.md`

Together, these documents define:

- the family/gateway target,
- the event-level classifier surface,
- the normalization rules that connect OBS rows to that surface,
- and the implementation sequence for the canonicalization layer.
