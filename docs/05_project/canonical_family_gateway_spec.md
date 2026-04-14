# Provisional Canonical Spec — Family and Gateway Layer

## Purpose

Define a provisional canonical family-and-gateway layer for the PAM Observatory.

This document exists to stabilize the current scientific picture of:

- canonical route families,
- gateway structure,
- the relation between family structure and gateway behavior,
- the family-level tables that should anchor downstream canonicalization work.

This is a project-level specification. It is intended to make the current direction explicit and implementation-targetable without pretending that every quantity is already fully consolidated in code.

## Status

This document is **provisional but normative** for the current canonicalization effort.

That means:

- it describes the current best stabilized picture,
- it is allowed to mark uncertainty explicitly,
- it should be treated as a target for downstream implementation and documentation cleanup.

It does **not** claim that every concept here is already exposed through one canonical runtime artifact.

## Scientific role

The family-and-gateway layer exists to provide a compact and reproducible account of seam-organized dynamics.

Its scientific role is to:

- separate distinct seam interaction modes into a small canonical family set,
- explain why pooled gateway laws were weak,
- connect local gateway behavior, path-context dependence, temporal depth, and memory compression,
- provide a stable family vocabulary for future observatory work.

In the current picture, family structure is not merely descriptive. It is one of the main explanatory partitions of seam dynamics.

## Canonical family picture

The current stabilized family set is:

- `branch_exit`
- `stable_seam_corridor`
- `reorganization_heavy`

These names are canonical at the family level.

### `branch_exit`

Current stabilized picture:

- directed/downstream family
- strongest association with escape-internal behavior
- immediate regime at current observatory scale
- weak memory compression relative to the other families

Interpretation:

- its defining structure lies mainly after release or at an already-directed boundary
- it is less strongly organized around prolonged seam-core residence than the other families

Caution:

- “immediate at current scale” is safer than stronger claims such as “memoryless”

### `stable_seam_corridor`

Current stabilized picture:

- canonical local gateway family
- strongest seam residency
- strongest local gateway dependence
- short/one-step effective memory regime
- rapid compression

Interpretation:

- this is the clearest case where local seam-boundary state explains whether a crossing occurs
- it is the main seam-resident and boundary-sampling family

Caution:

- no single variable or threshold is sufficient on its own to define corridor behavior

### `reorganization_heavy`

Current stabilized picture:

- canonical path-context family
- extended-memory regime
- strongest memory compression
- strongest dependence on broader path context

Interpretation:

- this family is not well explained by local gateway state alone
- broader route history matters for crossing behavior
- it is long-memory globally but internally compressive

Caution:

- the robust statement is “extended-memory,” not one final exact short horizon value

## Why families were introduced

Families were introduced to compress recurring classes of seam-adjacent trajectory behavior into a small number of stable route types.

Their role is not merely visual or descriptive. They exist to explain:

- why trajectories interact with the seam differently,
- why pooled gateway laws were weak,
- why some routes are local-gateway dominated while others are context-dominated,
- why temporal depth and memory compression differ systematically across seam behaviors.

## Gateway role

The gateway is the operational transition regime between reversible core structure and directed escape structure.

In the current family picture, gateway behavior is expressed through the contrast between:

- `core_internal`
- `core_to_escape`

with related continuation behavior including:

- `escape_internal`
- `escape_to_core`

### Gateway versus seam

Current stabilized distinction:

- the **seam** is the broader boundary/reorganization regime
- the **gateway** is the operational transition law governing whether boundary-adjacent behavior remains internal or becomes escape-directed

Interpretation:

- the seam is the larger organization field
- the gateway is one operational law inside that field

### Family-conditioned gateway picture

The current stabilized gateway picture is family-specific:

- `branch_exit`
  - downstream/immediate
  - strongest escape-internal tendency
  - weakly compressive

- `stable_seam_corridor`
  - local seam-boundary gateway family
  - strongest local gateway signal
  - short/one-step regime
  - rapid compression

- `reorganization_heavy`
  - context-dependent gateway family
  - strongest path-context gain
  - extended-memory regime
  - strongest compression

This is the main reason the family partition matters scientifically.

## Failure of simple pooled gateway models

Simple pooled gateway explanations were too weak.

The current interpretation is that gateway behavior is not adequately explained by:

- one pooled predictor law,
- symbolic resolution alone,
- one-step memory alone.

The replacement picture is family-conditioned:

- downstream/immediate for `branch_exit`
- local seam-boundary for `stable_seam_corridor`
- broader path-context for `reorganization_heavy`

## Canonical family-level tables

The current canonicalization effort should treat the following family-level tables as primary targets:

1. `gateway_comparison`
   - one row per family
   - stores family-level gateway and routing quantities

2. `temporal_depth_summary`
   - one row per family, or family × encoding if needed
   - stores memory-depth and horizon summaries

3. `memory_compression_summary`
   - one row per family
   - stores forgetting-share and dominant compression-state summaries

4. `family_aggregation`
   - one row per family
   - aggregates downstream assigned event rows once event-level classification exists

These tables are the family-level anchor layer for later event-level canonicalization.

## Minimal family-level classifier-supporting quantities

The smallest current family-supporting quantity set is:

- `seam_residency`
- `escape_internal_share`
- `core_to_escape_share`
- `local_gateway_strength`
- `path_context_gain`
- `effective_temporal_depth`
- `forgetting_share`

Strongly recommended:

- `dominant_compression_state`

These quantities are not all equally native yet. Some are currently stabilized more strongly at family level than at event or trajectory level.

## Provenance principles

The canonicalization effort should distinguish explicitly between:

- native local measurements,
- mapped values,
- inherited study-level quantities.

At the family-and-gateway layer, the main principle is:

- prefer direct family-level operational summaries where they already exist,
- prefer conservative mapping over invented precision,
- prefer null or explicit provisional status over hidden projection.

## Ambiguity principles

The family picture is strong at the family level, but boundary cases still exist.

Most likely ambiguity:

- `stable_seam_corridor` vs `reorganization_heavy`

Reason:

- both are seam-engaged and not purely downstream
- their key distinction is local-gateway versus path-context dependence, not simple seam contact alone

The canonicalization layer should preserve ambiguity explicitly rather than force false precision.

## Validation principles

The family-and-gateway layer is coherent only if the stabilized family distinctions remain visible after consolidation.

The minimum scientific coherence checks are:

- `branch_exit` has highest `escape_internal_share`
- `stable_seam_corridor` has strongest local gateway signal
- `stable_seam_corridor` has strongest core-to-escape gateway identity
- `reorganization_heavy` has strongest `path_context_gain`
- temporal ordering remains:
  - `branch_exit` = immediate
  - `stable_seam_corridor` = one-step / short-memory
  - `reorganization_heavy` = extended-memory
- compression ordering remains:
  - `branch_exit` weakest
  - `stable_seam_corridor` rapid/intermediate
  - `reorganization_heavy` strongest

## Known limitations

The following remain unresolved or only partly stabilized:

- no single final scalar gateway equation
- no universal threshold set for family assignment
- no one final production predictor architecture
- some classifier-supporting quantities are strongest at family level, not yet event-native
- `branch_exit` has lower effective support in some analyses and should be handled with moderate confidence
- `effective_temporal_depth` for `reorganization_heavy` should be treated categorically as extended-memory
- compression-state reporting must guard against non-structural padding artifacts

## Implementation consequence

This specification implies the following implementation order:

1. stabilize family-level gateway summaries
2. stabilize family-level temporal-depth summaries
3. stabilize family-level compression summaries
4. only then build the event-level canonical row layer that consumes them conservatively

So the family-and-gateway layer should be treated as the first canonicalization target, not as a later byproduct of row-level normalization.

## Relationship to later specs

This document is complemented by:

- `canonical_event_family_classification_spec.md`
- `canonical_event_normalization_contract.md`
- `canonicalization_implementation_plan.md`

Together, these documents define:

- the conceptual family/gateway target,
- the event-level classifier surface,
- the row normalization contract,
- and the implementation sequence.
