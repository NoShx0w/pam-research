# Linked Response Artifacts

## Purpose

This note indexes the main artifacts produced during the linked-response study.

It is intended as a lightweight repository pointer rather than a full interpretive document.

## Primary bundle

Use the patched bundle as the current canonical artifact for this stage.

### Bundle
- `linked_response_family_taxonomy_bundle_patched.zip`

## Included files

### `linked_response_taxonomy.csv`
Canonical response-level taxonomy.

One row per linked response.

Key columns:
- `corpus`
- `response_id`
- `text_index`
- `n_links`
- `domains`
- `family_mode`
- `subtype`
- `packet_architecture`
- `primary_motif_1`
- `primary_motif_2`
- `primary_motif_3`
- `pressure_valve_judgment`
- `text_integrity`
- `exemplar_status`
- `canonical_summary`

### `linked_response_source_enriched.csv`
The filtered source table enriched with taxonomy columns.

Useful for:
- auditing against the original filtered link set
- checking copied text against taxonomy assignments
- tracing summary rows back to source rows

### `linked_response_family_summary.csv`
Small summary table by:
- `corpus`
- `family_mode`

Useful for quick stage-level reporting.

### `linked_response_family_notes.csv`
Short textual signature notes for the three family modes.

Useful as a compact dictionary for downstream writeups.

### `linked_response_controlled_vocab.csv`
Controlled vocabulary and field notes.

Useful for:
- freezing ontology at this stage
- keeping downstream docs aligned with the same label set

## Current canonical family modes

- `geometric_externalization`
- `formal_structural_packetization`
- `distributed_emergence_packetization`

## Important patch note

`Cp:27` was removed from the linked-response set after later GUI verification showed that the original response contained no links.

The patched bundle supersedes earlier versions.

## Scope reminder

All corpora in this stage were generated with **GPT-5.2**.

Interpret the artifacts as:
- within-model
- annotation-backed
- provisional

## Recommended use

For repository-facing work:

1. cite `linked_response_taxonomy.csv` for the current row-level taxonomy
2. cite `linked_response_family_summary.csv` for compact summary numbers
3. cite `linked_response_family_taxonomy.md` for interpretation
4. cite `linked_response_annotation_protocol.md` for method and scope

## Suggested future additions

Likely follow-on artifacts:
- a lightweight quantitative summary notebook
- family-wise plots of subtype counts
- direct-image vs non-image link summaries
- domain/source distribution summaries
- an explicit exemplar gallery for canonical rows
