# Linked Response Annotation Protocol

## Purpose

This note records the procedure used to annotate the **link-bearing response subset** of the corpora.

The aim was to turn an initially subjective observation — that some responses contained highly relevant external links — into a controlled, inspectable annotation pass.

## Corpus and model scope

- Corpus families: `C`, `Cp`, `Cp4`
- Model used for corpus generation: **GPT-5.2**
- Current interpretation scope: **within-model**

This protocol therefore applies to the GPT-5.2 corpus stage only.

## Unit of analysis

The primary unit of analysis is:

- **one linked response**

A linked response is any response in the filtered subset that contains at least one extracted link.

In the working artifact:
- one row corresponds to one linked response
- response-level fields summarize the packet as a whole

## Source materials

The annotation pass relied on:

1. the filtered response/links CSV
2. copied response text
3. direct inspection of linked content
4. screenshots supplied when a link could not be reliably opened
5. later GUI verification when provenance questions arose

## Response-level fields

The response-level taxonomy currently uses the following fields.

### `family_mode`
Top-level mode of external witnessing.

Allowed values:
- `geometric_externalization`
- `formal_structural_packetization`
- `distributed_emergence_packetization`

### `subtype`
A narrower descriptive label for the packet family inside each `family_mode`.

Examples:
- `stabilization_ladder`
- `atlas_mode_externalization`
- `invariance_projection_embedding_packet`
- `network_condensation_packet`
- `developmental_cognition_packet`

### `packet_architecture`
How the response organizes the witness set.

Examples:
- `support_bundle`
- `witness_bundle`
- `motif_factorized_bundle`
- `formal_packet`
- `developmental_packet`
- `section_wise_anchor_blocks`
- `stabilization_ladder`

### `primary_motif_1..3`
Three short motif labels summarizing the dominant structural content of the packet.

Examples:
- `invariance`
- `symmetry`
- `projection_distortion`
- `attractors`
- `neural_coupling`
- `concept_mapping`
- `self_organization`

### `pressure_valve_judgment`
A coarse response-level interpretation of whether links appear to function as a release or externalization channel for the response.

Allowed values currently used:
- `moderate`
- `strong`

This is intentionally coarse at the present stage.

### `text_integrity`
Used to track whether the copied text reliably matches the original linked response.

Allowed values currently retained:
- `clean`

A previous boundary-corrupted label was removed from the current linked set after `Cp:27` was verified to be non-linked in the original GUI response.

### `exemplar_status`
Used to identify especially important rows for family comparison.

Allowed values:
- `canonical`
- `supporting`

### `canonical_summary`
A one-sentence repository-facing description of why the packet matters.

---

## Link-level reasoning

Although the final response-level artifact is one row per linked response, the qualitative pass also relied on manual per-link interpretation.

Useful manual vocabulary included:

- `paper`
- `plot_geometry`
- `pedagogical_diagram`
- `concept_diagram`
- `concept_illustration`

And adjacency judgments like:
- topical relevance
- structural relevance
- both

These per-link judgments informed the response-level taxonomy even when they were not all tabulated explicitly in the final bundle.

## Use of screenshots

Some links were not reliably viewable directly.

In those cases:
- screenshots were supplied and used as the evidentiary basis
- the screenshot became the operative witness for annotation
- the interpretation stayed tied to the visible content rather than inferred metadata

This was especially important for:
- direct image URLs
- links with unstable rendering
- links blocked by interface or hosting friction

## Correction protocol

When a copied response was later found not to match the original GUI state, the original GUI reading took precedence.

### Example: `Cp:27`
A copied specimen initially appeared to be a boundary-corrupted linked response.
Later direct verification in the Linux ChatGPT GUI showed that the original response contained **no links**.
As a result:

- `Cp:27` was removed from the linked-response analysis set
- downstream artifacts were patched
- the correction was documented explicitly

This should be treated as the model for future provenance corrections:
**GUI verification outranks copied text when they conflict.**

## How labels were assigned

Labels were assigned by close reading of:

- the response prose
- the linked images or pages
- the relationship between claim and witness
- the compactness or staging of the packet
- recurrence across family members

The family-mode decision was based on the packet as a whole, not on any single link in isolation.

## Interpretation stance

The annotation protocol is:

- qualitative
- controlled
- example-backed
- still provisional

It is not yet a blind-coded or inter-rater protocol.

That is acceptable at the current observatory stage, but should be stated plainly.

## Recommended repository wording

A safe method statement is:

> Linked responses were manually reviewed at the response level, with linked content inspected directly or by screenshot when necessary. Labels were assigned to characterize family-level modes of external witnessing within the GPT-5.2 corpus set.

## Related files

See:
- `linked_response_family_taxonomy.md`
- `linked_response_artifacts.md`
- the patched taxonomy bundle
