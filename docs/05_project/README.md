# Document roles

This folder mixes two kinds of material:

## Canonical project reference

These documents define current repository-facing project objects and contracts, such as:

- family-layer status
- canonical classification and normalization specs
- gateway spec
- canonical response-guided flow
- linked-response family taxonomy and annotation protocol

## Planning / historical design material

Some files in this folder are retained as implementation plans or stabilization notes. These are useful for historical context and development tracking, but they are not always the current canonical reference surface.

When reading this folder, prefer the canonical reference docs first, then consult planning documents for implementation history or roadmap context.

---

# Project

This section contains project-level documentation for the PAM Observatory.

It is the home for:

- stabilization plans
- canonicalization specs
- implementation roadmaps
- milestone and release notes
- repository maintenance procedures
- canonical layer status and architecture notes
- observatory-stage repository-facing documentation

## Current role

The repository now has:

- a canonical implemented runtime for the downstream observatory pipeline
- a downstream canonical family/gateway layer implemented in `scripts/canonical/`
- a first repository-facing canonical dynamical-layer object for OBS-043
- validated canonical artifacts under `outputs/canonical/`
- a repository-facing linked-response observatory layer preserved through OBS-053 and OBS-054

Primary runtime entrypoints:

- `scripts/run_full_pipeline.sh`
- `src/pam/pipeline/runner.py`

Primary architecture reference:

- [`../architecture.md`](../architecture.md)

---

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

---

## Canonicalization and stabilization

The documents in this section define **provisional canonical contracts** for research structures that are scientifically mature enough to stabilize, but not yet ready to be treated as fully internalized runtime primitives.

These documents are intended to:

- make evolving observatory structures explicit
- provide stable implementation targets
- distinguish canonical direction from transitional study outputs
- improve public-facing transparency as the repository catches up to the science
- mark what is implemented, what is validated, and what remains provisional

---

## Canonical dynamical layer

The repository now also includes a first canonical dynamical-layer object:

- [`canonical_response_guided_flow.md`](./canonical_response_guided_flow.md)

This document formalizes the response-guided flow construction introduced in OBS-043:
- dominant response-eigenvector direction field
- embedded discrete flow paths
- seam engagement
- phase crossing
- seam-bundle scalar modulation
- current route-family refinement

Related study scripts include:

- `experiments/studies/obs043_response_flow.py`
- `experiments/studies/obs043b_response_flow_path_families.py`

This dynamical layer is now scientifically established at first pass, but remains methodologically discretized and is not yet promoted into `src/pam/`.

---

## Linked-response observatory layer

The repository now also includes a repository-facing linked-response observatory layer.

This layer was formalized through:

- `OBS-053` — family-structured external witnessing in the GPT-5.2 linked-response subset
- `OBS-054` — consolidation of the linked-response taxonomy instrument at repository level

Current linked-response reference docs:

- [`linked_response_family_taxonomy.md`](./linked_response_family_taxonomy.md)
- [`linked_response_annotation_protocol.md`](./linked_response_annotation_protocol.md)
- [`linked_response_artifacts.md`](./linked_response_artifacts.md)

Current linked-response artifact set:
- patched linked-response taxonomy bundle
- response-level taxonomy CSV
- enriched source CSV
- family summary CSV
- controlled vocabulary CSV

Current linked-response family modes:

- `geometric_externalization`
- `formal_structural_packetization`
- `distributed_emergence_packetization`

Important scope note:

- this linked-response stage is currently **within-model**
- all corpora in this stage were generated with **GPT-5.2**
- the taxonomy is **annotation-backed** and **provisional**
- `Cp:27` was removed after GUI verification showed it was not truly link-bearing in the original response

This layer is not yet a runtime primitive. It is best understood as a repository-facing observatory instrument and documentation layer.

---

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

The linked-response observatory layer is now:

- documented
- artifactized
- vocabulary-stabilized
- provenance-corrected

But it remains a qualitative observatory surface rather than a promoted runtime layer.

---

## Documents

Canonical family-layer documents include:

- [`canonical_family_gateway_spec.md`](./canonical_family_gateway_spec.md)
- [`canonical_event_family_classification_spec.md`](./canonical_event_family_classification_spec.md)
- [`canonical_event_normalization_contract.md`](./canonical_event_normalization_contract.md)
- [`canonicalization_implementation_plan.md`](./canonicalization_implementation_plan.md)

Canonical dynamical-layer document:

- [`canonical_response_guided_flow.md`](./canonical_response_guided_flow.md)

Linked-response observatory documents:

- [`linked_response_family_taxonomy.md`](./linked_response_family_taxonomy.md)
- [`linked_response_annotation_protocol.md`](./linked_response_annotation_protocol.md)
- [`linked_response_artifacts.md`](./linked_response_artifacts.md)

Artifact registry:

- [`artifact_registry.md`](./artifact_registry.md) — file-first index of major runtime, downstream canonical, and repository-facing observatory artifacts, including status and scope caveats

Observatory notes directly relevant to the current project layer include:

- `OBS-052` — attractor basin mapping
- `OBS-053` — linked-response family taxonomy finding
- `OBS-054` — linked-response instrument consolidation

---

## Guidance

For current runtime and high-level structure, prefer:

- the repository root documentation
- [`../architecture.md`](../architecture.md)

For project-level stabilization, canonicalization, implementation planning, observatory-stage consolidation, and repository-facing instrument notes, prefer the documents in this section.

For the linked-response stage specifically, start with:

1. [`linked_response_family_taxonomy.md`](./linked_response_family_taxonomy.md)
2. [`linked_response_annotation_protocol.md`](./linked_response_annotation_protocol.md)
3. [`linked_response_artifacts.md`](./linked_response_artifacts.md)