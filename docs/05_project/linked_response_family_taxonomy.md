# Linked Response Family Taxonomy

## Purpose

This note documents the current repository-facing interpretation of the **linked-response subset** of the corpus study.

The goal of this stage was not to explain all responses in the corpora, but to characterize the responses that contained explicit outbound links or direct image links and to ask whether those linked responses exhibited stable structural differences across corpus families.

## Scope

- Corpus families analyzed: `C`, `Cp`, `Cp4`
- Unit of analysis: **one linked response**
- Model scope: all corpora in this stage were generated with **GPT-5.2**
- Interpretation status: **within-model**, **annotation-backed**, **provisional**

This note should therefore be read as a description of **within-model corpus-family effects** rather than as a cross-model claim about LLMs in general.

## Core provisional result

The linked-response subset does **not** appear random or merely topic-relevant.

Instead, the current qualitative pass suggests that the mode of external witnessing differs by corpus family:

- `C` tends toward **geometric externalization**
- `Cp` tends toward **formal-structural packetization**
- `Cp4` tends toward **distributed-emergence packetization**

This is the main qualitative result of the linked-response study at the current stage.

---

## Family modes

### 1. `geometric_externalization`

**Typical family:** `C`

This mode tends to externalize through geometric witness sets that are often broader and more architected than the other families.

Observed tendencies include:

- geometric and topological figures
- phase-flow or attractor imagery
- ladder-like or staged motif development
- section-wise visual blocks
- atlas-like organization across longer responses

This mode often feels **exploratory but structured**: the response installs a visual field around the prose and then moves through that field.

**Canonical examples**
- `C:28`
- `C:33`
- `C:34`
- `C:36`
- `C:47`
- `C:49`

**Useful shorthand**
- witness bundle
- stabilization ladder
- atlas-mode externalization

---

### 2. `formal_structural_packetization`

**Typical family:** `Cp`

This mode tends to externalize through compact formal packets with very low slack between the prose claim and the chosen witnesses.

Observed tendencies include:

- invariance
- symmetry
- topology
- embedding
- projection
- attractors
- constrained deformation
- representation learning

The linked materials are usually tightly selected and concept-dense. The packet often behaves like a formal witness set for a small number of structural claims.

**Canonical examples**
- `Cp:3`
- `Cp:6`
- `Cp:13`
- `Cp:22`
- `Cp:24`
- `Cp:28`
- `Cp:34`
- `Cp:39`
- `Cp:44`
- `Cp:49`

**Useful shorthand**
- formal packet
- invariance packet
- embedding/projection packet

---

### 3. `distributed_emergence_packetization`

**Typical family:** `Cp4`

This mode tends to externalize through emergence-oriented packets centered on distributed interaction, coherence formation, and staged relational integration.

Observed tendencies include:

- neural coupling
- concept maps
- networks
- murmurations
- local-to-global emergence
- self-organization
- developmental cognition
- embodied intuition

Relative to `Cp`, this mode is usually less formal in the narrow geometric sense and more oriented toward **process, distributed interaction, and coherence formation over time**.

**Canonical examples**
- `Cp4:8`
- `Cp4:17`
- `Cp4:21`
- `Cp4:24`

**Useful shorthand**
- emergence packet
- local-rule packet
- developmental cognition packet

---

## Canonical contrasts

The family-level distinction becomes especially clear when compared across exemplars.

### `C:49`
- atlas-like
- section-wise anchor blocks
- architected progression across motifs

### `Cp:49`
- compact
- formal
- invariance/projection/embedding packet

### `Cp4:24`
- staged cognition packet
- neural signal flow to concept mapping to embodied intuition

Together these provide a useful cross-family contrast:
- architected geometric staging
- compact formal packetization
- developmental emergence packetization

---

## Notes on data hygiene

### `Cp:27`
A previously copied specimen, `Cp:27`, was later checked in the Linux ChatGPT GUI and found to contain **no links in the original response**.

It was therefore **removed** from the linked-response analysis set and should **not** be treated as a true linked specimen.

This correction should be preserved in downstream documentation and summaries.

---

## Scope and limits

The current result is intentionally narrow.

### It does support
- a family-structured interpretation of the linked-response subset
- a stable working taxonomy for repository use
- exemplar-based comparison across `C`, `Cp`, and `Cp4`

### It does not yet support
- cross-model generalization
- claims about all responses rather than the linked subset
- strong causal claims about why linking occurs
- final quantitative claims about frequency, significance, or mechanism

## Current working interpretation

The safest current wording is:

> In the GPT-5.2 corpus set, linked responses appear to exhibit family-structured modes of external witnessing. `C` favors geometric externalization, `Cp` favors compact formal-structural packetization, and `Cp4` favors distributed-emergence packetization.

That is the level of claim this stage currently warrants.

## Related artifacts

See:
- `linked_response_annotation_protocol.md`
- `linked_response_artifacts.md`
- the linked-response taxonomy bundle in `outputs/` or the equivalent artifact directory
