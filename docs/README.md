# Documentation

This directory contains the canonical explanatory surface of the PAM repository, along with project history, interface docs, planning notes, observatory notes, and research provenance.

The repository now has a fairly rich documentation tree. Not all documents play the same role, so the sections below indicate which materials should be treated as canonical reference, which are project-facing scientific docs, and which are planning or historical records.

## Canonical reference docs

These documents define the current repository-facing scientific and architectural surface.

- `architecture.md` — high-level repository architecture
- `observatory.md` — observatory-level framing
- `abstract.md` — compact project summary
- `observatory_philosophy.md` — observatory design stance
- `research_log.md` — canonical observatory log surface

Subsystem reference layers:

- `01_observatory/` — how to read the observatory and its core terms
- `02_geometry/` — geometry-layer reference docs
- `03_pipeline/` — phase, operators, topology, and pipeline-facing docs
- `architecture/` — cross-cutting architecture docs such as:
  - `observatory_spine.md`
  - `generative_spine.md`

These are the best starting points for understanding the current observatory.

## Project and research-arc docs

These documents capture the evolving scientific program, stabilization work, canonical project objects, and repository-facing observatory-stage instruments.

Primary project-facing docs include:

- `05_project/README.md`
- `05_project/canonical_event_family_classification_spec.md`
- `05_project/canonical_event_normalization_contract.md`
- `05_project/canonical_family_gateway_spec.md`
- `05_project/canonical_family_layer_status.md`
- `05_project/canonical_response_guided_flow.md`

The `05_project/` section now also includes repository-facing observatory-stage docs for the linked-response study:

- `05_project/linked_response_family_taxonomy.md`
- `05_project/linked_response_annotation_protocol.md`
- `05_project/linked_response_artifacts.md`

These should be read as the current project-facing scientific reference layer for stabilized or stabilization-ready observatory objects.

## Concept docs

These documents explain major concept objects used across the repository.

- `concepts/tip.md` — TIP as the first-order invariant measurement instrument
- `concepts/tim.md` — TIM as the second-order transformation-stability instrument built on TIP
- `concepts/topological_identity.md` — the relational identity / transport / obstruction program

These files are especially useful when reading the corresponding code in `src/pam/`.

## Interface docs

These documents describe the operational observatory surface and interface-related design work.

- `04_interface/README.md`
- `04_interface/observatory_tui.md`

Additional interface planning and historical design notes may also live in `04_interface/`, but not all interface docs are canonical reference surfaces.

## Planning and historical design docs

Some documents are intentionally retained as planning records, implementation notes, or historical stabilization plans.

Examples include:

- canonicalization plans
- stabilization plans
- interface implementation plans
- historical roadmap notes

These documents are useful for understanding how the observatory evolved, but they should not automatically be treated as the current canonical reference surface unless stated explicitly.

## Provenance and notes

Some documents are kept as research provenance, observatory notes, conceptual notes, or supporting context.

Examples include:

- observatory log entries and stage notes
- conversation excerpts
- notes
- vision or framing texts
- prompt / tooling notes

These belong in the repository because they preserve development context, but they are secondary to the canonical reference and project docs unless explicitly promoted.

## Reading order

A good reading path is:

1. `README.md`
2. `architecture.md`
3. `observatory.md`
4. `research_log.md`
5. `01_observatory/`
6. `02_geometry/`
7. `03_pipeline/`
8. `architecture/`
9. `concepts/`
10. `05_project/`
11. `04_interface/`

This order moves from high-level framing to observatory mechanics, then to architecture and concept objects, then to project-specific canonical layers and observatory-stage instruments, and finally to the operational interface.

## Status notes

When in doubt:

- treat `src/pam/` as the canonical implementation layer
- treat `docs/` as the canonical explanatory layer
- treat `research_log.md` as the primary observatory-log surface
- treat `experiments/` as reproducible study and entrypoint history unless explicitly promoted
- treat planning docs as historical or roadmap material unless they are marked as current reference
- treat `05_project/` as the main home for project-facing stabilization docs and repository-facing observatory-stage consolidations

The repository’s scientific core is increasingly organized around a clean separation between:

- the **generative spine**, which produces runs and evolving corpora
- the **observatory spine**, which turns those outputs into stable scientific artifact layers

That split is now documented explicitly in `docs/architecture/`.

The repository also now includes a second important distinction inside the documentation layer:

- **runtime/canonical implementation-facing documents**
- **repository-facing observatory-stage instruments**, such as the linked-response taxonomy layer

That distinction matters when reading `05_project/`.

## Practical note

The documentation tree is still evolving.

The current aim is not to remove every historical or planning document, but to make document roles explicit enough that a reader can distinguish between:

- canonical reference
- project-facing scientific status
- observatory-stage instrument documentation
- interface practice
- planning history
- and research provenance

That is the standard this README is meant to support.