Documentation

This directory contains the canonical reference surface for the PAM Observatory, along with project history, interface docs, planning notes, and research provenance.

Not all documents play the same role. The sections below indicate which materials should be treated as canonical reference, which are active project documentation, and which are planning or historical records.

Canonical reference docs

These documents define the current repository-facing scientific and architectural surface.

* architecture.md — high-level repository architecture
* observatory.md — observatory-level framing
* abstract.md — compact project summary
* observatory_philosophy.md — observatory design stance
* research_log.md — canonical observatory log surface
* 01_observatory/ — how to read the observatory and its core terms
* 02_geometry/ — geometry-layer reference docs
* 03_pipeline/ — phase, operators, topology, and pipeline-facing docs
* 05_project/ canonical specs — family layer, normalization, gateway, and response-flow reference docs

These are the best starting points for understanding the current observatory.

Project and research-arc docs

These documents capture the evolving scientific program, stabilization work, and current canonical project objects.

* 05_project/README.md
* 05_project/canonical_event_family_classification_spec.md
* 05_project/canonical_event_normalization_contract.md
* 05_project/canonical_family_gateway_spec.md
* 05_project/canonical_family_layer_status.md
* 05_project/canonical_response_guided_flow.md

These should be read as the current project-facing scientific reference layer.

Interface docs

These documents describe the operational observatory surface and interface-related design work.

* 04_interface/README.md
* 04_interface/observatory_tui.md

Additional interface planning and historical design notes may also live in 04_interface/, but not all interface docs are canonical reference surfaces.

Planning and historical design docs

Some documents are intentionally retained as planning records, implementation notes, or historical stabilization plans.

Examples include:

* canonicalization plans
* stabilization plans
* interface implementation plans

These documents are useful for understanding how the observatory evolved, but they should not automatically be treated as the current canonical reference surface unless stated explicitly.

Provenance and notes

Some documents are kept as research provenance, conceptual notes, or supporting context.

Examples include:

* conversation excerpts
* notes
* vision or framing texts
* prompt/tooling notes

These belong in the repository because they preserve development context, but they are secondary to the canonical reference and project docs.

Reading order

A good reading path is:

1. README.md
2. architecture.md
3. observatory.md
4. research_log.md
5. 01_observatory/
6. 02_geometry/
7. 03_pipeline/
8. 05_project/
9. 04_interface/

This order moves from high-level framing to observatory mechanics, then to project-specific canonical layers, and finally to the operational interface.

Status notes

When in doubt:

* treat src/pam/ as the canonical implementation layer
* treat docs/ as the canonical explanatory layer
* treat experiments/ as reproducible study and entrypoint history unless explicitly promoted
* treat planning docs as historical or roadmap material unless they are marked as current reference