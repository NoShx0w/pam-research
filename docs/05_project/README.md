# Project

This section contains project-level documentation for the PAM Observatory.

It is the home for:

- stabilization plans
- canonicalization specs
- implementation roadmaps
- milestone and release notes
- repository maintenance procedures

## Current role

The repository has a canonical implemented runtime and architecture for the downstream observatory pipeline.

Primary entrypoints:

- `scripts/run_full_pipeline.sh`
- `src/pam/pipeline/runner.py`

Primary architecture reference:

- [`../architecture.md`](../architecture.md)

## Canonicalization and stabilization

The documents in this section may define **provisional canonical contracts** for parts of the research system that are scientifically mature but not yet fully consolidated in code.

These documents are intended to:

- make evolving observatory structures explicit,
- provide stable implementation targets,
- distinguish canonical direction from transitional study outputs,
- improve public-facing transparency as the repository catches up to the science.

## Documents

Current project-level documents include:

- [`OBS-015.md`](./OBS-015.md)
- [`pam_identity_transport_holonomy_stabilization_plan.md`](./pam_identity_transport_holonomy_stabilization_plan.md)

Additional canonicalization specs and implementation plans may also live here as they are introduced.

## Guidance

For current runtime and high-level structure, prefer:

- the repository root documentation
- [`../architecture.md`](../architecture.md)

For project-level stabilization, canonicalization, and implementation planning, prefer the documents in this section.