# PAM Documentation

This directory is the canonical explanatory and navigational surface of the
PAM Observatory.

It connects four distinct repository layers:

1. reusable runtime implementation;
2. numbered scientific studies;
3. frozen protocols and generated evidence artifacts;
4. historical, planning, interface, and provenance records.

These layers do not have equal authority. A planning document does not
establish a result. A generated report does not automatically define a reusable
runtime contract. A numbered study can be scientifically canonical without
being promoted into `src/pam/`.

This index therefore has two purposes:

- direct readers to the correct material for a given question;
- make the scientific and operational status of that material explicit.

For the public-facing research overview, begin with [`../README.md`](../README.md).
For the chronological observatory record, use [`research_log.md`](research_log.md).

---

## Current Research Position

PAM has progressed from manifold and seam reconstruction through multiscale
transition analysis to reusable-invariance testing, blinded confirmation, and
detection-attainability studies.

The present evidence spine is organized around OBS-081–085:

- the Reusable Invariance Registry is operational within the tested artifact
  lineage;
- the registered relations remain diagnostic rather than intervention-ready;
- OBS-084 completed a fully frozen discovery–confirmation protocol and
  established no FL3 direct failure witness;
- OBS-085 is evaluating structural evidence feasibility, simulator
  qualification, conditional gate passage, and prospective campaign
  attainability without changing completed results retrospectively.

The current evidential boundary is therefore:

| Question | Current position |
|---|---|
| Is the foundational geometric and phase instrument implemented? | Yes. |
| Are compact stability relations present within tested contracts? | Yes, within the declared C/Cp2/Cp3 artifact lineage. |
| Is the Reusable Invariance Registry operational? | Yes. |
| Are any RIG records intervention-ready? | No. |
| Did OBS-084 confirm an FL3 direct failure witness? | No; 0 confirmed FL3 witnesses. |
| Did OBS-085b produce complete gate passage? | No; 0 passages across 600,000 replicates. |
| Why did OBS-085b not pass? | Effective independent-cluster support never exceeded 3, so the minimum exact one-sided sign-flip p-value was 0.125, above the frozen alpha of 0.10. |
| What does OBS-085c test? | Whether prospectively expanded independent-object support makes the frozen evidence contract attainable and reliably passable. |

OBS-085c is a prospective design study. It does not revise OBS-085b, increase
claim entitlement, or reinterpret the completed four-object campaign as though
it contained additional independent support.

---

## Repository Authority Model

Different questions require different authoritative surfaces.

| Surface | Primary authority |
|---|---|
| `src/pam/` | Reusable runtime implementation and package-level contracts |
| `scripts/canonical/` | Stabilized downstream execution and canonicalization layers |
| `experiments/studies/` | Numbered scientific instruments, audits, and frozen study implementations |
| `docs/` | Explanatory, architectural, conceptual, and protocol documentation |
| `docs/research_log.md` | Observatory-level chronology and concise scientific synthesis |
| `outputs/` | Generated data products, evidence tables, manifests, reports, and audit artifacts |
| Git commits | Frozen repository-lineage anchors |

### Study-level source of truth

For a numbered study, read the evidence chain in this order:

1. **Frozen protocol or design document** — what was declared before result
   inspection.
2. **Study script at the frozen commit** — what was actually implemented.
3. **Input manifest and validated upstream hashes** — which artifacts entered
   the study.
4. **Generated manifest** — what was executed and which outputs were written.
5. **Generated tables and report** — what the run established under its frozen
   contract.
6. **Research-log entry and README summaries** — how the result fits into the
   wider program.

Broad summaries must not override a study-specific manifest, report, or frozen
protocol. When two descriptions appear inconsistent, prefer the narrower,
artifact-grounded source and inspect its repository lineage.

---

## The Three Scientific Spines

The repository is now best understood as three coupled spines.

### 1. Generative spine

Produces runs, evolving corpora, and observable material.

Primary surfaces include:

- corpus-generation and campaign code;
- source and prompt provenance;
- run manifests;
- raw and cleaned observable tables.

See [`architecture/generative_spine.md`](architecture/generative_spine.md)
and the corpus/provenance material in [`05_project/`](05_project/).

### 2. Observatory spine

Turns generated material into geometric and structural artifact layers.

It includes:

- Fisher-information geometry;
- graph distances and embeddings;
- phase, seam, horizon, and Lazarus observables;
- identity transport and obstruction;
- route families and transition algebra;
- response fields, operators, and flow;
- scale-space and multiscale persistence.

See [`architecture/observatory_spine.md`](architecture/observatory_spine.md),
[`02_geometry/`](02_geometry/), [`03_pipeline/`](03_pipeline/), and the earlier
sections of [`research_log.md`](research_log.md).

### 3. Evidence spine

Regulates what the observatory is entitled to claim.

It includes:

- robustness and alternate-contract testing;
- compact stability signatures;
- reusable-invariance registration;
- negative controls and failure localization;
- blinded discovery and reserved confirmation;
- structural evidence feasibility;
- simulator qualification;
- gate-passage and attainability studies.

Primary artifact roots include:

```text
outputs/rig_registry/
outputs/rig_registry/obs085_detection_envelope/
```

Primary implementation surfaces include the OBS-076–085 studies under:

```text
experiments/studies/
```

The evidence spine is not merely another output layer. It determines whether a
structural observation remains descriptive, becomes a reusable diagnostic, is
supported by direct witnesses, or qualifies for stronger claims.

---

## Documentation Map

### Top-level reference documents

- [`architecture.md`](architecture.md) — high-level repository architecture.
- [`observatory.md`](observatory.md) — observatory framing and scientific role.
- [`abstract.md`](abstract.md) — compact project summary.
- [`observatory_philosophy.md`](observatory_philosophy.md) — design stance and
  methodological orientation.
- [`research_log.md`](research_log.md) — canonical observatory chronology.

### `01_observatory/`

How to read the observatory, its terms, and its operating model.

Use this layer for orientation before interpreting study-specific outputs.

### `02_geometry/`

Geometry-layer reference material:

- Fisher metric and dissimilarity structure;
- graph and geodesic constructions;
- embeddings;
- curvature and seam geometry;
- geometric diagnostics and limitations.

### `03_pipeline/`

Pipeline-facing documentation for phase, operators, topology-like relational
objects, and downstream execution layers.

### `04_interface/`

Operational interface and observatory-navigation documentation.

Key entry points include:

- [`04_interface/README.md`](04_interface/README.md)
- [`04_interface/observatory_tui.md`](04_interface/observatory_tui.md)

Interface plans and historical design notes in this folder are not
automatically canonical runtime references.

### `05_project/`

Project-level scientific documentation, stabilization contracts, protocols,
working notes, and research-arc consolidations.

This folder now spans several generations of the program:

- canonical family and gateway contracts;
- response-guided flow;
- linked-response taxonomy and annotation;
- corpus provenance and affordance structure;
- Cp2/Cp3 controls;
- scale-space and structural-persistence studies;
- RIG readiness and direct-support protocols;
- OBS-085 detection-feasibility and attainability design.

Important current-frontier documents include:

- [`05_project/076a_observable_scale_space_substrate.md`](05_project/076a_observable_scale_space_substrate.md)
- [`05_project/076b_observable_space_geometry_rebuild.md`](05_project/076b_observable_space_geometry_rebuild.md)
- [`05_project/076c_structural_object_persistence.md`](05_project/076c_structural_object_persistence.md)
- [`05_project/077b_path_label_projection.md`](05_project/077b_path_label_projection.md)
- [`05_project/077c_window_local_divergence_bridge.md`](05_project/077c_window_local_divergence_bridge.md)
- [`05_project/obs080b_stability_core_scale_band_sensitivity.md`](05_project/obs080b_stability_core_scale_band_sensitivity.md)
- [`05_project/082_rig_intervention_readiness_audit.md`](05_project/082_rig_intervention_readiness_audit.md)
- [`05_project/084_rig_direct_failure_support_witness_protocol.md`](05_project/084_rig_direct_failure_support_witness_protocol.md)
- [`05_project/085_failure_support_detection_power_and_confirmation_feasibility_protocol.md`](05_project/085_failure_support_detection_power_and_confirmation_feasibility_protocol.md)

Because `05_project/` contains both current contracts and historical planning
material, always inspect the document status and compare it with the relevant
study manifest.

### `architecture/`

Cross-cutting architecture documents, including the generative and observatory
spines. This layer explains how scientific objects move through the repository,
not merely how Python modules import one another.

### `concepts/`

Concept-level references for major observatory objects.

Key documents include:

- [`concepts/tip.md`](concepts/tip.md) — TIP as a first-order invariant
  measurement instrument;
- [`concepts/tim.md`](concepts/tim.md) — TIM as a second-order
  transformation-stability instrument built on TIP;
- [`concepts/topological_identity.md`](concepts/topological_identity.md) — the
  relational identity, transport, and obstruction program.

Concept documents explain intended scientific meaning. Study-specific
operational definitions and generated manifests remain authoritative for actual
measurements.

### `figures/`

Repository-facing visual artifacts.

A figure is not self-authenticating evidence. Its authority depends on:

- the generating script;
- its input artifacts;
- the associated manifest or report;
- and whether it is illustrative, diagnostic, or result-bearing.

### `prompts/`

Prompt instruments and prompt provenance.

Prompts may be experimental inputs, corpus sources, controls, or historical
records. They are not scientific conclusions unless incorporated into a frozen
study contract with declared identity, role, and provenance.

### Notes and provenance material

Files such as `conversation_excerpts.md`, `notes.md`, `allspark.md`, working
notes, and vision texts preserve development context. They may be valuable for
reconstructing the research trajectory, but they are secondary to frozen
protocols, scripts, manifests, and generated evidence.

---

## Research-Arc Index

This table is a navigation aid, not a replacement for the research log.

| Research arc | OBS range | Main question | Primary surfaces |
|---|---:|---|---|
| Foundational manifold, phase, identity, and seam | 001–027 | What geometric and relational organization is present? | `02_geometry/`, `03_pipeline/`, early study scripts, `research_log.md` |
| Embedding policy, transition algebra, route families, and dynamical structure | 028–051 | How does seam-mediated organization compose, route, and flow? | `05_project/`, `scripts/canonical/`, numbered studies |
| Corpus, affordance, linked-response, and control development | 052–075 | How do corpus and prompt conditions alter the observatory substrate? | `05_project/`, provenance docs, comparison studies |
| Multiscale stability | 076–080 | Which structural distinctions persist across scale, contracts, and resampling? | OBS-076–080 docs, studies, comparison outputs |
| Reusable invariance and diagnostic readiness | 081–083 | Which stable relations can be registered, contrasted, and localized? | `outputs/rig_registry/`, RIG studies and reports |
| Blinded direct-support evaluation | 084 | Do sealed discovery candidates survive reserved confirmation as direct witnesses? | frozen OBS-084 protocol, discovery and confirmation artifacts |
| Detection feasibility and campaign attainability | 085 | Are evidence gates estimable and attainable under qualified simulators and prospective independent support? | OBS-085 protocol, studies, and `outputs/rig_registry/obs085_detection_envelope/` |

For exact claims, dates, states, and operational consequences, consult
[`research_log.md`](research_log.md) and the study-specific generated report.

---

## Reading Paths by Intent

### Fast project orientation

1. [`../README.md`](../README.md)
2. [`abstract.md`](abstract.md)
3. [`observatory.md`](observatory.md)
4. [`architecture.md`](architecture.md)
5. the latest relevant entries in [`research_log.md`](research_log.md)

### Understand the runtime and repository architecture

1. [`architecture.md`](architecture.md)
2. [`architecture/generative_spine.md`](architecture/generative_spine.md)
3. [`architecture/observatory_spine.md`](architecture/observatory_spine.md)
4. [`03_pipeline/`](03_pipeline/)
5. `src/pam/`
6. `scripts/canonical/`

### Understand the foundational geometry

1. [`01_observatory/`](01_observatory/)
2. [`02_geometry/`](02_geometry/)
3. [`concepts/`](concepts/)
4. OBS-001–027 in [`research_log.md`](research_log.md)
5. corresponding scripts under `experiments/studies/`

### Understand the current evidence frontier

1. OBS-076 onward in [`research_log.md`](research_log.md)
2. the OBS-076–080 documents in [`05_project/`](05_project/)
3. [`05_project/082_rig_intervention_readiness_audit.md`](05_project/082_rig_intervention_readiness_audit.md)
4. [`05_project/084_rig_direct_failure_support_witness_protocol.md`](05_project/084_rig_direct_failure_support_witness_protocol.md)
5. [`05_project/085_failure_support_detection_power_and_confirmation_feasibility_protocol.md`](05_project/085_failure_support_detection_power_and_confirmation_feasibility_protocol.md)
6. OBS-081–085 scripts under `experiments/studies/`
7. manifests and reports under `outputs/rig_registry/`

### Reproduce or audit a numbered study

1. identify the numbered study and exact repository commit;
2. read its protocol or project document;
3. inspect the study script and CLI validation modes;
4. inspect the input manifest and upstream artifact hashes;
5. run `--self-test` and `--validate-only` when supported;
6. reproduce the declared smoke or full campaign;
7. inspect the generated manifest before reading aggregate conclusions;
8. compare the report with the raw summary and failure tables;
9. preserve the resulting commit and output lineage.

### Reconstruct project history

Use:

- [`research_log.md`](research_log.md);
- observatory-chain summaries in [`05_project/`](05_project/);
- `program_state_reflection.md`;
- planning and canonicalization documents;
- notes and conversation excerpts;
- Git history.

Historical material should explain how the program evolved, not silently define
current behavior.

---

## Document Status Vocabulary

Use these labels consistently when adding or revising documentation.

### Canonical reference

Defines the current explanatory, architectural, or runtime-facing contract.

### Frozen protocol

Defines a study before result inspection. It must not be altered
retrospectively to accommodate observed outcomes.

### Study implementation

Executable numbered instrument that operationalizes a protocol or research
question. It may be scientifically canonical without being part of the reusable
runtime package.

### Generated manifest

Machine-written execution identity containing validated inputs, parameters,
output hashes, lineage anchors, counts, and completion state.

### Generated report

Machine-written or artifact-grounded interpretation of an executed study. Its
scope is limited by the associated protocol, manifest, simulator family, and
input lineage.

### Diagnostic registry

Structured evidence object that records reusable relations, carriers,
contracts, limits, and readiness status. Registry inclusion does not imply
causal control or intervention readiness.

### Current project reference

Repository-facing synthesis of the latest supported interpretation for an
active scientific arc.

### Working note

Exploratory reasoning or provisional interpretation not yet promoted to a
frozen protocol or canonical reference.

### Planning document

Describes intended work. It does not establish implementation, execution, or
scientific evidence.

### Historical or superseded document

Retained for provenance. It should be marked clearly and must not silently
compete with current references.

---

## Evidence and Claim Discipline

Unless a study explicitly establishes otherwise, PAM results are:

- model-specific;
- corpus-specific;
- prompt- and source-specific where applicable;
- artifact-specific;
- contract-specific;
- conditional on declared preprocessing and dependence units;
- observational, comparative, or diagnostic rather than causal;
- not evidence of universal semantic topology;
- not evidence of intervention readiness;
- not evidence that a simulated mechanism caused an observed result.

Operational terms such as *topology*, *invariance*, *witness*, *power*,
*failure support*, and *attainability* must be interpreted according to the
study that defines them.

In particular:

- empirical relational or transport structure is not automatically a formal
  topological theorem;
- robustness across tested contracts is not universal invariance;
- a discovery candidate is not a confirmed witness;
- structural evidence feasibility is not effect existence;
- simulator qualification is not simulator truth;
- conditional gate-passage probability is not classical power unless a study
  explicitly establishes that equivalence;
- prospective attainability does not revise a completed historical campaign.

Study-specific reports and manifests override broad summaries.

---

## Generated Artifacts and Provenance

Generated outputs are part of the scientific instrument, not disposable build
products.

A mature evidence artifact should be traceable through:

```text
source identity
    -> preprocessing contract
    -> input artifact hash
    -> study script and commit
    -> frozen parameters
    -> generated manifest
    -> output hashes
    -> report and research-log synthesis
```

Important artifact families include:

```text
outputs/canonical/
outputs/rig_registry/
outputs/rig_registry/obs085_detection_envelope/
```

Do not infer artifact authority from filename, modification time, or directory
placement alone. Confirm the associated manifest and content lineage.

Large replicate-level artifacts may be retained when they are necessary for
independent audit, but summary tables and reports must remain derivable from the
frozen replicate contract.

---

## Runtime, Study, and Evidence Reproducibility

The repository has three distinct reproducibility surfaces.

### Runtime instrument

The packaged geometry, phase, operator, and topology-facing pipeline is rooted
in:

```text
src/pam/
scripts/run_full_pipeline.sh
scripts/canonical/
```

### Numbered observatory studies

Study implementations live primarily under:

```text
experiments/studies/
```

Each mature study should declare:

- expected inputs;
- output root;
- deterministic seeds or seed derivation;
- scope and guardrails;
- validation and self-test modes where practical;
- completion state;
- manifest and report outputs.

### Frozen evidence campaigns

Later evidence studies additionally validate:

- upstream manifest identities;
- artifact hashes;
- source and carrier identities;
- discovery/confirmation partition roles;
- simulator authorization;
- gate contracts;
- repository ancestry or frozen commit anchors.

This stricter layer is essential for OBS-084 and OBS-085.

---

## Documentation Maintenance Rules

When updating the documentation tree:

1. **Preserve status.** Mark a file as canonical, frozen, generated, working,
   planning, historical, or superseded.
2. **Do not rewrite history.** Completed study protocols and reports should not
   be silently edited to fit later findings.
3. **Prefer links over duplication.** Keep exact counts, hashes, thresholds, and
   commit identities in manifests and reports when they are likely to change.
4. **Update navigation when the research frontier changes.** A new arc that
   changes how the repository should be read belongs in this index.
5. **Update the research log after freezing a study.** The log should synthesize
   the result without replacing the underlying evidence.
6. **Separate runtime promotion from scientific maturity.** A study can be
   scientifically established at first pass while remaining outside `src/pam/`.
7. **Keep active-run state out of durable references.** Machine names, transient
   progress, and projected completion times belong in logs, not canonical docs.
8. **Preserve explicit limitations.** Model-specific, subset-specific,
   provisional, corrected, and provenance-patched scope must survive summaries.
9. **Avoid unsupported escalation.** Do not promote diagnostic evidence to
   causal, actionable, universal, or formally topological claims without a
   study that establishes that entitlement.

---

## Practical Rule of Thumb

When in doubt:

- use [`../README.md`](../README.md) for the public project overview;
- use this file to find the correct documentation and authority layer;
- use [`research_log.md`](research_log.md) for observatory chronology;
- use `src/pam/` for reusable runtime behavior;
- use `experiments/studies/` for numbered scientific implementations;
- use frozen protocols to understand what a study intended to test;
- use generated manifests and reports to understand what actually ran;
- use `outputs/` as evidence only after validating provenance;
- use planning and historical notes to understand evolution, not current truth.

The documentation tree is intentionally cumulative. Its purpose is not to erase
earlier stages, but to make their role, authority, and relationship to the
current research state explicit.
