OBS-082 — RIG Intervention-Readiness Audit

State

OBS-082 audits the OBS-081 Reusable Invariance Registry for intervention-readiness.

The audit does not perform interventions. It does not establish control, causality, causal sufficiency, or external generalization. It asks a narrower artifact-scoped question:

Which OBS-081 relation × carrier records are mature enough to define conservative, testable intervention hypotheses within the current PAM artifact lineage?

Artifact lineage

OBS-082 follows the registry layer established in OBS-081.

The immediate artifact chain is:

OBS-078:
  local stability signature is classifier-visible, compressible, and localizable
OBS-079:
  the 3-feature local stability core survives robustness checks
OBS-080:
  the stability core survives multiple measurement-contract perturbations
OBS-081:
  reusable-invariance records are registered over relation × carrier entries
OBS-082:
  registered reusable-invariance records are audited for intervention-readiness

The compact stability core remains:

mean_lambda_local_mean
mean_delta_d_mean
bounded_share_mean

The current canonical regime interpretation remains:

C   = bounded stability
Cp2 = high-divergence / low-boundedness sorting
Cp3 = high-displacement / low-boundedness instability-settlement

Inputs

OBS-082 reads the OBS-081 registry artifacts:

outputs/rig_registry/rig_relation_registry.csv
outputs/rig_registry/rig_survival_matrix.csv
outputs/rig_registry/rig_failure_localization.csv
outputs/rig_registry/rig_geometry_needed_ladder.csv
outputs/rig_registry/rig_repair_recommendations.csv

The audit output directory is:

outputs/rig_registry/obs082_intervention_readiness/

Generated outputs:

obs082_input_manifest.csv
obs082_relation_readiness_scores.csv
obs082_candidate_intervention_hypotheses.csv
obs082_negative_control_contrasts.csv
obs082_failure_mode_inventory.csv
obs082_blockers.csv
obs082_report.md

Scoring dimensions

OBS-082 computes a composite readiness score over six dimensions:

Dimension	Meaning
Invariance strength	How strongly the relation survives OBS-080-tested contract perturbations
Failure localization	Whether failures are localized rather than diffuse
Repair specificity	Whether the registry provides concrete next-test or repair directions
Geometry sufficiency	Whether the geometry requirement is known and operationally sufficient
Carrier convergence	Whether multiple carriers support the relation
Negative-control contrast	Whether the record is clearly stronger than weak or context-fragile controls

The scoring weights are:

0.25 invariance_strength_score
0.15 failure_localization_score
0.15 repair_specificity_score
0.15 geometry_sufficiency_score
0.20 carrier_convergence_score
0.10 negative_control_contrast_score

rig_status is treated as metadata. It is not the primary score basis. If invariance strength can only be inferred from categorical rig_status, readiness class cannot exceed candidate-ready.

Readiness classes

OBS-082 distinguishes:

A: hypothesis-ready
B: candidate-ready
C: diagnostic-only
D: registry-only
X: blocked / insufficient artifact support

For Class A, a record must have strong invariance, carrier convergence, negative-control contrast, no fatal blockers, and either localized failure evidence or specific repair evidence.

For Class B, a record must still clear a negative-control contrast threshold. This prevents high-survival descriptive invariants from being promoted to candidate intervention hypotheses without adequate contrast.

Class C means the record remains useful as diagnostic registry evidence but is not yet ready to define a testable intervention hypothesis.

Result summary

OBS-082 produced a conservative result:

Scoreable records:     24 / 24
Blocked records:        0
Class A records:        0
Class B records:        0
Class C records:       24
Class D records:        0
Class X records:        0

All 24 records were scored from direct OBS-080d evidence:

score_basis = obs080d_carrier_mean_ba

This means the audit did not collapse into categorical registry-status restatement.

Main finding

OBS-082 shows that the OBS-081 registry is intervention-adjacent but not intervention-ready.

The registry records are strong as diagnostic reusable-invariance evidence:

invariance_strength_score: high
carrier_convergence_score: high
geometry_sufficiency_score: high

However, none reach hypothesis-ready or candidate-ready status under strict readiness criteria because the limiting dimensions are systematic:

weak_negative_control_contrast: 24 / 24
generic_repair_specificity:     24 / 24
diffuse_failure_localization:   23 / 24

Therefore, the current registry supports diagnosis and theory grounding, but not yet intervention-hypothesis selection.

Interpretation

The important OBS-082 distinction is:

registered reusable invariance ≠ intervention-ready invariance

OBS-081 established that reusable-invariance records can be created.

OBS-082 shows that registered invariance is not automatically actionable. To become intervention-ready, a record needs more than contract survival. It also needs:

specific contrast against negative controls,
localized failure structure,
and concrete repair or enrichment direction.

This gives RIG a useful maturity ladder:

Level 1: invariant
  a relation survives perturbation
Level 2: diagnostic invariant
  a relation helps describe regime structure
Level 3: actionable invariant
  a relation has enough localization, contrast, and repair specificity
  to define a testable intervention hypothesis

Within the current artifact lineage, OBS-082 places the OBS-081 registry at Level 2.

What OBS-082 does not show

OBS-082 does not show that interventions work.

OBS-082 does not show that PAM transitions are controllable.

OBS-082 does not establish causal sufficiency.

OBS-082 does not establish external generalization.

OBS-082 does not prove that the registered invariants are universal.

Its result is narrower:

The registry is scoreable, diagnostic, and intervention-adjacent,
but no record is yet hypothesis-ready under strict readiness criteria.

Consequence for the next research step

OBS-082 points to a clear next research branch:

OBS-083 — RIG Negative-Control and Failure-Localization Strengthening

Candidate question:

Can the diagnostic-only OBS-082 records be separated into candidate-ready versus registry-only records by constructing stronger matched negative controls and sharper failure-localization evidence?

Likely focus areas:

matched negative controls
relation-specific control groups
carrier-specific contrast tests
contract-family failure localization
repair recommendation sharpening

The next step should not be intervention execution. It should strengthen the evidence needed to make intervention hypotheses legitimate.
