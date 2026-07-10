# OBS-083 — RIG Negative-Control and Failure-Localization Strengthening

## State

Diagnostic subclassing audit completed with v2 conservative prior-aware gates.

OBS-083 is a diagnostic refinement audit only: no interventions performed; no causality/control/actionability claims; C4 remains diagnostic-only. v2 uses OBS-082 diffuse/generic priors and evidence-completeness gates.

## Scope

- Input registry directory: `outputs/rig_registry`
- Output directory: `outputs/rig_registry/obs083_negative_control_localization`
- Uses OBS-081/OBS-082 artifacts when available.
- Uses conservative fallbacks when optional OBS-080/contract columns are missing.
- Missing evidence is recorded as missing and is never counted as positive evidence.

## Inputs

| artifact_label                           | exists   |   rows | read_status    |
|:-----------------------------------------|:---------|-------:|:---------------|
| obs081_input_manifest                    | True     |      4 | ok             |
| obs081_relation_registry                 | True     |     24 | ok             |
| obs081_survival_matrix                   | True     |    256 | ok             |
| obs081_failure_localization              | True     |     29 | ok             |
| obs081_geometry_needed_ladder            | True     |     24 | ok             |
| obs081_repair_recommendations            | True     |     24 | ok             |
| obs081_registry_report                   | True     |    nan | exists_non_csv |
| obs082_input_manifest                    | True     |      5 | ok             |
| obs082_readiness_scores                  | True     |     24 | ok             |
| obs082_negative_control_contrasts        | True     |     48 | ok             |
| obs082_failure_mode_inventory            | True     |     29 | ok             |
| obs082_blockers                          | True     |      0 | ok             |
| obs082_candidate_intervention_hypotheses | True     |      0 | ok             |
| obs082_report                            | True     |    nan | exists_non_csv |

## Thresholds

| threshold | value |
|---|---:|
| strong_delta | 0.100 |
| moderate_delta | 0.050 |
| contrast_threshold | 0.500 |
| localization_threshold | 0.500 |
| repair_threshold | 0.500 |
| c4_contrast | 0.600 |
| c4_localization | 0.600 |
| c4_repair | 0.500 |
| survival_floor | 0.450 |

## Method summary

OBS-083 refines the OBS-082 Class C result by constructing conservative prior-aware diagnostics:

1. matched negative-control design rows;
2. relation-control contrasts;
3. carrier-control contrasts;
4. contract/transformation control contrasts where artifact columns exist;
5. geometry-needed control contrasts;
6. failure-localization matrix that treats OBS-082 diffuse localization as the prior;
7. repair-specificity sharpening table gated by direct locus, metric, and matched-control evidence;
8. C0–C4 diagnostic subclass assignments.

C4 is explicitly retained as **diagnostic-only / promising next-test candidate**. It is not candidate-ready, actionable, causal, or intervention-ready. In v2, C4 additionally requires direct localization evidence, R4/R5 repair specificity, a candidate metric, and a matched negative control for the repair claim.

## Subclass counts

| subclass                |   count |
|:------------------------|--------:|
| C1_contrast-limited     |      12 |
| C2_localization-limited |      12 |

## Diagnostic subclass assignments

| record_id                           | subclass                |   negative_control_strength_score |   failure_localization_score |   repair_specificity_score | primary_limiter           |
|:------------------------------------|:------------------------|----------------------------------:|-----------------------------:|---------------------------:|:--------------------------|
| C_vs_Cp2__stability_core_3          | C2_localization-limited |                          0.567141 |                         0.35 |                       0.55 | failure_localization      |
| C_vs_Cp2__geometry_scores_only      | C1_contrast-limited     |                          0.437864 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp2__no_window                 | C1_contrast-limited     |                          0.316278 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp2__stability_plus_geometry   | C1_contrast-limited     |                          0.348302 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp2__strict_numeric_all        | C1_contrast-limited     |                          0.316333 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp2__path_shares_only          | C2_localization-limited |                          0.554407 |                         0.35 |                       0.55 | failure_localization      |
| C_vs_Cp3__stability_core_3          | C2_localization-limited |                          0.510174 |                         0.35 |                       0.55 | failure_localization      |
| C_vs_Cp3__no_window                 | C1_contrast-limited     |                          0.257413 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp3__stability_plus_geometry   | C1_contrast-limited     |                          0.244568 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp3__strict_numeric_all        | C1_contrast-limited     |                          0.257469 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp3__geometry_scores_only      | C1_contrast-limited     |                          0.326343 |                         0.35 |                       0.55 | negative_control_contrast |
| C_vs_Cp3__path_shares_only          | C2_localization-limited |                          0.544305 |                         0.35 |                       0.55 | failure_localization      |
| Cp2_vs_Cp3__geometry_scores_only    | C2_localization-limited |                          0.664922 |                         0.35 |                       0.55 | failure_localization      |
| Cp2_vs_Cp3__no_window               | C1_contrast-limited     |                          0.450304 |                         0.35 |                       0.55 | negative_control_contrast |
| Cp2_vs_Cp3__path_shares_only        | C2_localization-limited |                          0.686927 |                         0.35 |                       0.55 | failure_localization      |
| Cp2_vs_Cp3__stability_plus_geometry | C2_localization-limited |                          0.506592 |                         0.35 |                       0.55 | failure_localization      |
| Cp2_vs_Cp3__strict_numeric_all      | C1_contrast-limited     |                          0.45036  |                         0.35 |                       0.55 | negative_control_contrast |
| Cp2_vs_Cp3__stability_core_3        | C2_localization-limited |                          0.746711 |                         0.35 |                       0.55 | failure_localization      |
| three_way__geometry_scores_only     | C2_localization-limited |                          0.664922 |                         0.35 |                       0.55 | failure_localization      |
| three_way__no_window                | C1_contrast-limited     |                          0.450304 |                         0.35 |                       0.55 | negative_control_contrast |
| three_way__stability_plus_geometry  | C2_localization-limited |                          0.506592 |                         0.35 |                       0.55 | failure_localization      |
| three_way__strict_numeric_all       | C1_contrast-limited     |                          0.45036  |                         0.35 |                       0.55 | negative_control_contrast |
| three_way__path_shares_only         | C2_localization-limited |                          0.75     |                         0.35 |                       0.55 | failure_localization      |
| three_way__stability_core_3         | C2_localization-limited |                          0.75     |                         0.35 |                       0.55 | failure_localization      |

## Relation-control evidence

- Rows written: 72
- Evidence-available rows: 72

Relation-control contrast tests whether a target relation differs from matched relations under the same carrier. Weak relation contrast remains a negative-control limiter.

## Carrier-control evidence

- Rows written: 120
- Evidence-available rows: 120

Carrier-control contrast tests whether a carrier has a differentiated role rather than acting as an overbroad separability substrate.

## Failure-localization evidence

- Rows written: 24
- High/moderate localization rows: 0
- Direct artifact locus rows: 0

Failure localization is interpreted as diagnostic addressability only. It is not a causal mechanism or repair target unless future criteria are met.

## Repair-specificity evidence

- Rows written: 24
- R4 diagnostic repair-candidate annotations: 0
- R3 relation+carrier-specific annotations: 24
- Hypothesis-ready rows: 0

OBS-083 deliberately does not promote repair annotations to actionability. R3 means relation+carrier-specific annotation; R4 requires direct locus plus metric evidence and still is not a validated repair hypothesis.

## Canonical result statement

OBS-083 refines the OBS-082 diagnostic-only registry by constructing matched relation, carrier, contract, geometry-needed, and failure-localization contrasts over the OBS-081 relation × carrier records. The audit assigns each record to a diagnostic subclass: C0 descriptive-only, C1 contrast-limited, C2 localization-limited, C3 repair-specificity-limited, or C4 promising next-test candidate. OBS-083 performs no interventions and establishes no causality, control, actionability, external generalization, or formal topology.

## Outputs

- `obs083_input_manifest.csv`
- `obs083_matched_negative_control_design.csv`
- `obs083_relation_control_contrast.csv`
- `obs083_carrier_control_contrast.csv`
- `obs083_contract_control_contrast.csv`
- `obs083_geometry_needed_control_contrast.csv`
- `obs083_failure_localization_matrix.csv`
- `obs083_repair_specificity_sharpening.csv`
- `obs083_diagnostic_subclass_assignments.csv`
- `obs083_readiness_delta_from_obs082.csv`
- `obs083_blocker_refinement.csv`
- `obs083_report.md`

## Limitations

- Optional contract-family evidence is only scored when available in the loaded artifacts.
- Missing evidence is not imputed.
- Contrast-derived localization is capped as a conservative proxy and remains diagnostic; it cannot by itself overcome the OBS-082 diffuse-localization prior.
- C4 remains within Class C unless a separate readiness audit proves otherwise.
