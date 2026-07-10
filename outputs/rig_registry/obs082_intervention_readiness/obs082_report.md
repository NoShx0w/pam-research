# OBS-082 — RIG Intervention-Readiness Audit

## 1. Scope

- within OBS-081 relation × carrier registry records
- within OBS-080-tested contract families
- within OBS-078a / OBS-079 / OBS-080 stability-core lineage
- within C / Cp2 / Cp3 regime comparisons where present
- within current repo-generated artifacts

- OBS-082 does not perform interventions.
- OBS-082 does not establish control.
- OBS-082 does not establish causality.
- OBS-082 does not establish external generalization.
- OBS-082 audits whether registry records are mature enough to define testable intervention hypotheses.

## 2. Inputs and artifact lineage

| input_name                 | input_path                                          | exists   |   rows | columns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | required_columns_present   | missing_columns   | used_for_dimensions                                                                         | notes   |
|:---------------------------|:----------------------------------------------------|:---------|-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------|:------------------|:--------------------------------------------------------------------------------------------|:--------|
| rig_relation_registry      | outputs/rig_registry/rig_relation_registry.csv      | True     |     24 | relation_id;task;carrier;carrier_role;carrier_role_detail;rig_status;rig_status_rank;threshold;baseline;obs080c_carrier_ba;obs080d_carrier_mean_ba;obs080d_carrier_min_ci95_low;obs080d_carrier_min_p_above_threshold;task_geometry_needed_level;task_geometry_needed_label;task_geometry_needed_rationale;failure_notes;repair_recommendation;repair_rationale;n_survival_rows;mean_survival_score;min_survival_score;numeric_transform_n;numeric_transform_mean_score;numeric_transform_min_score;numeric_transform_status;scale_band_n;scale_band_mean_score;scale_band_min_score;scale_band_status;feature_family_n;feature_family_mean_score;feature_family_min_score;feature_family_status;structural_resampling_n;structural_resampling_mean_score;structural_resampling_min_score;structural_resampling_status;level_1_stability_core_ba;level_3_geometry_ba;level_4_paths_ba;level_3_stability_plus_geometry_ba;level_5_no_window_ba;level_5_strict_numeric_ba;obs080d_core_mean_ba;obs080d_core_min_ci95_low;obs080d_core_min_p_above_threshold;feature_family;tested_count;geometry_level;minimal_sufficient_geometry | relation_id;task           |                   | base records; task normalization; OBS-081 v2 carrier/survival evidence; rig_status metadata |         |
| rig_survival_matrix        | outputs/rig_registry/rig_survival_matrix.csv        | True     |    256 | relation_id;obs;contract_family;contract_name;task;feature_contract;carrier_role;score;threshold;status;source_path;best_model;best_scheme;stratified_cv_best_ba;leave_object_out_best_ba;leave_cohort_out_best_ba;leave_transition_out_best_ba;ci_low;p_above_threshold;failure_rate;n_success;n_fail;feature_family;rig_status;failure_location;carrier                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | relation_id;task           | passed            | optional D1 aggregate survival-matrix fallback; tested/survival/failure counts              |         |
| rig_failure_localization   | outputs/rig_registry/rig_failure_localization.csv   | True     |     29 | relation_id;task;feature_contract;carrier_role;contract_family;contract_name;score;threshold;margin;status;failure_type;feature_family;rig_status;failure_mode;failure_location;carrier                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | relation_id;task           |                   | D2 failure localization; failure inventory                                                  |         |
| rig_geometry_needed_ladder | outputs/rig_registry/rig_geometry_needed_ladder.csv | True     |     24 | relation_id;task;carrier;carrier_role;carrier_role_detail;task_geometry_needed_level;task_geometry_needed_label;task_geometry_needed_rationale;level_1_stability_core_ba;level_3_geometry_ba;level_4_paths_ba;level_3_stability_plus_geometry_ba;level_5_no_window_ba;level_5_strict_numeric_ba;feature_family;geometry_level;minimal_sufficient_geometry;rig_status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | relation_id;task           | geometry_needed   | D4 geometry sufficiency                                                                     |         |
| rig_repair_recommendations | outputs/rig_registry/rig_repair_recommendations.csv | True     |     24 | relation_id;task;carrier;carrier_role;rig_status;repair_recommendation;repair_rationale;failure_notes;feature_family                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | relation_id;task           |                   | D3 repair specificity                                                                       |         |

## 3. Scoring schema

Readiness is computed from invariance strength, failure localization, repair specificity, geometry sufficiency, carrier convergence, and negative-control contrast. `rig_status` is metadata; it is used only as a categorical fallback when direct invariance evidence is unavailable. If invariance evidence comes only from categorical fallback, readiness class cannot exceed B.

| dimension                       |   weight |
|:--------------------------------|---------:|
| invariance_strength_score       |     0.25 |
| failure_localization_score      |     0.15 |
| repair_specificity_score        |     0.15 |
| geometry_sufficiency_score      |     0.15 |
| carrier_convergence_score       |     0.2  |
| negative_control_contrast_score |     0.1  |

## 4. Readiness class summary

| readiness_class    |   n |
|:-------------------|----:|
| C: diagnostic-only |  24 |

## 5. Readiness limiter summary

| readiness_limiter              |   n |
|:-------------------------------|----:|
| generic_repair_specificity     |  24 |
| weak_negative_control_contrast |  24 |
| diffuse_failure_localization   |  23 |

## 6. Dimension-level results

| relation_id                         | task       |   invariance_strength_score |   failure_localization_score |   repair_specificity_score |   geometry_sufficiency_score |   carrier_convergence_score |   negative_control_contrast_score |   readiness_score | readiness_class    |
|:------------------------------------|:-----------|----------------------------:|-----------------------------:|---------------------------:|-----------------------------:|----------------------------:|----------------------------------:|------------------:|:-------------------|
| C_vs_Cp3__no_window                 | C_vs_Cp3   |                    1        |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241289 |          0.751629 | C: diagnostic-only |
| C_vs_Cp3__strict_numeric_all        | C_vs_Cp3   |                    1        |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241289 |          0.751629 | C: diagnostic-only |
| Cp2_vs_Cp3__strict_numeric_all      | Cp2_vs_Cp3 |                    1        |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241289 |          0.751629 | C: diagnostic-only |
| Cp2_vs_Cp3__no_window               | Cp2_vs_Cp3 |                    1        |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241289 |          0.751629 | C: diagnostic-only |
| C_vs_Cp2__no_window                 | C_vs_Cp2   |                    0.999971 |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241231 |          0.751616 | C: diagnostic-only |
| C_vs_Cp2__strict_numeric_all        | C_vs_Cp2   |                    0.999971 |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241231 |          0.751616 | C: diagnostic-only |
| three_way__no_window                | three_way  |                    0.999899 |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241086 |          0.751583 | C: diagnostic-only |
| three_way__strict_numeric_all       | three_way  |                    0.99988  |                         0.35 |                        0.5 |                            1 |                           1 |                          0.241049 |          0.751575 | C: diagnostic-only |
| Cp2_vs_Cp3__geometry_scores_only    | Cp2_vs_Cp3 |                    0.999094 |                         0.35 |                        0.5 |                            1 |                           1 |                          0.239477 |          0.751221 | C: diagnostic-only |
| Cp2_vs_Cp3__stability_plus_geometry | Cp2_vs_Cp3 |                    0.99582  |                         0.35 |                        0.5 |                            1 |                           1 |                          0.232928 |          0.749748 | C: diagnostic-only |
| C_vs_Cp2__stability_plus_geometry   | C_vs_Cp2   |                    0.993302 |                         0.35 |                        0.5 |                            1 |                           1 |                          0.227893 |          0.748615 | C: diagnostic-only |
| C_vs_Cp3__stability_plus_geometry   | C_vs_Cp3   |                    0.98613  |                         0.35 |                        0.5 |                            1 |                           1 |                          0.213549 |          0.745387 | C: diagnostic-only |

## 7. Candidate intervention-hypothesis records

_No rows._

## 8. Negative-control contrasts

| relation_id                       | task     | carrier                 | feature_family          | negative_control_type       | negative_control_relation_id   |   record_score |   control_score |   contrast | contrast_passed   | contrast_basis                               |
|:----------------------------------|:---------|:------------------------|:------------------------|:----------------------------|:-------------------------------|---------------:|----------------:|-----------:|:------------------|:---------------------------------------------|
| C_vs_Cp2__stability_core_3        | C_vs_Cp2 | stability_core_3        | stability_core_3        | weak_status_or_low_survival | median_control_group           |       0.982595 |        0.907043 |  0.0755519 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__stability_core_3        | C_vs_Cp2 | stability_core_3        | stability_core_3        | context_sensitive           | median_control_group           |       0.982595 |        0.879356 |  0.103239  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__geometry_scores_only    | C_vs_Cp2 | geometry_scores_only    | geometry_scores_only    | weak_status_or_low_survival | median_control_group           |       0.983478 |        0.907043 |  0.0764351 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__geometry_scores_only    | C_vs_Cp2 | geometry_scores_only    | geometry_scores_only    | context_sensitive           | median_control_group           |       0.983478 |        0.879356 |  0.104122  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__no_window               | C_vs_Cp2 | no_window               | no_window               | weak_status_or_low_survival | median_control_group           |       0.999971 |        0.907043 |  0.092928  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__no_window               | C_vs_Cp2 | no_window               | no_window               | context_sensitive           | median_control_group           |       0.999971 |        0.879356 |  0.120615  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__stability_plus_geometry | C_vs_Cp2 | stability_plus_geometry | stability_plus_geometry | weak_status_or_low_survival | median_control_group           |       0.993302 |        0.907043 |  0.086259  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__stability_plus_geometry | C_vs_Cp2 | stability_plus_geometry | stability_plus_geometry | context_sensitive           | median_control_group           |       0.993302 |        0.879356 |  0.113946  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__strict_numeric_all      | C_vs_Cp2 | strict_numeric_all      | strict_numeric_all      | weak_status_or_low_survival | median_control_group           |       0.999971 |        0.907043 |  0.092928  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__strict_numeric_all      | C_vs_Cp2 | strict_numeric_all      | strict_numeric_all      | context_sensitive           | median_control_group           |       0.999971 |        0.879356 |  0.120615  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__path_shares_only        | C_vs_Cp2 | path_shares_only        | path_shares_only        | weak_status_or_low_survival | median_control_group           |       0.898901 |        0.915185 | -0.0162833 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp2__path_shares_only        | C_vs_Cp2 | path_shares_only        | path_shares_only        | context_sensitive           | median_control_group           |       0.898901 |        0.879356 |  0.0195457 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__stability_core_3        | C_vs_Cp3 | stability_core_3        | stability_core_3        | weak_status_or_low_survival | median_control_group           |       0.984355 |        0.907043 |  0.0773121 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__stability_core_3        | C_vs_Cp3 | stability_core_3        | stability_core_3        | context_sensitive           | median_control_group           |       0.984355 |        0.879356 |  0.104999  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__no_window               | C_vs_Cp3 | no_window               | no_window               | weak_status_or_low_survival | median_control_group           |       1        |        0.907043 |  0.0929571 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__no_window               | C_vs_Cp3 | no_window               | no_window               | context_sensitive           | median_control_group           |       1        |        0.879356 |  0.120644  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__stability_plus_geometry | C_vs_Cp3 | stability_plus_geometry | stability_plus_geometry | weak_status_or_low_survival | median_control_group           |       0.98613  |        0.907043 |  0.0790873 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__stability_plus_geometry | C_vs_Cp3 | stability_plus_geometry | stability_plus_geometry | context_sensitive           | median_control_group           |       0.98613  |        0.879356 |  0.106775  | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__strict_numeric_all      | C_vs_Cp3 | strict_numeric_all      | strict_numeric_all      | weak_status_or_low_survival | median_control_group           |       1        |        0.907043 |  0.0929571 | False             | record_invariance_minus_control_group_median |
| C_vs_Cp3__strict_numeric_all      | C_vs_Cp3 | strict_numeric_all      | strict_numeric_all      | context_sensitive           | median_control_group           |       1        |        0.879356 |  0.120644  | False             | record_invariance_minus_control_group_median |

## 9. Failure localization and repair structure

| relation_id                 | failure_mode                         | failure_location      | readiness_impact                                                                            |
|:----------------------------|:-------------------------------------|:----------------------|:--------------------------------------------------------------------------------------------|
| three_way__stability_core_3 | object_support_sensitivity           | structural_resampling | failure_localization_score=0.6;basis=distinct_failure_type_concentration_2_to_3_types       |
| three_way__stability_core_3 | object_support_sensitivity           | structural_resampling | failure_localization_score=0.6;basis=distinct_failure_type_concentration_2_to_3_types       |
| three_way__stability_core_3 | cohort_support_sensitivity           | structural_resampling | failure_localization_score=0.6;basis=distinct_failure_type_concentration_2_to_3_types       |
| three_way__stability_core_3 | transition_support_sensitivity       | structural_resampling | failure_localization_score=0.6;basis=distinct_failure_type_concentration_2_to_3_types       |
| three_way__path_shares_only | feature_projection_sensitivity       | feature_family        | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |
| three_way__path_shares_only | object_support_sensitivity           | structural_resampling | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |
| three_way__path_shares_only | cohort_support_sensitivity           | structural_resampling | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |
| three_way__path_shares_only | transition_support_sensitivity       | structural_resampling | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |
| three_way__path_shares_only | structural_recomposition_sensitivity | structural_resampling | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |
| three_way__path_shares_only | object_support_sensitivity           | structural_resampling | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |
| three_way__path_shares_only | object_support_sensitivity           | structural_resampling | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |
| C_vs_Cp2__path_shares_only  | cohort_support_sensitivity           | structural_resampling | failure_localization_score=0.35;basis=distinct_failure_type_concentration_more_than_3_types |

## 10. Geometry sufficiency ladder

| relation_id                       | task     | geometry_level   | minimal_sufficient_geometry   |   geometry_sufficiency_score | geometry_sufficiency_basis   |
|:----------------------------------|:---------|:-----------------|:------------------------------|-----------------------------:|:-----------------------------|
| C_vs_Cp2__stability_core_3        | C_vs_Cp2 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp2__geometry_scores_only    | C_vs_Cp2 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp2__no_window               | C_vs_Cp2 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp2__stability_plus_geometry | C_vs_Cp2 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp2__strict_numeric_all      | C_vs_Cp2 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp2__path_shares_only        | C_vs_Cp2 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp3__stability_core_3        | C_vs_Cp3 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp3__no_window               | C_vs_Cp3 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp3__stability_plus_geometry | C_vs_Cp3 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp3__strict_numeric_all      | C_vs_Cp3 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp3__geometry_scores_only    | C_vs_Cp3 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |
| C_vs_Cp3__path_shares_only        | C_vs_Cp3 | Level 1          | compact core sufficient       |                            1 | geometry_level_mapping       |

## 11. Blocked / insufficient records

_No rows._

## 12. Interpretation

Class A and B records are candidates for defining future testable intervention hypotheses. They are not evidence that interventions have been performed or that the system is controllable. Records in Class C/D/X remain useful as diagnostics, registry evidence, or missing-artifact signals.

## 13. What this does not show
- OBS-082 does not perform interventions.
- OBS-082 does not establish control.
- OBS-082 does not establish causality.
- OBS-082 does not establish external generalization.

## 14. Recommended next tests

For Class A/B records, design a follow-up perturbation or withholding probe that targets the record's intervention axis, evaluates relation survival under matched OBS-080-style contracts, and compares against explicit negative controls.

## 15. Top readiness records

| relation_id                         | task       | carrier                 | feature_family          |   readiness_score | readiness_class    | readiness_limiter                                                                      | readiness_blockers   |
|:------------------------------------|:-----------|:------------------------|:------------------------|------------------:|:-------------------|:---------------------------------------------------------------------------------------|:---------------------|
| C_vs_Cp3__no_window                 | C_vs_Cp3   | no_window               | no_window               |          0.751629 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp3__strict_numeric_all        | C_vs_Cp3   | strict_numeric_all      | strict_numeric_all      |          0.751629 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| Cp2_vs_Cp3__strict_numeric_all      | Cp2_vs_Cp3 | strict_numeric_all      | strict_numeric_all      |          0.751629 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| Cp2_vs_Cp3__no_window               | Cp2_vs_Cp3 | no_window               | no_window               |          0.751629 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp2__no_window                 | C_vs_Cp2   | no_window               | no_window               |          0.751616 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp2__strict_numeric_all        | C_vs_Cp2   | strict_numeric_all      | strict_numeric_all      |          0.751616 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| three_way__no_window                | three_way  | no_window               | no_window               |          0.751583 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| three_way__strict_numeric_all       | three_way  | strict_numeric_all      | strict_numeric_all      |          0.751575 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| Cp2_vs_Cp3__geometry_scores_only    | Cp2_vs_Cp3 | geometry_scores_only    | geometry_scores_only    |          0.751221 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| Cp2_vs_Cp3__stability_plus_geometry | Cp2_vs_Cp3 | stability_plus_geometry | stability_plus_geometry |          0.749748 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp2__stability_plus_geometry   | C_vs_Cp2   | stability_plus_geometry | stability_plus_geometry |          0.748615 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp3__stability_plus_geometry   | C_vs_Cp3   | stability_plus_geometry | stability_plus_geometry |          0.745387 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp3__stability_core_3          | C_vs_Cp3   | stability_core_3        | stability_core_3        |          0.744589 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp2__geometry_scores_only      | C_vs_Cp2   | geometry_scores_only    | geometry_scores_only    |          0.744194 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| C_vs_Cp2__stability_core_3          | C_vs_Cp2   | stability_core_3        | stability_core_3        |          0.743797 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| three_way__stability_plus_geometry  | three_way  | stability_plus_geometry | stability_plus_geometry |          0.741259 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| three_way__stability_core_3         | three_way  | stability_core_3        | stability_core_3        |          0.733315 | C: diagnostic-only | generic_repair_specificity;weak_negative_control_contrast                              |                      |
| C_vs_Cp3__geometry_scores_only      | C_vs_Cp3   | geometry_scores_only    | geometry_scores_only    |          0.727149 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| three_way__geometry_scores_only     | three_way  | geometry_scores_only    | geometry_scores_only    |          0.718983 | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
| Cp2_vs_Cp3__path_shares_only        | Cp2_vs_Cp3 | path_shares_only        | path_shares_only        |          0.71609  | C: diagnostic-only | diffuse_failure_localization;generic_repair_specificity;weak_negative_control_contrast |                      |
