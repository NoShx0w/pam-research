# OBS-084a — Schema and Partition Reconnaissance

## State

Reconnaissance completed.

Overall status: `schema_ready_for_obs084a_discovery_design_with_three_way_partition_candidate`

Interpretation: no fatal schema-level blocker found

This is a schema and partition audit only. It performs no candidate generation,
confirmation, witness assignment, FL promotion, repair design, intervention, or
causal analysis.

## Protocol alignment

This script operationalizes the pre-discovery reconnaissance step beneath
**OBS-084 RIG Direct Failure-Support Witness Protocol**.

Canonical guardrails:

> Directness is artifact-direct, not metaphysically direct and not causally direct.

> Localization is not atomization.

> A site is direct only through its witness.

> Discovery nominates a support; reserved evidence earns the localization claim.

A positive feasibility result means only that the artifact schemas appear able
to support a later protocol step. It is not evidence that a failure support
exists.

## Configuration

| setting | value |
|---|---|
| script_version | `1.0.0` |
| repo_root | `.` |
| outputs_root | `outputs` |
| output_dir | `outputs/rig_registry/obs084_direct_failure_witness/reconnaissance` |
| max_csv_mb | 512.0 |
| sample_rows | 250000 |
| min_two_way_clusters | 12 |
| min_three_way_clusters | 24 |
| min_clusters_per_stratum | 3 |

## Input artifacts

| artifact_label                                          | artifact_path                                                                                                                                          |    size_mb |   rows_read |   column_count | read_status            |
|:--------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|------------:|---------------:|:-----------------------|
| obs073_feature_table                                    | outputs/comparisons/obs073_C_vs_Cp3_v5_smoke/obs073_feature_table.csv                                                                                  |  58.9503   |       30000 |            171 | ok                     |
| obs073_feature_table__2                                 | outputs/comparisons/obs073_Cp2_vs_Cp3_v5_smoke/obs073_feature_table.csv                                                                                |  59.3815   |       30000 |            171 | ok                     |
| obs073_feature_table__3                                 | outputs/comparisons/obs073_Cp_vs_Cp3_v5_smoke/obs073_feature_table.csv                                                                                 |  59.2173   |       30000 |            171 | ok                     |
| obs074_feature_table                                    | outputs/comparisons/obs074_Cp3_lexical_field_bridge_smoke/obs074_feature_table.csv                                                                     |  26.3358   |       15000 |            138 | ok                     |
| obs074_feature_table__2                                 | outputs/comparisons/obs074_Cp3_lexical_field_bridge_smoke_v2/obs074_feature_table.csv                                                                  |  42.7182   |       15000 |            238 | ok                     |
| obs074_feature_table__3                                 | outputs/comparisons/obs074_lexical_field_bridge/C_Cp_Cp2_v3_smoke/obs074_feature_table.csv                                                             | 141.305    |       45000 |            238 | ok                     |
| obs075d_feature_table                                   | outputs/comparisons/obs075d_cp3_path_lexical_control_logreg/obs075d_feature_table.csv                                                                  | 463.589    |      145000 |            286 | ok                     |
| obs075d_feature_table__2                                | outputs/comparisons/obs075d_cp3_path_lexical_control_logreg_v2/obs075d_feature_table.csv                                                               | 643.834    |           0 |            146 | header_only_size_limit |
| obs076d_scale_feature_table                             | outputs/comparisons/obs076d_Cp2_vs_Cp3_scale_conditioned_factor_separability_shared14_mds_pilot_logreg/obs076d_scale_feature_table.csv                 |   0.821536 |        1200 |             50 | ok                     |
| obs076d_scale_feature_table__2                          | outputs/comparisons/obs076d_Cp2_vs_Cp3_scale_conditioned_factor_separability_shared14_mds_pilot_logreg_dynamic_only_v2/obs076d_scale_feature_table.csv |   0.821536 |        1200 |             50 | ok                     |
| obs078a_confusion_matrices                              | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_confusion_matrices.csv                                                            |   0.015656 |         324 |              6 | ok                     |
| obs078a_feature_importance                              | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_feature_importance.csv                                                            |   0.199181 |        1740 |              7 | ok                     |
| obs078a_feature_manifest                                | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_feature_manifest.csv                                                              |   0.007609 |         111 |              6 | ok                     |
| obs078a_feature_table                                   | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_feature_table.csv                                                                 |   0.244096 |         168 |            111 | ok                     |
| obs078a_input_manifest                                  | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_input_manifest.csv                                                                |   0.002905 |          18 |              6 | ok                     |
| obs078a_panel_scores                                    | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_panel_scores.csv                                                                  |   0.096274 |          48 |             12 | ok                     |
| obs078a_permutation_scores                              | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_permutation_scores.csv                                                            |   0.237958 |        2400 |              7 | ok                     |
| obs078a_permutation_summary                             | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_permutation_summary.csv                                                           |   0.002419 |          24 |              8 | ok                     |
| obs078a_report                                          | outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_report.md                                                                         |   0.191663 |         nan |              0 | exists_non_tabular     |
| obs078a_confusion_matrices__2                           | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_confusion_matrices.csv                                                         |   0.034683 |         648 |              7 | ok                     |
| obs078a_feature_importance__2                           | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_importance.csv                                                         |   0.337316 |        2736 |              9 | ok                     |
| obs078a_feature_manifest__2                             | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_manifest.csv                                                           |   0.008272 |         112 |              7 | ok                     |
| obs078a_feature_table__2                                | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv                                                              |   0.244279 |         168 |            112 | ok                     |
| obs078a_input_manifest__2                               | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_input_manifest.csv                                                             |   0.002905 |          18 |              6 | ok                     |
| obs078a_mechanistic_signature_classifier_v2_summary     | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_mechanistic_signature_classifier_v2_summary.md                                 |   0.011397 |         nan |              0 | exists_non_tabular     |
| obs078a_panel_scores__2                                 | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_panel_scores.csv                                                               |   0.149744 |          96 |             13 | ok                     |
| obs078a_permutation_scores__2                           | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_permutation_scores.csv                                                         |   0.500563 |        4800 |              8 | ok                     |
| obs078a_permutation_summary__2                          | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_permutation_summary.csv                                                        |   0.005182 |          48 |              9 | ok                     |
| obs078a_report__2                                       | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_report.md                                                                      |   0.244486 |         nan |              0 | exists_non_tabular     |
| obs078b_confusion_matrices                              | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_confusion_matrices.csv                                                                  |   0.030663 |         828 |              5 | ok                     |
| obs078b_feature_importance                              | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_feature_importance.csv                                                                  |   0.216917 |        1940 |              7 | ok                     |
| obs078b_feature_sets                                    | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_feature_sets.csv                                                                        |   0.023239 |         388 |              4 | ok                     |
| obs078b_input_manifest                                  | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_input_manifest.csv                                                                      |   0.000253 |           2 |              5 | ok                     |
| obs078b_minimal_signature_summary                       | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_minimal_signature_summary.md                                                            |   0.026335 |         nan |              0 | exists_non_tabular     |
| obs078b_panel_scores                                    | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_panel_scores.csv                                                                        |   0.063185 |          92 |              9 | ok                     |
| obs078b_permutation_scores                              | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_permutation_scores.csv                                                                  |   0.600884 |        6900 |              6 | ok                     |
| obs078b_permutation_summary                             | outputs/comparisons/obs078b_minimal_signature_ablation/obs078b_permutation_summary.csv                                                                 |   0.00651  |          69 |              7 | ok                     |
| obs078c_case_cohort_matrix                              | outputs/comparisons/obs078c_stability_signature_localization/obs078c_case_cohort_matrix.csv                                                            |   0.001752 |           6 |             13 | ok                     |
| obs078c_case_object_matrix                              | outputs/comparisons/obs078c_stability_signature_localization/obs078c_case_object_matrix.csv                                                            |   0.002063 |           8 |             13 | ok                     |
| obs078c_feature_zscores                                 | outputs/comparisons/obs078c_stability_signature_localization/obs078c_feature_zscores.csv                                                               |   0.276444 |         168 |            123 | ok                     |
| obs078c_input_manifest                                  | outputs/comparisons/obs078c_stability_signature_localization/obs078c_input_manifest.csv                                                                |   0.000139 |           1 |              5 | ok                     |
| obs078c_pairwise_case_contrasts                         | outputs/comparisons/obs078c_stability_signature_localization/obs078c_pairwise_case_contrasts.csv                                                       |   0.104253 |         143 |             49 | ok                     |
| obs078c_report                                          | outputs/comparisons/obs078c_stability_signature_localization/obs078c_report.md                                                                         |   0.021869 |         nan |              0 | exists_non_tabular     |
| obs078c_signature_by_case                               | outputs/comparisons/obs078c_stability_signature_localization/obs078c_signature_by_case.csv                                                             |   0.002433 |           3 |             32 | ok                     |
| obs078c_signature_by_cohort                             | outputs/comparisons/obs078c_stability_signature_localization/obs078c_signature_by_cohort.csv                                                           |   0.011021 |          18 |             33 | ok                     |
| obs078c_signature_by_object                             | outputs/comparisons/obs078c_stability_signature_localization/obs078c_signature_by_object.csv                                                           |   0.012111 |          21 |             33 | ok                     |
| obs078c_signature_by_object_cohort                      | outputs/comparisons/obs078c_stability_signature_localization/obs078c_signature_by_object_cohort.csv                                                    |   0.056843 |         115 |             34 | ok                     |
| obs078c_signature_by_object_transition                  | outputs/comparisons/obs078c_stability_signature_localization/obs078c_signature_by_object_transition.csv                                                |   0.01748  |          30 |             34 | ok                     |
| obs078c_signature_by_transition                         | outputs/comparisons/obs078c_stability_signature_localization/obs078c_signature_by_transition.csv                                                       |   0.008698 |          14 |             33 | ok                     |
| obs078c_top_separating_groups                           | outputs/comparisons/obs078c_stability_signature_localization/obs078c_top_separating_groups.csv                                                         |   0.109191 |         149 |             49 | ok                     |
| obs078c_zscore_stats                                    | outputs/comparisons/obs078c_stability_signature_localization/obs078c_zscore_stats.csv                                                                  |   0.000281 |           3 |              5 | ok                     |
| obs079a_confusion_matrices                              | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_confusion_matrices.csv                                                     |   0.113258 |        2020 |              6 | ok                     |
| obs079a_feature_manifest                                | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_feature_manifest.csv                                                       |   0.000246 |           3 |              3 | ok                     |
| obs079a_input_manifest                                  | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_input_manifest.csv                                                         |   0.000139 |           1 |              5 | ok                     |
| obs079a_leave_structure_failures                        | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_leave_structure_failures.csv                                               |   1e-06    |           0 |              0 | empty_csv              |
| obs079a_leave_structure_predictions                     | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_leave_structure_predictions.csv                                            |   0.339712 |        4032 |             10 | ok                     |
| obs079a_leave_structure_scores                          | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_leave_structure_scores.csv                                                 |   0.056093 |         356 |             15 | ok                     |
| obs079a_report                                          | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_report.md                                                                  |   0.036119 |         nan |              0 | exists_non_tabular     |
| obs079a_scheme_summary                                  | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_scheme_summary.csv                                                         |   0.007496 |          44 |             12 | ok                     |
| obs079a_validation_groups                               | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_validation_groups.csv                                                      |   0.006381 |          89 |              6 | ok                     |
| obs079b_bootstrap_ci_by_case                            | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_ci_by_case.csv                                                          |   0.0046   |          30 |             12 | ok                     |
| obs079b_bootstrap_ci_by_cohort                          | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_ci_by_cohort.csv                                                        |   0.028234 |         180 |             13 | ok                     |
| obs079b_bootstrap_ci_by_object                          | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_ci_by_object.csv                                                        |   0.033277 |         210 |             13 | ok                     |
| obs079b_bootstrap_ci_by_object_cohort                   | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_ci_by_object_cohort.csv                                                 |   0.176132 |        1150 |             14 | ok                     |
| obs079b_bootstrap_ci_by_transition                      | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_ci_by_transition.csv                                                    |   0.021575 |         140 |             13 | ok                     |
| obs079b_bootstrap_pairwise_case_contrasts               | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_pairwise_case_contrasts.csv                                             |   0.004774 |          30 |             13 | ok                     |
| obs079b_bootstrap_pairwise_case_contrasts_by_cohort     | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_pairwise_case_contrasts_by_cohort.csv                                   |   0.029022 |         180 |             14 | ok                     |
| obs079b_bootstrap_pairwise_case_contrasts_by_object     | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_pairwise_case_contrasts_by_object.csv                                   |   0.030221 |         180 |             14 | ok                     |
| obs079b_bootstrap_pairwise_case_contrasts_by_transition | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_bootstrap_pairwise_case_contrasts_by_transition.csv                               |   0.020893 |         130 |             14 | ok                     |
| obs079b_feature_zscores                                 | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_feature_zscores.csv                                                               |   0.276444 |         168 |            123 | ok                     |
| obs079b_group_inventory                                 | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_group_inventory.csv                                                               |   0.006895 |         171 |              6 | ok                     |
| obs079b_input_manifest                                  | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_input_manifest.csv                                                                |   0.000139 |           1 |              5 | ok                     |
| obs079b_report                                          | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_report.md                                                                         |   0.019724 |         nan |              0 | exists_non_tabular     |
| obs079b_zscore_stats                                    | outputs/comparisons/obs079b_stability_signature_bootstrap_ci/obs079b_zscore_stats.csv                                                                  |   0.000281 |           3 |              5 | ok                     |
| obs079c_confusion_matrices                              | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_confusion_matrices.csv                                                              |   0.441763 |        6048 |             13 | ok                     |
| obs079c_failures                                        | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_failures.csv                                                                        |   1e-06    |           0 |              0 | empty_csv              |
| obs079c_feature_importance                              | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_feature_importance.csv                                                              |   0.283448 |        2160 |             13 | ok                     |
| obs079c_feature_panels                                  | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_feature_panels.csv                                                                  |   0.000493 |          12 |              4 | ok                     |
| obs079c_input_manifest                                  | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_input_manifest.csv                                                                  |   0.000139 |           1 |              5 | ok                     |
| obs079c_pairwise_axis_summary                           | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_axis_summary.csv                                                           |   0.00117  |          12 |              8 | ok                     |
| obs079c_pairwise_permutation_scores                     | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_permutation_scores.csv                                                     |   1.3005   |       12600 |              8 | ok                     |
| obs079c_pairwise_permutation_summary                    | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_permutation_summary.csv                                                    |   0.006968 |          63 |              9 | ok                     |
| obs079c_pairwise_predictions                            | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_predictions.csv                                                            |   3.72899  |       37632 |             12 | ok                     |
| obs079c_pairwise_scheme_summary                         | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_scheme_summary.csv                                                         |   0.103189 |         672 |             13 | ok                     |
| obs079c_pairwise_scores                                 | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_scores.csv                                                                 |   0.304486 |        1680 |             20 | ok                     |
| obs079c_report                                          | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_report.md                                                                           |   0.070065 |         nan |              0 | exists_non_tabular     |
| obs079c_validation_groups                               | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_validation_groups.csv                                                               |   0.003611 |          60 |              7 | ok                     |
| obs080a_confusion_matrices                              | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_confusion_matrices.csv                                                        |   0.7707   |       11760 |              8 | ok                     |
| obs080a_failures                                        | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_failures.csv                                                                  |   1e-06    |           0 |              0 | empty_csv              |
| obs080a_input_manifest                                  | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_input_manifest.csv                                                            |   0.000139 |           1 |              5 | ok                     |
| obs080a_pairwise_transform_summary                      | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_pairwise_transform_summary.csv                                                |   0.002967 |          21 |              9 | ok                     |
| obs080a_permutation_scores                              | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_permutation_scores.csv                                                        |   1.71241  |       16800 |              8 | ok                     |
| obs080a_permutation_summary                             | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_permutation_summary.csv                                                       |   0.009113 |          84 |              9 | ok                     |
| obs080a_report                                          | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_report.md                                                                     |   0.066232 |         nan |              0 | exists_non_tabular     |
| obs080a_transform_manifest                              | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_manifest.csv                                                        |   0.003246 |          21 |              9 | ok                     |
| obs080a_transform_predictions                           | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_predictions.csv                                                     |   5.51447  |       56448 |             12 | ok                     |
| obs080a_transform_scheme_summary                        | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_scheme_summary.csv                                                  |   0.140045 |         896 |             13 | ok                     |
| obs080a_transform_scores                                | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_scores.csv                                                          |   0.54706  |        2240 |             18 | ok                     |
| obs080a_transform_stability_matrix                      | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_stability_matrix.csv                                                |   0.003931 |          28 |              9 | ok                     |
| obs080a_transformed_feature_table                       | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transformed_feature_table.csv                                                 |   0.311296 |         168 |            135 | ok                     |
| obs080a_validation_groups                               | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_validation_groups.csv                                                         |   0.004908 |          80 |              7 | ok                     |
| obs080b_confusion_matrices                              | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_confusion_matrices.csv                                                       |   0.768007 |       12260 |              8 | ok                     |
| obs080b_failures                                        | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_failures.csv                                                                 |   0.00971  |         112 |             12 | ok                     |
| obs080b_input_manifest                                  | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_input_manifest.csv                                                           |   0.000139 |           1 |              5 | ok                     |
| obs080b_pairwise_scale_band_summary                     | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_pairwise_scale_band_summary.csv                                              |   0.003136 |          27 |              9 | ok                     |
| obs080b_permutation_scores                              | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_permutation_scores.csv                                                       |   2.08825  |       21600 |              8 | ok                     |
| obs080b_permutation_summary                             | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_permutation_summary.csv                                                      |   0.011109 |         108 |              9 | ok                     |
| obs080b_report                                          | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_report.md                                                                    |   0.129518 |         nan |              0 | exists_non_tabular     |
| obs080b_scale_band_manifest                             | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_manifest.csv                                                      |   0.000843 |          10 |             14 | ok                     |
| obs080b_scale_band_predictions                          | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_predictions.csv                                                   |   4.54923  |       43288 |             16 | ok                     |
| obs080b_scale_band_scheme_summary                       | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_scheme_summary.csv                                                |   0.156232 |        1060 |             13 | ok                     |
| obs080b_scale_band_scores                               | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_scores.csv                                                        |   0.469511 |        2320 |             18 | ok                     |
| obs080b_scale_band_stability_matrix                     | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_stability_matrix.csv                                              |   0.004226 |          36 |              9 | ok                     |
| obs080b_validation_groups                               | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_validation_groups.csv                                                        |   0.041224 |         608 |              8 | ok                     |
| obs080c_confusion_matrices                              | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_confusion_matrices.csv                                                         |   1.39543  |       20160 |              8 | ok                     |
| obs080c_contract_contrast_summary                       | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_contract_contrast_summary.csv                                                  |   0.000876 |           4 |             10 | ok                     |
| obs080c_failures                                        | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_failures.csv                                                                   |   0.000122 |           0 |             12 | ok                     |
| obs080c_feature_contract_manifest                       | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_manifest.csv                                                  |   0.014603 |         263 |              5 | ok                     |
| obs080c_feature_contract_predictions                    | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_predictions.csv                                               |   9.81546  |       96768 |             12 | ok                     |
| obs080c_feature_contract_scheme_summary                 | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_scheme_summary.csv                                            |   0.242692 |        1536 |             13 | ok                     |
| obs080c_feature_contract_scores                         | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_scores.csv                                                    |   3.42724  |        3840 |             19 | ok                     |
| obs080c_feature_contract_stability_matrix               | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_stability_matrix.csv                                          |   0.006527 |          48 |              9 | ok                     |
| obs080c_input_manifest                                  | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_input_manifest.csv                                                             |   0.000139 |           1 |              5 | ok                     |
| obs080c_pairwise_feature_contract_summary               | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_pairwise_feature_contract_summary.csv                                          |   0.004883 |          36 |              9 | ok                     |
| obs080c_permutation_scores                              | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_permutation_scores.csv                                                         |   3.04143  |       28800 |              8 | ok                     |
| obs080c_permutation_summary                             | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_permutation_summary.csv                                                        |   0.016001 |         144 |              9 | ok                     |
| obs080c_report                                          | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_report.md                                                                      |   0.126012 |         nan |              0 | exists_non_tabular     |
| obs080c_validation_groups                               | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_validation_groups.csv                                                          |   0.075697 |         960 |              9 | ok                     |
| obs080d_bootstrap_confusion_matrices                    | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_confusion_matrices.csv                                        | 100.416    |      250000 |              8 | sampled_row_limit      |
| obs080d_bootstrap_failures                              | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_failures.csv                                                  |   0.00011  |           0 |             11 | ok                     |
| obs080d_bootstrap_predictions                           | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_predictions.csv                                               |   8.1e-05  |           1 |              2 | ok                     |
| obs080d_bootstrap_scores                                | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_scores.csv                                                    | 258.407    |      250000 |             16 | sampled_row_limit      |
| obs080d_bootstrap_summary                               | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_summary.csv                                                   |   0.135241 |         576 |             22 | ok                     |
| obs080d_core_vs_reference_summary                       | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_core_vs_reference_summary.csv                                           |   0.006042 |          24 |             15 | ok                     |
| obs080d_feature_contract_manifest                       | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_feature_contract_manifest.csv                                           |   0.007891 |         139 |              5 | ok                     |
| obs080d_input_manifest                                  | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_input_manifest.csv                                                      |   0.000139 |           1 |              5 | ok                     |
| obs080d_report                                          | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_report.md                                                               |   0.13688  |         nan |              0 | exists_non_tabular     |
| obs080d_resampling_manifest                             | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_resampling_manifest.csv                                                 |   0.003258 |           6 |              8 | ok                     |
| obs080d_structural_stability_matrix                     | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_structural_stability_matrix.csv                                         |   0.025002 |         144 |             19 | ok                     |
| obs080d_bootstrap_confusion_matrices__2                 | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_bootstrap_confusion_matrices.csv                                  |   4.93577  |       75600 |              8 | ok                     |
| obs080d_bootstrap_failures__2                           | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_bootstrap_failures.csv                                            |   0.00011  |           0 |             11 | ok                     |
| obs080d_bootstrap_predictions__2                        | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_bootstrap_predictions.csv                                         |   8.1e-05  |           1 |              2 | ok                     |
| obs080d_bootstrap_scores__2                             | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_bootstrap_scores.csv                                              |  12.8972   |       14400 |             16 | ok                     |
| obs080d_bootstrap_summary__2                            | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_bootstrap_summary.csv                                             |   0.133271 |         576 |             22 | ok                     |
| obs080d_core_vs_reference_summary__2                    | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_core_vs_reference_summary.csv                                     |   0.005862 |          24 |             15 | ok                     |
| obs080d_feature_contract_manifest__2                    | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_feature_contract_manifest.csv                                     |   0.007891 |         139 |              5 | ok                     |
| obs080d_input_manifest__2                               | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_input_manifest.csv                                                |   0.000139 |           1 |              5 | ok                     |
| obs080d_report__2                                       | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_report.md                                                         |   0.136879 |         nan |              0 | exists_non_tabular     |
| obs080d_resampling_manifest__2                          | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_resampling_manifest.csv                                           |   0.003258 |           6 |              8 | ok                     |
| obs080d_structural_stability_matrix__2                  | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_structural_stability_matrix.csv                                   |   0.024631 |         144 |             19 | ok                     |
| obs073_feature_table__4                                 | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_C0_instant_vs_Cp_v5/obs073_feature_table.csv                                              | 393.882    |      200000 |            171 | ok                     |
| obs073_feature_table__5                                 | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_Cp_vs_Cp2_v5/obs073_feature_table.csv                                                     | 392.502    |      200000 |            171 | ok                     |
| obs073_feature_table__6                                 | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_continuous_field_groupoid_reduction_smoke/obs073_feature_table.csv                        |  57.6142   |       30000 |            166 | ok                     |
| obs073_feature_table__7                                 | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_continuous_field_groupoid_reduction_v2_smoke/obs073_feature_table.csv                     |  57.6142   |       30000 |            166 | ok                     |
| obs073_feature_table__8                                 | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_continuous_field_groupoid_reduction_v3_smoke/obs073_feature_table.csv                     |  57.6142   |       30000 |            166 | ok                     |
| obs073_feature_table__9                                 | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_continuous_field_groupoid_reduction_v4_full/obs073_feature_table.csv                      | 391.763    |      200000 |            171 | ok                     |
| obs073_feature_table__10                                | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_continuous_field_groupoid_reduction_v4_smoke/obs073_feature_table.csv                     |  59.2482   |       30000 |            171 | ok                     |
| obs073_feature_table__11                                | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_continuous_field_groupoid_reduction_v5_full/obs073_feature_table.csv                      | 391.763    |      200000 |            171 | ok                     |
| obs073_feature_table__12                                | outputs/corpora/C/campaigns/canonical_legacy/pipeline/obs073_continuous_field_groupoid_reduction_v5_smoke/obs073_feature_table.csv                     |  59.2482   |       30000 |            171 | ok                     |
| obs073_feature_table__13                                | outputs/obs073_C0_instant_vs_Cp_v5/obs073_feature_table.csv                                                                                            | 393.882    |      200000 |            171 | ok                     |
| obs073_feature_table__14                                | outputs/obs073_C_vs_Cp_v5/obs073_feature_table.csv                                                                                                     | 392.901    |      200000 |            171 | ok                     |
| obs073_feature_table__15                                | outputs/obs073_Cp_vs_Cp2_v5/obs073_feature_table.csv                                                                                                   | 392.502    |      200000 |            171 | ok                     |
| obs073_feature_table__16                                | outputs/obs073_continuous_field_groupoid_reduction_smoke/obs073_feature_table.csv                                                                      |  57.6142   |       30000 |            166 | ok                     |
| obs073_feature_table__17                                | outputs/obs073_continuous_field_groupoid_reduction_v2_smoke/obs073_feature_table.csv                                                                   |  57.6142   |       30000 |            166 | ok                     |
| obs073_feature_table__18                                | outputs/obs073_continuous_field_groupoid_reduction_v3_smoke/obs073_feature_table.csv                                                                   |  57.6142   |       30000 |            166 | ok                     |
| obs073_feature_table__19                                | outputs/obs073_continuous_field_groupoid_reduction_v4_full/obs073_feature_table.csv                                                                    | 391.763    |      200000 |            171 | ok                     |
| obs073_feature_table__20                                | outputs/obs073_continuous_field_groupoid_reduction_v4_smoke/obs073_feature_table.csv                                                                   |  59.2482   |       30000 |            171 | ok                     |
| obs073_feature_table__21                                | outputs/obs073_continuous_field_groupoid_reduction_v5_full/obs073_feature_table.csv                                                                    | 391.763    |      200000 |            171 | ok                     |
| obs073_feature_table__22                                | outputs/obs073_continuous_field_groupoid_reduction_v5_smoke/obs073_feature_table.csv                                                                   |  59.2482   |       30000 |            171 | ok                     |
| obs074_feature_table__4                                 | outputs/obs074_lexical_field_bridge_C_Cp_Cp2_v2_smoke/obs074_feature_table.csv                                                                         | 137.507    |       45000 |            235 | ok                     |
| rig_stability_core_geometry                             | outputs/rig_navigator/views/rig_stability_core_geometry.csv                                                                                            |   0.279871 |         168 |            126 | ok                     |
| obs082_blockers                                         | outputs/rig_registry/obs082_intervention_readiness/obs082_blockers.csv                                                                                 |   0.000105 |           0 |              9 | ok                     |
| obs082_candidate_intervention_hypotheses                | outputs/rig_registry/obs082_intervention_readiness/obs082_candidate_intervention_hypotheses.csv                                                        |   0.000299 |           0 |             18 | ok                     |
| obs082_failure_mode_inventory                           | outputs/rig_registry/obs082_intervention_readiness/obs082_failure_mode_inventory.csv                                                                   |   0.005846 |          29 |             13 | ok                     |
| obs082_input_manifest                                   | outputs/rig_registry/obs082_intervention_readiness/obs082_input_manifest.csv                                                                           |   0.002848 |           5 |              9 | ok                     |
| obs082_negative_control_contrasts                       | outputs/rig_registry/obs082_intervention_readiness/obs082_negative_control_contrasts.csv                                                               |   0.010448 |          48 |             11 | ok                     |
| obs082_relation_readiness_scores                        | outputs/rig_registry/obs082_intervention_readiness/obs082_relation_readiness_scores.csv                                                                |   0.016597 |          24 |             35 | ok                     |
| obs082_report                                           | outputs/rig_registry/obs082_intervention_readiness/obs082_report.md                                                                                    |   0.031426 |         nan |              0 | exists_non_tabular     |
| obs083_blocker_refinement                               | outputs/rig_registry/obs083_negative_control_localization/obs083_blocker_refinement.csv                                                                |   0.010603 |          24 |              9 | ok                     |
| obs083_carrier_control_contrast                         | outputs/rig_registry/obs083_negative_control_localization/obs083_carrier_control_contrast.csv                                                          |   0.044836 |         120 |             18 | ok                     |
| obs083_contract_control_contrast                        | outputs/rig_registry/obs083_negative_control_localization/obs083_contract_control_contrast.csv                                                         |   0.024408 |          96 |             16 | ok                     |
| obs083_diagnostic_subclass_assignments                  | outputs/rig_registry/obs083_negative_control_localization/obs083_diagnostic_subclass_assignments.csv                                                   |   0.018836 |          24 |             32 | ok                     |
| obs083_failure_localization_matrix                      | outputs/rig_registry/obs083_negative_control_localization/obs083_failure_localization_matrix.csv                                                       |   0.014255 |          24 |             24 | ok                     |
| obs083_geometry_needed_control_contrast                 | outputs/rig_registry/obs083_negative_control_localization/obs083_geometry_needed_control_contrast.csv                                                  |   0.001804 |           4 |             16 | ok                     |
| obs083_input_manifest                                   | outputs/rig_registry/obs083_negative_control_localization/obs083_input_manifest.csv                                                                    |   0.002162 |          14 |              8 | ok                     |
| obs083_matched_negative_control_design                  | outputs/rig_registry/obs083_negative_control_localization/obs083_matched_negative_control_design.csv                                                   |   0.200047 |         416 |             17 | ok                     |
| obs083_readiness_delta_from_obs082                      | outputs/rig_registry/obs083_negative_control_localization/obs083_readiness_delta_from_obs082.csv                                                       |   0.006102 |          24 |             13 | ok                     |
| obs083_relation_control_contrast                        | outputs/rig_registry/obs083_negative_control_localization/obs083_relation_control_contrast.csv                                                         |   0.024134 |          72 |             16 | ok                     |
| obs083_repair_specificity_sharpening                    | outputs/rig_registry/obs083_negative_control_localization/obs083_repair_specificity_sharpening.csv                                                     |   0.026888 |          24 |             20 | ok                     |
| obs083_report                                           | outputs/rig_registry/obs083_negative_control_localization/obs083_report.md                                                                             |   0.010449 |         nan |              0 | exists_non_tabular     |
| rig_failure_localization                                | outputs/rig_registry/rig_failure_localization.csv                                                                                                      |   0.005946 |          29 |             11 | ok                     |
| rig_geometry_needed_ladder                              | outputs/rig_registry/rig_geometry_needed_ladder.csv                                                                                                    |   0.008868 |          24 |             14 | ok                     |
| rig_relation_registry                                   | outputs/rig_registry/rig_relation_registry.csv                                                                                                         |   0.020925 |          24 |             47 | ok                     |
| rig_repair_recommendations                              | outputs/rig_registry/rig_repair_recommendations.csv                                                                                                    |   0.007351 |          24 |              8 | ok                     |
| rig_survival_matrix                                     | outputs/rig_registry/rig_survival_matrix.csv                                                                                                           |   0.077635 |         256 |             22 | ok                     |

## Observation and structural units

- Unit inventory rows: 1620
- Candidate partition assessment rows: 74
- Two-way feasible rows: 36
- Three-way feasible rows: 36

Partition feasibility is schema-level and cluster-count-based. It does not prove
that candidate supports will have adequate matched complements or balanced
reserved evidence.

| artifact_label           | unit_family   | unit_column   |   unique_clusters | feasible_two_way   | feasible_three_way   | recommended_design                           | limitation   |
|:-------------------------|:--------------|:--------------|------------------:|:-------------------|:---------------------|:---------------------------------------------|:-------------|
| obs075d_feature_table    | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__4  | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__5  | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__9  | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__11 | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__13 | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__14 | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__15 | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__19 | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__21 | route_id      | path_id       |            100000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__2  | route_id      | path_id       |             27666 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__10 | route_id      | path_id       |             27666 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__12 | route_id      | path_id       |             27666 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__20 | route_id      | path_id       |             27666 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__22 | route_id      | path_id       |             27666 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table     | route_id      | path_id       |             15000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs073_feature_table__3  | route_id      | path_id       |             15000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs074_feature_table     | route_id      | path_id       |             15000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs074_feature_table__2  | route_id      | path_id       |             15000 | True               | True                 | candidate_discovery_confirmation_replication |              |
| obs074_feature_table__3  | route_id      | path_id       |             15000 | True               | True                 | candidate_discovery_confirmation_replication |              |

## Candidate support vocabulary availability

| support_family        | available   |   artifact_count |   max_unique_values |
|:----------------------|:------------|-----------------:|--------------------:|
| boundary_relative     | False       |                0 |                   0 |
| cohort                | True        |               20 |                   6 |
| contract_or_transform | True        |               12 |                   7 |
| feature_family        | True        |               11 |                  23 |
| object                | True        |               21 |                   8 |
| provenance_slice      | True        |                1 |                   4 |
| route_or_path         | True        |               28 |              100000 |
| scale_band            | True        |               40 |                  10 |
| seam_relative         | True        |                2 |                 113 |
| transition            | True        |               15 |                   5 |
| window                | False       |                0 |                   0 |

Support availability means that an address family can potentially be indexed.
It does not nominate a support or establish degradation.

## Join-key audit

- Audit rows: 5754
- Artifact-local keys marked candidate primary/one-side: 18
- Cross-artifact schema bridges: 3594

Shared field names do not prove semantic key equivalence. Future analytical
joins must audit value domains, cardinality, namespace, and provenance.

## Control feasibility

| record_id                           | subclass                | confirmation_eligibility   |   relation_control_count |   carrier_control_count | control_feasibility                     |
|:------------------------------------|:------------------------|:---------------------------|-------------------------:|------------------------:|:----------------------------------------|
| C_vs_Cp2__stability_core_3          | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp2__geometry_scores_only      | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp2__no_window                 | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp2__stability_plus_geometry   | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp2__strict_numeric_all        | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp2__path_shares_only          | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp3__stability_core_3          | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp3__no_window                 | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp3__stability_plus_geometry   | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp3__strict_numeric_all        | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp3__geometry_scores_only      | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| C_vs_Cp3__path_shares_only          | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| Cp2_vs_Cp3__geometry_scores_only    | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| Cp2_vs_Cp3__no_window               | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| Cp2_vs_Cp3__path_shares_only        | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| Cp2_vs_Cp3__stability_plus_geometry | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| Cp2_vs_Cp3__strict_numeric_all      | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| Cp2_vs_Cp3__stability_core_3        | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| three_way__geometry_scores_only     | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| three_way__no_window                | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| three_way__stability_plus_geometry  | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| three_way__strict_numeric_all       | C1_contrast-limited     | discovery_only_or_unknown  |                        3 |                       5 | relation_and_carrier_controls_available |
| three_way__path_shares_only         | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |
| three_way__stability_core_3         | C2_localization-limited | fl3_confirmation_eligible  |                        3 |                       5 | relation_and_carrier_controls_available |

Registry-level relation and carrier controls are only candidate control sets.
Observation-level support overlap, baseline balance, contract exposure, and
failure-mode comparability remain unproven.

## Provenance and versioning readiness

- Artifacts audited: 180
- Strong schema-provenance artifacts: 0
- Artifact hashes available: 180

Artifact hashes support a future frozen source manifest. They do not by
themselves establish scientific lineage equivalence.

## Leakage-sensitive fields

- Fields flagged for explicit handling: 740

Identity, regime, label, record, and structural-unit fields may be required for
outcomes, grouping, matching, or provenance. Their presence does not imply
leakage. Future predictive carriers must explicitly exclude or audit them.

## Reconnaissance decision rules

The repository may proceed to OBS-084a candidate-discovery implementation only
when all of the following are supported:

1. at least one defensible structural partition unit exists;
2. discovery and confirmation can be separated at the cluster level;
3. C2/localization-limited records can be identified;
4. relation and/or carrier control sets can be constructed;
5. candidate support fields are addressable;
6. provenance and source hashes can be frozen;
7. leakage-sensitive identity fields can be separated from predictive carriers.

A three-way discovery/confirmation/replication split is preferred only when the
number and balance of independent clusters support it. Otherwise the protocol
should use a two-way split plus dependence-aware structural resampling.

## Outputs

- `obs084a_input_manifest.csv`
- `obs084a_schema_inventory.csv`
- `obs084a_join_key_audit.csv`
- `obs084a_observation_unit_inventory.csv`
- `obs084a_candidate_support_availability.csv`
- `obs084a_partition_feasibility.csv`
- `obs084a_control_feasibility.csv`
- `obs084a_provenance_completeness.csv`
- `obs084a_leakage_field_audit.csv`
- `obs084a_reconnaissance_summary.csv`
- `obs084a_reconnaissance_report.md`

## Limitations

- Filename discovery prioritizes OBS-078–083 and RIG-related artifacts; use
  `--include` for additional sources.
- CSV files may be sampled for reconnaissance. Sampled results must not be used
  as confirmation evidence.
- Schema aliases are heuristics. They do not prove semantic equivalence.
- Cluster counts do not prove statistical independence.
- The script does not inspect hidden or uncommitted artifacts.
- The script does not create or unlock a reserved confirmation partition.
- No FL maturity level is assigned.

## Canonical result statement

OBS-084a reconnaissance audits whether the committed PAM/RIG artifacts contain
sufficient schema, join-key, structural-unit, support-vocabulary, control, and
provenance infrastructure to design a frozen discovery/confirmation study. It
produces feasibility evidence only and establishes no direct failure support,
causal origin, repair target, actionability, external generalization, or formal
topology.
