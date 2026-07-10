# OBS-081 — Reusable Invariance Registry

## Purpose

OBS-081 converts OBS-080 contract-sensitivity results into an explicit registry of reusable invariants, carrier roles, failure localizations, task-level geometry-needed levels, and repair annotations.

OBS-081 is a synthesis layer. It does not introduce new raw data.

## v2 patch

```text
1. Adds carrier_role.
2. Adds weak_redundant_carrier status.
3. Separates task_geometry_needed_* from carrier_role.
4. Recalibrates non-core carrier status from OBS-080c and OBS-080d evidence.
5. Avoids treating marginal but evidenced carriers as insufficient evidence.
```

## Input artifact manifest

| artifact    | path                                                                   | status   |
|:------------|:-----------------------------------------------------------------------|:---------|
| obs080a_dir | outputs/comparisons/obs080a_stability_core_transform_sensitivity       | ok       |
| obs080b_dir | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity      | ok       |
| obs080c_dir | outputs/comparisons/obs080c_feature_family_contract_sensitivity        | ok       |
| obs080d_dir | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity | ok       |

## Registry status vocabulary

```text
stable_reusable_invariant:
  compact relation survives broadly with little repair pressure

context_sensitive_reusable_invariant:
  relation survives but has localized sensitivity or needs annotation

redundant_reusable_invariant:
  alternate carrier preserves or sharpens a relation already carried by core

weak_redundant_carrier:
  alternate carrier has evidence but lower-tail or projection sensitivity remains

fragile_candidate:
  relation appears above baseline but is not contract-stable enough

accidental_relation:
  relation collapses under tested contracts

insufficient_evidence:
  artifact coverage is missing or unusable
```

## Carrier roles

```text
stability_core_3:
  compact_core_carrier

geometry_scores_only:
  geometry_sharpening_carrier

path_shares_only:
  path_support_carrier

stability_plus_geometry:
  enriched_geometry_carrier

no_window:
  non_window_redundant_carrier

strict_numeric_all:
  strict_numeric_reference_carrier
```

## Relation registry

| relation_id                         | task       | carrier                 | carrier_role                     | rig_status                           |   obs080c_carrier_ba |   obs080d_carrier_mean_ba | task_geometry_needed_level   | task_geometry_needed_label   | repair_recommendation                                                                               |
|:------------------------------------|:-----------|:------------------------|:---------------------------------|:-------------------------------------|---------------------:|--------------------------:|:-----------------------------|:-----------------------------|:----------------------------------------------------------------------------------------------------|
| C_vs_Cp2__stability_core_3          | C_vs_Cp2   | stability_core_3        | compact_core_carrier             | stable_reusable_invariant            |             0.987554 |                  0.982595 | Level 1                      | compact core sufficient      | preserve compact core; no repair needed                                                             |
| C_vs_Cp2__geometry_scores_only      | C_vs_Cp2   | geometry_scores_only    | geometry_sharpening_carrier      | redundant_reusable_invariant         |             1        |                  0.983478 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| C_vs_Cp2__no_window                 | C_vs_Cp2   | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         |             0.961054 |                  0.999971 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| C_vs_Cp2__stability_plus_geometry   | C_vs_Cp2   | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         |             1        |                  0.993302 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| C_vs_Cp2__strict_numeric_all        | C_vs_Cp2   | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         |             0.979592 |                  0.999971 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| C_vs_Cp2__path_shares_only          | C_vs_Cp2   | path_shares_only        | path_support_carrier             | weak_redundant_carrier               |             0.904167 |                  0.898901 | Level 1                      | compact core sufficient      | record as weak redundant carrier; do not promote as primary support; link to localized failure rows |
| C_vs_Cp3__stability_core_3          | C_vs_Cp3   | stability_core_3        | compact_core_carrier             | stable_reusable_invariant            |             0.989796 |                  0.984355 | Level 1                      | compact core sufficient      | preserve compact core; no repair needed                                                             |
| C_vs_Cp3__no_window                 | C_vs_Cp3   | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         |             0.940505 |                  1        | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| C_vs_Cp3__stability_plus_geometry   | C_vs_Cp3   | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         |             0.989796 |                  0.98613  | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| C_vs_Cp3__strict_numeric_all        | C_vs_Cp3   | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         |             0.993056 |                  1        | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| C_vs_Cp3__geometry_scores_only      | C_vs_Cp3   | geometry_scores_only    | geometry_sharpening_carrier      | weak_redundant_carrier               |             0.9625   |                  0.9456   | Level 1                      | compact core sufficient      | record as weak redundant carrier; do not promote as primary support; link to localized failure rows |
| C_vs_Cp3__path_shares_only          | C_vs_Cp3   | path_shares_only        | path_support_carrier             | weak_redundant_carrier               |             0.879453 |                  0.915185 | Level 1                      | compact core sufficient      | record as weak redundant carrier; do not promote as primary support; link to localized failure rows |
| Cp2_vs_Cp3__geometry_scores_only    | Cp2_vs_Cp3 | geometry_scores_only    | geometry_sharpening_carrier      | redundant_reusable_invariant         |             1        |                  0.999094 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| Cp2_vs_Cp3__no_window               | Cp2_vs_Cp3 | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         |             0.966525 |                  1        | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| Cp2_vs_Cp3__path_shares_only        | Cp2_vs_Cp3 | path_shares_only        | path_support_carrier             | redundant_reusable_invariant         |             0.915819 |                  0.921024 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| Cp2_vs_Cp3__stability_plus_geometry | Cp2_vs_Cp3 | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         |             1        |                  0.99582  | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| Cp2_vs_Cp3__strict_numeric_all      | Cp2_vs_Cp3 | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         |             0.966525 |                  1        | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| Cp2_vs_Cp3__stability_core_3        | Cp2_vs_Cp3 | stability_core_3        | compact_core_carrier             | context_sensitive_reusable_invariant |             0.891667 |                  0.885451 | Level 1                      | compact core sufficient      | preserve compact core; add geometry-support annotation; add scale-position sensitivity note         |
| three_way__geometry_scores_only     | three_way  | geometry_scores_only    | geometry_sharpening_carrier      | redundant_reusable_invariant         |             0.907716 |                  0.927454 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| three_way__no_window                | three_way  | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         |             0.889198 |                  0.999899 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| three_way__stability_plus_geometry  | three_way  | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         |             0.972789 |                  0.976956 | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| three_way__strict_numeric_all       | three_way  | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         |             0.962737 |                  0.99988  | Level 1                      | compact core sufficient      | record as redundant carrier; link to compact stability core                                         |
| three_way__path_shares_only         | three_way  | path_shares_only        | path_support_carrier             | weak_redundant_carrier               |             0.796465 |                  0.819453 | Level 1                      | compact core sufficient      | record as weak redundant carrier; do not promote as primary support; link to localized failure rows |
| three_way__stability_core_3         | three_way  | stability_core_3        | compact_core_carrier             | context_sensitive_reusable_invariant |             0.916162 |                  0.87326  | Level 1                      | compact core sufficient      | preserve compact core; prefer enriched geometry for high precision; annotate structural sensitivity |

## Core relations

| relation_id                  | task       | rig_status                           |   obs080c_carrier_ba |   obs080d_carrier_mean_ba |   obs080d_carrier_min_ci95_low |   obs080d_carrier_min_p_above_threshold | task_geometry_needed_level   | task_geometry_needed_label   | failure_notes                                                                                  | repair_recommendation                                                                               |
|:-----------------------------|:-----------|:-------------------------------------|---------------------:|--------------------------:|-------------------------------:|----------------------------------------:|:-----------------------------|:-----------------------------|:-----------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|
| C_vs_Cp2__stability_core_3   | C_vs_Cp2   | stable_reusable_invariant            |             0.987554 |                  0.982595 |                       0.942857 |                                   1     | Level 1                      | compact core sufficient      | C-separating contrast is stable under compact core                                             | preserve compact core; no repair needed                                                             |
| C_vs_Cp3__stability_core_3   | C_vs_Cp3   | stable_reusable_invariant            |             0.989796 |                  0.984355 |                       0.945946 |                                   1     | Level 1                      | compact core sufficient      | C-separating contrast is stable under compact core                                             | preserve compact core; no repair needed                                                             |
| Cp2_vs_Cp3__stability_core_3 | Cp2_vs_Cp3 | context_sensitive_reusable_invariant |             0.891667 |                  0.885451 |                       0.778584 |                                   0.988 | Level 1                      | compact core sufficient      | Cp2_vs_Cp3 remains the sensitive diagnostic pair geometry carrier sharply improves Cp2_vs_Cp3  | preserve compact core; add geometry-support annotation; add scale-position sensitivity note         |
| three_way__stability_core_3  | three_way  | context_sensitive_reusable_invariant |             0.916162 |                  0.87326  |                       0.758017 |                                   0.936 | Level 1                      | compact core sufficient      | three-way compact core is reusable but structurally more sensitive than C-separating contrasts | preserve compact core; prefer enriched geometry for high precision; annotate structural sensitivity |

## Survival matrix summary

| obs      | contract_family       | task       | feature_contract        |   n |   mean_score |   min_score |   max_score |
|:---------|:----------------------|:-----------|:------------------------|----:|-------------:|------------:|------------:|
| OBS-080a | numeric_transform     | C_vs_Cp2   | stability_core_3        |   7 |     0.984153 |    0.975649 |    0.987554 |
| OBS-080a | numeric_transform     | C_vs_Cp3   | stability_core_3        |   7 |     0.987164 |    0.986111 |    0.989796 |
| OBS-080a | numeric_transform     | Cp2_vs_Cp3 | stability_core_3        |   7 |     0.895238 |    0.891667 |    0.916667 |
| OBS-080a | numeric_transform     | three_way  | stability_core_3        |   7 |     0.910606 |    0.899495 |    0.916162 |
| OBS-080b | scale_band            | C_vs_Cp2   | stability_core_3        |   9 |     0.984187 |    0.958333 |    1        |
| OBS-080b | scale_band            | C_vs_Cp3   | stability_core_3        |   9 |     0.989513 |    0.970588 |    1        |
| OBS-080b | scale_band            | Cp2_vs_Cp3 | stability_core_3        |   9 |     0.911728 |    0.833333 |    1        |
| OBS-080b | scale_band            | three_way  | stability_core_3        |   9 |     0.91171  |    0.837535 |    1        |
| OBS-080c | feature_family        | C_vs_Cp2   | geometry_scores_only    |   1 |     1        |    1        |    1        |
| OBS-080c | feature_family        | C_vs_Cp2   | no_geometry             |   1 |     0.981602 |    0.981602 |    0.981602 |
| OBS-080c | feature_family        | C_vs_Cp2   | no_paths                |   1 |     0.982143 |    0.982143 |    0.982143 |
| OBS-080c | feature_family        | C_vs_Cp2   | no_window               |   1 |     0.961054 |    0.961054 |    0.961054 |
| OBS-080c | feature_family        | C_vs_Cp2   | path_shares_only        |   1 |     0.904167 |    0.904167 |    0.904167 |
| OBS-080c | feature_family        | C_vs_Cp2   | stability_core_3        |   1 |     0.987554 |    0.987554 |    0.987554 |
| OBS-080c | feature_family        | C_vs_Cp2   | stability_plus_geometry |   1 |     1        |    1        |    1        |
| OBS-080c | feature_family        | C_vs_Cp2   | stability_plus_paths    |   1 |     0.987554 |    0.987554 |    0.987554 |
| OBS-080c | feature_family        | C_vs_Cp2   | stability_plus_z        |   1 |     1        |    1        |    1        |
| OBS-080c | feature_family        | C_vs_Cp2   | strict_numeric_all      |   1 |     0.979592 |    0.979592 |    0.979592 |
| OBS-080c | feature_family        | C_vs_Cp2   | window_means_only       |   1 |     0.987554 |    0.987554 |    0.987554 |
| OBS-080c | feature_family        | C_vs_Cp2   | window_z_only           |   1 |     0.84632  |    0.84632  |    0.84632  |
| OBS-080c | feature_family        | C_vs_Cp3   | geometry_scores_only    |   1 |     0.9625   |    0.9625   |    0.9625   |
| OBS-080c | feature_family        | C_vs_Cp3   | no_geometry             |   1 |     0.993056 |    0.993056 |    0.993056 |
| OBS-080c | feature_family        | C_vs_Cp3   | no_paths                |   1 |     0.989796 |    0.989796 |    0.989796 |
| OBS-080c | feature_family        | C_vs_Cp3   | no_window               |   1 |     0.940505 |    0.940505 |    0.940505 |
| OBS-080c | feature_family        | C_vs_Cp3   | path_shares_only        |   1 |     0.879453 |    0.879453 |    0.879453 |
| OBS-080c | feature_family        | C_vs_Cp3   | stability_core_3        |   1 |     0.989796 |    0.989796 |    0.989796 |
| OBS-080c | feature_family        | C_vs_Cp3   | stability_plus_geometry |   1 |     0.989796 |    0.989796 |    0.989796 |
| OBS-080c | feature_family        | C_vs_Cp3   | stability_plus_paths    |   1 |     0.993056 |    0.993056 |    0.993056 |
| OBS-080c | feature_family        | C_vs_Cp3   | stability_plus_z        |   1 |     0.989796 |    0.989796 |    0.989796 |
| OBS-080c | feature_family        | C_vs_Cp3   | strict_numeric_all      |   1 |     0.993056 |    0.993056 |    0.993056 |
| OBS-080c | feature_family        | C_vs_Cp3   | window_means_only       |   1 |     0.989796 |    0.989796 |    0.989796 |
| OBS-080c | feature_family        | C_vs_Cp3   | window_z_only           |   1 |     0.793056 |    0.793056 |    0.793056 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | geometry_scores_only    |   1 |     1        |    1        |    1        |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | no_geometry             |   1 |     0.916384 |    0.916384 |    0.916384 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | no_paths                |   1 |     0.983333 |    0.983333 |    0.983333 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | no_window               |   1 |     0.966525 |    0.966525 |    0.966525 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | path_shares_only        |   1 |     0.915819 |    0.915819 |    0.915819 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | stability_core_3        |   1 |     0.891667 |    0.891667 |    0.891667 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | stability_plus_geometry |   1 |     1        |    1        |    1        |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | stability_plus_paths    |   1 |     0.899435 |    0.899435 |    0.899435 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | stability_plus_z        |   1 |     0.916667 |    0.916667 |    0.916667 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | strict_numeric_all      |   1 |     0.966525 |    0.966525 |    0.966525 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | window_means_only       |   1 |     0.891667 |    0.891667 |    0.891667 |
| OBS-080c | feature_family        | Cp2_vs_Cp3 | window_z_only           |   1 |     0.85     |    0.85     |    0.85     |
| OBS-080c | feature_family        | three_way  | geometry_scores_only    |   1 |     0.907716 |    0.907716 |    0.907716 |
| OBS-080c | feature_family        | three_way  | no_geometry             |   1 |     0.90824  |    0.90824  |    0.90824  |
| OBS-080c | feature_family        | three_way  | no_paths                |   1 |     0.962737 |    0.962737 |    0.962737 |
| OBS-080c | feature_family        | three_way  | no_window               |   1 |     0.889198 |    0.889198 |    0.889198 |
| OBS-080c | feature_family        | three_way  | path_shares_only        |   1 |     0.796465 |    0.796465 |    0.796465 |
| OBS-080c | feature_family        | three_way  | stability_core_3        |   1 |     0.916162 |    0.916162 |    0.916162 |
| OBS-080c | feature_family        | three_way  | stability_plus_geometry |   1 |     0.972789 |    0.972789 |    0.972789 |
| OBS-080c | feature_family        | three_way  | stability_plus_paths    |   1 |     0.921717 |    0.921717 |    0.921717 |
| OBS-080c | feature_family        | three_way  | stability_plus_z        |   1 |     0.916667 |    0.916667 |    0.916667 |
| OBS-080c | feature_family        | three_way  | strict_numeric_all      |   1 |     0.962737 |    0.962737 |    0.962737 |
| OBS-080c | feature_family        | three_way  | window_means_only       |   1 |     0.916162 |    0.916162 |    0.916162 |
| OBS-080c | feature_family        | three_way  | window_z_only           |   1 |     0.750505 |    0.750505 |    0.750505 |
| OBS-080d | structural_resampling | C_vs_Cp2   | geometry_scores_only    |   6 |     0.983478 |    0.971729 |    0.996438 |
| OBS-080d | structural_resampling | C_vs_Cp2   | no_window               |   6 |     0.999971 |    0.99986  |    1        |
| OBS-080d | structural_resampling | C_vs_Cp2   | path_shares_only        |   6 |     0.898901 |    0.892202 |    0.910318 |
| OBS-080d | structural_resampling | C_vs_Cp2   | stability_core_3        |   6 |     0.982595 |    0.981528 |    0.9838   |
| OBS-080d | structural_resampling | C_vs_Cp2   | stability_plus_geometry |   6 |     0.993302 |    0.989168 |    0.998103 |
| OBS-080d | structural_resampling | C_vs_Cp2   | strict_numeric_all      |   6 |     0.999971 |    0.99986  |    1        |
| OBS-080d | structural_resampling | C_vs_Cp3   | geometry_scores_only    |   6 |     0.9456   |    0.91789  |    0.973925 |
| OBS-080d | structural_resampling | C_vs_Cp3   | no_window               |   6 |     1        |    1        |    1        |
| OBS-080d | structural_resampling | C_vs_Cp3   | path_shares_only        |   6 |     0.915185 |    0.910186 |    0.921196 |
| OBS-080d | structural_resampling | C_vs_Cp3   | stability_core_3        |   6 |     0.984355 |    0.983241 |    0.985088 |
| OBS-080d | structural_resampling | C_vs_Cp3   | stability_plus_geometry |   6 |     0.98613  |    0.985344 |    0.987013 |
| OBS-080d | structural_resampling | C_vs_Cp3   | strict_numeric_all      |   6 |     1        |    1        |    1        |
| OBS-080d | structural_resampling | Cp2_vs_Cp3 | geometry_scores_only    |   6 |     0.999094 |    0.998395 |    1        |
| OBS-080d | structural_resampling | Cp2_vs_Cp3 | no_window               |   6 |     1        |    1        |    1        |
| OBS-080d | structural_resampling | Cp2_vs_Cp3 | path_shares_only        |   6 |     0.921024 |    0.911773 |    0.926272 |
| OBS-080d | structural_resampling | Cp2_vs_Cp3 | stability_core_3        |   6 |     0.885451 |    0.879211 |    0.89508  |
| OBS-080d | structural_resampling | Cp2_vs_Cp3 | stability_plus_geometry |   6 |     0.99582  |    0.992833 |    0.998757 |
| OBS-080d | structural_resampling | Cp2_vs_Cp3 | strict_numeric_all      |   6 |     1        |    1        |    1        |
| OBS-080d | structural_resampling | three_way  | geometry_scores_only    |   6 |     0.927454 |    0.900048 |    0.964534 |
| OBS-080d | structural_resampling | three_way  | no_window               |   6 |     0.999899 |    0.999669 |    1        |
| OBS-080d | structural_resampling | three_way  | path_shares_only        |   6 |     0.819453 |    0.81451  |    0.830023 |
| OBS-080d | structural_resampling | three_way  | stability_core_3        |   6 |     0.87326  |    0.869845 |    0.87649  |
| OBS-080d | structural_resampling | three_way  | stability_plus_geometry |   6 |     0.976956 |    0.972346 |    0.983724 |
| OBS-080d | structural_resampling | three_way  | strict_numeric_all      |   6 |     0.99988  |    0.9996   |    1        |

## Geometry-needed ladder

| relation_id                         | task       | carrier                 | carrier_role                     | task_geometry_needed_level   | task_geometry_needed_label   |   level_1_stability_core_ba |   level_3_geometry_ba |   level_4_paths_ba |   level_3_stability_plus_geometry_ba |   level_5_no_window_ba |   level_5_strict_numeric_ba |
|:------------------------------------|:-----------|:------------------------|:---------------------------------|:-----------------------------|:-----------------------------|----------------------------:|----------------------:|-------------------:|-------------------------------------:|-----------------------:|----------------------------:|
| three_way__stability_core_3         | three_way  | stability_core_3        | compact_core_carrier             | Level 1                      | compact core sufficient      |                    0.916162 |              0.907716 |           0.796465 |                             0.972789 |               0.889198 |                    0.962737 |
| three_way__geometry_scores_only     | three_way  | geometry_scores_only    | geometry_sharpening_carrier      | Level 1                      | compact core sufficient      |                    0.916162 |              0.907716 |           0.796465 |                             0.972789 |               0.889198 |                    0.962737 |
| three_way__path_shares_only         | three_way  | path_shares_only        | path_support_carrier             | Level 1                      | compact core sufficient      |                    0.916162 |              0.907716 |           0.796465 |                             0.972789 |               0.889198 |                    0.962737 |
| three_way__stability_plus_geometry  | three_way  | stability_plus_geometry | enriched_geometry_carrier        | Level 1                      | compact core sufficient      |                    0.916162 |              0.907716 |           0.796465 |                             0.972789 |               0.889198 |                    0.962737 |
| three_way__no_window                | three_way  | no_window               | non_window_redundant_carrier     | Level 1                      | compact core sufficient      |                    0.916162 |              0.907716 |           0.796465 |                             0.972789 |               0.889198 |                    0.962737 |
| three_way__strict_numeric_all       | three_way  | strict_numeric_all      | strict_numeric_reference_carrier | Level 1                      | compact core sufficient      |                    0.916162 |              0.907716 |           0.796465 |                             0.972789 |               0.889198 |                    0.962737 |
| C_vs_Cp2__stability_core_3          | C_vs_Cp2   | stability_core_3        | compact_core_carrier             | Level 1                      | compact core sufficient      |                    0.987554 |              1        |           0.904167 |                             1        |               0.961054 |                    0.979592 |
| C_vs_Cp2__geometry_scores_only      | C_vs_Cp2   | geometry_scores_only    | geometry_sharpening_carrier      | Level 1                      | compact core sufficient      |                    0.987554 |              1        |           0.904167 |                             1        |               0.961054 |                    0.979592 |
| C_vs_Cp2__path_shares_only          | C_vs_Cp2   | path_shares_only        | path_support_carrier             | Level 1                      | compact core sufficient      |                    0.987554 |              1        |           0.904167 |                             1        |               0.961054 |                    0.979592 |
| C_vs_Cp2__stability_plus_geometry   | C_vs_Cp2   | stability_plus_geometry | enriched_geometry_carrier        | Level 1                      | compact core sufficient      |                    0.987554 |              1        |           0.904167 |                             1        |               0.961054 |                    0.979592 |
| C_vs_Cp2__no_window                 | C_vs_Cp2   | no_window               | non_window_redundant_carrier     | Level 1                      | compact core sufficient      |                    0.987554 |              1        |           0.904167 |                             1        |               0.961054 |                    0.979592 |
| C_vs_Cp2__strict_numeric_all        | C_vs_Cp2   | strict_numeric_all      | strict_numeric_reference_carrier | Level 1                      | compact core sufficient      |                    0.987554 |              1        |           0.904167 |                             1        |               0.961054 |                    0.979592 |
| C_vs_Cp3__stability_core_3          | C_vs_Cp3   | stability_core_3        | compact_core_carrier             | Level 1                      | compact core sufficient      |                    0.989796 |              0.9625   |           0.879453 |                             0.989796 |               0.940505 |                    0.993056 |
| C_vs_Cp3__geometry_scores_only      | C_vs_Cp3   | geometry_scores_only    | geometry_sharpening_carrier      | Level 1                      | compact core sufficient      |                    0.989796 |              0.9625   |           0.879453 |                             0.989796 |               0.940505 |                    0.993056 |
| C_vs_Cp3__path_shares_only          | C_vs_Cp3   | path_shares_only        | path_support_carrier             | Level 1                      | compact core sufficient      |                    0.989796 |              0.9625   |           0.879453 |                             0.989796 |               0.940505 |                    0.993056 |
| C_vs_Cp3__stability_plus_geometry   | C_vs_Cp3   | stability_plus_geometry | enriched_geometry_carrier        | Level 1                      | compact core sufficient      |                    0.989796 |              0.9625   |           0.879453 |                             0.989796 |               0.940505 |                    0.993056 |
| C_vs_Cp3__no_window                 | C_vs_Cp3   | no_window               | non_window_redundant_carrier     | Level 1                      | compact core sufficient      |                    0.989796 |              0.9625   |           0.879453 |                             0.989796 |               0.940505 |                    0.993056 |
| C_vs_Cp3__strict_numeric_all        | C_vs_Cp3   | strict_numeric_all      | strict_numeric_reference_carrier | Level 1                      | compact core sufficient      |                    0.989796 |              0.9625   |           0.879453 |                             0.989796 |               0.940505 |                    0.993056 |
| Cp2_vs_Cp3__stability_core_3        | Cp2_vs_Cp3 | stability_core_3        | compact_core_carrier             | Level 1                      | compact core sufficient      |                    0.891667 |              1        |           0.915819 |                             1        |               0.966525 |                    0.966525 |
| Cp2_vs_Cp3__geometry_scores_only    | Cp2_vs_Cp3 | geometry_scores_only    | geometry_sharpening_carrier      | Level 1                      | compact core sufficient      |                    0.891667 |              1        |           0.915819 |                             1        |               0.966525 |                    0.966525 |
| Cp2_vs_Cp3__path_shares_only        | Cp2_vs_Cp3 | path_shares_only        | path_support_carrier             | Level 1                      | compact core sufficient      |                    0.891667 |              1        |           0.915819 |                             1        |               0.966525 |                    0.966525 |
| Cp2_vs_Cp3__stability_plus_geometry | Cp2_vs_Cp3 | stability_plus_geometry | enriched_geometry_carrier        | Level 1                      | compact core sufficient      |                    0.891667 |              1        |           0.915819 |                             1        |               0.966525 |                    0.966525 |
| Cp2_vs_Cp3__no_window               | Cp2_vs_Cp3 | no_window               | non_window_redundant_carrier     | Level 1                      | compact core sufficient      |                    0.891667 |              1        |           0.915819 |                             1        |               0.966525 |                    0.966525 |
| Cp2_vs_Cp3__strict_numeric_all      | Cp2_vs_Cp3 | strict_numeric_all      | strict_numeric_reference_carrier | Level 1                      | compact core sufficient      |                    0.891667 |              1        |           0.915819 |                             1        |               0.966525 |                    0.966525 |

## Failure localization

| relation_id                    | task      | feature_contract     | carrier_role                | contract_family       | contract_name               |    score |   threshold |       margin | status               | failure_type                         |
|:-------------------------------|:----------|:---------------------|:----------------------------|:----------------------|:----------------------------|---------:|------------:|-------------:|:---------------------|:-------------------------------------|
| three_way__stability_core_3    | three_way | stability_core_3     | compact_core_carrier        | structural_resampling | object_bootstrap            | 0.869845 |         0.8 |  0.0698451   | pass_with_lower_tail | object_support_sensitivity           |
| three_way__stability_core_3    | three_way | stability_core_3     | compact_core_carrier        | structural_resampling | object_transition_bootstrap | 0.870083 |         0.8 |  0.0700835   | pass_with_lower_tail | object_support_sensitivity           |
| three_way__stability_core_3    | three_way | stability_core_3     | compact_core_carrier        | structural_resampling | cohort_bootstrap            | 0.872118 |         0.8 |  0.0721184   | pass_with_lower_tail | cohort_support_sensitivity           |
| three_way__stability_core_3    | three_way | stability_core_3     | compact_core_carrier        | structural_resampling | transition_bootstrap        | 0.874733 |         0.8 |  0.0747325   | pass_with_lower_tail | transition_support_sensitivity       |
| three_way__path_shares_only    | three_way | path_shares_only     | path_support_carrier        | feature_family        | path_shares_only            | 0.796465 |         0.8 | -0.00353535  | borderline           | feature_projection_sensitivity       |
| three_way__path_shares_only    | three_way | path_shares_only     | path_support_carrier        | structural_resampling | object_cohort_bootstrap     | 0.81451  |         0.8 |  0.01451     | pass_with_lower_tail | object_support_sensitivity           |
| three_way__path_shares_only    | three_way | path_shares_only     | path_support_carrier        | structural_resampling | cohort_bootstrap            | 0.815369 |         0.8 |  0.0153687   | pass_with_lower_tail | cohort_support_sensitivity           |
| three_way__path_shares_only    | three_way | path_shares_only     | path_support_carrier        | structural_resampling | transition_bootstrap        | 0.818655 |         0.8 |  0.0186548   | pass_with_lower_tail | transition_support_sensitivity       |
| three_way__path_shares_only    | three_way | path_shares_only     | path_support_carrier        | structural_resampling | row_bootstrap               | 0.818777 |         0.8 |  0.0187774   | pass_with_lower_tail | structural_recomposition_sensitivity |
| three_way__path_shares_only    | three_way | path_shares_only     | path_support_carrier        | structural_resampling | object_transition_bootstrap | 0.819381 |         0.8 |  0.0193811   | pass_with_lower_tail | object_support_sensitivity           |
| three_way__path_shares_only    | three_way | path_shares_only     | path_support_carrier        | structural_resampling | object_bootstrap            | 0.830023 |         0.8 |  0.0300233   | pass_with_lower_tail | object_support_sensitivity           |
| C_vs_Cp2__path_shares_only     | C_vs_Cp2  | path_shares_only     | path_support_carrier        | structural_resampling | cohort_bootstrap            | 0.892202 |         0.9 | -0.00779811  | borderline           | cohort_support_sensitivity           |
| C_vs_Cp2__path_shares_only     | C_vs_Cp2  | path_shares_only     | path_support_carrier        | structural_resampling | object_cohort_bootstrap     | 0.894069 |         0.9 | -0.0059308   | borderline           | object_support_sensitivity           |
| C_vs_Cp2__path_shares_only     | C_vs_Cp2  | path_shares_only     | path_support_carrier        | structural_resampling | row_bootstrap               | 0.895265 |         0.9 | -0.00473486  | borderline           | structural_recomposition_sensitivity |
| C_vs_Cp2__path_shares_only     | C_vs_Cp2  | path_shares_only     | path_support_carrier        | structural_resampling | transition_bootstrap        | 0.89954  |         0.9 | -0.000460049 | borderline           | transition_support_sensitivity       |
| C_vs_Cp2__path_shares_only     | C_vs_Cp2  | path_shares_only     | path_support_carrier        | structural_resampling | object_transition_bootstrap | 0.902013 |         0.9 |  0.00201302  | pass_with_lower_tail | object_support_sensitivity           |
| C_vs_Cp2__path_shares_only     | C_vs_Cp2  | path_shares_only     | path_support_carrier        | structural_resampling | object_bootstrap            | 0.910318 |         0.9 |  0.0103182   | pass_with_lower_tail | object_support_sensitivity           |
| C_vs_Cp3__geometry_scores_only | C_vs_Cp3  | geometry_scores_only | geometry_sharpening_carrier | structural_resampling | cohort_bootstrap            | 0.91789  |         0.9 |  0.0178897   | pass_with_lower_tail | cohort_support_sensitivity           |
| C_vs_Cp3__geometry_scores_only | C_vs_Cp3  | geometry_scores_only | geometry_sharpening_carrier | structural_resampling | row_bootstrap               | 0.921531 |         0.9 |  0.0215309   | pass_with_lower_tail | structural_recomposition_sensitivity |
| C_vs_Cp3__geometry_scores_only | C_vs_Cp3  | geometry_scores_only | geometry_sharpening_carrier | structural_resampling | object_cohort_bootstrap     | 0.923601 |         0.9 |  0.0236014   | pass_with_lower_tail | object_support_sensitivity           |
| C_vs_Cp3__geometry_scores_only | C_vs_Cp3  | geometry_scores_only | geometry_sharpening_carrier | structural_resampling | object_bootstrap            | 0.966582 |         0.9 |  0.0665817   | pass_with_lower_tail | object_support_sensitivity           |
| C_vs_Cp3__geometry_scores_only | C_vs_Cp3  | geometry_scores_only | geometry_sharpening_carrier | structural_resampling | transition_bootstrap        | 0.970074 |         0.9 |  0.0700738   | pass_with_lower_tail | transition_support_sensitivity       |
| C_vs_Cp3__path_shares_only     | C_vs_Cp3  | path_shares_only     | path_support_carrier        | feature_family        | path_shares_only            | 0.879453 |         0.9 | -0.0205465   | borderline           | feature_projection_sensitivity       |
| C_vs_Cp3__path_shares_only     | C_vs_Cp3  | path_shares_only     | path_support_carrier        | structural_resampling | cohort_bootstrap            | 0.910186 |         0.9 |  0.0101863   | pass_with_lower_tail | cohort_support_sensitivity           |
| C_vs_Cp3__path_shares_only     | C_vs_Cp3  | path_shares_only     | path_support_carrier        | structural_resampling | object_cohort_bootstrap     | 0.911714 |         0.9 |  0.0117143   | pass_with_lower_tail | object_support_sensitivity           |
| C_vs_Cp3__path_shares_only     | C_vs_Cp3  | path_shares_only     | path_support_carrier        | structural_resampling | row_bootstrap               | 0.912676 |         0.9 |  0.0126761   | pass_with_lower_tail | structural_recomposition_sensitivity |
| C_vs_Cp3__path_shares_only     | C_vs_Cp3  | path_shares_only     | path_support_carrier        | structural_resampling | object_transition_bootstrap | 0.915471 |         0.9 |  0.0154712   | pass_with_lower_tail | object_support_sensitivity           |
| C_vs_Cp3__path_shares_only     | C_vs_Cp3  | path_shares_only     | path_support_carrier        | structural_resampling | transition_bootstrap        | 0.919863 |         0.9 |  0.0198631   | pass_with_lower_tail | transition_support_sensitivity       |
| C_vs_Cp3__path_shares_only     | C_vs_Cp3  | path_shares_only     | path_support_carrier        | structural_resampling | object_bootstrap            | 0.921196 |         0.9 |  0.0211963   | pass_with_lower_tail | object_support_sensitivity           |

## Repair recommendations

| relation_id                         | task       | carrier                 | carrier_role                     | rig_status                           | repair_recommendation                                                                               | repair_rationale                                                               | failure_notes                                                                                  |
|:------------------------------------|:-----------|:------------------------|:---------------------------------|:-------------------------------------|:----------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| three_way__stability_core_3         | three_way  | stability_core_3        | compact_core_carrier             | context_sensitive_reusable_invariant | preserve compact core; prefer enriched geometry for high precision; annotate structural sensitivity | three-way compact core survives but enriched geometry improves precision       | three-way compact core is reusable but structurally more sensitive than C-separating contrasts |
| three_way__geometry_scores_only     | three_way  | geometry_scores_only    | geometry_sharpening_carrier      | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | three-way compact core is reusable but structurally more sensitive than C-separating contrasts |
| three_way__path_shares_only         | three_way  | path_shares_only        | path_support_carrier             | weak_redundant_carrier               | record as weak redundant carrier; do not promote as primary support; link to localized failure rows | carrier has evidence but lower-tail or projection sensitivity remains          | three-way compact core is reusable but structurally more sensitive than C-separating contrasts |
| three_way__stability_plus_geometry  | three_way  | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | three-way compact core is reusable but structurally more sensitive than C-separating contrasts |
| three_way__no_window                | three_way  | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | three-way compact core is reusable but structurally more sensitive than C-separating contrasts |
| three_way__strict_numeric_all       | three_way  | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | three-way compact core is reusable but structurally more sensitive than C-separating contrasts |
| C_vs_Cp2__stability_core_3          | C_vs_Cp2   | stability_core_3        | compact_core_carrier             | stable_reusable_invariant            | preserve compact core; no repair needed                                                             | compact core is sufficient and structurally stable                             | C-separating contrast is stable under compact core                                             |
| C_vs_Cp2__geometry_scores_only      | C_vs_Cp2   | geometry_scores_only    | geometry_sharpening_carrier      | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | C-separating contrast is stable under compact core                                             |
| C_vs_Cp2__path_shares_only          | C_vs_Cp2   | path_shares_only        | path_support_carrier             | weak_redundant_carrier               | record as weak redundant carrier; do not promote as primary support; link to localized failure rows | carrier has evidence but lower-tail or projection sensitivity remains          | C-separating contrast is stable under compact core                                             |
| C_vs_Cp2__stability_plus_geometry   | C_vs_Cp2   | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | C-separating contrast is stable under compact core                                             |
| C_vs_Cp2__no_window                 | C_vs_Cp2   | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | C-separating contrast is stable under compact core                                             |
| C_vs_Cp2__strict_numeric_all        | C_vs_Cp2   | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | C-separating contrast is stable under compact core                                             |
| C_vs_Cp3__stability_core_3          | C_vs_Cp3   | stability_core_3        | compact_core_carrier             | stable_reusable_invariant            | preserve compact core; no repair needed                                                             | compact core is sufficient and structurally stable                             | C-separating contrast is stable under compact core                                             |
| C_vs_Cp3__geometry_scores_only      | C_vs_Cp3   | geometry_scores_only    | geometry_sharpening_carrier      | weak_redundant_carrier               | record as weak redundant carrier; do not promote as primary support; link to localized failure rows | carrier has evidence but lower-tail or projection sensitivity remains          | C-separating contrast is stable under compact core                                             |
| C_vs_Cp3__path_shares_only          | C_vs_Cp3   | path_shares_only        | path_support_carrier             | weak_redundant_carrier               | record as weak redundant carrier; do not promote as primary support; link to localized failure rows | carrier has evidence but lower-tail or projection sensitivity remains          | C-separating contrast is stable under compact core                                             |
| C_vs_Cp3__stability_plus_geometry   | C_vs_Cp3   | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | C-separating contrast is stable under compact core                                             |
| C_vs_Cp3__no_window                 | C_vs_Cp3   | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | C-separating contrast is stable under compact core                                             |
| C_vs_Cp3__strict_numeric_all        | C_vs_Cp3   | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | C-separating contrast is stable under compact core                                             |
| Cp2_vs_Cp3__stability_core_3        | Cp2_vs_Cp3 | stability_core_3        | compact_core_carrier             | context_sensitive_reusable_invariant | preserve compact core; add geometry-support annotation; add scale-position sensitivity note         | Cp2_vs_Cp3 survives under core but geometry contracts sharpen it substantially | Cp2_vs_Cp3 remains the sensitive diagnostic pair geometry carrier sharply improves Cp2_vs_Cp3  |
| Cp2_vs_Cp3__geometry_scores_only    | Cp2_vs_Cp3 | geometry_scores_only    | geometry_sharpening_carrier      | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | Cp2_vs_Cp3 remains the sensitive diagnostic pair geometry carrier sharply improves Cp2_vs_Cp3  |
| Cp2_vs_Cp3__path_shares_only        | Cp2_vs_Cp3 | path_shares_only        | path_support_carrier             | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | Cp2_vs_Cp3 remains the sensitive diagnostic pair geometry carrier sharply improves Cp2_vs_Cp3  |
| Cp2_vs_Cp3__stability_plus_geometry | Cp2_vs_Cp3 | stability_plus_geometry | enriched_geometry_carrier        | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | Cp2_vs_Cp3 remains the sensitive diagnostic pair geometry carrier sharply improves Cp2_vs_Cp3  |
| Cp2_vs_Cp3__no_window               | Cp2_vs_Cp3 | no_window               | non_window_redundant_carrier     | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | Cp2_vs_Cp3 remains the sensitive diagnostic pair geometry carrier sharply improves Cp2_vs_Cp3  |
| Cp2_vs_Cp3__strict_numeric_all      | Cp2_vs_Cp3 | strict_numeric_all      | strict_numeric_reference_carrier | redundant_reusable_invariant         | record as redundant carrier; link to compact stability core                                         | carrier preserves or sharpens relation outside the compact core                | Cp2_vs_Cp3 remains the sensitive diagnostic pair geometry carrier sharply improves Cp2_vs_Cp3  |

## Canonical OBS-081 interpretation

```text
OBS-081 turns OBS-080 contract-sensitivity into relation-level reusable
invariance records.

C_vs_Cp2 and C_vs_Cp3 register as stable reusable invariants under
the compact stability core.

Cp2_vs_Cp3 registers as a context-sensitive reusable invariant:
the compact core survives, but geometry and broader contracts sharpen
the relation.

three_way registers as reusable but structurally and geometrically enriched.
```

## Scope

```text
OBS-081 summarizes within-table reusable-invariance evidence.
It does not establish external generalization, intervention, or causal control.
```

## Output artifacts

```text
rig_input_manifest.csv
rig_relation_registry.csv
rig_survival_matrix.csv
rig_failure_localization.csv
rig_geometry_needed_ladder.csv
rig_repair_recommendations.csv
rig_registry_report.md
```

---
END OBS-081
