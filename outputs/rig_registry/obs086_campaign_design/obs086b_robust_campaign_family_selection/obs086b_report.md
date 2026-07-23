# OBS-086b — Robust Campaign Family Selection

## State

`robust_campaign_family_selection_completed`

OBS-086b deterministically collapses the frozen OBS-086a scenario-conditioned candidates into fixed operational allocation families, preserves their nondominated Pareto frontier, and applies a separate frozen protocol-selection rule. No new simulation, threshold modification, gate modification, interpolation, extrapolation, or observed-evidence evaluation was performed.

## Frozen lineage

- OBS-086a commit: `96704612b3e16fd2b0c56413932aed83ba070493`
- OBS-086a manifest ID: `df6714be8b7e0ccaffd7b9df56e067db14426ee98f60c0aefb98bab3da2e72d7`
- OBS-086a manifest SHA256: `ab5f3a1c4123d7ac621747b7965c34583097dc970b9b9078978100525207bc91`
- OBS-086a script SHA256: `d2a4724d17774faebf0f93306d6fa34489143fc1545fbe462d301b76eb97d4d6`
- OBS-086a output artifacts validated: **12**
- Frozen partition rows: **900**
- Frozen paired rows: **450**
- Frozen sealed scenario-conditioned rows: **87**
- Current repository HEAD: `96704612b3e16fd2b0c56413932aed83ba070493`

## Operational-family contract

- Family identity: address, reliability target, discovery nominal k, confirmation nominal k.
- Each family is re-evaluated over all 25 frozen stress-test cells for its address and target.
- Discovery and confirmation remain separate; paired probability is their minimum.
- Both partition probabilities are simulator-robust minima from OBS-086a.
- Materially nonmonotone scenario evaluations are holds, not defensible target coverage.
- Stress-test coverage is a fraction of the frozen grid, not a probability over real campaigns.
- No support value outside k = [3, 4, 5, 6, 8, 10, 12] is evaluated.

## Protocol-selection contract

- Preserve every nondominated Pareto family.
- Within each address-target group, identify the maximum defensible stress-cell count.
- Enter families within **1** cell of that maximum into the near-maximum band.
- Require defensible stress-test coverage of at least **0.50** for advancement.
- Select one advancement family lexicographically by minimum total objects, minimum partition imbalance, highest minimum probability over defensible cells, highest median all-scenario probability, lowest partition gap, lowest simulator spread, and stable family ID.
- When the group maximum remains below majority coverage, retain one deterministic low-coverage hold rather than advancing it.

## Family synthesis by reliability target

| reliability_target | operational_families | nondominated_families | protocol_selected_families | protocol_held_families | addresses_with_families | maximum_stress_test_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 9 | 9 | 2 | 0 | 2 | 0.72 |
| 0.8 | 9 | 9 | 2 | 0 | 2 | 0.6 |
| 0.9 | 6 | 6 | 1 | 1 | 2 | 0.52 |

## Nondominated Pareto frontier

| operational_family_id | record_id | carrier | entitlement_status | reliability_target | discovery_nominal_k | confirmation_nominal_k | total_nominal_objects | defensible_target_reaching_scenario_cells | defensible_stress_test_coverage | coverage_class | worst_case_paired_probability_all_scenarios | minimum_paired_probability_defensible_scenarios | median_paired_probability_all_scenarios | partition_allocation_imbalance | selection_roles_json | protocol_selection_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OF-3d71624a7ef373c0600f420c | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 5 | 8 | 13 | 11 | 0.44 | restricted_tested_stress_coverage | 0 | 0.547 | 0.479 | 3 | ["cost_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-17a97f86750bc6e8b305317a | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 6 | 8 | 14 | 14 | 0.56 | majority_tested_stress_coverage | 0 | 0.548 | 0.57 | 2 | ["partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-0f7d22d4a500b33e49584c7a | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 6 | 10 | 16 | 15 | 0.6 | majority_tested_stress_coverage | 0 | 0.601 | 0.669 | 4 | [] | pareto_frontier_outside_near_maximum_band |
| OF-311882fce98aca90bae5f40e | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 8 | 10 | 18 | 17 | 0.68 | majority_tested_stress_coverage | 0 | 0.503 | 0.766 | 2 | ["partition_balance_anchor"] | protocol_selected_for_preregistration_review |
| OF-839130280c26f74afd06f2f7 | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 8 | 12 | 20 | 18 | 0.72 | majority_tested_stress_coverage | 0 | 0.532 | 0.88 | 4 | ["coverage_anchor"] | pareto_near_maximum_not_protocol_selected |
| OF-ae359db10e2d91b84d4dc8dd | three_way__no_window | no_window | fl3_entitlement_capped | 0.8 | 6 | 10 | 16 | 5 | 0.2 | restricted_tested_stress_coverage | 0 | 0.806 | 0.669 | 4 | ["cost_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-57663c3c1e833458f2127059 | three_way__no_window | no_window | fl3_entitlement_capped | 0.8 | 8 | 10 | 18 | 9 | 0.36 | restricted_tested_stress_coverage | 0 | 0.802 | 0.766 | 2 | ["partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-f4d6170f6448c54b7802cf9d | three_way__no_window | no_window | fl3_entitlement_capped | 0.8 | 8 | 12 | 20 | 14 | 0.56 | majority_tested_stress_coverage | 0 | 0.85 | 0.88 | 4 | ["coverage_anchor"] | protocol_selected_for_preregistration_review |
| OF-a2ed80d5454aca252ccbdac3 | three_way__no_window | no_window | fl3_entitlement_capped | 0.9 | 8 | 12 | 20 | 10 | 0.4 | restricted_tested_stress_coverage | 0 | 0.905 | 0.88 | 4 | ["coverage_anchor","cost_anchor","partition_balance_anchor"] | protocol_hold_low_coverage |
| OF-0f0e7dbfb15cd38ad5dfa64a | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 5 | 5 | 10 | 6 | 0.24 | restricted_tested_stress_coverage | 0 | 0.521 | 0.21 | 0 | ["cost_anchor","partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-68801c6e4229fd7d7cb2e71b | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 6 | 6 | 12 | 12 | 0.48 | restricted_tested_stress_coverage | 0 | 0.509 | 0.336 | 0 | ["partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-0f27a07be4cf2153e2888363 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 8 | 8 | 16 | 15 | 0.6 | majority_tested_stress_coverage | 0 | 0.577 | 0.606 | 0 | ["partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-903ef1edcd0001ff716016ad | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 12 | 12 | 24 | 17 | 0.68 | majority_tested_stress_coverage | 0 | 0.535 | 0.904 | 0 | ["coverage_anchor","partition_balance_anchor"] | protocol_selected_for_preregistration_review |
| OF-b4f092907d174f4aa1a9d5e9 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 6 | 6 | 12 | 1 | 0.04 | restricted_tested_stress_coverage | 0 | 0.817 | 0.336 | 0 | ["cost_anchor","partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-09e506e18b89dbd85c9b4161 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 6 | 8 | 14 | 2 | 0.08 | restricted_tested_stress_coverage | 0 | 0.803 | 0.342 | 2 | [] | pareto_frontier_outside_near_maximum_band |
| OF-ea727157ac063c776db25af3 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 8 | 8 | 16 | 9 | 0.36 | restricted_tested_stress_coverage | 0 | 0.809 | 0.606 | 0 | ["partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-613e84e39a5d41b0728fca09 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 8 | 10 | 18 | 11 | 0.44 | restricted_tested_stress_coverage | 0 | 0.839 | 0.642 | 2 | [] | pareto_frontier_outside_near_maximum_band |
| OF-79c405489a64294bca87ffb2 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 10 | 10 | 20 | 14 | 0.56 | majority_tested_stress_coverage | 0 | 0.802 | 0.805 | 0 | ["partition_balance_anchor"] | protocol_selected_for_preregistration_review |
| OF-a79731686e85a5f29621e700 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 12 | 12 | 24 | 15 | 0.6 | majority_tested_stress_coverage | 0 | 0.811 | 0.904 | 0 | ["coverage_anchor","partition_balance_anchor"] | pareto_near_maximum_not_protocol_selected |
| OF-b131d0eb3f1508fae3a1728e | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 8 | 8 | 16 | 5 | 0.2 | restricted_tested_stress_coverage | 0 | 0.913 | 0.606 | 0 | ["cost_anchor","partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-278a8f327cec4c8ae6239327 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 8 | 10 | 18 | 7 | 0.28 | restricted_tested_stress_coverage | 0 | 0.902 | 0.642 | 2 | [] | pareto_frontier_outside_near_maximum_band |
| OF-5d72af7ad78c658c9621b65e | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 10 | 10 | 20 | 10 | 0.4 | restricted_tested_stress_coverage | 0 | 0.936 | 0.805 | 0 | ["partition_balance_anchor"] | pareto_frontier_outside_near_maximum_band |
| OF-a7fdc0177f08f4b2eb58c8c1 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 10 | 12 | 22 | 11 | 0.44 | restricted_tested_stress_coverage | 0 | 0.912 | 0.829 | 2 | [] | pareto_frontier_outside_near_maximum_band |
| OF-41b1acd0b3d21dd4516efcfc | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 12 | 12 | 24 | 13 | 0.52 | majority_tested_stress_coverage | 0 | 0.904 | 0.904 | 0 | ["coverage_anchor","partition_balance_anchor"] | protocol_selected_for_preregistration_review |

## Protocol-selected campaign families

| operational_family_id | record_id | carrier | entitlement_status | reliability_target | discovery_nominal_k | confirmation_nominal_k | total_nominal_objects | defensible_target_reaching_scenario_cells | maximum_group_defensible_cells | defensible_cell_shortfall_from_group_maximum | defensible_stress_test_coverage | minimum_paired_probability_defensible_scenarios | median_paired_probability_all_scenarios | partition_allocation_imbalance | protocol_selection_rank | protocol_selection_status | protocol_selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OF-311882fce98aca90bae5f40e | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 8 | 10 | 18 | 17 | 18 | 1 | 0.68 | 0.503 | 0.766 | 2 | 1 | protocol_selected_for_preregistration_review | within_one_stress_cell_of_group_maximum; majority_coverage; lexicographically_best_under_frozen_rule |
| OF-f4d6170f6448c54b7802cf9d | three_way__no_window | no_window | fl3_entitlement_capped | 0.8 | 8 | 12 | 20 | 14 | 14 | 0 | 0.56 | 0.85 | 0.88 | 4 | 1 | protocol_selected_for_preregistration_review | within_one_stress_cell_of_group_maximum; majority_coverage; lexicographically_best_under_frozen_rule |
| OF-903ef1edcd0001ff716016ad | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 12 | 12 | 24 | 17 | 17 | 0 | 0.68 | 0.535 | 0.904 | 0 | 1 | protocol_selected_for_preregistration_review | within_one_stress_cell_of_group_maximum; majority_coverage; lexicographically_best_under_frozen_rule |
| OF-79c405489a64294bca87ffb2 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 10 | 10 | 20 | 14 | 15 | 1 | 0.56 | 0.802 | 0.805 | 0 | 1 | protocol_selected_for_preregistration_review | within_one_stress_cell_of_group_maximum; majority_coverage; lexicographically_best_under_frozen_rule |
| OF-41b1acd0b3d21dd4516efcfc | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 12 | 12 | 24 | 13 | 13 | 0 | 0.52 | 0.904 | 0.904 | 0 | 1 | protocol_selected_for_preregistration_review | within_one_stress_cell_of_group_maximum; majority_coverage; lexicographically_best_under_frozen_rule |

## Protocol holds

| operational_family_id | record_id | carrier | entitlement_status | reliability_target | discovery_nominal_k | confirmation_nominal_k | total_nominal_objects | defensible_target_reaching_scenario_cells | maximum_group_defensible_cells | defensible_cell_shortfall_from_group_maximum | defensible_stress_test_coverage | minimum_paired_probability_defensible_scenarios | median_paired_probability_all_scenarios | partition_allocation_imbalance | protocol_selection_rank | protocol_selection_status | protocol_selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OF-a2ed80d5454aca252ccbdac3 | three_way__no_window | no_window | fl3_entitlement_capped | 0.9 | 8 | 12 | 20 | 10 | 10 | 0 | 0.4 | 0.905 | 0.88 | 4 | 1 | protocol_hold_low_coverage | near_maximum_family_retained_for_review_but_group_maximum_is_below_majority_coverage |

## Address recommendations

| record_id | carrier | entitlement_status | reliability_target | operational_family_count | nondominated_family_count | protocol_selected_family_count | protocol_held_family_count | maximum_defensible_stress_test_coverage | minimum_total_nominal_objects_on_frontier | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 5 | 5 | 1 | 0 | 0.72 | 13 | advance_protocol_selected_family_to_preregistration_review |
| three_way__no_window | no_window | fl3_entitlement_capped | 0.8 | 3 | 3 | 1 | 0 | 0.56 | 16 | advance_protocol_selected_family_to_preregistration_review |
| three_way__no_window | no_window | fl3_entitlement_capped | 0.9 | 1 | 1 | 0 | 1 | 0.4 | 20 | hold_low_coverage_family_for_design_review |
| Cp2_vs_Cp3__strict_numeric_all | strict_numeric_all | fl3_entitlement_capped | 0.5 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| Cp2_vs_Cp3__strict_numeric_all | strict_numeric_all | fl3_entitlement_capped | 0.8 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| Cp2_vs_Cp3__strict_numeric_all | strict_numeric_all | fl3_entitlement_capped | 0.9 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| Cp2_vs_Cp3__stability_plus_geometry | stability_plus_geometry | fl3_entitled | 0.5 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| Cp2_vs_Cp3__stability_plus_geometry | stability_plus_geometry | fl3_entitled | 0.8 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| Cp2_vs_Cp3__stability_plus_geometry | stability_plus_geometry | fl3_entitled | 0.9 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| C_vs_Cp2__stability_core_3 | stability_core_3 | fl3_entitled | 0.5 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| C_vs_Cp2__stability_core_3 | stability_core_3 | fl3_entitled | 0.8 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| C_vs_Cp2__stability_core_3 | stability_core_3 | fl3_entitled | 0.9 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| C_vs_Cp3__geometry_scores_only | geometry_scores_only | fl3_entitlement_capped | 0.5 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| C_vs_Cp3__geometry_scores_only | geometry_scores_only | fl3_entitlement_capped | 0.8 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| C_vs_Cp3__geometry_scores_only | geometry_scores_only | fl3_entitlement_capped | 0.9 | 0 | 0 | 0 | 0 | 0 |  | no_operational_family_from_obs086a_sealed_set |
| C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 4 | 4 | 1 | 0 | 0.68 | 10 | advance_protocol_selected_family_to_preregistration_review |
| C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 6 | 6 | 1 | 0 | 0.6 | 12 | advance_protocol_selected_family_to_preregistration_review |
| C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 5 | 5 | 1 | 0 | 0.52 | 16 | advance_protocol_selected_family_to_preregistration_review |

## Dominance and protocol result

- Pairwise dominance relations: **0**
- Nondominated Pareto families: **24**
- Protocol-selected families: **5**
- Low-coverage protocol holds: **1**
- Rejected or dominated families: **0**

The all-scenario worst-case probability is retained as a global fragility diagnostic. When it is constant within a comparison group, it is not labeled as a selection anchor. The minimum probability over defensible scenarios is used only after the frozen near-maximum coverage and majority-coverage rules.

## Protocol decision rules

| rule_order | condition_id | decision | rationale |
| --- | --- | --- | --- |
| 1 | lineage_or_schema_invalid | invalidate | Family selection cannot proceed from unverified OBS-086a artifacts. |
| 2 | partition_boundary_breached | invalidate | Discovery and confirmation independence is frozen. |
| 3 | allocation_outside_tested_k_grid | invalidate | OBS-086b does not interpolate or extrapolate support. |
| 4 | origin_candidate_material_nonmonotonicity | reject_family | OBS-086a holds materially nonmonotone candidates rather than sealing them. |
| 5 | zero_defensible_stress_coverage | reject_family | The fixed allocation does not defensibly reach the target in any tested stress cell. |
| 6 | pareto_dominated | reject_family | Another family is no worse on all frozen dominance objectives and strictly better on at least one. |
| 7 | pareto_nondominated | retain_on_pareto_frontier | The allocation is a non-redundant cost/coverage/reliability/balance trade-off. |
| 8 | within_one_cell_of_group_maximum_and_majority_coverage | enter_protocol_selection_pool | The family is near maximum tested coverage and reaches at least half of the frozen 25-cell grid. |
| 9 | protocol_selection_pool | select_lexicographically | Choose minimum total objects, minimum imbalance, highest minimum defensible probability, highest median probability, lowest partition gap, lowest simulator spread, then stable family ID. |
| 10 | group_maximum_below_majority_coverage | hold_one_low_coverage_family | Retain the lexicographically preferred near-maximum family as a design hold rather than advancing it. |
| 11 | partial_stress_coverage | retain_uncertainty_warning | Stress-test coverage is combinatorial and does not authorize selecting delta or lambda. |
| 12 | future_effective_support_shortfall | apply_preregistered_continue_or_futility_rule | Continuation and replacement must not depend on observed effect direction, magnitude, or passage. |

## Output counts

- Operational families: **24**
- Pareto families retained: **24**
- Protocol-selected families: **5**
- Protocol holds: **1**
- Rejected families: **0**
- Address-target recommendations: **18**
- Failures: **0**

## Interpretation boundary

> OBS-086b is prospective family selection only.

> Delta and control-response lambda remain uncertainty axes; they cannot be selected as campaign properties.

> Pareto retention and protocol selection are distinct. Neither is observed evidence or a guarantee of passage.

> A low-coverage hold is not authorized for preregistration advancement.

> Discovery and confirmation may not be pooled, no frozen gate may be weakened, and claim entitlement remains unchanged.
