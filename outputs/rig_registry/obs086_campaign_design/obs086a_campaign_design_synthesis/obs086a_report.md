# OBS-086a — Campaign Design Synthesis

## State

`campaign_design_synthesis_completed`

OBS-086a deterministically synthesizes prospective campaign-design candidates from the frozen OBS-085d artifacts. No new simulation, threshold modification, gate modification, or observed-evidence evaluation was performed.

## Frozen lineage

- OBS-085d commit: `e78160bc6f88c7edce45dee83755c8b7caea7d3f`
- OBS-085d manifest ID: `32884243ca122cf8b88a39d9511b157da02e8456dfa11eec47f7c647bd018023`
- OBS-085d manifest SHA256: `701bba1cd6973f61169b5301ab0c34df870e6abca9d192639c602b8368e8362d`
- OBS-085d script SHA256: `4ac0b63d6784388f75546893c2e74eaf09589188ad8dac82a193402cf4ddfe6d`
- OBS-085d output artifacts validated: **15**
- Current repository HEAD: `e78160bc6f88c7edce45dee83755c8b7caea7d3f`

## Design contract

- Tested nominal-support grid: **[3, 4, 5, 6, 8, 10, 12]**
- Conditional gate-passage reliability targets: **[0.5, 0.8, 0.9]**
- Simulator robustness: minimum conditional gate-passage probability across both qualified simulators at each tested k.
- Partition requirement: discovery and confirmation must each independently reach the target.
- Extrapolation beyond the tested support grid: **not performed**.
- Materially nonmonotone candidates: **held, not sealed**.

> `delta` and `control_response_lambda` are simulator stress-test axes. They are not observed facts and are not operationally selectable properties of a real campaign.

## Candidate synthesis by reliability target

| reliability_target | paired_scenario_designs | sealed_candidates | addresses_with_candidates |
| --- | --- | --- | --- |
| 0.5 | 150 | 35 | 2 |
| 0.8 | 150 | 29 | 2 |
| 0.9 | 150 | 23 | 2 |

## Design decisions

| reliability_target | paired_design_action | scenario_conditioned_designs |
| --- | --- | --- |
| 0.5 | no_go_simulator_discordance | 1 |
| 0.5 | no_go_under_frozen_contract | 102 |
| 0.5 | outside_tested_reliability_envelope_no_extrapolation | 12 |
| 0.5 | seal_for_targeted_design_evaluation | 35 |
| 0.8 | no_go_partition_discordance | 3 |
| 0.8 | no_go_under_frozen_contract | 102 |
| 0.8 | outside_tested_reliability_envelope_no_extrapolation | 16 |
| 0.8 | seal_for_targeted_design_evaluation | 29 |
| 0.9 | no_go_partition_discordance | 6 |
| 0.9 | no_go_under_frozen_contract | 102 |
| 0.9 | outside_tested_reliability_envelope_no_extrapolation | 19 |
| 0.9 | seal_for_targeted_design_evaluation | 23 |

## Address-level candidate families

| address_id | record_id | carrier | entitlement_status | reliability_target | tested_scenario_cells | sealed_candidate_cells | sealed_candidate_share | minimum_total_nominal_objects_among_candidates | maximum_total_nominal_objects_among_candidates | candidate_family_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 03ef63db7584762143dc3914 | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 25 | 18 | 0.72 | 13 | 20 | sealed_candidate_for_restricted_scenario_subset |
| 03ef63db7584762143dc3914 | three_way__no_window | no_window | fl3_entitlement_capped | 0.8 | 25 | 14 | 0.56 | 16 | 20 | sealed_candidate_for_restricted_scenario_subset |
| 03ef63db7584762143dc3914 | three_way__no_window | no_window | fl3_entitlement_capped | 0.9 | 25 | 10 | 0.4 | 20 | 20 | sealed_candidate_for_restricted_scenario_subset |
| 1cce3e5fe7c50070e58676c9 | Cp2_vs_Cp3__strict_numeric_all | strict_numeric_all | fl3_entitlement_capped | 0.5 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| 1cce3e5fe7c50070e58676c9 | Cp2_vs_Cp3__strict_numeric_all | strict_numeric_all | fl3_entitlement_capped | 0.8 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| 1cce3e5fe7c50070e58676c9 | Cp2_vs_Cp3__strict_numeric_all | strict_numeric_all | fl3_entitlement_capped | 0.9 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| 1dbc920edffba9363b404b4e | Cp2_vs_Cp3__stability_plus_geometry | stability_plus_geometry | fl3_entitled | 0.5 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| 1dbc920edffba9363b404b4e | Cp2_vs_Cp3__stability_plus_geometry | stability_plus_geometry | fl3_entitled | 0.8 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| 1dbc920edffba9363b404b4e | Cp2_vs_Cp3__stability_plus_geometry | stability_plus_geometry | fl3_entitled | 0.9 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| bfa569fbae807b7f50c3389f | C_vs_Cp2__stability_core_3 | stability_core_3 | fl3_entitled | 0.5 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| bfa569fbae807b7f50c3389f | C_vs_Cp2__stability_core_3 | stability_core_3 | fl3_entitled | 0.8 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| bfa569fbae807b7f50c3389f | C_vs_Cp2__stability_core_3 | stability_core_3 | fl3_entitled | 0.9 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| d10b3aec4edc47fefb2bd92f | C_vs_Cp3__geometry_scores_only | geometry_scores_only | fl3_entitlement_capped | 0.5 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| d10b3aec4edc47fefb2bd92f | C_vs_Cp3__geometry_scores_only | geometry_scores_only | fl3_entitlement_capped | 0.8 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| d10b3aec4edc47fefb2bd92f | C_vs_Cp3__geometry_scores_only | geometry_scores_only | fl3_entitlement_capped | 0.9 | 25 | 0 | 0 |  |  | no_sealed_candidate_under_tested_envelope |
| fe17c18b39eb86c6e9f982bd | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 25 | 17 | 0.68 | 10 | 24 | sealed_candidate_for_restricted_scenario_subset |
| fe17c18b39eb86c6e9f982bd | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 25 | 15 | 0.6 | 12 | 24 | sealed_candidate_for_restricted_scenario_subset |
| fe17c18b39eb86c6e9f982bd | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 25 | 13 | 0.52 | 16 | 24 | sealed_candidate_for_restricted_scenario_subset |

## Support attrition

| partition | reliability_target | attrition_class | address_profiles |
| --- | --- | --- | --- |
| confirmation | 0.5 | lower_attrition_within_observed_envelope | 1 |
| confirmation | 0.5 | moderate_attrition | 1 |
| confirmation | 0.5 | not_applicable_no_candidate | 4 |
| confirmation | 0.8 | lower_attrition_within_observed_envelope | 1 |
| confirmation | 0.8 | moderate_attrition | 1 |
| confirmation | 0.8 | not_applicable_no_candidate | 4 |
| confirmation | 0.9 | lower_attrition_within_observed_envelope | 1 |
| confirmation | 0.9 | moderate_attrition | 1 |
| confirmation | 0.9 | not_applicable_no_candidate | 4 |
| discovery | 0.5 | lower_attrition_within_observed_envelope | 2 |
| discovery | 0.5 | not_applicable_no_candidate | 4 |
| discovery | 0.8 | lower_attrition_within_observed_envelope | 2 |
| discovery | 0.8 | not_applicable_no_candidate | 4 |
| discovery | 0.9 | lower_attrition_within_observed_envelope | 2 |
| discovery | 0.9 | not_applicable_no_candidate | 4 |

Nominal support is not treated as effective support. OBS-086a carries the frozen OBS-085d support-efficiency estimates into every candidate allocation and performs no numeric extrapolation beyond k=12.

## Entitlement overlay

| entitlement_status | reliability_target | paired_design_action | scenario_conditioned_designs | sealed_candidate_designs |
| --- | --- | --- | --- | --- |
| fl3_entitled | 0.5 | no_go_simulator_discordance | 1 | 0 |
| fl3_entitled | 0.5 | no_go_under_frozen_contract | 51 | 0 |
| fl3_entitled | 0.5 | outside_tested_reliability_envelope_no_extrapolation | 6 | 0 |
| fl3_entitled | 0.5 | seal_for_targeted_design_evaluation | 17 | 17 |
| fl3_entitled | 0.8 | no_go_under_frozen_contract | 51 | 0 |
| fl3_entitled | 0.8 | outside_tested_reliability_envelope_no_extrapolation | 9 | 0 |
| fl3_entitled | 0.8 | seal_for_targeted_design_evaluation | 15 | 15 |
| fl3_entitled | 0.9 | no_go_partition_discordance | 1 | 0 |
| fl3_entitled | 0.9 | no_go_under_frozen_contract | 51 | 0 |
| fl3_entitled | 0.9 | outside_tested_reliability_envelope_no_extrapolation | 10 | 0 |
| fl3_entitled | 0.9 | seal_for_targeted_design_evaluation | 13 | 13 |
| fl3_entitlement_capped | 0.5 | no_go_under_frozen_contract | 51 | 0 |
| fl3_entitlement_capped | 0.5 | outside_tested_reliability_envelope_no_extrapolation | 6 | 0 |
| fl3_entitlement_capped | 0.5 | seal_for_targeted_design_evaluation | 18 | 18 |
| fl3_entitlement_capped | 0.8 | no_go_partition_discordance | 3 | 0 |
| fl3_entitlement_capped | 0.8 | no_go_under_frozen_contract | 51 | 0 |
| fl3_entitlement_capped | 0.8 | outside_tested_reliability_envelope_no_extrapolation | 7 | 0 |
| fl3_entitlement_capped | 0.8 | seal_for_targeted_design_evaluation | 14 | 14 |
| fl3_entitlement_capped | 0.9 | no_go_partition_discordance | 5 | 0 |
| fl3_entitlement_capped | 0.9 | no_go_under_frozen_contract | 51 | 0 |
| fl3_entitlement_capped | 0.9 | outside_tested_reliability_envelope_no_extrapolation | 9 | 0 |
| fl3_entitlement_capped | 0.9 | seal_for_targeted_design_evaluation | 10 | 10 |

## Protocol decision rules

| rule_order | condition_id | decision | rationale |
| --- | --- | --- | --- |
| 1 | lineage_or_schema_invalid | invalidate | Design synthesis cannot proceed from unverified evidence artifacts. |
| 2 | partition_boundary_breached | invalidate | Partition independence is a frozen requirement. |
| 3 | paired_robust_target_reached | seal_for_targeted_design_evaluation | The design is inside the tested envelope under conservative simulator and partition rules. |
| 4 | material_nonmonotonicity | hold_for_nonmonotonicity_review | A nominal-support recommendation is unstable across tested k. |
| 5 | partition_discordant_target_reach | no_go_partition_discordance | Discovery performance cannot substitute for confirmation performance. |
| 6 | simulator_discordant_target_reach | no_go_simulator_discordance | The design is not robust across the qualified simulator family. |
| 7 | target_not_reached_by_max_tested_k | outside_tested_reliability_envelope_no_extrapolation | OBS-086a does not numerically extrapolate nominal support beyond the frozen tested grid. |
| 8 | empirically_never_passable | no_go_under_frozen_contract | Support expansion alone did not establish passage in the tested envelope. |
| 9 | future_effective_support_shortfall | apply_pre_registered_continue_or_futility_rule | Replacement or continuation must not depend on observed effect direction, magnitude, or gate passage. |

## Output counts

- Partition-specific design rows: **900**
- Paired partition design rows: **450**
- Sealed scenario-conditioned candidates: **87**
- Address decision profiles: **18**
- Failures: **0**

## Interpretation boundary

> OBS-086a is prospective design synthesis only.

> A sealed candidate is not observed evidence, is not a guarantee of future passage, and does not authorize post hoc object replacement.

> Discovery and confirmation may not be pooled, and candidate status does not justify weakening any frozen evidence gate.

> The study cannot create an FL3 witness, establish causal attribution, validate simulator truth, or increase claim entitlement.
