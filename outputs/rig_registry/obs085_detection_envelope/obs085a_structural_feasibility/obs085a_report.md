# OBS-085a — Structural Evidence Feasibility

## State

`structural_evidence_feasibility_completed`

OBS-085a performs a deterministic structural audit only. It does not inject effects, estimate simulated gate-passage probability, compute observed power, or modify the completed OBS-084 result.

## Immutable OBS-084 validation

```text
OBS-084c validation complete: sealed bundle valid; confirmation not opened
Candidates sealed: 13
Candidate manifest ID: 0d58d3abd25677683bb29b25c5b4e1fc2fdd1fab83866893c2151a80b97fd4f5
```

The OBS-084b discovery structural checks were independently reconstructed from the frozen support queries and discovery observation-loss frame. Exact agreement with the frozen matching audit was required before this report could be written.

## Address universe

- Registry records: **24**
- Global unique support templates: **85**
- Record-scoped support definitions: **1,434**
- Predicate-indexed addresses: **5,736**
- Failure predicates: **4**
- Sealed OBS-084b candidates: **13**

## Evidence feasibility

Discovery-feasible addresses: **1,944 / 5,736**
Confirmation-feasible addresses: **1,320 / 5,736**
Evidence-feasible in both partitions: **984 / 5,736**

At the underlying record-scoped support level, feasible addresses represented **486** discovery supports, **330** confirmation supports, and **246** supports feasible in both partitions.

## Structural-state Sankey

[Open the deterministic structural-state Sankey](obs085a_structural_state_sankey.html).

| structural_state              |   address_count |
|:------------------------------|----------------:|
| feasible_in_both_partitions   |             984 |
| discovery_feasible_only       |             960 |
| confirmation_feasible_only    |             336 |
| infeasible_in_both_partitions |            3456 |

The Sankey cross-classifies mutually exclusive discovery and confirmation feasibility states, then overlays deterministic FL3 claim entitlement on the both-feasible branch. It is not a sequential gate-passage, causal-attrition, effect-existence, or detection-probability diagram. Authoritative link values are stored in `obs085a_structural_state_sankey_links.csv`.

## Gate-failure audit

| partition    | gate                                 |   failed_addresses |   passed_addresses |
|:-------------|:-------------------------------------|-------------------:|-------------------:|
| confirmation | G6 matched-complement admissibility  |               4416 |               1320 |
| confirmation | G5 class-bearing coverage            |               4176 |               1560 |
| confirmation | G9 outcome estimability              |               4176 |               1560 |
| confirmation | G8 joint target-control estimability |               4080 |               1656 |
| confirmation | G7 control availability              |               3816 |               1920 |
| confirmation | G3 support-cluster coverage          |               3048 |               2688 |
| confirmation | G1 support presence                  |               2088 |               3648 |
| confirmation | G2 complement presence               |                  0 |               5736 |
| confirmation | G4 complement-cluster coverage       |                  0 |               5736 |
| confirmation | G10 multiplicity-family definition   |                  0 |               5736 |
| discovery    | G6 matched-complement admissibility  |               3792 |               1944 |
| discovery    | G7 control availability              |               2760 |               2976 |
| discovery    | G8 joint target-control estimability |               2760 |               2976 |
| discovery    | G5 class-bearing coverage            |               2496 |               3240 |
| discovery    | G9 outcome estimability              |               2496 |               3240 |
| discovery    | G3 support-cluster coverage          |               2016 |               3720 |
| discovery    | G4 complement-cluster coverage       |                 24 |               5712 |
| discovery    | G1 support presence                  |                  0 |               5736 |
| discovery    | G2 complement presence               |                  0 |               5736 |
| discovery    | G10 multiplicity-family definition   |                  0 |               5736 |

## Structural gate relationships

| partition    | comparison                        |   mismatched_addresses |
|:-------------|:----------------------------------|-----------------------:|
| confirmation | G5 vs G9                          |                      0 |
| confirmation | G7 vs G8                          |                    264 |
| confirmation | G6 vs EvidenceFeasible            |                      0 |
| confirmation | Pass all non-G6 gates             |                   1368 |
| confirmation | Pass all non-G6 gates and fail G6 |                     48 |
| discovery    | G5 vs G9                          |                      0 |
| discovery    | G7 vs G8                          |                      0 |
| discovery    | G6 vs EvidenceFeasible            |                      0 |
| discovery    | Pass all non-G6 gates             |                   1944 |
| discovery    | Pass all non-G6 gates and fail G6 |                      0 |

Under the frozen OBS-084 evidence structure, G6 exactly delimited the evidence-feasible set in both partitions. G5 and G9 were empirically coextensive, while G7 and G8 separated only in confirmation. These are observed relationships in this evidence spine, not claims that the gates are conceptually interchangeable.

## Predicate-level structural envelope

| partition    | failure_predicate                     |   addresses |   class_coverage_pass |   outcome_estimability_pass |   matching_pass |   control_available |   jointly_estimable |   evidence_feasible |
|:-------------|:--------------------------------------|------------:|----------------------:|----------------------------:|----------------:|--------------------:|--------------------:|--------------------:|
| confirmation | local_criterion_breach                |        1434 |                   390 |                         390 |             330 |                 480 |                 414 |                 330 |
| confirmation | log_loss_attenuation                  |        1434 |                   390 |                         390 |             330 |                 480 |                 414 |                 330 |
| confirmation | measurement_missingness_concentration |        1434 |                   390 |                         390 |             330 |                 480 |                 414 |                 330 |
| confirmation | relation_separation_attenuation       |        1434 |                   390 |                         390 |             330 |                 480 |                 414 |                 330 |
| discovery    | local_criterion_breach                |        1434 |                   810 |                         810 |             486 |                 744 |                 744 |                 486 |
| discovery    | log_loss_attenuation                  |        1434 |                   810 |                         810 |             486 |                 744 |                 744 |                 486 |
| discovery    | measurement_missingness_concentration |        1434 |                   810 |                         810 |             486 |                 744 |                 744 |                 486 |
| discovery    | relation_separation_attenuation       |        1434 |                   810 |                         810 |             486 |                 744 |                 744 |                 486 |

All failure predicates had identical structural pass counts. This reflects coextensive evidence availability under the frozen artifacts, not equivalence of predicate behavior, effect scale, or detectability.

## Confirmation focal-support row-floor diagnostic

**48** predicate-indexed addresses across **12** record-scoped supports and **6** records passed every other gate but failed G6.

The failed incremental G6 checks were: site rows = **48**, complement rows = **0**, shared clusters = **0**.

These cases are specifically focal-support-row-limited: their complements, shared-cluster structure, class coverage, controls, outcomes, and multiplicity definitions remained admissible.

|   hypothetical_min_site_rows |   additional_record_scoped_supports |   additional_predicate_indexed_addresses |
|-----------------------------:|------------------------------------:|-----------------------------------------:|
|                            8 |                                   0 |                                        0 |
|                            7 |                                   6 |                                       24 |
|                            6 |                                  12 |                                       48 |

The table above is a gate-relaxation diagnostic only. It does not recommend changing the frozen 8-row requirement.

## Structural classes

| partition    | evidence_class             |   address_count |
|:-------------|:---------------------------|----------------:|
| confirmation | multiple_structural_limits |            4368 |
| confirmation | evidence_feasible          |            1320 |
| confirmation | matching_limited           |              48 |
| discovery    | multiple_structural_limits |            3792 |
| discovery    | evidence_feasible          |            1944 |

## Claim-entitlement overlay

Claim entitlement is not included in `EvidenceFeasible`. It is a deterministic record-level overlay inherited from OBS-083/084.

| entitlement_status     |   record_count |
|:-----------------------|---------------:|
| fl3_entitled           |             12 |
| fl3_entitlement_capped |             12 |

Addresses that are structurally feasible in both partitions but remain FL3-entitlement capped: **456**

## Sealed OBS-084 candidate structural ceiling

| measure                        |   value |
|:-------------------------------|--------:|
| sealed_candidates              |      13 |
| confirmation_feasible          |       8 |
| both_partitions_feasible       |       8 |
| fl3_entitled                   |       1 |
| both_feasible_and_fl3_entitled |       0 |

Within the sealed 13-candidate family, 8 candidates retained confirmation structural feasibility. FL3 entitlement applied to 1 members of the sealed family, while 0 members were simultaneously feasible in both partitions and FL3-entitled. This is a deterministic structural ceiling, not a reinterpretation of the realized confirmation contrasts.

## Sealed OBS-084 candidate context

| record_id                         | failure_predicate               | support_definition                                                      | discovery_evidence_feasible   | confirmation_evidence_feasible   | end_to_end_evidence_feasible   | entitlement_status     | obs084c_confirmation_status                     |
|:----------------------------------|:--------------------------------|:------------------------------------------------------------------------|:------------------------------|:---------------------------------|:-------------------------------|:-----------------------|:------------------------------------------------|
| C_vs_Cp2__geometry_scores_only    | relation_separation_attenuation | scale_band:scale_band=early                                             | True                          | False                            | False                          | fl3_entitlement_capped | confirmation_complement_inadmissible            |
| C_vs_Cp2__geometry_scores_only    | local_criterion_breach          | scale_band:scale_band=early                                             | True                          | False                            | False                          | fl3_entitlement_capped | confirmation_complement_inadmissible            |
| C_vs_Cp2__geometry_scores_only    | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_signal_absent                      |
| C_vs_Cp2__no_window               | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_multiplicity_not_survived          |
| C_vs_Cp2__no_window               | local_criterion_breach          | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_signal_absent                      |
| C_vs_Cp2__no_window               | log_loss_attenuation            | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_multiplicity_not_survived          |
| C_vs_Cp2__stability_plus_geometry | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_signal_absent                      |
| C_vs_Cp2__strict_numeric_all      | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_uncertain_cluster_support          |
| C_vs_Cp3__no_window               | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_reproduced_but_claim_capped_at_fl2 |
| C_vs_Cp3__strict_numeric_all      | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near | True                          | True                             | True                           | fl3_entitlement_capped | confirmation_control_explained                  |
| C_vs_Cp3__geometry_scores_only    | relation_separation_attenuation | transition:transition=5→6 AND scale_band:scale_band=middle              | True                          | False                            | False                          | fl3_entitlement_capped | confirmation_complement_inadmissible            |
| C_vs_Cp3__geometry_scores_only    | log_loss_attenuation            | transition:transition=5→6 AND scale_band:scale_band=middle              | True                          | False                            | False                          | fl3_entitlement_capped | confirmation_complement_inadmissible            |
| three_way__geometry_scores_only   | log_loss_attenuation            | transition:transition=5→6 AND scale_band:scale_band=middle              | True                          | False                            | False                          | fl3_entitled           | confirmation_complement_inadmissible            |

## Input manifest

| artifact_role                   | artifact_path                                                                                               |   size_bytes | sha256                                                           |
|:--------------------------------|:------------------------------------------------------------------------------------------------------------|-------------:|:-----------------------------------------------------------------|
| obs083_carrier_controls         | outputs/rig_registry/obs083_negative_control_localization/obs083_carrier_control_contrast.csv               |        47014 | 58f0451b2b321fc4c192464134fb70566404ed4d169e41dee89204d90f5c37c5 |
| obs083_relation_controls        | outputs/rig_registry/obs083_negative_control_localization/obs083_relation_control_contrast.csv              |        25306 | d8362e49c97d1ba8566ab9704b400139e0315c5dd7f6201be891deb8032c70a9 |
| obs083_subclasses               | outputs/rig_registry/obs083_negative_control_localization/obs083_diagnostic_subclass_assignments.csv        |        19751 | c3fd28c1bbc31dde900cedf9ed6ea3b3f40cddbfb64d642e5521461b5a186701 |
| obs084a_freeze_manifest         | outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution/obs084a_freeze_manifest.json           |         3097 | 03b06c37945d82655814548595281264ef46faa62e40d9a130c8dec46dc0aa8a |
| obs084b_candidate_manifest_csv  | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_candidate_freeze_manifest.csv          |        48119 | 06575ec3bf85143f5ef3c11d18ae785f81019ae5470e2fa587c2ca0da5773f0f |
| obs084b_candidate_manifest_json | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_candidate_freeze_manifest.json         |        63130 | 4deb25d622ab1ecb377d32468975aea6c332f18c2a88b64aa3cc1c67fef29167 |
| obs084b_observation_losses      | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_discovery_observation_losses.csv       |      3691621 | c786394ad527d7879f0bf0e99a2a751868edd0f00f973374ffcff5ae3b860405 |
| obs084b_opening_lock            | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084c_confirmation_opening_lock.json         |         1004 | f47e62b8f67de9ba2678ff6c37c4c735487ee62ed8122415dd8e78e757a3bc3e |
| obs084b_support_inventory       | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_support_candidate_inventory.csv        |       641473 | dcc01eb3a2402cce13fa37431ea7fd61068f61a10e77a77c99be7a37c8f7deba |
| obs084b_support_matching        | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_support_complement_matching.csv        |      3815132 | cba3817a2ecd9b23a5feb3cf9929905c7ab222f8600a289aadfc9e1a97971fee |
| obs084c_candidate_outcomes      | outputs/rig_registry/obs084_direct_failure_witness/confirmation/obs084c_candidate_outcomes.csv              |        11498 | 8e2a2aaacdfbde875470075f04f51d5f2f1348c184c70aed26299eac9871dd9c |
| obs084c_confirmation_manifest   | outputs/rig_registry/obs084_direct_failure_witness/confirmation/obs084c_confirmation_manifest.json          |        37140 | 101cd3bf0db74f933a9eb4b845685812bdf0562988903fe18f95c23c271aacd4 |
| obs084c_observation_losses      | outputs/rig_registry/obs084_direct_failure_witness/confirmation/obs084c_confirmation_observation_losses.csv |      3188859 | cd5b2055126c9eafe4bc2348484396f544f35011bb26674d08e0d5665ae4e5d4 |
| obs084c_support_validation      | outputs/rig_registry/obs084_direct_failure_witness/confirmation/obs084c_support_complement_validation.csv   |         8739 | af238f6c87cf1de253fe46d7a08370451f76648410652b8ee5184d44f2bf97ae |
| obs084c_validation_script       | experiments/studies/obs084c_direct_failure_support_confirmation.py                                          |        76840 | 5090aefa7dfde231a95ed65982af45c8f9af40cb323cba84ba199590b8c36c9c |
| obs085_protocol                 | docs/05_project/085_failure_support_detection_power_and_confirmation_feasibility_protocol.md                |        62252 | b5c852ab7c8abbc8f497b8e4efd0768fb0b45a878ebcef0c8446b3ec568499b9 |
| rig_registry                    | outputs/rig_registry/rig_relation_registry.csv                                                              |        21941 | 211d6270d948503ffda4a866558ca95fec0bc9fe99a5fe201616b842389ae631 |

## Failures

| stage                | scope_id     | reason                                          | detail             | severity   |
|:---------------------|:-------------|:------------------------------------------------|:-------------------|:-----------|
| outcome_estimability | confirmation | predicate_metric_unestimable_for_some_addresses | address_count=4176 | warning    |
| outcome_estimability | discovery    | predicate_metric_unestimable_for_some_addresses | address_count=2496 | warning    |

## Interpretation boundary

OBS-085a establishes structural evidence feasibility only. A passing address has enough frozen empirical structure to support later simulator qualification and conditional gate-passage analysis. It is not evidence that an artifact-direct effect exists, and it does not alter the null FL3 result of OBS-084.

> Claim entitlement is an epistemic ceiling, not a component of structural estimability.

> OBS-085a does not compute observed power or simulated gate-passage probability.
