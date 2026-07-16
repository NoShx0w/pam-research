# OBS-084c — Direct Failure-Support Reserved Confirmation

## State

Confirmation completed with status: `confirmation_completed_no_fl3_with_capped_reproductions`

OBS-084a freeze manifest: `c360f3b4cd5d65067c8b1cca45800524b6ee5cf7ad31ed43aaedcdb0fcd0566d`
OBS-084b candidate manifest: `0d58d3abd25677683bb29b25c5b4e1fc2fdd1fab83866893c2151a80b97fd4f5`
OBS-084c confirmation manifest: `bb573a6565be03648d5eab191a768ba1c95e13ab4ead6a55939f252d42dd57cf`

This stage opened the frozen confirmation partition once and evaluated only the sealed OBS-084b FL2 candidate family. It performed no new support search, predicate search, threshold fitting, or candidate ranking.

## Canonical guardrails

> Directness is artifact-direct, not metaphysically direct and not causally direct.

> Discovery nominates a support; reserved evidence earns the localization claim.

Any FL3 result is scoped to the declared record, predicate, support, carrier contract, partition, and provenance. No result is actionable, causal, externally generalized, repaired, or formally topological.

## One-time opening

Confirmation opening lock: `outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084c_confirmation_opening_lock.json`

The lock was created before confirmation outcome evaluation and completed only after all confirmation artifacts were written.

## Frozen and sealed input validation

| validation_check                     | passed   | status   | detail                                                                                                                                               |
|:-------------------------------------|:---------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| candidate_manifest_internal_hash     | True     | pass     | declared=0d58d3abd25677683bb29b25c5b4e1fc2fdd1fab83866893c2151a80b97fd4f5; computed=0d58d3abd25677683bb29b25c5b4e1fc2fdd1fab83866893c2151a80b97fd4f5 |
| expected_candidate_manifest_identity | True     | pass     | expected=0d58d3abd25677683bb29b25c5b4e1fc2fdd1fab83866893c2151a80b97fd4f5; observed=0d58d3abd25677683bb29b25c5b4e1fc2fdd1fab83866893c2151a80b97fd4f5 |
| candidate_manifest_schema            | True     | pass     | obs084b_candidate_manifest_v1                                                                                                                        |
| candidate_manifest_status            | True     | pass     | sealed_FL2_candidates                                                                                                                                |
| obs084a_freeze_identity              | True     | pass     | candidate=c360f3b4cd5d65067c8b1cca45800524b6ee5cf7ad31ed43aaedcdb0fcd0566d; current=c360f3b4cd5d65067c8b1cca45800524b6ee5cf7ad31ed43aaedcdb0fcd0566d |
| confirmation_partition_identity      | True     | pass     | candidate=7198b2d34bc8c469a54c8a83ad7e6dfe975c084fde947c1992b7ea6882dc26d2; current=7198b2d34bc8c469a54c8a83ad7e6dfe975c084fde947c1992b7ea6882dc26d2 |
| candidate_count_consistency          | True     | pass     | declared=13; json=13; csv=13                                                                                                                         |
| candidate_required_fields            | True     | pass     | missing_json=[]; missing_csv=[]                                                                                                                      |
| candidate_ids_unique                 | True     | pass     |                                                                                                                                                      |
| candidate_csv_json_identity          | True     | pass     | candidate ID sets                                                                                                                                    |
| candidate_csv_json_critical_fields   | True     | pass     | mismatches=0                                                                                                                                         |
| sealed_support_queries_parse         | True     | pass     | []                                                                                                                                                   |
| sealed_fl2_statuses                  | True     | pass     |                                                                                                                                                      |
| obs084b_script_hash                  | True     | pass     | expected=7fcefb62fd43bff1da05ea4faa9e6554c0d487fa132795b120b3ced6224b90c3; actual=7fcefb62fd43bff1da05ea4faa9e6554c0d487fa132795b120b3ced6224b90c3   |

## Input artifact identity

| artifact_role                        | artifact_path                                                                                                | exists   | sha256                                                           |
|:-------------------------------------|:-------------------------------------------------------------------------------------------------------------|:---------|:-----------------------------------------------------------------|
| obs084a_freeze_manifest              | outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution/obs084a_freeze_manifest.json            | True     | 03b06c37945d82655814548595281264ef46faa62e40d9a130c8dec46dc0aa8a |
| canonical_feature_table              | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv                    | True     | 936802c75f758ab3b798d6c669e1cbc04323c2f1f610b5f7f39ad71c5437b6ca |
| canonical_feature_manifest           | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_manifest.csv                 | True     | 94ebfffb360c3d4f65d4ca480e65275ba651fa1b31d8dedf550222f6c58d1fe4 |
| leave_structure_predictions          | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_leave_structure_predictions.csv  | True     | 23b151b794a6d50aff9a5febe68c9478a52644974f959db62236d34bac283672 |
| pairwise_predictions                 | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_predictions.csv                  | True     | 9956be0583256642827694e9694a8ba1f1c97616fc2cc6d132505be90ad823fc |
| numeric_transform_predictions        | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_predictions.csv           | True     | b3706215227eeb5b2d000ebeace34ece2a902d46c6ba1fc1841ba3b1986174ad |
| scale_band_predictions               | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_predictions.csv         | True     | daaffddaca8dfe402348b889f627b25b30103fac4c94a7d2298181b4295c590b |
| feature_contract_predictions         | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_predictions.csv     | True     | a09ca38d92a6d63a437e9402035a854190aa767447caee10b7c2a32ab3fe4c9a |
| structural_resampling_summary        | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_summary.csv         | True     | 39762e5b3410d221566898d71eaaafc3601dd15db9e4920060f022e8bd46a6cb |
| registry                             | outputs/rig_registry/rig_relation_registry.csv                                                               | True     | 211d6270d948503ffda4a866558ca95fec0bc9fe99a5fe201616b842389ae631 |
| obs083_subclasses                    | outputs/rig_registry/obs083_negative_control_localization/obs083_diagnostic_subclass_assignments.csv         | True     | c3fd28c1bbc31dde900cedf9ed6ea3b3f40cddbfb64d642e5521461b5a186701 |
| obs083_relation_controls             | outputs/rig_registry/obs083_negative_control_localization/obs083_relation_control_contrast.csv               | True     | d8362e49c97d1ba8566ab9704b400139e0315c5dd7f6201be891deb8032c70a9 |
| obs083_carrier_controls              | outputs/rig_registry/obs083_negative_control_localization/obs083_carrier_control_contrast.csv                | True     | 58f0451b2b321fc4c192464134fb70566404ed4d169e41dee89204d90f5c37c5 |
| canonical_geometry_contract_manifest | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_manifest.csv        | True     | 8a21e7ab2ed24f3a5b544b6f46e8aa60bd592558478085252a49a93b421e132d |
| canonical_geometry_contract_manifest | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_feature_contract_manifest.csv | True     | f927eafec401d4aabc2934cc6d1ea366a428014cbfd0b3458bbee38fc16ae940 |
| freeze_script                        | experiments/studies/obs084a_bridge_resolution_and_partition_freeze.py    | True     | 8af100c08550526b08be5c7b0202572dc974abc6173676fb2c0589225ae8d6f3 |
| obs084b_candidate_json               | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_candidate_freeze_manifest.json          | True     | 4deb25d622ab1ecb377d32468975aea6c332f18c2a88b64aa3cc1c67fef29167 |
| obs084b_candidate_csv                | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_candidate_freeze_manifest.csv           | True     | 06575ec3bf85143f5ef3c11d18ae785f81019ae5470e2fa587c2ca0da5773f0f |
| obs084b_thresholds                   | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_support_thresholds.csv                  | True     | 38b97c1f3a6a822460a131907564eb29e2575ce46da3f750f3cb0d4fdcc2f49f |
| obs084b_input_manifest               | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_input_manifest.csv                      | True     | 34b9c0adac5b19f93c12bd70b4bd6b6fb10c6762bfc3c6d5e25b045b20c268b3 |
| obs084b_summary                      | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_discovery_summary.csv                   | True     | d901c765af1428faafdaa210cb5d84b327f57eb633398d838e69eb34d2e803e1 |
| obs084b_report                       | outputs/rig_registry/obs084_direct_failure_witness/discovery/obs084b_discovery_report.md                     | True     | 0db812dd7e20d1944b14c2d8e4208f695c380c27d76c3e23ef5a00fb638dc6cb |
| obs084b_script                       | experiments/studies/obs084b_direct_failure_support_discovery.py                                              | True     | 7fcefb62fd43bff1da05ea4faa9e6554c0d487fa132795b120b3ced6224b90c3 |
| obs084c_script                       | experiments/studies/obs084c_direct_failure_support_confirmation.py                                           | True     | 5090aefa7dfde231a95ed65982af45c8f9af40cb323cba84ba199590b8c36c9c |

## Discovery-threshold verification

| support_family   | source_field                | threshold_name   |   threshold_value_recorded |   threshold_value_recomputed | validation_pass   |
|:-----------------|:----------------------------|:-----------------|---------------------------:|-----------------------------:|:------------------|
| scale_band       | transition_midpoint         | q33              |                   4.5      |                     4.5      | True              |
| scale_band       | transition_midpoint         | q67              |                   5.5      |                     5.5      | True              |
| seam_relative    | seam__core__path_enrichment | q33              |                   0.862518 |                     0.862518 | True              |
| seam_relative    | seam__core__path_enrichment | q67              |                   1.02389  |                     1.02389  | True              |

Recorded OBS-084b scale and seam cut values were independently recomputed from frozen discovery feature values. The verified recorded values were then applied unchanged to confirmation.

## Confirmation-only diagnostic instrument

- Sealed candidates evaluated: 13
- Confirmation observation-loss rows: 1404
- Confirmation observations represented: 78
- Structural dependence unit: object (`cluster_id`)
- Diagnostic model: confirmation-only leave-one-object-out balanced logistic regression
- Multiplicity family: exactly the sealed OBS-084b candidate family

## Outcome counts

| confirmation_status                             |   count |
|:------------------------------------------------|--------:|
| confirmation_complement_inadmissible            |       5 |
| confirmation_signal_absent                      |       3 |
| confirmation_multiplicity_not_survived          |       2 |
| confirmation_reproduced_but_claim_capped_at_fl2 |       1 |
| confirmation_control_explained                  |       1 |
| confirmation_uncertain_cluster_support          |       1 |

## Candidate outcomes

| candidate_id                 | record_id                         | failure_predicate               | support_definition                                                      |   discovery_site_relative_contrast |   confirmation_site_relative_contrast |   confirmation_bootstrap_ci_low |   confirmation_permutation_p |   confirmation_q_sealed_family |   median_control_adjusted_contrast | confirmation_status                             |
|:-----------------------------|:----------------------------------|:--------------------------------|:------------------------------------------------------------------------|-----------------------------------:|--------------------------------------:|--------------------------------:|-----------------------------:|-------------------------------:|-----------------------------------:|:------------------------------------------------|
| OBS084B-a69d98b6b8217d4059ce | C_vs_Cp3__no_window               | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           0.659878 |                             0.214902  |                       0.0155564 |                     0.004995 |                      0.0649351 |                         0.0523813  | confirmation_reproduced_but_claim_capped_at_fl2 |
| OBS084B-51b6faee33ac6cc790bd | C_vs_Cp2__no_window               | log_loss_attenuation            | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           1.20944  |                             0.303926  |                       0.0856142 |                     0.016983 |                      0.11039   |                         0.146788   | confirmation_multiplicity_not_survived          |
| OBS084B-7ec663ae1af7424b1b72 | C_vs_Cp2__no_window               | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           0.824095 |                             0.159058  |                       0.0391909 |                     0.100899 |                      0.437229  |                         0.0630312  | confirmation_multiplicity_not_survived          |
| OBS084B-5c75c4f7df387f78f22c | C_vs_Cp3__strict_numeric_all      | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           0.736191 |                             0.13299   |                       0.0834228 |                     0.4995   |                      1         |                        -0.0574534  | confirmation_control_explained                  |
| OBS084B-590dc8c788380700d526 | C_vs_Cp2__strict_numeric_all      | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           0.795097 |                             0.104296  |                      -0.035748  |                     0.277722 |                      0.902597  |                        -0.00607811 | confirmation_uncertain_cluster_support          |
| OBS084B-2f8f8e8bd729832e4f1c | C_vs_Cp2__geometry_scores_only    | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           0.579023 |                             0.0877576 |                      -0.0918654 |                     0.832168 |                      1         |                        -0.0439192  | confirmation_signal_absent                      |
| OBS084B-c811fbc12bffe9acee2f | C_vs_Cp2__stability_plus_geometry | relation_separation_attenuation | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           0.521466 |                             0.0276433 |                      -0.161624  |                     0.582418 |                      1         |                        -0.104033   | confirmation_signal_absent                      |
| OBS084B-f4375eb30e72ed220004 | C_vs_Cp2__no_window               | local_criterion_breach          | scale_band:scale_band=early AND seam_relative:seam_relative_region=near |                           0.5      |                             0.0757576 |                       0         |                     0.461538 |                      1         |                         0.0112554  | confirmation_signal_absent                      |
| OBS084B-0f82f9c3b220ffa08e45 | C_vs_Cp2__geometry_scores_only    | relation_separation_attenuation | scale_band:scale_band=early                                             |                           0.764724 |                           nan         |                     nan         |                     1        |                      1         |                       nan          | confirmation_complement_inadmissible            |
| OBS084B-11e1ddfaec94b8ba1e20 | C_vs_Cp3__geometry_scores_only    | relation_separation_attenuation | transition:transition=5→6 AND scale_band:scale_band=middle              |                           0.613684 |                            -0.61351   |                      -0.743186  |                     1        |                      1         |                       nan          | confirmation_complement_inadmissible            |
| OBS084B-340c36cc5ee088d4fab9 | three_way__geometry_scores_only   | log_loss_attenuation            | transition:transition=5→6 AND scale_band:scale_band=middle              |                           1.8684   |                           nan         |                     nan         |                     1        |                      1         |                       nan          | confirmation_complement_inadmissible            |
| OBS084B-34d66eb07912a939e6bf | C_vs_Cp2__geometry_scores_only    | local_criterion_breach          | scale_band:scale_band=early                                             |                           0.666667 |                           nan         |                     nan         |                     1        |                      1         |                       nan          | confirmation_complement_inadmissible            |
| OBS084B-57dd73e04bc58c21dc10 | C_vs_Cp3__geometry_scores_only    | log_loss_attenuation            | transition:transition=5→6 AND scale_band:scale_band=middle              |                           2.05575  |                            -1.44524   |                      -1.76966   |                     1        |                      1         |                       nan          | confirmation_complement_inadmissible            |

## FL3 direct-witness registry

_No rows._

## Failures and exclusions

_No execution failures._

## Interpretation

Candidates that reproduced directionally but failed uncertainty, controls, multiplicity, or claim-entitlement gates remain unconfirmed FL2 evidence. C1 contrast-limited candidates cannot be promoted beyond FL2 in this stage even if their confirmation signal reproduces.

## Canonical result statement

OBS-084c evaluates the complete sealed OBS-084b candidate family once on the frozen confirmation partition. It may establish scoped FL3 artifact-direct witnesses, but establishes no causal origin, repair target, intervention readiness, actionability, external generalization, or formal topology.
