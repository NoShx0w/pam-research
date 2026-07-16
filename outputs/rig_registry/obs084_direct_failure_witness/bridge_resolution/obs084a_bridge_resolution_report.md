# OBS-084a — Bridge Resolution and Two-Way Partition Freeze

## State

Resolution/freeze pass completed.

Overall status: `frozen_ready_for_discovery`

Freeze manifest ID: `c360f3b4cd5d65067c8b1cca45800524b6ee5cf7ad31ed43aaedcdb0fcd0566d`

This is a pre-discovery design-freeze artifact only. It performs no candidate
generation, confirmation-outcome inspection, localization contrast, witness
assignment, FL promotion, repair design, intervention, or causal analysis.

## Canonical interpretation

This pass resolves the final bridge decisions required before OBS-084 discovery:
observation identity, carrier predictors, field roles, cluster unit, deterministic
discovery/confirmation partition, seam discretization protocol, support vocabulary,
and source identity.

A successful freeze means only that discovery may begin under an inspectable and
immutable protocol. It is not evidence that a failure support exists.

## Freeze gates

| gate                                 | passed   | status   | detail                                                                                                       |
|:-------------------------------------|:---------|:---------|:-------------------------------------------------------------------------------------------------------------|
| observation_key_resolved             | True     | pass     | selected key rows=1; scientific key complete/unique and alignment key unique=True                            |
| carrier_features_resolved            | True     | pass     | all six carriers have complete allowed feature manifests; exact canonical carriers match ordered definitions |
| field_roles_resolved                 | True     | pass     | manual-review fields remaining=0                                                                             |
| cluster_unit_selected                | True     | pass     | selected cluster rows=1                                                                                      |
| two_way_partition_created            | True     | pass     | partitions=['confirmation', 'discovery']                                                                     |
| eligible_record_partition_balance    | True     | pass     | eligible partition rows=24, passing=24                                                                       |
| support_vocabulary_frozen            | True     | pass     | included support families=10                                                                                 |
| source_hashes_complete               | True     | pass     | hashed sources=15/15                                                                                         |
| human_review_or_explicit_auto_freeze | True     | pass     | auto-proposed rows=161; allow_auto_freeze=True                                                               |

## Observation-key specification

| canonical_key_name         | feature_table_column                               | key_columns_json                                                   |   non_null_rows |   unique_values |   uniqueness_ratio |   null_key_rows |   duplicate_key_rows | key_role             | alignment_key_columns_json                     | alignment_key_unique   |   alignment_duplicate_rows | resolution_status                       | resolution_basis                                                                                 | review_note                                                                     | selected   |
|:---------------------------|:---------------------------------------------------|:-------------------------------------------------------------------|----------------:|----------------:|-------------------:|----------------:|---------------------:|:---------------------|:-----------------------------------------------|:-----------------------|---------------------------:|:----------------------------------------|:-------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:-----------|
| scientific_observation_key | case|object|cohort|scale_index_from|scale_index_to | ["case", "object", "cohort", "scale_index_from", "scale_index_to"] |             168 |             168 |                  1 |               0 |                    0 | scientific_composite | ["case", "object", "cohort", "candidate_rank"] | True                   |                          0 | resolved_validated_scientific_composite | predeclared OBS-078 scientific identity; independently validated for completeness and uniqueness | alignment key is diagnostic only and is not the scientific observation identity | True       |

## Carrier feature counts

| carrier                 |   feature_count |
|:------------------------|----------------:|
| geometry_scores_only    |               7 |
| no_window               |              19 |
| path_shares_only        |               4 |
| stability_core_3        |               3 |
| stability_plus_geometry |              10 |
| strict_numeric_all      |              25 |

## Cluster-unit selection

| cluster_unit   |   unique_clusters | selected   | selection_basis                                                           | resolution_status   |
|:---------------|------------------:|:-----------|:--------------------------------------------------------------------------|:--------------------|
| object         |                 8 | True       | preference order prioritizing object-level independence before route/path | proposed_auto       |

## Two-way partition

| partition    |   rows |   clusters |
|:-------------|-------:|-----------:|
| confirmation |     78 |          4 |
| discovery    |     90 |          4 |

Partition assignment is deterministic at the selected cluster level. Rows from
the same cluster cannot cross discovery and confirmation.

## Per-record partition balance

- Balance rows: 48
- FL3-eligible rows: 24
- Passing FL3-eligible rows: 24

A globally balanced split is insufficient. Every confirmation-eligible C2 record
must retain adequate cluster and class support in both partitions.

## Seam discretization protocol

| support_family   | source_field                | discretization_method     |   bin_count | threshold_basis                                                                      | threshold_values_json   | bin_labels_json                 | fit_partition   | outcome_blind   | resolution_status   |
|:-----------------|:----------------------------|:--------------------------|------------:|:-------------------------------------------------------------------------------------|:------------------------|:--------------------------------|:----------------|:----------------|:--------------------|
| seam_relative    | seam__core__path_enrichment | equal_frequency_quantiles |           3 | quantiles fitted on discovery partition only; then applied unchanged to confirmation | []                      | ["near", "intermediate", "far"] | discovery_only  | True            | proposed_auto       |

Seam thresholds are not fitted by this script. The method is frozen here; cut
values must be estimated on discovery only and applied unchanged to confirmation.

## Support vocabulary

| support_family        | included_in_discovery_vocabulary   | requires_predeclared_discretization   |   max_conjunction_depth | freeze_status                       | resolution_status   |
|:----------------------|:-----------------------------------|:--------------------------------------|------------------------:|:------------------------------------|:--------------------|
| object                | True                               | False                                 |                       2 | included                            | proposed_auto       |
| cohort                | True                               | False                                 |                       2 | included                            | proposed_auto       |
| transition            | True                               | False                                 |                       2 | included                            | proposed_auto       |
| scale_band            | True                               | False                                 |                       2 | included                            | proposed_auto       |
| contract_or_transform | True                               | False                                 |                       2 | included                            | proposed_auto       |
| feature_family        | True                               | False                                 |                       2 | included                            | proposed_auto       |
| seam_relative         | True                               | True                                  |                       2 | included_with_frozen_discretization | proposed_auto       |
| window                | True                               | False                                 |                       2 | included                            | proposed_auto       |
| route_or_path         | True                               | False                                 |                       2 | included                            | proposed_auto       |
| provenance_slice      | True                               | False                                 |                       2 | included                            | proposed_auto       |

Unavailable support families remain excluded. Candidate conjunction depth is
capped to prevent uncontrolled combinatorial discovery.

## Source identity

- Hashed source rows: 15
- Repository/code commit recorded: 2647b494eb688388bdad9b88d8dd32c8a5e95f69

## Discovery gate

OBS-084 candidate discovery may begin only when all hard gates pass and the
freeze status is `frozen_ready_for_discovery`.

If the status is `proposal_ready_for_human_review`, the generated artifacts are
inspectable proposals only. Review or explicitly rerun with `--allow-auto-freeze`
after accepting the rule-based decisions.

If any technical gate fails, discovery remains blocked.

## Outputs

- `obs084a_reviewed_observation_key_spec.csv`
- `obs084a_reviewed_carrier_feature_manifest.csv`
- `obs084a_reviewed_field_roles.csv`
- `obs084a_cluster_unit_selection.csv`
- `obs084a_two_way_partition_manifest.csv`
- `obs084a_partition_balance_final.csv`
- `obs084a_seam_discretization_protocol.csv`
- `obs084a_support_vocabulary_freeze.csv`
- `obs084a_source_hash_manifest.csv`
- `obs084a_bridge_resolution_summary.csv`
- `obs084a_freeze_manifest.json`
- `obs084a_bridge_resolution_report.md`

## Limitations

- Rule-based proposals are not silently described as human-reviewed.
- Observation-key overlap does not prove semantic equivalence across every
  prediction artifact; future discovery must use the frozen key specification.
- Carrier manifests derived from vocabulary rules should be reviewed against
  upstream feature manifests before irreversible candidate freeze.
- Cluster assignment supports structural separation, not proof of statistical
  independence.
- No confirmation outcomes are inspected or unlocked here.

## Canonical result statement

OBS-084a bridge resolution converts the canonical OBS-078–083 evidence spine into
an inspectable pre-discovery protocol covering observation identity, carrier
features, field roles, structural clustering, deterministic two-way partitioning,
support vocabulary, seam discretization, and source hashes. It establishes study-
design readiness only and no direct failure support, causal origin, repair target,
actionability, external generalization, or formal topology.
