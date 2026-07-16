# OBS-084a — Canonical Lineage and Observation Bridge

## State

Bridge completed.

Overall status: `lineage_bridge_partially_ready_with_manual_resolution_required`

This is a semantic-lineage and observation-bridge audit only. It performs no
candidate generation, confirmation, witness assignment, FL promotion, repair
design, intervention, or causal analysis.

## Canonical interpretation

The bridge asks whether the broad reconnaissance inventory can be reduced to a
small, explicit evidence spine connecting OBS-083 registry records to canonical
OBS-078–080 observation-level artifacts.

A positive mapping means only that a record has a plausible canonical or
reconstructible observation source. It does not establish a failure support.

## Canonical source manifest

| source_role                   | selected_path                                                                                               | exists   |   rows_read |   column_count | read_status   |
|:------------------------------|:------------------------------------------------------------------------------------------------------------|:---------|------------:|---------------:|:--------------|
| canonical_feature_table       | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv                   | True     |         168 |            112 | ok            |
| canonical_feature_manifest    | outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_manifest.csv                | True     |         112 |              7 | ok            |
| leave_structure_predictions   | outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_leave_structure_predictions.csv | True     |        4032 |             10 | ok            |
| pairwise_predictions          | outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_predictions.csv                 | True     |       37632 |             12 | ok            |
| numeric_transform_predictions | outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_predictions.csv          | True     |       56448 |             12 | ok            |
| scale_band_predictions        | outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_predictions.csv        | True     |       43288 |             16 | ok            |
| feature_contract_predictions  | outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_predictions.csv    | True     |       96768 |             12 | ok            |
| structural_resampling_summary | outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_summary.csv        | True     |         576 |             22 | ok            |
| registry                      | outputs/rig_registry/rig_relation_registry.csv                                                              | True     |          24 |             47 | ok            |
| obs083_subclasses             | outputs/rig_registry/obs083_negative_control_localization/obs083_diagnostic_subclass_assignments.csv        | True     |          24 |             32 | ok            |
| obs083_relation_controls      | outputs/rig_registry/obs083_negative_control_localization/obs083_relation_control_contrast.csv              | True     |          72 |             16 | ok            |
| obs083_carrier_controls       | outputs/rig_registry/obs083_negative_control_localization/obs083_carrier_control_contrast.csv               | True     |         120 |             18 | ok            |

## Registry-record source mapping

| mapping_status                        |   count |
|:--------------------------------------|--------:|
| canonical_observation_source_resolved |      24 |

- Records audited: 24
- FL3-eligible C2 records: 12
- Canonically resolved observation sources: 24
- Reconstructible source candidates: 0
- Unresolved records: 0

## Observation-key bridge

- Crosswalk rows: 42
- Value-overlap-supported bridges: 15

Shared field names remain insufficient by themselves. A future discovery script
must use only reviewed value-overlap or explicitly reconstructed keys.

## Carrier-feature resolution

- Carriers audited: 6
- Resolved carriers: 6
- Unresolved carriers: 0

Heuristic carrier definitions require manual review before candidate freeze.

## Cluster hierarchy and partition balance

- Candidate cluster families: 3
- Record-by-cluster balance rows: 72
- Three-way balance candidates: 0

Cluster counts and deterministic hash partitions are design candidates only.
They do not establish statistical independence or adequate matched support.

## Support-family resolution

| support_family        | available   | requires_predeclared_discretization   | discovery_status                      |
|:----------------------|:------------|:--------------------------------------|:--------------------------------------|
| object                | True        | False                                 | available                             |
| cohort                | True        | False                                 | available                             |
| transition            | True        | False                                 | available                             |
| scale_band            | True        | False                                 | available                             |
| contract_or_transform | True        | False                                 | available                             |
| feature_family        | True        | False                                 | available                             |
| seam_relative         | True        | True                                  | available_after_discretization_freeze |
| boundary_relative     | False       | False                                 | unavailable                           |
| window                | True        | False                                 | available                             |
| route_or_path         | True        | False                                 | available                             |
| provenance_slice      | True        | False                                 | available                             |

Continuous seam/boundary fields must be discretized under a predeclared rule
before outcome inspection. Unavailable families must not appear in the frozen
candidate vocabulary.

## Field-role classification

- Fields classified: 316
- Candidate predictors: 74
- Grouping/partition fields: 96
- Provenance fields: 9
- Forbidden predictive-leakage fields: 60
- Manual-review fields: 54

A field may be valid for matching, grouping, or provenance while remaining
forbidden as a predictive carrier field.

## Duplicate and lineage resolution

- Candidate artifact variants audited: 15
- Selected canonical artifacts: 12
- Content-identical mirrors: 12
- Alternate noncanonical artifacts: 3

Large OBS-073/075 route tables are not selected merely because they expose many
path identifiers. They remain optional enrichment unless an explicit semantic
bridge to the registry lineage is later established.

## Discovery gates

OBS-084 candidate discovery should remain blocked for any record lacking:

1. a reviewed canonical or reconstructible observation source;
2. a reviewed observation-key bridge;
3. an explicit carrier-feature definition;
4. a defensible cluster hierarchy;
5. per-record partition balance;
6. a frozen support vocabulary;
7. a reviewed field-role classification;
8. source hashes and repository commit identity.

## Outputs

- `obs084a_canonical_source_manifest.csv`
- `obs084a_artifact_duplicate_resolution.csv`
- `obs084a_registry_record_to_prediction_source.csv`
- `obs084a_observation_key_crosswalk.csv`
- `obs084a_carrier_feature_resolution.csv`
- `obs084a_cluster_hierarchy.csv`
- `obs084a_partition_balance_by_record.csv`
- `obs084a_support_family_resolution.csv`
- `obs084a_field_role_classification.csv`
- `obs084a_lineage_bridge_summary.csv`
- `obs084a_lineage_bridge_report.md`

## Limitations

- Canonical selection is rule-based and must be reviewed where multiple
  non-identical authoritative-looking artifacts exist.
- Value overlap does not prove that two columns have identical scientific
  semantics.
- Carrier resolution may be heuristic when upstream manifests do not encode the
  exact registry carrier.
- Deterministic partitions are proposals only; reserved evidence is not created
  or unlocked by this script.
- No localization predicate, support, contrast, witness, or FL level is created.

## Canonical result statement

OBS-084a lineage bridging reduces the broad repository artifact inventory to an
explicit OBS-078–083 evidence spine and audits whether registry records can be
connected to observation-level sources, carrier definitions, structural units,
support vocabularies, and leakage-safe field roles. It establishes semantic
lineage feasibility only and no direct failure support, causal origin, repair
target, actionability, external generalization, or formal topology.
