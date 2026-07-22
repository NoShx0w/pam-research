# OBS-085d — Campaign Bottleneck Localization

## State

`campaign_bottleneck_localization_completed`

OBS-085d deterministically localizes the frozen OBS-085c prospective-campaign bottlenecks. No new simulation or threshold modification was performed.

## Frozen lineage

- OBS-085c manifest ID: `3341203b8c1e0024847fa054548a3c2ad6c263f271f22c947069281f4cde00ac`
- OBS-085c manifest SHA256: `713dc5bd37d234de9f4c0a8bf366058478f99cfc79c11252f38026f4117b0e9e`
- OBS-085c script SHA256: `b3d850a2da3ce43d8d44661729b793ae4e438acb41039468d8cb70dd9221926b`
- OBS-085c script commit: `f40b442dc06e9d9ae19466e01b73a2314485d1dc`
- OBS-085c output commit: `83cbdb6bfb6cb185646c178578890eb4c02a5f21`
- OBS-085c output hashes checked: **26**
- Current repository HEAD: `83cbdb6bfb6cb185646c178578890eb4c02a5f21`

## Execution integrity

- Frozen replicate rows analyzed: **4,200,000**
- Cell trajectories: **600**
- Empirical passable-set coverage plateau begins at tested k=**6**.
- Nonmonotonicity tolerance: **0.02** absolute probability.
- Plateau epsilon: **0.01** absolute probability.

## Cell trajectory classification

| trajectory_class | probability_shape | cells |
| --- | --- | --- |
| early_passable | materially_nonmonotone | 11 |
| early_passable | minor_nonmonotonicity_within_tolerance | 22 |
| early_passable | observed_non_decreasing | 302 |
| empirically_never_passable | all_zero | 264 |
| late_passable | observed_non_decreasing | 1 |

## Persistent non-passage decomposition

| persistent_nonpassage | plateau_limiting_class | cells |
| --- | --- | --- |
| False | empirically_passable | 336 |
| True | mixed_gate_limited | 264 |

> A limiting class localizes frozen-gate behavior. It is not a recommendation to weaken or remove that gate.

## Leave-one-gate-out diagnostic rescue

| gate_name | single_gate_rescues | mean_single_gate_blocker_probability | mean_gate_failure_probability |
| --- | --- | --- | --- |
| control_adjusted_contrast_pass | 68435 | 0.016294 | 0.354242 |
| minimum_effect_pass | 933 | 0.000222143 | 0.200925 |
| cluster_uncertainty_pass | 0 | 0 | 0.621136 |
| complement_admissible_pass | 0 | 0 | 0.068695 |
| control_specificity_pass | 0 | 0 | 0.275466 |
| effect_direction_reproduced_pass | 0 | 0 | 0.177262 |
| multiplicity_adjusted_threshold_pass | 0 | 0 | 0.814958 |
| raw_statistical_threshold_pass | 0 | 0 | 0.814958 |
| support_available_pass | 0 | 0 | 0.00694214 |
| target_contrast_positive_pass | 0 | 0 | 0.177262 |

## Nominal versus effective support

| address_id | aggregation_level | base_scenario_id | carrier | control_response_lambda | delta | effective_cluster_histogram_json | effective_resolution_attainable_probability | entitlement_status | failure_predicate | maximum_effective_cluster_count | mean_effective_cluster_count | median_effective_cluster_count | minimum_effective_cluster_count | minimum_observed_raw_p | nominal_cluster_count | nominal_support_efficiency | partition | probability_effective_k_at_least_4 | probability_effective_k_at_least_6 | probability_effective_k_at_least_8 | prospective_cluster_count | record_id | relation | replicates | scenario_id | simulator_id | support_id | uncertainty_status_counts_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | global |  |  |  |  | {"0":321758,"1":103226,"2":103192,"3":71824} | 0 |  |  | 3 | 0.875137 | 0 | 0 | 0.125 | 3 | 0.291712 |  | 0 | 0 | 0 | 3 |  |  | 600000 |  |  |  |  |
|  | global |  |  |  |  | {"0":300982,"1":82829,"2":82330,"3":82716,"4":51143} | 0.0852383 |  |  | 4 | 1.16701 | 0 | 0 | 0.0625 | 4 | 0.291754 |  | 0.0852383 | 0 | 0 | 4 |  |  | 600000 |  |  |  |  |
|  | global |  |  |  |  | {"0":287396,"1":69155,"2":68601,"3":68276,"4":69410,"5":37162} | 0.17762 |  |  | 5 | 1.45772 | 1 | 0 | 0.03125 | 5 | 0.291545 |  | 0.17762 | 0 | 0 | 5 |  |  | 600000 |  |  |  |  |
|  | global |  |  |  |  | {"0":277557,"1":58809,"2":61038,"3":54918,"4":61474,"5":58755,"6":27449} | 0.24613 |  |  | 6 | 1.75001 | 1 | 0 | 0.015625 | 6 | 0.291668 |  | 0.24613 | 0.0457483 | 0 | 6 |  |  | 600000 |  |  |  |  |
|  | global |  |  |  |  | {"0":265244,"1":41597,"2":52980,"3":45464,"4":39854,"5":45251,"6":52854,"7":41683,"8":15073} | 0.324525 |  |  | 8 | 2.33185 | 1 | 0 | 0.00390625 | 8 | 0.291481 |  | 0.324525 | 0.182683 | 0.0251217 | 8 |  |  | 600000 |  |  |  |  |
|  | global |  |  |  |  | {"0":258562,"1":28603,"2":44652,"3":43801,"4":34651,"5":29849,"6":34699,"7":43636,"8":44476,"9":28713,"10":8358} | 0.37397 |  |  | 10 | 2.91435 | 2 | 0 | 0.000976562 | 10 | 0.291435 |  | 0.37397 | 0.26647 | 0.135912 | 10 |  |  | 600000 |  |  |  |  |
|  | global |  |  |  |  | {"0":254830,"1":19079,"2":35885,"3":41252,"4":35667,"5":26713,"6":23462,"7":26948,"8":35413,"9":41405,"10":35523,"11":19173,"12":4650} | 0.414923 |  |  | 12 | 3.49688 | 2 | 0 | 0.000244141 | 12 | 0.291407 |  | 0.414923 | 0.310957 | 0.22694 | 12 |  |  | 600000 |  |  |  |  |

## Marginal value of added support

| aggregation_scope | partition | simulator_id | previous_cluster_count | next_cluster_count | evaluated_cells | newly_empirically_passable_cells | lost_empirically_passable_cells | mean_gate_passage_probability_gain | median_gate_passage_probability_gain | minimum_gate_passage_probability_gain | maximum_gate_passage_probability_gain | positive_probability_gain_share | mean_effective_cluster_gain | newly_reached_probability_0_5 | lost_probability_0_5 | newly_reached_probability_0_8 | lost_probability_0_8 | newly_reached_probability_0_9 | lost_probability_0_9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | all | all | 3 | 4 | 600 | 307 | 0 | 0.042085 | 0.001 | 0 | 0.351 | 0.511667 | 0.291878 | 0 | 0 | 0 | 0 | 0 | 0 |
| global | all | all | 4 | 5 | 600 | 28 | 1 | 0.0542133 | 0.004 | -0.001 | 0.345 | 0.556667 | 0.29071 | 47 | 0 | 0 | 0 | 0 | 0 |
| global | all | all | 5 | 6 | 600 | 2 | 0 | 0.0486067 | 0.007 | 0 | 0.246 | 0.56 | 0.292282 | 31 | 0 | 19 | 0 | 0 | 0 |
| global | all | all | 6 | 8 | 600 | 0 | 0 | 0.0687483 | 0.0205 | -0.001 | 0.306 | 0.556667 | 0.581842 | 46 | 0 | 52 | 0 | 52 | 0 |
| global | all | all | 8 | 10 | 600 | 0 | 0 | 0.0464067 | 0.0155 | -0.028 | 0.223 | 0.525 | 0.582502 | 6 | 0 | 36 | 0 | 19 | 0 |
| global | all | all | 10 | 12 | 600 | 0 | 0 | 0.0356017 | 0.002 | -0.042 | 0.173 | 0.506667 | 0.58253 | 11 | 0 | 15 | 0 | 36 | 0 |

## Diagnostic design stopping table

| aggregation_scope | partition | simulator_id | prospective_cluster_count | evaluated_cells | empirically_passable_cells | empirically_passable_share | newly_passable_cells_from_previous | lost_passable_cells_from_previous | mean_gate_passage_probability | mean_probability_gain_from_previous | cells_reaching_0_50 | cells_reaching_0_80 | cells_reaching_0_90 | mean_effective_cluster_count | coverage_plateau_start_k | diagnostic_design_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | all | all | 3 | 600 | 0 | 0 | 0 | 0 | 0 |  | 0 | 0 | 0 | 0.875137 | 6 | structurally_unattainable |
| global | all | all | 4 | 600 | 307 | 0.511667 | 307 | 0 | 0.042085 | 0.042085 | 0 | 0 | 0 | 1.16702 | 6 | coverage_expanding |
| global | all | all | 5 | 600 | 334 | 0.556667 | 28 | 1 | 0.0962983 | 0.0542133 | 47 | 0 | 0 | 1.45772 | 6 | coverage_expanding |
| global | all | all | 6 | 600 | 336 | 0.56 | 2 | 0 | 0.144905 | 0.0486067 | 78 | 19 | 0 | 1.75001 | 6 | coverage_plateau_probability_gain_continues |
| global | all | all | 8 | 600 | 336 | 0.56 | 0 | 0 | 0.213653 | 0.0687483 | 124 | 71 | 52 | 2.33185 | 6 | coverage_plateau_probability_gain_continues |
| global | all | all | 10 | 600 | 336 | 0.56 | 0 | 0 | 0.26006 | 0.0464067 | 130 | 107 | 71 | 2.91435 | 6 | coverage_plateau_probability_gain_continues |
| global | all | all | 12 | 600 | 336 | 0.56 | 0 | 0 | 0.295662 | 0.0356017 | 141 | 122 | 107 | 3.49688 | 6 | coverage_plateau_probability_gain_within_epsilon |

Coverage plateau and probability saturation are distinct. A stable passable-cell set can coexist with continuing probability gains inside that set.

## Partition concordance

| matched_cells | mean_absolute_probability_difference | first_pass_k_agreement_share | empirical_passability_agreement_share |
| --- | --- | --- | --- |
| 300 | 0.0665238 | 0.52 | 0.52 |

## Simulator concordance

| matched_cells | mean_absolute_probability_difference | first_pass_k_agreement_share | empirical_passability_agreement_share |
| --- | --- | --- | --- |
| 300 | 0.0050819 | 0.953333 | 1 |

## Failures

_No rows._

## Interpretation boundary

> OBS-085d is diagnostic localization only.

> Leave-one-gate-out passage is not an alternative evidence result and does not justify removing a frozen gate.

> Prospective template replication is not additional observed evidence.

> The study cannot create an FL3 witness, establish causal attribution, validate simulator truth, or increase claim entitlement.
