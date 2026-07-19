# OBS-085b — Conditional Gate-Passage Sensitivity

## State

`conditional_gate_passage_sensitivity_completed`

OBS-085b estimates conditional frozen-gate passage only for the OBS-085b0-authorized missingness simulator cells. It does not estimate classical power or establish a minimum detectable effect.

## Frozen lineage

- OBS-085b0 manifest ID: `3015094cef1ee6a3f2b098662b75668109491634be827cd62cffc7b598fc66e6`
- OBS-085b0 script version: `1.0.2`
- OBS-085b0 output hashes checked: **17**
- Current repository HEAD: `f98ba656c4e836e90c6ccd737b7f536c4851d5bb`

## Authorized scope

- Frozen qualification addresses reused: **6**
- Authorized predicate: `measurement_missingness_concentration`
- Authorized predicate × partition × simulator cells: **4**

| failure_predicate                     | partition    | simulator_id                       | qualification_status       | scope_limits_json                                                                                                                                                                                                                                                    |
|:--------------------------------------|:-------------|:-----------------------------------|:---------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| measurement_missingness_concentration | confirmation | joint_gaussian_regularized_cluster | qualified_with_scope_limit | ["Gaussian tails and covariance shrinkage are declared simulator assumptions","baseline_covariance_preservation_unscorable","missingness mechanism is simulator-defined because frozen baseline is sparse or zero"]                                                  |
| measurement_missingness_concentration | confirmation | joint_wild_cluster_rademacher      | qualified_with_scope_limit | ["baseline_covariance_preservation_unscorable","missingness mechanism is simulator-defined because frozen baseline is sparse or zero","wild residual symmetry is a declared model assumption with four clusters"]                                                    |
| measurement_missingness_concentration | discovery    | joint_gaussian_regularized_cluster | qualified_with_scope_limit | ["Gaussian tails and covariance shrinkage are declared simulator assumptions","baseline_covariance_preservation_unscorable","missingness mechanism is simulator-defined because frozen baseline is sparse or zero","structural_leave_one_object_out_unavailability"] |
| measurement_missingness_concentration | discovery    | joint_wild_cluster_rademacher      | qualified_with_scope_limit | ["baseline_covariance_preservation_unscorable","missingness mechanism is simulator-defined because frozen baseline is sparse or zero","structural_leave_one_object_out_unavailability","wild residual symmetry is a declared model assumption with four clusters"]   |

No address was added, removed, or ranked after observing OBS-085b simulation results.

## Conditional estimand

For each frozen address, partition, qualified simulator, effect level, and control-response condition, the primary quantity is `conditional_gate_passage_probability`.

Discovery and confirmation remain separate. Gaussian and wild simulator results remain separate.

## Frozen design

- Scenarios: **25**
- Replicates per address × partition × simulator × scenario: **1,000**
- Effect grid: `0.00,0.25,0.50,0.75,1.00,1.50,2.00`
- Control-response grid: `0.00,0.25,0.50,1.00`
- Complete replicate vectors retained: **True**

## Gate contract

| gate_name                            |   gate_order | estimand_field                                                | pass_rule                                                                               | threshold                                                      | provenance                                                            | required_for_overall_pass   |
|:-------------------------------------|-------------:|:--------------------------------------------------------------|:----------------------------------------------------------------------------------------|:---------------------------------------------------------------|:----------------------------------------------------------------------|:----------------------------|
| support_available_pass               |            1 | target support and metric availability                        | metric present and focal support has at least one row                                   | >= 1 support row                                               | frozen address/support identity                                       | True                        |
| complement_admissible_pass           |            2 | focal/complement rows and shared object clusters              | site rows, complement rows, and shared clusters meet frozen minima                      | site >= 8; complement >= 8; shared objects >= 2                | OBS-085a structural evidence contract                                 | True                        |
| effect_direction_reproduced_pass     |            3 | target_response_from_simulated_null                           | positive injected target response for delta > 0                                         | > 1e-12                                                        | OBS-085b0 direct injection-response qualification estimand            | True                        |
| target_contrast_positive_pass        |            4 | target_contrast                                               | focal support missingness contrast is positive                                          | > 0                                                            | frozen positive failure direction                                     | True                        |
| minimum_effect_pass                  |            5 | target_contrast                                               | focal support missingness contrast reaches declared minimum effect                      | >= 0.1                                                         | predeclared OBS-085b missingness contrast gate                        | True                        |
| cluster_uncertainty_pass             |            6 | object-cluster bootstrap and leave-one-object-out diagnostics | bootstrap lower bound positive, direction consistency sufficient, and >=2 LOO estimates | CI low > 0; direction consistency >= 0.75; LOO successful >= 2 | OBS-084c cluster-sensitivity form; object retained as dependence unit | True                        |
| raw_statistical_threshold_pass       |            7 | one-sided object-cluster sign-flip p-value                    | raw p-value reaches declared alpha                                                      | p <= 0.1                                                       | fixed-address conditional randomization diagnostic                    | True                        |
| multiplicity_adjusted_threshold_pass |            8 | M1 adjusted p-value                                           | fixed-address M1 adjusted p-value reaches alpha                                         | q_M1 <= 0.1                                                    | M1 only; M13 not identified for the missingness panel                 | True                        |
| control_adjusted_contrast_pass       |            9 | target contrast minus median frozen-control contrast          | median control-adjusted contrast reaches declared minimum                               | >= 0.05                                                        | OBS-084c control-adjusted effect form                                 | True                        |
| control_specificity_pass             |           10 | share of target-minus-control contrasts that are positive     | positive target-minus-control share reaches declared minimum                            | >= 0.5                                                         | OBS-084c positive control-adjusted share form                         | True                        |

## Null calibration

| partition    | simulator_id                       |   addresses |   macro_mean_null_gate_passage |   maximum_address_null_gate_passage |   macro_mean_null_target_contrast |
|:-------------|:-----------------------------------|------------:|-------------------------------:|------------------------------------:|----------------------------------:|
| confirmation | joint_gaussian_regularized_cluster |           6 |                              0 |                                   0 |                                 0 |
| confirmation | joint_wild_cluster_rademacher      |           6 |                              0 |                                   0 |                                 0 |
| discovery    | joint_gaussian_regularized_cluster |           6 |                              0 |                                   0 |                                 0 |
| discovery    | joint_wild_cluster_rademacher      |           6 |                              0 |                                   0 |                                 0 |

## Highest tested effect

| partition    | simulator_id                       |   control_response_lambda |   delta |   addresses |   macro_mean_gate_passage_probability |   minimum_address_probability |   maximum_address_probability |
|:-------------|:-----------------------------------|--------------------------:|--------:|------------:|--------------------------------------:|------------------------------:|------------------------------:|
| discovery    | joint_wild_cluster_rademacher      |                      0    |       2 |           6 |                                     0 |                             0 |                             0 |
| discovery    | joint_wild_cluster_rademacher      |                      0.25 |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_wild_cluster_rademacher      |                      1    |       2 |           6 |                                     0 |                             0 |                             0 |
| discovery    | joint_gaussian_regularized_cluster |                      0.5  |       2 |           6 |                                     0 |                             0 |                             0 |
| discovery    | joint_gaussian_regularized_cluster |                      0.25 |       2 |           6 |                                     0 |                             0 |                             0 |
| discovery    | joint_gaussian_regularized_cluster |                      0    |       2 |           6 |                                     0 |                             0 |                             0 |
| discovery    | joint_wild_cluster_rademacher      |                      0.5  |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_wild_cluster_rademacher      |                      0.5  |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_wild_cluster_rademacher      |                      0.25 |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_wild_cluster_rademacher      |                      0    |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_gaussian_regularized_cluster |                      1    |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_gaussian_regularized_cluster |                      0.5  |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_gaussian_regularized_cluster |                      0.25 |       2 |           6 |                                     0 |                             0 |                             0 |
| confirmation | joint_gaussian_regularized_cluster |                      0    |       2 |           6 |                                     0 |                             0 |                             0 |
| discovery    | joint_gaussian_regularized_cluster |                      1    |       2 |           6 |                                     0 |                             0 |                             0 |
| discovery    | joint_wild_cluster_rademacher      |                      1    |       2 |           6 |                                     0 |                             0 |                             0 |

## Tested-grid thresholds

| partition    | simulator_id                       |   target_gate_passage_probability | threshold_status           |   address_control_cells |
|:-------------|:-----------------------------------|----------------------------------:|:---------------------------|------------------------:|
| confirmation | joint_gaussian_regularized_cluster |                               0.5 | not_reached_on_tested_grid |                      24 |
| confirmation | joint_gaussian_regularized_cluster |                               0.8 | not_reached_on_tested_grid |                      24 |
| confirmation | joint_gaussian_regularized_cluster |                               0.9 | not_reached_on_tested_grid |                      24 |
| confirmation | joint_wild_cluster_rademacher      |                               0.5 | not_reached_on_tested_grid |                      24 |
| confirmation | joint_wild_cluster_rademacher      |                               0.8 | not_reached_on_tested_grid |                      24 |
| confirmation | joint_wild_cluster_rademacher      |                               0.9 | not_reached_on_tested_grid |                      24 |
| discovery    | joint_gaussian_regularized_cluster |                               0.5 | not_reached_on_tested_grid |                      24 |
| discovery    | joint_gaussian_regularized_cluster |                               0.8 | not_reached_on_tested_grid |                      24 |
| discovery    | joint_gaussian_regularized_cluster |                               0.9 | not_reached_on_tested_grid |                      24 |
| discovery    | joint_wild_cluster_rademacher      |                               0.5 | not_reached_on_tested_grid |                      24 |
| discovery    | joint_wild_cluster_rademacher      |                               0.8 | not_reached_on_tested_grid |                      24 |
| discovery    | joint_wild_cluster_rademacher      |                               0.9 | not_reached_on_tested_grid |                      24 |

Threshold rows report the smallest tested delta reaching 50%, 80%, or 90% passage. No interpolation or extrapolation is used.

## Failures

_No rows._

## Multiplicity boundary

The primary multiplicity contract is M1 because each estimand is conditional on one fixed predeclared address. The sealed OBS-084b M13 family contained no authorized missingness candidate family; OBS-085b therefore does not invent or approximate an M13 Benjamini-Hochberg probability.

## Interpretation boundary

> Monte Carlo intervals quantify simulation-sampling error only; they do not quantify simulator-model uncertainty.

> A gate-passage probability is conditional on the frozen address, qualified simulator, declared missingness mechanism, and fixed gate contract. It is not evidence that missingness is the true cause of an observed failure.

> Between-simulator spread is a model-sensitivity diagnostic, not a confidence interval.

OBS-085b does not alter the null FL3 result of OBS-084 and cannot increase any address beyond its frozen OBS-085a claim entitlement.
