# OBS-086c — Prospective Campaign Protocol Preregistration

## State

`prospective_campaign_protocol_preregistration_completed`

OBS-086c converts the five frozen OBS-086b protocol-selected families into complete prospective protocol profiles. It does not activate a campaign, assign a roster, reveal a randomization seed, run a simulation, or inspect observed scientific evidence.

## Frozen lineage

- OBS-086b commit: `dbd0b2171c854934ea6bd23dfbeecc80b44d7dbb`
- OBS-086b manifest ID: `d32f61955a41abf1ecd036863a766fbf0617fb2a0bf0a34b691487f9bed94119`
- OBS-086b manifest SHA256: `0adbab5209bf3ac4d8ee36df43ebcb5db0150a066d47235440b65e77f4c157d2`
- OBS-086b script SHA256: `0f9b4704f0ad3499f713bc0b0ce05c13ac4d37aea21776f45ad1ac26b85b6147`
- OBS-086b output artifacts validated: **13**
- Protocol semantic anchors frozen: **3**
- Current repository HEAD: `dbd0b2171c854934ea6bd23dfbeecc80b44d7dbb`

## Completion result

- Executable protocol profiles: **5**
- Nonactivatable held references: **1**
- Globally selected campaigns: **0**
- New simulations: **0**
- Observed evidence inspected: **0**
- Validation failures: **0**

## Executable protocol profiles

| protocol_profile_id | record_id | carrier | entitlement_status | reliability_target | discovery_nominal_k | confirmation_nominal_k | total_nominal_objects | defensible_stress_test_coverage | protocol_profile_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PR-b0a098e76aa6219c1eef6e42 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.5 | 12 | 12 | 24 | 0.68 | executable_pending_evidence_independent_activation |
| PR-b7df17242f938b67d3f9da24 | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.8 | 10 | 10 | 20 | 0.56 | executable_pending_evidence_independent_activation |
| PR-0acc5b724625a5033327986b | C_vs_Cp3__path_shares_only | path_shares_only | fl3_entitled | 0.9 | 12 | 12 | 24 | 0.52 | executable_pending_evidence_independent_activation |
| PR-8d5c884228be9454cd794d62 | three_way__no_window | no_window | fl3_entitlement_capped | 0.5 | 8 | 10 | 18 | 0.68 | executable_pending_evidence_independent_activation |
| PR-ae1bc88174da78efeb9bc1b5 | three_way__no_window | no_window | fl3_entitlement_capped | 0.8 | 8 | 12 | 20 | 0.56 | executable_pending_evidence_independent_activation |

No profile is activated by OBS-086c. A future activation record must use only evidence-independent entitlement, resource, carrier, measurement, structural-feasibility, and predeclared target criteria.

## Activation boundary

- Activation must precede outcome access.
- Delta and control-response lambda remain uncertainty axes and may not be selected as campaign properties.
- The two entitlement-capped profiles cannot be activated for an FL3 claim.
- OBS-086c selects no global winner among the five profiles.

## Commit–reveal partition contract

The assignment algorithm is frozen but no objects are assigned in OBS-086c. A seed commitment must be recorded before the eligible roster identity is known. After the roster is frozen and hashed, the seed is revealed and exact discovery/confirmation quotas are assigned deterministically while preventing object or effective-cluster overlap across partitions.

Discovery and confirmation may never be pooled, and no object may migrate after roster freeze.

## Artifact-derived reserve recommendations

| protocol_profile_id | partition | planned_analysis_objects | origin_minimum_support_efficiency | artifact_derived_maximum_screened_or_reserved_objects | artifact_derived_reserve_objects |
| --- | --- | --- | --- | --- | --- |
| PR-b0a098e76aa6219c1eef6e42 | confirmation | 12 | 0.742667 | 17 | 5 |
| PR-b0a098e76aa6219c1eef6e42 | discovery | 12 | 0.742667 | 17 | 5 |
| PR-b7df17242f938b67d3f9da24 | confirmation | 10 | 0.745 | 14 | 4 |
| PR-b7df17242f938b67d3f9da24 | discovery | 10 | 0.745 | 14 | 4 |
| PR-0acc5b724625a5033327986b | confirmation | 12 | 0.746833 | 17 | 5 |
| PR-0acc5b724625a5033327986b | discovery | 12 | 0.746833 | 17 | 5 |
| PR-8d5c884228be9454cd794d62 | confirmation | 10 | 0.4853 | 21 | 11 |
| PR-8d5c884228be9454cd794d62 | discovery | 8 | 0.4853 | 17 | 9 |
| PR-ae1bc88174da78efeb9bc1b5 | confirmation | 12 | 0.49725 | 25 | 13 |
| PR-ae1bc88174da78efeb9bc1b5 | discovery | 8 | 0.49725 | 17 | 9 |

The screening envelopes are calculated from the frozen origin minimum support-efficiency field. They are replacement-planning recommendations only: they do not enlarge the analysis allocation, guarantee effective support, or extrapolate the tested k grid.

## Held family reference

| held_reference_id | record_id | carrier | reliability_target | discovery_nominal_k | confirmation_nominal_k | defensible_stress_test_coverage | held_reference_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HR-33ca9c6662b7918c592e18c5 | three_way__no_window | no_window | 0.9 | 8 | 12 | 0.4 | nonactivatable_low_coverage_reference |

The held family remains nonactivatable because its maximum frozen stress-grid coverage is below majority coverage. OBS-086c cannot promote it.

## Frozen gate identities

| gate_order | gate_id | partition_scope | failure_status |
| --- | --- | --- | --- |
| 1 | discovery_partition_only | discovery | invalidate |
| 2 | candidate_family_sealed_before_confirmation | discovery | do_not_open_confirmation |
| 3 | protocol_match | confirmation | confirmation_protocol_mismatch |
| 4 | record_testable | confirmation | confirmation_not_testable |
| 5 | support_columns_available | confirmation | confirmation_support_unavailable |
| 6 | complement_admissible | confirmation | confirmation_complement_inadmissible |
| 7 | direction_match | confirmation | confirmation_direction_reversed |
| 8 | predicate_semantics_pass | confirmation | confirmation_signal_absent |
| 9 | minimum_effect_pass | confirmation | confirmation_signal_absent |
| 10 | cluster_sensitivity_pass | confirmation | confirmation_uncertain_cluster_support |
| 11 | control_robustness_pass | confirmation | confirmation_control_explained |
| 12 | confirmation_multiplicity_pass | confirmation | confirmation_multiplicity_not_survived |
| 13 | confirmation_eligible | confirmation | confirmation_reproduced_but_claim_capped_at_fl2 |

All thresholds, test definitions, controls, multiplicity logic, cluster semantics, and conjunction rules remain frozen. Threshold weakening, gate deletion, test substitution, one-sided conversion, partition pooling, and post hoc cluster redefinition are prohibited.

## Confirmation opening

Confirmation may be opened exactly once, only after the discovery roster, exclusions, evaluation, candidate identity, candidate family, manifest, and artifact hashes are frozen and verified. A failed discovery result does not authorize confirmation search.

## Claim entitlement

- `three_way__no_window`: maximum claim remains FL2 localized support; FL3 is prohibited.
- `C_vs_Cp3__path_shares_only`: FL3 artifact-direct witness remains conditionally available only if every frozen discovery and confirmation gate later passes.
- No profile authorizes causal origin, repair target, intervention readiness, actionability, external generalization, or formal topology claims.

## Interpretation boundary

> OBS-086c is a prospective protocol freeze only.

> It creates no witness, performs no campaign, and evaluates no scientific outcome.

> A protocol profile is not a guarantee of passage and does not increase claim entitlement.

> Discovery and confirmation remain separate; no frozen evidence gate may be weakened.
