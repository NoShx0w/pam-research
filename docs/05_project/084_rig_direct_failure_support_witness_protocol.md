# OBS-084 — RIG Direct Failure-Support Witness Protocol

**Status:** Canonical study protocol  
**Scope:** PAM/RIG artifact lineage through OBS-083  
**Primary role:** Operational specification for direct failure-support discovery, confirmation, and witness assignment  
**Claim level:** Diagnostic only; no intervention, causal, actionability, external-generalization, or formal-topology claim

---

## 1. Purpose

OBS-084 operationalizes Theory Memo 009 v2.

The study asks whether an OBS-083 diagnostic record can be tied to a versioned, predicate-specific, minimal sufficient artifact support through admissible matching, control adjustment, dependence-aware uncertainty, and reserved confirmation evidence.

The target claim is:

> A declared property of a specific registered relation degrades at this artifact support, relative to admissible complements and controls, within the declared PAM artifact lineage.

OBS-084 does not test repairs.

OBS-084 does not establish causal origin.

OBS-084 does not promote any record to actionable or intervention-ready status.

---

## 2. Canonical Guardrails

The following statements govern the study:

> **Directness is artifact-direct, not metaphysically direct and not causally direct.**

> **Localization is not atomization.**

> **A site is direct only through its witness.**

> **A localized degradation is record-, predicate-, contract-, and provenance-relative.**

> **Discovery nominates a support; reserved evidence earns the localization claim.**

The study must preserve the distinction:

\[
\text{weak record}
\neq
\text{failure support}
\neq
\text{causal origin}
\neq
\text{repair target}.
\]

It must also preserve:

\[
\text{signature concentration}
\neq
\text{failure concentration}.
\]

Earlier PAM localization artifacts may nominate candidate supports. They do not by themselves establish direct RIG failure supports.

---

## 3. Empirical Starting Point

OBS-083 leaves the current registry in two subclasses:

- 12 C1 contrast-limited records;
- 12 C2 localization-limited records.

It reports:

- zero direct artifact loci;
- zero R4/R5 repair annotations;
- zero C4 promising next-test candidates;
- zero candidate-ready or actionable records.

The immediate bottleneck is direct failure-support evidence.

### 3.1 Confirmation-eligible cohort

The 12 C2 localization-limited records are eligible for FL3 confirmation because their primary unresolved limitation is localization.

### 3.2 Discovery-only cohort

The 12 C1 contrast-limited records may enter candidate generation, but they remain capped at FL2 until record-specific negative-control evidence becomes adequate.

Formally:

\[
\text{C2 records}
\rightarrow
\text{FL3 confirmation eligible}
\]

while:

\[
\text{C1 records}
\rightarrow
\text{FL1/FL2 exploratory only}.
\]

---

## 4. Epistemic Unit

The unit of analysis is not:

\[
(\text{record},\text{localization score}).
\]

The unit is a versioned witness:

\[
w=
\left(
 r,
 q,
 \mu,
 z,
 m,
 \mathcal R_0,
 A,
 \Delta,
 \Lambda,
 \mathcal U,
 \Pi,
 V
\right).
\]

Where:

- \(r\): target registry record;
- \(q\): declared failure predicate;
- \(\mu\): failure mode;
- \(z\): minimal sufficient artifact support;
- \(m\): admissible matched complement;
- \(\mathcal R_0\): admissible control-record set;
- \(A\): retrievable artifact pointers;
- \(\Delta\): site-relative contrast;
- \(\Lambda\): control-adjusted contrast;
- \(\mathcal U\): dependence-aware uncertainty and replication evidence;
- \(\Pi\): provenance, matching, and analysis protocol;
- \(V\): version and reproducibility identity.

The direct support is not merely \(z\).

It is \(z\) as warranted by \(w\).

---

## 5. Study Architecture

OBS-084 is divided into mandatory stages.

## 5.1 OBS-084a — Candidate Discovery and Freeze

Purpose:

- construct candidate supports;
- declare failure predicates and modes;
- define site complements;
- define admissible control-record sets;
- create structural partitions;
- freeze the confirmation protocol;
- write immutable candidate manifests.

Maximum maturity:

\[
\mathrm{FL2}.
\]

## 5.2 OBS-084b — Reserved Confirmation and Witness Assignment

Purpose:

- validate the frozen manifest;
- unlock reserved evidence;
- compute predicate-specific site-relative contrasts;
- compute control-adjusted contrasts;
- perform dependence-aware uncertainty analysis;
- assign, reject, or retain candidate witnesses.

Maximum maturity:

\[
\mathrm{FL3}.
\]

## 5.3 OBS-084c — Internal Replication Audit

Purpose:

- test confirmed witnesses under held-out structural partitions;
- perform clustered structural resampling;
- assess support recurrence;
- assess contract-neighborhood stability;
- identify non-dominated support families.

Possible maturity:

\[
\mathrm{FL4}
\quad\text{or}\quad
\mathrm{FL5}.
\]

OBS-084c may remain part of the same study program.

---

## 6. Observation-Level Diagnostic Instrument

OBS-084 should construct a per-observation out-of-fold diagnostic table for every eligible relation × carrier record.

The classifier or separation model remains a measurement instrument. It must not be described as a causal or mechanistic model.

Each observation row should include, where available:

- `record_id`
- `relation`
- `carrier`
- `contract`
- `transformation`
- `observation_id`
- `object_id`
- `route_id`
- `transition_id`
- `window_id`
- `cohort`
- `true_regime`
- `predicted_regime`
- `predicted_probability`
- `signed_margin`
- `true_class_margin`
- `log_loss`
- `correct`
- `fold_id`
- `partition_role`
- `scale_band`
- `feature_family`
- `seam_relative_region`
- `boundary_relative_region`
- `provenance_id`

For pairwise records, a signed margin or target-class probability may be used.

For three-way records, a true-class margin may be defined as:

\[
m_i
=
p_i(y_i)
-
\max_{y\neq y_i}p_i(y).
\]

The chosen local loss must be declared before confirmation.

---

## 7. Failure Predicates and Modes

OBS-084 must not collapse all degradation into one score.

Each candidate declares a failure predicate \(q\) and a failure mode \(\mu\).

## 7.1 Relation-separation attenuation

The relation remains recoverable but separation is weaker within support \(z\).

Failure mode:

`attenuation`

## 7.2 Local criterion breach

The record falls below a declared diagnostic threshold within \(z\), while its admissible complement remains above that threshold.

Failure mode:

`threshold_breach`

## 7.3 Sign reversal

The relation changes direction within \(z\).

Failure mode:

`sign_reversal`

## 7.4 Contrast collapse

Target-control separation enters a declared equivalence or indistinguishability region within \(z\).

Failure mode:

`contrast_collapse`

## 7.5 Uncertainty breach

The estimate becomes too unstable or imprecise to support the declared property within \(z\).

Failure mode:

`uncertainty_breach`

## 7.6 Reproducibility loss

The relation becomes unstable across structural clusters or reserved partitions within \(z\).

Failure mode:

`reproducibility_loss`

## 7.7 Leakage dependence

The apparent relation degrades after removing or neutralizing leakage-prone information.

Failure mode:

`leakage_dependence`

## 7.8 Missingness concentration

Measurement undefinedness becomes concentrated within \(z\).

Failure mode:

`missingness_concentration`

Missingness follows the separate protocol in Section 16.

---

## 8. Candidate Support Vocabulary

OBS-084a may nominate candidate supports from a predeclared address vocabulary.

## 8.1 Structural supports

- object class;
- cohort;
- transition class;
- transition phase;
- bounded local window;
- route or path family.

## 8.2 Contract supports

- numeric transformation;
- feature-family projection;
- structural-resampling scheme;
- scale-band restriction.

## 8.3 Representational supports

- compact stability carrier;
- geometry-only carrier;
- stability-plus-geometry carrier;
- path-share carrier;
- non-window carrier;
- strict numeric carrier.

## 8.4 Geometry-relative supports

- seam-adjacent versus seam-far;
- boundary-adjacent versus boundary-far;
- response-ridge region;
- energy-ridge region;
- support-density region;
- route-relative region.

## 8.5 Provenance supports

- corpus origin;
- shaped-preamble lineage;
- model or generation campaign;
- contract-specific provenance slice.

## 8.6 Conjunctive supports

The protocol permits supports such as:

\[
z=(\text{middle scale},\text{exit transition})
\]

or:

\[
z=(\text{seam-adjacent window},\text{geometry carrier}).
\]

Initial discovery should cap conjunction depth, preferably at two address dimensions, unless a deeper conjunction was predeclared from prior evidence.

---

## 9. Minimal Sufficient Supports

A failure support is not defined as the narrowest row or smallest geometric region.

It is:

> The minimal sufficient artifact support at which a declared failure predicate can be distinguished from its admissibly matched complement.

Let:

\[
\operatorname{Sufficient}_q(r,z)
\]

mean that support \(z\) contains enough evidence to establish criterion-specific degradation for record \(r\).

Let:

\[
z_1\preceq z_2
\]

mean that \(z_1\) is an admissible refinement of \(z_2\).

Then \(z\) is minimally sufficient when:

\[
\operatorname{Sufficient}_q(r,z)
\]

and no proper admissible refinement \(z'\prec z\) also satisfies:

\[
\operatorname{Sufficient}_q(r,z').
\]

Minimality is relative to the artifact representation and evidence resolution.

It is not an ontological claim.

---

## 10. Non-Unique Supports

A failure may admit several incomparable minimal sufficient supports.

OBS-084 should report the non-dominated support family:

\[
\mathcal Z_{\min}(r,q)
=
\left\{
z:
\operatorname{Sufficient}_q(r,z)
\land
\nexists z'\prec z
\text{ such that }
\operatorname{Sufficient}_q(r,z')
\right\}.
\]

The study should distinguish:

- syntactic refinement;
- empirical containment;
- empirical overlap;
- functional equivalence;
- incomparability.

Required fields should include:

- `support_dominance_status`
- `support_relationship`
- `equivalent_or_overlapping_supports`
- `preferred_support_reason`

Canonical rule:

> A failure may admit multiple direct supports. OBS-084 reports the non-dominated support family rather than forcing a unique point of failure.

---

## 11. Structural Partitioning

Discovery and confirmation must use structurally separated evidence.

Preferred partitioning units are:

1. object;
2. route or trajectory;
3. transition block;
4. non-overlapping structural cluster;
5. provenance campaign.

Overlapping windows from the same route or transition must remain in the same partition.

The partition must be:

- deterministic;
- hash-based where practical;
- stratified by regime and provenance where support permits;
- written to a committed manifest;
- inaccessible to discovery-stage confirmation metrics.

Possible roles:

- `discovery`
- `confirmation`
- `replication`

A 60/20/20 split is acceptable only when independent-cluster support is sufficient.

Where support is limited, a discovery/confirmation split plus dependence-aware resampling may be preferable.

---

## 12. Frozen Candidate Manifest

Before confirmation begins, OBS-084 must freeze each candidate.

Required fields:

- `candidate_id`
- `record_id`
- `failure_predicate`
- `failure_mode`
- `support_definition`
- `support_query`
- `complement_definition`
- `matching_variables`
- `exposure_normalization`
- `control_admissibility_rule`
- `eligible_control_records`
- `metric`
- `expected_direction`
- `threshold_basis`
- `minimum_effect`
- `uncertainty_method`
- `resampling_unit`
- `confirmation_partition_id`
- `multiplicity_family`
- `exclusion_rules`
- `source_artifacts`

The candidate manifest must receive a deterministic identifier:

`candidate_manifest_id`

The confirmation command must refuse to run if the frozen manifest, partition manifest, source hashes, or protocol identity do not validate.

Any material post-freeze change must be recorded.

Material changes include:

- support definition;
- metric;
- expected direction;
- complement;
- control set;
- threshold;
- exclusion rule;
- confirmation partition.

Canonical rule:

> A materially altered candidate returns to FL2 until evaluated on fresh reserved evidence.

---

## 13. Admissible Site Complements

For each candidate support \(z\), OBS-084 must construct an admissible complement \(m(z)\).

The comparison should differ on the candidate address while remaining balanced on declared nuisance structure.

The admissibility audit must cover:

### 13.1 Support overlap

Both site and complement must contain enough comparable observations and independent clusters.

### 13.2 Match balance

Balance should be reported over relevant variables such as:

- regime;
- object or route identity;
- scale support;
- transition opportunity;
- provenance;
- contract exposure;
- valid observation count;
- baseline uncertainty.

### 13.3 Exposure normalization

Where relevant, normalize for:

- route length;
- window count;
- seam-contact duration;
- transition opportunity;
- object occupancy;
- support density;
- scale exposure.

This is structural comparability discipline, not causal matching.

Candidates with inadequate overlap or balance remain FL2.

---

## 14. Admissible Control Records

Define:

\[
\mathcal R_0(r,q,z)
\]

as the set of records admissible as controls for target record \(r\), predicate \(q\), and support \(z\).

Control admissibility should consider:

- baseline survival;
- carrier capacity or dimensionality;
- scale support;
- valid observation count;
- provenance;
- contract exposure;
- baseline uncertainty;
- failure-mode comparability;
- structural opportunity.

Two control families are especially useful.

## 14.1 Relation controls

Same carrier, different relation.

These test whether the support is difficult specifically for the target relation.

## 14.2 Carrier controls

Same relation, different carrier.

These test whether degradation is specific to the target representation.

The study should report:

- single-control evidence;
- multi-control directional robustness;
- control-sensitive localization.

Required fields:

- `control_record_id`
- `control_admissibility_rule`
- `control_balance_summary`
- `alternative_control_count`
- `control_robustness_status`

A candidate that depends on one weakly justified control remains FL2.

---

## 15. Predicate-Specific Contrasts

Let:

\[
L_q(r,z)
\]

be the predicate-indexed loss or degradation measure.

The site-relative contrast is:

\[
\Delta_q(r,z)
=
L_q(r,z)-L_q(r,m(z)).
\]

The control-adjusted contrast is:

\[
\Lambda_q(r,z;r_0)
=
\Delta_q(r,z)-\Delta_q(r_0,z).
\]

The first asks whether degradation is stronger at the proposed support than at its complement.

The second asks whether the support degrades the target more than an admissible control record.

Thresholds must derive from at least one of:

1. a scientifically meaningful minimum effect;
2. a null or negative-control distribution;
3. an uncertainty-based criterion.

For attenuation, a conservative confirmation rule may require:

\[
\operatorname{LCB}_{1-\alpha}
[\Delta_q]
>
\delta_q
\]

and:

\[
\operatorname{LCB}_{1-\alpha}
[\Lambda_q]
>
\lambda_q.
\]

Other predicates require their own confirmation semantics.

No generic scalar rule should be imposed on every failure mode.

---

## 16. Missingness Subprotocol

Missingness may reflect:

- structural absence;
- insufficient support;
- a mathematically undefined statistic;
- numerical failure;
- pipeline defect;
- unavailable artifacts;
- contract-defined exclusion;
- unexplained missingness.

Recommended reason taxonomy:

- `zero_variance`
- `zero_denominator`
- `empty_support`
- `insufficient_event_count`
- `non_finite_upstream_value`
- `failed_join`
- `unavailable_artifact`
- `contract_defined_exclusion`
- `unexplained_missingness`

Define:

\[
M(r,z)
=
\Pr(
\text{measurement undefined}
\mid r,z
).
\]

Then:

\[
\Delta_M(r,z)
=
M(r,z)-M(r,m(z))
\]

and:

\[
\Lambda_M(r,z;r_0)
=
\Delta_M(r,z)-\Delta_M(r_0,z).
\]

A confirmed missingness result initially licenses only:

> Measurement undefinedness is localized at this support for this record relative to its matched complement and admissible controls.

It does not automatically establish reusable-invariance failure.

Required fields:

- `missingness_reason`
- `reason_resolution_status`
- `technical_failure_excluded`
- `numerical_failure_excluded`
- `support_failure_excluded`
- `structural_interpretation_eligible`
- `rig_property_link_evidence`

---

## 17. Multiplicity

OBS-084 may search across many records, predicates, supports, contracts, and partitions.

The study must report the full candidate denominator.

Required counts include:

- total candidates generated;
- candidates per record;
- candidates per predicate;
- candidates per support family;
- candidates frozen;
- candidates confirmed;
- candidates rejected;
- multiplicity family;
- correction or filtering rule.

Multiplicity may be handled through:

- reserved confirmation;
- null distributions;
- family-wise thresholds;
- false-discovery control;
- hierarchical testing;
- explicit exploratory labeling.

The protocol need not impose one universal correction.

It must make the search space visible.

---

## 18. Dependence-Aware Uncertainty

PAM artifact rows are not necessarily independent.

Rows may share:

- objects;
- routes;
- overlapping windows;
- transitions;
- model runs;
- corpus campaigns;
- provenance lineage.

Ordinary row-level resampling is not sufficient.

Possible resampling units include:

- object bootstrap;
- route bootstrap;
- transition-block bootstrap;
- cohort-stratified bootstrap;
- non-overlapping window-block bootstrap;
- provenance-cluster bootstrap.

Each candidate or witness should report:

- `replication_unit`
- `resampling_unit`
- `cluster_definition`
- `independent_cluster_count`
- `effective_support`
- `point_estimate`
- `confidence_interval`
- `direction_consistency`
- `successful_resample_count`

Canonical rule:

> Resampling must occur at the level at which evidence can reasonably be treated as structurally independent.

Candidates with inadequate independent support remain FL2.

---

## 19. Reproducibility and Replication

The study must distinguish:

### 19.1 Computational reproducibility

The same code and committed artifacts recover the same witness.

### 19.2 Resampling robustness

The result survives dependence-aware structural resampling.

### 19.3 Partition replication

The support recurs in held-out objects, routes, transitions, or reserved artifact partitions.

### 19.4 Generation replication

The support recurs in an independently generated artifact campaign.

### 19.5 External replication

The support recurs under another model, corpus lineage, or broader scope.

OBS-084 may reasonably target the first three.

Internal replication must not be described as external generalization.

---

## 20. Versioned Witness Identity

Each witness must have a stable identity.

Conceptually:

\[
\operatorname{witness\_id}
=
H(
 r,
 q,
 \mu,
 z,
 m,
 \mathcal R_0,
 A,
 \Pi,
 V_s
),
\]

where \(H\) is a deterministic hash and \(V_s\) is the witness-schema version.

Canonical serialization is required so field ordering, path ordering, dictionary order, and floating-point formatting do not create accidental identity changes.

Required version fields:

- `witness_id`
- `witness_schema_version`
- `record_version`
- `source_artifact_hashes`
- `source_artifact_manifest_id`
- `code_commit`
- `candidate_manifest_id`
- `confirmation_protocol_id`
- `confirmation_partition_id`
- `matching_protocol_id`
- `created_at`
- `supersedes_witness_id`
- `witness_status`

Suggested statuses:

- `candidate`
- `frozen`
- `confirmed`
- `replicated`
- `superseded`
- `invalidated`
- `non_reproducible`
- `scope_restricted`

A later repair hypothesis must reference the exact `witness_id`.

---

## 21. Direct-Witness Admissibility

A direct witness requires:

\[
\operatorname{RecordDeclared}(r)
\]

\[
\land
\operatorname{CriterionDeclared}(q)
\]

\[
\land
\operatorname{FailureModeDeclared}(\mu)
\]

\[
\land
\operatorname{MinimalSufficientSupport}(z)
\]

\[
\land
\operatorname{ArtifactAddressable}(z,A)
\]

\[
\land
\operatorname{AdmissiblyMatched}(z,m)
\]

\[
\land
\operatorname{ControlAdmissible}(r,\mathcal R_0)
\]

\[
\land
\operatorname{PredicateSpecificEvidence}(\Delta,\Lambda)
\]

\[
\land
\operatorname{DependenceAwareUncertainty}(\mathcal U)
\]

\[
\land
\operatorname{ProvenanceComplete}(\Pi)
\]

\[
\land
\operatorname{LeakageSafe}(\Pi)
\]

\[
\land
\operatorname{ConfirmedOnReservedEvidence}(\Pi)
\]

\[
\land
\operatorname{VersionIdentified}(V).
\]

If any required component is missing or invalid, the candidate cannot reach FL3.

---

## 22. Failure-Localization Maturity Ladder

### FL0 — Unlocalized limitation

A readiness limitation is known, but no credible support has been identified.

### FL1 — Contrast-proxy localization

Aggregate relation, carrier, contract, scale, feature, or geometry contrasts suggest an address family.

Evidence remains indirect.

### FL2 — Artifact-indexed candidate support

A retrievable candidate exists, but one or more required audits remain incomplete.

### FL3 — Confirmed direct witness

A frozen candidate is confirmed through the complete direct-witness protocol.

FL3 is the minimum level that counts as direct failure-support evidence.

### FL4 — Internally replicated direct support

The direct witness survives dependence-aware resampling and held-out structural partitions.

### FL5 — Contract-neighborhood-stable localization

The replicated support remains identifiable across scientifically reasonable variations within the declared contract neighborhood while preserving the expected scope and direction of degradation.

A contract-local failure need not survive outside the contract family that defines it.

No FL level establishes causal origin, repairability, intervention-readiness, external generalization, or formal topology.

No FL level automatically changes the OBS-082 readiness class.

---

## 23. Hard Promotion Rules

### 23.1 FL1

Aggregate contrast points toward a support family.

### 23.2 FL2

The candidate is artifact-indexed but discovery-stage or incompletely matched, controlled, or confirmed.

### 23.3 FL3

Requires all of:

- C2 confirmation eligibility;
- frozen candidate manifest;
- valid support/complement matching;
- admissible record control;
- predicate-specific site-relative contrast;
- control-adjusted contrast;
- dependence-aware uncertainty;
- reserved confirmation evidence;
- complete provenance;
- leakage-safe evidence;
- stable witness identity.

### 23.4 FL4

FL3 plus internal partition replication and structural resampling.

### 23.5 FL5

FL4 plus robustness within the declared contract neighborhood.

C1 records remain capped at FL2 in OBS-084 unless a separate negative-control audit changes their eligibility.

---

## 24. Required Outputs

Default output directory:

`outputs/rig_registry/obs084_direct_failure_witness/`

### 24.1 Discovery and freeze

- `obs084_input_manifest.csv`
- `obs084_observation_loss_table.csv`
- `obs084_discovery_candidate_manifest.csv`
- `obs084_candidate_support_index.csv`
- `obs084_confirmation_protocol.json`
- `obs084_reserved_partition_manifest.csv`
- `obs084_candidate_changes_after_freeze.csv`

### 24.2 Matching and controls

- `obs084_site_complement_admissibility.csv`
- `obs084_control_record_admissibility.csv`
- `obs084_match_balance_audit.csv`
- `obs084_exposure_normalization_audit.csv`

### 24.3 Confirmation

- `obs084_predicate_specific_contrasts.csv`
- `obs084_control_adjusted_contrasts.csv`
- `obs084_dependence_aware_uncertainty.csv`
- `obs084_missingness_reason_audit.csv`
- `obs084_confirmation_results.csv`

### 24.4 Witness layer

- `obs084_witness_manifest.csv`
- `obs084_direct_witnesses.csv`
- `obs084_non_dominated_support_families.csv`
- `obs084_support_overlap_and_equivalence.csv`
- `obs084_replication_audit.csv`
- `obs084_witness_version_history.csv`
- `obs084_report.md`

---

## 25. Required Witness Fields

Each direct-witness row should include at least:

- `witness_id`
- `witness_schema_version`
- `record_id`
- `relation`
- `carrier`
- `contract`
- `transformation`
- `failure_predicate`
- `failure_mode`
- `support_definition`
- `support_id`
- `support_dominance_status`
- `equivalent_or_overlapping_supports`
- `matched_complement_definition`
- `control_record_id`
- `control_admissibility_rule`
- `site_relative_contrast`
- `control_adjusted_contrast`
- `threshold_basis`
- `minimum_effect`
- `confidence_interval`
- `resampling_unit`
- `independent_cluster_count`
- `confirmation_partition_id`
- `artifact_pointers`
- `source_artifact_hashes`
- `candidate_manifest_id`
- `code_commit`
- `provenance_complete`
- `leakage_check`
- `localization_level`
- `witness_status`
- `scope_statement`
- `supersedes_witness_id`

---

## 26. Command-Level Execution Contract

The implementation should expose separate commands:

```bash
python experiments/studies/obs084_rig_direct_failure_witness.py discover
python experiments/studies/obs084_rig_direct_failure_witness.py freeze
python experiments/studies/obs084_rig_direct_failure_witness.py confirm
python experiments/studies/obs084_rig_direct_failure_witness.py replicate
```

### 26.1 `discover`

May read discovery partitions and upstream artifacts.

May produce FL1/FL2 candidates.

Must not inspect reserved confirmation outcomes.

### 26.2 `freeze`

Writes canonical candidate, protocol, source-hash, and partition manifests.

Must assign deterministic manifest identifiers.

### 26.3 `confirm`

Must refuse to run unless:

- the candidate manifest exists;
- the confirmation protocol exists;
- source hashes validate;
- the partition manifest validates;
- the candidate-manifest hash validates;
- no undeclared material changes are present.

### 26.4 `replicate`

May operate only on confirmed witnesses.

Must preserve the original witness identity and write a new version or replication record rather than silently replacing evidence.

The refusal behavior is part of the scientific protocol.

---

## 27. Legitimate Outcomes

### 27.1 Positive result

One or more C2 records reach FL3 or higher with versioned direct witnesses.

### 27.2 Partial result

Candidates reach FL2 but fail confirmation because of:

- weak support;
- generic site difficulty;
- control sensitivity;
- poor matching;
- unstable direction;
- inadequate independent clusters;
- failed reserved confirmation.

### 27.3 Null result

No direct witnesses are confirmed.

Canonical interpretation:

> Current reusable-invariance failures remain distributed, under-resolved, or inaccessible through the present artifact hierarchy.

A null result may indicate a need for:

- finer artifact generation;
- improved transition or window alignment;
- stronger provenance indexing;
- more appropriate structural units;
- stronger matching;
- larger independent support;
- a different localization instrument.

A null result does not justify inventing a repair target.

---

## 28. Repair-Hypothesis Boundary

A future repair hypothesis may be represented as:

\[
H_\rho=
(
 w,
 \rho,
 M_\rho,
 \mathcal C_\rho,
 \Phi,
 \Omega
),
\]

where:

- \(w\): exact confirmed witness;
- \(\rho\): proposed repair;
- \(M_\rho\): repair-success metric;
- \(\mathcal C_\rho\): repair-specific controls;
- \(\Phi\): falsification condition;
- \(\Omega\): scope.

OBS-084 must not construct or execute \(\rho\).

A later repair hypothesis must reference the exact `witness_id`.

---

## 29. Canonical Claim Templates

### FL1

> Matched-control evidence suggests that degradation may be concentrated in the declared support family, but the evidence remains contrast-derived and does not establish a direct artifact support.

### FL2

> The record has a retrievable candidate failure support, but matching, control adjustment, uncertainty, or confirmation evidence remains incomplete.

### FL3

> Within the declared PAM artifact lineage, contract, and provenance scope, degradation of the specified registry record under the declared predicate is directly supported at the named minimal sufficient artifact support relative to its admissible complement and controls.

### FL4

> The direct failure support survives dependence-aware resampling and recurs across the declared held-out structural partitions.

### FL5

> The replicated failure support remains identifiable across scientifically reasonable variations within its declared contract neighborhood while preserving the expected scope and direction of degradation.

### Missingness

> Measurement undefinedness is concentrated within the declared support for this record relative to its matched complement and admissible controls. This does not by itself establish reusable-invariance failure.

### Required guardrail

> This witness identifies where degradation is empirically addressable. It does not establish causal origin, repairability, intervention-readiness, external generalization, or formal topology.

---

## 30. Scope Discipline

All OBS-084 claims remain conditional on:

- the current PAM artifact lineage;
- the current model and corpus provenance;
- the C/Cp2/Cp3 comparison family where applicable;
- the OBS-078–083 feature and contract lineage;
- the tested relation and carrier families;
- the declared matching and control procedures;
- the current committed source artifacts;
- the declared confirmation partitions.

The term *direct* must never imply:

- direct access to cognition;
- direct access to human phenomenology;
- direct access to hidden-state mechanisms;
- model-independent structure;
- corpus-independent structure;
- universal reusable invariance;
- formal topological singularity;
- causal control;
- successful repair.

---

## 31. Canonical Result Statement Template

> OBS-084 constructs and audits versioned, predicate-specific failure-support witnesses over the OBS-083 diagnostic registry. Candidate supports are generated from existing PAM artifact structure, frozen before confirmation, evaluated against admissible complements and control records, and tested using dependence-aware uncertainty on reserved evidence. Any confirmed witness establishes artifact-level diagnostic addressability only. OBS-084 performs no interventions and establishes no causal origin, repairability, actionability, external generalization, or formal topology.

---

## 32. Final Position

OBS-084 is not a search for visually persuasive locations.

It is a test of whether PAM has earned the right to make a stronger diagnostic statement.

The transition is:

\[
\text{contrast-derived suspicion}
\rightarrow
\text{artifact-indexed candidate}
\rightarrow
\text{frozen candidate}
\rightarrow
\text{confirmed direct witness}
\rightarrow
\text{replicated failure support}.
\]

The study succeeds only when claim entitlement is made inspectable.

---

## Canonical One-Sentence Summary

> OBS-084 tests whether a declared property of a specific RIG registry record can be shown to degrade at a versioned, minimal sufficient artifact support relative to admissible complements and controls under dependence-aware uncertainty and reserved confirmation evidence; any resulting witness establishes diagnostic addressability, not causal origin, repairability, or intervention-readiness.
