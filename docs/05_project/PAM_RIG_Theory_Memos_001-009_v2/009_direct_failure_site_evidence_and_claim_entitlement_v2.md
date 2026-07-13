# Theory Memo 009 v2 — Direct Failure-Site Evidence and Claim Entitlement in RIG

**Status:** Canonical conceptual and operational specification  
**Scope:** PAM/RIG artifact lineage through OBS-083  
**Primary role:** Epistemic protocol and design specification for OBS-084  
**Claim level:** Theory-guided diagnostic specification; no intervention, causal, or universal claim

## 1. Purpose

The Reusable Invariance Geometry program has reached a methodological boundary.

OBS-081 established a registry of relation × carrier records.

OBS-082 audited those records for readiness and found all 24 diagnostic-only.

OBS-083 strengthened matched relation, carrier, contract, and geometry-needed contrasts. It refined the flat diagnostic-only class into:

\[
12\ \text{contrast-limited records}
\]

and:

\[
12\ \text{localization-limited records}.
\]

However:

\[
\text{direct artifact locus rows}=0
\]

and:

\[
\text{R4/R5 repair annotations}=0.
\]

The current RIG registry can therefore say:

- a relation survives;
- it survives better or worse than matched alternatives;
- its weakness appears associated with a carrier, contract, or geometry-needed condition;
- its readiness is limited by contrast or localization.

It cannot yet responsibly say:

> Reuse fails here.

This memo defines the evidence required to make that stronger statement.

Its purpose is not merely to propose a new study. It specifies what OBS-084 is and is not allowed to claim.

The methodological transition is:

\[
\text{registry classification}
\longrightarrow
\text{artifact-indexed diagnosis}.
\]

The deeper transition is:

\[
\text{pattern recognition}
\longrightarrow
\text{claim entitlement}.
\]

## 2. Canonical Guardrails

> **Directness is artifact-direct, not metaphysically direct and not causally direct.**

> **Localization is not atomization.**

> **A site is direct only through its witness.**

> **A localized degradation is record-, predicate-, contract-, and provenance-relative.**

> **Discovery nominates a support; reserved evidence earns the localization claim.**

These guardrails prevent five common errors:

1. treating an artifact location as self-evidently explanatory;
2. treating a narrow site as more direct than a distributed support;
3. treating association as causal origin;
4. treating one artifact lineage as universal;
5. treating discovery evidence as confirmation.

## 3. Central Distinctions

\[
\text{weak record}
\neq
\text{failure support}
\neq
\text{causal origin}
\neq
\text{repair target}.
\]

A weak record may show reduced survival or contrast without revealing where the weakness is concentrated.

A failure support is an artifact address or distributed artifact support where a declared property degrades relative to admissible comparisons.

A causal origin would explain why the degradation occurs.

A repair target is a confirmed support attached to a separate falsifiable proposal for modifying the system or measurement process.

None of these transitions is automatic.

Similarly:

\[
\text{signature concentration}
\neq
\text{failure concentration}.
\]

Earlier PAM studies may show where a regime signature is strongest. That does not establish where a registered relation loses reusable support.

## 4. The Revised Epistemic Unit

The original localization unit was approximately:

\[
(\text{record},\text{localization score}).
\]

The revised epistemic unit is a versioned witness:

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
- \(V\): reproducibility and version identity.

The scientific claim does not exist through \(z\) alone.

It exists through the conjunction:

\[
(r,q,\mu,z,m,\mathcal R_0,A,\Delta,\Lambda,\mathcal U,\Pi,V).
\]

Therefore:

\[
\text{the direct site is not merely }z.
\]

Rather:

\[
\text{the direct site is }z\text{ as warranted by }w.
\]

## 5. Core Definitions

### 5.1 Registry record

\[
r=(R,c,k,t,\Omega)
\]

where \(R\) is the relation, \(c\) the carrier, \(k\) the contract, \(t\) the transformation, and \(\Omega\) the declared scope.

### 5.2 Failure predicate

A predicate \(q\) defines which property is expected to hold and what counts as degradation.

Possible predicates include survival, matched-control contrast, sign consistency, contract stability, carrier breadth, scale robustness, reproducibility, uncertainty tolerance, leakage independence, and measurement validity.

A site cannot be localized without a declared failure predicate.

### 5.3 Failure mode

The failure mode \(\mu\) identifies how the predicate degrades.

OBS-084 should distinguish at least:

- `attenuation`;
- `threshold_breach`;
- `sign_reversal`;
- `contrast_collapse`;
- `uncertainty_breach`;
- `reproducibility_loss`;
- `leakage_dependence`;
- `missingness_concentration`.

### 5.4 Structural address

A structural address is an inspectable part or conjunction within the artifact system.

Let:

\[
z=(b,\mathcal F,u,g,p)
\]

where the components may include scale band, feature family, structural unit, geometry-relative region, and provenance slice.

### 5.5 Candidate failure support

A structural address where discovery evidence suggests criterion-specific degradation.

### 5.6 Direct failure support

A candidate confirmed through a declared predicate and mode, artifact addressability, admissible matching, admissible control comparison, predicate-specific evidence, dependence-aware uncertainty, provenance preservation, anti-leakage checks, reserved confirmation evidence, and reproducible witness identity.

### 5.7 Failure site

Shorthand for a direct or replicated failure support.

A site need not be spatial, atomic, or unique.

### 5.8 Causal origin

The explanation of why degradation occurs. Direct localization does not establish this.

### 5.9 Repair target

A replicated failure support attached to a separate falsifiable repair hypothesis.

## 6. Minimal Sufficient Artifact Support

A support should not be defined as the smallest available row, object, or geometric region.

> A failure support is a minimal sufficient artifact support at which a declared failure predicate can be distinguished from its admissibly matched complement.

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

Then \(z\) is minimally sufficient when it is sufficient and no proper admissible refinement remains sufficient.

Minimality is relative to the artifact representation and evidence resolution. It is not an ontological claim.

## 7. Non-Unique and Incomparable Supports

A failure may admit several incomparable minimal sufficient supports.

Define:

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

Canonical rule:

> A failure may admit multiple direct supports. PAM should report the set of non-dominated supports rather than forcing a single preferred site.

OBS-084 should record support dominance, overlap, equivalence, and any operational preference reason.

## 8. Representation-Aware Support Relations

The artifact hierarchy should not be assumed to form one clean lattice.

OBS-084 should distinguish:

- syntactic refinement;
- empirical containment;
- empirical overlap;
- functional equivalence;
- incomparability.

## 9. Site-Relative Contrast

Let:

\[
L_q(r,z)
\]

be a predicate-indexed degradation measure.

Let \(m(z)\) be an admissibly matched complement.

\[
\Delta_q(r,z)
=
L_q(r,z)-L_q(r,m(z)).
\]

This asks whether degradation is stronger at the proposed support than in its matched complement.

## 10. Control-Adjusted Contrast

To guard against generic site difficulty, define:

\[
\Lambda_q(r,z;r_0)
=
\Delta_q(r,z)-\Delta_q(r_0,z).
\]

This asks whether the support degrades the target record more than an admissible control record.

A direct witness should ordinarily include both site-relative and control-adjusted evidence.

## 11. Admissible Site Complements

The complement must be comparable on nuisance structure while differing on the candidate support.

Minimum audits:

- support overlap;
- match balance;
- exposure normalization.

Relevant exposure variables may include route length, window count, seam-contact duration, transition opportunity, object occupancy, support density, and scale exposure.

This is structural comparability discipline, not causal matching.

## 12. Admissible Control Records

Define:

\[
\mathcal R_0(r,q,z)
\]

as the set of records admissible as controls.

Controls should be comparable in baseline survival, carrier capacity, scale support, observation count, provenance, contract exposure, uncertainty, failure mode, and structural opportunity.

OBS-084 should distinguish single-control evidence, multi-control robustness, and control-sensitive localization.

## 13. Predicate-Specific Evidence

Thresholds must be evidential, not decorative.

They should derive from:

1. a scientifically meaningful minimum effect size;
2. a null or negative-control distribution;
3. an uncertainty-based criterion.

For attenuation, one possible rule is:

\[
\operatorname{LCB}_{1-\alpha}[\Delta_q(r,z)]>\delta_q
\]

and:

\[
\operatorname{LCB}_{1-\alpha}[\Lambda_q(r,z;r_0)]>\lambda_q.
\]

Other failure modes require their own predicates:

- threshold breach;
- sign reversal;
- contrast collapse;
- uncertainty breach;
- reproducibility loss;
- leakage dependence;
- missingness concentration.

## 14. Local Attenuation Versus Failure

OBS-084 should distinguish:

- localized attenuation;
- localized criterion breach;
- localized reversal;
- localized ambiguity;
- localized uncertainty failure.

A mild decrement should not be narrated as a complete breakdown.

## 15. Missingness Subprotocol

Missingness may indicate structural absence, insufficient support, mathematical undefinedness, numerical failure, pipeline defect, unavailable artifact, contract-defined exclusion, or unexplained missingness.

Recommended reason taxonomy:

- `zero_variance`;
- `zero_denominator`;
- `empty_support`;
- `insufficient_event_count`;
- `non_finite_upstream_value`;
- `failed_join`;
- `unavailable_artifact`;
- `contract_defined_exclusion`;
- `unexplained_missingness`.

Define:

\[
M(r,z)=\Pr(\text{measurement undefined}\mid r,z).
\]

Then:

\[
\Delta_M(r,z)=M(r,z)-M(r,m(z))
\]

and:

\[
\Lambda_M(r,z;r_0)=\Delta_M(r,z)-\Delta_M(r_0,z).
\]

A confirmed missingness witness initially licenses only a claim about localized measurement undefinedness, not automatically reusable-invariance failure.

## 16. Discovery and Confirmation

OBS-084 must separate discovery from confirmation.

### Discovery

Use existing artifacts to nominate candidate supports. Discovery can establish FL1 or FL2 only.

### Frozen candidate stage

Before reserved evidence is inspected, freeze:

- target record;
- predicate;
- failure mode;
- support;
- complement rule;
- control rule;
- metric;
- expected direction;
- thresholds;
- matching variables;
- exclusions;
- resampling unit;
- confirmation partition;
- multiplicity family.

Suggested artifacts:

- `obs084_discovery_candidate_manifest.csv`;
- `obs084_confirmation_protocol.json`;
- `obs084_reserved_partition_manifest.csv`.

### Confirmation

Only reserved evidence may establish FL3 or higher.

### Post-freeze changes

Any material change should be recorded and return the candidate to FL2 until reconfirmed on fresh reserved evidence.

## 17. Multiplicity

OBS-084 should report the full candidate denominator and the chosen multiplicity strategy.

At minimum, report total candidates generated, candidates per record and predicate, number frozen, number confirmed, number rejected, and the multiplicity family.

## 18. Dependence-Aware Uncertainty

PAM artifact rows may share objects, routes, overlapping windows, transitions, model runs, or provenance campaigns.

Resampling should occur at the level of the scientific unit.

Witness fields should include:

- replication unit;
- resampling unit;
- independent cluster count;
- effective support;
- confidence interval;
- direction consistency;
- cluster definition.

## 19. Reproducibility and Replication

Distinguish:

1. computational reproducibility;
2. resampling robustness;
3. partition replication;
4. generation replication;
5. external replication.

OBS-084 may reasonably target the first three.

## 20. Versioned Witness Identity

Each witness must have a stable identity:

\[
\operatorname{witness\_id}
=
H(r,q,\mu,z,m,\mathcal R_0,A,\Pi,V_s).
\]

The hash must use canonical serialization.

Required fields include witness schema version, source artifact hashes, code commit, candidate manifest, confirmation protocol, partition ID, matching protocol, creation time, status, and supersession link.

## 21. Direct-Witness Admissibility

A direct witness requires all of the following:

- declared record;
- declared predicate;
- declared failure mode;
- minimal sufficient support;
- artifact addressability;
- admissible complement;
- admissible control set;
- predicate-specific site and control contrasts;
- dependence-aware uncertainty;
- complete provenance;
- leakage safety;
- reserved confirmation;
- version identity.

## 22. Failure-Localization Maturity Ladder

### FL0 — Unlocalized limitation

A limitation is known but no credible support exists.

### FL1 — Contrast-proxy localization

Aggregate evidence suggests an address family.

### FL2 — Artifact-indexed candidate support

A retrievable candidate exists, but matching, control adjustment, uncertainty, provenance, freeze, or confirmation remains incomplete.

### FL3 — Confirmed direct witness

A frozen candidate is confirmed through the full witness protocol.

FL3 is the minimum level that counts as direct failure-site evidence.

### FL4 — Internally replicated direct support

The witness survives dependence-aware resampling and held-out structural partitions.

### FL5 — Contract-neighborhood-stable localization

The replicated support remains identifiable across scientifically reasonable variations within its declared contract neighborhood.

Even FL5 does not establish causal origin, repairability, intervention readiness, external generalization, or formal topology.

## 23. What Does Not Count as Direct Evidence

Insufficient on their own:

- registry metadata;
- aggregate score differences;
- large relation or carrier contrast;
- feature importance;
- geometry enrichment;
- seam proximity;
- strong regime localization;
- missingness without the subprotocol;
- a single extreme row;
- post hoc subgroup selection;
- repair language;
- causal narration.

## 24. Falsification and Disqualification

A claim should fail or remain below FL3 under:

- diffusion;
- generic site difficulty;
- sign instability;
- sparse support;
- identity leakage;
- match failure;
- exposure confounding;
- contract collapse;
- provenance ambiguity;
- global weakness;
- post-selection failure;
- metric substitution;
- control dependence;
- non-reproducibility.

## 25. Null Results

A null result is scientifically valid:

> Current reusable-invariance failures remain distributed, under-resolved, or inaccessible through the present artifact hierarchy.

A null result does not justify inventing a repair target.

## 26. Consequences for OBS-084

### Proposed title

**OBS-084 — RIG Direct Failure-Support Witness Construction and Confirmation**

### Core question

> Can OBS-083 contrast-limited and localization-limited records be tied to confirmed, versioned, predicate-specific artifact supports through admissible matching, control adjustment, dependence-aware uncertainty, and reserved evidence?

### Primary analysis unit

\[
(r,q,\mu,z,m,\mathcal R_0,w).
\]

### Study phases

\[
\text{candidate generation}
\rightarrow
\text{candidate freeze}
\rightarrow
\text{reserved evidence unlock}
\rightarrow
\text{confirmation}
\rightarrow
\text{witness assignment}
\rightarrow
\text{replication audit}.
\]

### Suggested artifacts

Discovery and freeze:

- `obs084_input_manifest.csv`;
- `obs084_discovery_candidate_manifest.csv`;
- `obs084_candidate_support_index.csv`;
- `obs084_confirmation_protocol.json`;
- `obs084_reserved_partition_manifest.csv`;
- `obs084_candidate_changes_after_freeze.csv`.

Matching and controls:

- `obs084_site_complement_admissibility.csv`;
- `obs084_control_record_admissibility.csv`;
- `obs084_match_balance_audit.csv`;
- `obs084_exposure_normalization_audit.csv`.

Confirmation:

- `obs084_predicate_specific_contrasts.csv`;
- `obs084_control_adjusted_contrasts.csv`;
- `obs084_dependence_aware_uncertainty.csv`;
- `obs084_missingness_reason_audit.csv`;
- `obs084_confirmation_results.csv`.

Witness layer:

- `obs084_witness_manifest.csv`;
- `obs084_direct_witnesses.csv`;
- `obs084_non_dominated_support_families.csv`;
- `obs084_support_overlap_and_equivalence.csv`;
- `obs084_replication_audit.csv`;
- `obs084_witness_version_history.csv`;
- `obs084_report.md`.

No FL assignment should automatically alter the OBS-082 readiness class. Any promotion requires a separate audit.

## 27. Repair-Hypothesis Boundary

A future repair hypothesis may be represented as:

\[
H_\rho=(w,\rho,M_\rho,\mathcal C_\rho,\Phi,\Omega).
\]

It must reference an exact witness identity.

OBS-084 should not construct or execute the repair.

## 28. Canonical Claim Templates

### FL1

> Matched-control evidence suggests that degradation may be concentrated in the declared address family, but the current evidence remains contrast-derived and does not establish a direct artifact support.

### FL2

> The record has a retrievable candidate failure support, but matching, control adjustment, uncertainty, or confirmation evidence remains incomplete.

### FL3

> Within the declared PAM artifact lineage, contract, and provenance scope, degradation of the specified registry record under the declared predicate is directly supported at the named minimal sufficient artifact support relative to its admissible complement and control records.

### FL4

> The direct failure support survives dependence-aware resampling and recurs across the declared held-out structural partitions.

### FL5

> The replicated failure support remains identifiable across scientifically reasonable variations within its declared contract neighborhood while preserving the expected scope and direction of degradation.

### Required guardrail

> This witness identifies where degradation is empirically addressable. It does not establish causal origin, repairability, intervention-readiness, external generalization, or formal topology.

## 29. Scope Discipline

All claims remain conditional on the current PAM artifact lineage, model and corpus provenance, C/Cp2/Cp3 comparisons where applicable, OBS-078–083 evidence lineage, tested relation and carrier families, declared matching and control procedures, committed artifacts, and confirmation partitions.

The term *direct* must never imply direct access to cognition, human phenomenology, hidden-state mechanisms, model-independent structure, corpus-independent structure, universal invariance, formal topology, causal control, or successful repair.

## 30. Revised Conceptual Hierarchy

- **Structural address:** inspectable part or conjunction within the artifact system.
- **Candidate failure support:** discovery evidence suggests predicate-specific degradation.
- **Frozen candidate support:** support and protocol fixed before reserved evidence.
- **Direct failure support:** confirmed through the full witness protocol.
- **Replicated failure support:** survives resampling and held-out partitions.
- **Contract-neighborhood-stable support:** remains identifiable within its declared contract neighborhood.
- **Repair target:** replicated support attached to a separate falsifiable repair hypothesis.

No transition is implicit.

## 31. Final Position

Theory Memo 009 v2 asks:

> What precisely would make PAM entitled to say that a declared property of a registered relation degrades at this artifact support?

The answer is not a high localization score.

It is a versioned, retrievable, predicate-specific, admissibly matched, control-adjusted, dependence-aware, provenance-preserved witness confirmed on reserved evidence.

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

Only after that sequence should RIG attempt to transform repair annotations into repair hypotheses.

## Canonical One-Sentence Summary

> Direct failure-site evidence in RIG is a versioned, retrievable witness showing that a declared property of a specific registry record degrades at a minimal sufficient artifact support relative to admissible complements and controls, under dependence-aware uncertainty, preserved provenance, and reserved confirmation evidence; it establishes diagnostic addressability, not causal origin, repairability, or intervention-readiness.
