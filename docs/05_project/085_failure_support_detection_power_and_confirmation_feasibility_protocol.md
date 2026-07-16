# OBS-085 — Failure-Support Detection Power and Confirmation Feasibility Protocol

**VERSION:** 2.0  
**DATE:** 2026-07-16  
**STATUS:** Pre-implementation design protocol  
**SCOPE:** Diagnostic-only instrument metrology  
**PREDECESSOR:** OBS-084 fully blinded artifact-direct failure-support study  
**PRIMARY EVIDENCE SPINE:** OBS-078 through OBS-084  
**CLAIM CEILING:** Conditional instrument sensitivity, feasibility, and operating-envelope characterization only

---

## 1. Purpose

OBS-085 characterizes the operating envelope of the PAM/RIG artifact-direct failure-support instrument established in OBS-084.

OBS-084 completed a fully separated discovery–confirmation study:

- OBS-084a froze the scientific observation identity, carrier definitions, field roles, support vocabulary, discovery–confirmation partition, seam discretization, source identities, and provenance chain.
- OBS-084b evaluated 5,736 predicate-indexed discovery tests and sealed thirteen non-dominated FL2 candidates.
- OBS-084c evaluated those thirteen candidates once on the reserved confirmation partition and established zero FL3 direct witnesses.

The null FL3 result does not, by itself, distinguish among several possibilities:

- no qualifying artifact-direct witness was present in the tested evidence space;
- qualifying effects were smaller than the instrument could recover;
- support or complement coverage was structurally inadequate;
- too few independent object clusters were available;
- object-level heterogeneity obscured a stable contrast;
- control adjustment removed non-specific signal;
- multiplicity exceeded the evidential strength of otherwise positive contrasts;
- discovery-stage selection inflated candidate effects;
- effects did not transport from discovery to confirmation;
- or prior claim-entitlement rules capped otherwise reproducible evidence.

OBS-085 therefore asks a separate metrological question:

> What kinds of artifact-direct failure support was the frozen OBS-084 instrument capable of structurally estimating, statistically recovering, discovering, sealing, and confirming?

OBS-085 does not reinterpret the OBS-084 result. It studies the instrument that produced it.

---

## 2. Core Question

> Under declared simulation-generating assumptions, what artifact-direct failure supports could the frozen OBS-084 instrument structurally estimate, statistically recover, discover, seal, and confirm?

This question is conditional on:

- the frozen OBS-084 evidence hierarchy;
- the frozen scientific observation and clustering units;
- the frozen support vocabulary;
- the frozen discovery and confirmation procedures;
- the observed object-level dependence structure;
- and a predeclared family of simulation-generating models.

OBS-085 does not estimate a model-independent probability that the PAM/RIG instrument would detect any possible witness.

---

## 3. Primary Objectives

OBS-085 has four primary analytical objectives.

### 3.1 Evidence feasibility

Determine which frozen record–support addresses possess the empirical structure required for valid statistical evaluation.

This includes:

- support availability;
- complement availability;
- scientific-observation coverage;
- independent object-cluster coverage;
- class-bearing cluster coverage;
- matched support/complement availability;
- control availability;
- target-control joint estimability;
- outcome estimability;
- and defined multiplicity-family membership.

### 3.2 Conditional confirmation sensitivity

Estimate the probability that a fixed, preregistered support address would satisfy the frozen OBS-084c statistical confirmation gates under specified:

- true-effect scenarios;
- object-cluster counts;
- support-coverage regimes;
- control-response regimes;
- multiplicity families;
- and simulation-generating models.

This analysis is conditional on the address already having been selected for confirmation.

### 3.3 End-to-end campaign sensitivity

Estimate the probability that a true synthetic support address would:

1. generate sufficient evidence during discovery;
2. survive the frozen OBS-084b search procedure;
3. survive multiplicity and non-dominance rules;
4. be sealed as an FL2 candidate;
5. remain admissible in an independent confirmation partition;
6. satisfy the frozen OBS-084c statistical gates;
7. and remain within its deterministic claim-entitlement ceiling.

This is the full discovery-to-confirmation operating envelope.

### 3.4 Instrument-model uncertainty

Determine how strongly the estimated operating characteristics depend on the declared simulation-generating model.

OBS-085 must distinguish:

- Monte Carlo uncertainty within a simulator;
- simulator-family dispersion across scientifically plausible models;
- and leave-one-object-out sensitivity to the small observed cluster set.

---

## 4. Non-Goals

OBS-085 must not:

- reopen the OBS-084 confirmation partition as a new empirical search;
- rerun OBS-084 discovery to seek additional empirical candidates;
- alter the thirteen sealed OBS-084 candidates;
- modify frozen support predicates;
- refit or replace frozen support thresholds;
- introduce new predicates based on observed OBS-084 outcomes;
- promote any OBS-084 record from FL2 to FL3;
- compute or report observed power from the realized OBS-084 effect estimate, test statistic, confidence interval, or p-value;
- infer that an artifact-direct witness exists because a synthetic effect is detectable;
- infer that no witness exists because a synthetic effect is detectable with high probability;
- infer that a hidden witness probably exists because estimated sensitivity is low;
- identify causal origins;
- identify repair targets;
- establish intervention readiness;
- claim actionability;
- generalize beyond the frozen PAM/RIG evidence spine;
- or claim formal topology.

Synthetic effect injection is an instrument test. It is not empirical evidence that the injected effect exists.

---

## 5. Canonical Terminology

### 5.1 Simulated gate-passage probability

The canonical sensitivity quantity is:

```text
simulated gate-passage probability
```

The term `power` may be used as shorthand within scripts or tables, but primary reporting must identify the quantity as simulation based and conditional on a declared generating model.

### 5.2 Statistical gate-passage probability

For support address \(a=(r,q,z)\), scenario \(s\), predicate-specific injection operator \(\mathcal I_q(\delta)\), and simulator family member \(h\):

\[
\pi_{\mathrm{stat}}(a,s;h)
=
P_h\!\left(
G_{\mathrm{stat}}=1
\mid
\mathcal I_q(\delta),s
\right).
\]

This is the simulated probability that the frozen statistical confirmation gates pass.

### 5.3 Campaign gate-passage probability

For the full discovery–confirmation campaign:

\[
\pi_{\mathrm{campaign}}(a,s;h)
=
P_h\!\left(
\begin{array}{l}
\text{address passes discovery}\\
\land\ \text{address is sealed}\\
\land\ \text{confirmation is admissible}\\
\land\ \text{statistical gates pass}
\end{array}
\right).
\]

### 5.4 Conditional operating envelope

The canonical result is the set:

\[
\left\{
\pi(a,s;h):
h\in\mathcal H
\right\},
\]

where \(\mathcal H\) is the frozen simulator family.

A summary range across \(h\in\mathcal H\) is an operating envelope. It is not a frequentist confidence interval over an empirically known superpopulation.

### 5.5 Minimum tested injected contrast

The term:

```text
minimum detectable contrast
```

must not be used as the canonical label.

The canonical quantity is:

```text
minimum tested injected contrast reaching the declared gate-passage probability
```

This wording makes clear that the result is:

- conditional on the tested effect grid;
- conditional on the simulator;
- conditional on the frozen decision procedure;
- and not a model-independent property of the experiment.

---

## 6. Two Distinct Forms of Uncertainty

OBS-085 must report two different uncertainty classes.

### 6.1 Monte Carlo uncertainty

Monte Carlo uncertainty measures how precisely a chosen simulation procedure was executed.

It depends primarily on:

- the number of simulation replicates;
- the estimated gate-passage probability;
- and the Monte Carlo interval procedure.

Increasing the number of simulation replicates reduces Monte Carlo uncertainty.

### 6.2 Simulator or design-model uncertainty

Simulator uncertainty measures how strongly the estimated operating characteristics change under scientifically plausible generating assumptions.

It includes uncertainty about:

- between-object heterogeneity;
- residual distributions;
- tail behavior;
- leverage;
- support prevalence;
- target-control covariance;
- missingness;
- variance pooling;
- class composition;
- and the exchangeability of hypothetical future objects.

Increasing the number of simulation replicates does not eliminate simulator uncertainty.

The following statement is mandatory in all OBS-085 reports:

> Monte Carlo precision is not instrument-model certainty. Sensitivity estimates remain conditional on the declared simulation-generating assumptions, particularly given the small number of observed independent object clusters.

---

## 7. Frozen Inputs

OBS-085 treats the completed OBS-084 study as immutable.

### 7.1 OBS-084a inputs

Required frozen inputs include:

- scientific observation-key specification;
- carrier-feature manifest;
- field-role classification;
- object-cluster selection;
- discovery–confirmation partition manifest;
- partition-balance audit;
- support-vocabulary freeze;
- seam-discretization protocol;
- canonical source manifest;
- source-hash manifest;
- bridge-resolution summary;
- and freeze manifest.

### 7.2 OBS-084b inputs

Required discovery inputs include:

- discovery observation losses;
- support-vocabulary execution audit;
- support candidate inventory;
- support/complement matching;
- site-relative contrasts;
- control-adjusted contrasts;
- cluster uncertainty;
- multiplicity audit;
- minimal support families;
- candidate freeze manifests;
- discovery failures;
- discovery summary;
- and discovery report.

### 7.3 OBS-084c inputs

Required confirmation inputs include:

- confirmation observation losses;
- candidate-manifest validation;
- support/complement validation;
- confirmation site contrasts;
- control adjustment;
- cluster uncertainty;
- multiplicity audit;
- candidate outcomes;
- direct-witness registry;
- confirmation failures;
- confirmation opening lock;
- confirmation manifest;
- confirmation summary;
- and confirmation report.

### 7.4 Identity preservation

Before analysis, OBS-085 must verify:

- the OBS-084a freeze identity;
- the OBS-084b candidate-manifest identity;
- the OBS-084c opening-lock identity;
- the OBS-084c confirmation identity;
- all required source hashes;
- and the exact frozen script identities required by the OBS-084 validation chain.

OBS-085 must not regenerate any OBS-084 artifact.

A validation failure is a stop condition.

Historical absolute execution paths retained inside sealed OBS-084 artifacts are part of the completed provenance record and must not be normalized or rewritten by OBS-085.

---

## 8. Units of Analysis

OBS-085 distinguishes five analytical units.

### 8.1 Scientific observation

The frozen scientific observation identity is:

```text
case
+ object
+ cohort
+ scale_index_from
+ scale_index_to
```

### 8.2 Independent cluster

The independent clustering unit is:

```text
object
```

Observation rows belonging to the same object must not be treated as mutually independent.

### 8.3 Registry record

The registry-record unit is:

```text
relation × carrier
```

There are twenty-four frozen OBS-081 registry records.

### 8.4 Support address

A support address is the complete frozen location at which a support claim may be evaluated.

It includes, where applicable:

```text
registry record
+ diagnostic predicate
+ support family
+ support predicate
+ frozen threshold or category
+ complement definition
+ control contract
```

Let:

\[
a=(r,q,z)
\]

denote a support address, where:

- \(r\) is the registry record;
- \(q\) is the failure predicate;
- \(z\) is the frozen support specification and associated comparison contract.

### 8.5 Candidate

A candidate is a support address sealed by OBS-084b.

The thirteen OBS-084b candidates form a fixed candidate family.

They must remain distinct from the complete frozen support-address universe.

---

## 9. Evidence Feasibility and Claim Entitlement

Claim entitlement is not a structural feasibility gate.

Define:

\[
\operatorname{EvidenceFeasible}(a)
\]

as the deterministic property that address \(a\) has the support, complement, cluster, class, matching, control, and outcome structure required for statistical evaluation.

Define:

\[
E(r)
=
\mathbf 1\{
\operatorname{ClaimEntitled}(r)
\}
\]

as the deterministic claim-entitlement ceiling for registry record \(r\).

For a fixed record:

\[
\pi_{\mathrm{FL3}}(a,s;h)
=
E(r)\,
\pi_{\mathrm{stat}}(a,s;h).
\]

Entitlement is not simulated as a random event.

OBS-085 must separately report whether an address is:

- evidence-feasible;
- statistically detectable;
- campaign-detectable;
- FL3-entitled;
- or statistically detectable but entitlement-capped.

An address must not be described as structurally unconfirmable solely because its registry record is not entitled to an FL3 claim.

---

## 10. Analytical Architecture

OBS-085 is divided into five operational stages.

### OBS-085a — Structural Evidence Feasibility

A deterministic audit of:

- support coverage;
- complement admissibility;
- independent-cluster availability;
- class coverage;
- matching;
- joint target-control estimability;
- outcome estimability;
- multiplicity-family membership;
- and deterministic claim entitlement.

No synthetic outcomes are generated.

### OBS-085b0 — Injection and Simulator Qualification

A pre-simulation qualification stage that freezes and validates:

- predicate-specific injection operators;
- simulator families;
- scenario definitions;
- range preservation;
- target-control covariance behavior;
- object influence;
- null behavior at the estimator level;
- and Monte Carlo execution rules.

No canonical sensitivity estimate may be produced before OBS-085b0 passes.

### OBS-085b — Conditional Gate-Passage Sensitivity

A simulation study estimating confirmation-gate passage for fixed OBS-084 candidates under frozen effect, cluster-count, coverage, control, multiplicity, and simulator scenarios.

No candidate search or candidate ranking is performed.

### OBS-085c — Joint Campaign Sensitivity

A full correlated campaign simulation applying the frozen OBS-084b discovery and OBS-084c confirmation procedures to independently generated discovery and confirmation partitions.

This stage preserves:

- all 5,736 discovery tests;
- candidate competition;
- multiplicity;
- non-dominance;
- candidate-family limits;
- and confirmation selection.

### OBS-085d — Operating-Envelope Synthesis

A final synthesis of:

- structural feasibility;
- simulator-conditional gate passage;
- end-to-end campaign behavior;
- model uncertainty;
- selection bias;
- null calibration;
- object-count extrapolations;
- and entitlement ceilings.

---

# OBS-085a — Structural Evidence Feasibility

## 11. Structural Evidence Gates

Each frozen support address must be evaluated against the following deterministic evidence gates.

### G1 — Support presence

At least one eligible scientific observation satisfies the frozen support predicate.

### G2 — Complement presence

At least one eligible scientific observation satisfies the frozen complement definition.

### G3 — Support-cluster coverage

The support occurs in the minimum number of independent object clusters required by the frozen OBS-084 contract.

### G4 — Complement-cluster coverage

The complement occurs in the minimum number of independent object clusters required by the frozen OBS-084 contract.

### G5 — Class-bearing coverage

Support and complement contain the class structure required by the diagnostic predicate.

### G6 — Matched-complement admissibility

The frozen matching procedure produces an admissible support–complement comparison.

### G7 — Control availability

The required relation, carrier, or combined controls are present.

### G8 — Joint target-control estimability

The target and required controls can be jointly estimated at the object-cluster level.

This includes sufficient overlap to estimate their dependence structure.

### G9 — Outcome estimability

The target diagnostic loss, attenuation, breach, reversal, collapse, uncertainty, reproducibility, or missingness quantity is finite and estimable.

### G10 — Multiplicity-family definition

The address belongs to a defined multiplicity family for the analysis in which it is evaluated.

### E1 — Claim entitlement

The frozen OBS-083/084 claim hierarchy permits the registry record to progress to FL3 if all statistical gates are satisfied.

`E1` is recorded separately and is not part of `EvidenceFeasible`.

---

## 12. Effective Evidence Measures

For each support address, OBS-085a must record:

- total observation rows;
- unique scientific observations;
- total object clusters;
- support observations;
- complement observations;
- support-bearing object clusters;
- complement-bearing object clusters;
- clusters containing both support and complement;
- class-bearing support clusters;
- class-bearing complement clusters;
- matched support observations;
- matched complement observations;
- matched clusters;
- usable target observations;
- usable target clusters;
- usable relation-control observations and clusters;
- usable carrier-control observations and clusters;
- usable combined-control observations and clusters;
- target-control jointly estimable clusters;
- missingness rate;
- support prevalence;
- complement prevalence;
- and object-level leverage diagnostics where calculable.

The primary effective-evidence quantities are cluster based.

Observation-row counts are descriptive and must not be presented as independent sample sizes.

---

## 13. Structural Evidence Classes

Each address receives a complete gate vector.

A non-canonical summary class may also be assigned from:

```text
evidence_feasible
support_absent
complement_absent
support_cluster_limited
complement_cluster_limited
class_coverage_limited
matching_limited
control_limited
joint_estimability_limited
outcome_not_estimable
multiplicity_undefined
multiple_structural_limits
```

Claim entitlement is reported separately as:

```text
fl3_entitled
fl3_entitlement_capped
```

A support address may fail more than one evidence gate.

The complete gate vector is canonical. Any single summary label is secondary.

---

## 14. OBS-085a Success Criterion

OBS-085a succeeds if it produces a deterministic, fully auditable map of:

- the complete frozen support-address universe;
- all evidence-feasibility gates;
- effective cluster-level evidence;
- target-control joint estimability;
- structural bottlenecks;
- and the separate claim-entitlement overlay.

OBS-085a does not estimate gate-passage probability.

---

# OBS-085b0 — Injection and Simulator Qualification

## 15. Purpose

OBS-085b0 qualifies the machinery used by OBS-085b and OBS-085c.

Because the observed design contains only four independent object clusters per partition, no simulator should produce canonical sensitivity estimates until its:

- artifact semantics;
- injection behavior;
- dependence preservation;
- range preservation;
- null behavior;
- object influence;
- and scenario boundaries

have been audited and frozen.

---

## 16. Predicate-Specific Injection Contract

Synthetic effect injection must be indexed by the frozen failure predicate:

\[
\mathcal I_q(\delta).
\]

A universal additive injection rule is prohibited.

For every predicate \(q\), the injection contract must freeze:

- the failure-predicate identity;
- the lowest valid artifact level for injection;
- the mathematical scale of injection;
- the range-preserving transform;
- the target field or fields modified;
- the fields that must remain unchanged;
- the control fields modified;
- the target-control response relationship;
- the missingness behavior;
- the discovery–confirmation transport behavior;
- the estimator rerun after injection;
- the inverse transform, if any;
- invalid parameter combinations;
- and the interpretation ceiling of the resulting simulation.

### 16.1 Continuous diagnostic losses

An additive or standardized shift may be permissible when:

- the native scale is unbounded;
- or a suitable transformed scale makes the injection range valid.

The frozen estimator must be rerun after injection.

### 16.2 Bounded probabilities

Injection must occur on a range-preserving scale, such as:

- logit;
- probit;
- or another predeclared bounded-response transform.

Direct additive shifts outside the valid range are prohibited.

### 16.3 Binary outcomes

Injection requires a declared probability model, such as:

- Bernoulli-logistic;
- beta-binomial;
- logistic-hierarchical;
- or another justified object-aware binary model.

### 16.4 Attenuation predicates

Injection must alter the magnitude of the relevant signal relative to its frozen baseline.

Adding a constant to the final attenuation statistic is insufficient unless the study is explicitly limited to gate-logic behavior.

### 16.5 Threshold-breach predicates

Injection must operate on the quantity compared with the frozen threshold.

The threshold itself must remain unchanged.

### 16.6 Sign-reversal predicates

Injection must modify the direction-generating artifact or contrast so that the frozen estimator can recover a reversal.

Directly flipping the final sign label is prohibited.

### 16.7 Contrast-collapse predicates

Target and control quantities must be modified jointly.

The simulation must preserve the meaning of collapse relative to the declared comparison.

### 16.8 Uncertainty predicates

An uncertainty failure requires a variance, heterogeneity, leverage, or missing-information injection.

A mean shift alone is not an adequate uncertainty injection.

### 16.9 Reproducibility-loss predicates

Discovery and confirmation effects require a declared transport model.

Possible frozen models may include:

- stable transport;
- attenuated transport;
- object-varying transport;
- sign-unstable transport;
- or partition-specific heterogeneity.

### 16.10 Missingness predicates

Injection requires an explicit missingness mechanism.

Where relevant, the contract must distinguish:

- missing completely at random;
- conditionally missing;
- support-concentrated missingness;
- class-concentrated missingness;
- and object-concentrated missingness.

### 16.11 Contrast-level approximation

When injection is possible only at an already-computed contrast level, the output must state:

> This result characterizes the frozen gate logic conditional on the existing estimator. It does not characterize the complete artifact-generation and estimation instrument.

---

## 17. Joint Target-Control Simulation

Targets and controls must be simulated jointly.

For object cluster \(j\), the primary simulation object is:

\[
\left(
D_{\mathrm{target},j},
D_{\mathrm{relation},j},
D_{\mathrm{carrier},j},
D_{\mathrm{combined},j}
\right),
\]

where those components exist.

The simulator must preserve or explicitly model:

- object-level covariance;
- target-control dependence;
- unequal object leverage;
- shared missingness;
- support/complement incidence;
- class composition;
- matched-comparison structure;
- and joint estimator availability.

Independent target and control resampling is prohibited as the primary method.

---

## 18. Control-Response Grid

Let \(\lambda\) describe the amount of the synthetic target effect also expressed in the relevant control.

The primary grid is:

\[
\lambda\in
\{
0.00,\ 0.25,\ 0.50,\ 1.00
\}.
\]

Interpretation:

```text
λ = 0.00   target-specific effect
λ = 0.25   weak shared effect
λ = 0.50   partially generic effect
λ = 1.00   fully control-explained effect
```

A secondary adverse-control grid may include:

\[
\lambda\in
\{
-0.50,\ -0.25,\ 1.25,\ 1.50
\}.
\]

These scenarios test:

- counter-directional control behavior;
- suppressor-like control behavior;
- and control response exceeding the target response.

The adverse grid is a robustness analysis and must not replace the primary grid.

---

## 19. Baseline Construction

For each evidence-feasible address:

1. reconstruct the frozen support and complement assignment;
2. reconstruct the frozen matched comparison;
3. compute the required target object-level contribution;
4. compute all required control object-level contributions;
5. form the joint target-control object vector;
6. remove the observed mean target and control contrasts;
7. retain the centered object-level dependence structure;
8. apply the predicate-specific injection operator;
9. rerun the frozen estimator;
10. apply the frozen decision gates.

Centering removes the realized empirical effect before sensitivity estimation.

OBS-085 must not treat the realized OBS-084 effect estimate as the true effect in a post hoc power calculation.

---

## 20. Standardized Effect Grid

Where a standardized effect scale is valid, the primary frozen grid is:

```text
0.00
0.25
0.50
0.75
1.00
1.50
2.00
```

The standardized effect is defined relative to a frozen object-cluster residual scale.

The exact scale estimator must be part of the simulator specification.

The zero-effect scenario is mandatory for null calibration.

Predicate-specific native-scale grids may supplement or replace the standardized grid where standardization would distort the failure semantics.

No effect grid may be altered after simulation results are inspected.

---

## 21. Simulator Family

The outer simulator family \(\mathcal H\) must be frozen before primary execution.

It must contain a computationally bounded set of complete simulator specifications spanning scientifically plausible assumptions.

### 21.1 Residual generation

The family should include, where valid:

- wild-cluster residual resampling;
- object-level residual bootstrap;
- leave-one-object-out residual construction;
- and hierarchical parametric residual generation.

### 21.2 Scale estimation

The family should include:

- ordinary object-cluster scale;
- robust object-cluster scale;
- pooled predicate-level scale;
- address-specific scale where estimable;
- and a conservative upper-scale scenario.

### 21.3 Between-object heterogeneity

The family should include:

- empirical heterogeneity;
- reduced heterogeneity;
- conservatively inflated heterogeneity;
- and a heavy-tail or high-leverage scenario.

### 21.4 Support prevalence

The family should include:

- empirical support incidence;
- leave-one-object-out support incidence;
- minimum-admissible incidence;
- balanced hypothetical incidence;
- and conservative rare-support incidence.

### 21.5 Target-control covariance

The family should include:

- empirical joint covariance;
- regularized covariance;
- conservative high-correlation covariance;
- and weak-correlation covariance.

### 21.6 Missingness

The family should include, where predicate-valid:

- empirical frozen missingness;
- conditionally increased missingness;
- support-concentrated missingness;
- and a conservative complete-case loss scenario.

Not every Cartesian combination must be executed.

The simulator-family manifest must define complete, named model specifications rather than allowing post hoc mixing of favorable assumptions.

---

## 22. Four-Cluster Limitation

The observed design contains four independent objects per partition.

This is enough to define the frozen empirical study.

It is not a strong empirical basis for estimating:

- the full residual distribution;
- between-object variance;
- tail behavior;
- leverage;
- target-control covariance;
- support-prevalence variation;
- or a general future-object distribution.

OBS-085 must therefore treat the current-design simulation as a conditional reconstruction of the frozen instrument.

It must not present smooth gate-passage curves as though four clusters identify a precise general sampling distribution.

Leave-one-object-out qualification must report how strongly results depend on each observed object.

---

## 23. Simulator Qualification Audits

Before OBS-085b begins, each primary simulator must pass the following audits.

### Q1 — Range preservation

Injected artifacts remain within their valid mathematical ranges.

### Q2 — Predicate validity

The injected change represents the declared failure predicate.

### Q3 — Frozen-estimator compatibility

The original frozen estimator can be rerun without modification.

### Q4 — Joint covariance preservation

The simulator preserves or validly models the target-control dependence structure.

### Q5 — Cluster integrity

Object-level dependence is preserved.

Observation rows are not treated as independent clusters.

### Q6 — Support/complement integrity

The simulator preserves or explicitly models support/complement incidence according to the frozen scenario.

### Q7 — Missingness integrity

Missingness follows the declared mechanism and does not silently alter unrelated fields.

### Q8 — Null estimator calibration

At \(\delta=0\), the estimator behaves within declared calibration tolerances before campaign selection is introduced.

### Q9 — Object influence

Leave-one-object-out analyses identify whether one object dominates the simulated gate-passage behavior.

### Q10 — Reproducibility

Repeated runs with the same seed and manifest reproduce identical outputs.

Failure of a primary simulator on any required qualification gate stops canonical sensitivity execution for that simulator.

---

## 24. OBS-085b0 Success Criterion

OBS-085b0 succeeds if it freezes a validated set of:

- predicate-specific injection contracts;
- complete simulator specifications;
- scenario definitions;
- range-preservation rules;
- object-level dependence rules;
- target-control covariance rules;
- null-calibration tolerances;
- object-influence diagnostics;
- and deterministic random-seed controls.

No canonical gate-passage probability may be reported before OBS-085b0 passes.

---

# OBS-085b — Conditional Confirmation Sensitivity

## 25. Primary Estimand

For a fixed support address and simulation scenario, the primary estimand is:

> The probability that the address satisfies the frozen OBS-084c statistical confirmation gates when the declared predicate-specific synthetic effect has magnitude \(\delta\).

This estimand is conditional on the address already having been selected for confirmation.

OBS-085b does not simulate candidate discovery or candidate competition.

---

## 26. Statistical and Entitlement Outputs

OBS-085b reports:

### Statistical gate-passage probability

\[
\pi_{\mathrm{stat}}(a,s;h).
\]

### Entitlement-adjusted FL3 probability

\[
\pi_{\mathrm{FL3}}(a,s;h)
=
E(r)\,
\pi_{\mathrm{stat}}(a,s;h).
\]

Because \(E(r)\) is deterministic, it is reported as an overlay.

The study must not report:

\[
P(
\text{FL3 entitlement}
\mid
\text{statistical confirmation}
)
\]

as though entitlement were a simulated event.

---

## 27. Confirmation Gate Contract

Each replicate must apply the exact frozen OBS-084c decision contract.

At minimum, the audit must retain whether the replicate passes:

```text
support available
complement admissible
effect direction reproduced
target contrast positive
cluster-aware uncertainty requirement
raw statistical threshold
multiplicity-adjusted threshold
control-adjusted contrast requirement
control-specificity requirement
```

The exact gate names, threshold values, correction method, and ordering must be extracted from the validated OBS-084c implementation and frozen in the OBS-085 input manifest.

OBS-085 must not introduce a more favorable confirmation rule.

Claim entitlement is applied only after the statistical gate vector is complete.

---

## 28. Object-Count Scenarios

The canonical current design contains:

```text
4 objects per partition
```

Hypothetical scenarios may include:

```text
6 objects per partition
8 objects per partition
12 objects per partition
```

These correspond to full discovery–confirmation campaign sizes of:

```text
8 independent objects
12 independent objects
16 independent objects
24 independent objects
```

Scenarios with more than four objects per partition require the explicit assumption:

> Hypothetical future object clusters are exchangeable draws from the same scoped object-generating population represented by the frozen OBS-084 objects, under the declared simulator.

This assumption is strong.

All larger object-count scenarios must be labeled:

```text
hypothetical design extrapolation
```

They must not be described as empirical forecasts.

Duplicating observed objects or observation rows while treating them as new independent evidence is prohibited.

---

## 29. Support-Coverage Scenarios

Sensitivity may be evaluated under:

### Empirical coverage

The observed support/complement incidence pattern is preserved.

### Leave-one-object-out coverage

Support prevalence is estimated after withholding each object in turn.

### Minimum-admissible coverage

Support and complement satisfy only the minimum frozen cluster requirements.

### Balanced coverage

Support and complement are approximately balanced across the available clusters.

### Rare-support coverage

Support prevalence is reduced under a frozen conservative rule.

Only empirical coverage is canonical for the current design.

The remaining coverage regimes are conditional design scenarios.

---

## 30. Multiplicity Scenarios

The following multiplicity families may be evaluated:

```text
M1      one preregistered primary address
M13     the thirteen sealed OBS-084 candidates
M24     one primary address per registry record
M5736   the complete OBS-084b predicate-indexed discovery family
```

`M13` is the canonical OBS-084 confirmation burden.

`M1` and `M24` are design comparisons.

`M5736` is an exploratory upper-bound diagnostic and must not be described as the actual OBS-084c confirmation correction.

The same correction method used in OBS-084 must be retained.

---

## 31. Minimum Tested Injected Contrast

For each estimable address, scenario, and simulator family, OBS-085b must report:

- gate-passage probability at every tested effect size;
- Monte Carlo interval;
- simulator-family range;
- deterministic entitlement status;
- and the smallest tested injected contrast reaching:

```text
50% gate-passage probability
80% gate-passage probability
90% gate-passage probability
```

If the target probability is not reached within the frozen grid, report:

```text
greater_than_tested_range
```

No extrapolation beyond the tested grid is canonical.

---

## 32. Conditional Gate-Failure Reporting

Each replicate must retain the complete gate-failure vector.

For every scenario, OBS-085b must report:

- marginal failure rate for each gate;
- joint gate-failure combinations;
- ordered cumulative passage through the frozen protocol;
- number of failed gates per replicate;
- simulator-family dispersion;
- and Monte Carlo uncertainty.

No replicate receives a canonical exclusive failure cause.

---

## 33. Gate-Relaxation Diagnostic

An optional decision-procedure diagnostic is:

\[
L_j
=
P(
\text{success with gate }j\text{ relaxed}
)
-
P(
\text{success under the frozen protocol}
).
\]

This measures the simulated increase in success probability when one decision gate is hypothetically relaxed while all other components remain unchanged.

It does not estimate a causal effect on the PAM phenomenon.

It must not be used to recommend relaxing the real evidential standard.

---

## 34. OBS-085b Success Criterion

OBS-085b succeeds if it produces model-conditional gate-passage envelopes for the fixed OBS-084 candidate family, with:

- predicate-valid injection;
- joint target-control simulation;
- object-cluster dependence;
- Monte Carlo uncertainty;
- simulator uncertainty;
- leave-one-object-out sensitivity;
- multiplicity comparisons;
- and deterministic entitlement overlays.

---

# OBS-085c — Joint Campaign Sensitivity

## 35. Campaign Estimand

For support address \(a\) containing a synthetic effect of magnitude \(\delta\), OBS-085c estimates the probability that the complete frozen campaign:

1. detects sufficient evidence during discovery;
2. retains the address after uncertainty and control gates;
3. retains the address after multiplicity correction;
4. retains the address after non-dominance and candidate-family rules;
5. seals the address as a candidate;
6. finds the address admissible in independent confirmation;
7. satisfies the frozen confirmation statistical gates;
8. and remains within its deterministic entitlement ceiling.

The stage probabilities must be reported separately:

```text
P(discovery threshold passed)
P(discovery statistical gates passed)
P(candidate sealed)
P(confirmation admissible | sealed)
P(confirmation statistical gates passed | sealed and admissible)
P(end-to-end statistical success)
deterministic entitlement overlay
P(end-to-end FL3-entitled success)
```

---

## 36. Full Correlated Campaign Requirement

Canonical end-to-end sensitivity requires simulation of the full correlated campaign.

Independent per-address simulation is insufficient because the 5,736 discovery tests share:

- object clusters;
- scientific observations;
- outcomes;
- carrier features;
- predicates;
- support memberships;
- controls;
- thresholds;
- missingness;
- and candidate-selection rules.

Each canonical campaign replicate must:

1. generate or resample a complete discovery object-level dataset;
2. generate an independent confirmation object-level dataset;
3. inject the target effect at the predicate-valid artifact level;
4. preserve the joint target-control structure;
5. execute all 5,736 frozen discovery tests jointly;
6. apply the actual discovery uncertainty and control gates;
7. apply the actual multiplicity procedure;
8. apply the frozen non-dominance rules;
9. apply the frozen candidate-family limits;
10. seal the resulting candidate set;
11. evaluate only the sealed candidate set on the independent confirmation realization;
12. apply the frozen confirmation admissibility gates;
13. apply the frozen confirmation uncertainty, control, and multiplicity gates;
14. and overlay deterministic claim entitlement.

The same synthetic object realization must drive all correlated addresses within a campaign replicate.

---

## 37. Discovery–Confirmation Independence

Each simulated campaign must maintain separate discovery and confirmation object clusters.

Within a replicate:

- no object realization may appear in both partitions;
- no residual draw may be shared across partitions unless explicitly required by a frozen superpopulation parameter;
- no outcome realization may appear in both partitions;
- no confirmation information may influence discovery threshold fitting;
- and no confirmation result may alter the sealed candidate family.

Threshold fitting and candidate selection occur only in simulated discovery.

Simulated confirmation applies the discovery-fitted definitions without additional search.

---

## 38. Candidate Competition

The injected address competes with the complete frozen discovery family.

OBS-085c must report whether it fails because it:

- does not generate sufficient discovery contrast;
- fails discovery uncertainty requirements;
- fails discovery controls;
- fails discovery multiplicity;
- is dominated by another address;
- is excluded by candidate-family limits;
- is not selected because a correlated alternative ranks higher;
- fails confirmation admissibility;
- fails confirmation signal;
- fails confirmation uncertainty;
- fails confirmation multiplicity;
- fails confirmation controls;
- or is statistically successful but entitlement-capped.

Candidate competition must not be omitted from campaign-level sensitivity.

---

## 39. Discovery Selection Bias

OBS-085c must quantify the winner’s-curse effect created by discovery selection.

For synthetic true contrast \(\delta\), report:

\[
\text{discovery inflation}
=
\widehat{\delta}_{\mathrm{discovery}}
-
\delta,
\]

\[
\text{confirmation bias}
=
\widehat{\delta}_{\mathrm{confirmation}}
-
\delta,
\]

and, where numerically stable:

\[
\text{confirmation transport ratio}
=
\frac{
\widehat{\delta}_{\mathrm{confirmation}}
}{
\widehat{\delta}_{\mathrm{discovery}}
}.
\]

The transport ratio must not be reported when the discovery denominator is zero or lies within a frozen near-zero stability band.

Also report:

- discovery-to-confirmation shrinkage;
- effect-sign transport;
- candidate-rank inflation;
- expected regression following selection;
- and gate-specific loss after candidate sealing.

This distinguishes expected selection behavior from simulator-defined cross-partition instability.

---

## 40. Null Calibration

The zero-effect campaign must report multiple error quantities.

### 40.1 Per-address error

\[
P(
\text{a specified null address passes}
).
\]

### 40.2 Candidate-sealing error

\[
P(
\text{a specified null address is sealed as FL2}
).
\]

### 40.3 Conditional confirmation error

\[
P(
\text{a null candidate passes confirmation}
\mid
\text{candidate was selected}
).
\]

### 40.4 Campaign-wise false-witness rate

\[
P(
N_{\mathrm{false\ FL3}}\geq 1
).
\]

This is the primary OBS-085c null-calibration quantity.

### 40.5 Expected false-witness count

\[
E[
N_{\mathrm{false\ FL3}}
].
\]

### 40.6 Expected false-candidate count

\[
E[
N_{\mathrm{false\ FL2\ sealed}}
].
\]

### 40.7 Null candidate-family size distribution

The full distribution of the number of sealed null candidates must be retained.

---

## 41. Mixed Campaign Calibration

A mixed campaign is required in which:

- one frozen address contains a real synthetic effect;
- all competing addresses remain null.

This scenario must report:

- probability the true address passes discovery;
- probability the true address is sealed;
- probability the true address reaches confirmation;
- probability the true address passes confirmation;
- probability a correlated null alternative outranks it;
- expected number of false alternatives sealed;
- probability of at least one false FL3 result;
- probability of a false FL3 instead of the true address;
- and probability of a false FL3 alongside the true address.

---

## 42. Bottleneck Reporting

No campaign replicate receives a canonical exclusive failure cause.

The canonical record is the complete gate-failure vector.

For every campaign scenario, report:

- marginal failure rate for every gate;
- joint failure combinations;
- ordered cumulative passage;
- number of failed gates;
- stage of first failure;
- stage of final exclusion;
- candidate-competition outcomes;
- and simulator-family dispersion.

A primary bottleneck label may be included only as a secondary summary convention.

Its ordering must be frozen before execution.

---

## 43. Approximate Campaign Simulation

If computational constraints require an approximation:

- the approximation must be declared before primary execution;
- it must preserve the full-family dependence structure as far as possible;
- it must be validated against a smaller exact campaign;
- its error relative to exact execution must be reported;
- and it must not support claims requiring exact candidate competition unless equivalence is established.

Independent per-address simulations may be used for screening or engineering diagnostics.

They may not be presented as canonical end-to-end campaign sensitivity.

---

## 44. OBS-085c Success Criterion

OBS-085c succeeds if it executes a validated, correlated discovery–confirmation campaign simulation that preserves:

- all frozen discovery addresses;
- object-level dependence;
- target-control covariance;
- candidate competition;
- multiplicity;
- non-dominance;
- candidate-family limits;
- confirmation independence;
- selection bias;
- null calibration;
- and deterministic entitlement ceilings.

---

# Cross-Study Design

## 45. Computational Hierarchy

The scenario design must use three computational tiers.

### Tier 1 — Canonical current-design analysis

Population:

- all thirteen sealed OBS-084 candidates.

Frozen settings:

- four objects per partition;
- empirical support coverage;
- `M13` multiplicity;
- primary target-control model;
- complete primary effect grid;
- and the complete primary simulator family.

This is the canonical OBS-085b analysis.

### Tier 2 — Hypothetical design extrapolation

Population:

- all thirteen candidates;
- or predefined candidate strata.

Settings may include:

- six, eight, and twelve objects per partition;
- balanced coverage;
- minimum-admissible coverage;
- rare-support coverage;
- alternate control-response values;
- `M1`;
- `M24`;
- and conservative heterogeneity models.

All Tier 2 results must be labeled hypothetical design extrapolations.

### Tier 3 — Full campaign stress tests

Population:

- a frozen set of representative target addresses.

Each replicate must still execute the complete 5,736-address discovery family.

Representative addresses must be selected before simulation to cover:

- relation types;
- carriers;
- failure predicates;
- support families;
- support prevalence;
- evidence-feasibility strata;
- entitlement classes;
- and observed OBS-084 outcome classes.

The selection algorithm and resulting target-address manifest must be frozen before simulation.

No target address may be added because early simulation results appear interesting.

---

## 46. Monte Carlo Precision

Each primary simulation cell begins with a frozen minimum number of replicates.

Recommended initial minimum:

```text
2,000 replicates per primary cell
```

Cells near a declared decision boundary may be extended to:

```text
10,000 replicates
```

Extension rules must be based on Monte Carlo uncertainty.

They must not depend on whether results are favorable.

For every probability estimate, report:

- number of replicates;
- number of successes;
- point estimate;
- Monte Carlo interval;
- random-seed identity;
- simulator identity;
- and scenario identity.

A small Monte Carlo interval must not be presented as evidence that simulator uncertainty is small.

---

## 47. Primary Reporting Dimensions

Results must be reportable by:

```text
registry record
diagnostic predicate
support family
support address
OBS-083 subclass
entitlement ceiling
partition
effect size
object-cluster count
coverage scenario
control-response scenario
multiplicity scenario
simulator family
simulation replicate count
```

A single global power or sensitivity estimate is prohibited as the primary result.

---

## 48. Detection-Envelope Summary Classes

Each address may receive a non-canonical summary class under the canonical current-design scenario:

```text
detectable_under_current_design
detectable_only_for_large_injected_effects
support_coverage_limited
complement_limited
cluster_count_limited
matching_limited
joint_estimability_limited
control_specificity_limited
multiplicity_limited
simulator_sensitive
entitlement_capped
multiple_limits
not_estimable
```

Recommended interpretation:

- `detectable_under_current_design`: reaches at least 80% statistical gate-passage probability within the frozen moderate-effect range under every required primary simulator, or under a separately frozen robust-envelope rule;
- `detectable_only_for_large_injected_effects`: reaches 80% only at a standardized injected effect of 1.0 or greater;
- `simulator_sensitive`: the classification changes materially across primary simulator families;
- limiting classes: does not reach the declared threshold primarily because of the named evidence or decision constraint;
- `entitlement_capped`: statistically detectable but not eligible for FL3;
- `not_estimable`: a valid simulation cannot be constructed from the frozen evidence structure.

The exact classification thresholds and simulator-envelope rule must be frozen before final execution.

Continuous operating envelopes remain canonical.

Classes are summaries only.

---

## 49. Negative Controls

Required negative controls include:

### 49.1 Null-effect calibration

At \(\delta=0\), estimate false-positive behavior at:

- estimator level;
- fixed-address confirmation level;
- candidate-sealing level;
- conditional confirmation level;
- and full campaign level.

### 49.2 Support-label permutation

Permute support labels only within frozen structural constraints and object clusters.

### 49.3 Relation-control injection

Inject the same effect into the target and relation control.

The frozen control adjustment should suppress generic signal according to its declared contract.

### 49.4 Carrier-control injection

Inject the same effect into the target and carrier control.

### 49.5 Combined-control injection

Where a combined control exists, inject the declared shared effect into target and combined control.

### 49.6 Partition-exchange robustness

Where structurally valid, exchange the discovery and confirmation object assignments as a robustness diagnostic.

This does not alter the canonical OBS-084 result.

### 49.7 Inadmissible-support control

Verify that structurally inadmissible addresses are never reported as statistically powered or campaign-detectable.

### 49.8 No-injection identity test

With zero injection and a fixed seed, the simulator must reproduce the declared centered baseline behavior without silently changing unrelated fields.

---

## 50. Permitted Sensitivity Analyses

Permitted sensitivity analyses include:

- alternative cluster-residual resampling schemes;
- pooled versus partition-specific residual scales;
- robust versus ordinary scale estimates;
- conservative versus empirical between-object heterogeneity;
- empirical versus regularized target-control covariance;
- fixed versus variable support prevalence;
- conservative versus empirical missingness;
- raw versus standardized effect scales;
- exclusion of high-leverage objects;
- alternative Monte Carlo replicate counts;
- adverse control-response scenarios;
- and frozen candidate-family size comparisons.

All sensitivity analyses must retain:

- the frozen support vocabulary;
- the frozen observation identity;
- the frozen cluster unit;
- the frozen discovery and confirmation estimators;
- the frozen gate definitions;
- and the deterministic entitlement rules.

---

## 51. Required Outputs

Recommended root directory:

```text
outputs/rig_registry/obs085_detection_envelope/
```

### 51.1 OBS-085a outputs

```text
obs085a_input_manifest.csv
obs085a_support_address_inventory.csv
obs085a_support_coverage_matrix.csv
obs085a_effective_evidence.csv
obs085a_complement_admissibility.csv
obs085a_control_availability.csv
obs085a_joint_target_control_estimability.csv
obs085a_structural_gate_matrix.csv
obs085a_evidence_feasibility.csv
obs085a_claim_entitlement_overlay.csv
obs085a_detection_envelope_summary.csv
obs085a_failures.csv
obs085a_report.md
```

### 51.2 OBS-085b0 outputs

```text
obs085b0_input_manifest.csv
obs085b0_predicate_injection_contract.csv
obs085b0_simulator_family_manifest.csv
obs085b0_scenario_freeze_manifest.csv
obs085b0_range_preservation_audit.csv
obs085b0_predicate_validity_audit.csv
obs085b0_joint_covariance_audit.csv
obs085b0_cluster_integrity_audit.csv
obs085b0_missingness_integrity_audit.csv
obs085b0_object_influence_audit.csv
obs085b0_null_estimator_calibration.csv
obs085b0_reproducibility_audit.csv
obs085b0_qualification_summary.csv
obs085b0_failures.csv
obs085b0_report.md
```

### 51.3 OBS-085b outputs

```text
obs085b_input_manifest.csv
obs085b_simulation_scenario_manifest.csv
obs085b_residual_scale_manifest.csv
obs085b_simulated_gate_passage.csv
obs085b_confirmation_gate_vectors.csv
obs085b_simulator_family_envelope.csv
obs085b_leave_one_object_out_envelopes.csv
obs085b_object_influence_on_gate_passage.csv
obs085b_minimum_tested_injected_contrast.csv
obs085b_joint_gate_failure_patterns.csv
obs085b_gate_relaxation_diagnostics.csv
obs085b_false_positive_calibration.csv
obs085b_monte_carlo_precision.csv
obs085b_summary.csv
obs085b_failures.csv
obs085b_report.md
```

### 51.4 OBS-085c outputs

```text
obs085c_input_manifest.csv
obs085c_campaign_scenario_manifest.csv
obs085c_representative_target_manifest.csv
obs085c_discovery_pass_rates.csv
obs085c_candidate_sealing_rates.csv
obs085c_confirmation_admissibility.csv
obs085c_confirmation_pass_rates.csv
obs085c_end_to_end_gate_passage.csv
obs085c_candidate_competition.csv
obs085c_discovery_effect_inflation.csv
obs085c_confirmation_transport.csv
obs085c_selection_bias_summary.csv
obs085c_campaign_wise_error.csv
obs085c_expected_false_witness_count.csv
obs085c_expected_false_candidate_count.csv
obs085c_mixed_campaign_results.csv
obs085c_joint_gate_failure_patterns.csv
obs085c_gate_relaxation_diagnostics.csv
obs085c_monte_carlo_precision.csv
obs085c_summary.csv
obs085c_failures.csv
obs085c_report.md
```

### 51.5 OBS-085d synthesis outputs

```text
obs085_detection_envelope_registry.csv
obs085_operating_envelope_summary.csv
obs085_simulator_uncertainty_summary.csv
obs085_claim_boundary_audit.csv
obs085_design_extrapolation_summary.csv
obs085_manifest.json
obs085_summary.csv
obs085_failures.csv
obs085_report.md
```

---

## 52. Provenance Requirements

Every OBS-085 stage must record:

- input artifact paths;
- input hashes;
- upstream OBS-084 identities;
- script path;
- script hash;
- repository commit;
- Python version;
- package versions;
- random seeds;
- simulator-family identity;
- scenario-manifest identity;
- execution timestamp;
- and output hashes.

All newly serialized repository paths should be repository-relative where possible.

Internal execution may use absolute paths.

Path representation must not be included in a scientific identity when a repository-relative identity is sufficient.

OBS-085 must not modify historical OBS-084 path representations.

---

## 53. Implementation Order

Implementation must proceed in the following order:

1. write and commit this protocol;
2. build the OBS-085 input-manifest validator;
3. verify all OBS-084 hashes and identities;
4. implement OBS-085a structural evidence feasibility only;
5. review and freeze the evidence-gate definitions;
6. freeze the separate claim-entitlement overlay;
7. implement predicate-specific injection contracts;
8. implement the simulator-family manifest;
9. implement OBS-085b0 qualification audits;
10. pass range, covariance, cluster, missingness, null, object-influence, and reproducibility qualification;
11. freeze the OBS-085b scenario manifest;
12. execute Tier 1 conditional gate-passage sensitivity;
13. execute permitted Tier 2 design extrapolations;
14. freeze the representative-address manifest for OBS-085c;
15. implement a full correlated campaign simulator;
16. validate any campaign approximation against exact execution;
17. execute Tier 3 campaign stress tests;
18. produce OBS-085d synthesis;
19. write the canonical observatory log entry;
20. and commit the complete OBS-085 lineage without modifying OBS-084.

OBS-085b may not begin before OBS-085b0 passes.

OBS-085c may not begin before:

- OBS-085a is frozen;
- OBS-085b0 is frozen;
- the OBS-085b scenario manifest is frozen;
- and the representative-address manifest is frozen.

---

## 54. Stop Conditions

Execution must stop if:

- any required OBS-084 identity fails validation;
- a frozen OBS-084 artifact has changed;
- a support predicate cannot be reconstructed exactly;
- a complement definition cannot be reconstructed exactly;
- the confirmation gate contract cannot be reproduced;
- the discovery candidate-selection contract cannot be reproduced;
- discovery and confirmation clusters are accidentally pooled;
- observation rows are treated as independent clusters;
- target and controls are independently simulated in a primary analysis;
- an injection violates the mathematical range of the artifact;
- an injection does not represent the declared failure predicate;
- missingness changes outside the declared mechanism;
- a simulator fails null-estimator calibration;
- a simulator fails reproducibility;
- a simulation scenario is added after results are inspected;
- a representative target address is added after campaign results are inspected;
- a larger object-count scenario duplicates rows or objects without a generating model;
- a campaign approximation materially disagrees with exact execution and is still used as canonical;
- entitlement rules differ from OBS-084;
- or observed effect estimates are substituted for true-effect parameters in an observed-power calculation.

All failures must be written to a dedicated audit artifact.

---

## 55. Interpretation Rules

### 55.1 Permitted conclusions

OBS-085 may conclude that:

- a support address is or is not structurally evidence-feasible;
- the current design has limited simulated gate-passage probability below a specified injected-effect range;
- an address is sensitive to the declared simulator family;
- a support family is structurally unconfirmable under the observed object layout;
- multiplicity materially reduces simulated confirmation probability;
- control adjustment limits detection of non-specific synthetic effects;
- additional hypothetical independent object clusters improve simulated sensitivity under an explicit exchangeability assumption;
- selection inflates discovery estimates under the declared campaign simulator;
- confirmation shrinkage is expected after discovery selection;
- a statistically detectable address remains entitlement-capped;
- the current evidence is insufficient to estimate sensitivity for a particular address;
- or a simulator family is too unstable to support a narrow operating-envelope claim.

### 55.2 Prohibited conclusions

OBS-085 must not conclude that:

- an FL3 witness exists because a synthetic effect is detectable;
- no FL3 witness exists because simulated sensitivity is high;
- an undetected witness is probably present because simulated sensitivity is low;
- increasing object count will necessarily produce a witness;
- a simulated bottleneck identifies a causal mechanism;
- a statistically detectable support is actionable;
- a support address is repair-ready;
- a control-limited simulation identifies the correct intervention;
- a larger-object scenario forecasts future empirical success;
- or one simulator family provides the uniquely true operating characteristic.

---

## 56. Binding Guardrails

The following statements are binding:

> Monte Carlo precision is not instrument-model certainty.

> Claim entitlement is an epistemic ceiling, not a component of structural estimability.

> End-to-end sensitivity requires joint campaign simulation. Multiplicity, candidate competition, and non-dominance may not be represented through independent address-level simulations without explicit qualification.

> Synthetic injection must be predicate-valid. The injection mechanism must preserve the mathematical range, dependence structure, control covariance, and artifact semantics of the declared failure predicate.

> Larger object-count scenarios are hypothetical design extrapolations under an explicit exchangeability assumption.

> A simulated probability is not evidence that the simulated effect exists.

> OBS-085 does not compute observed power.

> OBS-085 does not reinterpret or alter the null FL3 result of OBS-084.

---

## 57. Success Criterion

OBS-085 succeeds if it produces a validated, cluster-aware, address-specific, and simulator-conditional map of:

- evidence feasibility;
- support and complement coverage;
- effective object-level evidence;
- joint target-control estimability;
- statistical gate-passage behavior;
- full campaign gate-passage behavior;
- simulator-family uncertainty;
- Monte Carlo uncertainty;
- object-level influence;
- predicate-specific injection validity;
- control-specificity burden;
- multiplicity burden;
- candidate competition;
- discovery effect inflation;
- confirmation transport;
- campaign-wise false-witness behavior;
- hypothetical object-count extrapolations;
- and deterministic claim-entitlement ceilings.

No single model-conditional probability is sufficient as the canonical result.

OBS-085 does not require:

- high simulated gate-passage probability;
- identification of an FL3 witness;
- a recommended intervention;
- or a positive empirical result.

---

## 58. Canonical Claim

> OBS-085 characterizes the conditional operating envelope of the frozen PAM/RIG artifact-direct failure-support instrument. It separates evidence feasibility, statistical gate-passage behavior, end-to-end campaign behavior, and deterministic claim entitlement; evaluates predicate-valid synthetic effects under joint object-level target-control models; and reports sensitivity across a frozen family of simulation-generating assumptions. Because the observed design contains few independent object clusters, the results are model-conditional instrument diagnostics rather than model-independent power estimates. They do not establish that an artifact-direct witness exists, alter the null FL3 result of OBS-084, identify a causal origin, or imply repair or intervention readiness.

---

## 59. Canonical Study Progression

OBS-085 completes the following methodological progression:

\[
\text{OBS-081: register reusable invariance}
\]

\[
\text{OBS-082/083: audit readiness and evidential limitations}
\]

\[
\text{OBS-084: seek artifact-direct failure witnesses under blinding}
\]

\[
\text{OBS-085: characterize the conditional resolving power of the witness instrument}
\]

The purpose of OBS-085 is not to give OBS-084 a second opportunity to obtain a positive result.

Its purpose is to establish, within explicit and auditable limits, what the current artifact hierarchy can see, what it cannot see, and which forms of additional evidence would change its resolving capacity under declared assumptions.
