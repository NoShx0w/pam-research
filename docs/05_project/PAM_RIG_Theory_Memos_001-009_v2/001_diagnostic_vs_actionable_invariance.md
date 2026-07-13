# Theory Memo 001 — Diagnostic vs Actionable Invariance

**Status:** Canonical conceptual memo  
**Scope:** PAM/RIG artifact lineage through OBS-082  
**Claim level:** Diagnostic theory; no intervention or causal claim

## Purpose

This memo establishes the first maturity distinction in Reusable Invariance Geometry (RIG): a relation may be stable enough to register and interpret without being sufficiently specific to support an intervention hypothesis.

The central rule is:

> Survival is not actionability.

A registered invariant is an indexed evidence object. It is not automatically a control variable, repair target, or intervention-ready hypothesis.

## Core definitions

### Invariant

A relation or structural distinction that remains sufficiently stable under a declared family of transformations or contracts.

\[
\operatorname{Invariant}(R,T\mid\Omega)
\]

means that relation \(R\) is preserved under tested transformations \(T\) within scope \(\Omega\).

An invariance claim is incomplete unless it names:

- the relation;
- the tested transformation family;
- the evidence surface;
- the scope.

### Reusable invariant

An invariant that survives enough variation in carrier, contract, or sampling structure to be used again within the declared artifact lineage.

\[
\operatorname{ReusableInvariant}(R,C,T\mid\Omega)
\]

Reusable does not mean universal. It means reusable under the tested conditions.

### Registered invariant

A reusable-invariance candidate represented as a relation × carrier record with explicit metadata.

A minimal record may be written as:

\[
r=(R,C,s,g,f,\rho)
\]

where:

- \(R\): relation;
- \(C\): carrier;
- \(s\): survival or registry status;
- \(g\): geometry-needed assessment;
- \(f\): failure-localization state;
- \(\rho\): repair annotation.

Registration creates inspectability. It does not establish readiness.

### Diagnostic invariant

A registered invariant that reliably characterizes observed regime structure but lacks sufficient negative-control contrast, failure localization, or repair specificity to support a conservative intervention hypothesis.

A diagnostic invariant answers:

> What structure is reliably visible within this artifact lineage?

It does not yet answer:

> What should be changed, where, and with what expected effect?

### Actionable invariant

A future maturity category in which a diagnostic invariant has sufficiently strong:

- invariant survival;
- negative-control contrast;
- failure localization;
- repair specificity;
- provenance and leakage discipline.

A minimal formal gate is:

\[
\operatorname{Actionable}(r)
\iff
S(r)\ge s_0
\land N(r)\ge n_0
\land L(r)\ge l_0
\land Q(r)\ge q_0.
\]

Current PAM/RIG artifacts do not establish this category.

## The OBS-081 → OBS-082 maturity break

OBS-081 showed that the OBS-080 contract-sensitivity results could be represented as a Reusable Invariance Registry over relation × carrier records.

OBS-082 then audited whether those records were ready to support intervention hypotheses.

The result was not a distinction between “real” and “fake” invariants. It was a distinction between:

- invariants good enough to describe;
- invariants good enough to intervene from.

All 24 records remained diagnostic-only.

The dominant limitations were:

- weak negative-control contrast;
- diffuse failure localization;
- generic repair specificity.

## Artifact-to-theory mapping

| Artifact layer | Theoretical role |
|---|---|
| OBS-080 contract sensitivity | Tests survival under declared transformations |
| OBS-081 registry | Converts survival evidence into indexed relation × carrier records |
| OBS-082 readiness audit | Tests whether registry evidence is specific and addressable enough for hypothesis formation |

## Falsification conditions

A registered-invariance claim should be weakened or rejected if:

- the relation does not survive the stated contracts;
- the result depends on one narrow transformation;
- the carrier merely encodes the target through leakage;
- provenance is incomplete;
- matched controls show equivalent behavior.

A diagnostic-invariance claim should be weakened if:

- the structural interpretation cannot be reproduced;
- apparent robustness disappears under reasonable resampling;
- the relation is not distinguishable from negative controls.

An actionable-invariance claim should fail if:

- no direct failure site exists;
- the repair annotation remains generic;
- the proposed repair lacks a measurable target and falsifier;
- no admissible control exists;
- intervention success has not been demonstrated.

## Consequence for the research program

The correct maturity ladder is:

\[
\text{observed pattern}
\rightarrow
\text{robust diagnostic pattern}
\rightarrow
\text{reusable invariant}
\rightarrow
\text{registered invariant}
\rightarrow
\text{diagnostic invariant}
\rightarrow
\text{candidate-ready invariant}
\rightarrow
\text{actionable invariant}.
\]

No transition is automatic.

## Canonical summary

> A registered reusable invariant is not automatically intervention-ready. Diagnostic invariance establishes stable, interpretable structure within a declared artifact lineage; actionability additionally requires strong negative controls, direct failure localization, repair specificity, and a separate readiness audit.
