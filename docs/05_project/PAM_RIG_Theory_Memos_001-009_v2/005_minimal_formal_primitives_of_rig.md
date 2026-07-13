# Theory Memo 005 — Minimal Formal Primitives of RIG

**Status:** Provisional canonical vocabulary  
**Scope:** PAM/RIG artifact lineage through OBS-082  
**Claim level:** Operational formalization, not a completed mathematical theory

## Purpose

This memo defines the smallest formal vocabulary needed to describe RIG without claiming a universal learning theory, formal topology, or causal control framework.

## Artifact object space

Let:

\[
X
\]

be the current artifact object space.

Its elements may include:

- supports;
- windows;
- cohorts;
- transitions;
- feature rows;
- structural resampling units;
- registry records;
- C/Cp2/Cp3 artifact objects.

\(X\) is not a universal cognitive state space.

## Relation

Let:

\[
R
\]

be a structural comparison, distinction, or equivalence under study.

Examples include pairwise and three-way regime relations.

## Carrier

Let:

\[
C:X\rightarrow V_C
\]

be a representation or measurement map into carrier space \(V_C\).

A carrier may contain compact stability features, geometry scores, path shares, window features, or other artifact-grounded representations.

## Context

Let:

\[
\Omega
\]

encode the artifact scope:

- model and corpus provenance;
- comparison family;
- study lineage;
- committed source artifacts;
- declared limitations.

## Contract

Let:

\[
K
\]

be a testing rule defining fixed conditions, allowed variation, and survival criteria.

## Transformation

Let:

\[
T
\]

be the concrete operation executed under contract \(K\).

## Survival

Let:

\[
S(R,C,K,T\mid\Omega)
\]

measure the degree to which relation \(R\) survives through carrier \(C\) under the declared contract and transformation.

## Negative-control contrast

Let:

\[
N(r)
\]

measure whether the record’s survival is more specific than matched relation, carrier, contract, geometry, and permutation controls.

## Failure localization

Let:

\[
L(r)
\]

measure how directly degradation can be assigned to a specific artifact address.

OBS-082 found this dimension diffuse for nearly all records.

## Repair specificity

Let:

\[
Q(r)
\]

measure how specifically a proposed repair names the relation, carrier, failure site, modification, metric, control, and falsifier.

OBS-082 found this dimension generic for all records.

## Geometry-needed

Let:

\[
GNeed(R,K\mid\Omega)
\]

represent how much geometry is required for survival, sharpening, localization, or repair specificity.

## Failure site

Let:

\[
Z
\]

denote an artifact address at which degradation may be localized.

In the OBS-082 state, \(Z\) remained unresolved for most records.

## Repair annotation

Let:

\[
\rho
\]

be a repair annotation or operator sketch.

Current values are diagnostic metadata, not validated operators.

## Registry record

A rich registry record may be written as:

\[
r=(R,C,\Omega,K,T,S,N,L,Q,GNeed,Z,\rho).
\]

The registry is:

\[
\mathcal R_\Omega=\{r_i\}_{i=1}^{n}.
\]

For the OBS-081 registry:

\[
n=24.
\]

## Maturity predicates

### Registered

\[
\operatorname{Registered}(r)
\iff
S(r)\ge s_{reg}.
\]

### Diagnostic

\[
\operatorname{Diagnostic}(r)
\iff
\operatorname{Registered}(r)
\land
\operatorname{StructurallyInterpretable}(r).
\]

### Candidate-ready

\[
\operatorname{CandidateReady}(r)
\iff
\operatorname{Diagnostic}(r)
\land N(r)\ge n_0
\land L(r)\ge l_0
\land Q(r)\ge q_0.
\]

### Actionable

Actionable is a later category requiring a separate readiness and intervention standard.

Current PAM/RIG artifacts contain no actionable records.

## Evidence survival versus repair hypothesis

The key non-implication is:

\[
S(r)\text{ high}
\not\Rightarrow
\operatorname{RepairHypothesisReady}(r).
\]

A repair hypothesis requires direct failure-site evidence and a falsifiable modification claim beyond survival evidence.

## Falsification

The formal primitives should be revised if:

- artifact objects cannot be consistently indexed;
- relations cannot be specified independently of labels;
- carriers encode the target through leakage;
- contracts are not reproducible;
- survival cannot be separated from generic carrier strength;
- context cannot be preserved;
- proposed failure sites cannot be retrieved.

## Scope guardrails

This vocabulary does not establish:

- a universal learning theory;
- formal algebraic topology;
- strict groupoids or inverses;
- causal mechanisms;
- model-independent invariance;
- intervention or control.

## Canonical summary

> RIG minimally requires artifact objects, relations, carriers, contexts, contracts, transformations, survival, matched contrast, failure localization, repair specificity, geometry-needed, failure sites, and versioned registry records. The vocabulary is operational and artifact-scoped, not a completed universal formalism.
