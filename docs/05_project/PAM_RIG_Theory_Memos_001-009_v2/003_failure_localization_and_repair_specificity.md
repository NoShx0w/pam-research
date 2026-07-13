# Theory Memo 003 — Failure Localization and Repair Specificity

**Status:** Canonical conceptual memo  
**Scope:** PAM/RIG artifact lineage through OBS-082  
**Claim level:** Diagnostic maturity specification

## Purpose

OBS-082 identified two major blockers beyond negative-control contrast:

- diffuse failure localization in nearly all records;
- generic repair specificity in all records.

This memo defines those limitations and the bridge required to move from diagnosis toward a falsifiable repair hypothesis.

## Failure

A RIG failure is a criterion-specific degradation in a registered relation’s reusable-invariance profile.

It may include:

- weakening;
- instability;
- reversal;
- ambiguity;
- loss of matched-control separation;
- contract sensitivity;
- carrier dependence;
- reproducibility loss.

Failure is an artifact-level diagnostic term. It does not imply a broken system or a causal mechanism.

## Failure localization

Failure localization is the degree to which degradation can be assigned to a specific artifact address.

Possible addresses include:

- relation;
- carrier;
- contract;
- transformation;
- scale band;
- feature family;
- cohort;
- transition;
- window;
- route segment;
- seam or boundary region;
- provenance slice.

The operational question is:

> Does reuse weaken here, rather than everywhere?

Let:

\[
L(r)
\]

denote localization strength for registry record \(r\).

### Diffuse failure

A failure is diffuse when its source cannot be isolated and similar degradation appears broadly across relations, carriers, contracts, or structural units.

### Localized failure

A failure is localized when evidence can be tied to a specific, inspectable artifact locus relative to an appropriate complement.

Localization does not establish causal origin.

## Repair annotation

A repair annotation is a registry-level note suggesting a direction for further investigation.

Examples include:

- strengthen matched controls;
- add geometry;
- inspect seam-adjacent windows;
- refine the scale band;
- separate transition classes.

A repair annotation is useful metadata. It is not a hypothesis.

## Repair specificity

Repair specificity is the degree to which a proposed repair names:

- the target relation;
- the target carrier;
- the failure address;
- the proposed modification;
- the measurable expected effect;
- the matched negative control;
- the falsification condition;
- the scope.

Let:

\[
Q(r)
\]

represent repair specificity.

## Repair hypothesis

A repair hypothesis is a falsifiable, contrast-aware claim about how a declared modification might improve or restore a registered relation within a declared artifact context.

A minimal structure is:

\[
H_\rho=(R,C,Z,\rho,M,R^{-},\Phi,\Omega),
\]

where:

- \(Z\): localized failure site;
- \(\rho\): proposed repair;
- \(M\): success metric;
- \(R^{-}\): matched control;
- \(\Phi\): falsifier.

A repair hypothesis is still not evidence that the repair works.

## Maturity bridge

The intended sequence is:

\[
\text{diagnostic invariant}
\rightarrow
\text{localized failure}
\rightarrow
\text{specific repair hypothesis}.
\]

The OBS-082 state was instead:

\[
\text{diagnostic invariant}
\rightarrow
\text{diffuse failure}
\rightarrow
\text{generic repair annotation}.
\]

This explains why the registry remained diagnostic-only.

## Localization taxonomy

A localization may be:

- relation-local;
- carrier-local;
- contract-local;
- geometry-level-local;
- scale-local;
- cohort-local;
- transition-local;
- boundary- or seam-local;
- provenance-local.

These labels should remain separate because they support different future tests.

## Repair-specificity ladder

- **R0:** no repair annotation;
- **R1:** generic repair annotation;
- **R2:** relation-specific annotation;
- **R3:** relation + carrier-specific annotation;
- **R4:** localized repair candidate with a named metric;
- **R5:** contrastive repair hypothesis with control and falsifier.

R5 is the minimum plausible repair-hypothesis-ready level.

It is not intervention success.

## Record structure

A richer registry record may be written as:

\[
r_i=(R_i,C_i,T_i,S_i,N_i,L_i,Q_i,G_i,Z_i,\rho_i).
\]

Candidate-readiness requires minimum thresholds in survival, contrast, localization, and repair specificity.

OBS-082 found that localization and specificity remained below those thresholds.

## Falsification conditions

A localization claim should fail if:

- degradation is equally diffuse outside the proposed site;
- the same site appears for all matched controls;
- the result depends on one outlier;
- provenance is incomplete;
- the site is identified through leakage;
- the direction is unstable across resamples.

A repair-specificity claim should fail if:

- no failure site is named;
- no measurable target is stated;
- no matched control exists;
- no falsification condition is possible;
- the repair is only generic language.

## Canonical summary

> Failure localization is the addressability of failed reuse. Repair specificity begins only after a failure can be tied to a declared artifact site, metric, control, and falsifier. Generic repair annotations are diagnostic metadata, not intervention hypotheses.
