# Theory Memo 002 — Negative Controls in RIG

**Status:** Canonical conceptual memo  
**Scope:** PAM/RIG artifact lineage through OBS-082  
**Claim level:** Diagnostic specification; no causal claim

## Purpose

This memo defines negative controls for Reusable Invariance Geometry.

A relation that survives multiple tests may still survive for the wrong reason. It may reflect generic carrier capacity, leakage, broad separability, or a transformation that fails to challenge the relevant structure.

The central question is therefore not only:

> Does the relation survive?

It is also:

> Does it survive more specifically than the appropriate alternatives?

## Definition

A RIG negative control is a matched comparison designed to test whether an apparent invariant is specific to the declared:

- relation;
- carrier;
- contract;
- failure mode;
- geometry-needed level;
- artifact lineage.

For target record:

\[
r=(R,C,T\mid\Omega),
\]

controls may include:

\[
(R^{-},C,T),
\qquad
(R,C^{-},T),
\qquad
(R,C,T^{-}).
\]

A strong target should be more stable, more differentiated, or more localized than the matched control under the declared criterion.

## Negative-control dimensions

### Relation controls

Hold carrier and contract fixed while changing the relation.

Examples include comparing:

- C-vs-Cp2;
- C-vs-Cp3;
- Cp2-vs-Cp3;
- three-way classification

through the same carrier.

This tests whether the carrier expresses the target relation specifically rather than supporting generic separability.

### Carrier controls

Hold relation and contract fixed while changing the carrier.

Carrier examples include:

- `stability_core_3`;
- `geometry_scores_only`;
- `path_shares_only`;
- `stability_plus_geometry`;
- `no_window`;
- `strict_numeric_all`.

This tests whether a carrier has a differentiated role or simply acts as an overbroad substrate.

### Contract and transformation controls

Hold relation and carrier fixed while changing the testing contract or operation.

Relevant controls may include:

- numeric transforms;
- scale-band restrictions;
- feature-family projections;
- structural resampling;
- holdout choices;
- normalization choices.

A useful invariant should survive meaningful transformations and may weaken under transformations expected to disrupt its support.

### Failure-mode controls

A proposed failure should localize where the theory predicts rather than appearing everywhere.

If every carrier, relation, and contract fails similarly, the failure is diffuse rather than specific.

### Geometry-needed controls

Test whether the declared geometry is actually required.

A geometry-enriched carrier outperforming a compact carrier does not by itself show geometric necessity. The comparison must determine whether geometry:

- is unnecessary;
- sharpens an already stable relation;
- is required for survival;
- is required only for failure localization;
- remains insufficient.

### Shuffled and permutation controls

Destroy the target relation while preserving data shape or marginal structure.

These controls are necessary for leakage and chance-separation checks, but they are not sufficient for RIG specificity. A model may outperform shuffled labels while still relying on an overbroad carrier or weak relation contrast.

## Why OBS-082 remained weak on negative controls

OBS-081 built a survival and registry layer.

It did not yet build a fully matched contrast layer.

OBS-082 therefore found that all 24 records remained weak in negative-control contrast. The evidence showed reusable structure, but not yet enough specificity to determine whether each record survived for the intended reason.

## Aggregate contrast score

A provisional negative-control score may combine several dimensions:

\[
N_i=
 w_RN_R+w_CN_C+w_KN_K+w_GN_G+w_PN_P,
\]

where the terms represent relation, carrier, contract, geometry-needed, and permutation-control evidence.

The score is secondary to the underlying evidence. A single aggregate must not hide which control family remains weak.

## Failure and leakage risks

Negative controls should guard against:

- generic carrier strength;
- label or identifier leakage;
- overbroad robustness;
- geometry inflation;
- diffuse failure;
- generic repair recommendations;
- the permutation-adequacy fallacy;
- unmatched provenance;
- unequal support density.

## Candidate-readiness implication

A candidate-ready record would need evidence that:

- the target relation is stronger than matched relations;
- the target carrier has a specific role;
- the relation survives relevant contracts;
- failure is not generic across records;
- the geometry-needed claim is justified;
- shuffled, decoy, or leakage-prone controls do not reproduce the result.

## Canonical summary

> In RIG, a negative control tests whether reusable-invariance evidence is specific to the declared relation, carrier, contract, and failure mode. Survival without matched contrast is diagnostic evidence, not readiness evidence.
