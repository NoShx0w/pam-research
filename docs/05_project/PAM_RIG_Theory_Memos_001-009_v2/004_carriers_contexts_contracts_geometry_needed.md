# Theory Memo 004 — Carriers, Contexts, Contracts, and Geometry-Needed

**Status:** Canonical conceptual memo  
**Scope:** PAM/RIG artifact lineage through OBS-082

## Purpose

RIG depends on several concepts that can be easily conflated: relation, carrier, context, contract, transformation, geometry-needed level, and repair target.

This memo keeps those categories distinct.

## Relation

A relation is the structural distinction, comparison, or equivalence being tested for reuse.

Examples include:

- C-vs-Cp2;
- C-vs-Cp3;
- Cp2-vs-Cp3;
- three-way regime structure.

The relation is the target of invariance.

## Carrier

A carrier is the representational or measurement substrate through which a relation is expressed.

Examples include:

- `stability_core_3`;
- `geometry_scores_only`;
- `path_shares_only`;
- `stability_plus_geometry`;
- `no_window`;
- `strict_numeric_all`.

A carrier is not the relation itself.

## Context

Context is the artifact and interpretive boundary within which a claim is valid.

It may include:

- model and corpus lineage;
- C/Cp2/Cp3 comparison family;
- matched shared-14 substrate;
- OBS-078–082 evidence lineage;
- current generated artifacts.

A claim should be written as:

\[
\operatorname{Claim}(R,C,T\mid\Omega).
\]

Context prevents artifact-local results from becoming universal claims.

## Contract

A contract is the declared testing condition specifying:

- what remains fixed;
- what is allowed to vary;
- how survival is measured;
- what counts as acceptable degradation.

OBS-080 contracts included transformations such as:

- numeric transforms;
- scale-band restrictions;
- feature-family projections;
- structural resampling.

## Transformation

A transformation is the actual operation performed under a contract.

The contract defines the rule; the transformation is the executed instance.

## Geometry-needed level

Geometry-needed is a diagnostic assessment of how much geometric or structural representation is required for a relation to survive, sharpen, localize failure, or support repair specificity.

Geometry-needed is not a carrier identity.

A provisional ladder is:

- **G0:** no geometry claim;
- **G1:** compact local carrier sufficient;
- **G2:** geometry sharpens but is not required;
- **G3:** geometry required for relation survival;
- **G4:** geometry required for failure localization;
- **G5:** geometry required for repair specificity;
- **G6:** geometry remains insufficient.

## Repair target

A repair target is a localized artifact address at which a future falsifiable repair hypothesis would apply.

It requires:

- relation;
- carrier;
- failure site;
- context;
- contract;
- measurable property.

A carrier or geometry-needed label alone is not a repair target.

## Why OBS-081 separated relation × carrier records from geometry-needed

These concepts answer different questions:

- relation × carrier asks where and how a relation is expressed;
- geometry-needed asks how much geometry is necessary for the evidence task.

Keeping them separate prevents:

- treating a carrier as a relation;
- treating geometry-needed as carrier identity;
- treating carrier convergence as actionability;
- treating geometry enrichment as proof of intrinsic geometry.

## Carrier-role taxonomy

A carrier may be classified as:

- primary;
- compact;
- redundant;
- weak redundant;
- geometry-sharpening;
- context-sensitive;
- insufficient;
- overbroad.

Carrier role is relative to a relation, contract, and context.

## Formal representation

Let:

\[
r_i=(R_i,C_i,K_i,T_i,\Omega_i).
\]

Then:

\[
S_i=S(R_i,C_i,K_i,T_i\mid\Omega_i)
\]

measures survival,

\[
G_i=\operatorname{GNeed}(R_i,K_i\mid\Omega_i)
\]

measures geometry-needed,

and:

\[
\operatorname{Role}_i=\operatorname{Role}(C_i\mid R_i,K_i,\Omega_i).
\]

The distinctions are:

\[
G_i\neq C_i
\]

and:

\[
\operatorname{Role}_i\neq G_i.
\]

## Risks

This memo guards against:

- carrier–relation conflation;
- geometry inflation;
- context erasure;
- weak contract definition;
- overbroad carrier strength;
- geometry-needed mislabeling;
- repair-target ambiguity.

## Canonical summary

> A relation is what is tested, a carrier is how it is represented, a contract defines the test, a transformation executes it, context bounds the claim, geometry-needed assesses representational necessity, and a repair target requires a separately localized failure site.
