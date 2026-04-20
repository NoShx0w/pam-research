# Topological Identity

## Purpose

This document explains the **topological identity** program in PAM.

Topological identity is the part of the observatory that asks:

- what counts as the local structural identity of a state?
- how should two local identities be compared?
- can identity change be treated as a field over the manifold?
- is identity transport path-dependent?
- where does obstruction localize?

This layer lives primarily in:

- `src/pam/topology/identity.py`
- `src/pam/topology/identity_proxy.py`
- `src/pam/topology/identity_field.py`
- `src/pam/topology/identity_metric.py`
- `src/pam/topology/identity_metric_full.py`
- `src/pam/topology/identity_transport.py`
- `src/pam/topology/identity_obstruction.py`

The central idea is:

**identity is not treated as a scalar label, but as local relational organization.**

---

## Why this layer exists

Geometry and phase already tell the observatory a great deal:

- where the manifold bends
- where the seam lies
- how far states sit from transition structure
- where operator-derived fields concentrate

But geometry alone does not answer a deeper question:

- when are two local states structurally “the same” or “different” in their organization?

The topological identity layer exists to answer that question.

It gives the observatory a way to treat local structural organization as a first-class object rather than only as a collection of scalar values.

---

## Identity is relational, not scalar

One of the strongest design choices in PAM is that local identity is not reduced to:

- one metric value
- one phase sign
- one family label
- one hand-picked score

Instead, identity is represented as **relational structure**.

This means that a state’s identity depends on how local objects and transitions are arranged, not only on what a single coordinate says.

That is why the identity program begins with graphs rather than scalar summaries.

---

## `IdentityGraph`: the local identity object

The central object is the `IdentityGraph`.

An `IdentityGraph` is a local relational representation of a state’s topological organization.

Depending on the local construction, it may encode things such as:

- basin-like objects
- critical objects
- adjacency relations
- transition relations
- directional or connectivity summaries

The exact implementation details are less important than the conceptual role:

an `IdentityGraph` is meant to represent the **minimal relational organization** of a local topological state.

This is the core object of the identity layer.

---

## Why graphs are used

Graphs are used because they preserve structural relations better than scalar proxies.

A graph can express:

- what kinds of local objects exist
- how many there are
- how they connect
- how transitions enter or leave them
- how local organization differs from one region to another

This allows the observatory to talk about identity in a way that is:

- local
- structured
- comparison-friendly
- and less dependent on one arbitrary scalar reduction

That is exactly what this layer needs.

---

## Identity distance

Once local identity objects exist, the next question is:

- how should two identities be compared?

PAM answers this with a **structural identity distance**.

This is not intended to be an exact graph-isomorphism theorem. Instead, it is an operational comparison between relational organizations.

Typical ingredients of the comparison include:

- node-signature histograms
- adjacency-signature summaries
- transition-signature summaries
- degree-like local structure counts

So identity distance should be understood as:

**difference in local relational organization**, measured in an operationally stable way.

This is one of the most important conceptual moves in the whole repository.

---

## Why structural multisets matter

A particularly good design choice in PAM is that identity comparison is based on structural multisets and signature counters rather than exact named-node matching.

That matters because it makes the comparison:

- label-invariant
- robust to local naming conventions
- lightweight enough to compute operationally
- still sensitive to real structural differences

So the identity layer does not require expensive exact graph matching to become scientifically useful.

It measures identity difference through structural summary distributions.

---

## Proxy versus object

The repository makes an important distinction between:

- the richer identity object
- operational identity proxies

This distinction is preserved through modules such as `identity_proxy.py`.

That is scientifically important.

Not every downstream analysis needs the full graph object, and not every quick diagnostic can afford the full identity-comparison machinery. Proxy objects therefore exist as lower-cost approximations.

But the repository does not collapse the distinction.

This is one of the strengths of the design:

- the full object is acknowledged
- the proxy is acknowledged
- and the two are not confused

That keeps the science honest.

---

## Identity as a field

Once pairwise identity differences can be measured, PAM promotes identity change into a field.

This is the role of `identity_field.py`.

The idea is:

- compare neighboring states
- estimate how identity changes across local directions
- record that change as a field-like object over the control manifold

This means the observatory can ask not only:

- what is the local identity here?

but also:

- how rapidly does identity change nearby?
- where is identity variation large?
- does identity change organize directionally?
- are there local circulation-like patterns?

This is where identity becomes more than a local object. It becomes a manifold-facing field.

---

## Identity magnitude

One simple but important output of the identity field is **identity magnitude**.

Identity magnitude is the local size of identity change across neighboring states.

This can be read as:

- how quickly relational organization changes here
- how sensitive local identity is to nearby motion on the manifold
- where the identity field is calm versus active

This is often the first useful scalar summary of the identity layer.

But it should still be understood as derived from relational comparison, not as a primitive quantity.

---

## Legacy spin and later refinements

Earlier identity-field work in PAM used spin-like or curl-like summaries as a first operational handle on local circulation structure.

These remain useful as historical or comparison objects, but later canonicalization shifted toward more explicit transport- and obstruction-facing quantities.

That means legacy spin should be read as:

- a useful precursor
- a comparison field
- not necessarily the final canonical identity object

This distinction also appears in the TUI’s identity hierarchy.

---

## Identity metric layers

The repository includes `identity_metric.py` and `identity_metric_full.py` because the identity program does not stop at local scalar summaries.

Once identity distance exists, it becomes natural to ask:

- is there a larger geometry of identity difference itself?
- can identity distance organize the manifold in its own right?

These modules push identity comparison toward a metric-like layer.

Conceptually, this means identity is not treated as a one-off local descriptor. It is promoted toward a larger comparison geometry.

That is an ambitious and important move.

---

## Identity transport

Once identity can be compared locally, the next major question is:

- does identity change depend on the path taken through the manifold?

This is the role of `identity_transport.py`.

Identity transport is the attempt to compare how local identity evolves when moved along different local routes.

This is where PAM begins to ask holonomy-like questions in an operational way.

The central idea is:

- transport identity around local loops
- compare the residual
- ask whether different routes produce the same identity evolution or not

This is one of the deepest transitions in the whole observatory.

---

## Loop residual and holonomy-like behavior

A particularly important operational object is the local loop residual.

For a small cell or loop in the control lattice, PAM compares identity transport along two different routes that connect the same endpoints.

If the transported identity disagrees, there is a nonzero residual.

This residual is not treated as a grand theorem of formal holonomy. It is treated as a **holonomy-like operational signal**:

- identity transport is path-dependent here
- local organization is not naively integrable
- loop transport leaves structural residue

That is the right level of scientific claim.

It is bold enough to be meaningful, but restrained enough to remain honest.

---

## Obstruction

Once loop residuals exist, the next question is:

- where do these residuals concentrate?

This is the role of `identity_obstruction.py`.

Obstruction is the node-local concentration of loopwise identity path-dependence.

It is built by aggregating loop residual information back onto nearby nodes, producing quantities such as:

- mean holonomy
- mean absolute holonomy
- max absolute holonomy
- signed or weighted residual summaries
- unsigned obstruction magnitude

The key idea is:

**obstruction is not postulated; it is constructed from identity transport mismatch.**

That makes obstruction one of the strongest derived objects in the observatory.

---

## Signed and unsigned obstruction

The obstruction layer can be read in two ways.

### Unsigned obstruction

Unsigned obstruction measures the size of local transport-derived obstruction without regard to orientation.

It answers:

- where is identity transport strongly path-dependent?

### Signed obstruction

Signed obstruction keeps track of orientation or polarity in the local obstruction field.

It answers:

- how is local path-dependent identity structure oriented?
- do neighboring regions organize with opposite obstruction sign?

This is why signed obstruction is especially rich in grid-facing observatory views.

It allows the identity layer to express not only intensity, but oriented structural organization.

---

## Absolute holonomy

Absolute holonomy is one of the main canonical summaries of the identity-transport layer.

It measures the local magnitude of loop transport residual without regard to sign.

This makes it an especially useful bridge quantity:

- more directly tied to transport than generic field magnitude
- easier to compare spatially than a raw signed residual
- useful as a canonical identity-mode overlay in the TUI

Absolute holonomy and obstruction are closely related, but they are not identical in interpretation:

- holonomy emphasizes loop-transport residual
- obstruction emphasizes the localized concentration of that path dependence

---

## Why this matters scientifically

The topological identity program matters because it lets PAM ask deeper structural questions than scalar field inspection alone can answer.

For example:

- are two nearby states geometrically close but topologically different?
- are seam-adjacent regions also identity-unstable?
- where does path dependence localize?
- do transition corridors coincide with identity obstruction?
- can local organization be transported coherently?

These are genuinely structural questions.

Without this layer, the observatory would have geometry and operators, but not yet a robust way to talk about local organizational identity and its failure of naive transport.

---

## Relationship to geometry and phase

Topological identity is downstream of the geometry and phase layers.

Geometry provides:
- the manifold substrate
- neighborhood structure
- intrinsic distances
- directional context

Phase provides:
- signed regime structure
- seam organization
- seam-local coordinates

The identity layer then asks how local topological organization behaves on top of that substrate.

So identity is not a replacement for geometry or phase. It is a deeper organizational layer built on them.

---

## Relationship to operators

Identity also interacts naturally with operator-derived fields.

For example, later observatory analysis can ask how identity magnitude, holonomy, or obstruction relate to:

- Lazarus
- criticality
- seam distance
- response-flow structure
- route-family organization

This is one reason the identity program matters so much.

It is not isolated. It creates a new structural language that can be compared against the rest of the observatory.

---

## Relationship to the TUI

The TUI exposes the identity layer in one of its clearest public forms.

Identity mode in the TUI already reflects the canonical hierarchy of the identity program:

- identity magnitude
- absolute holonomy
- unsigned obstruction
- signed obstruction
- legacy spin as comparison

This is useful because it makes the topological identity program inspectable in practice, not only in code.

The TUI also helps make clear that the identity layer is not merely another scalar map. It is a comparative structural console.

---

## What topological identity is not

It is helpful to say what this layer is **not**.

Topological identity is not:

- a single class label
- a generic graph-embedding trick
- exact formal algebraic topology
- a completed holonomy theory
- a substitute for all other observatory layers

Instead, it is an operational program for representing, comparing, transporting, and localizing structural organization on the observatory manifold.

---

## Limitations

This layer also has real limits.

Its outputs depend on:

- the quality of local object extraction
- the chosen identity summaries
- the proxy/object distinction
- the local loop or neighborhood construction
- the aggregation rule used to produce obstruction-like fields

So the identity program should be read as:

- operational
- inspectable
- scientifically meaningful
- but not identical to the strongest possible formal topological theory

That is not a weakness. It is the right level of honesty for an observatory program.

---

## Why the name matters

The phrase “topological identity” matters because it signals a real shift in how identity is understood.

Identity is not being treated as:

- label identity
- or raw coordinate identity

It is being treated as:

- local structural organization
- preserved or altered under manifold motion
- comparable through relational summaries
- and sometimes obstructed under loop transport

That is a much richer notion of identity than most ordinary metric pipelines support.

---

## Summary

Topological identity is PAM’s program for representing and comparing local structural organization on the manifold.

Its key ideas are:

- local identity is represented as a relational object (`IdentityGraph`)
- identities are compared through structural distance rather than scalar labels
- identity change can be promoted into a field over the manifold
- pathwise identity transport can leave loop residuals
- those residuals can be localized into holonomy- and obstruction-like observables
- proxy and full-object distinctions are preserved explicitly

The central principle is:

**topological identity in PAM is not a label attached to a state, but a locally structured organization whose differences, transport, and path dependence can be measured.**

That is why this layer is one of the deepest and most distinctive parts of the repository.
