# Generative Spine

## Purpose

This document describes the **generative spine** of the PAM repository.

The generative spine is the part of the system that produces evolving corpora, recursive runs, quenches, and trajectory-facing outputs. It is the substrate from which the observatory later derives geometry, phase structure, operators, topology, and interface-facing artifacts.

The main components are:

- `src/pam/engine/`
- `src/pam/dynamics/`
- `src/pam/injectors.py`
- `src/pam/corpora.py`
- top-level generative façades such as `pam.engine`, `pam.dynamics_mixture`, and `pam.injectors`

The central design fact is:

**generation and observation are separated.**

The generative spine is responsible for producing evolving text populations and run traces. It does not, by itself, define the full observatory interpretation of those runs.

---

## The generative spine as the substrate layer

PAM is not only an observatory. It is also a controlled generator of recursive language-system trajectories.

The generative spine exists to answer questions such as:

- how is a corpus evolved through time?
- what is fixed and what is mutable?
- how is anchor influence injected?
- how does endogenous continuation occur?
- how is a quench schedule imposed?
- when are snapshots and state labels recorded?

This is the layer where the repository defines the **mechanics of evolution**.

Its outputs later feed the observatory spine, but the generative spine itself remains comparatively lean and disciplined.

---

## Separation from the observatory spine

The repository has two major architectural halves.

### Generative spine

The generative spine includes:

- `engine/`
- `dynamics/`
- `injectors`
- `corpora`

Its job is to produce runs and evolving corpora.

### Observatory spine

The observatory spine includes:

- `types.py`
- `io/paths.py`
- `pipeline/state.py`
- `pipeline/runner.py`
- `pipeline/stages/*`

Its job is to turn generated outputs into stable observatory layers.

This split matters because it keeps PAM from collapsing into a single opaque application.

The repository distinguishes between:

- **what the system does**
- **how the observatory interprets what it does**

That separation is one of the strongest design choices in the codebase.

---

## `engine/`: execution skeleton

The `src/pam/engine/` package contains the runtime execution skeleton for PAM experiments.

This layer is responsible for:

- stepping a corpus forward in time
- applying the chosen dynamics or mixture law
- recording snapshots
- optionally extracting macrostate labels
- wrapping quench schedules into reproducible run contracts

The key conceptual point is that `engine/` does **not** define the full scientific meaning of a run. It defines the execution loop.

### Interpretation is injected, not hardcoded

The engine accepts a macrostate function rather than hardwiring one canonical state notion into the runtime itself.

That means the engine remains agnostic about whether the user wants to read a run in terms of:

- frozen / mixed state
- phase sign
- family labels
- or some later observatory classification

This is a very strong design choice.

The runtime produces an evolving corpus trace; higher-order interpretation is layered on later.

### Snapshot convention

The engine records state and snapshots according to a consistent temporal convention:

- inspect current corpus
- optionally record current state
- then apply the next update

That matters because all later lag, transition, and horizon logic depends on consistent time indexing.

---

## `engine/core.py`: minimal runtime contract

The core engine functions define the minimal PAM runtime.

At each step, the engine does three things:

1. evaluate the current corpus
2. optionally record a state or snapshot
3. apply the next corpus update

This minimality is deliberate.

The runtime does not need to know about:

- Fisher geometry
- seam structure
- curvature
- Lazarus
- obstruction
- TUI overlays

Those are all downstream observatory constructs.

The engine only needs:

- a corpus
- an update rule
- an optional state extractor

This makes the generative substrate reusable and conceptually clean.

---

## `engine/mixture.py`: canonical quench law

`src/pam/engine/mixture.py` contains the main quench-style population update logic.

This file is one of the most important in the generative spine because it defines how anchor influence and endogenous continuation are combined over time.

The key mechanism is a **rolling replacement process**.

At each step:

- a fraction of the mutable population is replaced
- replacement texts are generated from a mixture source
- the mixture is controlled by a time-dependent schedule, usually through `alpha`

This is not a model of every text mutating continuously. It is a population-turnover process.

That is an important conceptual distinction.

---

## Anchor and mutable split

One of the most defining choices in PAM generation is the split between:

- **anchor texts**
- **mutable texts**

The anchor set remains fixed and acts as a stable reference population.

The mutable portion evolves over time through replacement and resampling.

This means the system is never just a homogeneous recursive population. It is always a tension between:

- persistent anchor influence
- endogenous continuation from recent mutable history

This split is central to the meaning of the quench.

---

## Alpha schedule as control law

The parameter `alpha` should be read as a control law, not merely as a blend knob.

At each step, the engine applies an `alpha` schedule that determines how much replacement mass is drawn from:

- anchor-conditioned generation
- recent self-resampling or self-conditioned generation

So `alpha(t)` controls the balance between:

- exogenous stabilizing or steering influence
- endogenous self-continuation

This is the heart of the quench interpretation.

A PAM quench is therefore best understood as a scheduled modulation of anchor-vs-self influence over an evolving corpus population.

---

## Source window as finite memory

The mutable source pool is not drawn from the entire history of the run.

Instead, the engine typically uses a recent trailing window of mutable texts.

This gives the generative system an explicit **finite-memory mechanism**.

That means the endogenous component is not a full-history recursion. It is a recency-weighted continuation process.

This is one of the key reasons later observatory analyses of temporal depth and compression have real generative grounding.

The runtime itself already contains a memory design.

---

## `injectors.py`: semantic injection layer

`src/pam/injectors.py` is the bridge between the raw generative loop and semantic or invariant-aware control.

This layer is where the generative spine becomes distinctively PAM-like.

Injectors can:

- sample from anchor material
- sample from endogenous pools
- mutate candidate texts
- target or filter by TIP signatures
- participate in the alpha-controlled mixture law

So the injector layer determines not only **where texts come from**, but also **what kind of semantic structure is preserved or targeted during generation**.

---

## TIP-conditioned injection

One of the strongest features of the generative spine is that injection can be conditioned on invariant-signature structure.

This means mutation or anchor-based generation is not purely blind.

Instead, candidate texts can be evaluated against TIP signatures and filtered or accepted according to target signature classes.

Conceptually, this is a major step.

It means the generative process is not only:

- lexical mutation
- or random resampling

It can also be:

- invariant-conditioned
- signature-aware
- semantically constrained

This is one of the places where the measurement layer and the generative layer connect directly.

---

## `dynamics/`: local text variation toolkit

The `src/pam/dynamics/` package contains the actual local text-variation mechanisms used by the runtime.

This layer defines **what kinds of change are allowed**.

Typical functions include:

- self-resampling
- controlled mutation
- sentence reordering
- clause shuffling
- small lexical substitution
- light framing or “lens” changes

The important point is that PAM dynamics are intentionally conservative.

They do not try to produce arbitrarily wild rewrites. They aim to create bounded local perturbations while preserving enough continuity for geometric and observatory structure to emerge.

---

## Conservative transforms

The transform library is small and careful by design.

This is not a generic text-generation engine. It is a controlled perturbation toolkit.

That choice matters scientifically.

If the transforms were too unconstrained, later observatory structure would be harder to interpret. By keeping transforms conservative, the repository preserves a clearer relation between:

- source ancestry
- semantic continuity
- and observed structural transition

This is part of why later geometry and phase results remain meaningful.

---

## `dynamics/mutation.py`: compositional local perturbation

Mutation in PAM is not a full rewrite of a text and not merely a single token edit.

Instead, a mutation is usually a small composition of safe transforms.

That gives the dynamics an intermediate scale:

- enough change to allow regime motion
- enough continuity to preserve traceability

This scale is important.

It keeps the system in the regime where iterative structure can form without collapsing immediately into arbitrary noise.

---

## `dynamics/generators.py`: endogenous continuation

The endogenous side of the mixture law is intentionally simple.

The self-resampling generator draws from the recent mutable source pool.

This means the self component acts as a persistence mechanism:

- it preserves local continuity
- it propagates recently realized forms
- it acts as the endogenous inheritance channel of the population

This makes the two sides of the mixture asymmetrical in a useful way:

- anchor side: structured, potentially signature-aware perturbation
- self side: endogenous persistence and continuation

That asymmetry is one of the core generative facts of PAM.

---

## `corpora.py`: corpus construction and source surfaces

`src/pam/corpora.py` provides the corpus-facing inputs used by the generative spine.

This layer determines the source textual material from which runs are built.

Its role is straightforward but important:

- define corpus construction surfaces
- provide named source populations
- stabilize what counts as anchor material and mutable material at run start

This means the generative spine is not only about how texts change. It also depends on what initial source surfaces exist to be changed.

---

## Top-level façades

The repository also exposes simplified public entrypoints such as:

- `pam.engine`
- `pam.dynamics_mixture`
- `pam.injectors`

These modules act as façades over the deeper generative implementation.

Their purpose is to provide usable public handles without forcing callers to understand the entire internal structure of the runtime.

This mirrors the broader repository design:

- deep internal structure where needed
- simple top-level entrypoints where possible

---

## Generative contract

Taken together, the generative spine follows a clear contract:

1. construct a starting corpus
2. split it into anchor and mutable components
3. define an update law through mixture and dynamics
4. apply a time-dependent control schedule
5. record snapshots and optional state labels
6. emit run-facing outputs for later observatory derivation

This contract is what turns PAM into a controlled recursive language-system generator rather than a static analysis tool.

---

## What the generative spine is not

It is helpful to state clearly what this layer is **not**.

The generative spine is not:

- the observatory pipeline
- the phase / seam layer
- the operator layer
- the topology layer
- the TUI
- the full conceptual theory of PAM

It is the substrate that produces evolving corpora and run traces from which those later layers derive their objects.

---

## Why this architecture is good

The generative spine has several strengths.

### It is lean

The runtime remains small and legible rather than over-absorbing observatory semantics.

### It is interpretable

The main generative ingredients are explicit:
- anchor influence
- endogenous continuation
- finite-memory source windows
- conservative local transforms

### It is extensible

New macrostate readers and downstream observatory layers can be added without rewriting the engine.

### It is scientifically honest

It keeps a clean distinction between:
- what is generated
- how it is generated
- and how it is later interpreted

That distinction is one of the main reasons the repository remains conceptually coherent.

---

## Relationship to the observatory

The observatory builds on top of the generative spine.

The generative spine produces:
- evolving corpora
- trajectory-facing traces
- run-facing outputs

The observatory spine then builds:
- geometry
- phase structure
- operator fields
- topology
- interface-facing artifacts

This ordering matters.

The observatory should not define the generative substrate retroactively. It should derive its objects from the substrate’s outputs.

That is the proper direction of dependency.

---

## Summary

The generative spine is the runtime substrate of PAM.

Its key parts are:

- `engine/` for execution
- `mixture.py` for quench-style update laws
- `injectors.py` for anchor/self and signature-aware generation
- `dynamics/` for conservative local text variation
- `corpora.py` for source corpus construction

The central principle is:

**PAM generates evolving corpora through a controlled mixture of anchor influence and endogenous continuation, then lets the observatory derive higher-order structure from those runs.**

That principle explains how the repository remains both generative and observational without collapsing the two into one opaque system.
