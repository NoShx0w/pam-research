# Observatory Spine

## Purpose

This document describes the **observatory spine** of the PAM repository.

The observatory spine is the file-first architectural layer that turns generated runs into stable, inspectable scientific artifacts. It is the part of the repository that takes the output of experiments and builds the observatory surface used by geometry, phase, operators, topology, documentation, and the TUI.

The key modules are:

- `src/pam/types.py`
- `src/pam/io/paths.py`
- `src/pam/pipeline/state.py`
- `src/pam/pipeline/runner.py`
- `src/pam/pipeline/stages/*`

The central design fact is simple:

**the observatory is artifact-first.**

It is not primarily a large in-memory world model. It is a structured artifact graph rooted in repository paths, built stage by stage, and surfaced for later scientific inspection.

---

## The observatory as a file-first system

The repository distinguishes between two related things:

- the **generative substrate**, which produces runs and raw trajectory-facing outputs
- the **observatory spine**, which turns those outputs into stable derived layers

The observatory spine is therefore not the simulation itself. It is the canonical build path that constructs scientific objects such as:

- Fisher geometry
- distance graphs
- MDS embeddings
- phase coordinates
- seam-local observables
- operator-derived fields
- topology-facing summaries

This architecture matters because PAM is not organized as a monolithic application object. Its durable scientific state lives primarily in artifacts on disk.

That design has several consequences:

- scientific truth is anchored in generated files
- observatory layers can be refreshed independently
- downstream tools such as the TUI can read canonical outputs without re-deriving the full system
- the build order of scientific layers becomes explicit and reviewable

---

## `types.py`: minimal run contracts

`src/pam/types.py` defines the small foundational contracts used to move generated results through the broader repository.

The important point is not the complexity of these types, but their restraint.

The run-facing types do **not** attempt to encode the whole observatory. They define a minimal contract for:

- run parameters
- run results
- lightweight exchange between generation and downstream processing

This is a deliberate design choice.

PAM does not pretend that the full scientific state of the observatory can be captured by one giant Python object. Instead:

- generated outputs are produced
- those outputs are written to disk
- the observatory spine consumes them and builds a richer artifact surface

So `types.py` should be read as the minimal run boundary, not as the repository’s full ontology.

---

## `io/paths.py`: filesystem ontology

`src/pam/io/paths.py` is one of the most important architectural files in the repository.

It defines the filesystem ontology of the observatory.

Two path surfaces matter most:

### `OutputPaths`

`OutputPaths` represents the currently active legacy output layout under `outputs/`.

This is where many generated and derived artifacts still live today, including surfaces such as:

- run index and coverage artifacts
- Fisher metric outputs
- distance graph outputs
- MDS embeddings
- curvature surfaces
- phase outputs
- operator outputs

This surface is real, active, and still heavily used.

### `ObservatoryPaths`

`ObservatoryPaths` defines the more explicit canonical observatory layout.

This path system expresses the intended stable repository-facing organization of:

- corpora
- runs
- derived geometry
- derived phase
- derived topology
- derived operators
- figures
- reports

This means `io/paths.py` is not just a utility helper. It is a statement of architectural intent.

It tells us that the repository is in a transitional but coherent state:

- `outputs/` remains the main active artifact surface
- `observatory/` names the canonical structure the repository is moving toward

This is why artifact promotion from `outputs/` into `observatory/` is an architectural task rather than just a cleanup task.

---

## `pipeline/state.py`: the observatory contract

`src/pam/pipeline/state.py` defines the object that moves through the observatory pipeline.

The important object here is `PipelineState`.

Conceptually, `PipelineState` is **not** a scientific world model. It is a compact contract containing:

- root paths
- observatory path references
- lightweight metadata
- stage-to-stage context

This means the pipeline carries:

- where artifacts live
- what stage has been built
- what metadata has been accumulated

rather than a giant live representation of the manifold.

That is exactly the right design for a file-first observatory.

### Metadata accumulation

A particularly good design feature is that metadata is accumulated incrementally rather than by mutating a hidden global state object.

This makes the pipeline easier to reason about:

- each stage consumes a known state
- writes its artifact family
- returns an updated state with additional metadata

This gives the observatory spine a clean compositional form.

### Why this matters

The stage contract becomes:

**consume file-backed state → build one derived observatory layer → return updated state**

That is the real operational contract of the observatory.

---

## `pipeline/runner.py`: canonical stage order

`src/pam/pipeline/runner.py` is the central ordering document of the observatory spine.

This file matters because it defines the canonical scientific sequence in which observatory layers are built.

The current order is:

1. geometry
2. phase
3. operators
4. topology

This order is not incidental.

It encodes the repository’s current scientific worldview:

### Geometry first

The observatory first builds the intrinsic manifold substrate:
- metric
- distance
- embedding
- curvature
- geodesic structure

### Phase second

Once geometry exists, the observatory constructs:
- signed phase
- seam structure
- distance to seam
- phase-local coordinates

### Operators third

With the manifold and seam in place, the observatory can define controlled probes and derived operator fields, such as:
- Lazarus
- transition-rate diagnostics
- probe-family structures

### Topology fourth

Only after geometry, phase, and operator layers exist does the observatory move to deeper structural organization:
- criticality
- field alignment
- identity structure
- obstruction-like organization

This ordering is one of the clearest expressions of PAM as a layered scientific instrument rather than a bag of scripts.

---

## `pipeline/stages/*`: scientific choreography

The files in `src/pam/pipeline/stages/` are best understood as **scientific choreography**.

They are not where the deepest mathematics lives. The deeper algorithms usually live in:
- `geometry/`
- `phase/`
- `operators/`
- `topology/`

The stage files do something different and equally important:

- select canonical producers
- call them in the right order
- write artifacts
- preserve stage-level metadata and path structure

So the stage files are the place where the repository states:

- what counts as a buildable observatory layer
- when it is allowed to be built
- what it depends on
- where it should write

That means these files are architectural, even when they look thin.

### Why they drift easily

Stage files often drift because they sit between:
- evolving code
- evolving artifact surfaces
- evolving scientific interpretation

That is why they deserve explicit documentation and should not be treated as trivial wrappers.

---

## The observatory build contract

Taken together, the observatory spine follows a simple contract:

1. accept generated, file-backed scientific inputs
2. resolve canonical repository paths
3. build one derived layer at a time
4. write stable artifacts
5. accumulate metadata
6. expose those artifacts to later layers and tools

This contract gives the observatory its distinctive character.

PAM is not only:
- a simulator
- a metrics library
- a figure generator

It is a layered observatory builder.

---

## What the observatory spine is not

It is helpful to state clearly what this layer is **not**.

The observatory spine is not:

- the generative engine
- the text-dynamics law
- the TUI itself
- a monolithic stateful app
- the full conceptual theory of PAM

Instead, it is the file-first layer that makes all of those later readings possible by building the stable observatory artifact graph.

---

## Relationship to the generative spine

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

This split is one of the most important facts about the repository.

It means PAM cleanly separates:
- generation
from
- observation

That separation is one of the strongest design choices in the codebase.

---

## Relationship to the TUI

The TUI sits downstream of the observatory spine.

It does not define observatory truth. It reads observatory truth from generated artifacts.

This is why the TUI can function as a serious scientific inspection console:
- it is not inventing structure
- it is loading the output of a canonical layered build process

The observatory spine therefore provides the artifact surface that the TUI inspects.

---

## Current transition state

The repository is currently in a partially transitional but coherent state.

In practice:

- many canonical read-path artifacts still live in `outputs/`
- the architectural intent points toward `observatory/`
- the code already knows about both surfaces
- future artifact promotion is part of architectural completion

So the observatory spine should be understood as both:
- a current working build system
- and a migration path toward a clearer canonical observatory artifact surface

---

## Why this architecture is good

The observatory spine has several strengths.

### It is inspectable

Scientific objects become files, tables, and surfaces that can be checked directly.

### It is compositional

Each stage builds on earlier ones rather than hiding work inside a monolith.

### It is interface-friendly

Tools like the TUI can inspect file-backed truth rather than re-running the science.

### It is documentation-friendly

The architecture can be described as stable layered contracts rather than a mass of implicit runtime behavior.

### It is scientifically honest

It keeps a clear distinction between:
- generated substrate
- derived observatory object
- interface representation

That honesty is one of the reasons PAM works as an observatory rather than only as a simulation project.

---

## Summary

The observatory spine is the file-first architectural layer that builds PAM’s canonical scientific surface.

Its key parts are:

- `types.py` for minimal run contracts
- `io/paths.py` for filesystem ontology
- `pipeline/state.py` for stage-to-stage observatory state
- `pipeline/runner.py` for canonical scientific ordering
- `pipeline/stages/*` for layered artifact construction

The central principle is:

**the observatory is built as a sequence of derived artifact layers, not as a single in-memory world model.**

That principle explains how PAM turns generated runs into a navigable scientific observatory.
