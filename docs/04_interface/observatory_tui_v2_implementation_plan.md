# PAM Observatory TUI v2 — Implementation Plan

## Purpose

Define a practical, staged implementation path for the next canonical PAM observatory TUI.

The implementation must preserve the file-first architecture and reflect the repository’s scientific ontology:

- one shared manifold
- multiple derived layers
- one selected location
- multiple structural lenses

The TUI is not a dashboard. It is an instrument for inspecting the manifold through geometry, phase, topology, operators, and identity.

---

## Guiding principle

> The manifold designs the interface.

The implementation should proceed from shared state and shared manifold rendering outward, rather than from isolated panels inward.

That means:

- selection is global
- overlays are mode-specific lenses on the same object
- all panes derive from the same selected manifold node
- artifact files remain the source of truth

---

## Phase 0 — Freeze the data contract

### Goal

Define exactly which artifact files TUI v2 reads in each mode.

### Deliverables

Create a stable loader contract mapping:

- **Run**
  - `outputs/index.csv`
  - manifest / validation outputs
  - selected trajectory files
- **Geometry**
  - `outputs/fim/*`
  - `outputs/fim_mds/*`
  - `outputs/fim_curvature/*`
- **Phase**
  - `outputs/fim_phase/*`
- **Topology**
  - `outputs/fim_critical/*`
  - future topology summaries
- **Operators**
  - `outputs/fim_lazarus/*`
  - `outputs/fim_transition_rate/*`
  - `outputs/fim_ops*/*`
- **Identity**
  - `outputs/fim_identity/*`
  - `outputs/fim_identity_maps/*`
  - `outputs/fim_identity_diagnostics/*`
  - `outputs/fim_identity_singularity_overlay/*`

### Why first

This prevents the interface from inventing hidden state, bypassing the artifact graph, or drifting away from the repository’s architecture.

### Acceptance

- every mode has an explicit file contract
- missing optional artifacts are identified in advance
- no mode depends on hidden in-memory-only scientific state

---

## Phase 1 — Build the shared observatory skeleton

### Goal

Implement the stable frame of the instrument before adding heavy scientific mode logic.

### Core pieces

- app shell
- persistent left inspector
- center manifold pane
- right detail pane
- footer strip
- shared selected-node state
- mode state
- overlay state
- Grid/MDS toggle
- refresh state

### Suggested modules

- `observatory/app.py`
- `observatory/state.py`
- `observatory/loaders.py`
- `observatory/views/layout.py`
- `observatory/views/inspector.py`
- `observatory/views/manifold.py`
- `observatory/views/detail.py`
- `observatory/views/footer.py`

### Shared state model

Minimum state should include:

- `mode`
- `view_space` = `grid` or `mds`
- `overlay`
- `selected_node_id`
- `selected_r`
- `selected_alpha`
- `selected_i`
- `selected_j`
- refresh status
- artifact snapshot timestamps / mtimes

### Acceptance

- app launches successfully
- user can switch modes
- user can move selection on the grid
- left pane updates from selection
- center pane renders a placeholder or basic grid
- right pane shows selected-node metadata
- footer shows controls and status

This is the first true milestone.

---

## Phase 2 — Implement the manifold renderer

### Goal

Make the center pane reflect the shared manifold directly.

### Step 2A — Grid view first

Start with the parameter lattice.

Support:

- scalar heatmap rendering in terminal cells
- selected-node highlight
- optional markers for seam / singularities later

### Why grid first

- it matches the native sweep structure
- selection is discrete and robust
- color mapping is straightforward
- it supports Run, Geometry, Phase, Topology, Operators, and Identity immediately

### Step 2B — MDS view second

Once grid rendering is stable, add the intrinsic manifold embedding.

Support:

- scatter-like node rendering in terminal coordinates
- nearest-node selection
- field coloring by overlay
- optional seam / singularity overlays

### Acceptance

- toggling Grid/MDS preserves selected node
- same node can be inspected in both coordinate systems
- overlays render correctly in both views

---

## Phase 3 — Implement mode-specific loaders and overlays

Build mode support one conceptual family at a time.

### 3.1 Run mode

#### Reads

- `outputs/index.csv`
- manifest / integrity outputs
- selected trajectory artifact if present

#### Shows

- coverage heatmap
- run status summary
- selected cell summary
- optional mini trajectory display

#### Acceptance

- provides parity or improvement relative to the legacy run monitor

---

### 3.2 Geometry mode

#### Reads

- `outputs/fim/fim_surface.csv`
- `outputs/fim_mds/mds_coords.csv`
- `outputs/fim_curvature/curvature_surface.csv`

#### Overlays

- determinant
- curvature
- condition number

#### Detail pane

- local metric tensor summary
- determinant
- curvature
- anisotropy / condition number

---

### 3.3 Phase mode

#### Reads

- `outputs/fim_phase/signed_phase_coords.csv`
- `outputs/fim_phase/phase_distance_to_seam.csv`
- seam artifact

#### Overlays

- signed phase
- distance to seam

#### Detail pane

- signed phase
- seam distance
- seam-adjacent status
- local phase interpretation

---

### 3.4 Topology mode

#### Reads

- `outputs/fim_critical/criticality_surface.csv`
- future topology summaries

#### Overlays

- criticality
- topology summaries as they stabilize

#### Detail pane

- selected node local topology summary
- neighborhood organization
- critical / seam / stable composition

---

### 3.5 Operators mode

#### Reads

- `outputs/fim_lazarus/lazarus_scores.csv`
- `outputs/fim_transition_rate/*`
- operator summaries and path metadata

#### Overlays

- Lazarus
- transition rate
- operator response summaries

#### Detail pane

- local operator metrics
- transition likelihood
- probe/path availability

---

### 3.6 Identity mode

#### Reads

- `outputs/fim_identity/identity_field_nodes.csv`
- `outputs/fim_identity_maps/identity_maps_nodes.csv`
- singularity tables / overlays where available

#### Overlays

- identity magnitude
- identity spin

#### Detail pane

- `identity_vx`
- `identity_vy`
- `identity_magnitude`
- `identity_spin`
- patch counts:
  - `patch_n_nodes`
  - `patch_n_edges`
  - `patch_n_seam`
  - `patch_n_critical`
  - `patch_n_stable`

#### Acceptance

Identity must be treated as a first-class mode from the start, not bolted on later.

---

## Phase 4 — Add selection-aware detail lenses

### Goal

Make the right pane scientifically useful rather than merely descriptive.

### Per selected node, expose

- canonical coordinates:
  - `node_id`
  - `r`
  - `alpha`
  - `i`
  - `j`
  - `mds1`
  - `mds2` if available
- mode-relevant metrics
- local patch or neighborhood summaries
- artifact availability
- ranked local context when relevant

### Stable node synopsis block

Across all modes, maintain a compact node synopsis:

- curvature
- seam distance
- criticality
- Lazarus
- identity magnitude
- identity spin

This becomes the observatory’s persistent local chart.

### Acceptance

- every mode has a structured, non-chaotic detail pane
- selected-node interpretation feels mode-specific but selection-consistent

---

## Phase 5 — Add lightweight interaction upgrades

### Goal

Improve usability without destabilizing the observatory.

### Add

- overlay cycling
- singularity table view
- optional trajectory lens for Run / Operators
- help screen
- manual refresh and refresh-freeze
- optional selection jump commands later

### Avoid early

- mouse-dependent logic
- overly animated transitions
- large collections of tiny subpanels
- web-dashboard style clutter

### Acceptance

- interactions remain fast and legible
- the interface still feels like an instrument, not a dashboard

---

## Recommended implementation order

### Milestone 1

- app shell
- shared state
- 3-pane layout
- footer
- grid selection

### Milestone 2

- Run mode parity with legacy TUI

### Milestone 3

- Geometry + Phase modes on grid view

### Milestone 4

- MDS view support

### Milestone 5

- Topology + Operators modes

### Milestone 6

- Identity mode

### Milestone 7

- singularity overlays
- richer detail pane
- contextual tables / ranking views

This order gets the instrument usable early while aligning implementation with the scientific stack.

---

## Suggested code architecture

### State

One central app state object.

Responsible for:

- selection
- active mode
- active overlay
- active view space
- refresh / freeze state
- loader snapshots

### Loaders

One loader family per artifact family.

Responsible for:

- file reads
- mtime-aware refresh
- schema normalization
- graceful handling of missing optional outputs

### Views

Pure rendering functions from:

- `(state, loaded_data) -> panel content`

### Mode controllers

Each mode defines:

- default overlay
- required artifact families
- detail-pane schema
- optional mode-specific actions

This keeps the UI modular and faithful to the architecture.

---

## Testing strategy

### 1. Loader tests

Verify each loader reads current artifact files successfully.

### 2. Selection tests

Verify moving selection updates all panes consistently.

### 3. Mode tests

Verify each mode renders or degrades gracefully if optional files are absent.

### 4. Cross-view tests

Verify Grid/MDS toggle preserves selected node.

### 5. Identity tests

Verify identity overlays and singularity views work when identity artifacts exist.

### 6. Refresh tests

Verify file updates appear correctly after refresh without restarting the app.

---

## Recommended initial PR sequence

### PR A — observatory shell

- app state
- 3-pane layout
- footer
- selection model
- mode switching scaffolding

### PR B — loaders + Run mode

- migrate the useful core of the legacy monitor into the new shell

### PR C — Geometry / Phase mode

- first real manifold-centered scientific inspection views

### PR D — MDS support

- preserve shared selection across Grid and MDS

### PR E — Topology / Operators mode

- expose criticality, Lazarus, transition summaries

### PR F — Identity mode

- identity magnitude
- identity spin
- singularity highlighting

This sequence keeps the observatory useful at every stage.

---

## Key design discipline

At every implementation step, ask:

> does this feature help inspect one shared manifold through multiple structural lenses?

If yes, it belongs.

If it feels like “just another useful panel,” it probably does not belong in the canonical observatory.

---

## Canonical non-goals

The TUI should not become:

- a generic dashboard
- a replacement for offline paper figures
- a catch-all log browser
- an in-memory analysis notebook

Its purpose is:

- live scientific inspection
- structural comparison across fields
- local manifold interrogation
- ontology-aligned observation

---

## Relationship to future GUI

Any future GUI should inherit this same conceptual model:

- same modes
- same selected-node semantics
- same Grid/MDS duality
- same overlay logic
- same first-class identity treatment

The GUI should be an implementation variant, not a conceptual redesign.

---

## Summary

The next canonical PAM observatory should be implemented as a manifold-centered instrument with:

- one selected node
- one shared manifold
- multiple structural lenses
- persistent local inspection
- first-class identity support

In short:

> The implementation should move from shell → manifold → modes → identity, while preserving the file-first instrument architecture at every step.
