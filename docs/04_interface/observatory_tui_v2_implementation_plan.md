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

The implementation should proceed from shared state and shared manifold rendering outward.

That means:

- selection is global
- overlays are mode-specific lenses on the same object
- all panes derive from the same selected manifold node
- artifact files remain the source of truth

---

## Phase 0 — Freeze the data contract

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
  - `outputs/fim_identity_holonomy/*`
  - `outputs/fim_identity_obstruction/*`

The identity loader contract should now explicitly prioritize:

1. holonomy
2. unsigned local obstruction
3. signed local obstruction
4. legacy spin

---

## Phase 1 — Build the shared observatory skeleton

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

---

## Phase 2 — Implement the manifold renderer

### Step 2A — Grid view first

Support:

- scalar heatmap rendering in terminal cells
- selected-node highlight
- optional markers for seam / obstruction later

### Step 2B — MDS view second

Support:

- scatter-like node rendering in terminal coordinates
- nearest-node selection
- field coloring by overlay
- optional seam / obstruction overlays

### Acceptance

- toggling Grid/MDS preserves selected node
- same node can be inspected in both coordinate systems
- overlays render correctly in both views

---

## Phase 3 — Implement mode-specific loaders and overlays

### 3.1 Run mode

Reads:
- `outputs/index.csv`
- manifest / integrity outputs
- selected trajectory artifact if present

Shows:
- coverage heatmap
- run status summary
- selected cell summary
- optional mini trajectory display

---

### 3.2 Geometry mode

Reads:
- `outputs/fim/fim_surface.csv`
- `outputs/fim_mds/mds_coords.csv`
- `outputs/fim_curvature/curvature_surface.csv`

Overlays:
- determinant
- curvature
- condition number

---

### 3.3 Phase mode

Reads:
- `outputs/fim_phase/signed_phase_coords.csv`
- `outputs/fim_phase/phase_distance_to_seam.csv`
- seam artifact

Overlays:
- signed phase
- distance to seam

---

### 3.4 Topology mode

Reads:
- `outputs/fim_critical/criticality_surface.csv`
- future topology summaries

Overlays:
- criticality
- topology summaries as they stabilize

---

### 3.5 Operators mode

Reads:
- `outputs/fim_lazarus/lazarus_scores.csv`
- `outputs/fim_transition_rate/*`
- operator summaries and path metadata

Overlays:
- Lazarus
- transition rate
- operator response summaries

---

### 3.6 Identity mode

Reads:
- `outputs/fim_identity/identity_field_nodes.csv`
- `outputs/fim_identity_holonomy/identity_holonomy_cells.csv`
- `outputs/fim_identity_obstruction/identity_obstruction_nodes.csv`
- `outputs/fim_identity_obstruction/identity_obstruction_signed_nodes.csv`

Primary overlays:
- `identity_magnitude`
- `abs_holonomy_residual` (cell-centered view)
- `obstruction_mean_abs_holonomy`
- `obstruction_max_abs_holonomy`
- `obstruction_signed_sum_holonomy`

Secondary / comparison overlays:
- `identity_spin`
- `obstruction_signed_weighted_holonomy`
- `obstruction_mean_holonomy`

Detail pane should prioritize:
- holonomy summaries
- unsigned local obstruction
- signed local obstruction
- legacy spin only as comparison

Acceptance:
Identity mode must be transport-centered, with spin clearly demoted to comparison status.

---

## Phase 4 — Add selection-aware detail lenses

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
- unsigned local obstruction
- signed local obstruction
- legacy spin (comparison)

---

## Phase 5 — Add lightweight interaction upgrades

Add:

- overlay cycling
- obstruction / holonomy table view
- optional trajectory lens for Run / Operators
- help screen
- manual refresh and refresh-freeze

Avoid early:

- mouse-dependent logic
- overly animated transitions
- large collections of tiny subpanels
- web-dashboard style clutter

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
- Identity mode with transport-centered overlays:
  - absolute holonomy
  - unsigned local obstruction
  - signed local obstruction
  - legacy spin comparison

### Milestone 7
- obstruction overlays
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

Identity loaders should expose both:
- node-centered obstruction summaries
- cell-centered holonomy summaries

### Views
Pure rendering functions from:
- `(state, loaded_data) -> panel content`

### Mode controllers
Each mode defines:
- default overlay
- required artifact families
- detail-pane schema
- optional mode-specific actions

Identity mode defaults should now prefer transport-derived fields over legacy spin.

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
Verify:
- holonomy overlays render
- unsigned obstruction overlays render
- signed obstruction overlays render
- legacy spin comparison renders when available

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
- absolute holonomy
- unsigned local obstruction
- signed local obstruction
- legacy spin as comparison only

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
- same transport-founded identity treatment

The GUI should be an implementation variant, not a conceptual redesign.

---

## Summary

The next canonical PAM observatory should be implemented as a manifold-centered instrument with:

- one selected node
- one shared manifold
- multiple structural lenses
- persistent local inspection
- first-class identity support grounded in transport and obstruction

In short:

> The implementation should move from shell → manifold → modes → transport-centered identity, while preserving the file-first instrument architecture at every step.
