# PAM Observatory TUI v2 — Canonical Design Spec

## Purpose

Define the next canonical terminal observatory for PAM.

The interface must reflect the actual scientific architecture of the repository:

- one shared manifold
- multiple derived layers
- one selected location
- multiple structural lenses

The TUI is not a dashboard. It is an instrument for inspecting a manifold through geometry, phase, topology, operators, and identity.

---

## Core principle

> The manifold designs the interface.

The observatory should be organized around a single selected manifold state and a set of mode-dependent field overlays.

This replaces the earlier batch-monitor-first design with a manifold-inspection-first design, while preserving live run monitoring as an explicit mode.

---

## System invariant

Every major interface view is a different lens on the same underlying manifold.

That means:

- the selected state is shared across modes
- all overlays refer to the same `(r, alpha)` node or its MDS embedding
- geometry, phase, topology, operators, and identity are not separate dashboards
- they are fields on one common object

---

## Primary interaction model

### Selection

The interface always maintains one active selected node.

Canonical selected coordinates:

- `r`
- `alpha`

Derived selected coordinates:

- `node_id`
- `i`, `j`
- `mds1`, `mds2` when available

All panes respond to this shared selection.

### View space

The central view should support two coordinate systems:

1. **Grid view**
   - direct `(r, alpha)` parameter lattice
   - best for sweep coverage and discrete field inspection

2. **MDS view**
   - intrinsic manifold embedding
   - best for geometric and structural interpretation

The user can toggle between them at any time.

### Lens / overlay model

The central manifold pane always displays one chosen field overlay.

Examples:

- coverage
- curvature
- seam distance
- signed phase
- criticality
- Lazarus
- transition rate
- identity magnitude
- absolute holonomy
- unsigned local obstruction
- signed local obstruction
- legacy spin

The interface should make it easy to switch overlays without changing the selected node.

---

## Top-level modes

The TUI should expose the following canonical modes.

### 1. Run

Purpose:
- live execution monitoring
- coverage tracking
- selected-run detail

Primary overlays:
- seed coverage
- run completion
- missing trajectories

Center pane:
- parameter grid coverage view

Detail pane:
- selected run / selected cell summary
- optional trajectory mini-plots

This mode preserves the spirit of the legacy TUI.

### 2. Geometry

Purpose:
- inspect the intrinsic geometry of the manifold

Primary overlays:
- `fim_det`
- `scalar_curvature`
- `fim_cond`
- geodesic / local structural metrics

Center pane:
- grid or MDS manifold colored by chosen geometric scalar

Detail pane:
- selected node geometry summary:
  - `fim_rr`
  - `fim_ra`
  - `fim_aa`
  - determinant
  - curvature
  - anisotropy / condition number

### 3. Phase

Purpose:
- inspect phase organization and seam structure

Primary overlays:
- signed phase
- distance to seam
- seam mask / seam corridor

Center pane:
- phase field on grid or MDS
- seam overlay where available

Detail pane:
- signed phase value
- seam distance
- local phase interpretation
- seam-adjacent status

### 4. Topology

Purpose:
- inspect structural organization beyond raw geometry

Primary overlays:
- criticality
- field/topology summaries
- local organizational patch statistics

Center pane:
- criticality or topology-derived structural overlays

Detail pane:
- selected node local topology summary
- neighborhood organization
- critical / seam / stable composition

### 5. Operators

Purpose:
- inspect how the manifold behaves under intervention

Primary overlays:
- Lazarus score
- transition rate
- operator response / probe intensity

Center pane:
- operator-derived field overlays on grid or MDS

Detail pane:
- selected node operator summary:
  - Lazarus
  - transition risk
  - local operator-conditioned metrics
  - probe-path availability

### 6. Identity

Purpose:
- inspect local structural identity, transport, and obstruction

Primary overlays:
- identity magnitude
- absolute holonomy
- unsigned local obstruction
- signed local obstruction
- legacy spin (comparison only)

Center pane:
- identity/transport/obstruction fields on grid or MDS
- singularity / high-obstruction overlays where relevant

Detail pane:
- selected node identity summary:
  - `identity_magnitude`
  - `obstruction_mean_abs_holonomy`
  - `obstruction_max_abs_holonomy`
  - `obstruction_signed_sum_holonomy`
  - `obstruction_signed_weighted_holonomy`
  - `identity_spin` (comparison only)

This mode must be first-class from the start.

---

## Identity geometry stack

The identity layer now resolves into four ordered levels:

1. **Metric layer**
   - identity distance
   - local identity metric
   - identity magnitude

2. **Transport layer**
   - path composition
   - loop-based holonomy residual

3. **Local obstruction layer**
   - unsigned local obstruction from incident holonomy
   - signed local obstruction from oriented incident holonomy

4. **Legacy proxy layer**
   - chart-sensitive node-based spin approximation
   - retained for comparison only

The TUI should reflect this hierarchy directly.

---

## Canonical layout

The next TUI should use a stable 3-region structure.

### A. Left inspector column

Persistent across all modes.

Contains:

- current mode
- current view (`Grid` / `MDS`)
- selected node:
  - `r`
  - `alpha`
  - `node_id`
- core derived values:
  - curvature
  - seam distance
  - criticality
  - Lazarus
  - identity magnitude
  - absolute holonomy / obstruction summaries
  - signed local obstruction
  - legacy spin (comparison only)
- refresh status / data freshness

This is the observatory spine.

### B. Center manifold pane

Primary visual object.

Shows:

- grid view or MDS view
- one overlay at a time
- optional contour / seam / singularity overlays
- selection highlight

This pane is the heart of the interface.

### C. Right detail pane

Context-sensitive.

Possible content:

- node metrics table
- local patch summary
- transport / holonomy summaries
- obstruction summaries
- operator summary
- selected trajectory sparkline block
- artifact availability
- nearest-neighbor summary

### D. Footer strip

Persistent controls / status.

Contains:

- keybindings
- refresh timer
- dataset / output roots
- current overlay
- current cursor mode
- optional event messages

---

## Selection semantics

### Grid selection

Arrow keys move the selected node across `(i, j)` / `(r, alpha)`.

### MDS selection

When in MDS view:
- selection still maps to the nearest existing node
- cursor moves node-to-node, not pixel-to-pixel

### Cross-view persistence

Switching between Grid and MDS must preserve the same selected node.

---

## Overlay semantics

Every overlay must be bound to one of the canonical artifact families.

### Geometry overlays
- determinant
- curvature
- condition number

### Phase overlays
- signed phase
- seam distance

### Topology overlays
- criticality
- local topology class

### Operator overlays
- Lazarus
- transition rate

### Identity overlays
- identity magnitude
- absolute holonomy
- unsigned local obstruction
- signed local obstruction
- legacy spin (comparison)

The overlay switch must not change mode by itself. Mode selects the conceptual family; overlay selects the field within that family.

---

## Detail-pane content by mode

### Run detail
- seed coverage for selected cell
- latest run values
- optional ASCII trajectory mini-plots

### Geometry detail
- local metric tensor summary
- curvature
- determinant
- nearest geometric neighbors

### Phase detail
- signed phase
- seam distance
- seam membership / seam corridor status

### Topology detail
- criticality
- local patch composition
- neighborhood class counts

### Operators detail
- Lazarus
- transition-rate summary
- selected operator path metadata if available

### Identity detail
- identity magnitude
- holonomy / transport summaries
- unsigned local obstruction
- signed local obstruction
- legacy spin as comparison
- local identity neighborhood composition

---

## Priority visual products for the center pane

### Grid renderings
- heatmap-style discrete field map
- optional marker overlay for selected node
- optional seam / singularity / high-obstruction marker overlays

### MDS renderings
- scatter manifold colored by chosen field
- optional seam / obstruction overlays
- selection highlight

Avoid overloading the pane with too many simultaneous encodings.

---

## Recommended keybindings

### Mode switching
- `1` → Run
- `2` → Geometry
- `3` → Phase
- `4` → Topology
- `5` → Operators
- `6` → Identity

### View / overlay
- `g` → toggle Grid / MDS
- `o` → cycle overlays within mode
- `O` → reverse cycle overlays

### Selection
- arrow keys → move selected node
- `h/j/k/l` optionally mirror arrows

### Detail / context
- `d` → expand/collapse detail pane
- `t` → trajectory/detail lens where applicable
- `s` → singularity / obstruction table view
- `p` → operator path / probe view where applicable

### Refresh / utility
- `r` → manual refresh
- `f` → freeze/unfreeze auto-refresh
- `?` → help
- `q` → quit

---

## Data contract expectations

Canonical artifact families include:

- `outputs/index.csv`
- `outputs/fim_*`
- `outputs/fim_phase/*`
- `outputs/fim_critical/*`
- `outputs/fim_ops*/*`
- `outputs/fim_identity/*`
- `outputs/fim_identity_holonomy/*`
- `outputs/fim_identity_obstruction/*`

The interface must remain file-first.

---

## Initial implementation scope for TUI v2

### Must-have
- persistent selected node
- mode system
- Grid/MDS toggle
- center manifold pane
- left inspector pane
- right detail pane
- Run / Geometry / Phase / Topology / Operators / Identity modes
- basic overlay switching

### Nice-to-have later
- embedded trajectory mini-plots
- obstruction ranking table view
- operator path overlays
- neighborhood graph rendering
- artifact browser
- screenshot/export system

---

## Canonical non-goals

The TUI should not become:

- a general-purpose dashboard
- a web-like chart grid
- a replacement for offline publication figures
- an in-memory analysis notebook

Its job is:
- live scientific inspection
- structural comparison across fields
- local manifold interrogation

---

## Relationship to future GUI

Any future GUI should inherit the same ontology:

- same modes
- same selected-node semantics
- same Grid/MDS duality
- same overlay model
- same transport-founded identity treatment

The GUI should be an implementation variant, not a conceptual redesign.

---

## Summary

The next canonical PAM observatory should be a manifold-centered instrument with:

- one selected node
- one shared manifold
- multiple structural lenses
- persistent local inspection
- first-class identity support grounded in transport and obstruction

In short:

> The observatory should no longer be organized around runs alone.
> It should be organized around the manifold that the runs reveal, with identity mode centered on holonomy and transport-derived local obstruction rather than legacy spin.
