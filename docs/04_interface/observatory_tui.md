PAM Observatory TUI

Purpose

The PAM Observatory TUI is the live, file-backed inspection console for the observatory.

It is not a simulation dashboard and not a decorative viewer. Its role is to let a user inspect derived observatory artifacts across scientific layers while keeping one shared selected node stable across multiple interpretive lenses.

The TUI exists to support manifold inspection, overlay comparison, structural ranking, node-level detail reading, and file-faithful observatory navigation.

Position in the repository

The TUI is the operational observatory surface.

It sits downstream of the canonical artifact pipeline:

* engine/ and dynamics/ generate runs
* pipeline/ and src/pam/ produce observatory artifacts
* the TUI loads those artifacts and exposes them for inspection

The TUI does not define scientific truth. It reads truth from file-backed observatory outputs.

Design principles

File-backed truth

The TUI reads from generated artifacts and does not invent its own hidden data model.

One shared selected node

All panes refer to the same selected lattice cell or manifold node.

Layered observatory modes

Modes correspond to scientific layers of the observatory, not arbitrary UI tabs.

Overlay switching as epistemic lensing

Changing overlays changes how the same manifold position is interpreted.

No false smoothness

The TUI preserves sparse, discrete structure rather than visually faking continuity.

Comparative inspection

Markers, overlays, and ranking allow one field to be read against another.

Signed and unsigned fields are distinct

Signed fields are rendered as oriented structure, not generic magnitude heatmaps.

Core interaction model

The TUI is organized around a shared state:

* mode — current scientific layer
* overlay — active scalar field within that layer
* selection — currently selected node
* view space — grid or mds
* marker mode — optional second-order structural highlights
* right pane mode — detail or ranking

This means the TUI is fundamentally organized around stable position and changing interpretation, rather than changing page, changing object, and changing context.

Scientific modes

The current TUI supports these observatory layers.

Run

Run coverage and sweep completion structure.

Typical questions:

* what has run?
* what is missing?
* where is coverage concentrated?

Geometry

Intrinsic manifold structure.

Typical questions:

* where is curvature high?
* where is the metric stressed?
* how does geometry organize the state space?

Phase

Phase-coordinate and seam structure.

Typical questions:

* where is the seam?
* how far is this node from the seam?
* how does the phase field partition the manifold?

Topology

Topological or structural concentration layer.

Typical questions:

* where are critical regions?
* how do structural objects localize?

Operators

Probe-derived regime structure.

Typical questions:

* where is Lazarus active?
* where do operator-defined response regions localize?

Identity

Identity, transport, and obstruction layer.

Typical questions:

* how strong is local identity change?
* where is absolute holonomy high?
* where does obstruction localize?
* how do canonical identity objects compare with legacy spin?

View spaces

Grid view

Grid view displays observatory structure over the sampled parameter lattice.

It is best for:

* local adjacency
* block structure
* corridor reading
* lattice-faithful inspection

MDS view

MDS view displays observatory structure in the canonical low-dimensional manifold chart.

It is best for:

* intrinsic geometric organization
* seam-relative structure
* manifold-wide shape
* embedding-based comparison

The same selected node persists across both spaces.

Overlays

Each mode contains one or more overlays.

An overlay is the active field being rendered for the current mode.

Examples include:

* run coverage
* curvature
* distance to seam
* criticality
* Lazarus
* identity magnitude
* absolute holonomy
* unsigned obstruction
* signed obstruction
* legacy spin

The overlay defines what scalar field is being shown, whether it is signed or unsigned, and what semantic cue appears in the detail pane.

Signed and unsigned field rendering

The TUI distinguishes between unsigned overlays and signed overlays.

Unsigned overlays

Unsigned overlays encode magnitude only.

Examples include:

* coverage
* curvature
* criticality
* Lazarus
* absolute holonomy
* unsigned obstruction

These should be read as low-to-high intensity or concentration.

Signed overlays

Signed overlays encode both polarity and magnitude.

Examples include:

* signed phase
* signed obstruction

These should be read as oriented structural organization, not merely intensity.

Blue↔red renderings indicate sign-bearing field structure rather than generic heatmap coloration.

Markers

Markers are optional second-order highlights layered on top of the active overlay.

Typical marker families include:

* seam
* critical
* obstruction
* Lazarus

Markers exist to support comparative inspection.

They allow the user to inspect one field while marking another structural object, without collapsing both into a single visual encoding.

This is especially useful for:

* seam vs field overlays
* holonomy vs obstruction
* Lazarus vs seam-adjacent regions

Panes

Inspector pane

The inspector pane is the compact operational summary.

It typically shows:

* mode
* view
* overlay
* selected node
* grid indices
* compact numeric summary
* refresh state
* status text

Manifold pane

The manifold pane is the primary spatial inspection surface.

It shows:

* the active overlay in grid or MDS space
* the selected node
* optional markers
* sparse structural rendering

Detail pane

The detail pane is the interpretive local report for the selected node.

It typically includes:

* selected node identifiers
* local observatory values
* active overlay
* field cue
* semantic meaning
* mode-specific local ontology

Ranking pane

The ranking pane is the alternative right-pane mode listing nodes ranked by the current overlay.

It is used for:

* discovering high-value or extreme nodes
* jumping selection to ranked nodes
* probing structural hotspots

Footer

The footer is the interaction summary and current mode / overlay status line.

It should be treated as the authoritative interaction cue surface.

Ranking as navigational probe

Ranking is not merely a summary table.

It is a navigational probe.

The ranking view lets the user sort nodes by the active field, inspect extremes, jump selection to those nodes, and compare local detail against global field structure.

For signed fields, ranking is typically by absolute magnitude. For unsigned fields, ranking is by raw magnitude.

Detail semantics and field cues

The detail pane does not only display numbers.

It also explains how the active field should be read.

Field cues may include:

* overlay label
* signed / unsigned type
* rendering convention
* meaning of the field

Examples include:

* run coverage across the sweep lattice
* node-local absolute transport obstruction summary
* signed local transport-derived obstruction

This makes the TUI a self-interpreting instrument panel rather than a raw numerical dump.

Identity mode and canonical hierarchy

Identity mode is special because it supports comparison between current canonical identity objects and earlier proxy objects.

The current hierarchy is:

* primary: identity magnitude / holonomy / obstruction
* comparison: legacy spin

This lets the TUI render successive canonicalization:

* the newer preferred object
* alongside the historical comparison object
* while keeping the current hierarchy explicit

This is especially important for:

* absolute holonomy
* unsigned obstruction
* signed obstruction
* legacy spin comparison

File-backed artifact sources

The TUI loads file-backed observatory artifacts from generated repository outputs.

Typical sources include:

* run index / coverage tables
* MDS coordinates
* curvature surfaces
* signed phase coordinates
* distance-to-seam surfaces
* criticality summaries
* Lazarus summaries
* identity / holonomy / obstruction outputs

The TUI should be understood as artifact-loading, normalization, and inspection, not as recomputation of observatory science inside the interface.

Current limitations

The current TUI is intentionally conservative.

Known limits include:

* the trajectory lens is not yet fully integrated into the main manifold view
* manifold connectivity is still more node-centric than web-centric
* canonical artifact promotion from outputs/ to observatory/ is still incomplete
* the interface trails the science by design and therefore does not always surface the latest research object immediately

These limits are intentional consequences of keeping science first, canonicalization second, and interface third.

Design lineage

The current TUI emerged in stages.

Early concept sketch

A manifold-centered observatory vision with seam, probes, network structure, and log semantics.

Legacy batch monitor

A live sweep and trajectory monitor focused on:

* run status
* seed coverage
* phase diagram
* selected trajectory detail

Current observatory console

A layered, file-backed scientific inspection surface organized by:

* mode
* overlay
* selection
* grid/MDS space
* markers
* detail and ranking

The current TUI should be understood as the mature operational continuation of that lineage.

Why the TUI matters

The TUI is not part of the canonical scientific theory, but it is part of canonical observatory practice.

Its role is to let users enter the manifold, hold one node stable, compare structural lenses, read local semantic cues, and inspect file-backed observatory truth without false smoothing or abstraction loss.

In that sense, the TUI is the observatory’s live console.

Key interactions

Typical controls include:

* mode switching
* grid/MDS switching
* overlay cycling
* marker cycling
* node movement
* detail/ranking toggle
* ranking navigation
* rank jump
* snapshot export
* refresh / freeze
* quit

The footer should always be treated as the authoritative interaction cue surface.

Future directions

Important future directions include:

* richer manifold web / edge representation
* seam-thread and path overlays
* tighter canonical artifact loading from observatory/
* deeper trajectory inspection
* improved route-family and response-flow integration

The TUI should evolve carefully, preserving sparse inspectability, file-backed truth, semantic explicitness, and mode/overlay discipline.
