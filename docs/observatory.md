Observatory

The PAM Observatory is a manifold-centered research instrument for inspecting the system through a shared parameter space and a set of structural lenses.

It is not primarily a batch monitor, and it is not a collection of unrelated dashboards.

Its core assumption is simple:
	•	there is one manifold
	•	every major field lives on that manifold
	•	one selected node should remain stable across views
	•	different modes are different ways of interrogating the same object

In practice, the observatory lets you move between execution state, geometry, phase structure, topology, operator response, and identity transport while keeping the same underlying location in focus.

⸻

Conceptual model

The observatory is organized around a single selected state on the parameter manifold.

That selected state can be viewed in two coordinate systems:
	•	Grid view: the direct (r, α) sweep lattice
	•	MDS view: the intrinsic low-dimensional embedding of the manifold

These are not different datasets. They are two views of the same manifold state.

The rest of the observatory is built around this invariant:
	•	the selected node is shared
	•	overlays change what you see, not what node you are on
	•	ranking, markers, and snapshots all refer back to the same manifold state

This is what makes the observatory useful as an instrument rather than merely a display.

⸻

The six canonical modes

1. Run

Run mode shows execution and coverage across the sweep lattice.

It answers questions like:
	•	which parameter cells are populated?
	•	how much run coverage exists here?
	•	where are there gaps or missing regions?

This is the operational lens. It is closest to the legacy monitoring role of the system, but in the observatory it is now integrated into the same selection model as the scientific views.

Use Run mode to sanity-check dataset completeness and confirm which parts of the manifold are actually represented in the outputs.

⸻

2. Geometry

Geometry mode shows intrinsic geometric structure on the manifold.

Typical overlays include:
	•	scalar curvature
	•	Fisher metric determinant
	•	Fisher condition number

It answers questions like:
	•	where is the manifold sharply curved?
	•	where does the metric become compressed or distorted?
	•	where are geometric anomalies or structurally sensitive regions?

This is the mode to use when the question is about the shape of the manifold itself rather than the phase or transport structure that lives on it.

⸻

3. Phase

Phase mode shows large-scale regime organization.

Typical overlays include:
	•	signed phase
	•	distance to seam

It answers questions like:
	•	where are the major phase regions?
	•	how close is this node to the phase boundary?
	•	how does the seam organize the manifold?

This is often the best first scientific lens after Geometry mode, because it reveals the broad regime structure that later topology, operator, and identity signals often organize around.

⸻

4. Topology

Topology mode shows structural organization beyond raw geometry.

Typical overlays include:
	•	criticality

It answers questions like:
	•	where are the structurally important nodes?
	•	where does local organization become especially unstable or transitional?
	•	where do topological signals concentrate?

Topology mode is useful when the question is not just “what is the manifold shaped like?” but “where is the manifold structurally decisive?”

⸻

5. Operators

Operators mode shows how the manifold behaves under intervention-related observables.

Typical overlays include:
	•	Lazarus score

It answers questions like:
	•	where do recovery-like signals appear?
	•	where are trajectories more sensitive to operator probing?
	•	where do intervention-associated responses concentrate?

This mode helps connect local structure to dynamical or operator-conditioned behavior.

⸻

6. Identity

Identity mode shows the transport-centered identity layer.

This is the most interpretively delicate mode and should be read with the refined identity hierarchy in mind.

Typical overlays include:
	•	identity magnitude
	•	absolute holonomy
	•	unsigned local obstruction
	•	signed local obstruction
	•	legacy spin

It answers questions like:
	•	where is identity changing strongly?
	•	where is transport obstruction concentrated?
	•	where does local directional obstruction change sign?
	•	how does the legacy spin proxy compare to the transport-derived obstruction layer?

Identity mode is not centered on spin anymore. It is centered on holonomy and transport-derived local obstruction, with spin retained only as a secondary comparison field.

That distinction is important.

⸻

Identity interpretation

Identity mode should be read according to the following hierarchy.

Primary layer
	•	identity magnitude
	•	absolute holonomy
	•	unsigned local obstruction
	•	signed local obstruction

Comparison layer
	•	legacy spin

The current observatory interpretation is:
	•	holonomy is the invariant obstruction object
	•	local obstruction is derived from transport
	•	signed local obstruction is the canonical local directional field
	•	legacy spin is a chart-sensitive historical comparison signal

This means:
	•	if you want the intrinsic obstruction story, look at holonomy and obstruction
	•	if you want to compare against the earlier proxy, look at legacy spin
	•	do not treat spin as the primary identity object anymore

In compact form:

holonomy is primary; local obstruction is derived; spin is comparison-only.

⸻

Core interaction model

The observatory is built around a small number of repeated interactions.

Selection

The observatory always maintains one active selected node.

That node is shared across all modes and both view spaces.

Moving the selection changes what is highlighted in the center pane and what is summarized in the inspector and detail panes.

Grid / MDS toggle

You can switch between:
	•	Grid view
	•	MDS view

without changing the underlying selected node.

This is one of the central affordances of the observatory.

Grid view is often better for sweep-complete structural inspection.
MDS view is often better for geometric intuition and manifold-level pattern reading.

Overlay cycling

Each mode contains one or more overlays.

Changing overlay changes the field being shown, but not the selection or the underlying mode.

This lets you compare multiple related signals within one conceptual lens.

Ranking pane

The ranking pane provides a compact top-N list for the active overlay.

It answers questions like:
	•	what are the strongest nodes right now?
	•	where should I look next?
	•	what dominates this field?

This complements the spatial map by giving a fast scalar ranking view.

Jump-to-ranked-node

From ranking mode, you can move the ranking cursor and jump directly to a ranked node.

This closes the loop between:
	•	field visualization
	•	ranking
	•	local inspection

Marker overlays

Marker overlays let you place a second structural layer on top of the current field.

Current marker types include:
	•	seam
	•	critical
	•	obstruction
	•	Lazarus

These are useful for comparative reading, for example:
	•	signed phase with seam markers
	•	curvature with critical markers
	•	identity obstruction with seam markers
	•	Lazarus with critical markers

Snapshot export

Snapshots write a lightweight markdown record of the current observatory state.

They capture:
	•	mode
	•	overlay
	•	marker mode
	•	view space
	•	selected node
	•	selected summary
	•	ranking context

This makes the observatory usable as part of a research workflow, not only as a live interface.

⸻

Recommended workflows

1. Inspect strong geometric regions

Start in Geometry mode with scalar curvature.
Open the ranking pane.
Jump to the top-ranked nodes.
Toggle to MDS view to understand how the strongest geometric regions sit in the embedded manifold.

2. Compare phase and seam structure

Go to Phase mode.
Switch between signed phase and distance to seam.
Turn on seam markers.
Move between Grid and MDS to understand both lattice position and manifold organization.

3. Inspect critical structural nodes

Go to Topology mode.
Use the criticality overlay.
Open ranking and jump to the strongest nodes.
Then compare those same locations in Geometry or Phase mode.

4. Study operator-sensitive regions

Go to Operators mode.
Inspect Lazarus.
Use ranking to find the strongest nodes.
Then compare those nodes against seam markers or critical markers.

5. Read the identity layer correctly

Go to Identity mode.
Start with signed local obstruction, not legacy spin.
Then inspect:
	•	absolute holonomy
	•	unsigned local obstruction
	•	identity magnitude

Only after that, compare against legacy spin.

This preserves the intended transport-centered interpretation.

6. Capture a useful observatory state

Once you find an interesting manifold region or comparison:
	•	choose your mode
	•	choose your overlay
	•	choose markers if helpful
	•	jump to a relevant ranked node
	•	export a snapshot

That gives you a stable research artifact you can refer to later.

⸻

Keybindings

The exact bindings may evolve slightly, but the current observatory interaction set is centered on:
	•	1-6 — switch modes
	•	g — toggle Grid / MDS
	•	o — next overlay
	•	O — previous overlay
	•	m — cycle marker mode
	•	arrow keys — move selection
	•	s — toggle detail / ranking pane
	•	j / k — move ranking cursor
	•	enter — jump to ranked node
	•	x — export snapshot
	•	r — refresh
	•	f — freeze / unfreeze refresh
	•	q — quit

The most important loop is:
	•	select lens
	•	inspect field
	•	rank nodes
	•	jump to node
	•	compare in Grid / MDS
	•	snapshot if useful

⸻

Reading the center pane

The center pane is the heart of the observatory.

A few interpretation notes help:

Unsigned fields

Unsigned overlays encode magnitude only.

These should be read as:
	•	low → high intensity
	•	presence/absence where relevant
	•	concentration or strength

Signed fields

Signed overlays encode both sign and magnitude.

These should be read as:
	•	one side of the color scale for negative values
	•	the other side for positive values
	•	stronger intensity for larger magnitude

The most important signed fields currently are:
	•	signed phase
	•	signed local obstruction
	•	legacy spin

Selection vs markers

Selection always remains dominant.
Markers are secondary overlays for comparison.

So:
	•	selected node answers “where am I?”
	•	markers answer “what else should I compare against here?”

⸻

File-first architecture

The observatory is file-first.

It reads canonical outputs from outputs/ rather than maintaining its own hidden analytical state.

This is important for reproducibility.

The observatory is not a separate analysis engine.
It is a live instrument over the canonical artifact stack.

That means:
	•	if outputs change, the observatory can refresh
	•	if a field exists canonically in the pipeline, the observatory should read it from that artifact family
	•	if an artifact is missing, the observatory should degrade gracefully rather than inventing structure

This is one of the reasons the observatory and the research pipeline remain aligned.

⸻

What the observatory is for

The observatory is best used when you want to think with the manifold.

It is especially good for:
	•	local inspection of interesting nodes
	•	cross-comparison between manifold layers
	•	locating extreme or structurally important regions
	•	comparing Grid and MDS interpretations
	•	reading the identity transport layer without losing the rest of the system
	•	capturing meaningful observatory states as snapshots

It is not meant to replace:
	•	publication figures
	•	batch analysis scripts
	•	notebooks
	•	downstream statistical analysis

It is the live interpretive layer that sits between raw outputs and formal writeup.

⸻

Current status

The observatory now supports the first full canonical scientific stack:
	•	Run
	•	Geometry
	•	Phase
	•	Topology
	•	Operators
	•	Identity

with:
	•	manifold-centered selection
	•	Grid / MDS duality
	•	ranking and jump workflows
	•	marker overlays
	•	snapshot export
	•	transport-centered Identity interpretation

That is the first complete observatory checkpoint.

⸻

One-line summary

The PAM Observatory is a manifold-centered instrument for moving between geometry, phase, topology, operators, and identity while keeping one shared node in view and reading every field as structure on the same underlying object.
