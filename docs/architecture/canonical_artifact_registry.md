# Canonical Artifact Registry

## Purpose

This document defines the first-pass canonical artifact registry for the PAM Observatory.

Its role is to move the repository out of the long-running `outputs/` ↔ `observatory/` transition phase by making explicit:

- which artifact families are first-class observatory surfaces
- which stage or script produces them
- which tools consume them
- what status they currently have
- and where they should canonically live

This is not a request to migrate every historical file immediately. It is a registry of artifact truth.

The main question behind this document is:

**which artifacts should the observatory now trust as canonical scientific surfaces?**

---

## Status classes

Use these status labels consistently.

### `canonical`

A first-class observatory artifact.

A canonical artifact has:

- stable scientific meaning
- a clear producing stage or script
- a path contract that downstream tools are allowed to trust
- likely consumers in the TUI, docs, downstream pipeline stages, or stable analysis code

### `intermediate`

A useful pipeline artifact, but not a primary public observatory surface.

An intermediate artifact may:

- be needed between stages
- support downstream derivation
- remain important operationally

but it does not need to be foregrounded as part of the stable observatory surface.

### `study`

A study-specific artifact.

These are valuable scientific outputs, but they belong to:

- one experiment line
- one OBS series
- one staging workflow
- or one temporary analysis thread

They should not automatically become part of the core observatory contract.

### `legacy`

A historical or compatibility artifact.

Legacy artifacts may still be read by older scripts or interface code, but they should not determine the future canonical structure of the repository.

---

## Canonicalization principles

The registry follows a few simple rules.

### 1. Scientific meaning comes first

An artifact should become canonical because it represents a stable observatory object, not because it already exists in a convenient folder.

### 2. Stage ownership must be clear

Every canonical artifact should have a clear owning producer:
- geometry
- phase
- operators
- topology
- or an explicitly named pre-stage dependency

### 3. Consumers should not guess

The TUI, documentation, and later-stage code should not have to discover canonical truth by scanning ad hoc output trees.

### 4. Compatibility may remain temporarily

Early canonicalization passes may still mirror or preserve legacy `outputs/` paths for compatibility. That is acceptable during transition.

### 5. Not every useful file becomes canonical

Plots, study snapshots, debug summaries, and one-off diagnostics may remain intermediate or study artifacts even when they are scientifically useful.

---

## Registry table

| family | artifact(s) | scientific meaning | producer stage / script | current path(s) | consumed by | status | target canonical path | compatibility notes |
|---|---|---|---|---|---|---|---|---|

---

## Geometry

| family | artifact(s) | scientific meaning | producer stage / script | current path(s) | consumed by | status | target canonical path | compatibility notes |
|---|---|---|---|---|---|---|---|---|
| `geometry.metric_surface` | `fim_surface.csv`, metadata, plots | Fisher / response metric surface over control lattice | geometry stage / FIM scripts | `outputs/fim/` | docs, analysis, derived geometry workflows | canonical | `observatory/geometry/metric/` | keep legacy mirror initially |
| `geometry.graph_nodes` | `fisher_nodes.csv` | canonical manifold node table | geometry stage | `outputs/fim_distance/` | phase, topology, TUI | canonical | `observatory/geometry/graph/nodes.csv` | very high-priority surface |
| `geometry.graph_edges` | `fisher_edges.csv` | canonical manifold edge graph | geometry stage | `outputs/fim_distance/` | geodesic logic, topology, TUI later | canonical | `observatory/geometry/graph/edges.csv` | very high-priority surface |
| `geometry.distance_matrix` | `fisher_distance_matrix.csv` | intrinsic graph geodesic distance matrix | geometry stage | `outputs/fim_distance/` | MDS, analysis, topology | canonical | `observatory/geometry/distance/distance_matrix.csv` | large but canonical |
| `geometry.mds_coords` | `mds_coords.csv` | low-dimensional intrinsic manifold chart | geometry stage | `outputs/fim_mds/` | TUI, figures, docs | canonical | `observatory/geometry/mds/coords.csv` | extremely high-priority surface |
| `geometry.curvature` | curvature csv/plots | curvature proxy over manifold | geometry stage | `outputs/fim_curvature/` | TUI, docs, analysis | canonical | `observatory/geometry/curvature/` | tables and figures may split |
| `geometry.geodesic_paths` | path metadata, path plots | sampled geodesic traces and diagnostics | geometry stage / geodesic scripts | `outputs/fim_distance/`, `outputs/fim_geodesic*` | analysis, figures | intermediate | `observatory/geometry/geodesics/` | not every derivative file needs promotion |

---

## Phase

| family | artifact(s) | scientific meaning | producer stage / script | current path(s) | consumed by | status | target canonical path | compatibility notes |
|---|---|---|---|---|---|---|---|---|
| `phase.boundary` | `phase_boundary*.csv`, plots | seam / phase-boundary object | phase stage | `outputs/fim_phase/` | TUI, docs, figures | canonical | `observatory/phase/boundary/` | boundary nodes and rendered forms may split |
| `phase.distance_to_seam` | `phase_distance_to_seam.csv` | intrinsic distance-to-boundary coordinate | phase stage | `outputs/fim_phase/` | TUI, operators, transition-law work | canonical | `observatory/phase/distance_to_seam.csv` | very high-priority surface |
| `phase.signed_phase` | `signed_phase_coords.csv` | signed phase coordinate field | phase stage | `outputs/fim_phase/` | TUI, docs, figures | canonical | `observatory/phase/signed_phase.csv` | very high-priority surface |
| `phase.phase_plots` | signed-phase and seam plots | rendered inspection surfaces | phase stage | `outputs/fim_phase/` | docs, figures | intermediate | `observatory/phase/figures/` | figure promotion can remain selective |

---

## Operators

| family | artifact(s) | scientific meaning | producer stage / script | current path(s) | consumed by | status | target canonical path | compatibility notes |
|---|---|---|---|---|---|---|---|---|
| `operators.lazarus_scores` | `lazarus_scores.csv`, summary | canonical trajectory-level compression observable | operator stage | `outputs/fim_lazarus/` | TUI, docs, transition-law work | canonical | `observatory/operators/lazarus/` | high-priority surface |
| `operators.operator_S` | `operator_S_*` | operator-S probe outputs | operator stage / legacy scripts | `outputs/fim_ops/` | analysis | intermediate | `observatory/operators/operator_S/` | may remain partly legacy |
| `operators.canonical_probes` | `canonical_probes.csv`, paths, summary | canonical operator probe surface | operator stage | `outputs/fim_ops/` | docs, analysis | canonical | `observatory/operators/canonical_probes/` | verify which files are truly canonical |
| `operators.scaled_probes` | `scaled_probe_*` | scale-conditioned probe line | operator stage | `outputs/fim_ops_scaled/` | analysis, predictive tests | intermediate | `observatory/operators/scaled_probes/` | not first-pass TUI-facing |
| `operators.transition_rate` | `transition_rate_*` | finite-horizon transition-rate artifacts | operator stage | `outputs/fim_transition_rate/` | docs, analysis | canonical | `observatory/operators/transition_rate/` | stable public observatory surface |
| `operators.field_alignment` | field alignment csv/plots | Lazarus vs seam / ordering relation | operator stage | `outputs/fim_field_alignment/` | analysis, docs | intermediate | `observatory/operators/field_alignment/` | secondary public importance |
| `operators.gradient_alignment` | gradient alignment csv/plots | gradient / field alignment diagnostics | operator stage | `outputs/fim_gradient_alignment/` | analysis, docs | intermediate | `observatory/operators/gradient_alignment/` | secondary public importance |

---

## Topology

| family | artifact(s) | scientific meaning | producer stage / script | current path(s) | consumed by | status | target canonical path | compatibility notes |
|---|---|---|---|---|---|---|---|---|
| `topology.criticality` | `criticality_surface.csv`, `critical_points.csv` | canonical criticality layer | topology stage | `outputs/fim_critical/` | TUI, docs, figures | canonical | `observatory/topology/criticality/` | high-priority surface |
| `topology.initial_conditions` | `initial_conditions_outcome_summary.csv`, related outputs | initial-condition outcome summary and basin/outcome structure | dedicated prerequisite script / future explicit pipeline step | `outputs/fim_initial_conditions/` | topology organization, analysis | canonical | `observatory/topology/initial_conditions/` | must be wired into pipeline explicitly |
| `topology.organization` | organization summaries / plots | route-family / manifold organization layer | topology stage | current topology / initial-conditions-adjacent outputs | docs, analysis | canonical | `observatory/topology/organization/` | depends on stable initial-condition ownership |
| `topology.identity_core` | identity node summaries, identity magnitude, absolute holonomy, obstruction fields | canonical identity / transport / obstruction layer | topology stage | current topology outputs | TUI, docs, analysis | canonical | `observatory/topology/identity/` | extremely high-priority surface |
| `topology.identity_proxy` | proxy identity graph summaries | lower-cost identity intermediates | topology stage | current topology outputs | topology internals | intermediate | `observatory/topology/identity/proxy/` | not necessarily public |
| `topology.hubs_hotspots` | hub lists, hotspot summaries, obstruction hotspot summaries | structural node classes for observatory and TUI | topology stage or OBS scripts | scattered in `outputs/` and study surfaces | TUI, figures | canonical | `observatory/topology/annotations/` | important precondition for TUI manifold upgrade |
| `topology.family_bundles` | family path bundles, route-family node chains | family as route object | OBS studies / family scripts | scattered in `outputs/` / study folders | future TUI, figures, docs | study (promote later) | `observatory/topology/families/` | likely second-pass canonicalization |

---

## Interface-facing artifacts

| family | artifact(s) | scientific meaning | producer stage / script | current path(s) | consumed by | status | target canonical path | compatibility notes |
|---|---|---|---|---|---|---|---|---|
| `interface.tui_loader_surface` | minimal node / field tables trusted by TUI | canonical TUI read surface | derived from multiple stages | mixed `outputs/*` reads | TUI | canonical | `observatory/interface/tui/` or direct per-family canonical reads | must decide whether to consolidate or keep per-family reading |
| `interface.snapshots` | TUI or observatory plate snapshots | interface-facing observatory captures | TUI / plotting tools | ad hoc | docs, figures | intermediate | `observatory/interface/snapshots/` | later-phase cleanup |

---

## Study-specific families

| family | artifact(s) | scientific meaning | producer stage / script | current path(s) | consumed by | status | target canonical path | compatibility notes |
|---|---|---|---|---|---|---|---|---|
| `studies.obs044_047` | continuous flow artifacts | continuous response-flow reconstruction line | `experiments/studies/obs044*` etc. | study-specific outputs | docs, analysis | study | remain under study outputs for now | promote only stabilized reference summaries later |
| `studies.obs048` | recoverability artifacts | distributed family recoverability | `obs0xx_route_family_recoverability.py` | `outputs/obs0xx_route_family_recoverability/` | docs, analysis | study | remain study-specific for now | summary promotion possible later |
| `studies.obs024_bundle_plate` | bundle-rendered observatory plate outputs | family / seam / hotspot visual synthesis | OBS figure scripts | figure outputs | docs, future TUI ontology | study | not yet canonical artifact | ontology should inform TUI later |

---

## Immediate decisions to resolve

The following questions need explicit answers during Pass 1.

### A. Which surfaces are canonical right now?

At minimum, the following should be treated as first-pass canonical surfaces:

- manifold node table
- manifold edge graph
- graph geodesic distance matrix
- MDS coordinates
- signed phase
- phase boundary / seam
- distance to seam
- Lazarus scores / summary
- criticality surface / critical points
- initial-conditions outcome summary
- identity magnitude / absolute holonomy / obstruction surfaces

### B. Which surfaces need stable filename contracts?

Every canonical surface should have:

- a canonical filename
- a canonical folder
- an owning stage
- and a clear consumer contract

### C. Which surfaces does the TUI trust now or later?

This determines which artifacts must be stabilized first.

High-priority TUI-facing families include:

- MDS coordinates
- signed phase
- distance to seam
- identity magnitude / holonomy / obstruction
- hub / hotspot annotation summaries
- later: family bundle summaries

### D. Which families need compatibility mirrors?

Early canonicalization can preserve old `outputs/` paths temporarily while also writing to canonical `observatory/` locations.

That is acceptable during transition and may be the safest first-pass strategy.

---

## Recommended first-pass targets

### Pass 1

Promote these first:

- geometry graph nodes / edges / distances
- MDS coordinates
- signed phase
- phase boundary / seam
- distance to seam
- Lazarus scores / summary
- criticality surface / critical points
- initial-conditions outcome summary
- identity magnitude / absolute holonomy / obstruction surfaces

### Pass 2

Promote these next:

- hubs / hotspots / structural annotation files
- family bundle surfaces
- operator probe surfaces beyond Lazarus and transition rate
- field alignment / gradient alignment summaries

### Pass 3

Later:

- stabilized study-derived reference products
- TUI-oriented consolidated read surfaces
- snapshot / plate surfaces
- historical compatibility cleanup

---

## First-pass PR shape

### Proposed title

`Canonicalize first-pass observatory artifact surfaces`

### Proposed scope

- define canonical locations for core geometry / phase / topology artifact families
- wire or mirror producers there
- preserve `outputs/` compatibility for now
- do not migrate every study output yet
- do not redesign the TUI yet

### Proposed goal

Exit the ambiguous artifact phase for the core observatory surfaces the pipeline and TUI most need to trust.

---

## Canonical reading

This registry should be read as the first formal statement of artifact truth in the observatory.

The observatory is now in a position where:

- full corpus-Cp coverage exists
- the full pipeline closes
- the TUI operates on full Cp outputs

That means the remaining work is no longer proof-of-closure work.

It is canonicalization work.

This document therefore marks the transition from:

- exploratory artifact accumulation

to:

- explicit observatory artifact governance

---

## Summary

The registry does not attempt to clean every output path immediately.

It does something more important first:

- it names the artifact families
- assigns them status
- identifies their producers and consumers
- and defines which ones are now ready to become first-class observatory surfaces
