# 057 — Multiscale Diffusion Robustness Design

Status: design note  
Role: future robustness sandbox, not canonical replacement

## Purpose

This note records the scale-space idea before implementation.

The goal is not to smooth plots. The goal is to test whether PAM structural objects persist across scale.

Core question:

```text
Which PAM structures persist across scale,
and which are local texture?
```

## Architectural Placement

Scale-space should sit as an optional robustness branch after raw observable extraction.

```text
[Raw Observables X]
        │
        ├── Canonical Pipeline
        │      → Fisher metric
        │      → core geometry
        │      → phase / seams / attractors / operators
        │
        └── Robustness Sandbox
               → graph diffusion over observables
               → X(t)
               → rebuild geometry at each scale
               → persistence summaries
```

This branch must not replace the canonical pipeline until it has earned that role.

## Field Choice Contract

First implementation choice:

```text
Diffuse raw observable vectors X over the graph manifold,
then rebuild the Fisher geometry from X(t).
```

Let:

```text
X ∈ R^(N × D)
```

where rows are parameter nodes and columns are observables.

For each observable dimension d, diffuse the field across the graph:

```math
∂x_d(t) / ∂t = -L_sym x_d(t)
```

This produces a scale-parameterized observable matrix:

```math
X(t)
```

The standard Fisher / geometry pipeline is then rerun on each X(t).

## Graph Weighting Contract

Use a local adaptive Gaussian kernel rather than raw inverse distance.

For nodes i and j:

```math
W_ij = exp(-d_Fisher(i,j)^2 / (2 σ_i σ_j))
```

where j is in the k-nearest neighborhood of i.

Default provisional settings:

```text
k = 15
local scale σ_i = Fisher distance from i to its 7th nearest neighbor
```

This avoids near-zero inverse-distance instability and adapts to dense / sparse regions.

## Operator Contract

Use the symmetric normalized graph Laplacian:

```math
L_sym = I - D^(-1/2) W D^(-1/2)
```

Diffusion is computed by:

```math
X(t) = exp(-t L_sym) X
```

For interactive tools, precompute a scale ladder rather than computing matrix exponentials at runtime.

## Scale Ladder

Provisional default:

```text
M = 8 scale levels
t_min = 0.1
t_max = 10.0
geometric spacing
```

```math
t_m = t_min (t_max / t_min)^(m/(M-1))
```

## Recomputed Objects at Each Scale

At each scale level t_m:

- recompute scale-dependent geometry D(t_m)
- recompute embedding if needed
- recompute seams / phase structures if applicable
- recompute attractor basins if applicable
- track Lazarus concentration dynamics if available
- compare route-family / gateway / canonical structures where compatible

## Persistence Questions

### Seam Persistence

```text
Does the seam remain localized across t?
```

Possible metric:

```math
J_S(t_m) = |S(t_m) ∩ S(t_{m-1})| / |S(t_m) ∪ S(t_{m-1})|
```

### Attractor Basin Persistence

```text
Which attractor basins vanish quickly under diffusion?
Which persist across scale?
```

Interpretation:

```text
shallow basin:
  collapses at low t

canonical anchor basin:
  persists across large t
```

### Lazarus Concentration Dynamics

```text
Does diffusion sharpen Lazarus peaks at intermediate scales,
or wash them out?
```

This distinguishes micro-scale noise from macro-scale structural concentration.

### Gateway / Canonical Persistence

```text
Do route-family, gateway, and canonical-family structures survive scale diffusion?
```

This should be treated carefully because symbolic structures may not transfer directly across all geometry rebuilds.

## Visual Intuition

Cellular automata imagery provided a useful analogy:

```text
local rule
→ repeated transformation
→ nested motifs
→ seams / patches / defects
→ apparent scale-depth
```

The imported question is not “PAM is fractal.”

The imported question is:

```text
Which observed structures are scale-persistent,
and which are local texture?
```

## Guardrails

- Do not smooth plots and call it scale-space.
- Do not diffuse coordinates in arbitrary Euclidean embedding space.
- Do not promote scale-space into the canonical pipeline before validation.
- Be explicit that this is scale-space over the observable field, not a unique scale-space of PAM in the abstract.

## Minimal First Implementation

A first implementation should output:

```text
outputs/scale_space/<corpus_or_campaign>/
  scale_ladder.json
  graph_affinity_summary.csv
  observables_t*.csv or .npz
  geometry_t*/...
  persistence_summary.csv
  scale_space_report.md
```

## Resulting Research Role

Scale-space becomes a robustness instrument:

```text
canonical pipeline finds structures;
scale-space tests whether they persist across diffusion scale.
```
