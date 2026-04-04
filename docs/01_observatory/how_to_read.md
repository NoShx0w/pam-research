# How to Read the PAM Observatory

This document explains how to interpret the outputs of the PAM Observatory.

The observatory turns parameter sweeps and recursive experiments into a layered structural object. Different files and plots expose different aspects of that object.

---

## 1. The Control Manifold

The system is defined over a 2D control manifold:

```math
\theta = (r, \alpha)
```

where:

- **r** controls recursion / replacement strength
- **α** controls anchoring / mixture dynamics

Each parameter configuration contributes measurements that can later be aggregated into geometry, phase, operator, and topology structure.

---

## 2. Experiment Outputs (`outputs/index.csv`)

The first major ledger is:

```text
outputs/index.csv
```

This file records run-level experiment summaries.

In practice, rows typically correspond to specific run configurations, often including:

- corpus
- $(r)$
- $(\alpha)$
- seed
- summary observables derived from the run

These observables provide the measurable state from which the downstream observatory layers are built.

Examples include:

- freeze-related summaries
- entropy-related summaries
- lag / regression statistics
- other derived observables used by the geometry layer

See:

- `observable_glossary.md`

for definitions.

---

## 3. The Canonical Interpretation Stack

The observatory is best read as a layered system:

```text
experiments
↓
observables
↓
geometry
↓
phase
↓
operators
↓
topology
```

A useful shorthand is:

| Layer | Meaning |
|------|--------|
| Observables | what is measured from recursive runs |
| Geometry | how parameter states are intrinsically arranged |
| Phase | where behavioral regimes divide |
| Operators | how manifold paths behave under probing |
| Topology | how the field is structurally organized |

---

## 4. Geometry Outputs

The geometry layer constructs the manifold from observable behavior.

Important artifact families include:

- `outputs/fim/`
- `outputs/fim_distance/`
- `outputs/fim_mds/`
- `outputs/fim_curvature/`

### 4.1 Parameter heatmaps

Examples include:

- Fisher determinant
- curvature diagnostics
- seam-relative quantities

These are plotted over the raw \((r, \alpha)\) grid.

Use these to see where sensitivity, curvature, and transition structure concentrate in parameter space.

### 4.2 MDS manifold

The MDS embedding provides low-dimensional coordinates for the intrinsic manifold.

Interpretation:

- each point = one parameter configuration
- nearby points = behaviorally similar configurations under the intrinsic metric
- separated or folded regions = distinct manifold organization

### 4.3 Curvature

Curvature highlights regions where the intrinsic geometry changes rapidly.

Interpretation:

- high curvature → sensitive / transition-like regions
- low curvature → comparatively stable regions

---

## 5. Phase Outputs

The phase layer builds regime structure on top of geometry.

Important artifacts include:

- `outputs/fim_phase/phase_boundary_mds_backprojected.csv`
- `outputs/fim_phase/phase_distance_to_seam.csv`
- `outputs/fim_phase/signed_phase_coords.csv`
- `outputs/fim_phase/signed_phase_on_grid.png`
- `outputs/fim_phase/signed_phase_on_mds.png`

### 5.1 Signed phase

Each point is assigned a signed phase coordinate.

Interpretation:

- positive values → one side of the regime structure
- negative values → the other side
- values near zero → seam-adjacent / transition-like structure

### 5.2 Phase boundary (seam)

The seam is the boundary between phase regimes.

Interpretation:

- not imposed externally
- inferred from manifold structure
- central to transition and operator analysis

### 5.3 Distance to seam

Distance to seam measures how deeply a point lies within a regime or how close it is to the boundary.

Interpretation:

- small distance → boundary-adjacent
- large distance → regime interior

---

## 6. Operator Outputs

The observatory also supports active probing of the manifold.

Important artifacts include:

- `outputs/fim_ops/`
- `outputs/fim_ops_scaled/`
- `outputs/fim_lazarus/`
- `outputs/fim_transition_rate/`

These outputs describe how paths behave relative to seam structure and manifold constraints.

Key ideas:

- **canonical probes** compare different endpoint families
- **scaled probes** estimate transition structure over many sampled paths
- **Lazarus** identifies boundary-adjacent compression / pre-transition structure
- **transition rate** estimates short-horizon probability of phase change

When reading operator outputs, look for:

- seam graze
- seam crossing
- phase flip
- Lazarus exposure
- lag from compression to transition

---

## 7. Topology Outputs

The topology layer summarizes how the manifold-organized field is structurally put together.

Important artifacts include:

- `outputs/fim_field_alignment/`
- `outputs/fim_gradient_alignment/`
- `outputs/fim_critical/`
- `outputs/fim_initial_conditions/`

These help answer questions like:

- where does structural organization concentrate?
- how do different fields align?
- which regions behave like transition corridors?
- how are outcomes organized by initial condition and seam-relative structure?

---

## Identity Transport and Obstruction

The observatory now resolves an identity layer on top of the manifold.

This layer separates into three related parts:

- **identity magnitude** — local strength of structural identity change
- **identity transport / holonomy** — path dependence of structural identity transport around local loops
- **identity spin** — local obstruction signal associated with that path dependence

Operationally:
- identity magnitude behaves as a local metric-adjacent field
- identity spin is weakly explained by local metric structure
- loop-based holonomy provides finite-path confirmation that structural identity transport is not path-independent

This means the observatory now resolves not only where structure changes, but also where those changes fail to compose consistently across the manifold.

---

## 8. How to Read the Observatory as a Whole

A useful reading strategy is:

1. start with `outputs/index.csv` to understand measured observables  
2. inspect geometry to see intrinsic manifold structure  
3. inspect phase to locate seams and regime organization  
4. inspect operators to see how paths behave relative to the seam  
5. inspect topology to understand structural organization across the field  

In other words:

- geometry tells you what exists
- phase tells you where regimes divide
- operators tell you how the manifold behaves under probing
- topology tells you how the whole structure is organized

---

## 9. Key Insight

The PAM Observatory treats a parameter sweep not as a loose collection of runs, but as a structured object.

Rather than asking only:

> what happens at each parameter setting?

it asks:

> what is the intrinsic organization of the system across parameter space?

Phase structure, seam behavior, operator response, and topology are then understood as properties of that organization.
