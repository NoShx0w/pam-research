# Observatory Chain — OBS-028c to OBS-051

## Scope

This document defines the canonical reproducible observatory chain from the seam-bundle layer through the first predictive and first Lyapunov-like follow-up studies.

The chain covers:

- OBS-028c canonical seam bundle
- scale-conditioned family substrate construction
- OBS-022 scene-bundle preparation
- OBS-024 family hotspot occupancy
- OBS-050 structural coupling persistence
- OBS-051 local divergence in seam-coupled escalation windows

This document is intended to stabilize the observatory pipeline across machines and corpora.

---

## Current Status

### Canonical / stable enough to rely on
- OBS-028c
- scale family substrate build
- OBS-022 scene bundle preparation
- OBS-024 family hotspot occupancy
- OBS-050 structural coupling persistence

### Running and reproducible, but not yet canonically stable in interpretation
- OBS-051 local divergence in coupled windows

### Cross-corpus note
- OBS-050 reproduces qualitatively across corpus C and corpus Cp
- OBS-051 runs on both, but the dynamical conclusion is currently corpus-sensitive and should be treated as provisional

---

## Environment Notes

### Machine roles
- **MacBook**: corpus C, canonical smaller observatory baseline
- **Mac mini**: corpus Cp, larger corpus observatory regime

### General invocation pattern
Most commands assume:

```bash
PYTHONPATH=src .venv/bin/python <script.py>
```

On the Mac mini environment, equivalent project-local Python invocation is also acceptable if the virtual environment differs.

---

## Step 0 — Base Canonical Field Stack

### Purpose
Provide the canonical non-scale-local observatory fields used downstream by seam, substrate, and predictive studies.

### Required inputs
Expected to already exist from earlier observatory construction:
- `outputs/fim_distance/`
- `outputs/fim_mds/`
- `outputs/fim_phase/`
- `outputs/fim_identity/`
- `outputs/fim_identity_obstruction/`
- `outputs/fim_critical/`
- `outputs/fim_lazarus/`
- `outputs/fim_response_operator/`

### Required outputs
At minimum:
- `outputs/fim_distance/fisher_nodes.csv`
- `outputs/fim_distance/fisher_edges.csv`
- `outputs/fim_mds/mds_coords.csv`
- `outputs/fim_phase/signed_phase_coords.csv`
- `outputs/fim_phase/phase_distance_to_seam.csv`
- `outputs/fim_identity/identity_field_nodes.csv`
- `outputs/fim_identity_obstruction/identity_obstruction_nodes.csv`
- `outputs/fim_identity_obstruction/identity_obstruction_signed_nodes.csv`
- `outputs/fim_critical/criticality_surface.csv`
- `outputs/fim_lazarus/lazarus_scores.csv`
- `outputs/fim_response_operator/response_operator_nodes.csv`

### Verification
Confirm these files exist before continuing.

---

## Step 1 — OBS-028c Prerequisite Chain

### Purpose
Build the precursor studies needed for canonical seam-bundle export.

### Commands

```bash
PYTHONPATH=src .venv/bin/python experiments/studies/fim_response_complex_compatibility.py
PYTHONPATH=src .venv/bin/python experiments/studies/fim_response_operator_decomposition.py
PYTHONPATH=src .venv/bin/python experiments/studies/obs025_anisotropy_vs_relational_obstruction.py
PYTHONPATH=src .venv/bin/python experiments/studies/obs025_two_field_seam_panel.py
PYTHONPATH=src .venv/bin/python experiments/studies/obs026_family_two_field_occupancy.py
PYTHONPATH=src .venv/bin/python experiments/studies/obs028_embedding_comparison.py
PYTHONPATH=src .venv/bin/python experiments/studies/obs028b_diffusion_mode_analysis.py
```

### Expected outputs
- `outputs/fim_response_complex_compatibility/...`
- `outputs/fim_response_operator_decomposition/...`
- `outputs/obs025_anisotropy_vs_relational_obstruction/...`
- `outputs/obs025_two_field_seam_panel/...`
- `outputs/obs026_family_two_field_occupancy/...`
- `outputs/obs028_embedding_comparison/...`
- `outputs/obs028b_diffusion_mode_analysis/...`

### Verification
All summary text files should exist and be nonempty.

---

## Step 2 — OBS-028c Canonical Seam Bundle

### Purpose
Export the canonical seam bundle from the base observatory stack.

### Command

```bash
PYTHONPATH=src .venv/bin/python experiments/studies/obs028c_export_canonical_seam_bundle.py
```

### Expected outputs
- `outputs/obs028c_canonical_seam_bundle/seam_nodes.csv`
- `outputs/obs028c_canonical_seam_bundle/seam_family_summary.csv`
- `outputs/obs028c_canonical_seam_bundle/seam_embedding_summary.csv`
- `outputs/obs028c_canonical_seam_bundle/seam_metadata.txt`

### Status
Canonical.

---

## Step 3 — Build Scale Family Substrate

### Purpose
Construct a scale-conditioned path-family substrate from scale-local probe paths, using the base canonical field stack as the reference observatory geometry.

### Canonical note
This is the crucial point where:
- path data is scale-local
- field diagnostics are base-output canonical

This distinction should remain explicit.

### Command

```bash
PYTHONPATH=src .venv/bin/python experiments/toy/build_scale_family_substrate.py \
  --scale-root outputs/scales/100000 \
  --outputs-root outputs
```

### Expected outputs
Under:
- `outputs/scales/100000/family_substrate/`

At minimum:
- `path_nodes_for_family.csv`
- `path_node_diagnostics.csv`
- `path_diagnostics.csv`
- `path_family_assignments.csv`
- `path_family_summary.csv`
- `family_substrate_metadata.txt`

### Verification
Expected rough structure:
- path-node tables nonempty
- path diagnostics row count approximately equals unique path count
- family summary contains multiple families, not a collapsed singleton row

### Status
Canonical.

---

## Step 4 — Prepare OBS-022 Scene Inputs

### Purpose
Refresh the scale-family substrate, scene bundle, hotspot occupancy, canonical seam bundle mirror, and optional pass-2 annotation mirrors from one orchestration point.

### Canonical command

```bash
PYTHONPATH=src .venv/bin/python experiments/toy/prepare_obs022_scene_inputs.py \
  --scale-root outputs/scales/100000 \
  --outputs-root outputs \
  --run-hotspot-occupancy \
  --run-canonical-seam-bundle \
  --run-pass2-annotations
```

### Expected outputs
At minimum:
- `outputs/obs022_scene_bundle/scene_nodes.csv`
- `outputs/obs022_scene_bundle/scene_edges.csv`
- `outputs/obs022_scene_bundle/scene_seam.csv`
- `outputs/obs022_scene_bundle/scene_hubs.csv`
- `outputs/obs022_scene_bundle/scene_routes.csv`
- `outputs/obs022_scene_bundle/scene_glyphs.csv`
- `outputs/obs022_scene_bundle/scene_metadata.txt`

Also refreshes:
- family substrate outputs
- OBS-024 outputs
- OBS-028c outputs
- pass-2 annotation mirrors if enabled

### Status
Canonical orchestration entry point.

---

## Step 5 — OBS-024 Family Hotspot Occupancy

### Purpose
Measure how route families occupy hotspot-like node regions in the scene representation.

### Canonical invocation
Usually produced via `prepare_obs022_scene_inputs.py`, but can be run directly if prerequisites already exist.

### Direct command

```bash
PYTHONPATH=src .venv/bin/python experiments/toy/obs024_family_hotspot_occupancy.py
```

### Expected outputs
- `outputs/obs024_family_hotspot_occupancy/family_hotspot_occupancy_nodes.csv`
- `outputs/obs024_family_hotspot_occupancy/family_hotspot_occupancy_summary.csv`
- `outputs/obs024_family_hotspot_occupancy/obs024_family_hotspot_occupancy_summary.txt`
- `outputs/obs024_family_hotspot_occupancy/obs024_family_hotspot_occupancy_figure.png`

### Status
Canonical.

---

## Step 6 — OBS-050 Structural Coupling Persistence

### Purpose
Test whether recovery-like roughness-escalation windows remain structurally coupled to the seam more often than nonrecovering windows.

### Canonical command

```bash
PYTHONPATH=src .venv/bin/python experiments/studies/obs050_structural_coupling_persistence.py
```

### Expected outputs
Under:
- `outputs/obs050_structural_coupling_persistence/`

At minimum:
- `structural_coupling_segments.csv`
- `structural_coupling_path_summary.csv`
- `structural_coupling_family_summary.csv`
- `structural_coupling_seam_band_summary.csv`
- `structural_coupling_coupled_vs_decoupled_summary.csv`
- `obs050_structural_coupling_persistence_summary.txt`
- `obs050_m_seam_vs_mean_distance_to_seam.png`
- `obs050_seam_band_distribution_by_outcome.png`
- `obs050_coupled_vs_decoupled_by_outcome.png`

### Canonical interpretation
OBS-050 is currently treated as cross-corpus robust at the qualitative level.

### Verification targets
Summary should report:
- recovering coupled share
- nonrecovering coupled share
- coupled risk ratio
- coupled odds ratio

Direction should remain:
- recovering more coupled than nonrecovering

### Status
Canonical, cross-corpus qualitatively replicated.

---

## Step 7 — OBS-051 Local Divergence in Coupled Windows

### Purpose
Measure Lyapunov-like local divergence inside seam-coupled roughness-escalation windows.

### Canonical commands
Run at least these three passes:

#### All coupled windows
```bash
PYTHONPATH=src .venv/bin/python experiments/studies/obs051_local_divergence_in_coupled_windows.py \
  --seam-band-filter all
```

#### Core-only windows
```bash
PYTHONPATH=src .venv/bin/python experiments/studies/obs051_local_divergence_in_coupled_windows.py \
  --seam-band-filter core
```

#### Near-only windows
```bash
PYTHONPATH=src .venv/bin/python experiments/studies/obs051_local_divergence_in_coupled_windows.py \
  --seam-band-filter near
```

### Expected outputs
Under:
- `outputs/obs051_local_divergence_in_coupled_windows/`

At minimum:
- `obs051_window_divergence.csv`
- `obs051_outcome_summary.csv`
- `obs051_family_summary.csv`
- `obs051_local_divergence_summary.txt`
- `obs051_lambda_boxplot_by_outcome.png`
- `obs051_lambda_vs_initial_separation.png`

### Instrument note
The refined version includes:
- nonzero minimum initial separation
- standardized matching features
- secondary boundedness proxies:
  - `mean_delta_d`
  - `mean_bounded_share`

### Current status
Runs reproducibly on both machines, but interpretation is currently **corpus-sensitive**.

### Cross-corpus note
- On corpus C, refined OBS-051 suggests stronger boundedness in recovery-like near-band windows
- On corpus Cp, refined OBS-051 does not preserve that directional conclusion
- Therefore OBS-051 should currently be treated as **provisional**, not canonical

### Operational interpretation
- `core` currently behaves more like a high-pressure contact regime than a clean bounded recovery regime
- `near` is the most informative band, but remains corpus-sensitive in direction

### Status
Provisional.

---

## Canonical vs Provisional Summary

### Canonical chain components
- OBS-028c prerequisites
- OBS-028c canonical seam bundle
- scale family substrate build
- OBS-022 scene input preparation
- OBS-024 hotspot occupancy
- OBS-050 structural coupling persistence

### Provisional chain components
- OBS-051 local divergence in coupled windows

---

## Verification Checklist

### Minimal file-level checks
- all expected summary text files exist
- all expected primary csv outputs exist
- no zero-row summary tables where meaningful results are expected

### Scientific checks

#### OBS-050
- recovering coupled share > nonrecovering coupled share
- coupled risk ratio > 1

#### OBS-051
- script runs successfully for `all`, `core`, and `near`
- summary files are produced
- results explicitly recorded as corpus-sensitive until further stabilization

---

## Recommended Verification Note

Maintain a companion note such as:

- `docs/05_project/observatory_chain_verification_C_vs_Cp.md`

This should record:
- machine
- corpus
- date
- command set used
- whether each stage ran
- whether qualitative conclusions replicated
- whether stage is canonical or provisional

---

## Current Recommended Interpretation

The observatory chain currently supports the following structure:

- seam bundle and family substrate layers are reproducible
- predictive seam-coupling persistence is stable enough to treat as canonical
- local divergence inside coupled windows is scientifically promising, but not yet stable enough to treat as corpus-invariant

That distinction should remain explicit in downstream use.
