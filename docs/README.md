# PAM Observatory

**Exploring phase structure in recursive language systems**

PAM Observatory is a research instrument for studying the dynamics of recursive language systems under controlled parameter sweeps.

The system performs batches of experiments across a grid of parameters and records dynamical observables such as freeze probability, entropy, and cross-correlation. A live terminal interface allows researchers to monitor experiments in real time, explore phase diagrams, inspect trajectories, and generate visual artifacts from the evolving sweep.

The observatory is designed to make **phase structure visible while experiments are running**.

---

## What this repository is doing

The repository currently supports a layered workflow:

```text
quench experiments
        ↓
trajectory statistics
        ↓
observable phase surfaces
        ↓
Fisher-type metric tensor
        ↓
Fisher geodesic distances
        ↓
manifold embedding (MDS)
        ↓
phase geometry interpretation
```

This gives the project four practical pillars:

- **experiments**
- **observatory**
- **geometry**
- **documentation**

---

## Observatory Interface

The PAM Observatory TUI exposes three conceptual layers:

1. **Coverage**  
   Which parameter combinations have already been computed.

2. **Phase Diagram**  
   Aggregated observables across parameter space.

3. **Detail View**  
   Local dynamics for a specific parameter configuration.

The interface supports row, cell, and trajectory inspection modes, plus screenshot export for documentation and movie generation.

---

## Parameter Sweep

Experiments explore the parameter manifold

\[
\theta = (r, \alpha)
\]

with a typical grid:

```text
r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
α ∈ linspace(0.03, 0.15, 15)
seeds = 10
```

Total experiments:

```text
5 × 15 × 10 = 750 quenches
```

Each run appends a summary row to:

```text
outputs/index.csv
```

and may also write trajectory files to:

```text
outputs/trajectories/
```

---

## Core Observables

Typical summary observables include:

- `piF_tail`
- `H_joint_mean`
- `corr0`
- `best_corr`
- `delta_r2_freeze`
- `delta_r2_entropy`
- `K_max`

These fields form the bridge between raw dynamics and phase-level geometry.

---

## Running the Experiment Sweep

```bash
python experiments/exp_batch.py
```

The batch runner is resume-aware and continues from existing rows in `outputs/index.csv`.

---

## Launching the Observatory

```bash
PYTHONPATH=. python tui/app.py
```

The TUI reads the live experiment outputs and updates automatically while the sweep progresses.

---

## Controls

```text
↑ ↓     change r
← →     change α
Enter   toggle row / cell inspection
T       trajectory view
S       save SVG screenshot
```

Screenshots are saved to:

```text
tui/screenshots/
```

---

## Observatory Artifacts

The repository includes a movie tool for turning saved Observatory screenshots into GIF or MP4 artifacts.

```bash
python tools/phase_movie.py \
  --input-dir tui/screenshots \
  --output tui/screenshots/phase_movie.gif \
  --fps 4 \
  --hold 3
```

### macOS note

For SVG screenshot conversion on macOS, you may need:

```bash
brew install cairo
brew install ffmpeg
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
```

Python dependencies:

```bash
pip install cairosvg pillow
```

---

## Fisher Geometry Layer

A first-pass Fisher-type metric estimator lives at:

```text
experiments/fim.py
```

It operates on the observable summaries in `outputs/index.csv` and estimates a metric of the form:

\[
g_{ij} = \partial_i m^\top \Sigma^{-1} \partial_j m
\]

where `m(r, α)` is the observable vector and `Σ` is an empirical noise covariance estimated from seed variability.

Current downstream geometry work includes:

- local Fisher surfaces
- determinant and anisotropy diagnostics
- Fisher-distance graphs
- MDS embeddings of the parameter manifold
- curvature and candidate phase-boundary detection

---

## Repository Structure

```text
pam-research/
│
├─ experiments/
│  ├─ exp_batch.py
│  ├─ common_quench_metrics.py
│  ├─ fim.py
│  └─ ...
│
├─ outputs/
│  ├─ index.csv
│  ├─ trajectories/
│  ├─ fim/
│  └─ ...
│
├─ tools/
│  └─ phase_movie.py
│
├─ tui/
│  ├─ app.py
│  ├─ widgets/
│  ├─ controllers/
│  ├─ screenshots/
│  └─ sweep_spec.json
│
├─ docs/
│  ├─ README.md
│  ├─ architecture.md
│  ├─ observatory_philosophy.md
│  ├─ allspark.md
│  ├─ fim_branch.md
│  └─ figures/
│
└─ README.md
```

---

## Documentation Policy

Repository markdown files are the **canonical source of truth** for the project’s documentation.

Chat is used for drafting and refinement, but final structure and formatting should be stabilized in the repository itself.

In short:

```text
chat is trajectory
markdown is state
git is time
```

---

## Research Context

The broader project asks how recursive language systems organize into **phase structure** under parameter variation.

Key questions include:

- When does the system freeze into self-reference?
- How does entropy evolve during collapse?
- Which parameters control transitions between regimes?
- What geometry is induced on the parameter manifold by observable statistics?

The observatory exists to make those structures legible while the experiments are still unfolding.

---

## Future Work

Planned and active directions include:

- automated phase ridge extraction
- direct movie generation from `index.csv`
- Fisher-distance geodesics on the parameter grid
- manifold embeddings via MDS
- curvature-based phase boundary detection
- protocol and capability geometry research branches

---

## The Allspark

The repository explores a simple but powerful idea:

> Meaning, structure, and interaction can be studied as **phase phenomena** in dynamical systems.

Experiments generate trajectories.  
The observatory reveals structure.  
Geometry describes the manifold on which those structures live.

The purpose of this repository is not merely to produce results, but to build the **instruments** that make such structures visible.
