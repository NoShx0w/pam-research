PAM Observatory Architecture

This document describes the structure and data flow of the PAM Observatory system.

The repository is organized around three core layers:
	1.	Experiment Engine
	2.	Observatory Interface
	3.	Visualization Tools

These layers communicate through simple, durable data artifacts.

⸻

System Overview

exp_batch.py
      │
      │ runs parameter sweeps
      ▼
outputs/index.csv
      │
      │ live experiment state
      ▼
PAM Observatory (TUI)
      │
      ├─ screenshots
      │
      ▼
tools/phase_movie.py

The architecture intentionally uses files as interfaces between layers. This keeps the experiment runner, observatory, and analysis tools loosely coupled.

⸻

1. Experiment Engine

The experiment engine is responsible for executing parameter sweeps.

Primary entry point:

exp_batch.py

Each experiment run (called a quench) evolves a recursive language system for a fixed number of iterations.

Parameter sweeps explore combinations of:

r      reinforcement strength
α      mixture rate
seed   stochastic initialization

Typical sweep size:

r values     = 5
α values     = 15
seeds        = 10
total runs   = 750


⸻

Execution Model

Experiments are executed in parallel using a bounded process pool.

ProcessPoolExecutor

Each worker performs:

run_one_job()
      │
      ▼
run_one_summary()
      │
      ▼
compute summary metrics
      │
      ▼
append row to index.csv

Workers operate independently and return only compact summary data.

⸻

2. Data Interface

The central data artifact is:

outputs/index.csv

Each row represents one completed quench.

Example columns:

corpus
r
alpha
seed
iters
W

piF_mean
piF_tail
H_joint_mean
var_H_joint
corr0
best_corr
delta_r2_freeze
delta_r2_entropy
K_max

This file serves as the live experiment state.

Important properties:
	•	append-only
	•	human readable
	•	restart-safe
	•	suitable for streaming analysis

⸻

3. Trajectory Data

In addition to summary metrics, the experiment engine stores lightweight trajectory files.

outputs/trajectories/

Each trajectory file contains time-series data for a single run.

Example contents:

F_raw      freeze state over time
H_joint    entropy series
K          cluster count series

These files allow detailed inspection of system dynamics.

⸻

4. PAM Observatory (TUI)

The observatory is a terminal interface for monitoring experiments.

Entry point:

tui/app.py

The interface reads index.csv and updates automatically.

The observatory is composed of modular widgets.

tui/widgets


⸻

Observatory Layers

The interface is organized into three conceptual layers.

Coverage

Displays which parameter combinations have been executed.

CoverageHeatmap

Each cell shows seed coverage for a given (r, α) pair.

⸻

Phase Diagram

Displays aggregated observables across parameter space.

PhaseDiagram

Each cell represents the mean value of a chosen metric.

Default metric:

πF_tail


⸻

Detail View

Displays local information for the selected parameter configuration.

Three modes:

row view
cell view
trajectory view

Trajectory view renders ASCII plots of the time series.

⸻

5. Selection Model

Navigation is handled by a small state controller.

tui/controllers/selection.py

The selection state tracks:

selected r
selected α
view mode

Navigation controls:

↑ ↓    change r
← →    change α
Enter  toggle row/cell
T      trajectory view


⸻

6. Screenshot System

The observatory can export vector screenshots.

Key binding:

S

Screenshots are saved to:

tui/screenshots/

Example filename:

obs_r0.20_a0.039_2026-03-13_11-01-53.svg

SVG format preserves the exact terminal layout and is ideal for documentation.

⸻

7. Visualization Tools

The repository includes tools for converting observatory artifacts into visualizations.

Example:

tools/phase_movie.py

This tool converts screenshot sequences into time-lapse movies showing the phase diagram emerging as experiments complete.

⸻

8. Design Principles

The system follows several architectural principles.

File-Based Interfaces

Components communicate via files rather than in-memory APIs.

Benefits:
	•	robustness
	•	easy debugging
	•	simple tooling
	•	reproducibility

⸻

Incremental Computation

Experiments append results as they finish.

Benefits:
	•	safe interruption
	•	restart capability
	•	live observability

⸻

Loose Coupling

The experiment engine and observatory are independent.

The observatory can analyze past runs without modification.

⸻

9. Future Architecture

Planned extensions include:

phase boundary detection
direct movie generation from index.csv
cluster-scale parameter sweeps
interactive trajectory animation
information-geometric visualizations

These additions will build on the same file-based architecture.

⸻

Summary

The PAM Observatory architecture is designed to make experimental phase structure visible while experiments are running.

The combination of:

parameter sweeps
live observatory interface
visual artifact generation

creates a flexible environment for exploring recursive language systems and their emergent dynamics.
