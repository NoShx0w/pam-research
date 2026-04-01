PAM Observatory

Exploring phase structure in recursive language systems

PAM Observatory is a research instrument for studying the dynamics of recursive language systems under controlled parameter sweeps.

The system performs large batches of experiments across a grid of parameters and records dynamical observables such as freeze probability, entropy, and cross-correlation. A live terminal interface allows researchers to monitor the experiment in real time, explore phase diagrams, and inspect trajectory dynamics.

The observatory is designed to make phase transitions in recursive systems visible while experiments are running.

⸻

Observatory Interface

The observatory provides a terminal interface that monitors experiments as they progress.

It exposes three conceptual layers of information:
	1.	Coverage
Which parameter combinations have already been computed.
	2.	Phase Diagram
Aggregated observables across parameter space.
	3.	Detail View
Local dynamics for a specific parameter configuration.

The interface allows interactive navigation across the parameter grid and inspection of both summary statistics and individual trajectories.

Example interface snapshot:

docs/observatory_ui.svg


⸻

Parameter Sweep

Experiments explore a two-dimensional parameter space:

Parameter	Meaning
r	reinforcement strength
α	mixture rate between corpus and generated output
seed	stochastic initialization

A typical sweep configuration:

r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
α ∈ [0.03 … 0.15] (15 steps)
seeds = 10

Total experiments:

5 × 15 × 10 = 750 quenches

Each quench evolves the system for a fixed number of iterations while recording entropy and freeze dynamics.

Results are written incrementally to:

outputs/index.csv


⸻

Observables

Each experiment produces a compact set of observables used to construct the phase diagram.

Metric	Description
πF_tail	freeze probability in the final portion of the run
H_joint_mean	average joint entropy
corr0	correlation between freeze and entropy
ΔR²_freeze	Granger-style predictability of freeze
ΔR²_entropy	entropy predictability
best_corr	strongest lag correlation between freeze and entropy
K_max	maximum cluster count observed

These observables allow the system’s dynamical regimes to be mapped across the parameter grid.

⸻

Running Experiments

Start the parameter sweep:

python exp_batch.py

The batch runner automatically resumes from previous runs by reading outputs/index.csv.

⸻

Launching the Observatory

Start the live monitoring interface:

PYTHONPATH=. python tui/app.py

The observatory reads outputs/index.csv and updates automatically while experiments run.

⸻

Controls

The interface supports interactive exploration of the parameter grid.

↑ ↓     change r
← →     change α
Enter   toggle row / cell inspection
T       trajectory view
S       save SVG screenshot

Screenshots are saved to:

tui/screenshots/

These SVG snapshots capture the exact terminal layout and can be used for documentation or analysis.

⸻

Phase Movies

The repository includes a tool to create time-lapse movies from observatory screenshots.

python tools/phase_movie.py

This stitches saved screenshots into a movie showing the phase diagram emerging as experiments complete.

Future versions will render movies directly from the experimental data in index.csv.

⸻

Repository Structure

pam-research
│
├─ exp_batch.py
│
├─ outputs
│   └─ index.csv
│
├─ tui
│   ├─ app.py
│   ├─ widgets
│   └─ screenshots
│
├─ tools
│   └─ phase_movie.py
│
└─ docs
    └─ observatory_ui.svg


⸻

Research Context

Recursive language systems exhibit complex dynamical behavior when they consume their own generated output.

This project investigates questions such as:

• Under which conditions does the system freeze into self-reference?
• How does entropy evolve during collapse?
• Which parameters control phase transitions between stable and unstable regimes?

The observatory provides a practical environment for exploring these phenomena experimentally.

⸻

Future Work

Planned extensions include:

• automatic phase boundary detection
• direct movie generation from experimental data
• higher-resolution parameter sweeps
• improved trajectory visualization
• information-geometric analysis of entropy dynamics

⸻

Philosophy

The goal of PAM Observatory is not only to run experiments, but to make their structure visible.

Instead of waiting until the end of a parameter sweep to analyze results, the observatory allows researchers to watch phase structure emerge in real time.
