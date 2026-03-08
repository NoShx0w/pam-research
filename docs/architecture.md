Excellent — this is the document that makes people instantly understand the system when they open the repo.

A good docs/architecture.md should:

• explain the conceptual pipeline
• show how the modules connect
• connect code → research idea

I’ll also include a Mermaid diagram, since GitHub renders those automatically (like your flowcharts).

You can place this in:

docs/architecture.md


⸻

docs/architecture.md

# PAM Architecture

This document explains the conceptual architecture of the PAM research framework.

PAM is designed to study **recursive language dynamics** as a dynamical system. The framework evolves a corpus under controlled mutation and anchoring dynamics and measures macroscopic observables such as entropy and freeze occupancy.

The system consists of four major layers:

1. Invariant detection (TIP)
2. Corpus dynamics (PAM engine)
3. Macroscopic metrics
4. Experiment runners

---

# Conceptual Pipeline

```mermaid
flowchart LR

A[Initial Corpus] --> B[PAM Dynamics Engine]

B --> C[Corpus Snapshots]

C --> D1[Entropy Metrics]
C --> D2[Macrostate Analysis]
C --> D3[TIM Trajectory Analysis]

D1 --> E[Phase Observables]
D2 --> E
D3 --> E

E --> F[Experiments / Phase Diagram]

The engine evolves the corpus while the metrics layer observes macroscopic behavior.

⸻

Core Components

TIP — Text Invariant Perceptron

TIP detects semantic invariant signatures in texts.

Each text is mapped to a boolean signature vector:

{
  reflective: True,
  coherent: True,
  playful_serious: False,
  geometric: True
}

These signatures allow the system to measure structural properties of the evolving corpus.

Responsibilities:
	•	detect invariant signatures
	•	produce signature vectors
	•	support entropy and mutation filtering

Source:

src/pam/tip.py


⸻

TIM — Trajectory Invariance Metric

TIM measures semantic trajectory stability.

A text is perturbed along a time axis:
	•	truncation
	•	sentence removal
	•	resampling

The metric measures how invariant the semantic signature remains under these transformations.

TIM provides a measure of structural robustness.

Source:

src/pam/tim.py


⸻

PAM Dynamics Engine

The engine evolves the corpus under mixture dynamics.

At each iteration:
	1.	A subset of the corpus is replaced.
	2.	Replacement samples are generated via:
	•	anchor mutation
	•	self-resampling
	3.	Anchor texts are injected with probability α.

Control parameters:

r  = replacement fraction
α  = anchor injection probability

This creates a discrete-time dynamical system over corpus states.

Source:

src/pam/dynamics/
src/pam/engine.py


⸻

Mutation System

Mutation operators introduce controlled variation.

Examples:
	•	synonym substitution
	•	clause shuffling
	•	sentence reordering
	•	rhetorical lens toggles

Mutations are designed to preserve semantic structure while introducing variability.

Source:

src/pam/injectors.py
src/pam/mutation.py


⸻

Macroscopic Metrics

Metrics observe the emergent behavior of the system.

Entropy

Measures diversity of invariant signatures.

Two variants:

• Marginal entropy
• Joint signature entropy

Source:

src/pam/metrics/entropy.py


⸻

Macrostate Detection

Microstructure analysis identifies:
	•	boundary density
	•	grain size

These features classify windows into macrostates:

F = Freeze
M = Mixed
E = Entropy

Freeze occupancy:

π_F

Source:

src/pam/metrics/macrostate.py


⸻

Lag Correlation

Measures temporal relationship between observables.

Example:

corr(π_F(t), H(t + lag))

Used to detect dynamic coupling.

Source:

src/pam/metrics/lag.py


⸻

Minimal Dynamical Models

Two minimal autoregressive models are fitted:

F_{t+1} = a + bF_t + cH_t
H_{t+1} = d + eH_t + fF_t

These models test cross-variable predictive power.

Source:

src/pam/metrics/regression.py


⸻

Experiment Layer

Experiments orchestrate runs and parameter sweeps.

Examples:

experiments/exp_quench.py
experiments/exp_alpha_sweep.py
experiments/exp_batch.py

Experiments typically perform:
	1.	corpus evolution
	2.	macrostate extraction
	3.	entropy analysis
	4.	regression tests
	5.	logging results

⸻

Data Logging

Experiments produce two output types.

Deep Runs

Detailed run data:

outputs/deep_*.json

Contains:
	•	entropy time series
	•	macrostate sequence
	•	regression results
	•	lag correlations

⸻

Sweep Index

Summary of runs:

outputs/index.csv

Used to aggregate results across seeds and parameter sweeps.

⸻

Research Workflow

Typical research workflow:

flowchart TD

A[Define experiment parameters] --> B[Run batch experiments]

B --> C[Log outputs]

C --> D[Analyze index.csv]

D --> E[Generate phase diagrams]

E --> F[Interpret regimes]


⸻

Phase Discovery Goal

The goal of the PAM framework is to map phase structure in recursive language systems.

Parameter space:

(r, α)

Primary observables:

π_F  = freeze occupancy
H    = signature entropy
K    = microstructure complexity
ΔR²  = causal coupling

The system exhibits distinct regimes depending on these parameters.

Mapping these regimes produces a phase diagram of recursive language dynamics.

⸻

Repository

Code and experiments:

https://github.com/NoShx0w/pam-research

---

## Why this document is powerful

When someone opens your repo, they will now see:

README.md
docs/notes.md
docs/architecture.md

That combination is **exactly what serious research repositories look like**.

It tells readers:

• what the project is  
• how it works  
• how it was discovered

