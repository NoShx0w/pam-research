# PAM Observatory — Documentation

## Overview

This documentation describes the architecture and analysis pipeline of the PAM (Phase Analysis of Meaning) Observatory.

The repository is structured as an **experimental instrument** for studying the parameter manifold:

\[
\theta = (r, \alpha)
\]

through:

- large-scale parameter sweeps  
- observable extraction  
- information geometry  
- field topology  
- operator-driven probing  

The system has evolved from a descriptive pipeline into a **geometry + dynamics + topology instrument**.

---

## Documentation Structure

### 01 — Foundations

- [`architecture.md`](../architecture.md)  
  High-level structure of the repository and observatory components

- [`observable_glossary.md`](../observable_glossary.md)  
  Definitions of all recorded observables

---

### 02 — Geometry

- [`geometry_pipeline.md`](../02_geometry/geometry_pipeline.md)  
  End-to-end pipeline from observables to manifold geometry

- [`parameter_sweep_geometry.md`](../02_geometry/parameter_sweep_geometry.md)  
  Structure and coverage of the (r, α) parameter space

---

### 03 — Pipeline (Dynamics & Topology)

- [`phase_geometry.md`](../03_pipeline/phase_geometry.md)  
  Signed phase, seam structure, and phase interpretation

- [`field_topology.md`](../03_pipeline/field_topology.md)  
  Continuous field construction, flow dynamics, and topological structure

- [`operators.md`](../03_pipeline/operators.md)  
  Active transformations (e.g. Geodesic Extraction) and experimental probing

---

## Analysis Pipeline

The PAM Observatory processes data through the following stages:
```text
experiments (exp_batch.py)
↓
observables (index.csv)
↓
Fisher Information Metric (fim.py)
↓
Fisher distance graph (fim_distance.py)
↓
MDS embedding (fim_mds.py)
↓
curvature estimation (fim_curvature.py)
↓
phase field (fim_signed_phase.py)
↓
continuous scalar field φ(x, y)
↓
flow field v = -∇φ
↓
field topology (critical points, basins, saddles)
↓
operators (GE / S) → experimental probing
```
---

## 4. Observatory Interface

- `04_interface/` reserved for TUI and observatory interface documentation

## 5. Project

- `05_project/` reserved for roadmap and reproducibility guides

---

## Conceptual Layers

The system can be understood in three layers:

### Geometry

- parameter manifold
- Fisher metric
- distances and embedding
- curvature

### Dynamics

- phase field
- flow field \( v = -\nabla \phi \)
- trajectory behavior

### Topology

- critical points (sinks, saddles, sources)
- basin structure
- seam interaction
- invariant organization

---

## Key Principle

> **Topology is the part of the field that does not disappear when representation changes.**

The observatory does not identify point identity.

It identifies:

> **relational identity of behavior**

Two runs are equivalent if they share:
- critical point structure  
- connectivity  
- seam relationships  

---

## Operators and Experimental Mode

With the introduction of operators (e.g. Geodesic Extraction):

- trajectories are no longer passive  
- the system can be actively probed  

This enables:

- controlled interaction with the manifold  
- measurement of collapse, divergence, and recovery  
- identification of dynamical constraint surfaces  

---

## Current State

The observatory now supports:

- full parameter sweep (≈750 runs)  
- trajectory recovery and validation  
- Fisher geometry extraction  
- manifold embedding  
- phase and seam detection  
- field topology extraction  
- operator-driven probing  

---

## Summary

The PAM Observatory is no longer just a visualization pipeline.

It is an instrument for:

- extracting structure from data  
- measuring behavior under transformation  
- identifying invariant organization across representations  

In short:

> Geometry describes the manifold.  
> Topology defines its structure.  
> Operators reveal how it behaves.
