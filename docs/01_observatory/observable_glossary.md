# Observable Glossary

This document defines the observables stored in:

outputs/index.csv

Each row corresponds to one parameter configuration (r, α), aggregated over multiple seeds.

These observables form the input to the geometry pipeline.

## Overview

The PAM Observatory extracts a set of scalar observables from each trajectory.

These observables capture:
- entropy and uncertainty
- coupling between variables
- temporal structure
- correlation patterns

Together, they define a feature vector x(θ) ∈ ℝⁿ used to construct the Fisher Information Metric.

## Core Observables

### 1. F (Freeze)
Measures the degree of convergence or stabilization in the system.

### 2. H (Entropy)
Measures the uncertainty or diversity of system states.

### 3. H_joint (Joint Entropy)
Entropy computed over joint variables or combined state representations.

### 4. K
A coupling or interaction measure derived from the system dynamics.

### 5. π (pi)
A probability-like or distributional observable derived from state frequencies.

### 6. Hj_sm (Smoothed Joint Entropy)
A smoothed version of joint entropy to reduce noise.

### 7. Correlation Structure
Derived from:
- `lags`
- `corrs`

Measures memory and temporal structure.

## Aggregation

Each observable is computed from trajectories and then aggregated across time and across seeds, yielding one scalar per observable per (r, α).

## Role in Geometry

The observables define a feature vector x(θ) = [F, H, K, π, …].

From this, we construct a Fisher metric that measures sensitivity of observables to parameter changes and defines a local geometry on parameter space.

## Summary

- Observables are extracted from trajectories
- They define a feature space over parameters
- The Fisher metric uses them to define geometry
- Geometry reveals structure (curvature, phase, boundaries)

They are the **interface between dynamics and geometry**.
