# PAM Observatory Data Adapter

This module is the thin bridge between real experiment outputs and `ObservatoryState`.

## Current scope

- reads `outputs/index.csv`
- groups rows by `(r, alpha)`
- estimates per-cell coverage
- normalizes:
  - `K_max` -> `curvature`
  - `piF_tail`
  - `H_joint_mean`

## Intended next extensions

- load real MDS coordinates
- load real trajectory arrays from `.npz`
- compute probe data from Fisher metric artifacts
