# Trajectory Adapter Notes

## Purpose

Provide a thin bridge from `outputs/trajectories/*.npz` into the observatory's
`Trajectory Signatures` panel.

## Current behavior

The adapter tries to extract likely arrays from `.npz` files using flexible key names.

### Preferred direct fields
- `F_raw`
- `H_joint`
- `K`
- `piF_smooth`

### Accepted fallbacks
- `F`, `freeze`, `freeze_series`
- `H`, `entropy`, `entropy_series`
- `K_series`, `curvature`, `k_series`
- `pif_smooth`, `piF`, `piF_tail_series`

## Derived fields

If the file does not contain:
- `K(t)`, it is approximated from discrete derivative magnitude
- `πF_smooth(t)`, it is approximated with a moving average over `F_raw`

## Fallback behavior

If no file is found, the adapter returns deterministic placeholder series so the
TUI remains alive during integration.

## Intended next refinement

Once the exact PAM `.npz` schema is known, tighten:
- filename matching
- key names
- the `K(t)` derivation or direct loading path
