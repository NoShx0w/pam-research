# Embedding Adapter Notes

## Purpose

Provide a thin bridge from geometry outputs into the observatory's
`Manifold Embedding` panel.

## Preferred input formats

### 1. `outputs/geometry/mds_points.csv`
Recommended schema:

- `r`
- `alpha`
- `x`
- `y`

This is the most explicit and robust format.

### 2. `outputs/geometry/mds_coords.npy`
Expected shape:

- `(n_cells, 2)`

Interpretation:

- row-major over the `(r_values, alpha_values)` grid
- first column = x
- second column = y

## Fallback behavior

If no geometry output is found, the adapter generates a deterministic placeholder
embedding with:

- horizontal banding by `r`
- within-band spread by `alpha`
- seam-like distortion near `r ≈ 0.15`

This keeps the TUI structurally alive during integration.

## Intended next refinement

If you later export:

- neighbor graph edges
- curvature per point
- cluster labels

those can be folded into richer embedding renderers without changing the
overall adapter boundary.
