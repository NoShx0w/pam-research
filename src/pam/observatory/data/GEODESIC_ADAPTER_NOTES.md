# Geodesic Adapter Notes

## Purpose

Provide a thin bridge from geometry outputs into the observatory's
`Geodesic Probe` panel.

## Preferred input formats

### 1. `outputs/geometry/geodesic_probe.json`

Suggested schema:

```json
{
  "mode": "fan",
  "origin": {"r": 0.15, "alpha": 0.064},
  "path_points": [[0.30, 0.032], [0.25, 0.048]],
  "fan_rays": [
    [[0.15, 0.064], [0.12, 0.048]],
    [[0.15, 0.064], [0.13, 0.064]]
  ],
  "shear_level": "high"
}
```

### 2. `outputs/geometry/geodesic_probe.csv`

Suggested columns:

- `kind`
- `index`
- `r`
- `alpha`
- `ray`

## Fallback behavior

If no serialized probe artifact is present, the adapter returns a deterministic
placeholder probe that behaves seam-like near `r ≈ 0.15`.

## Intended next refinement

Once the exact PAM geodesic export format is fixed, tighten:

- the file naming
- parsing rules
- shear estimation
