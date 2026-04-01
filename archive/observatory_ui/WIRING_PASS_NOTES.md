# PAM Observatory Wiring Pass

This pass connects the geometry-aware shell to real outputs.

## What is now wired

- `outputs/index.csv` -> `ObservatoryState`
- `outputs/geometry/*` -> embedding panel
- `outputs/trajectories/*.npz` -> trajectory panel

## Remaining synthetic area

- Geodesic Probe still uses a placeholder renderer

## Current boot path

```python
PamObservatory(outputs_dir="outputs").run()
```

## Recommended next move

Replace the Geodesic Probe placeholder with a small adapter that reads:
- Fisher path outputs
- geodesic fan results
- or a serialized probe artifact
