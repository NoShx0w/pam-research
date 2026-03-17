# Geodesic Wiring Example

## App-side integration

```python
from pam.observatory.data.geodesic_adapter import (
    GeodesicAdapterConfig,
    load_geodesic_probe,
)

self.geodesic_probe = load_geodesic_probe(
    GeodesicAdapterConfig(outputs_dir=self.outputs_dir),
    r=self.state.selected_r,
    alpha=self.state.selected_alpha,
)
```

## Refresh contract

Reload probe data whenever:
- the selected cell changes
- probe mode changes
- a new geometry artifact is written
```
