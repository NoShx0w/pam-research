from __future__ import annotations

from pathlib import Path

from pam.observatory.data.geodesic_adapter import (
    GeodesicAdapterConfig,
    load_geodesic_probe,
)


if __name__ == "__main__":
    probe = load_geodesic_probe(
        GeodesicAdapterConfig(outputs_dir=Path("outputs")),
        r=0.15,
        alpha=0.064,
    )

    print("mode:", probe.mode)
    print("source_kind:", probe.source_kind)
    print("origin:", (probe.origin_r, probe.origin_alpha))
    print("path_points:", probe.path_points[:4])
    print("n_fan_rays:", len(probe.fan_rays))
    print("shear_level:", probe.shear_level)
