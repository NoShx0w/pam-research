from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json


@dataclass(slots=True)
class GeodesicProbeData:
    mode: str
    source_kind: str
    origin_r: float
    origin_alpha: float
    path_points: list[tuple[float, float]]
    fan_rays: list[list[tuple[float, float]]]
    shear_level: str = "unknown"
    source_path: Path | None = None


@dataclass(slots=True)
class GeodesicAdapterConfig:
    outputs_dir: Path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_probe_json(path: Path) -> GeodesicProbeData | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    mode = payload.get("mode", "fan")
    origin = payload.get("origin", {})
    path_points = [
        (_safe_float(p[0]), _safe_float(p[1]))
        for p in payload.get("path_points", [])
        if isinstance(p, (list, tuple)) and len(p) >= 2
    ]
    fan_rays = []
    for ray in payload.get("fan_rays", []):
        parsed = [
            (_safe_float(p[0]), _safe_float(p[1]))
            for p in ray
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        if parsed:
            fan_rays.append(parsed)

    return GeodesicProbeData(
        mode=mode,
        source_kind="json",
        origin_r=_safe_float(origin.get("r")),
        origin_alpha=_safe_float(origin.get("alpha")),
        path_points=path_points,
        fan_rays=fan_rays,
        shear_level=str(payload.get("shear_level", "unknown")),
        source_path=path,
    )


def _load_probe_csv(path: Path) -> GeodesicProbeData | None:
    """
    Expected schema for path CSV:
    kind,index,r,alpha,ray
    path,0,0.30,0.032,
    path,1,0.25,0.048,
    fan,0,0.15,0.048,0
    fan,1,0.15,0.064,0
    fan,0,0.15,0.048,1
    """
    if not path.exists():
        return None

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))

    path_points: list[tuple[float, float]] = []
    rays: dict[int, list[tuple[float, float]]] = {}

    for row in rows:
        kind = (row.get("kind") or "").strip().lower()
        r = _safe_float(row.get("r"))
        alpha = _safe_float(row.get("alpha") or row.get("α"))
        if kind == "path":
            path_points.append((r, alpha))
        elif kind == "fan":
            ray_idx = int(_safe_float(row.get("ray"), 0))
            rays.setdefault(ray_idx, []).append((r, alpha))

    all_points = path_points[:]
    for ray in rays.values():
        all_points.extend(ray)

    if not all_points:
        return None

    origin_r, origin_alpha = all_points[0]
    return GeodesicProbeData(
        mode="fan" if rays else "path",
        source_kind="csv",
        origin_r=origin_r,
        origin_alpha=origin_alpha,
        path_points=path_points,
        fan_rays=[rays[k] for k in sorted(rays)],
        shear_level="unknown",
        source_path=path,
    )


def _placeholder_probe(r: float, alpha: float) -> GeodesicProbeData:
    seam_like = abs(r - 0.15) < 1e-9

    if seam_like:
        path_points = [
            (r + 0.15, 0.032),
            (r + 0.10, 0.048),
            (r + 0.05, 0.064),
            (r + 0.00, 0.064),
            (r - 0.05, 0.080),
            (r - 0.05, 0.096),
        ]
        fan_rays = [
            [(r, alpha), (r - 0.03, alpha - 0.016), (r - 0.05, alpha - 0.032)],
            [(r, alpha), (r - 0.02, alpha + 0.000), (r - 0.04, alpha + 0.000)],
            [(r, alpha), (r - 0.01, alpha + 0.016), (r + 0.02, alpha + 0.032)],
        ]
        shear = "high"
    else:
        path_points = [
            (r, max(0.032, alpha - 0.032)),
            (r, alpha),
            (r, min(0.096, alpha + 0.032)),
        ]
        fan_rays = [
            [(r, alpha), (r - 0.02, alpha - 0.016), (r - 0.04, alpha - 0.032)],
            [(r, alpha), (r - 0.02, alpha + 0.000), (r - 0.04, alpha + 0.000)],
            [(r, alpha), (r - 0.02, alpha + 0.016), (r - 0.04, alpha + 0.032)],
        ]
        shear = "low"

    return GeodesicProbeData(
        mode="fan",
        source_kind="placeholder",
        origin_r=r,
        origin_alpha=alpha,
        path_points=path_points,
        fan_rays=fan_rays,
        shear_level=shear,
        source_path=None,
    )


def load_geodesic_probe(
    config: GeodesicAdapterConfig,
    *,
    r: float,
    alpha: float,
) -> GeodesicProbeData:
    """
    Preferred load order:
    1. outputs/geometry/geodesic_probe.json
    2. outputs/geometry/geodesic_probe.csv
    3. deterministic placeholder probe for the selected cell
    """
    geometry_dir = config.outputs_dir / "geometry"

    json_path = geometry_dir / "geodesic_probe.json"
    data = _load_probe_json(json_path)
    if data is not None:
        return data

    csv_path = geometry_dir / "geodesic_probe.csv"
    data = _load_probe_csv(csv_path)
    if data is not None:
        return data

    return _placeholder_probe(r=r, alpha=alpha)
