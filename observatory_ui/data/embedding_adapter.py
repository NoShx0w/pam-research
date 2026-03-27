from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import csv
import numpy as np


@dataclass(slots=True)
class EmbeddingPoint:
    r: float
    alpha: float
    x: float
    y: float
    source_kind: str = "placeholder"


@dataclass(slots=True)
class EmbeddingAdapterConfig:
    outputs_dir: Path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize(values: np.ndarray, out_min: float = 0.0, out_max: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full_like(values, (out_min + out_max) / 2)
    return out_min + (values - vmin) * (out_max - out_min) / (vmax - vmin)


def _load_points_csv(path: Path) -> list[EmbeddingPoint] | None:
    if not path.exists():
        return None

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))

    points: list[EmbeddingPoint] = []
    for row in rows:
        r = _safe_float(row.get("r"))
        alpha = _safe_float(row.get("alpha") or row.get("α"))
        x = _safe_float(row.get("x") or row.get("mds_x") or row.get("dim1"))
        y = _safe_float(row.get("y") or row.get("mds_y") or row.get("dim2"))
        points.append(EmbeddingPoint(r=r, alpha=alpha, x=x, y=y, source_kind="csv"))

    return points or None


def _load_points_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def _placeholder_embedding(r_values: list[float], alpha_values: list[float]) -> list[EmbeddingPoint]:
    points: list[EmbeddingPoint] = []
    for r in r_values:
        for alpha in alpha_values:
            y = r * 10.0
            x = alpha * 100.0
            if abs(r - 0.15) < 1e-9:
                y -= 0.8
                x += (alpha - 0.064) * 8.0
            points.append(EmbeddingPoint(r=r, alpha=alpha, x=x, y=y, source_kind="placeholder"))
    return points


def load_embedding_points(
    config: EmbeddingAdapterConfig,
    *,
    r_values: list[float],
    alpha_values: list[float],
) -> list[EmbeddingPoint]:
    """
    Preferred load order:
    1. outputs/geometry/mds_points.csv
       columns: r, alpha, x, y
    2. outputs/geometry/mds_coords.npy
       shape: (n_cells, 2), assumed row-major over (r_values, alpha_values)
    3. deterministic placeholder embedding
    """
    geometry_dir = config.outputs_dir / "geometry"

    csv_path = geometry_dir / "mds_points.csv"
    csv_points = _load_points_csv(csv_path)
    if csv_points is not None:
        return csv_points

    npy_path = geometry_dir / "mds_coords.npy"
    coords = _load_points_npy(npy_path)
    if coords is not None:
        expected = len(r_values) * len(alpha_values)
        if coords.shape[0] >= expected:
            coords = coords[:expected]
            xs = _normalize(coords[:, 0], 0.0, 1.0)
            ys = _normalize(coords[:, 1], 0.0, 1.0)
            points: list[EmbeddingPoint] = []
            idx = 0
            for r in r_values:
                for alpha in alpha_values:
                    points.append(
                        EmbeddingPoint(
                            r=r,
                            alpha=alpha,
                            x=float(xs[idx]),
                            y=float(ys[idx]),
                            source_kind="npy",
                        )
                    )
                    idx += 1
            return points

    return _placeholder_embedding(r_values=r_values, alpha_values=alpha_values)
