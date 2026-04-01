from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class GeometryData:
    geometry_df: pd.DataFrame
    curvature_mtime: float | None
    fim_mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_geometry_data(outputs_root: str | Path = "outputs") -> GeometryData:
    outputs_root = Path(outputs_root)

    curvature_csv = outputs_root / "fim_curvature" / "curvature_surface.csv"
    fim_csv = outputs_root / "fim" / "fim_surface.csv"

    curv = pd.read_csv(curvature_csv).copy() if curvature_csv.exists() else pd.DataFrame()
    fim = pd.read_csv(fim_csv).copy() if fim_csv.exists() else pd.DataFrame()

    if not curv.empty:
        for col in ["r", "alpha", "scalar_curvature"]:
            if col in curv.columns:
                curv[col] = pd.to_numeric(curv[col], errors="coerce")

    if not fim.empty:
        for col in ["r", "alpha", "fim_det", "fim_cond"]:
            if col in fim.columns:
                fim[col] = pd.to_numeric(fim[col], errors="coerce")

    if curv.empty and fim.empty:
        merged = pd.DataFrame(columns=["r", "alpha", "scalar_curvature", "fim_det", "fim_cond"])
    elif curv.empty:
        merged = fim.copy()
    elif fim.empty:
        merged = curv.copy()
    else:
        merged = curv.merge(
            fim[["r", "alpha", "fim_det", "fim_cond"]],
            on=["r", "alpha"],
            how="outer",
        )

    return GeometryData(
        geometry_df=merged.sort_values(["r", "alpha"]).reset_index(drop=True) if not merged.empty else merged,
        curvature_mtime=_safe_mtime(curvature_csv),
        fim_mtime=_safe_mtime(fim_csv),
    )
