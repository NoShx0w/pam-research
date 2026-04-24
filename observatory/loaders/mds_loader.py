from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MDSData:
    mds_df: pd.DataFrame
    mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_mds_data(
    outputs_root: str | Path = "outputs",
    observatory_root: str | Path = "observatory",
) -> MDSData:
    outputs_root = Path(outputs_root)
    observatory_root = Path(observatory_root)

    canonical_csv = observatory_root / "derived" / "geometry" / "mds" / "coords.csv"
    legacy_csv = outputs_root / "fim_mds" / "mds_coords.csv"

    mds_csv = canonical_csv if canonical_csv.exists() else legacy_csv

    if not mds_csv.exists():
        return MDSData(mds_df=pd.DataFrame(), mtime=None)

    df = pd.read_csv(mds_csv).copy()
    for col in ["r", "alpha", "mds1", "mds2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return MDSData(
        mds_df=df.sort_values(["r", "alpha"]).reset_index(drop=True),
        mtime=_safe_mtime(mds_csv),
    )