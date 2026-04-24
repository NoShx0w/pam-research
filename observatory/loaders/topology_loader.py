from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TopologyData:
    topology_df: pd.DataFrame
    criticality_mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_topology_data(
    outputs_root: str | Path = "outputs",
    observatory_root: str | Path = "observatory",
) -> TopologyData:
    outputs_root = Path(outputs_root)
    observatory_root = Path(observatory_root)

    canonical_criticality_csv = (
        observatory_root / "derived" / "topology" / "criticality" / "criticality_surface.csv"
    )
    legacy_criticality_csv = outputs_root / "fim_critical" / "criticality_surface.csv"

    criticality_csv = (
        canonical_criticality_csv if canonical_criticality_csv.exists() else legacy_criticality_csv
    )

    if not criticality_csv.exists():
        return TopologyData(
            topology_df=pd.DataFrame(columns=["r", "alpha", "criticality"]),
            criticality_mtime=None,
        )

    df = pd.read_csv(criticality_csv).copy()
    for col in ["r", "alpha", "criticality"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in ["r", "alpha", "criticality"] if c in df.columns]
    df = df[keep].sort_values(["r", "alpha"]).reset_index(drop=True)

    return TopologyData(
        topology_df=df,
        criticality_mtime=_safe_mtime(criticality_csv),
    )