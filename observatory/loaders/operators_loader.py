from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class OperatorsData:
    operators_df: pd.DataFrame
    lazarus_mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_operators_data(
    outputs_root: str | Path = "outputs",
    observatory_root: str | Path = "observatory",
) -> OperatorsData:
    outputs_root = Path(outputs_root)
    observatory_root = Path(observatory_root)

    canonical_lazarus_csv = observatory_root / "derived" / "operators" / "lazarus" / "lazarus_scores.csv"
    legacy_lazarus_csv = outputs_root / "fim_lazarus" / "lazarus_scores.csv"

    lazarus_csv = canonical_lazarus_csv if canonical_lazarus_csv.exists() else legacy_lazarus_csv

    if not lazarus_csv.exists():
        return OperatorsData(
            operators_df=pd.DataFrame(columns=["r", "alpha", "lazarus_score"]),
            lazarus_mtime=None,
        )

    df = pd.read_csv(lazarus_csv).copy()

    # support a couple of likely column names
    rename_map = {}
    if "lazarus" in df.columns and "lazarus_score" not in df.columns:
        rename_map["lazarus"] = "lazarus_score"
    df = df.rename(columns=rename_map)

    for col in ["r", "alpha", "lazarus_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in ["r", "alpha", "lazarus_score"] if c in df.columns]
    df = df[keep].sort_values(["r", "alpha"]).reset_index(drop=True)

    return OperatorsData(
        operators_df=df,
        lazarus_mtime=_safe_mtime(lazarus_csv),
    )