from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RunData:
    index_df: pd.DataFrame
    coverage_df: pd.DataFrame
    index_mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_run_data(outputs_root: str | Path = "outputs") -> RunData:
    outputs_root = Path(outputs_root)
    index_csv = outputs_root / "index.csv"

    if not index_csv.exists():
        empty = pd.DataFrame(columns=["r", "alpha", "n_rows", "n_seeds"])
        return RunData(index_df=pd.DataFrame(), coverage_df=empty, index_mtime=None)

    df = pd.read_csv(index_csv).copy()

    for col in ["r", "alpha", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"r", "alpha"}.issubset(df.columns):
        grouped = df.groupby(["r", "alpha"], dropna=False)
        coverage = grouped.agg(
            n_rows=("alpha", "size"),
            n_seeds=("seed", "nunique") if "seed" in df.columns else ("alpha", "size"),
        ).reset_index()
    else:
        coverage = pd.DataFrame(columns=["r", "alpha", "n_rows", "n_seeds"])

    return RunData(
        index_df=df,
        coverage_df=coverage.sort_values(["r", "alpha"]).reset_index(drop=True),
        index_mtime=_safe_mtime(index_csv),
    )
