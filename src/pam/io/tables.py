from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)
    return path


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
