
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_read_csv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def unique_sorted(values) -> np.ndarray:
    return np.sort(pd.Series(values).dropna().unique())


def pivot_surface(
    df: pd.DataFrame,
    value_col: str,
    r_col: str = "r",
    a_col: str = "alpha",
    r_values: np.ndarray | None = None,
    a_values: np.ndarray | None = None,
) -> np.ndarray:
    if r_values is None:
        r_values = unique_sorted(df[r_col])
    if a_values is None:
        a_values = unique_sorted(df[a_col])

    grid = df.pivot_table(index=r_col, columns=a_col, values=value_col, aggfunc="mean")
    grid = grid.reindex(index=r_values, columns=a_values)
    return grid.to_numpy(dtype=float)


def pivot_dataframe(
    df: pd.DataFrame,
    value_col: str,
    r_col: str = "r",
    a_col: str = "alpha",
    r_values: np.ndarray | None = None,
    a_values: np.ndarray | None = None,
) -> pd.DataFrame:
    if r_values is None:
        r_values = unique_sorted(df[r_col])
    if a_values is None:
        a_values = unique_sorted(df[a_col])

    grid = df.pivot_table(index=r_col, columns=a_col, values=value_col, aggfunc="mean")
    return grid.reindex(index=r_values, columns=a_values)


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x)
    return (x - m) / s


def central_diff(arr: np.ndarray, axis: int, h: float) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if axis == 0:
        for i in range(1, arr.shape[0] - 1):
            out[i] = (arr[i + 1] - arr[i - 1]) / (2 * h)
    elif axis == 1:
        for j in range(1, arr.shape[1] - 1):
            out[:, j] = (arr[:, j + 1] - arr[:, j - 1]) / (2 * h)
    else:
        raise ValueError("axis must be 0 or 1")
    return out


def finite_laplacian(arr: np.ndarray, dr: float, da: float) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    nr, na = arr.shape
    for i in range(1, nr - 1):
        for j in range(1, na - 1):
            if not np.isfinite(arr[i, j]):
                continue
            drr = (arr[i + 1, j] - 2 * arr[i, j] + arr[i - 1, j]) / (dr ** 2)
            daa = (arr[i, j + 1] - 2 * arr[i, j] + arr[i, j - 1]) / (da ** 2)
            out[i, j] = drr + daa
    return out


def render_heatmap(
    grid: np.ndarray,
    r_values: np.ndarray,
    a_values: np.ndarray,
    title: str,
    outpath: str | Path,
    cbar_label: str | None = None,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(a_values)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_values], rotation=45, ha="right")
    ax.set_yticks(range(len(r_values)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_values])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def drop_node_id_collision(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["node_id"], errors="ignore")


def match_nodes_by_r_alpha(
    source_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    keep_cols: Iterable[str] = ("r", "alpha"),
) -> pd.DataFrame:
    clean = drop_node_id_collision(source_df)
    return clean.merge(
        nodes_df[["node_id", *keep_cols]],
        on=list(keep_cols),
        how="left",
    )
