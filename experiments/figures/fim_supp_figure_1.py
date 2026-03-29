#!/usr/bin/env python3
from __future__ import annotations

"""
fim_supp_figure_1.py

Supplementary Figure S1 — Geometric transition law.

Renders a 2D geometric phase map:
- x-axis: distance_to_seam
- y-axis: log10(1 + scalar_curvature)
- color: P(transition within k steps)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_binned_surface(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    x_bins: int = 24,
    y_bins: int = 24,
) -> pd.DataFrame:
    work = df[[x_col, y_col, z_col]].dropna().copy()
    if len(work) == 0:
        raise ValueError("No valid rows available after dropping NaNs.")

    x_edges = np.linspace(work[x_col].min(), work[x_col].max(), x_bins + 1)
    y_edges = np.linspace(work[y_col].min(), work[y_col].max(), y_bins + 1)

    work["x_bin"] = pd.cut(work[x_col], bins=x_edges, include_lowest=True)
    work["y_bin"] = pd.cut(work[y_col], bins=y_edges, include_lowest=True)

    grouped = (
        work.groupby(["y_bin", "x_bin"], observed=False)[z_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped["x_bin_label"] = grouped["x_bin"].astype(str)
    grouped["y_bin_label"] = grouped["y_bin"].astype(str)
    return grouped


def render_figure(
    df: pd.DataFrame,
    outpath: Path,
    *,
    gridsize: int = 28,
) -> None:
    work = _safe_numeric(
        df,
        ["distance_to_seam", "scalar_curvature", "transition_within_k"],
    )
    work = work.dropna(subset=["distance_to_seam", "scalar_curvature", "transition_within_k"]).copy()
    if len(work) == 0:
        raise ValueError("No valid rows available for plotting.")

    work["log_curvature"] = np.log10(1.0 + work["scalar_curvature"].clip(lower=0.0))

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    hb = ax.hexbin(
        work["distance_to_seam"],
        work["log_curvature"],
        C=work["transition_within_k"],
        reduce_C_function=np.mean,
        gridsize=gridsize,
        mincnt=1,
    )

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("P(transition within k steps)")

    ax.set_xlabel("distance_to_seam")
    ax.set_ylabel("log10(1 + curvature)")
    ax.set_title("Supplementary Figure S1 — Geometric transition law")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Supplementary Figure S1.")
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
        help="CSV containing distance_to_seam, scalar_curvature, transition_within_k",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_supp",
        help="Output directory",
    )
    parser.add_argument("--x-bins", type=int, default=24)
    parser.add_argument("--y-bins", type=int, default=24)
    parser.add_argument("--gridsize", type=int, default=28)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    df = _safe_numeric(df, ["distance_to_seam", "scalar_curvature", "transition_within_k"])
    df["log_curvature"] = np.log10(1.0 + df["scalar_curvature"].clip(lower=0.0))

    surface_df = build_binned_surface(
        df,
        x_col="distance_to_seam",
        y_col="log_curvature",
        z_col="transition_within_k",
        x_bins=args.x_bins,
        y_bins=args.y_bins,
    )
    surface_path = outdir / "supp_figure_1_binned_surface.csv"
    surface_df.to_csv(surface_path, index=False)

    fig_path = outdir / "supp_figure_1_geometric_transition_law.png"
    render_figure(df, fig_path, gridsize=args.gridsize)

    print(fig_path)
    print(surface_path)


if __name__ == "__main__":
    main()
