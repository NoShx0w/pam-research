#!/usr/bin/env python3
from __future__ import annotations

"""
fim_supp_figure_1_v2.py

Supplementary Figure S1 — Geometric transition law (smoothed phase surface).

Renders a 2D geometric phase map:
- x-axis: Distance to phase boundary
- y-axis: Log curvature
- color: P(transition within k steps)

Changes from v1:
- uses a binned 2D surface instead of sparse hexes
- clips extreme curvature for readability
- overlays contour lines
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


def build_surface(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    x_bins: int = 30,
    y_bins: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
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

    pivot = grouped.pivot(index="y_bin_label", columns="x_bin_label", values="mean")
    surface = pivot.to_numpy(dtype=float)

    return surface, x_edges, y_edges, grouped


def render_figure(
    df: pd.DataFrame,
    outpath: Path,
    *,
    x_bins: int = 30,
    y_bins: int = 30,
    curvature_clip: float = 3.0,
) -> None:
    work = _safe_numeric(
        df,
        ["distance_to_seam", "scalar_curvature", "transition_within_k"],
    )
    work = work.dropna(subset=["distance_to_seam", "scalar_curvature", "transition_within_k"]).copy()
    if len(work) == 0:
        raise ValueError("No valid rows available for plotting.")

    work["log_curvature"] = np.log10(1.0 + work["scalar_curvature"].clip(lower=0.0))
    work["log_curvature"] = work["log_curvature"].clip(upper=curvature_clip)

    surface, x_edges, y_edges, _ = build_surface(
        work,
        x_col="distance_to_seam",
        y_col="log_curvature",
        z_col="transition_within_k",
        x_bins=x_bins,
        y_bins=y_bins,
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.9))

    im = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        interpolation="nearest",
    )

    finite = surface[np.isfinite(surface)]
    if finite.size > 1:
        levels = np.linspace(finite.min(), finite.max(), 5)
        Xc = np.linspace(x_edges[0], x_edges[-1], surface.shape[1])
        Yc = np.linspace(y_edges[0], y_edges[-1], surface.shape[0])
        ax.contour(
            Xc,
            Yc,
            surface,
            levels=levels,
            colors="black",
            linewidths=0.55,
        )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("P(transition within k steps)")

    ax.set_xlabel("Distance to phase boundary")
    ax.set_ylabel("Log curvature")
    ax.set_title("Supplementary Figure S1 — Geometric transition law")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Supplementary Figure S1 (smoothed).")
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
        help="CSV containing distance_to_seam, scalar_curvature, transition_within_k",
    )
    parser.add_argument("--outdir", default="outputs/fim_supp", help="Output directory")
    parser.add_argument("--x-bins", type=int, default=30)
    parser.add_argument("--y-bins", type=int, default=30)
    parser.add_argument("--curvature-clip", type=float, default=3.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    df = _safe_numeric(df, ["distance_to_seam", "scalar_curvature", "transition_within_k"])
    df["log_curvature"] = np.log10(1.0 + df["scalar_curvature"].clip(lower=0.0))
    df["log_curvature"] = df["log_curvature"].clip(upper=args.curvature_clip)

    _, _, _, surface_df = build_surface(
        df,
        x_col="distance_to_seam",
        y_col="log_curvature",
        z_col="transition_within_k",
        x_bins=args.x_bins,
        y_bins=args.y_bins,
    )
    surface_path = outdir / "supp_figure_1_binned_surface_v2.csv"
    surface_df.to_csv(surface_path, index=False)

    fig_path = outdir / "supp_figure_1_geometric_transition_law_v2.png"
    render_figure(
        df,
        fig_path,
        x_bins=args.x_bins,
        y_bins=args.y_bins,
        curvature_clip=args.curvature_clip,
    )

    print(fig_path)
    print(surface_path)


if __name__ == "__main__":
    main()
