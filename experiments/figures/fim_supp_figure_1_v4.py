#!/usr/bin/env python3
from __future__ import annotations

"""
fim_supp_figure_1_v4.py

Supplementary Figure S1 — Geometric transition law (publication-grade v4).

Changes from v3:
- support-threshold masking using per-bin counts
- std-centered color scaling for contrast
- simpler, stronger contour levels
- empirical seam/ridge marker from highest transition region
- marginal 1D transition profile over distance
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
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

    mean_pivot = grouped.pivot(index="y_bin_label", columns="x_bin_label", values="mean")
    count_pivot = grouped.pivot(index="y_bin_label", columns="x_bin_label", values="count")

    surface = mean_pivot.to_numpy(dtype=float)
    counts = count_pivot.to_numpy(dtype=float)

    return surface, counts, x_edges, y_edges, grouped


def smooth_surface(surface: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if not np.isfinite(surface).any():
        return surface.copy()

    mask = np.isnan(surface)
    filled = surface.copy()
    filled[mask] = np.nanmean(surface)
    smooth = ndimage.gaussian_filter(filled, sigma=sigma)
    return smooth


def render_figure(
    df: pd.DataFrame,
    outpath: Path,
    *,
    x_bins: int = 30,
    y_bins: int = 30,
    curvature_clip: float = 3.0,
    smooth_sigma: float = 1.0,
    min_count: int = 20,
) -> tuple[float, np.ndarray, np.ndarray]:
    work = _safe_numeric(
        df,
        ["distance_to_seam", "scalar_curvature", "transition_within_k"],
    )
    work = work.dropna(subset=["distance_to_seam", "scalar_curvature", "transition_within_k"]).copy()
    if len(work) == 0:
        raise ValueError("No valid rows available for plotting.")

    work["log_curvature"] = np.log10(1.0 + work["scalar_curvature"].clip(lower=0.0))
    work["log_curvature"] = work["log_curvature"].clip(upper=curvature_clip)

    surface, counts, x_edges, y_edges, _ = build_surface(
        work,
        x_col="distance_to_seam",
        y_col="log_curvature",
        z_col="transition_within_k",
        x_bins=x_bins,
        y_bins=y_bins,
    )

    surface = surface.copy()
    counts = counts.copy()

    # mask unsupported regions
    surface[counts < min_count] = np.nan

    # mask unsupported regions
    surface[counts < min_count] = np.nan

    surface_smooth = smooth_surface(surface, sigma=smooth_sigma)

    finite = surface_smooth[np.isfinite(surface_smooth)]
    if finite.size == 0:
        raise ValueError("No finite values available in smoothed surface after masking.")

    mean = float(np.nanmean(finite))
    std = float(np.nanstd(finite))
    vmin = mean - 1.5 * std
    vmax = mean + 1.5 * std
    if vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))

    # empirical ridge marker: x position of maximal column-averaged transition probability
    x_profile = np.nanmean(surface_smooth, axis=0)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    ridge_x = float(x_centers[int(np.nanargmax(x_profile))])

    fig, ax = plt.subplots(figsize=(7.8, 6.1))

    im = ax.imshow(
        surface_smooth,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )

    levels = np.linspace(vmin, vmax, 4)
    Xc = np.linspace(x_edges[0], x_edges[-1], surface_smooth.shape[1])
    Yc = np.linspace(y_edges[0], y_edges[-1], surface_smooth.shape[0])
    ax.contour(
        Xc,
        Yc,
        surface_smooth,
        levels=levels,
        colors="black",
        linewidths=1.5,
        alpha=0.9,
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("P(transition within k steps)")

    ax.axvline(x=ridge_x, linestyle="--", linewidth=1.2, alpha=0.9)

    # marginal profile over distance (scaled into top band for visibility)
    prof = x_profile.copy()
    prof = (prof - np.nanmin(prof)) / (np.nanmax(prof) - np.nanmin(prof) + 1e-12)
    y_lo = y_edges[-1] - 0.12 * (y_edges[-1] - y_edges[0])
    y_hi = y_edges[-1] - 0.02 * (y_edges[-1] - y_edges[0])
    prof_y = y_lo + prof * (y_hi - y_lo)
    ax.plot(x_centers, prof_y, color="white", linewidth=2.2)

    ax.set_xlabel("Distance to phase boundary")
    ax.set_ylabel("Log curvature")
    ax.set_title("Supplementary Figure S1 — Geometric transition law")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    return ridge_x, x_centers, x_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Supplementary Figure S1 (v4).")
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
        help="CSV containing distance_to_seam, scalar_curvature, transition_within_k",
    )
    parser.add_argument("--outdir", default="outputs/fim_supp", help="Output directory")
    parser.add_argument("--x-bins", type=int, default=30)
    parser.add_argument("--y-bins", type=int, default=30)
    parser.add_argument("--curvature-clip", type=float, default=3.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    df = _safe_numeric(df, ["distance_to_seam", "scalar_curvature", "transition_within_k"])
    df["log_curvature"] = np.log10(1.0 + df["scalar_curvature"].clip(lower=0.0))
    df["log_curvature"] = df["log_curvature"].clip(upper=args.curvature_clip)

    _, _, _, _, grouped = build_surface(
        df,
        x_col="distance_to_seam",
        y_col="log_curvature",
        z_col="transition_within_k",
        x_bins=args.x_bins,
        y_bins=args.y_bins,
    )
    grouped.to_csv(outdir / "supp_figure_1_binned_surface_v4.csv", index=False)

    fig_path = outdir / "supp_figure_1_geometric_transition_law_v4.png"
    ridge_x, x_centers, x_profile = render_figure(
        df,
        fig_path,
        x_bins=args.x_bins,
        y_bins=args.y_bins,
        curvature_clip=args.curvature_clip,
        smooth_sigma=args.smooth_sigma,
        min_count=args.min_count,
    )

    pd.DataFrame({"x_center": x_centers, "x_profile": x_profile}).to_csv(
        outdir / "supp_figure_1_distance_profile_v4.csv", index=False
    )
    with open(outdir / "supp_figure_1_metadata_v4.txt", "w", encoding="utf-8") as f:
        f.write(f"ridge_x: {ridge_x}\n")

    print(fig_path)
    print(outdir / "supp_figure_1_binned_surface_v4.csv")
    print(outdir / "supp_figure_1_distance_profile_v4.csv")
    print(outdir / "supp_figure_1_metadata_v4.txt")


if __name__ == "__main__":
    main()
