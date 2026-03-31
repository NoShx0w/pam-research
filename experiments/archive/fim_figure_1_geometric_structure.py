#!/usr/bin/env python3
from __future__ import annotations

"""
fim_figure_1_geometric_structure.py

Polished Figure 1 — Geometric structure of the state space.

Builds a 4-panel opener figure:
A. State space schematic (states + empirical seam band)
B. Distance to phase boundary
C. Local curvature
D. Transition localization

Inputs:
    outputs/fim_transition_rate/transition_rate_labeled.csv

Outputs:
    outputs/fim_figure_1/figure_1_geometric_structure.png
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


def _distance_regime(d: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(d < 0.2, "boundary", np.where(d < 0.6, "intermediate", "interior")),
        index=d.index,
    )


def _curve_by_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 24,
    min_count: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(x) == 0:
        return np.array([]), np.array([])

    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1

    means = []
    for i in range(bins):
        m = idx == i
        if np.sum(m) < min_count:
            means.append(np.nan)
        else:
            means.append(float(np.nanmean(y[m])))
    return centers, np.array(means)


def render_figure(df: pd.DataFrame, outpath: Path) -> None:
    work = _safe_numeric(
        df,
        ["distance_to_seam", "scalar_curvature", "transition_within_k", "lazarus_score"],
    )
    work = work.dropna(subset=["distance_to_seam", "scalar_curvature", "transition_within_k"]).copy()
    if len(work) == 0:
        raise ValueError("No valid rows available for plotting.")

    work["scalar_curvature"] = work["scalar_curvature"].clip(lower=0.0)
    work["log_curvature"] = np.log10(1.0 + work["scalar_curvature"])
    work["distance_regime"] = _distance_regime(work["distance_to_seam"])

    x = work["distance_to_seam"].to_numpy()
    y = work["log_curvature"].to_numpy()
    t = work["transition_within_k"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    axA, axB, axC, axD = axes.flatten()

    # Shared visual anchors
    x_boundary_max = 0.2
    x_intermediate_max = 0.6

    # A. State space schematic
    axA.scatter(x, y, s=14, alpha=0.18)
    axA.axvspan(0.0, x_boundary_max, alpha=0.12)
    axA.axvspan(x_boundary_max, x_intermediate_max, alpha=0.06)
    xc, yc = _curve_by_bins(x, t, bins=28, min_count=30)
    valid = np.isfinite(yc)
    if np.any(valid):
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        yc_scaled = ymin + (yc[valid] - np.nanmin(yc[valid])) / (np.nanmax(yc[valid]) - np.nanmin(yc[valid]) + 1e-12) * (0.22 * (ymax - ymin))
        yc_scaled = ymax - 0.12 * (ymax - ymin) + yc_scaled
        axA.plot(xc[valid], yc_scaled, linewidth=2.4, color="black")
    axA.text(0.04, 0.95, "boundary", transform=axA.transAxes, va="top")
    axA.text(0.42, 0.95, "intermediate", transform=axA.transAxes, va="top")
    axA.text(0.78, 0.95, "interior", transform=axA.transAxes, va="top")
    axA.set_title("A. State space and phase structure")
    axA.set_xlabel("Distance to phase boundary")
    axA.set_ylabel("Log curvature")

    # B. Distance to phase boundary
    scB = axB.scatter(x, y, c=work["distance_to_seam"], s=18, alpha=0.9)
    axB.axvspan(0.0, x_boundary_max, alpha=0.08)
    axB.axvspan(x_boundary_max, x_intermediate_max, alpha=0.04)
    cbB = fig.colorbar(scB, ax=axB, shrink=0.86)
    cbB.set_label("Distance to phase boundary")
    axB.set_title("B. Distance field")
    axB.set_xlabel("Distance to phase boundary")
    axB.set_ylabel("Log curvature")

    # C. Curvature field
    scC = axC.scatter(x, y, c=work["log_curvature"], s=18, alpha=0.9)
    cbC = fig.colorbar(scC, ax=axC, shrink=0.86)
    cbC.set_label("Log curvature")
    axC.set_title("C. Local curvature")
    axC.set_xlabel("Distance to phase boundary")
    axC.set_ylabel("Log curvature")

    # D. Transition localization
    scD = axD.scatter(x, y, c=t, s=18, alpha=0.9)
    axD.axvspan(0.0, x_boundary_max, alpha=0.08)
    axD.axvspan(x_boundary_max, x_intermediate_max, alpha=0.04)
    xc2, yc2 = _curve_by_bins(x, t, bins=28, min_count=30)
    valid2 = np.isfinite(yc2)
    if np.any(valid2):
        axD_t = axD.twinx()
        axD_t.plot(xc2[valid2], yc2[valid2], linewidth=2.2, color="black")
        axD_t.set_ylabel("Mean transition probability")
        axD_t.set_ylim(0.0, max(0.05, float(np.nanmax(yc2[valid2]) * 1.15)))
    cbD = fig.colorbar(scD, ax=axD, shrink=0.86)
    cbD.set_label("Transition within k steps")
    axD.set_title("D. Transition localization")
    axD.set_xlabel("Distance to phase boundary")
    axD.set_ylabel("Log curvature")

    fig.suptitle("Figure 1 — Geometric structure of the state space", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render polished Figure 1 opener.")
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
    )
    parser.add_argument("--outdir", default="outputs/fim_figure_1")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    outpath = outdir / "figure_1_geometric_structure.png"
    render_figure(df, outpath)

    print(outpath)


if __name__ == "__main__":
    main()
