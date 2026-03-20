#!/usr/bin/env python3
from __future__ import annotations

"""
fim_figure_1_geometric_structure_v3.py

Polished Figure 1 — Geometric structure of the state space (v3).

Changes from v2:
- monotone-smoothed transition ridge in panel D
- explicitly schematic ridge placement in panel A
- dashed regime boundaries only in analytical panels (B, D)
- stronger semantic labels and refined colorbar text
- subtle panel takeaways
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _distance_regime(d: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(d < 0.2, "boundary", np.where(d < 0.6, "transition zone", "stable interior")),
        index=d.index,
    )


def _binned_mean_curve(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 24,
    min_count: int = 25,
    smooth_sigma: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1

    means = []
    counts = []
    for i in range(bins):
        m = idx == i
        n = int(np.sum(m))
        counts.append(n)
        if n < min_count:
            means.append(np.nan)
        else:
            means.append(float(np.nanmean(y[m])))

    means = np.array(means, dtype=float)
    counts = np.array(counts, dtype=float)

    valid = np.isfinite(means)
    if np.any(valid):
        filled = means.copy()
        missing = ~valid
        if np.any(missing):
            filled[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(valid), means[valid])
        smooth = gaussian_filter1d(filled, sigma=smooth_sigma)
        smooth[~valid] = np.nan
    else:
        smooth = means

    return centers, smooth, counts


def _monotone_nonincreasing(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    idx = np.flatnonzero(valid)
    for j in range(1, len(idx)):
        i_prev = idx[j - 1]
        i_cur = idx[j]
        if out[i_cur] > out[i_prev]:
            out[i_cur] = out[i_prev]
    return out


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

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.8))
    axA, axB, axC, axD = axes.flatten()

    x_boundary_max = 0.2
    x_transition_max = 0.6

    def decorate_regimes(ax, with_lines: bool) -> None:
        ax.axvspan(0.0, x_boundary_max, alpha=0.12)
        ax.axvspan(x_boundary_max, x_transition_max, alpha=0.06)
        if with_lines:
            ax.axvline(x_boundary_max, linestyle="--", alpha=0.4, linewidth=1.0)
            ax.axvline(x_transition_max, linestyle="--", alpha=0.4, linewidth=1.0)

    # Panel A — conceptual opener
    decorate_regimes(axA, with_lines=False)
    axA.scatter(x, y, s=8, alpha=0.08)
    xc, ridge, counts = _binned_mean_curve(x, t, bins=28, min_count=30, smooth_sigma=1.2)
    valid = np.isfinite(ridge)
    if np.any(valid):
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        ridge_norm = (ridge.copy() - np.nanmin(ridge[valid])) / (np.nanmax(ridge[valid]) - np.nanmin(ridge[valid]) + 1e-12)
        y_anchor = ymax - 0.12 * (ymax - ymin)
        amplitude = 0.08 * (ymax - ymin)
        ridge_scaled = y_anchor + ridge_norm * amplitude
        axA.plot(xc[valid], ridge_scaled[valid], linewidth=3.0, color="black")
    axA.text(0.04, 0.95, "boundary\n(high curvature)", transform=axA.transAxes, va="top")
    axA.text(0.43, 0.95, "transition\nzone", transform=axA.transAxes, va="top")
    axA.text(0.78, 0.95, "stable\ninterior", transform=axA.transAxes, va="top")
    axA.text(0.02, 0.02, "geometry defines regimes", transform=axA.transAxes, fontsize=9, alpha=0.7)
    axA.set_title("A. State space and phase structure")
    axA.set_xlabel("Distance to phase boundary")
    axA.set_ylabel("Log curvature")

    # Panel B — analytical
    decorate_regimes(axB, with_lines=True)
    scB = axB.scatter(x, y, c=work["distance_to_seam"], s=18, alpha=0.9)
    cbB = fig.colorbar(scB, ax=axB, shrink=0.86)
    cbB.set_label("Distance to phase boundary")
    axB.set_title("B. Distance field")
    axB.set_xlabel("Distance to phase boundary")
    axB.set_ylabel("Log curvature")

    # Panel C — conceptual field
    decorate_regimes(axC, with_lines=False)
    scC = axC.scatter(x, y, c=work["log_curvature"], s=18, alpha=0.9)
    cbC = fig.colorbar(scC, ax=axC, shrink=0.86)
    cbC.set_label("Log curvature")
    axC.set_title("C. Local curvature")
    axC.set_xlabel("Distance to phase boundary")
    axC.set_ylabel("Log curvature")

    # Panel D — money panel
    decorate_regimes(axD, with_lines=True)
    t_vmin = float(np.nanpercentile(t, 5))
    t_vmax = float(np.nanpercentile(t, 95))
    scD = axD.scatter(x, y, c=t, s=10, alpha=0.25, vmin=t_vmin, vmax=t_vmax)
    cbD = fig.colorbar(scD, ax=axD, shrink=0.86)
    cbD.set_label("Empirical transition probability (k-step)")

    xc2, ridge2, counts2 = _binned_mean_curve(x, t, bins=28, min_count=30, smooth_sigma=1.2)
    ridge2 = _monotone_nonincreasing(ridge2)
    valid2 = np.isfinite(ridge2)
    if np.any(valid2):
        axD_t = axD.twinx()
        axD_t.plot(xc2[valid2], ridge2[valid2], linewidth=3.0, color="black")
        axD_t.set_ylabel("Mean transition probability")
        axD_t.set_ylim(0.0, max(0.05, float(np.nanmax(ridge2[valid2]) * 1.15)))
        peak_x = float(xc2[valid2][np.nanargmax(ridge2[valid2])])
        axD.axvline(peak_x, linestyle="--", linewidth=1.5, alpha=0.7)
        label_idx = max(0, min(len(xc2[valid2]) - 1, int(len(xc2[valid2]) * 0.35)))
        axD_t.text(
            xc2[valid2][label_idx],
            ridge2[valid2][label_idx],
            "transition ridge",
            fontsize=10,
            rotation=-25,
            va="bottom",
        )
    axD.text(0.02, 0.02, "transitions localize on ridge", transform=axD.transAxes, fontsize=9, alpha=0.7)
    axD.set_title("D. Transition localization")
    axD.set_xlabel("Distance to phase boundary")
    axD.set_ylabel("Log curvature")

    # panel letters
    for label, ax in zip(["A", "B", "C", "D"], [axA, axB, axC, axD]):
        ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=14)

    fig.suptitle("Figure 1 — Geometric structure of the state space", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render polished Figure 1 opener (v3).")
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
    )
    parser.add_argument("--outdir", default="outputs/fim_figure_1")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    outpath = outdir / "figure_1_geometric_structure_v3.png"
    render_figure(df, outpath)

    print(outpath)


if __name__ == "__main__":
    main()
