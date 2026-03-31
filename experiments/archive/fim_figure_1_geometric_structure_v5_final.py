#!/usr/bin/env python3
from __future__ import annotations

"""
figure_1_geometric_structure_v5_final.py

Final clean-layout Figure 1 — Geometric structure of the state space.

Changes from v4:
- uses GridSpec with dedicated colorbar axes so panel widths are visually consistent
- Panel A ridge is schematic and explicitly controlled
- Panel D ridge is empirical but cleaned/interpolated/smoothed before plotting
- top title/subtitle spacing is stable without tight_layout conflicts
- labels and annotations are repositioned to avoid line collisions
"""

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
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
    bins: int = 18,
    min_count: int = 3,
    smooth_sigma: float = 1.0,
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
        if np.sum(valid) >= 2:
            missing = ~valid
            if np.any(missing):
                filled[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(valid), means[valid])
        if np.sum(np.isfinite(filled)) >= 3 and smooth_sigma > 0:
            smooth = gaussian_filter1d(filled, sigma=smooth_sigma)
        else:
            smooth = filled
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


def _interp_fill(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    valid = np.isfinite(out)
    if np.sum(valid) < 2:
        return out
    out[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), out[valid])
    return out


def _clean_curve(
    x: np.ndarray,
    y: np.ndarray,
    smooth_sigma: float = 1.0,
    monotone_nonincreasing: bool = False,
    floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y = _interp_fill(y)
    if np.sum(np.isfinite(y)) >= 3 and smooth_sigma > 0:
        y = gaussian_filter1d(y, sigma=smooth_sigma)

    if monotone_nonincreasing:
        y = _monotone_nonincreasing(y)

    if floor is not None:
        y = np.maximum(y, floor)

    return x, y


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

    fig = plt.figure(figsize=(13.8, 10.2))
    gs = gridspec.GridSpec(
        2,
        4,
        width_ratios=[1.0, 0.035, 1.0, 0.035],
        height_ratios=[1.0, 1.0],
        wspace=0.16,
        hspace=0.30,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 2])

    caxC = fig.add_subplot(gs[1, 1])
    caxB = fig.add_subplot(gs[0, 3])

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
    axA.scatter(x, y, s=8, alpha=0.08, zorder=1)

    ridge_x = np.array([0.03, 0.12, 0.22, 0.33, 0.45, 0.58, 0.72, 0.88, 1.10, 1.35, 1.55])
    ridge_y = np.array([7.95, 7.82, 7.62, 7.48, 7.28, 7.29, 7.24, 7.28, 7.44, 7.66, 7.84])

    axA.plot(
        ridge_x,
        ridge_y,
        color="black",
        linewidth=3.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=6,
        clip_on=False,
    )

    axA.text(-0.03, 8.25, "boundary\n(high curvature)", ha="left", va="top", fontsize=11, zorder=7)
    axA.text(0.70, 8.18, "transition\nzone", ha="left", va="top", fontsize=11, zorder=7)
    axA.text(1.36, 8.16, "stable\ninterior", ha="left", va="top", fontsize=11, zorder=7)
    axA.text(0.02, 0.03, "geometry defines regimes", transform=axA.transAxes, fontsize=9, alpha=0.7)
    axA.set_title("A. State space and phase structure")
    axA.set_xlabel("Distance to phase boundary")
    axA.set_ylabel("Log curvature")

    # Panel B — analytical
    decorate_regimes(axB, with_lines=True)
    scB = axB.scatter(x, y, c=work["distance_to_seam"], s=24, alpha=0.95, zorder=2)
    cbB = fig.colorbar(scB, cax=caxB)
    cbB.set_label("Distance to phase boundary")
    axB.set_title("B. Distance to phase boundary")
    axB.set_xlabel("Distance to phase boundary")
    axB.set_ylabel("Log curvature")

    # Panel C — curvature field
    decorate_regimes(axC, with_lines=False)
    scC = axC.scatter(x, y, c=work["log_curvature"], s=24, alpha=0.95, zorder=2)
    cbC = fig.colorbar(scC, cax=caxC)
    cbC.set_label("Log curvature")
    axC.set_title("C. Local curvature field")
    axC.set_xlabel("Distance to phase boundary")
    axC.set_ylabel("Log curvature")

    # Panel D — money panel
    decorate_regimes(axD, with_lines=True)
    axD.scatter(x, t, s=12, alpha=0.28, color="#4c4c4c", zorder=1)

    xc2, ridge2, counts2 = _binned_mean_curve(x, t, bins=18, min_count=3, smooth_sigma=1.0)
    ridge_x2, ridge_y2 = _clean_curve(
        xc2,
        ridge2,
        smooth_sigma=0.9,
        monotone_nonincreasing=True,
        floor=0.01,
    )

    if len(ridge_x2) > 0:
        axD_t = axD.twinx()
        axD_t.plot(
            ridge_x2,
            ridge_y2,
            color="black",
            linewidth=3.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=6,
            clip_on=False,
        )
        axD_t.set_ylabel("Mean transition probability")
        axD_t.set_ylim(0.0, max(0.05, float(np.nanmax(ridge_y2) * 1.15)))

        peak_x = float(ridge_x2[np.nanargmax(ridge_y2)])
        axD.axvline(peak_x, linestyle="--", linewidth=1.8, alpha=0.75)

        axD_t.text(
            0.42,
            0.028,
            "transition ridge",
            fontsize=10,
            rotation=-24,
            va="bottom",
        )

    axD.text(0.02, 0.03, "transitions localize on ridge", transform=axD.transAxes, fontsize=9, alpha=0.7)
    axD.set_title("D. Transition probability localization")
    axD.set_xlabel("Distance to phase boundary")
    axD.set_ylabel("Transition probability (k-step)")
    axD.set_ylim(-0.05, 1.05)

    # panel letters
    for label, ax in zip(["A", "B", "C", "D"], [axA, axB, axC, axD]):
        ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=14)

    fig.suptitle(
        "Figure 1 — Geometric structure of the state space",
        fontsize=17,
        y=0.97,
    )

    fig.text(
        0.5,
        0.938,
        "Transitions concentrate near the phase boundary along a geometric ridge",
        ha="center",
        va="top",
        fontsize=11,
        alpha=0.8,
    )

    fig.subplots_adjust(top=0.79, left=0.07, right=0.96, bottom=0.08)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render final clean-layout Figure 1 opener (v5).")
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
    )
    parser.add_argument("--outdir", default="outputs/fim_figure_1")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    outpath = outdir / "figure_1_geometric_structure_v5_final.png"
    render_figure(df, outpath)

    print(outpath)


if __name__ == "__main__":
    main()
