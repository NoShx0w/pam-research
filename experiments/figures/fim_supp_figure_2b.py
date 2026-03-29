#!/usr/bin/env python3
from __future__ import annotations

"""
fim_supp_figure_2b.py

Supplementary Figure S2b — Conditional collapse by distance regime.

Plots P(transition within k steps) versus log curvature, stratified by
distance-to-seam regimes with binomial standard errors.
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


def assign_distance_regime(d: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            d < 0.2, "near",
            np.where(d < 0.6, "intermediate", "far")
        ),
        index=d.index,
    )


def binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 20, min_count: int = 30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(x, edges) - 1

    means, ses, counts = [], [], []
    for i in range(bins):
        m = idx == i
        n = int(np.sum(m))
        if n < min_count:
            means.append(np.nan)
            ses.append(np.nan)
            counts.append(n)
            continue
        p = float(np.nanmean(y[m]))
        se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n))
        means.append(p)
        ses.append(se)
        counts.append(n)

    return centers, np.array(means), np.array(ses), np.array(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Supplementary Figure S2b.")
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
    )
    parser.add_argument("--outdir", default="outputs/fim_supp")
    parser.add_argument("--bins", type=int, default=18)
    parser.add_argument("--min-count", type=int, default=40)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    df = _safe_numeric(df, ["distance_to_seam", "scalar_curvature", "transition_within_k"])
    df = df.dropna(subset=["distance_to_seam", "scalar_curvature", "transition_within_k"]).copy()

    df["scalar_curvature"] = df["scalar_curvature"].clip(lower=0.0)
    df["log_curvature"] = np.log10(1.0 + df["scalar_curvature"])
    df["distance_regime"] = assign_distance_regime(df["distance_to_seam"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    # Panel A: regime-specific curves
    ax = axes[0]
    summary_rows = []

    regime_order = ["near", "intermediate", "far"]
    for regime in regime_order:
        g = df[df["distance_regime"] == regime].copy()
        xc, ym, ys, counts = binned_curve(
            g["log_curvature"].values,
            g["transition_within_k"].values,
            bins=args.bins,
            min_count=args.min_count,
        )

        valid = np.isfinite(ym)
        if np.any(valid):
            ax.plot(xc[valid], ym[valid], linewidth=2, label=regime)
            ax.fill_between(
                xc[valid],
                ym[valid] - ys[valid],
                ym[valid] + ys[valid],
                alpha=0.15,
            )

        for xci, ymi, ysi, ni in zip(xc, ym, ys, counts):
            summary_rows.append(
                {
                    "regime": regime,
                    "log_curvature_center": xci,
                    "transition_rate": ymi,
                    "transition_se": ysi,
                    "count": ni,
                }
            )

    ax.set_xlabel("log10(1 + curvature)")
    ax.set_ylabel("P(transition within k steps)")
    ax.set_title("A. Conditional curves by distance regime")
    ax.legend()

    # Panel B: regime-level summary
    ax = axes[1]
    reg_summary = (
        df.groupby("distance_regime", observed=False)
        .agg(
            n=("transition_within_k", "count"),
            mean_transition=("transition_within_k", "mean"),
            mean_distance=("distance_to_seam", "mean"),
            mean_log_curvature=("log_curvature", "mean"),
        )
        .reset_index()
    )
    reg_summary["distance_regime"] = pd.Categorical(
        reg_summary["distance_regime"], categories=regime_order, ordered=True
    )
    reg_summary = reg_summary.sort_values("distance_regime")

    ax.bar(reg_summary["distance_regime"], reg_summary["mean_transition"])
    ax.set_ylim(0, max(0.05, float(reg_summary["mean_transition"].max()) * 1.25))
    ax.set_ylabel("mean P(transition)")
    ax.set_title("B. Regime-level transition rate")

    fig.suptitle("Supplementary Figure S2b — Conditional collapse by distance regime", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = outdir / "supp_figure_2b_conditional_collapse.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    summary_path = outdir / "supp_figure_2b_curve_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    reg_path = outdir / "supp_figure_2b_regime_summary.csv"
    reg_summary.to_csv(reg_path, index=False)

    print(fig_path)
    print(summary_path)
    print(reg_path)


if __name__ == "__main__":
    main()
