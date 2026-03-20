#!/usr/bin/env python3
from __future__ import annotations

"""
fim_supp_figure_2.py

Supplementary Figure S2 — Scaling collapse test

Tests whether transition probability collapses onto a 1D function of
combined geometric variables.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_scaling_variables(df):
    df = df.copy()

    # base variables
    d = df["distance_to_seam"].values
    k = df["scalar_curvature"].values

    # candidates (you can expand this later)
    df["Z1"] = k / (1 + d)
    df["Z2"] = np.log1p(k) / (1 + d)
    df["Z3"] = k * np.exp(-d)
    df["Z4"] = np.log1p(k) - d

    return df


def bin_and_average(x, y, bins=30):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(x, edges) - 1

    means = []
    counts = []

    for i in range(bins):
        mask = idx == i
        if np.sum(mask) < 20:
            means.append(np.nan)
            counts.append(0)
        else:
            means.append(np.nanmean(y[mask]))
            counts.append(np.sum(mask))

    return centers, np.array(means), np.array(counts)


def plot_collapse(df, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    candidates = ["Z1", "Z2", "Z3", "Z4"]
    axes = axes.flatten()

    for ax, z in zip(axes, candidates):
        x = df[z].values
        y = df["transition_within_k"].values

        centers, means, counts = bin_and_average(x, y)

        ax.plot(centers, means, linewidth=2)
        ax.set_title(z)
        ax.set_xlabel("scaled variable")
        ax.set_ylabel("P(transition)")
        ax.set_ylim(np.nanmin(means), np.nanmax(means))

    fig.suptitle("Supplementary Figure S2 — Scaling collapse test")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition-labeled-csv",
        default="outputs/fim_transition_rate/transition_rate_labeled.csv",
    )
    parser.add_argument("--outdir", default="outputs/fim_supp")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    df = _safe_numeric(
        df,
        ["distance_to_seam", "scalar_curvature", "transition_within_k"],
    )
    df = df.dropna()

    df = build_scaling_variables(df)

    outpath = outdir / "supp_figure_2_scaling_collapse.png"
    plot_collapse(df, outpath)

    print(outpath)


if __name__ == "__main__":
    main()
