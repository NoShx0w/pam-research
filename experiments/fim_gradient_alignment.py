#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_nodes(mds_csv: str | Path, phase_csv: str | Path, lazarus_csv: str | Path) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    phase = pd.read_csv(phase_csv)
    laz = pd.read_csv(lazarus_csv)

    df = mds.copy()
    if {"mds1", "mds2"}.issubset(phase.columns):
        keep_phase = [c for c in ["r", "alpha", "mds1", "mds2", "signed_phase", "distance_to_seam"] if c in phase.columns]
        df = phase[keep_phase].copy()
    else:
        keep_phase = [c for c in ["r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]
        df = df.merge(phase[keep_phase], on=["r", "alpha"], how="left")

    keep_laz = [c for c in ["r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns]
    df = df.merge(laz[keep_laz], on=["r", "alpha"], how="left")
    return df


def estimate_local_gradients(
    df: pd.DataFrame,
    value_col: str,
    x_col: str = "mds1",
    y_col: str = "mds2",
    k: int = 8,
) -> pd.DataFrame:
    work = df[[x_col, y_col, value_col]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    X = work[[x_col, y_col]].to_numpy(dtype=float)
    z = work[value_col].to_numpy(dtype=float)

    gx = np.full(len(work), np.nan, dtype=float)
    gy = np.full(len(work), np.nan, dtype=float)
    gnorm = np.full(len(work), np.nan, dtype=float)

    for i in range(len(work)):
        if not np.isfinite(z[i]):
            continue

        d2 = np.sum((X - X[i]) ** 2, axis=1)
        order = np.argsort(d2)
        neigh = order[1 : min(k + 1, len(order))]
        neigh = neigh[np.isfinite(z[neigh])]
        if len(neigh) < 3:
            continue

        dx = X[neigh, 0] - X[i, 0]
        dy = X[neigh, 1] - X[i, 1]
        dz = z[neigh] - z[i]

        A = np.column_stack([dx, dy])
        try:
            beta, *_ = np.linalg.lstsq(A, dz, rcond=None)
        except np.linalg.LinAlgError:
            continue

        gx[i] = float(beta[0])
        gy[i] = float(beta[1])
        gnorm[i] = float(np.sqrt(beta[0] ** 2 + beta[1] ** 2))

    return pd.DataFrame(
        {
            f"grad_{value_col}_x": gx,
            f"grad_{value_col}_y": gy,
            f"grad_{value_col}_norm": gnorm,
        },
        index=df.index,
    )


def build_gradient_alignment(df: pd.DataFrame):
    work = df.copy()
    work["signed_phase"] = pd.to_numeric(work["signed_phase"], errors="coerce")
    work["lazarus_score"] = pd.to_numeric(work["lazarus_score"], errors="coerce")
    work["distance_to_seam"] = pd.to_numeric(work["distance_to_seam"], errors="coerce")

    phase_grads = estimate_local_gradients(work, "signed_phase")
    laz_grads = estimate_local_gradients(work, "lazarus_score")
    work = pd.concat([work, phase_grads, laz_grads], axis=1)

    gx1 = work["grad_signed_phase_x"].to_numpy(dtype=float)
    gy1 = work["grad_signed_phase_y"].to_numpy(dtype=float)
    gx2 = work["grad_lazarus_score_x"].to_numpy(dtype=float)
    gy2 = work["grad_lazarus_score_y"].to_numpy(dtype=float)

    dot = gx1 * gx2 + gy1 * gy2
    n1 = np.sqrt(gx1 ** 2 + gy1 ** 2)
    n2 = np.sqrt(gx2 ** 2 + gy2 ** 2)
    denom = n1 * n2
    cos = np.where((denom > 0) & np.isfinite(denom), dot / denom, np.nan)

    work["gradient_dot"] = dot
    work["gradient_cosine_alignment"] = np.clip(cos, -1.0, 1.0)
    work["gradient_magnitude_product"] = n1 * n2

    median_laz = float(work["lazarus_score"].median())
    work["lazarus_group"] = work["lazarus_score"].apply(
        lambda x: "high" if pd.notna(x) and x >= median_laz else "low"
    )

    summary = pd.DataFrame(
        {
            "n_nodes": [int(len(work))],
            "mean_alignment_all": [pd.to_numeric(work["gradient_cosine_alignment"], errors="coerce").mean()],
            "median_alignment_all": [pd.to_numeric(work["gradient_cosine_alignment"], errors="coerce").median()],
            "mean_alignment_high_lazarus": [
                pd.to_numeric(
                    work.loc[work["lazarus_group"] == "high", "gradient_cosine_alignment"], errors="coerce"
                ).mean()
            ],
            "mean_alignment_low_lazarus": [
                pd.to_numeric(
                    work.loc[work["lazarus_group"] == "low", "gradient_cosine_alignment"], errors="coerce"
                ).mean()
            ],
            "mean_abs_alignment_near_seam": [
                pd.to_numeric(
                    work.loc[work["distance_to_seam"] <= work["distance_to_seam"].median(), "gradient_cosine_alignment"],
                    errors="coerce",
                ).abs().mean()
            ],
            "mean_abs_alignment_far_from_seam": [
                pd.to_numeric(
                    work.loc[work["distance_to_seam"] > work["distance_to_seam"].median(), "gradient_cosine_alignment"],
                    errors="coerce",
                ).abs().mean()
            ],
        }
    )
    return work, summary


def render_plots(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plot_df = df[["distance_to_seam", "gradient_cosine_alignment"]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(plot_df["distance_to_seam"], plot_df["gradient_cosine_alignment"], alpha=0.75)
    ax.set_xlabel("distance_to_seam")
    ax.set_ylabel("cosine(∇phase, ∇lazarus)")
    ax.set_title("Gradient alignment vs distance to seam")
    fig.tight_layout()
    fig.savefig(outdir / "gradient_alignment_vs_seam_distance.png", dpi=220)
    plt.close(fig)

    plot_df = df[["lazarus_score", "gradient_cosine_alignment"]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(plot_df["lazarus_score"], plot_df["gradient_cosine_alignment"], alpha=0.75)
    ax.set_xlabel("lazarus_score")
    ax.set_ylabel("cosine(∇phase, ∇lazarus)")
    ax.set_title("Gradient alignment vs Lazarus score")
    fig.tight_layout()
    fig.savefig(outdir / "gradient_alignment_vs_lazarus.png", dpi=220)
    plt.close(fig)

    vals = pd.to_numeric(df["gradient_cosine_alignment"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(vals, bins=20)
    ax.set_xlabel("cosine(∇phase, ∇lazarus)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of gradient alignment")
    fig.tight_layout()
    fig.savefig(outdir / "gradient_alignment_hist.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate local gradient alignment between phase and Lazarus fields.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--outdir", default="outputs/fim_gradient_alignment")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(args.mds_csv, args.phase_csv, args.lazarus_csv)
    grad_df, summary = build_gradient_alignment(nodes)

    grad_df.to_csv(outdir / "gradient_alignment_nodes.csv", index=False)
    summary.to_csv(outdir / "gradient_alignment_summary.csv", index=False)
    render_plots(grad_df, outdir)

    print(outdir / "gradient_alignment_nodes.csv")
    print(outdir / "gradient_alignment_summary.csv")
    print(outdir / "gradient_alignment_vs_seam_distance.png")
    print(outdir / "gradient_alignment_vs_lazarus.png")
    print(outdir / "gradient_alignment_hist.png")


if __name__ == "__main__":
    main()
