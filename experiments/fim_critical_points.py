import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or not np.isfinite(s):
        return np.zeros_like(x)
    return (x - m) / s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--phase-distance-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/fim_critical")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fim = pd.read_csv(args.fim_csv)
    curv = pd.read_csv(args.curvature_csv)
    dist = pd.read_csv(args.phase_distance_csv)

    df = fim.merge(
        curv[["r", "alpha", "scalar_curvature"]],
        on=["r", "alpha"],
        how="left",
    ).merge(
        dist[["r", "alpha", "distance_to_seam"]],
        on=["r", "alpha"],
        how="left",
    )

    df["log_det"] = np.log10(np.clip(df["fim_det"].to_numpy(dtype=float), 1e-12, None))
    df["log_abs_curv"] = np.log10(1.0 + np.abs(df["scalar_curvature"].to_numpy(dtype=float)))
    df["criticality"] = (
        zscore(df["log_det"])
        + zscore(df["log_abs_curv"])
        - zscore(df["distance_to_seam"])
    )

    out_csv = outdir / "criticality_surface.csv"
    df.to_csv(out_csv, index=False)

    crit_pts = df.sort_values("criticality", ascending=False).head(args.top_k).copy()
    crit_pts.to_csv(outdir / "critical_points.csv", index=False)

    r_vals = np.sort(df["r"].unique())
    a_vals = np.sort(df["alpha"].unique())

    grid = (
        df.pivot_table(index="r", columns="alpha", values="criticality", aggfunc="mean")
        .reindex(index=r_vals, columns=a_vals)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_vals])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title("Criticality score")
    fig.colorbar(im, ax=ax, label="criticality")
    fig.tight_layout()
    fig.savefig(outdir / "criticality_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df["alpha"], df["r"], s=20, alpha=0.25)
    ax.scatter(crit_pts["alpha"], crit_pts["r"], s=100)
    for _, row in crit_pts.iterrows():
        ax.text(row["alpha"], row["r"], f"({row['r']:.2f},{row['alpha']:.3f})", fontsize=8)
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title("Top critical points")
    fig.tight_layout()
    fig.savefig(outdir / "critical_points_overlay.png", dpi=180)
    plt.close(fig)

    print(out_csv)
    print(outdir / "critical_points.csv")


if __name__ == "__main__":
    main()
