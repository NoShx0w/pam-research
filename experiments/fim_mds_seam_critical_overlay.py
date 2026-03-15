
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Overlay phase seam and critical points on the PAM MDS manifold."
    )
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--critical-csv", default="outputs/fim_critical/critical_points.csv")
    parser.add_argument("--outdir", default="outputs/fim_critical")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mds = pd.read_csv(args.mds_csv)
    seam = pd.read_csv(args.seam_csv)
    crit = pd.read_csv(args.critical_csv)

    # Keep only columns needed for matching, avoid duplicate column collisions
    seam_small = seam.drop(columns=[c for c in seam.columns if c not in ["r", "alpha", "mds1", "mds2"]], errors="ignore")
    if "mds1" not in seam_small.columns or "mds2" not in seam_small.columns:
        seam_small = seam[["r", "alpha"]].merge(mds[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")

    crit_small = crit.drop(columns=[c for c in crit.columns if c not in ["r", "alpha", "criticality"]], errors="ignore")
    crit_small = crit_small.merge(mds[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")

    # Order seam points approximately along the manifold using MDS1
    seam_small = seam_small.dropna(subset=["mds1", "mds2"]).drop_duplicates(subset=["r", "alpha"]).sort_values("mds1")

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    # Background manifold nodes
    ax.scatter(
        mds["mds1"],
        mds["mds2"],
        s=28,
        alpha=0.28,
        label="manifold nodes",
    )

    # Seam polyline and nodes
    if not seam_small.empty:
        ax.plot(
            seam_small["mds1"],
            seam_small["mds2"],
            linewidth=2.5,
            label="phase seam",
        )
        ax.scatter(
            seam_small["mds1"],
            seam_small["mds2"],
            s=90,
            label="seam points",
        )

    # Critical points
    crit_small = crit_small.dropna(subset=["mds1", "mds2"]).drop_duplicates(subset=["r", "alpha"])
    if "criticality" in crit_small.columns:
        sc = ax.scatter(
            crit_small["mds1"],
            crit_small["mds2"],
            c=crit_small["criticality"],
            s=160,
            cmap="plasma",
            marker="*",
            label="critical points",
            zorder=5,
        )
        fig.colorbar(sc, ax=ax, label="criticality")
    else:
        ax.scatter(
            crit_small["mds1"],
            crit_small["mds2"],
            s=160,
            marker="*",
            label="critical points",
            zorder=5,
        )

    for _, row in crit_small.iterrows():
        ax.text(
            row["mds1"],
            row["mds2"],
            f"({row['r']:.2f},{row['alpha']:.3f})",
            fontsize=8,
            alpha=0.8,
        )

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("PAM manifold with phase seam and critical points")
    ax.legend(loc="best")
    fig.tight_layout()

    outpath = outdir / "mds_seam_critical_overlay.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    print(outpath)


if __name__ == "__main__":
    main()
