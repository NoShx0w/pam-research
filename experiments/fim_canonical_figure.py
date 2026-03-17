import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Build the canonical PAM manifold figure."
    )
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--critical-csv", default="outputs/fim_critical/critical_points.csv")
    parser.add_argument("--outdir", default="outputs/fim_report")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phase = pd.read_csv(args.signed_phase_csv)
    seam = pd.read_csv(args.seam_csv)
    curv = pd.read_csv(args.curvature_csv)
    crit = pd.read_csv(args.critical_csv)

    df = phase.merge(
        curv[["r", "alpha", "scalar_curvature"]],
        on=["r", "alpha"],
        how="left",
    )

    seam_small = seam.copy()
    if "mds1" not in seam_small.columns or "mds2" not in seam_small.columns:
        seam_small = seam_small[["r", "alpha"]].merge(
            df[["r", "alpha", "mds1", "mds2"]],
            on=["r", "alpha"],
            how="left",
        )
    seam_small = seam_small.dropna(subset=["mds1", "mds2"]).drop_duplicates(subset=["r", "alpha"])
    seam_small = seam_small.sort_values("mds1")

    crit_small = crit[["r", "alpha", "criticality"]].merge(
        df[["r", "alpha", "mds1", "mds2"]],
        on=["r", "alpha"],
        how="left",
    ).dropna(subset=["mds1", "mds2"]).drop_duplicates(subset=["r", "alpha"])

    # curvature glow
    glow = np.log10(1.0 + np.abs(df["scalar_curvature"].to_numpy(dtype=float)))
    glow = np.nan_to_num(glow, nan=np.nanmin(glow[np.isfinite(glow)]) if np.isfinite(glow).any() else 0.0)
    gmin, gmax = np.nanmin(glow), np.nanmax(glow)
    if gmax > gmin:
        glow_norm = (glow - gmin) / (gmax - gmin)
    else:
        glow_norm = np.zeros_like(glow)

    fig, ax = plt.subplots(figsize=(8.0, 6.4))

    # subtle curvature halo behind points
    ax.scatter(
        df["mds1"],
        df["mds2"],
        s=260 * (0.15 + glow_norm),
        c=glow_norm,
        cmap="inferno",
        alpha=0.18,
        linewidths=0,
        zorder=1,
    )

    # signed phase field
    sc = ax.scatter(
        df["mds1"],
        df["mds2"],
        c=df["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=72,
        linewidths=0.3,
        edgecolors=(0, 0, 0, 0.15),
        zorder=2,
    )

    # seam
    ax.plot(
        seam_small["mds1"],
        seam_small["mds2"],
        color="black",
        linewidth=2.4,
        zorder=4,
    )

    # critical points
    if "criticality" in crit_small.columns and not crit_small.empty:
        ax.scatter(
            crit_small["mds1"],
            crit_small["mds2"],
            c=crit_small["criticality"],
            cmap="plasma",
            s=170,
            marker="*",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
    else:
        ax.scatter(
            crit_small["mds1"],
            crit_small["mds2"],
            s=170,
            marker="*",
            color="gold",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )

    cbar = fig.colorbar(sc, ax=ax, label="signed phase")
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Phase Flow on the PAM Manifold")
    ax.grid(False)

    # optional minimal labels for critical points only
    for _, row in crit_small.iterrows():
        ax.text(
            row["mds1"] + 0.05,
            row["mds2"] + 0.03,
            f"({row['r']:.2f},{row['alpha']:.3f})",
            fontsize=8,
            alpha=0.8,
            zorder=6,
        )

    fig.tight_layout()
    outpath = outdir / "phase_flow_on_manifold.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

    print(outpath)


if __name__ == "__main__":
    main()
