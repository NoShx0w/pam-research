#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(
    *,
    identity_nodes_csv: str | Path,
    criticality_csv: str | Path,
    phase_distance_csv: str | Path,
) -> pd.DataFrame:
    ident = pd.read_csv(identity_nodes_csv)
    crit = pd.read_csv(criticality_csv)
    phase = pd.read_csv(phase_distance_csv)

    df = (
        ident.merge(
            crit[["r", "alpha", "criticality", "scalar_curvature"]],
            on=["r", "alpha"],
            how="left",
        )
        .merge(
            phase[["node_id", "distance_to_seam"]],
            on="node_id",
            how="left",
            suffixes=("", "_phase"),
        )
        .copy()
    )

    for col in [
        "r",
        "alpha",
        "identity_magnitude",
        "identity_spin",
        "criticality",
        "scalar_curvature",
        "distance_to_seam",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def grid_from_values(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_vals = np.sort(df["r"].dropna().unique())
    a_vals = np.sort(df["alpha"].dropna().unique())

    grid = (
        df.pivot_table(index="r", columns="alpha", values=value_col, aggfunc="mean")
        .reindex(index=r_vals, columns=a_vals)
        .to_numpy(dtype=float)
    )
    return r_vals, a_vals, grid


def render_panel(
    df: pd.DataFrame,
    outpath: Path,
) -> None:
    r_vals, a_vals, seam_grid = grid_from_values(df, "distance_to_seam")
    _, _, crit_grid = grid_from_values(df, "criticality")
    _, _, mag_grid = grid_from_values(df, "identity_magnitude")
    _, _, spin_grid = grid_from_values(df, "identity_spin")

    extent = [float(a_vals.min()), float(a_vals.max()), float(r_vals.min()), float(r_vals.max())]
    spin_abs_max = float(np.nanmax(np.abs(spin_grid)))
    if not np.isfinite(spin_abs_max) or spin_abs_max <= 0:
        spin_abs_max = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    im0 = axes[0, 0].imshow(seam_grid, origin="lower", extent=extent, aspect="auto")
    axes[0, 0].set_title("Distance to Seam")
    axes[0, 0].set_xlabel("alpha")
    axes[0, 0].set_ylabel("r")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.9)

    im1 = axes[0, 1].imshow(crit_grid, origin="lower", extent=extent, aspect="auto")
    axes[0, 1].set_title("Criticality")
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("r")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.9)

    im2 = axes[1, 0].imshow(mag_grid, origin="lower", extent=extent, aspect="auto")
    axes[1, 0].set_title("Identity Magnitude")
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("r")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.9)

    im3 = axes[1, 1].imshow(
        spin_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-spin_abs_max,
        vmax=spin_abs_max,
    )
    axes[1, 1].set_title("Identity Spin")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("r")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def build_singularity_table(df: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    work = df.copy()
    work["abs_identity_spin"] = work["identity_spin"].abs()
    cols = [
        "node_id",
        "r",
        "alpha",
        "identity_spin",
        "abs_identity_spin",
        "identity_magnitude",
        "criticality",
        "distance_to_seam",
        "scalar_curvature",
        "mds1",
        "mds2",
    ]
    keep = [c for c in cols if c in work.columns]
    out = (
        work.sort_values("abs_identity_spin", ascending=False)
        .loc[:, keep]
        .head(top_k)
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PAM identity diagnostics and singularity table.")
    parser.add_argument(
        "--identity-nodes-csv",
        default="outputs/fim_identity_maps/identity_maps_nodes.csv",
    )
    parser.add_argument(
        "--criticality-csv",
        default="outputs/fim_critical/criticality_surface.csv",
    )
    parser.add_argument(
        "--phase-distance-csv",
        default="outputs/fim_phase/phase_distance_to_seam.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_identity_diagnostics",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(
        identity_nodes_csv=args.identity_nodes_csv,
        criticality_csv=args.criticality_csv,
        phase_distance_csv=args.phase_distance_csv,
    )
    df.to_csv(outdir / "identity_diagnostics_nodes.csv", index=False)

    render_panel(
        df=df,
        outpath=outdir / "identity_diagnostic_panel.png",
    )

    singularities = build_singularity_table(df, top_k=args.top_k)
    singularities.to_csv(outdir / "identity_singularity_table.csv", index=False)

    print(outdir / "identity_diagnostics_nodes.csv")
    print(outdir / "identity_diagnostic_panel.png")
    print(outdir / "identity_singularity_table.csv")


if __name__ == "__main__":
    main()
