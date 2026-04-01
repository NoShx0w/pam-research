#!/usr/bin/env python3
from __future__ import annotations

"""
Render first-pass PAM identity maps.

Outputs:
- identity_magnitude_on_grid.png
- identity_spin_on_grid.png
- identity_magnitude_on_mds.png
- identity_maps_nodes.csv

Run:
    PYTHONPATH=src .venv/bin/python experiments/studies/fim_identity_maps.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_identity_nodes(identity_nodes_csv: str | Path, mds_csv: str | Path) -> pd.DataFrame:
    ident = pd.read_csv(identity_nodes_csv)
    mds = pd.read_csv(mds_csv)

    required_ident = {"node_id", "i", "j", "r", "alpha", "identity_magnitude", "identity_spin"}
    missing_ident = required_ident - set(ident.columns)
    if missing_ident:
        raise ValueError(
            f"identity field nodes CSV missing required columns: {sorted(missing_ident)}"
        )

    required_mds = {"r", "alpha", "mds1", "mds2"}
    missing_mds = required_mds - set(mds.columns)
    if missing_mds:
        raise ValueError(f"MDS CSV missing required columns: {sorted(missing_mds)}")

    df = ident.merge(
        mds[["r", "alpha", "mds1", "mds2"]],
        on=["r", "alpha"],
        how="left",
    ).copy()

    numeric_cols = [
        "i",
        "j",
        "r",
        "alpha",
        "identity_magnitude",
        "identity_spin",
        "mds1",
        "mds2",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def grid_from_node_values(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = df.copy()
    work["i"] = pd.to_numeric(work["i"], errors="raise").astype(int)
    work["j"] = pd.to_numeric(work["j"], errors="raise").astype(int)

    r_vals = np.sort(pd.to_numeric(work["r"], errors="coerce").unique())
    a_vals = np.sort(pd.to_numeric(work["alpha"], errors="coerce").unique())

    grid = (
        work.pivot_table(index="r", columns="alpha", values=value_col, aggfunc="mean")
        .reindex(index=r_vals, columns=a_vals)
        .to_numpy(dtype=float)
    )
    return r_vals, a_vals, grid


def render_grid_map(
    r_vals: np.ndarray,
    a_vals: np.ndarray,
    grid: np.ndarray,
    *,
    outpath: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    extent = [float(a_vals.min()), float(a_vals.max()), float(r_vals.min()), float(r_vals.max())]

    plt.figure(figsize=(7.2, 5.4))
    plt.imshow(
        grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=colorbar_label)
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def render_mds_scatter(
    df: pd.DataFrame,
    *,
    value_col: str,
    outpath: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    work = df[["mds1", "mds2", value_col]].dropna().copy()

    plt.figure(figsize=(7.0, 5.6))
    sc = plt.scatter(
        work["mds1"],
        work["mds2"],
        c=work[value_col],
        s=70,
        alpha=0.9,
        cmap=cmap,
    )
    plt.xlabel("MDS 1")
    plt.ylabel("MDS 2")
    plt.title(title)
    plt.colorbar(sc, label=colorbar_label)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render first-pass PAM identity maps.")
    parser.add_argument(
        "--identity-nodes-csv",
        default="outputs/fim_identity/identity_field_nodes.csv",
    )
    parser.add_argument(
        "--mds-csv",
        default="outputs/fim_mds/mds_coords.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_identity_maps",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_identity_nodes(
        identity_nodes_csv=args.identity_nodes_csv,
        mds_csv=args.mds_csv,
    )
    df.to_csv(outdir / "identity_maps_nodes.csv", index=False)

    # Magnitude on grid
    r_vals, a_vals, mag_grid = grid_from_node_values(df, "identity_magnitude")
    render_grid_map(
        r_vals,
        a_vals,
        mag_grid,
        outpath=outdir / "identity_magnitude_on_grid.png",
        title="Identity Magnitude on Grid",
        colorbar_label="identity magnitude",
        cmap="viridis",
    )

    # Spin on grid
    _, _, spin_grid = grid_from_node_values(df, "identity_spin")
    spin_abs_max = float(
        max(
            abs(pd.to_numeric(df["identity_spin"], errors="coerce").min()),
            abs(pd.to_numeric(df["identity_spin"], errors="coerce").max()),
        )
    )
    if not np.isfinite(spin_abs_max) or spin_abs_max <= 0:
        spin_abs_max = 1.0

    render_grid_map(
        r_vals,
        a_vals,
        spin_grid,
        outpath=outdir / "identity_spin_on_grid.png",
        title="Identity Spin on Grid",
        colorbar_label="identity spin",
        cmap="coolwarm",
        vmin=-spin_abs_max,
        vmax=spin_abs_max,
    )

    # Magnitude on MDS
    render_mds_scatter(
        df,
        value_col="identity_magnitude",
        outpath=outdir / "identity_magnitude_on_mds.png",
        title="Identity Magnitude on MDS",
        colorbar_label="identity magnitude",
        cmap="viridis",
    )

    print(outdir / "identity_maps_nodes.csv")
    print(outdir / "identity_magnitude_on_grid.png")
    print(outdir / "identity_spin_on_grid.png")
    print(outdir / "identity_magnitude_on_mds.png")


if __name__ == "__main__":
    main()
