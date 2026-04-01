#!/usr/bin/env python3
from __future__ import annotations

"""
Canonical identity transport comparison panel.

Renders a side-by-side panel of:
- node-based identity spin on the grid
- cell-based absolute holonomy on the grid

Outputs:
- outputs/fim_identity_holonomy/identity_transport_panel.png

Run:
    PYTHONPATH=src .venv/bin/python experiments/studies/fim_identity_transport_panel.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_identity_spin(nodes_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(nodes_csv).copy()

    required = {"r", "alpha", "identity_spin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"identity field nodes CSV missing required columns: {sorted(missing)}"
        )

    for col in ["r", "alpha", "identity_spin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_identity_holonomy(cells_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(cells_csv).copy()

    required = {"r_center", "alpha_center", "abs_holonomy_residual"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"identity holonomy cells CSV missing required columns: {sorted(missing)}"
        )

    for col in ["r_center", "alpha_center", "abs_holonomy_residual", "holonomy_residual"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def grid_from_values(
    df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_vals = np.sort(pd.to_numeric(df[row_col], errors="coerce").dropna().unique())
    a_vals = np.sort(pd.to_numeric(df[col_col], errors="coerce").dropna().unique())

    grid = (
        df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
        .reindex(index=r_vals, columns=a_vals)
        .to_numpy(dtype=float)
    )
    return r_vals, a_vals, grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Render canonical identity transport comparison panel.")
    parser.add_argument(
        "--identity-nodes-csv",
        default="outputs/fim_identity/identity_field_nodes.csv",
    )
    parser.add_argument(
        "--identity-holonomy-cells-csv",
        default="outputs/fim_identity_holonomy/identity_holonomy_cells.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_identity_holonomy",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    spin_df = load_identity_spin(args.identity_nodes_csv)
    hol_df = load_identity_holonomy(args.identity_holonomy_cells_csv)

    spin_r, spin_a, spin_grid = grid_from_values(
        spin_df,
        row_col="r",
        col_col="alpha",
        value_col="identity_spin",
    )
    hol_r, hol_a, hol_grid = grid_from_values(
        hol_df,
        row_col="r_center",
        col_col="alpha_center",
        value_col="abs_holonomy_residual",
    )

    spin_abs_max = float(np.nanmax(np.abs(spin_grid)))
    if not np.isfinite(spin_abs_max) or spin_abs_max <= 0:
        spin_abs_max = 1.0

    spin_extent = [
        float(spin_a.min()),
        float(spin_a.max()),
        float(spin_r.min()),
        float(spin_r.max()),
    ]
    hol_extent = [
        float(hol_a.min()),
        float(hol_a.max()),
        float(hol_r.min()),
        float(hol_r.max()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    im0 = axes[0].imshow(
        spin_grid,
        origin="lower",
        extent=spin_extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-spin_abs_max,
        vmax=spin_abs_max,
    )
    axes[0].set_title("Identity Spin on Grid")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("r")
    fig.colorbar(im0, ax=axes[0], shrink=0.9, label="identity spin")

    im1 = axes[1].imshow(
        hol_grid,
        origin="lower",
        extent=hol_extent,
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_title("Absolute Identity Holonomy on Grid")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("r")
    fig.colorbar(im1, ax=axes[1], shrink=0.9, label="|holonomy residual|")

    fig.suptitle("Identity Spin and Holonomy Panel", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "identity_transport_panel.png", dpi=220)
    plt.close(fig)

    print(outdir / "identity_transport_panel.png")


if __name__ == "__main__":
    main()
