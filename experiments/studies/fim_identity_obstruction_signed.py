#!/usr/bin/env python3
from __future__ import annotations

"""
Derive a signed, orientation-aware local obstruction field from holonomy.

Outputs:
- identity_obstruction_signed_nodes.csv
- identity_obstruction_signed_on_grid.png
- identity_obstruction_signed_abs_on_grid.png
- identity_obstruction_signed_alignment.csv

Run:
    PYTHONPATH=src .venv/bin/python experiments/studies/fim_identity_obstruction_signed.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.topology.identity_obstruction import (
    IdentityObstructionConfig,
    build_identity_obstruction_table,
    load_identity_obstruction_inputs,
)


def grid_from_values(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_vals = np.sort(pd.to_numeric(df["r"], errors="coerce").dropna().unique())
    a_vals = np.sort(pd.to_numeric(df["alpha"], errors="coerce").dropna().unique())

    grid = (
        df.pivot_table(index="r", columns="alpha", values=value_col, aggfunc="mean")
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
    cmap: str = "viridis",
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


def corr_summary(df: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    work = df[[x, y]].dropna()
    if len(work) < 3:
        return {
            "metric_x": x,
            "metric_y": y,
            "n": len(work),
            "pearson": np.nan,
            "spearman": np.nan,
        }
    return {
        "metric_x": x,
        "metric_y": y,
        "n": int(len(work)),
        "pearson": float(work[x].corr(work[y], method="pearson")),
        "spearman": float(work[x].corr(work[y], method="spearman")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a signed transport-based local obstruction field from holonomy."
    )
    parser.add_argument(
        "--identity-nodes-csv",
        default="outputs/fim_identity/identity_field_nodes.csv",
    )
    parser.add_argument(
        "--holonomy-cells-csv",
        default="outputs/fim_identity_holonomy/identity_holonomy_cells.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_identity_obstruction",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, cells = load_identity_obstruction_inputs(
        identity_nodes_csv=args.identity_nodes_csv,
        holonomy_cells_csv=args.holonomy_cells_csv,
    )

    obs_df = build_identity_obstruction_table(
        nodes=nodes,
        cells=cells,
        config=IdentityObstructionConfig(
            weight_signed_by_abs_holonomy=True,
        ),
    )

    # merge legacy spin only for comparison
    spin_df = pd.read_csv(args.identity_nodes_csv).copy()
    spin_df["node_id"] = pd.to_numeric(spin_df["node_id"], errors="coerce").astype("Int64").astype(str)
    spin_df["identity_spin"] = pd.to_numeric(spin_df["identity_spin"], errors="coerce")
    spin_df["abs_identity_spin"] = spin_df["identity_spin"].abs()

    obs_df = obs_df.merge(
        spin_df[["node_id", "identity_spin", "abs_identity_spin"]],
        on="node_id",
        how="left",
    )

    signed_col = "obstruction_signed_weighted_holonomy"
    obs_df["abs_obstruction_signed_weighted_holonomy"] = obs_df[signed_col].abs()

    out_csv = outdir / "identity_obstruction_signed_nodes.csv"
    obs_df.to_csv(out_csv, index=False)

    # signed field map
    r_vals, a_vals, signed_grid = grid_from_values(obs_df, signed_col)
    signed_abs_max = float(np.nanmax(np.abs(signed_grid)))
    if not np.isfinite(signed_abs_max) or signed_abs_max <= 0:
        signed_abs_max = 1.0

    render_grid_map(
        r_vals,
        a_vals,
        signed_grid,
        outpath=outdir / "identity_obstruction_signed_on_grid.png",
        title="Signed Transport-Derived Local Obstruction",
        colorbar_label="signed weighted holonomy",
        cmap="coolwarm",
        vmin=-signed_abs_max,
        vmax=signed_abs_max,
    )

    # abs field map
    _, _, abs_grid = grid_from_values(obs_df, "abs_obstruction_signed_weighted_holonomy")
    render_grid_map(
        r_vals,
        a_vals,
        abs_grid,
        outpath=outdir / "identity_obstruction_signed_abs_on_grid.png",
        title="Absolute Signed Transport-Derived Local Obstruction",
        colorbar_label="|signed weighted holonomy|",
        cmap="viridis",
    )

    align = pd.DataFrame(
        [
            corr_summary(obs_df, "obstruction_signed_weighted_holonomy", "identity_spin"),
            corr_summary(obs_df, "obstruction_mean_holonomy", "identity_spin"),
            corr_summary(obs_df, "obstruction_signed_sum_holonomy", "identity_spin"),
            corr_summary(obs_df, "abs_obstruction_signed_weighted_holonomy", "abs_identity_spin"),
        ]
    )
    align.to_csv(outdir / "identity_obstruction_signed_alignment.csv", index=False)

    print(out_csv)
    print(outdir / "identity_obstruction_signed_on_grid.png")
    print(outdir / "identity_obstruction_signed_abs_on_grid.png")
    print(outdir / "identity_obstruction_signed_alignment.csv")


if __name__ == "__main__":
    main()
