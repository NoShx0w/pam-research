#!/usr/bin/env python3
from __future__ import annotations

"""
First-pass PAM identity holonomy experiment.

Builds cell-based loop residuals over the parameter grid and compares them
to node-based identity spin.

Outputs:
- identity_holonomy_cells.csv
- identity_holonomy_on_grid.png
- identity_abs_holonomy_on_grid.png
- identity_holonomy_alignment.csv

Run:
    PYTHONPATH=src .venv/bin/python experiments/studies/fim_identity_holonomy.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.topology.identity_proxy import (
    IdentityProxyConfig,
    build_local_identity_graphs,
    load_identity_proxy_inputs,
)
from pam.topology.identity_transport import (
    IdentityHolonomyConfig,
    build_identity_holonomy_table,
    load_identity_transport_nodes,
)


def grid_from_cells(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_vals = np.sort(pd.to_numeric(df["r_center"], errors="coerce").dropna().unique())
    a_vals = np.sort(pd.to_numeric(df["alpha_center"], errors="coerce").dropna().unique())

    grid = (
        df.pivot_table(index="r_center", columns="alpha_center", values=value_col, aggfunc="mean")
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
    parser = argparse.ArgumentParser(description="Compute first-pass identity holonomy / loop residuals.")
    parser.add_argument("--identity-nodes-csv", default="outputs/fim_identity/identity_field_nodes.csv")
    parser.add_argument("--proxy-nodes-csv", default="outputs/fim_distance/fisher_nodes.csv")
    parser.add_argument("--proxy-edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--criticality-csv", default="outputs/fim_critical/criticality_surface.csv")
    parser.add_argument("--phase-distance-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/fim_identity_holonomy")
    parser.add_argument("--normalized-distance", action="store_true", default=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # rebuild same identity graphs used by field layer
    proxy_node_df, proxy_edge_df = load_identity_proxy_inputs(
        nodes_csv=args.proxy_nodes_csv,
        edges_csv=args.proxy_edges_csv,
        criticality_csv=args.criticality_csv,
        phase_distance_csv=args.phase_distance_csv,
    )
    identity_graphs = build_local_identity_graphs(
        node_df=proxy_node_df,
        edge_df=proxy_edge_df,
        config=IdentityProxyConfig(),
    )

    nodes_df = load_identity_transport_nodes(identity_nodes_csv=args.identity_nodes_csv)

    hol_df = build_identity_holonomy_table(
        nodes_df=nodes_df,
        identity_graphs=identity_graphs,
        config=IdentityHolonomyConfig(
            normalized_distance=args.normalized_distance,
        ),
    )
    hol_df.to_csv(outdir / "identity_holonomy_cells.csv", index=False)

    r_vals, a_vals, signed_grid = grid_from_cells(hol_df, "holonomy_residual")
    signed_abs_max = float(np.nanmax(np.abs(signed_grid))) if np.isfinite(np.nanmax(np.abs(signed_grid))) else 1.0
    if not np.isfinite(signed_abs_max) or signed_abs_max <= 0:
        signed_abs_max = 1.0

    render_grid_map(
        r_vals,
        a_vals,
        signed_grid,
        outpath=outdir / "identity_holonomy_on_grid.png",
        title="Identity Holonomy Residual on Grid",
        colorbar_label="holonomy residual",
        cmap="coolwarm",
        vmin=-signed_abs_max,
        vmax=signed_abs_max,
    )

    _, _, abs_grid = grid_from_cells(hol_df, "abs_holonomy_residual")
    render_grid_map(
        r_vals,
        a_vals,
        abs_grid,
        outpath=outdir / "identity_abs_holonomy_on_grid.png",
        title="Absolute Identity Holonomy on Grid",
        colorbar_label="|holonomy residual|",
        cmap="viridis",
    )

    summary = pd.DataFrame(
        [
            corr_summary(hol_df, "holonomy_residual", "mean_abs_corner_spin"),
            corr_summary(hol_df, "holonomy_residual", "max_abs_corner_spin"),
            corr_summary(hol_df, "abs_holonomy_residual", "mean_abs_corner_spin"),
            corr_summary(hol_df, "abs_holonomy_residual", "max_abs_corner_spin"),
        ]
    )
    summary.to_csv(outdir / "identity_holonomy_alignment.csv", index=False)

    print(outdir / "identity_holonomy_cells.csv")
    print(outdir / "identity_holonomy_on_grid.png")
    print(outdir / "identity_abs_holonomy_on_grid.png")
    print(outdir / "identity_holonomy_alignment.csv")


if __name__ == "__main__":
    main()
