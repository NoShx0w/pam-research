#!/usr/bin/env python3
from __future__ import annotations

"""
First-pass full PAM identity metric experiment.

Fits a local symmetric quadratic form over a 3x3 lattice stencil:
    d^2 ≈ g_rr Δr^2 + 2 g_ra Δr Δα + g_aa Δα^2

Outputs:
- identity_metric_full_nodes.csv
- identity_metric_full_det_on_grid.png
- identity_metric_full_cond_on_grid.png
- identity_metric_full_gra_on_grid.png
- identity_metric_full_alignment.csv

Run:
    PYTHONPATH=src .venv/bin/python experiments/studies/fim_identity_metric_full.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.topology.identity_metric_full import (
    IdentityMetricFullConfig,
    estimate_full_identity_metric_table,
    load_identity_metric_full_inputs,
)
from pam.topology.identity_proxy import (
    IdentityProxyConfig,
    build_local_identity_graphs,
    load_identity_proxy_inputs,
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
    parser = argparse.ArgumentParser(description="Estimate a first-pass full local identity metric.")
    parser.add_argument(
        "--identity-nodes-csv",
        default="outputs/fim_identity/identity_field_nodes.csv",
    )
    parser.add_argument(
        "--proxy-nodes-csv",
        default="outputs/fim_distance/fisher_nodes.csv",
    )
    parser.add_argument(
        "--proxy-edges-csv",
        default="outputs/fim_distance/fisher_edges.csv",
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
        default="outputs/fim_identity_metric_full",
    )
    parser.add_argument(
        "--patch-radius",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--normalized-distance",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build the same identity proxy graphs used by the identity field
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

    nodes_df = load_identity_metric_full_inputs(
        identity_nodes_csv=args.identity_nodes_csv,
    )

    metric_df = estimate_full_identity_metric_table(
        nodes_df=nodes_df,
        identity_graphs=identity_graphs,
        config=IdentityMetricFullConfig(
            patch_radius=args.patch_radius,
            ridge=args.ridge,
            min_samples=args.min_samples,
            normalized_distance=args.normalized_distance,
        ),
    )

    # merge spin from identity nodes
    ident_nodes = pd.read_csv(args.identity_nodes_csv).copy()
    ident_nodes["node_id"] = pd.to_numeric(ident_nodes["node_id"], errors="coerce").astype("Int64").astype(str)
    ident_nodes["identity_spin"] = pd.to_numeric(ident_nodes["identity_spin"], errors="coerce")
    metric_df = metric_df.merge(
        ident_nodes[["node_id", "identity_spin"]],
        on="node_id",
        how="left",
    )
    metric_df["abs_identity_spin"] = metric_df["identity_spin"].abs()

    metric_df.to_csv(outdir / "identity_metric_full_nodes.csv", index=False)

    # determinant map
    r_vals, a_vals, det_grid = grid_from_values(metric_df, "identity_metric_det")
    render_grid_map(
        r_vals,
        a_vals,
        det_grid,
        outpath=outdir / "identity_metric_full_det_on_grid.png",
        title="Full Identity Metric Determinant on Grid",
        colorbar_label="identity metric det",
        cmap="viridis",
    )

    # condition number map
    _, _, cond_grid = grid_from_values(metric_df, "identity_metric_cond")
    finite_cond = pd.to_numeric(metric_df["identity_metric_cond"], errors="coerce")
    finite_cond = finite_cond[np.isfinite(finite_cond)]
    vmax_cond = float(finite_cond.quantile(0.95)) if len(finite_cond) else None
    render_grid_map(
        r_vals,
        a_vals,
        cond_grid,
        outpath=outdir / "identity_metric_full_cond_on_grid.png",
        title="Full Identity Metric Condition Number on Grid",
        colorbar_label="identity metric cond",
        cmap="magma",
        vmax=vmax_cond,
    )

    # mixed term map
    _, _, gra_grid = grid_from_values(metric_df, "identity_g_ra")
    gra_abs_max = float(np.nanmax(np.abs(gra_grid))) if np.isfinite(np.nanmax(np.abs(gra_grid))) else 1.0
    if not np.isfinite(gra_abs_max) or gra_abs_max <= 0:
        gra_abs_max = 1.0
    render_grid_map(
        r_vals,
        a_vals,
        gra_grid,
        outpath=outdir / "identity_metric_full_gra_on_grid.png",
        title="Full Identity Metric Mixed Term on Grid",
        colorbar_label="identity g_ra",
        cmap="coolwarm",
        vmin=-gra_abs_max,
        vmax=gra_abs_max,
    )

    summary = pd.DataFrame(
        [
            corr_summary(metric_df, "identity_metric_det", "abs_identity_spin"),
            corr_summary(metric_df, "identity_metric_cond", "abs_identity_spin"),
            corr_summary(metric_df, "identity_g_ra", "abs_identity_spin"),
            corr_summary(metric_df, "identity_g_rr", "abs_identity_spin"),
            corr_summary(metric_df, "identity_g_aa", "abs_identity_spin"),
        ]
    )
    summary.to_csv(outdir / "identity_metric_full_alignment.csv", index=False)

    print(outdir / "identity_metric_full_nodes.csv")
    print(outdir / "identity_metric_full_det_on_grid.png")
    print(outdir / "identity_metric_full_cond_on_grid.png")
    print(outdir / "identity_metric_full_gra_on_grid.png")
    print(outdir / "identity_metric_full_alignment.csv")


if __name__ == "__main__":
    main()
