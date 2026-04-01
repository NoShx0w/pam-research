#!/usr/bin/env python3
from __future__ import annotations

"""
First-pass PAM identity metric experiment.

Estimates a local diagonal metric induced by identity distance using
axis-aligned incident neighbor edges.

Outputs:
- identity_metric_nodes.csv
- identity_metric_det_on_grid.png
- identity_metric_anisotropy_on_grid.png
- identity_metric_vs_spin.png
- identity_metric_singularity_alignment.csv

Run:
    PYTHONPATH=src .venv/bin/python experiments/studies/fim_identity_metric.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.topology.identity_metric import (
    IdentityMetricConfig,
    build_identity_metric_table,
    load_identity_metric_inputs,
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


def render_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str) -> None:
    work = df[[x, y]].dropna().copy()
    plt.figure(figsize=(6.6, 5.0))
    plt.scatter(work[x], work[y], alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
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
    parser = argparse.ArgumentParser(description="Estimate a first-pass local identity metric.")
    parser.add_argument(
        "--identity-nodes-csv",
        default="outputs/fim_identity/identity_field_nodes.csv",
    )
    parser.add_argument(
        "--identity-edges-csv",
        default="outputs/fim_identity/identity_field_edges.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_identity_metric",
    )
    parser.add_argument(
        "--axis-tol",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--min-samples-per-axis",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_identity_metric_inputs(
        identity_nodes_csv=args.identity_nodes_csv,
        identity_edges_csv=args.identity_edges_csv,
    )

    config = IdentityMetricConfig(
        axis_tol=args.axis_tol,
        min_samples_per_axis=args.min_samples_per_axis,
    )

    metric_df = build_identity_metric_table(
        nodes=nodes,
        edges=edges,
        config=config,
    )
    metric_df.to_csv(outdir / "identity_metric_nodes.csv", index=False)

    # determinant map
    r_vals, a_vals, det_grid = grid_from_values(metric_df, "identity_metric_det")
    render_grid_map(
        r_vals,
        a_vals,
        det_grid,
        outpath=outdir / "identity_metric_det_on_grid.png",
        title="Identity Metric Determinant on Grid",
        colorbar_label="identity metric det",
        cmap="viridis",
    )

    # anisotropy map
    _, _, aniso_grid = grid_from_values(metric_df, "identity_metric_anisotropy")
    finite_aniso = pd.to_numeric(metric_df["identity_metric_anisotropy"], errors="coerce")
    finite_aniso = finite_aniso[np.isfinite(finite_aniso)]
    vmax = float(finite_aniso.quantile(0.95)) if len(finite_aniso) else None

    render_grid_map(
        r_vals,
        a_vals,
        aniso_grid,
        outpath=outdir / "identity_metric_anisotropy_on_grid.png",
        title="Identity Metric Anisotropy on Grid",
        colorbar_label="identity metric anisotropy",
        cmap="magma",
        vmax=vmax,
    )

    # compare against spin
    metric_df["abs_identity_spin"] = pd.to_numeric(metric_df["identity_spin"], errors="coerce").abs()

    render_scatter(
        metric_df,
        "identity_metric_det",
        "abs_identity_spin",
        outdir / "identity_metric_det_vs_abs_spin.png",
        "Identity Metric Determinant vs |Identity Spin|",
    )
    render_scatter(
        metric_df,
        "identity_metric_anisotropy",
        "abs_identity_spin",
        outdir / "identity_metric_anisotropy_vs_abs_spin.png",
        "Identity Metric Anisotropy vs |Identity Spin|",
    )

    summary = pd.DataFrame(
        [
            corr_summary(metric_df, "identity_metric_det", "abs_identity_spin"),
            corr_summary(metric_df, "identity_metric_anisotropy", "abs_identity_spin"),
            corr_summary(metric_df, "identity_g_rr", "abs_identity_spin"),
            corr_summary(metric_df, "identity_g_aa", "abs_identity_spin"),
        ]
    )
    summary.to_csv(outdir / "identity_metric_singularity_alignment.csv", index=False)

    print(outdir / "identity_metric_nodes.csv")
    print(outdir / "identity_metric_det_on_grid.png")
    print(outdir / "identity_metric_anisotropy_on_grid.png")
    print(outdir / "identity_metric_det_vs_abs_spin.png")
    print(outdir / "identity_metric_anisotropy_vs_abs_spin.png")
    print(outdir / "identity_metric_singularity_alignment.csv")


if __name__ == "__main__":
    main()
