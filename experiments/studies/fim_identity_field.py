#!/usr/bin/env python3
from __future__ import annotations

"""
Real PAM identity field experiment.

Builds a first-pass local structural identity graph per manifold node from:
- manifold adjacency
- seam distance
- criticality

Then computes:
- identity change magnitude
- identity change field
- identity spin

Outputs:
- identity_node_summary.csv
- identity_field_edges.csv
- identity_field_spin.csv
- identity_magnitude.png
- identity_field_quiver.png
- identity_spin.png

Run:
    PYTHONPATH=src .venv/bin/python experiments/studies/fim_identity_field.py
"""

from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.topology.identity_field import compute_identity_field
from pam.topology.identity_proxy import (
    IdentityProxyConfig,
    build_local_identity_graphs,
    identity_grid_from_node_graphs,
    load_identity_proxy_inputs,
)


def normalize_node_id_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def build_node_summary(
    node_df: pd.DataFrame,
    identity_graphs: dict[str, object],
) -> pd.DataFrame:
    work = node_df.copy()
    work["node_id"] = pd.to_numeric(work["node_id"], errors="coerce").astype("Int64").astype(str)
    rows: list[dict[str, object]] = []
    for _, row in work.iterrows():
        node_id = str(row["node_id"])
        graph = identity_graphs[node_id]
        kinds = [n.kind for n in graph.nodes.values()]
        rows.append(
            {
                "node_id": node_id,
                "i": int(row["i"]),
                "j": int(row["j"]),
                "r": float(row["r"]),
                "alpha": float(row["alpha"]),
                "criticality": pd.to_numeric(row.get("criticality"), errors="coerce"),
                "distance_to_seam": pd.to_numeric(row.get("distance_to_seam"), errors="coerce"),
                "patch_n_nodes": len(graph.nodes),
                "patch_n_edges": len(graph.edges),
                "patch_n_seam": int(sum(1 for k in kinds if k == "seam")),
                "patch_n_critical": int(sum(1 for k in kinds if k == "critical")),
                "patch_n_stable": int(sum(1 for k in kinds if k == "stable")),
            }
        )
    return pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)


def build_edge_field_table(
    node_df: pd.DataFrame,
    vx: np.ndarray,
    vy: np.ndarray,
) -> pd.DataFrame:
    work = node_df.copy().sort_values(["i", "j"]).reset_index(drop=True)
    work["node_id"] = pd.to_numeric(work["node_id"], errors="coerce").astype("Int64").astype(str)
    lookup = {
        (int(row["i"]), int(row["j"])): row
        for _, row in work.iterrows()
    }

    rows: list[dict[str, object]] = []

    n_i, n_j = vx.shape
    for i in range(n_i):
        for j in range(n_j):
            src = lookup[(i, j)]

            if j < n_j - 1:
                dst = lookup[(i, j + 1)]
                rows.append(
                    {
                        "direction": "alpha",
                        "src_node_id": src["node_id"],
                        "dst_node_id": dst["node_id"],
                        "src_i": i,
                        "src_j": j,
                        "dst_i": i,
                        "dst_j": j + 1,
                        "src_r": float(src["r"]),
                        "src_alpha": float(src["alpha"]),
                        "dst_r": float(dst["r"]),
                        "dst_alpha": float(dst["alpha"]),
                        "identity_distance": float(vx[i, j]),
                    }
                )

            if i < n_i - 1:
                dst = lookup[(i + 1, j)]
                rows.append(
                    {
                        "direction": "r",
                        "src_node_id": src["node_id"],
                        "dst_node_id": dst["node_id"],
                        "src_i": i,
                        "src_j": j,
                        "dst_i": i + 1,
                        "dst_j": j,
                        "src_r": float(src["r"]),
                        "src_alpha": float(src["alpha"]),
                        "dst_r": float(dst["r"]),
                        "dst_alpha": float(dst["alpha"]),
                        "identity_distance": float(vy[i, j]),
                    }
                )

    return pd.DataFrame(rows)


def build_spin_table(
    node_df: pd.DataFrame,
    spin: np.ndarray,
) -> pd.DataFrame:
    work = node_df.copy().sort_values(["i", "j"]).reset_index(drop=True)
    work["node_id"] = pd.to_numeric(work["node_id"], errors="coerce").astype("Int64").astype(str)
    lookup = {
        (int(row["i"]), int(row["j"])): row
        for _, row in work.iterrows()
    }

    rows: list[dict[str, object]] = []
    n_i, n_j = spin.shape
    for i in range(n_i):
        for j in range(n_j):
            row = lookup[(i, j)]
            rows.append(
                {
                    "node_id": row["node_id"],
                    "i": i,
                    "j": j,
                    "r": float(row["r"]),
                    "alpha": float(row["alpha"]),
                    "identity_spin": float(spin[i, j]),
                }
            )

    return pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)


def build_field_node_table(
    node_df: pd.DataFrame,
    vx: np.ndarray,
    vy: np.ndarray,
    magnitude: np.ndarray,
    spin: np.ndarray,
) -> pd.DataFrame:
    work = node_df.copy().sort_values(["i", "j"]).reset_index(drop=True)
    work["node_id"] = normalize_node_id_series(work["node_id"])

    lookup = {
        (int(row["i"]), int(row["j"])): row
        for _, row in work.iterrows()
    }

    rows: list[dict[str, object]] = []
    n_i, n_j = magnitude.shape
    for i in range(n_i):
        for j in range(n_j):
            row = lookup[(i, j)]
            rows.append(
                {
                    "node_id": row["node_id"],
                    "i": i,
                    "j": j,
                    "r": float(row["r"]),
                    "alpha": float(row["alpha"]),
                    "identity_vx": float(vx[i, j]),
                    "identity_vy": float(vy[i, j]),
                    "identity_magnitude": float(magnitude[i, j]),
                    "identity_spin": float(spin[i, j]),
                }
            )

    return pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)


def render_magnitude(
    r_vals: np.ndarray,
    a_vals: np.ndarray,
    magnitude: np.ndarray,
    outpath: Path,
) -> None:
    extent = [float(a_vals.min()), float(a_vals.max()), float(r_vals.min()), float(r_vals.max())]

    plt.figure(figsize=(7.2, 5.4))
    plt.imshow(magnitude, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="|identity change|")
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Change Magnitude")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def render_quiver(
    r_vals: np.ndarray,
    a_vals: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    outpath: Path,
) -> None:
    x, y = np.meshgrid(a_vals, r_vals)
    mag = np.sqrt(vx**2 + vy**2)
    mask = mag > 1e-12

    plt.figure(figsize=(7.2, 5.4))
    plt.quiver(
        x[mask],
        y[mask],
        vx[mask],
        vy[mask],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        pivot="mid",
    )
    plt.xlim(float(a_vals.min()), float(a_vals.max()))
    plt.ylim(float(r_vals.min()), float(r_vals.max()))
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Change Field")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def render_spin(
    r_vals: np.ndarray,
    a_vals: np.ndarray,
    spin: np.ndarray,
    outpath: Path,
) -> None:
    extent = [float(a_vals.min()), float(a_vals.max()), float(r_vals.min()), float(r_vals.max())]

    plt.figure(figsize=(7.2, 5.4))
    plt.imshow(spin, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="identity spin")
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Spin Field")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute first-pass real PAM identity field.")
    parser.add_argument("--nodes-csv", default="outputs/fim_distance/fisher_nodes.csv")
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--criticality-csv", default="outputs/fim_critical/criticality_surface.csv")
    parser.add_argument("--phase-distance-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/fim_identity")
    parser.add_argument("--seam-eps", type=float, default=0.15)
    parser.add_argument("--criticality-quantile", type=float, default=0.90)
    parser.add_argument("--normalized", action="store_true", default=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    node_df, edge_df = load_identity_proxy_inputs(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        criticality_csv=args.criticality_csv,
        phase_distance_csv=args.phase_distance_csv,
    )

    config = IdentityProxyConfig(
        seam_eps=args.seam_eps,
        criticality_quantile=args.criticality_quantile,
    )

    identity_graphs = build_local_identity_graphs(
        node_df=node_df,
        edge_df=edge_df,
        config=config,
    )

    identity_grid = identity_grid_from_node_graphs(
        node_df=node_df,
        identity_graphs=identity_graphs,
    )

    field = compute_identity_field(identity_grid, normalized=args.normalized)

    node_summary = build_node_summary(node_df, identity_graphs)
    edge_table = build_edge_field_table(node_df, field.vx, field.vy)
    spin_table = build_spin_table(node_df, field.spin)

    field_node_table = build_field_node_table(
        node_df=node_df,
        vx=field.vx,
        vy=field.vy,
        magnitude=field.magnitude,
        spin=field.spin,
    )

    node_summary.to_csv(outdir / "identity_node_summary.csv", index=False)
    edge_table.to_csv(outdir / "identity_field_edges.csv", index=False)
    spin_table.to_csv(outdir / "identity_spin.csv", index=False)

    r_vals = np.sort(pd.to_numeric(node_df["r"], errors="coerce").unique())
    a_vals = np.sort(pd.to_numeric(node_df["alpha"], errors="coerce").unique())

    field_node_table.to_csv(outdir / "identity_field_nodes.csv", index=False)

    render_magnitude(r_vals, a_vals, field.magnitude, outdir / "identity_magnitude.png")
    render_quiver(r_vals, a_vals, field.vx, field.vy, outdir / "identity_field_quiver.png")
    render_spin(r_vals, a_vals, field.spin, outdir / "identity_spin.png")

    print(outdir / "identity_node_summary.csv")
    print(outdir / "identity_field_nodes.csv")
    print(outdir / "identity_field_edges.csv")
    print(outdir / "identity_spin.csv")
    print(outdir / "identity_magnitude.png")
    print(outdir / "identity_field_quiver.png")
    print(outdir / "identity_spin.png")


if __name__ == "__main__":
    main()
