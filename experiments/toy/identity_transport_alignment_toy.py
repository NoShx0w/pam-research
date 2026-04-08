#!/usr/bin/env python3
"""
identity_transport_alignment_toy.py

Canonical toy experiment for transport-aware directional mismatch on the PAM manifold.

Uses:
- pam.geometry.directional_field.DirectionalField
- pam.geometry.transport.edge_transport_table
- pam.geometry.transport.node_transport_summary

Outputs
-------
outputs/toy_identity_transport_alignment/
  node_transport_alignment.csv
  edge_transport_alignment.csv
  identity_transport_alignment_summary.txt
  identity_transport_alignment_panel.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.geometry.directional_field import DirectionalField
from pam.geometry.parallel_transport import (
    edge_parallel_transport_table,
    node_parallel_transport_summary,
)


@dataclass(frozen=True)
class Config:
    nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    edges_csv: str = "outputs/obs022_scene_bundle/scene_edges.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/toy_identity_transport_alignment"
    seam_threshold: float = 0.15
    top_k_labels: int = 10


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def load_field(cfg: Config) -> DirectionalField:
    return DirectionalField.from_csv(
        cfg.nodes_csv,
        cfg.edges_csv,
        connection_theta_col="fim_theta",
        response_theta_col="rsp_theta",
    )


def build_edge_table(field: DirectionalField) -> pd.DataFrame:
    edge_base = edge_transport_table(field)
    edges = field.edges.copy()
    nodes = field.nodes.copy()
    lookup = nodes.set_index(field.node_id_col, drop=False)

    rows = []
    for _, row in edge_base.iterrows():
        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        src = lookup.loc[src_id]
        dst = lookup.loc[dst_id]

        out = dict(row)
        out["src_mds1"] = float(src["mds1"])
        out["src_mds2"] = float(src["mds2"])
        out["dst_mds1"] = float(dst["mds1"])
        out["dst_mds2"] = float(dst["mds2"])

        out["src_signed_phase"] = float(src["signed_phase"]) if pd.notna(src["signed_phase"]) else np.nan
        out["dst_signed_phase"] = float(dst["signed_phase"]) if pd.notna(dst["signed_phase"]) else np.nan
        out["src_distance_to_seam"] = float(src["distance_to_seam"]) if pd.notna(src["distance_to_seam"]) else np.nan
        out["dst_distance_to_seam"] = float(dst["distance_to_seam"]) if pd.notna(dst["distance_to_seam"]) else np.nan

        out["src_theta_transport"] = float(src["fim_theta"])
        out["dst_theta_transport"] = float(dst["fim_theta"])
        out["src_rsp_theta"] = float(src["rsp_theta"])
        out["dst_rsp_theta"] = float(dst["rsp_theta"])

        out["edge_signed_phase_mid"] = 0.5 * (out["src_signed_phase"] + out["dst_signed_phase"])
        out["edge_distance_to_seam_mid"] = 0.5 * (out["src_distance_to_seam"] + out["dst_distance_to_seam"])

        rows.append(out)

    edge_df = pd.DataFrame(rows)

    if "response_misalignment_deg" in edge_df.columns and "misalignment_deg" not in edge_df.columns:
        edge_df["misalignment_deg"] = edge_df["response_misalignment_deg"]

    # merge any existing edge proxy from bundle edges if present
    if {"src_id", "dst_id"}.issubset(edges.columns):
        keep = [c for c in ["src_id", "dst_id", "edge_holonomy_proxy"] if c in edges.columns]
        if len(keep) >= 2:
            edge_df = edge_df.merge(edges[keep], on=["src_id", "dst_id"], how="left")

    return edge_df


def build_node_table(field: DirectionalField, edge_df: pd.DataFrame) -> pd.DataFrame:
    local_df = field.local_direction_mismatch(degrees=True).rename(
        columns={"local_direction_mismatch": "local_direction_mismatch_deg"}
    )
    neighbor_df = field.node_neighbor_mismatch(degrees=True)
    transport_df = node_parallel_transport_summary(field)

    nodes = field.attach_node_metrics(local_df, neighbor_df, transport_df)

    agg_spec = {}
    if "edge_holonomy_proxy" in edge_df.columns:
        agg_spec["transport_holonomy_edge_mean"] = ("edge_holonomy_proxy", "mean")
    if "edge_distance_to_seam_mid" in edge_df.columns:
        agg_spec["transport_edge_seam_mid_mean"] = ("edge_distance_to_seam_mid", "mean")
    if "response_misalignment_deg" in edge_df.columns:
        agg_spec["transport_edge_misalignment_mean"] = ("response_misalignment_deg", "mean")

    if agg_spec:
        extra = (
            edge_df.groupby("src_id", as_index=False)
            .agg(**agg_spec)
            .rename(columns={"src_id": "node_id"})
        )
        nodes = nodes.merge(extra, on="node_id", how="left")

    if "transport_holonomy_edge_mean" not in nodes.columns:
        nodes["transport_holonomy_edge_mean"] = np.nan
    if "transport_edge_seam_mid_mean" not in nodes.columns:
        nodes["transport_edge_seam_mid_mean"] = np.nan
    if "transport_edge_misalignment_mean" not in nodes.columns:
        nodes["transport_edge_misalignment_mean"] = np.nan

    return nodes


def build_summary(nodes: pd.DataFrame, edge_df: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    edge_mis_col = (
        "misalignment_deg"
        if "misalignment_deg" in edge_df.columns
        else "response_misalignment_deg"
        if "response_misalignment_deg" in edge_df.columns
        else None
    )

    edge_seam_col = (
        "edge_distance_to_seam_mid"
        if "edge_distance_to_seam_mid" in edge_df.columns
        else None
    )

    edge_hol_col = "edge_holonomy_proxy" if "edge_holonomy_proxy" in edge_df.columns else None

    edge_seam_corr = (
        safe_corr(edge_df[edge_mis_col], edge_df[edge_seam_col])
        if edge_mis_col is not None and edge_seam_col is not None
        else float("nan")
    )
    edge_hol_corr = (
        safe_corr(edge_df[edge_mis_col], edge_df[edge_hol_col])
        if edge_mis_col is not None and edge_hol_col is not None
        else float("nan")
    )

    lines = [
        "=== Identity Transport Alignment Toy Summary ===",
        "",
        "transport_theta_column = fim_theta",
        f"n_nodes = {len(nodes)}",
        f"n_edges_undirected = {len(edge_df)}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Node-level means",
        f"  mean transport_align_mean_deg = {float(pd.to_numeric(nodes['transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        f"  mean transport_align_max_deg  = {float(pd.to_numeric(nodes['transport_align_max_deg'], errors='coerce').mean()):.4f}",
        f"  seam-band mean misalignment   = {float(pd.to_numeric(nodes.loc[seam_mask, 'transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        f"  off-seam mean misalignment    = {float(pd.to_numeric(nodes.loc[~seam_mask, 'transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        "",
        "Correlations",
        f"  corr(node misalignment, distance_to_seam) = {safe_corr(nodes['transport_align_mean_deg'], nodes['distance_to_seam']):.4f}",
        f"  corr(node misalignment, node holonomy)    = {safe_corr(nodes['transport_align_mean_deg'], nodes['node_holonomy_proxy']):.4f}",
        f"  corr(edge misalignment, edge seam mid)    = {edge_seam_corr:.4f}",
        f"  corr(edge misalignment, edge holonomy)    = {edge_hol_corr:.4f}",
        "",
        "Highest-misalignment nodes",
    ]

    top = nodes.sort_values("transport_align_mean_deg", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, "
            f"r={float(row['r']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"transport_align_mean_deg={float(row['transport_align_mean_deg']):.4f}, "
            f"distance_to_seam={float(row['distance_to_seam']):.4f}, "
            f"node_holonomy_proxy={float(row['node_holonomy_proxy']) if pd.notna(row['node_holonomy_proxy']) else float('nan'):.4f}"
        )

    return "\n".join(lines)


def render_panel(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    top = nodes.sort_values("transport_align_mean_deg", ascending=False).head(cfg.top_k_labels)

    fig = plt.figure(figsize=(15.5, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.5], height_ratios=[1.0, 1.0])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[1, 1])

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=6.0, alpha=0.65, zorder=1)
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.8, alpha=0.96, zorder=2)

    sc = ax_main.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce"),
        cmap="viridis",
        s=95,
        alpha=0.95,
        linewidths=0.35,
        edgecolors="white",
        zorder=3,
    )

    seam_nodes = nodes[seam_mask]
    if len(seam_nodes):
        ax_main.scatter(
            seam_nodes["mds1"], seam_nodes["mds2"],
            s=175, facecolors="none", edgecolors="black", linewidths=1.4, zorder=4
        )

    for _, row in top.iterrows():
        ax_main.scatter([row["mds1"]], [row["mds2"]], s=145, facecolors="none", edgecolors="#FFD166", linewidths=1.8, zorder=5)
        ax_main.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=6,
        )

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.038, pad=0.02)
    cbar.set_label("mean transport misalignment (deg)")

    ax_main.set_title("Transport-aware response-field misalignment", fontsize=15, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")

    vals = [
        float(pd.to_numeric(nodes.loc[seam_mask, "transport_align_mean_deg"], errors="coerce").mean()),
        float(pd.to_numeric(nodes.loc[~seam_mask, "transport_align_mean_deg"], errors="coerce").mean()),
    ]
    ax_bar.bar(["seam-band", "off-seam"], vals, alpha=0.9)
    ax_bar.set_ylabel("mean misalignment (deg)")
    ax_bar.set_title("Misalignment vs seam band", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    x = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
    y = pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce")
    mask = x.notna() & y.notna()
    ax_sc.scatter(x[mask], y[mask], s=38, alpha=0.88)
    ax_sc.set_xlabel("distance to seam")
    ax_sc.set_ylabel("mean misalignment (deg)")
    ax_sc.set_title("Misalignment vs seam distance", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)
    ax_sc.text(
        0.98, 0.05,
        f"corr = {safe_corr(y, x):.3f}",
        transform=ax_sc.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("Identity Transport Alignment Toy", fontsize=19)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transport-aware alignment toy built on canonical geometry modules.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--edges-csv", default=Config.edges_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seam = pd.read_csv(cfg.seam_csv)
    for col in ["mds1", "mds2", "signed_phase", "distance_to_seam"]:
        if col in seam.columns:
            seam[col] = pd.to_numeric(seam[col], errors="coerce")

    field = load_field(cfg)
    edge_df = edge_parallel_transport_table(field)
    node_df = build_node_table(field, edge_df)

    node_csv = outdir / "node_transport_alignment.csv"
    edge_csv = outdir / "edge_transport_alignment.csv"
    txt_path = outdir / "identity_transport_alignment_summary.txt"
    png_path = outdir / "identity_transport_alignment_panel.png"

    node_df.to_csv(node_csv, index=False)
    edge_df.to_csv(edge_csv, index=False)
    txt_path.write_text(build_summary(node_df, edge_df, cfg.seam_threshold), encoding="utf-8")
    render_panel(cfg, node_df, seam, png_path)

    print(node_csv)
    print(edge_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()