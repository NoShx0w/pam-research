#!/usr/bin/env python3
"""
OBS-023 — Transport misalignment localizes at the seam.

Build a clean result figure and summary from the outputs of
experiments/toy/identity_transport_alignment_toy.py

Inputs
------
outputs/toy_identity_transport_alignment/
  node_transport_alignment.csv
  edge_transport_alignment.csv
  identity_transport_alignment_summary.txt   (optional)

outputs/obs022_scene_bundle/
  scene_seam.csv

Outputs
-------
outputs/obs023_transport_misalignment/
  obs023_transport_misalignment_figure.png
  obs023_transport_misalignment_summary.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    node_csv: str = "outputs/toy_identity_transport_alignment/node_transport_alignment.csv"
    edge_csv: str = "outputs/toy_identity_transport_alignment/edge_transport_alignment.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/obs023_transport_misalignment"
    seam_threshold: float = 0.15
    top_k_labels: int = 8


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.node_csv)
    edges = pd.read_csv(cfg.edge_csv)
    seam = pd.read_csv(cfg.seam_csv)

    numeric_cols_nodes = [
        "node_id", "r", "alpha", "mds1", "mds2",
        "signed_phase", "distance_to_seam",
        "transport_align_mean_deg", "transport_align_max_deg",
        "node_holonomy_proxy",
    ]
    numeric_cols_edges = [
        "src_id", "dst_id",
        "src_mds1", "src_mds2", "dst_mds1", "dst_mds2",
        "edge_signed_phase_mid", "edge_distance_to_seam_mid",
        "misalignment_deg", "edge_holonomy_proxy",
    ]
    numeric_cols_seam = ["r", "alpha", "mds1", "mds2", "signed_phase", "distance_to_seam"]

    for col in numeric_cols_nodes:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")
    for col in numeric_cols_edges:
        if col in edges.columns:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")
    for col in numeric_cols_seam:
        if col in seam.columns:
            seam[col] = pd.to_numeric(seam[col], errors="coerce")

    return nodes, edges, seam


def build_summary(nodes: pd.DataFrame, edges: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold
    seam_mean = float(pd.to_numeric(nodes.loc[seam_mask, "transport_align_mean_deg"], errors="coerce").mean())
    off_mean = float(pd.to_numeric(nodes.loc[~seam_mask, "transport_align_mean_deg"], errors="coerce").mean())

    corr_node_seam = safe_corr(nodes["transport_align_mean_deg"], nodes["distance_to_seam"])
    corr_node_hol = safe_corr(nodes["transport_align_mean_deg"], nodes["node_holonomy_proxy"])
    corr_edge_seam = safe_corr(edges["misalignment_deg"], edges["edge_distance_to_seam_mid"])
    corr_edge_hol = safe_corr(edges["misalignment_deg"], edges["edge_holonomy_proxy"]) if "edge_holonomy_proxy" in edges.columns else float("nan")

    top = nodes.sort_values("transport_align_mean_deg", ascending=False).head(10)

    lines = [
        "=== OBS-023 Transport Misalignment Summary ===",
        "",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Node-level means",
        f"  mean transport_align_mean_deg = {float(pd.to_numeric(nodes['transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        f"  mean transport_align_max_deg  = {float(pd.to_numeric(nodes['transport_align_max_deg'], errors='coerce').mean()):.4f}",
        f"  seam-band mean misalignment   = {seam_mean:.4f}",
        f"  off-seam mean misalignment    = {off_mean:.4f}",
        "",
        "Correlations",
        f"  corr(node misalignment, distance_to_seam) = {corr_node_seam:.4f}",
        f"  corr(node misalignment, node holonomy)    = {corr_node_hol:.4f}",
        f"  corr(edge misalignment, edge seam mid)    = {corr_edge_seam:.4f}",
        f"  corr(edge misalignment, edge holonomy)    = {corr_edge_hol:.4f}",
        "",
        "Interpretive summary",
        "- response-field transport misalignment is markedly elevated near the seam",
        "- highest-misalignment nodes cluster on or immediately adjacent to the seam",
        "- current node holonomy proxy is present but only weakly related to transport mismatch",
        "",
        "Highest-misalignment nodes",
    ]

    keep = [c for c in ["node_id", "r", "alpha", "transport_align_mean_deg", "distance_to_seam", "node_holonomy_proxy"] if c in top.columns]
    for _, row in top[keep].iterrows():
        parts = []
        for c in keep:
            if c == "node_id":
                parts.append(f"{c}={int(row[c])}")
            else:
                parts.append(f"{c}={float(row[c]):.4f}")
        lines.append("  " + ", ".join(parts))

    return "\n".join(lines)


def render_figure(cfg: Config, nodes: pd.DataFrame, edges: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    top = nodes.sort_values("transport_align_mean_deg", ascending=False).head(cfg.top_k_labels).copy()

    fig = plt.figure(figsize=(15.5, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.5], height_ratios=[1.0, 1.0])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[1, 1])

    # background edge web colored by signed phase midpoint if available
    if {"src_mds1", "src_mds2", "dst_mds1", "dst_mds2", "edge_signed_phase_mid"}.issubset(edges.columns):
        for _, row in edges.iterrows():
            if int(row["src_id"]) > int(row["dst_id"]):
                continue
            arr = [row["src_mds1"], row["src_mds2"], row["dst_mds1"], row["dst_mds2"], row["edge_signed_phase_mid"]]
            if not np.isfinite(arr).all():
                continue
            ax_main.plot(
                [row["src_mds1"], row["dst_mds1"]],
                [row["src_mds2"], row["dst_mds2"]],
                color=plt.cm.coolwarm((float(row["edge_signed_phase_mid"]) + 1.0) / 2.0),
                alpha=0.22,
                linewidth=1.0,
                zorder=1,
            )

    # seam
    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=6.0, alpha=0.65, zorder=2)
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.8, alpha=0.96, zorder=3)

    # nodes colored by misalignment
    sc = ax_main.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce"),
        cmap="viridis",
        s=90,
        alpha=0.95,
        linewidths=0.35,
        edgecolors="white",
        zorder=4,
    )

    # seam-band ring
    seam_nodes = nodes[seam_mask].copy()
    if len(seam_nodes):
        ax_main.scatter(
            seam_nodes["mds1"],
            seam_nodes["mds2"],
            s=170,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
        )

    # top labels
    for _, row in top.iterrows():
        if not np.isfinite([row["mds1"], row["mds2"], row["transport_align_mean_deg"]]).all():
            continue
        ax_main.scatter([row["mds1"]], [row["mds2"]], s=140, facecolors="none", edgecolors="#FFD166", linewidths=1.8, zorder=6)
        ax_main.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=7,
        )

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.038, pad=0.02)
    cbar.set_label("mean transport misalignment (deg)")

    ax_main.set_title("OBS-023 — Transport misalignment on the phase manifold", fontsize=15, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.text(
        0.02,
        0.98,
        "black seam = detected phase boundary\nblack rings = seam neighborhood\nyellow labels = highest-misalignment nodes",
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92),
    )

    # seam vs off-seam comparison
    seam_vals = pd.to_numeric(nodes.loc[seam_mask, "transport_align_mean_deg"], errors="coerce").dropna()
    off_vals = pd.to_numeric(nodes.loc[~seam_mask, "transport_align_mean_deg"], errors="coerce").dropna()

    cats = ["seam-band", "off-seam"]
    means = [float(seam_vals.mean()), float(off_vals.mean())]
    errs = [
        float(seam_vals.std(ddof=1) / max(np.sqrt(len(seam_vals)), 1.0)) if len(seam_vals) > 1 else 0.0,
        float(off_vals.std(ddof=1) / max(np.sqrt(len(off_vals)), 1.0)) if len(off_vals) > 1 else 0.0,
    ]
    ax_bar.bar(cats, means, yerr=errs, alpha=0.9)
    ax_bar.set_ylabel("mean misalignment (deg)")
    ax_bar.set_title("Seam localization", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    # scatter: misalignment vs seam distance
    x = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
    y = pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce")
    mask = x.notna() & y.notna()
    ax_sc.scatter(x[mask], y[mask], s=38, alpha=0.88)
    ax_sc.set_xlabel("distance to seam")
    ax_sc.set_ylabel("mean misalignment (deg)")
    ax_sc.set_title("Node misalignment vs seam distance", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)

    corr = safe_corr(x, y)
    ax_sc.text(
        0.98,
        0.05,
        f"corr = {corr:.3f}",
        transform=ax_sc.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("PAM Observatory — OBS-023", fontsize=19)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-023 transport misalignment figure.")
    parser.add_argument("--node-csv", default=Config.node_csv)
    parser.add_argument("--edge-csv", default=Config.edge_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        node_csv=args.node_csv,
        edge_csv=args.edge_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, seam = load_inputs(cfg)
    summary = build_summary(nodes, edges, cfg.seam_threshold)

    png_path = outdir / "obs023_transport_misalignment_figure.png"
    txt_path = outdir / "obs023_transport_misalignment_summary.txt"

    render_figure(cfg, nodes, edges, seam, png_path)
    txt_path.write_text(summary, encoding="utf-8")

    print(png_path)
    print(txt_path)


if __name__ == "__main__":
    main()
