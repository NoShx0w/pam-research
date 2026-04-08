#!/usr/bin/env python3
"""
OBS-023 — transport misalignment on the phase manifold.

Canonical figure built on pam.geometry.directional_field + pam.geometry.transport.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.geometry.directional_field import DirectionalField
from pam.geometry.transport import edge_transport_table, node_transport_summary


@dataclass(frozen=True)
class Config:
    nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    edges_csv: str = "outputs/obs022_scene_bundle/scene_edges.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/obs023_transport_misalignment"
    seam_threshold: float = 0.15
    top_k_labels: int = 10


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def load_inputs(cfg: Config) -> tuple[DirectionalField, pd.DataFrame]:
    field = DirectionalField.from_csv(
        cfg.nodes_csv,
        cfg.edges_csv,
        connection_theta_col="fim_theta",
        response_theta_col="rsp_theta",
    )
    seam = pd.read_csv(cfg.seam_csv)
    for col in ["mds1", "mds2", "signed_phase", "distance_to_seam"]:
        if col in seam.columns:
            seam[col] = pd.to_numeric(seam[col], errors="coerce")
    return field, seam


def build_node_table(field: DirectionalField) -> pd.DataFrame:
    nodes = field.nodes.copy()
    transport_nodes = node_transport_summary(field)
    nodes = nodes.merge(transport_nodes, on="node_id", how="left")
    return nodes


def write_summary(path: Path, nodes: pd.DataFrame, seam_threshold: float) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold
    lines = [
        "=== OBS-023 Transport Misalignment Summary ===",
        "",
        f"n_nodes = {len(nodes)}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Node-level means",
        f"  mean transport_align_mean_deg = {float(pd.to_numeric(nodes['transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        f"  seam-band mean                = {float(pd.to_numeric(nodes.loc[seam_mask, 'transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        f"  off-seam mean                 = {float(pd.to_numeric(nodes.loc[~seam_mask, 'transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        "",
        "Correlations",
        f"  corr(node misalignment, distance_to_seam) = {safe_corr(nodes['transport_align_mean_deg'], nodes['distance_to_seam']):.4f}",
        f"  corr(node misalignment, node holonomy)    = {safe_corr(nodes['transport_align_mean_deg'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        "",
        "Top nodes",
    ]
    top = nodes.sort_values("transport_align_mean_deg", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, r={float(row['r']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"transport_align_mean_deg={float(row['transport_align_mean_deg']):.4f}, "
            f"distance_to_seam={float(row['distance_to_seam']):.4f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_figure(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    top = nodes.sort_values("transport_align_mean_deg", ascending=False).head(cfg.top_k_labels)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 1.25], height_ratios=[1.0, 1.0])

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
        s=92,
        alpha=0.96,
        linewidths=0.35,
        edgecolors="white",
        zorder=3,
    )

    seam_nodes = nodes[seam_mask]
    if len(seam_nodes):
        ax_main.scatter(
            seam_nodes["mds1"], seam_nodes["mds2"],
            s=170, facecolors="none", edgecolors="black", linewidths=1.3, zorder=4
        )

    for _, row in top.iterrows():
        ax_main.scatter([row["mds1"]], [row["mds2"]], s=140, facecolors="none", edgecolors="#FFD166", linewidths=1.8, zorder=5)
        ax_main.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=6,
        )

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.038, pad=0.02)
    cbar.set_label("mean transport misalignment (deg)")

    ax_main.set_title("OBS-023 — Transport misalignment on the phase manifold", fontsize=18, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.text(
        0.02, 0.97,
        "black seam = detected phase boundary\nblack rings = seam neighborhood\nyellow labels = highest-misalignment nodes",
        transform=ax_main.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )

    vals = [
        float(pd.to_numeric(nodes.loc[seam_mask, "transport_align_mean_deg"], errors="coerce").mean()),
        float(pd.to_numeric(nodes.loc[~seam_mask, "transport_align_mean_deg"], errors="coerce").mean()),
    ]
    ax_bar.bar(["seam-band", "off-seam"], vals, alpha=0.9)
    ax_bar.set_ylabel("mean misalignment (deg)")
    ax_bar.set_title("Seam localization", fontsize=14, pad=8)
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
        0.98, 0.05, f"corr = {safe_corr(y, x):.3f}",
        transform=ax_sc.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("PAM Observatory — OBS-023", fontsize=22)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="OBS-023 transport misalignment figure.")
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

    field, seam = load_inputs(cfg)
    nodes = build_node_table(field)

    png_path = outdir / "obs023_transport_misalignment_figure.png"
    txt_path = outdir / "obs023_transport_misalignment_summary.txt"
    csv_path = outdir / "obs023_transport_misalignment_nodes.csv"

    nodes.to_csv(csv_path, index=False)
    write_summary(txt_path, nodes, cfg.seam_threshold)
    render_figure(cfg, nodes, seam, png_path)

    print(csv_path)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()