#!/usr/bin/env python3
"""
OBS-024 — Relational vs pointwise directional mismatch.

Build a dedicated figure comparing:
- local_direction_mismatch_deg
- neighbor_direction_mismatch_deg

Inputs
------
outputs/obs023_local_direction_mismatch/
  local_direction_mismatch_nodes.csv

outputs/obs022_scene_bundle/
  scene_seam.csv

Outputs
-------
outputs/obs024_relational_vs_pointwise/
  obs024_relational_vs_pointwise_figure.png
  obs024_relational_vs_pointwise_summary.txt
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
    nodes_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/obs024_relational_vs_pointwise"
    seam_threshold: float = 0.15
    top_k_labels: int = 6


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    seam = pd.read_csv(cfg.seam_csv)

    node_cols = [
        "node_id", "r", "alpha", "mds1", "mds2", "distance_to_seam",
        "local_direction_mismatch_deg",
        "neighbor_direction_mismatch_deg",
        "transport_align_mean_deg",
        "node_holonomy_proxy",
    ]
    seam_cols = ["mds1", "mds2", "distance_to_seam"]

    for c in node_cols:
        if c in nodes.columns:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
    for c in seam_cols:
        if c in seam.columns:
            seam[c] = pd.to_numeric(seam[c], errors="coerce")

    return nodes, seam


def build_summary(nodes: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    loc = pd.to_numeric(nodes["local_direction_mismatch_deg"], errors="coerce")
    nei = pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce")
    d2s = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
    hol = pd.to_numeric(nodes["node_holonomy_proxy"], errors="coerce")

    lines = [
        "=== OBS-024 Relational vs Pointwise Summary ===",
        "",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Pointwise field",
        f"  mean local_direction_mismatch_deg = {float(loc.mean()):.4f}",
        f"  seam-band mean                    = {float(loc[seam_mask].mean()):.4f}",
        f"  off-seam mean                     = {float(loc[~seam_mask].mean()):.4f}",
        "",
        "Relational field",
        f"  mean neighbor_direction_mismatch_deg = {float(nei.mean()):.4f}",
        f"  seam-band mean                       = {float(nei[seam_mask].mean()):.4f}",
        f"  off-seam mean                        = {float(nei[~seam_mask].mean()):.4f}",
        "",
        "Correlations",
        f"  corr(local mismatch, distance_to_seam)    = {safe_corr(loc, d2s):.4f}",
        f"  corr(neighbor mismatch, distance_to_seam) = {safe_corr(nei, d2s):.4f}",
        f"  corr(local mismatch, node holonomy)       = {safe_corr(loc, hol):.4f}",
        f"  corr(neighbor mismatch, node holonomy)    = {safe_corr(nei, hol):.4f}",
        "",
        "Interpretation",
        "- pointwise mismatch is present but only mildly seam-localized",
        "- relational mismatch is strongly seam-localized",
        "- seam obstruction is therefore primarily relational rather than pointwise",
    ]
    return "\n".join(lines)


def draw_field_panel(ax, nodes: pd.DataFrame, seam: pd.DataFrame, value_col: str, title: str, cmap: str, seam_threshold: float, top_ids: set[int]) -> None:
    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.8, alpha=0.65, zorder=1)
        ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.7, alpha=0.95, zorder=2)

    vals = pd.to_numeric(nodes[value_col], errors="coerce")
    sc = ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=vals,
        cmap=cmap,
        s=88,
        alpha=0.95,
        linewidths=0.35,
        edgecolors="white",
        zorder=3,
    )

    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold
    seam_nodes = nodes[seam_mask]
    if len(seam_nodes):
        ax.scatter(
            seam_nodes["mds1"],
            seam_nodes["mds2"],
            s=165,
            facecolors="none",
            edgecolors="black",
            linewidths=1.25,
            zorder=4,
        )

    label_rows = nodes[nodes["node_id"].isin(top_ids)].copy()
    for _, row in label_rows.iterrows():
        ax.scatter([row["mds1"]], [row["mds2"]], s=140, facecolors="none", edgecolors="#FFD166", linewidths=1.7, zorder=5)
        ax.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=6,
        )

    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.08)
    ax.set_aspect("equal", adjustable="box")
    return sc


def render_figure(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    top_local = set(nodes.sort_values("local_direction_mismatch_deg", ascending=False).head(cfg.top_k_labels)["node_id"].tolist())
    top_neighbor = set(nodes.sort_values("neighbor_direction_mismatch_deg", ascending=False).head(cfg.top_k_labels)["node_id"].tolist())

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.6, 1.1], height_ratios=[1.0, 1.0])

    ax_loc = fig.add_subplot(gs[:, 0])
    ax_nei = fig.add_subplot(gs[:, 1])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_sc = fig.add_subplot(gs[1, 2])

    sc1 = draw_field_panel(
        ax_loc, nodes, seam,
        "local_direction_mismatch_deg",
        "Pointwise mismatch",
        "magma",
        cfg.seam_threshold,
        top_local,
    )
    sc2 = draw_field_panel(
        ax_nei, nodes, seam,
        "neighbor_direction_mismatch_deg",
        "Relational / neighbor mismatch",
        "viridis",
        cfg.seam_threshold,
        top_neighbor,
    )

    cbar1 = fig.colorbar(sc1, ax=ax_loc, fraction=0.040, pad=0.02)
    cbar1.set_label("local mismatch (deg)")
    cbar2 = fig.colorbar(sc2, ax=ax_nei, fraction=0.040, pad=0.02)
    cbar2.set_label("neighbor mismatch (deg)")

    loc = pd.to_numeric(nodes["local_direction_mismatch_deg"], errors="coerce")
    nei = pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce")

    cats = ["pointwise\nseam", "pointwise\noff", "relational\nseam", "relational\noff"]
    vals = [
        float(loc[seam_mask].mean()),
        float(loc[~seam_mask].mean()),
        float(nei[seam_mask].mean()),
        float(nei[~seam_mask].mean()),
    ]
    ax_bar.bar(cats, vals, alpha=0.9)
    ax_bar.set_ylabel("mean mismatch (deg)")
    ax_bar.set_title("Seam localization comparison", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    mask = loc.notna() & nei.notna()
    ax_sc.scatter(loc[mask], nei[mask], s=36, alpha=0.88)
    ax_sc.set_xlabel("pointwise mismatch (deg)")
    ax_sc.set_ylabel("relational mismatch (deg)")
    ax_sc.set_title("Field comparison", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)
    ax_sc.text(
        0.98, 0.05,
        f"corr = {safe_corr(loc, nei):.3f}",
        transform=ax_sc.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("PAM Observatory — OBS-024 relational vs pointwise obstruction", fontsize=19)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-024 relational-vs-pointwise comparison figure.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, seam = load_inputs(cfg)

    png_path = outdir / "obs024_relational_vs_pointwise_figure.png"
    txt_path = outdir / "obs024_relational_vs_pointwise_summary.txt"

    render_figure(cfg, nodes, seam, png_path)
    txt_path.write_text(build_summary(nodes, cfg.seam_threshold), encoding="utf-8")

    print(png_path)
    print(txt_path)


if __name__ == "__main__":
    main()
