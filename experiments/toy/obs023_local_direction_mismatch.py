#!/usr/bin/env python3
"""
OBS-023 local directional mismatch proxy.

Compare local Fisher-direction and response-direction structure directly.

Outputs
-------
outputs/obs023_local_direction_mismatch/
  local_direction_mismatch_nodes.csv
  obs023_local_direction_mismatch_summary.txt
  obs023_local_direction_mismatch_figure.png
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
    outdir: str = "outputs/obs023_local_direction_mismatch"
    seam_threshold: float = 0.15
    top_k_labels: int = 8


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def axial_angle_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    d = np.abs(wrap_angle(aa - bb))
    d = np.minimum(d, np.pi - d)
    return np.degrees(d)


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

    node_cols = [
        "node_id", "r", "alpha", "mds1", "mds2",
        "signed_phase", "distance_to_seam",
        "fim_theta", "rsp_theta",
        "transport_align_mean_deg", "node_holonomy_proxy",
    ]
    edge_cols = [
        "src_id", "dst_id",
        "src_mds1", "src_mds2", "dst_mds1", "dst_mds2",
        "src_theta_transport", "dst_theta_transport",
        "src_rsp_theta", "dst_rsp_theta",
        "misalignment_deg", "edge_distance_to_seam_mid",
    ]
    seam_cols = ["mds1", "mds2", "signed_phase", "distance_to_seam"]

    for c in node_cols:
        if c in nodes.columns:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
    for c in edge_cols:
        if c in edges.columns:
            edges[c] = pd.to_numeric(edges[c], errors="coerce")
    for c in seam_cols:
        if c in seam.columns:
            seam[c] = pd.to_numeric(seam[c], errors="coerce")

    return nodes, edges, seam


def compute_node_proxy(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()

    if "fim_theta" not in out.columns or "rsp_theta" not in out.columns:
        raise ValueError("node csv must contain fim_theta and rsp_theta")

    out["local_direction_mismatch_deg"] = axial_angle_diff_deg(out["fim_theta"], out["rsp_theta"])

    if len(edges):
        # Compare local directional-change mismatch along edges:
        # |Δrsp_theta - Δfim_theta| in axial form, aggregated back to source node
        d_fim = wrap_angle(
            pd.to_numeric(edges["dst_theta_transport"], errors="coerce")
            - pd.to_numeric(edges["src_theta_transport"], errors="coerce")
        )
        d_rsp = wrap_angle(
            pd.to_numeric(edges["dst_rsp_theta"], errors="coerce")
            - pd.to_numeric(edges["src_rsp_theta"], errors="coerce")
        )
        d = np.abs(wrap_angle(d_rsp - d_fim))
        d = np.minimum(d, np.pi - d)
        edges = edges.copy()
        edges["neighbor_direction_mismatch_deg"] = np.degrees(d)

        agg = (
            edges.groupby("src_id", as_index=False)
            .agg(
                neighbor_direction_mismatch_deg=("neighbor_direction_mismatch_deg", "mean"),
            )
            .rename(columns={"src_id": "node_id"})
        )
        out = out.merge(agg, on="node_id", how="left")
    else:
        out["neighbor_direction_mismatch_deg"] = np.nan

    return out


def build_summary(nodes: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    loc = pd.to_numeric(nodes["local_direction_mismatch_deg"], errors="coerce")
    nei = pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce")
    trn = pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce")
    d2s = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
    hol = pd.to_numeric(nodes["node_holonomy_proxy"], errors="coerce")

    lines = [
        "=== OBS-023 Local Directional Mismatch Summary ===",
        "",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Local mismatch",
        f"  mean local_direction_mismatch_deg = {float(loc.mean()):.4f}",
        f"  seam-band mean                    = {float(loc[seam_mask].mean()):.4f}",
        f"  off-seam mean                     = {float(loc[~seam_mask].mean()):.4f}",
        "",
        "Neighbor mismatch",
        f"  mean neighbor_direction_mismatch_deg = {float(nei.mean()):.4f}",
        f"  seam-band mean                       = {float(nei[seam_mask].mean()):.4f}",
        f"  off-seam mean                        = {float(nei[~seam_mask].mean()):.4f}",
        "",
        "Correlations",
        f"  corr(local mismatch, transport misalignment)    = {safe_corr(loc, trn):.4f}",
        f"  corr(neighbor mismatch, transport misalignment) = {safe_corr(nei, trn):.4f}",
        f"  corr(local mismatch, distance_to_seam)          = {safe_corr(loc, d2s):.4f}",
        f"  corr(neighbor mismatch, distance_to_seam)       = {safe_corr(nei, d2s):.4f}",
        f"  corr(local mismatch, node holonomy proxy)       = {safe_corr(loc, hol):.4f}",
        f"  corr(neighbor mismatch, node holonomy proxy)    = {safe_corr(nei, hol):.4f}",
        "",
        "Top local-mismatch nodes",
    ]

    top = nodes.sort_values("local_direction_mismatch_deg", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, r={float(row['r']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"local_direction_mismatch_deg={float(row['local_direction_mismatch_deg']):.4f}, "
            f"neighbor_direction_mismatch_deg={float(row['neighbor_direction_mismatch_deg']):.4f}, "
            f"transport_align_mean_deg={float(row['transport_align_mean_deg']):.4f}, "
            f"distance_to_seam={float(row['distance_to_seam']):.4f}"
        )

    return "\n".join(lines)


def render_figure(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    top = nodes.sort_values("local_direction_mismatch_deg", ascending=False).head(cfg.top_k_labels)

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
        c=pd.to_numeric(nodes["local_direction_mismatch_deg"], errors="coerce"),
        cmap="magma",
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
    cbar.set_label("local directional mismatch (deg)")

    ax_main.set_title("OBS-023 addendum — Local directional mismatch on the phase manifold", fontsize=15, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")

    loc = pd.to_numeric(nodes["local_direction_mismatch_deg"], errors="coerce")
    cats = ["seam-band", "off-seam"]
    vals = [float(loc[seam_mask].mean()), float(loc[~seam_mask].mean())]
    ax_bar.bar(cats, vals, alpha=0.9)
    ax_bar.set_ylabel("mean local mismatch (deg)")
    ax_bar.set_title("Seam localization", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    trn = pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce")
    mask = loc.notna() & trn.notna()
    ax_sc.scatter(loc[mask], trn[mask], s=38, alpha=0.88)
    ax_sc.set_xlabel("local directional mismatch (deg)")
    ax_sc.set_ylabel("transport misalignment (deg)")
    ax_sc.set_title("Local mismatch vs transport misalignment", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)
    ax_sc.text(
        0.98, 0.05,
        f"corr = {safe_corr(loc, trn):.3f}",
        transform=ax_sc.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("PAM Observatory — OBS-023 local directional mismatch", fontsize=19)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-023 local directional mismatch proxy.")
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
    nodes = compute_node_proxy(nodes, edges)

    csv_path = outdir / "local_direction_mismatch_nodes.csv"
    txt_path = outdir / "obs023_local_direction_mismatch_summary.txt"
    png_path = outdir / "obs023_local_direction_mismatch_figure.png"

    nodes.to_csv(csv_path, index=False)
    txt_path.write_text(build_summary(nodes, cfg.seam_threshold), encoding="utf-8")
    render_figure(cfg, nodes, seam, png_path)

    print(csv_path)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
