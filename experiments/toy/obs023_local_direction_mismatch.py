#!/usr/bin/env python3
"""
OBS-023 — local vs neighbor directional mismatch.

Canonical computation built on pam.geometry.directional_field.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pam.geometry.directional_field import DirectionalField


@dataclass(frozen=True)
class Config:
    nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    edges_csv: str = "outputs/obs022_scene_bundle/scene_edges.csv"
    outdir: str = "outputs/obs023_local_direction_mismatch"
    seam_threshold: float = 0.15


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(aa[mask].corr(bb[mask]))


def build_summary(nodes: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold
    lines = [
        "=== OBS-023 Local Directional Mismatch Summary ===",
        "",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Local mismatch",
        f"  mean local_direction_mismatch_deg = {float(pd.to_numeric(nodes['local_direction_mismatch_deg'], errors='coerce').mean()):.4f}",
        f"  seam-band mean                    = {float(pd.to_numeric(nodes.loc[seam_mask, 'local_direction_mismatch_deg'], errors='coerce').mean()):.4f}",
        f"  off-seam mean                     = {float(pd.to_numeric(nodes.loc[~seam_mask, 'local_direction_mismatch_deg'], errors='coerce').mean()):.4f}",
        "",
        "Neighbor mismatch",
        f"  mean neighbor_direction_mismatch_mean = {float(pd.to_numeric(nodes['neighbor_direction_mismatch_mean'], errors='coerce').mean()):.4f}",
        f"  seam-band mean                         = {float(pd.to_numeric(nodes.loc[seam_mask, 'neighbor_direction_mismatch_mean'], errors='coerce').mean()):.4f}",
        f"  off-seam mean                          = {float(pd.to_numeric(nodes.loc[~seam_mask, 'neighbor_direction_mismatch_mean'], errors='coerce').mean()):.4f}",
        "",
        "Correlations",
        f"  corr(local mismatch, distance_to_seam)    = {safe_corr(nodes['local_direction_mismatch_deg'], nodes['distance_to_seam']):.4f}",
        f"  corr(neighbor mismatch, distance_to_seam) = {safe_corr(nodes['neighbor_direction_mismatch_mean'], nodes['distance_to_seam']):.4f}",
        f"  corr(local mismatch, node holonomy)       = {safe_corr(nodes['local_direction_mismatch_deg'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        f"  corr(neighbor mismatch, node holonomy)    = {safe_corr(nodes['neighbor_direction_mismatch_mean'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        "",
        "Top local-mismatch nodes",
    ]
    top = nodes.sort_values("local_direction_mismatch_deg", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, r={float(row['r']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"local_direction_mismatch_deg={float(row['local_direction_mismatch_deg']):.4f}, "
            f"neighbor_direction_mismatch_mean={float(row['neighbor_direction_mismatch_mean']):.4f}, "
            f"distance_to_seam={float(row['distance_to_seam']):.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="OBS-023 local vs neighbor directional mismatch.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--edges-csv", default=Config.edges_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    field = DirectionalField.from_csv(
        cfg.nodes_csv,
        cfg.edges_csv,
        connection_theta_col="fim_theta",
        response_theta_col="rsp_theta",
    )

    local_df = field.local_direction_mismatch(degrees=True).rename(
        columns={"local_direction_mismatch": "local_direction_mismatch_deg"}
    )
    neighbor_df = field.node_neighbor_mismatch(degrees=True)
    nodes = field.attach_node_metrics(local_df, neighbor_df)

    csv_path = outdir / "local_direction_mismatch_nodes.csv"
    txt_path = outdir / "obs023_local_direction_mismatch_summary.txt"

    nodes.to_csv(csv_path, index=False)
    txt_path.write_text(build_summary(nodes, cfg.seam_threshold), encoding="utf-8")

    print(csv_path)
    print(txt_path)


if __name__ == "__main__":
    main()