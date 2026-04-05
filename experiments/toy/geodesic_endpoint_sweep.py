#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_endpoint_sweep.py

Run structured pairwise geodesics over an endpoint manifest.

Expected manifest columns
-------------------------
- endpoint_id
- node_id
- r
- alpha
- seam_bin
- dew_sector

Outputs
-------
- geodesic_endpoint_pairs.csv
- geodesic_endpoint_sweep_path_nodes.csv
- geodesic_endpoint_sweep_summary.txt
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from pam.geometry.geodesics import dijkstra, load_geodesic_inputs


@dataclass(frozen=True)
class Config:
    manifest_csv: str
    nodes_csv: str
    edges_csv: str
    coords_csv: str
    outdir: str = "outputs/toy_geodesic_endpoint_sweep"


def build_path_nodes_df(
    path: list[int],
    path_id: str,
    nodes: pd.DataFrame,
    coords: pd.DataFrame,
    source_node_id: int,
    target_node_id: int,
    meta: dict[str, object],
) -> pd.DataFrame:
    node_lookup = nodes.set_index("node_id")
    rows = []
    for step, node_id in enumerate(path):
        rec = {
            "path_id": path_id,
            "step": int(step),
            "node_id": int(node_id),
            "source_node_id": int(source_node_id),
            "target_node_id": int(target_node_id),
            "r": np.nan,
            "alpha": np.nan,
            "mds1": np.nan,
            "mds2": np.nan,
            **meta,
        }
        if int(node_id) in node_lookup.index:
            rec["r"] = float(node_lookup.loc[int(node_id), "r"])
            rec["alpha"] = float(node_lookup.loc[int(node_id), "alpha"])
        if int(node_id) in coords.index:
            rec["mds1"] = float(coords.loc[int(node_id), "mds1"])
            rec["mds2"] = float(coords.loc[int(node_id), "mds2"])
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(pair_df: pd.DataFrame, path_nodes_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== Geodesic Endpoint Sweep Summary ===")
    lines.append("")
    lines.append(f"n_pairs = {len(pair_df)}")
    lines.append(f"n_paths = {pair_df['path_found'].sum() if 'path_found' in pair_df.columns else 0}")
    lines.append(f"n_path_nodes = {len(path_nodes_df)}")
    lines.append("")
    if not pair_df.empty:
        counts = (
            pair_df.groupby(["src_seam_bin", "dst_seam_bin"], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        lines.append("Pair counts by seam-bin class")
        for _, row in counts.iterrows():
            lines.append(f"  {row['src_seam_bin']} -> {row['dst_seam_bin']}: {int(row['n'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured pairwise geodesics from endpoint manifest.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--edges-csv", required=True)
    parser.add_argument("--coords-csv", required=True)
    parser.add_argument("--outdir", default="outputs/toy_geodesic_endpoint_sweep")
    args = parser.parse_args()

    cfg = Config(
        manifest_csv=args.manifest_csv,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        coords_csv=args.coords_csv,
        outdir=args.outdir,
    )
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg.manifest_csv).copy()
    nodes, edges, coords, graph = load_geodesic_inputs(cfg.nodes_csv, cfg.edges_csv, cfg.coords_csv)

    pair_rows: list[dict[str, object]] = []
    path_frames: list[pd.DataFrame] = []

    for idx, (a, b) in enumerate(combinations(manifest.itertuples(index=False), 2), start=1):
        src_node_id = int(a.node_id)
        dst_node_id = int(b.node_id)
        path_id = f"path_{idx:03d}"

        path = dijkstra(graph, src_node_id, dst_node_id)
        found = len(path) > 0

        pair_rows.append(
            {
                "path_id": path_id,
                "src_endpoint_id": a.endpoint_id,
                "dst_endpoint_id": b.endpoint_id,
                "src_node_id": src_node_id,
                "dst_node_id": dst_node_id,
                "src_r": float(a.r),
                "src_alpha": float(a.alpha),
                "dst_r": float(b.r),
                "dst_alpha": float(b.alpha),
                "src_seam_bin": a.seam_bin,
                "dst_seam_bin": b.seam_bin,
                "src_sector": a.dew_sector,
                "dst_sector": b.dew_sector,
                "path_found": found,
                "n_nodes": len(path) if found else 0,
                "n_steps": max(0, len(path) - 1) if found else 0,
            }
        )

        if found:
            meta = {
                "src_endpoint_id": a.endpoint_id,
                "dst_endpoint_id": b.endpoint_id,
                "src_seam_bin": a.seam_bin,
                "dst_seam_bin": b.seam_bin,
                "src_sector": a.dew_sector,
                "dst_sector": b.dew_sector,
            }
            path_frames.append(
                build_path_nodes_df(
                    path,
                    path_id,
                    nodes,
                    coords,
                    src_node_id,
                    dst_node_id,
                    meta,
                )
            )

    pair_df = pd.DataFrame(pair_rows)
    path_nodes_df = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()

    pair_df.to_csv(outdir / "geodesic_endpoint_pairs.csv", index=False)
    path_nodes_df.to_csv(outdir / "geodesic_endpoint_sweep_path_nodes.csv", index=False)
    (outdir / "geodesic_endpoint_sweep_summary.txt").write_text(
        summarize(pair_df, path_nodes_df),
        encoding="utf-8",
    )

    print(outdir / "geodesic_endpoint_pairs.csv")
    print(outdir / "geodesic_endpoint_sweep_path_nodes.csv")
    print(outdir / "geodesic_endpoint_sweep_summary.txt")


if __name__ == "__main__":
    main()
