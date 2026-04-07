#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    paths_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    critical_csv: str = "outputs/fim_critical/critical_points.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    phase_csv: str = "outputs/fim_phase/signed_phase_surface.csv"
    outdir: str = "outputs/toy_geodesic_branch_hubs"
    top_k: int = 20
    seam_quantile: float = 0.15
    lazarus_quantile: float = 0.85


def load_paths(paths_csv: str) -> pd.DataFrame:
    df = pd.read_csv(paths_csv).copy()
    if "probe_id" in df.columns and "path_id" not in df.columns:
        df = df.rename(columns={"probe_id": "path_id"})
    need = {"path_id", "node_id", "r", "alpha"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"paths csv missing columns: {sorted(missing)}")
    return df


def load_node_annotations(
    phase_csv: str,
    lazarus_csv: str,
    critical_csv: str,
) -> pd.DataFrame:
    phase = pd.read_csv(phase_csv).copy()
    laz = pd.read_csv(lazarus_csv).copy()
    crit = pd.read_csv(critical_csv).copy() if Path(critical_csv).exists() else pd.DataFrame()

    keep_phase = [c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase", "distance_to_seam"] if c in phase.columns]
    df = phase[keep_phase].copy()

    keep_laz = [c for c in ["node_id", "r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns]
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in df.columns and c in laz.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]
    df = df.merge(laz[keep_laz], on=join_cols, how="left")

    if not crit.empty:
        crit = crit.copy()
        crit["is_critical_point"] = 1
        keep_crit = [c for c in ["node_id", "r", "alpha", "is_critical_point"] if c in crit.columns]
        join_cols = [c for c in ["node_id", "r", "alpha"] if c in df.columns and c in crit.columns]
        if not join_cols:
            join_cols = ["r", "alpha"]
        df = df.merge(crit[keep_crit], on=join_cols, how="left")
    else:
        df["is_critical_point"] = 0

    if "is_critical_point" not in df.columns:
        df["is_critical_point"] = 0
    df["is_critical_point"] = pd.to_numeric(df["is_critical_point"], errors="coerce").fillna(0).astype(int)

    return df


def build_node_traffic(paths: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        paths.groupby(["node_id", "r", "alpha"], as_index=False)
        .agg(
            n_visits=("path_id", "size"),
            n_unique_paths=("path_id", "nunique"),
        )
    )

    n_total_paths = paths["path_id"].nunique()
    grouped["path_occupancy"] = grouped["n_unique_paths"] / max(n_total_paths, 1)
    return grouped.sort_values(["n_unique_paths", "n_visits"], ascending=False).reset_index(drop=True)


def summarize_overlap(
    nodes: pd.DataFrame,
    seam_quantile: float,
    lazarus_quantile: float,
    top_k: int,
) -> tuple[pd.DataFrame, str]:
    out = nodes.copy()

    seam_thr = float(pd.to_numeric(out["distance_to_seam"], errors="coerce").quantile(seam_quantile))
    laz_thr = float(pd.to_numeric(out["lazarus_score"], errors="coerce").quantile(lazarus_quantile))

    out["is_seam_hub_zone"] = (
        pd.to_numeric(out["distance_to_seam"], errors="coerce") <= seam_thr
    ).astype(int)
    out["is_high_lazarus_zone"] = (
        pd.to_numeric(out["lazarus_score"], errors="coerce") >= laz_thr
    ).astype(int)

    top = out.head(top_k).copy()

    lines = []
    lines.append("=== Geodesic Branch Hub Summary ===")
    lines.append("")
    lines.append(f"n_nodes = {len(out)}")
    lines.append(f"top_k = {top_k}")
    lines.append(f"seam_quantile = {seam_quantile:.2f} -> seam_threshold = {seam_thr:.4f}")
    lines.append(f"lazarus_quantile = {lazarus_quantile:.2f} -> lazarus_threshold = {laz_thr:.4f}")
    lines.append("")
    lines.append("Top-hub overlap")
    lines.append(
        f"  seam-near hubs in top_k = {int(top['is_seam_hub_zone'].sum())}/{len(top)}"
    )
    lines.append(
        f"  critical hubs in top_k = {int(top['is_critical_point'].sum())}/{len(top)}"
    )
    lines.append(
        f"  high-Lazarus hubs in top_k = {int(top['is_high_lazarus_zone'].sum())}/{len(top)}"
    )
    lines.append(
        f"  seam+critical hubs in top_k = {int(((top['is_seam_hub_zone'] == 1) & (top['is_critical_point'] == 1)).sum())}/{len(top)}"
    )
    lines.append(
        f"  seam+high-Lazarus hubs in top_k = {int(((top['is_seam_hub_zone'] == 1) & (top['is_high_lazarus_zone'] == 1)).sum())}/{len(top)}"
    )
    lines.append("")
    lines.append("Top hubs")
    for _, row in top.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, "
            f"r={row['r']:.3f}, alpha={row['alpha']:.6f}, "
            f"n_unique_paths={int(row['n_unique_paths'])}, "
            f"path_occupancy={row['path_occupancy']:.4f}, "
            f"distance_to_seam={row['distance_to_seam']:.4f}, "
            f"lazarus_score={row['lazarus_score']:.4f}, "
            f"is_critical_point={int(row['is_critical_point'])}"
        )

    return top, "\n".join(lines)


def plot_hubs(nodes: pd.DataFrame, outpath: Path, top_k: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        s=20,
        alpha=0.25,
    )

    top = nodes.head(top_k).copy()
    sizes = 40 + 260 * (top["path_occupancy"] / max(top["path_occupancy"].max(), 1e-9))
    sc = ax.scatter(
        top["mds1"],
        top["mds2"],
        s=sizes,
        c=top["distance_to_seam"],
        alpha=0.95,
    )

    for _, row in top.iterrows():
        ax.text(row["mds1"], row["mds2"], str(int(row["node_id"])), fontsize=8)

    fig.colorbar(sc, ax=ax, label="distance_to_seam")
    ax.set_title("Top geodesic branch hubs")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_hub_vs_fields(nodes: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
    y = pd.to_numeric(nodes["path_occupancy"], errors="coerce")
    ax.scatter(x, y, s=28, alpha=0.65)
    ax.set_title("Node traffic vs seam distance")
    ax.set_xlabel("distance_to_seam")
    ax.set_ylabel("path_occupancy")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure geodesic branch-hub occupancy and overlap with seam/critical/Lazarus structure.")
    parser.add_argument("--paths-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--critical-csv", default="outputs/fim_critical/critical_points.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_surface.csv")
    parser.add_argument("--outdir", default="outputs/toy_geodesic_branch_hubs")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seam-quantile", type=float, default=0.15)
    parser.add_argument("--lazarus-quantile", type=float, default=0.85)
    args = parser.parse_args()

    cfg = Config(
        paths_csv=args.paths_csv,
        critical_csv=args.critical_csv,
        lazarus_csv=args.lazarus_csv,
        phase_csv=args.phase_csv,
        outdir=args.outdir,
        top_k=args.top_k,
        seam_quantile=args.seam_quantile,
        lazarus_quantile=args.lazarus_quantile,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(cfg.paths_csv)
    traffic = build_node_traffic(paths)
    ann = load_node_annotations(cfg.phase_csv, cfg.lazarus_csv, cfg.critical_csv)

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in traffic.columns and c in ann.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    nodes = traffic.merge(ann, on=join_cols, how="left")
    nodes = nodes.sort_values(["n_unique_paths", "n_visits"], ascending=False).reset_index(drop=True)

    top, summary_text = summarize_overlap(
        nodes,
        seam_quantile=cfg.seam_quantile,
        lazarus_quantile=cfg.lazarus_quantile,
        top_k=cfg.top_k,
    )

    nodes.to_csv(outdir / "geodesic_branch_hub_nodes.csv", index=False)
    top.to_csv(outdir / "geodesic_branch_hub_top_nodes.csv", index=False)
    (outdir / "geodesic_branch_hub_summary.txt").write_text(summary_text, encoding="utf-8")

    plot_hubs(nodes, outdir / "geodesic_branch_hub_map.png", cfg.top_k)
    plot_hub_vs_fields(nodes, outdir / "geodesic_branch_hub_vs_seam.png")

    print(outdir / "geodesic_branch_hub_nodes.csv")
    print(outdir / "geodesic_branch_hub_top_nodes.csv")
    print(outdir / "geodesic_branch_hub_summary.txt")
    print(outdir / "geodesic_branch_hub_map.png")
    print(outdir / "geodesic_branch_hub_vs_seam.png")


if __name__ == "__main__":
    main()
