#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    paths_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    seam_csv: str = "outputs/fim_phase/phase_distance_to_seam.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    critical_csv: str = "outputs/fim_critical/critical_points.csv"
    outdir: str = "outputs/toy_family_branch_hub_usage"
    top_k: int = 20
    seam_quantile: float = 0.15
    lazarus_quantile: float = 0.85


def load_paths(paths_csv: str, family_csv: str) -> pd.DataFrame:
    paths = pd.read_csv(paths_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    need = {"path_id", "node_id", "r", "alpha"}
    missing = need - set(paths.columns)
    if missing:
        raise ValueError(f"paths csv missing columns: {sorted(missing)}")

    if "path_family" not in fam.columns:
        raise ValueError("family csv must contain path_family")

    out = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")
    return out


def load_node_annotations(
    phase_csv: str,
    seam_csv: str,
    lazarus_csv: str,
    critical_csv: str,
) -> pd.DataFrame:
    phase = pd.read_csv(phase_csv).copy()
    seam = pd.read_csv(seam_csv).copy() if Path(seam_csv).exists() else pd.DataFrame()
    laz = pd.read_csv(lazarus_csv).copy()
    crit = pd.read_csv(critical_csv).copy() if Path(critical_csv).exists() else pd.DataFrame()

    keep_phase = [c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase"] if c in phase.columns]
    df = phase[keep_phase].copy()

    if not seam.empty:
        keep_seam = [c for c in ["node_id", "r", "alpha", "distance_to_seam"] if c in seam.columns]
        join_cols = [c for c in ["node_id", "r", "alpha"] if c in df.columns and c in seam.columns]
        if not join_cols:
            join_cols = ["r", "alpha"]
        df = df.merge(seam[keep_seam], on=join_cols, how="left")

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

    if "distance_to_seam" not in df.columns:
        df["distance_to_seam"] = np.nan
    if "is_critical_point" not in df.columns:
        df["is_critical_point"] = 0

    df["is_critical_point"] = pd.to_numeric(df["is_critical_point"], errors="coerce").fillna(0).astype(int)
    return df


def build_family_node_traffic(paths: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        paths.groupby(["path_family", "node_id", "r", "alpha"], as_index=False)
        .agg(
            n_visits=("path_id", "size"),
            n_unique_paths=("path_id", "nunique"),
        )
    )

    totals = (
        paths.groupby("path_family", as_index=False)
        .agg(total_paths=("path_id", "nunique"))
    )

    grouped = grouped.merge(totals, on="path_family", how="left")
    grouped["path_occupancy"] = grouped["n_unique_paths"] / grouped["total_paths"].clip(lower=1)
    return grouped.sort_values(["path_family", "n_unique_paths", "n_visits"], ascending=[True, False, False]).reset_index(drop=True)


def herfindahl(shares: np.ndarray) -> float:
    shares = np.asarray(shares, dtype=float)
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return float("nan")
    s = shares.sum()
    if s <= 0:
        return float("nan")
    p = shares / s
    return float(np.sum(p ** 2))


def shannon_entropy(shares: np.ndarray) -> float:
    shares = np.asarray(shares, dtype=float)
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return float("nan")
    s = shares.sum()
    if s <= 0:
        return float("nan")
    p = shares / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def summarize_by_family(
    nodes: pd.DataFrame,
    top_k: int,
    seam_quantile: float,
    lazarus_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = nodes.copy()

    seam_thr = float(pd.to_numeric(out["distance_to_seam"], errors="coerce").quantile(seam_quantile))
    laz_thr = float(pd.to_numeric(out["lazarus_score"], errors="coerce").quantile(lazarus_quantile))

    out["is_seam_hub_zone"] = (
        pd.to_numeric(out["distance_to_seam"], errors="coerce") <= seam_thr
    ).astype(int)
    out["is_high_lazarus_zone"] = (
        pd.to_numeric(out["lazarus_score"], errors="coerce") >= laz_thr
    ).astype(int)

    rows = []
    top_rows = []

    for fam, sub in out.groupby("path_family", sort=False):
        sub = sub.sort_values(["n_unique_paths", "n_visits"], ascending=False).reset_index(drop=True)
        top = sub.head(top_k).copy()

        total_unique_path_hits = float(sub["n_unique_paths"].sum())
        total_visits = float(sub["n_visits"].sum())

        top_k_share = float(top["n_unique_paths"].sum() / total_unique_path_hits) if total_unique_path_hits > 0 else float("nan")
        seam_share = float(sub.loc[sub["is_seam_hub_zone"] == 1, "n_unique_paths"].sum() / total_unique_path_hits) if total_unique_path_hits > 0 else float("nan")
        high_laz_share = float(sub.loc[sub["is_high_lazarus_zone"] == 1, "n_unique_paths"].sum() / total_unique_path_hits) if total_unique_path_hits > 0 else float("nan")
        critical_share = float(sub.loc[sub["is_critical_point"] == 1, "n_unique_paths"].sum() / total_unique_path_hits) if total_unique_path_hits > 0 else float("nan")

        rows.append(
            {
                "path_family": fam,
                "n_nodes_used": int(len(sub)),
                "top_k_hub_share": top_k_share,
                "traffic_herfindahl": herfindahl(sub["n_unique_paths"].to_numpy(dtype=float)),
                "traffic_entropy": shannon_entropy(sub["n_unique_paths"].to_numpy(dtype=float)),
                "seam_hub_traffic_share": seam_share,
                "high_lazarus_hub_traffic_share": high_laz_share,
                "critical_hub_traffic_share": critical_share,
                "top_k_seam_hubs": int(top["is_seam_hub_zone"].sum()),
                "top_k_high_lazarus_hubs": int(top["is_high_lazarus_zone"].sum()),
                "top_k_critical_hubs": int(top["is_critical_point"].sum()),
                "mean_top_k_distance_to_seam": float(pd.to_numeric(top["distance_to_seam"], errors="coerce").mean()),
                "mean_top_k_lazarus": float(pd.to_numeric(top["lazarus_score"], errors="coerce").mean()),
            }
        )

        top["top_rank"] = np.arange(1, len(top) + 1)
        top_rows.append(top)

    return pd.DataFrame(rows), pd.concat(top_rows, ignore_index=True)


def write_summary_text(summary: pd.DataFrame, outpath: Path, top_k: int) -> None:
    lines = ["=== Family Branch Hub Usage Summary ===", ""]
    lines.append(f"top_k = {top_k}")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(
            f"{row['path_family']}: "
            f"n_nodes_used={int(row['n_nodes_used'])}, "
            f"top_k_hub_share={row['top_k_hub_share']:.4f}, "
            f"traffic_herfindahl={row['traffic_herfindahl']:.4f}, "
            f"traffic_entropy={row['traffic_entropy']:.4f}, "
            f"seam_hub_traffic_share={row['seam_hub_traffic_share']:.4f}, "
            f"high_lazarus_hub_traffic_share={row['high_lazarus_hub_traffic_share']:.4f}, "
            f"critical_hub_traffic_share={row['critical_hub_traffic_share']:.4f}, "
            f"top_k_seam_hubs={int(row['top_k_seam_hubs'])}, "
            f"top_k_high_lazarus_hubs={int(row['top_k_high_lazarus_hubs'])}, "
            f"top_k_critical_hubs={int(row['top_k_critical_hubs'])}, "
            f"mean_top_k_distance_to_seam={row['mean_top_k_distance_to_seam']:.4f}, "
            f"mean_top_k_lazarus={row['mean_top_k_lazarus']:.4f}"
        )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(summary: pd.DataFrame, value_col: str, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    ax.bar(x, pd.to_numeric(summary[value_col], errors="coerce").to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(summary["path_family"], rotation=20, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare branch-hub usage across canonical route families.")
    parser.add_argument("--paths-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--critical-csv", default="outputs/fim_critical/critical_points.csv")
    parser.add_argument("--outdir", default="outputs/toy_family_branch_hub_usage")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seam-quantile", type=float, default=0.15)
    parser.add_argument("--lazarus-quantile", type=float, default=0.85)
    args = parser.parse_args()

    cfg = Config(
        paths_csv=args.paths_csv,
        family_csv=args.family_csv,
        phase_csv=args.phase_csv,
        seam_csv=args.seam_csv,
        lazarus_csv=args.lazarus_csv,
        critical_csv=args.critical_csv,
        outdir=args.outdir,
        top_k=args.top_k,
        seam_quantile=args.seam_quantile,
        lazarus_quantile=args.lazarus_quantile,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(cfg.paths_csv, cfg.family_csv)
    traffic = build_family_node_traffic(paths)
    ann = load_node_annotations(cfg.phase_csv, cfg.seam_csv, cfg.lazarus_csv, cfg.critical_csv)

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in traffic.columns and c in ann.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    nodes = traffic.merge(ann, on=join_cols, how="left")
    summary, top_nodes = summarize_by_family(
        nodes,
        top_k=cfg.top_k,
        seam_quantile=cfg.seam_quantile,
        lazarus_quantile=cfg.lazarus_quantile,
    )

    nodes.to_csv(outdir / "family_branch_hub_nodes.csv", index=False)
    top_nodes.to_csv(outdir / "family_branch_hub_top_nodes.csv", index=False)
    summary.to_csv(outdir / "family_branch_hub_summary.csv", index=False)
    write_summary_text(summary, outdir / "family_branch_hub_summary.txt", cfg.top_k)

    plot_metric(summary, "top_k_hub_share", "Top-k hub traffic share by family", outdir / "family_branch_hub_topk_share.png")
    plot_metric(summary, "seam_hub_traffic_share", "Seam-hub traffic share by family", outdir / "family_branch_hub_seam_share.png")
    plot_metric(summary, "high_lazarus_hub_traffic_share", "High-Lazarus hub traffic share by family", outdir / "family_branch_hub_lazarus_share.png")

    print(outdir / "family_branch_hub_nodes.csv")
    print(outdir / "family_branch_hub_top_nodes.csv")
    print(outdir / "family_branch_hub_summary.csv")
    print(outdir / "family_branch_hub_summary.txt")
    print(outdir / "family_branch_hub_topk_share.png")
    print(outdir / "family_branch_hub_seam_share.png")
    print(outdir / "family_branch_hub_lazarus_share.png")


if __name__ == "__main__":
    main()
