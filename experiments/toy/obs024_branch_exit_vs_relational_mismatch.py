#!/usr/bin/env python3
"""
OBS-024 — Branch exit vs relational mismatch.

Test whether branch-away / seam-exit routes preferentially sample nodes with
high relational directional obstruction, compared with seam-corridor routes and
the background node set.

Inputs
------
outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv
outputs/obs022_scene_bundle/scene_routes.csv
outputs/obs022_scene_bundle/scene_seam.csv

Outputs
-------
outputs/obs024_branch_exit_vs_relational_mismatch/
  obs024_branch_exit_vs_relational_mismatch_summary.txt
  obs024_branch_exit_vs_relational_mismatch.csv
  obs024_branch_exit_vs_relational_mismatch_figure.png
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
    routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/obs024_branch_exit_vs_relational_mismatch"
    seam_threshold: float = 0.15
    top_k_labels: int = 8
    max_paths_per_class: int = 8


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def safe_std(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.std(ddof=1)) if s.notna().sum() > 1 else float("nan")


def safe_sem(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.notna().sum())
    if n <= 1:
        return 0.0
    return float(s.std(ddof=1) / np.sqrt(n))


def safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.median()) if s.notna().any() else float("nan")


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    routes = pd.read_csv(cfg.routes_csv)
    seam = pd.read_csv(cfg.seam_csv)

    node_cols = [
        "node_id", "r", "alpha", "mds1", "mds2",
        "distance_to_seam",
        "neighbor_direction_mismatch_deg",
        "local_direction_mismatch_deg",
        "transport_align_mean_deg",
    ]
    route_cols = [
        "path_id", "path_family", "step", "node_id", "r", "alpha", "mds1", "mds2",
        "distance_to_seam", "is_representative", "is_branch_away",
    ]
    seam_cols = ["mds1", "mds2"]

    for c in node_cols:
        if c in nodes.columns:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
    for c in route_cols:
        if c in routes.columns and c not in {"path_id", "path_family"}:
            routes[c] = pd.to_numeric(routes[c], errors="coerce")
    for c in seam_cols:
        if c in seam.columns:
            seam[c] = pd.to_numeric(seam[c], errors="coerce")

    if "neighbor_direction_mismatch_deg" not in nodes.columns:
        raise ValueError("nodes csv must contain neighbor_direction_mismatch_deg")
    if "node_id" not in nodes.columns:
        raise ValueError("nodes csv must contain node_id")
    if "is_branch_away" not in routes.columns:
        routes["is_branch_away"] = 0
    if "is_representative" not in routes.columns:
        routes["is_representative"] = 0

    return nodes, routes, seam


def prepare_route_table(nodes: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    enrich_cols = [
        c for c in [
            "node_id",
            "neighbor_direction_mismatch_deg",
            "local_direction_mismatch_deg",
            "transport_align_mean_deg",
        ] if c in nodes.columns
    ]
    out = routes.merge(nodes[enrich_cols].drop_duplicates(subset=["node_id"]), on="node_id", how="left")
    out = out.sort_values(["path_id", "step"]).reset_index(drop=True)
    return out


def summarize_path_classes(routes: pd.DataFrame, seam_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build both:
    - row-level class table
    - path-level class summary
    """
    work = routes.copy()

    work["route_class"] = np.select(
        [
            work["is_branch_away"] == 1,
            (work["is_representative"] == 1) & (work["path_family"] == "stable_seam_corridor"),
            (work["is_representative"] == 1) & (work["path_family"] == "reorganization_heavy"),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )

    path_rows = []
    for path_id, grp in work.groupby("path_id", sort=False):
        grp = grp.sort_values("step")
        cls = str(grp["route_class"].iloc[0])
        fam = str(grp["path_family"].iloc[0]) if "path_family" in grp.columns else "unknown"

        rel = pd.to_numeric(grp["neighbor_direction_mismatch_deg"], errors="coerce")
        loc = pd.to_numeric(grp["local_direction_mismatch_deg"], errors="coerce")
        d2s = pd.to_numeric(grp["distance_to_seam"], errors="coerce")

        seam_mask = (d2s <= seam_threshold).fillna(False)

        path_rows.append(
            {
                "path_id": path_id,
                "route_class": cls,
                "path_family": fam,
                "n_steps": int(pd.to_numeric(grp["step"], errors="coerce").max()) if "step" in grp.columns else len(grp) - 1,
                "mean_relational_mismatch": safe_mean(rel),
                "max_relational_mismatch": float(rel.max()) if rel.notna().any() else float("nan"),
                "mean_local_mismatch": safe_mean(loc),
                "mean_distance_to_seam": safe_mean(d2s),
                "min_distance_to_seam": float(d2s.min()) if d2s.notna().any() else float("nan"),
                "seam_fraction": float(seam_mask.mean()) if len(seam_mask) else float("nan"),
            }
        )

    path_df = pd.DataFrame(path_rows)

    summary_rows = []
    for cls, grp in path_df.groupby("route_class", sort=False):
        summary_rows.append(
            {
                "route_class": cls,
                "n_paths": len(grp),
                "mean_relational_mismatch": safe_mean(grp["mean_relational_mismatch"]),
                "median_relational_mismatch": safe_median(grp["mean_relational_mismatch"]),
                "sem_relational_mismatch": safe_sem(grp["mean_relational_mismatch"]),
                "mean_local_mismatch": safe_mean(grp["mean_local_mismatch"]),
                "mean_distance_to_seam": safe_mean(grp["mean_distance_to_seam"]),
                "mean_seam_fraction": safe_mean(grp["seam_fraction"]),
            }
        )

    return work, pd.DataFrame(summary_rows).sort_values("route_class").reset_index(drop=True), path_df


def build_summary(nodes: pd.DataFrame, path_df: pd.DataFrame, class_summary: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold
    bg_rel = pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce")

    lines = [
        "=== OBS-024 Branch Exit vs Relational Mismatch Summary ===",
        "",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Background node field",
        f"  mean relational mismatch      = {safe_mean(bg_rel):.4f}",
        f"  seam-band mean                = {safe_mean(bg_rel[seam_mask]):.4f}",
        f"  off-seam mean                 = {safe_mean(bg_rel[~seam_mask]):.4f}",
        "",
        "Path-class summary",
    ]

    for _, row in class_summary.iterrows():
        lines.append(
            f"  {row['route_class']}: "
            f"n_paths={int(row['n_paths'])}, "
            f"mean_relational_mismatch={float(row['mean_relational_mismatch']):.4f}, "
            f"median_relational_mismatch={float(row['median_relational_mismatch']):.4f}, "
            f"mean_local_mismatch={float(row['mean_local_mismatch']):.4f}, "
            f"mean_distance_to_seam={float(row['mean_distance_to_seam']):.4f}, "
            f"mean_seam_fraction={float(row['mean_seam_fraction']):.4f}"
        )

    top_branch = (
        path_df[path_df["route_class"] == "branch_exit"]
        .sort_values("mean_relational_mismatch", ascending=False)
        .head(10)
    )
    lines.extend(["", "Top branch-exit paths by mean relational mismatch"])
    for _, row in top_branch.iterrows():
        lines.append(
            f"  {row['path_id']}: "
            f"mean_relational_mismatch={float(row['mean_relational_mismatch']):.4f}, "
            f"max_relational_mismatch={float(row['max_relational_mismatch']):.4f}, "
            f"mean_distance_to_seam={float(row['mean_distance_to_seam']):.4f}, "
            f"min_distance_to_seam={float(row['min_distance_to_seam']):.4f}, "
            f"seam_fraction={float(row['seam_fraction']):.4f}"
        )

    return "\n".join(lines)


def render_figure(
    cfg: Config,
    nodes: pd.DataFrame,
    routes: pd.DataFrame,
    seam: pd.DataFrame,
    class_summary: pd.DataFrame,
    path_df: pd.DataFrame,
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 1.7, 1.2], height_ratios=[1.0, 1.0])

    ax_main = fig.add_subplot(gs[:, 0:2])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_box = fig.add_subplot(gs[1, 2])

    # background nodes
    ax_main.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce"),
        cmap="viridis",
        s=46,
        alpha=0.25,
        linewidths=0,
        zorder=1,
    )

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=6.0, alpha=0.65, zorder=2)
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.8, alpha=0.96, zorder=3)

    # representative paths by class
    class_styles = {
        "branch_exit": dict(color="#7FDBFF", linewidth=2.4, alpha=0.92, zorder=6),
        "stable_seam_corridor": dict(color="#D4A72C", linewidth=2.8, alpha=0.94, zorder=5),
        "reorganization_heavy": dict(color="#B23A48", linewidth=1.9, alpha=0.68, zorder=4),
    }

    path_rank = (
        path_df.sort_values(["route_class", "mean_relational_mismatch"], ascending=[True, False])
        .groupby("route_class")["path_id"]
        .apply(list)
        .to_dict()
    )

    for cls, style in class_styles.items():
        keep_ids = path_rank.get(cls, [])[: cfg.max_paths_per_class]
        cls_df = routes[routes["path_id"].isin(keep_ids) & (routes["route_class"] == cls)].copy()

        for _, grp in cls_df.groupby("path_id", sort=False):
            grp = grp.sort_values("step")
            ax_main.plot(
                grp["mds1"], grp["mds2"],
                color=style["color"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                zorder=style["zorder"],
            )

    # top branch-exit hotspot labels
    top_branch = (
        path_df[path_df["route_class"] == "branch_exit"]
        .sort_values("mean_relational_mismatch", ascending=False)
        .head(cfg.top_k_labels)
    )
    label_ids = set(top_branch["path_id"].tolist())

    label_points = (
        routes[routes["path_id"].isin(label_ids) & (routes["route_class"] == "branch_exit")]
        .groupby("path_id", as_index=False)
        .agg(
            mds1=("mds1", "mean"),
            mds2=("mds2", "mean"),
            mean_relational_mismatch=("neighbor_direction_mismatch_deg", "mean"),
        )
    )
    for _, row in label_points.iterrows():
        ax_main.scatter([row["mds1"]], [row["mds2"]], s=150, facecolors="none", edgecolors="#FFD166", linewidths=1.7, zorder=7)
        ax_main.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            str(row["path_id"]),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=8,
        )

    ax_main.set_title("OBS-024 follow-up — Branch-exit routes on the relational obstruction field", fontsize=15, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.text(
        0.02,
        0.98,
        "cyan = branch exit\ngold = stable seam corridor\ncrimson = reorganization heavy\nyellow labels = highest branch-exit relational-mismatch paths",
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92),
        zorder=9,
    )

    # bar chart
    plot_order = ["branch_exit", "stable_seam_corridor", "reorganization_heavy", "other"]
    bar_df = class_summary[class_summary["route_class"].isin(plot_order)].copy()
    bar_df["order"] = bar_df["route_class"].map({k: i for i, k in enumerate(plot_order)})
    bar_df = bar_df.sort_values("order")

    ax_bar.bar(
        bar_df["route_class"],
        bar_df["mean_relational_mismatch"],
        yerr=bar_df["sem_relational_mismatch"],
        alpha=0.9,
    )
    ax_bar.set_ylabel("mean relational mismatch (deg)")
    ax_bar.set_title("Path-class comparison", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")
    ax_bar.tick_params(axis="x", rotation=20)

    # box/strip style scatter
    cls_vals = []
    cls_names = []
    for cls in ["branch_exit", "stable_seam_corridor", "reorganization_heavy"]:
        vals = path_df.loc[path_df["route_class"] == cls, "mean_relational_mismatch"].dropna().to_numpy()
        if len(vals):
            cls_vals.append(vals)
            cls_names.append(cls)

    ax_box.boxplot(cls_vals, labels=cls_names, vert=True)
    for i, vals in enumerate(cls_vals, start=1):
        x = np.full(len(vals), i, dtype=float) + np.random.normal(0, 0.04, size=len(vals))
        ax_box.scatter(x, vals, s=18, alpha=0.55)
    ax_box.set_ylabel("path mean relational mismatch (deg)")
    ax_box.set_title("Path-level distribution", fontsize=14, pad=8)
    ax_box.grid(alpha=0.15, axis="y")
    ax_box.tick_params(axis="x", rotation=20)

    fig.suptitle("PAM Observatory — OBS-024 branch exit vs relational mismatch", fontsize=19)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-024 branch-exit vs relational-mismatch figure.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    parser.add_argument("--max-paths-per-class", type=int, default=Config.max_paths_per_class)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        routes_csv=args.routes_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_labels=args.top_k_labels,
        max_paths_per_class=args.max_paths_per_class,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, routes, seam = load_inputs(cfg)
    routes, class_summary, path_df = summarize_path_classes(prepare_route_table(nodes, routes), cfg.seam_threshold)

    csv_path = outdir / "obs024_branch_exit_vs_relational_mismatch.csv"
    txt_path = outdir / "obs024_branch_exit_vs_relational_mismatch_summary.txt"
    png_path = outdir / "obs024_branch_exit_vs_relational_mismatch_figure.png"

    class_summary.to_csv(csv_path, index=False)
    txt_path.write_text(build_summary(nodes, path_df, class_summary, cfg.seam_threshold), encoding="utf-8")
    render_figure(cfg, nodes, routes, seam, class_summary, path_df, png_path)

    print(csv_path)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
