#!/usr/bin/env python3
"""
OBS-024 — Family hotspot occupancy.

Compare how route classes occupy the highest relational-mismatch nodes.

Goal
----
Test whether:
- stable_seam_corridor repeatedly occupies the same seam-core obstruction nodes
- reorganization_heavy spreads across a broader stressed set
- branch_exit touches hotspot nodes more transiently

Inputs
------
outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv
outputs/obs022_scene_bundle/scene_routes.csv
outputs/obs022_scene_bundle/scene_seam.csv

Outputs
-------
outputs/obs024_family_hotspot_occupancy/
  family_hotspot_occupancy_nodes.csv
  family_hotspot_occupancy_summary.csv
  obs024_family_hotspot_occupancy_summary.txt
  obs024_family_hotspot_occupancy_figure.png
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
    outdir: str = "outputs/obs024_family_hotspot_occupancy"
    seam_threshold: float = 0.15
    hotspot_quantile: float = 0.85
    top_label_k: int = 10


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def safe_sem(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.notna().sum())
    if n <= 1:
        return 0.0
    return float(s.std(ddof=1) / np.sqrt(n))


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
        "step", "node_id", "r", "alpha", "mds1", "mds2",
        "distance_to_seam", "is_branch_away", "is_representative",
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

    return nodes, routes, seam


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
    out = routes.copy()
    out["route_class"] = np.select(
        [
            out["is_branch_away"] == 1,
            (out["is_representative"] == 1) & (out["path_family"] == "stable_seam_corridor"),
            (out["is_representative"] == 1) & (out["path_family"] == "reorganization_heavy"),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )
    return out


def prepare_tables(nodes: pd.DataFrame, routes: pd.DataFrame, hotspot_quantile: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    nodes = nodes.copy()
    routes = classify_routes(routes.copy())

    threshold = float(
        pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce").quantile(hotspot_quantile)
    )
    nodes["is_hotspot"] = (
        pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce") >= threshold
    ).astype(int)

    # Do not merge distance_to_seam here; routes already has it.
    enrich = nodes[[
        "node_id",
        "neighbor_direction_mismatch_deg",
        "is_hotspot",
    ]].drop_duplicates(subset=["node_id"])

    routes = routes.merge(enrich, on="node_id", how="left")

    return nodes, routes, threshold


def compute_occupancy(nodes: pd.DataFrame, routes: pd.DataFrame, seam_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if "distance_to_seam" not in routes.columns:
        if "distance_to_seam_x" in routes.columns:
            routes = routes.rename(columns={"distance_to_seam_x": "distance_to_seam"})
        elif "distance_to_seam_y" in routes.columns:
            routes = routes.rename(columns={"distance_to_seam_y": "distance_to_seam"})

    work = routes[routes["route_class"].isin(CLASS_ORDER)].copy()
    hotspot_nodes = set(nodes.loc[nodes["is_hotspot"] == 1, "node_id"].dropna().astype(int).tolist())

    node_rows = []
    class_rows = []

    for cls in CLASS_ORDER:
        cls_df = work[work["route_class"] == cls].copy()
        if len(cls_df) == 0:
            continue

        n_paths = int(cls_df["path_id"].nunique())
        total_rows = len(cls_df)
        hotspot_rows = int((pd.to_numeric(cls_df["is_hotspot"], errors="coerce") == 1).sum())
        hotspot_row_share = hotspot_rows / max(total_rows, 1)

        hotspot_path_counts = (
            cls_df[pd.to_numeric(cls_df["is_hotspot"], errors="coerce") == 1]
            .groupby("node_id", as_index=False)
            .agg(
                n_rows=("path_id", "size"),
                n_paths=("path_id", "nunique"),
                mean_distance_to_seam=("distance_to_seam", "mean"),
                mean_relational_mismatch=("neighbor_direction_mismatch_deg", "mean"),
            )
        )
        hotspot_path_counts["route_class"] = cls

        n_hotspot_nodes_used = int(hotspot_path_counts["node_id"].nunique())
        hotspot_node_coverage = n_hotspot_nodes_used / max(len(hotspot_nodes), 1)
        hotspot_touch_paths = int(
            cls_df.loc[pd.to_numeric(cls_df["is_hotspot"], errors="coerce") == 1, "path_id"]
            .dropna()
            .nunique()
        )
        hotspot_path_touch_share = hotspot_touch_paths / max(n_paths, 1)

        per_path_hotspot_nodes = (
            cls_df.loc[pd.to_numeric(cls_df["is_hotspot"], errors="coerce") == 1]
            .groupby("path_id")["node_id"]
            .nunique()
        )
        mean_distinct_hotspots_per_path = float(per_path_hotspot_nodes.mean()) if len(per_path_hotspot_nodes) else 0.0

        # concentration across hotspot nodes
        if len(hotspot_path_counts):
            p = hotspot_path_counts["n_rows"] / hotspot_path_counts["n_rows"].sum()
            herfindahl = float((p ** 2).sum())
            entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
            top3_share = float(hotspot_path_counts.nlargest(3, "n_rows")["n_rows"].sum() / hotspot_path_counts["n_rows"].sum())
        else:
            herfindahl = float("nan")
            entropy = float("nan")
            top3_share = float("nan")

        class_rows.append(
            {
                "route_class": cls,
                "n_paths": n_paths,
                "n_rows": total_rows,
                "hotspot_row_share": hotspot_row_share,
                #"hotspot_path_share": hotspot_path_share,
                "n_hotspot_nodes_used": n_hotspot_nodes_used,
                "hotspot_node_coverage": hotspot_node_coverage,
                "hotspot_path_touch_share": hotspot_path_touch_share,
                "hotspot_traffic_herfindahl": herfindahl,
                "hotspot_traffic_entropy": entropy,
                "hotspot_top3_share": top3_share,
                "mean_distinct_hotspots_per_path": mean_distinct_hotspots_per_path,
                "mean_hotspot_distance_to_seam": safe_mean(hotspot_path_counts["mean_distance_to_seam"]) if len(hotspot_path_counts) else float("nan"),
                "mean_hotspot_relational_mismatch": safe_mean(hotspot_path_counts["mean_relational_mismatch"]) if len(hotspot_path_counts) else float("nan"),
            }
        )

        if len(hotspot_path_counts):
            sub = hotspot_path_counts.merge(
                nodes[["node_id", "r", "alpha", "mds1", "mds2", "distance_to_seam", "neighbor_direction_mismatch_deg"]].drop_duplicates("node_id"),
                on="node_id",
                how="left",
                suffixes=("", "_node"),
            )
            node_rows.append(sub)

    node_df = pd.concat(node_rows, ignore_index=True) if node_rows else pd.DataFrame()
    class_df = pd.DataFrame(class_rows)

    if len(class_df):
        class_df["seam_hotspot_bias"] = class_df["mean_hotspot_distance_to_seam"] <= seam_threshold

    return node_df, class_df


def build_summary(class_df: pd.DataFrame, hotspot_threshold: float) -> str:
    lines = [
        "=== OBS-024 Family Hotspot Occupancy Summary ===",
        "",
        f"hotspot_threshold_quantile = {hotspot_threshold:.6f}",
        "",
    ]
    for _, row in class_df.iterrows():
        lines.append(
            f"{row['route_class']}: "
            f"n_paths={int(row['n_paths'])}, "
            f"hotspot_row_share={float(row['hotspot_row_share']):.4f}, "
            f"hotspot_path_touch_share={float(row['hotspot_path_touch_share']):.4f}, "
            f"n_hotspot_nodes_used={int(row['n_hotspot_nodes_used'])}, "
            f"hotspot_node_coverage={float(row['hotspot_node_coverage']):.4f}, "
            f"hotspot_traffic_herfindahl={float(row['hotspot_traffic_herfindahl']):.4f}, "
            f"hotspot_traffic_entropy={float(row['hotspot_traffic_entropy']):.4f}, "
            f"hotspot_top3_share={float(row['hotspot_top3_share']):.4f}, "
            f"mean_hotspot_distance_to_seam={float(row['mean_hotspot_distance_to_seam']):.4f}, "
            f"mean_hotspot_relational_mismatch={float(row['mean_hotspot_relational_mismatch']):.4f}"
        )

    lines.extend(
        [
            "",
            "Interpretive guide",
            "- higher hotspot_row_share means more route-time spent on high relational-mismatch nodes",
            "- lower hotspot_node_coverage with higher top3_share means stronger concentration on a small seam-core set",
            "- higher hotspot_node_coverage suggests broader stressed-node usage",
            "- hotspot_path_touch_share = fraction of class paths that touch at least one hotspot",      
        ]
    )
    return "\n".join(lines)


def render_figure(nodes: pd.DataFrame, seam: pd.DataFrame, node_df: pd.DataFrame, class_df: pd.DataFrame, hotspot_threshold: float, outpath: Path) -> None:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_bar1 = fig.add_subplot(gs[0, 1])
    ax_bar2 = fig.add_subplot(gs[0, 2])
    ax_bar3 = fig.add_subplot(gs[1, 1])
    ax_bar4 = fig.add_subplot(gs[1, 2])

    # background nodes
    vals = pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce")
    sc = ax_main.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=vals,
        cmap="viridis",
        s=50,
        alpha=0.22,
        linewidths=0,
        zorder=1,
    )

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=6.0, alpha=0.65, zorder=2)
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.8, alpha=0.96, zorder=3)

    hotspot = nodes[nodes["is_hotspot"] == 1].copy()
    ax_main.scatter(
        hotspot["mds1"],
        hotspot["mds2"],
        s=130,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
    )

    colors = {
        "branch_exit": "#7FDBFF",
        "stable_seam_corridor": "#D4A72C",
        "reorganization_heavy": "#B23A48",
    }

    if len(node_df):
        for cls in CLASS_ORDER:
            sub = node_df[node_df["route_class"] == cls].copy()
            if len(sub) == 0:
                continue
            # label top hotspot nodes by route usage for each class
            sub = sub.sort_values("n_rows", ascending=False).head(5)
            ax_main.scatter(
                sub["mds1"],
                sub["mds2"],
                s=160,
                facecolors="none",
                edgecolors=colors[cls],
                linewidths=2.0,
                zorder=5,
            )

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.040, pad=0.02)
    cbar.set_label("relational mismatch (deg)")

    ax_main.set_title("OBS-024 follow-up — family occupancy of relational-mismatch hotspots", fontsize=15, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.text(
        0.02,
        0.98,
        f"black rings = hotspot nodes (q >= {hotspot_threshold:.2f})\ncyan/gold/crimson rings = top family-used hotspots",
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92),
    )

    x = np.arange(len(class_df))
    labels = class_df["route_class"].tolist()

    ax_bar1.bar(labels, class_df["hotspot_row_share"], alpha=0.9)
    ax_bar1.set_title("Hotspot row share")
    ax_bar1.set_ylabel("share")
    ax_bar1.grid(alpha=0.15, axis="y")
    ax_bar1.tick_params(axis="x", rotation=20)

    ax_bar2.bar(labels, class_df["hotspot_node_coverage"], alpha=0.9)
    ax_bar2.set_title("Hotspot node coverage")
    ax_bar2.set_ylabel("coverage")
    ax_bar2.grid(alpha=0.15, axis="y")
    ax_bar2.tick_params(axis="x", rotation=20)

    ax_bar3.bar(labels, class_df["hotspot_top3_share"], alpha=0.9)
    ax_bar3.set_title("Top-3 hotspot traffic share")
    ax_bar3.set_ylabel("share")
    ax_bar3.grid(alpha=0.15, axis="y")
    ax_bar3.tick_params(axis="x", rotation=20)

    ax_bar4.bar(labels, class_df["mean_hotspot_distance_to_seam"], alpha=0.9)
    ax_bar4.set_title("Mean hotspot distance to seam")
    ax_bar4.set_ylabel("distance")
    ax_bar4.grid(alpha=0.15, axis="y")
    ax_bar4.tick_params(axis="x", rotation=20)

    fig.suptitle("PAM Observatory — OBS-024 family hotspot occupancy", fontsize=18)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-024 family hotspot occupancy analysis.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        routes_csv=args.routes_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        hotspot_quantile=args.hotspot_quantile,
        top_label_k=Config.top_label_k,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, routes, seam = load_inputs(cfg)
    nodes, routes, threshold = prepare_tables(nodes, routes, cfg.hotspot_quantile)
    node_df, class_df = compute_occupancy(nodes, routes, cfg.seam_threshold)

    nodes_csv = outdir / "family_hotspot_occupancy_nodes.csv"
    summary_csv = outdir / "family_hotspot_occupancy_summary.csv"
    summary_txt = outdir / "obs024_family_hotspot_occupancy_summary.txt"
    fig_png = outdir / "obs024_family_hotspot_occupancy_figure.png"

    node_df.to_csv(nodes_csv, index=False)
    class_df.to_csv(summary_csv, index=False)
    summary_txt.write_text(build_summary(class_df, threshold), encoding="utf-8")
    render_figure(nodes, seam, node_df, class_df, threshold, fig_png)

    print(nodes_csv)
    print(summary_csv)
    print(summary_txt)
    print(fig_png)


if __name__ == "__main__":
    main()
