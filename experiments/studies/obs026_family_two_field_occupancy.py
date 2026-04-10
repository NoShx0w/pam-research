#!/usr/bin/env python3
"""
OBS-026 — Family occupancy on the two-field seam structure.

Compare realized route classes against three hotspot regimes:

1. anisotropy-only hotspots
2. relational-only hotspots
3. shared hotspots

This bridges:
- OBS-024 family dynamics
- OBS-025 two-field seam structure

Inputs
------
outputs/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_nodes.csv
outputs/obs022_scene_bundle/scene_routes.csv

Outputs
-------
outputs/obs026_family_two_field_occupancy/
  family_two_field_node_usage.csv
  family_two_field_class_summary.csv
  obs026_family_two_field_occupancy_summary.txt
  obs026_family_two_field_occupancy_figure.png
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
    nodes_csv: str = "outputs/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_nodes.csv"
    routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    outdir: str = "outputs/obs026_family_two_field_occupancy"
    include_other: bool = False


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

HOTSPOT_ORDER = [
    "anisotropy_only",
    "relational_only",
    "shared",
    "non_hotspot",
]


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def herfindahl_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    return float((p ** 2).sum())


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
    out = routes.copy()
    out["route_class"] = np.select(
        [
            pd.to_numeric(out.get("is_branch_away", 0), errors="coerce").fillna(0).eq(1),
            (
                pd.to_numeric(out.get("is_representative", 0), errors="coerce").fillna(0).eq(1)
                & out.get("path_family", pd.Series(index=out.index, dtype=object)).eq("stable_seam_corridor")
            ),
            (
                pd.to_numeric(out.get("is_representative", 0), errors="coerce").fillna(0).eq(1)
                & out.get("path_family", pd.Series(index=out.index, dtype=object)).eq("reorganization_heavy")
            ),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )
    return out


def assign_hotspot_class(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()

    aniso = pd.to_numeric(out["anisotropy_hotspot"], errors="coerce").fillna(0).astype(int)
    rel = pd.to_numeric(out["relational_hotspot"], errors="coerce").fillna(0).astype(int)
    shared = pd.to_numeric(out["shared_hotspot"], errors="coerce").fillna(0).astype(int)

    hotspot_class = np.full(len(out), "non_hotspot", dtype=object)
    hotspot_class[(aniso == 1) & (shared == 0)] = "anisotropy_only"
    hotspot_class[(rel == 1) & (shared == 0)] = "relational_only"
    hotspot_class[shared == 1] = "shared"

    out["hotspot_class"] = hotspot_class
    return out


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    routes = pd.read_csv(cfg.routes_csv)

    for df in (nodes, routes):
        for col in df.columns:
            if col not in {"path_id", "path_family", "route_class", "hotspot_class"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    nodes = assign_hotspot_class(nodes)
    routes = classify_routes(routes)

    keep_node_cols = [
        c
        for c in [
            "node_id",
            "r",
            "alpha",
            "distance_to_seam",
            "sym_traceless_norm",
            "neighbor_direction_mismatch_mean",
            "anisotropy_hotspot",
            "relational_hotspot",
            "shared_hotspot",
            "hotspot_class",
        ]
        if c in nodes.columns
    ]

    routes = routes.merge(
        nodes[keep_node_cols].drop_duplicates(subset=["node_id"]),
        on="node_id",
        how="left",
        suffixes=("", "_node"),
    )

    for base_col in [
        "r",
        "alpha",
        "distance_to_seam",
        "sym_traceless_norm",
        "neighbor_direction_mismatch_mean",
        "anisotropy_hotspot",
        "relational_hotspot",
        "shared_hotspot",
        "hotspot_class",
    ]:
        node_col = f"{base_col}_node"
        if base_col not in routes.columns and node_col in routes.columns:
            routes[base_col] = routes[node_col]
        elif base_col in routes.columns and node_col in routes.columns:
            routes[base_col] = routes[base_col].where(routes[base_col].notna(), routes[node_col])

    drop_cols = [c for c in routes.columns if c.endswith("_node")]
    if drop_cols:
        routes = routes.drop(columns=drop_cols)

    return nodes, routes


def compute_node_usage(routes: pd.DataFrame, include_other: bool) -> pd.DataFrame:
    work = routes.copy()
    if not include_other:
        work = work[work["route_class"].isin(CLASS_ORDER)].copy()

    usage = (
        work.groupby(["route_class", "node_id", "hotspot_class"], as_index=False)
        .agg(
            n_rows=("path_id", "size"),
            n_paths=("path_id", "nunique"),
            mean_distance_to_seam=("distance_to_seam", "mean"),
            mean_anisotropy=("sym_traceless_norm", "mean"),
            mean_relational=("neighbor_direction_mismatch_mean", "mean"),
        )
    )
    return usage.sort_values(["route_class", "n_rows"], ascending=[True, False]).reset_index(drop=True)


def compute_class_summary(routes: pd.DataFrame, include_other: bool) -> pd.DataFrame:
    work = routes.copy()
    if not include_other:
        work = work[work["route_class"].isin(CLASS_ORDER)].copy()

    # path-level hotspot touch summary
    path_touch = (
        work.groupby(["route_class", "path_id"], as_index=False)
        .agg(
            touches_anisotropy_only=("hotspot_class", lambda s: int((s == "anisotropy_only").any())),
            touches_relational_only=("hotspot_class", lambda s: int((s == "relational_only").any())),
            touches_shared=("hotspot_class", lambda s: int((s == "shared").any())),
            touches_any_hotspot=("hotspot_class", lambda s: int((s != "non_hotspot").any())),
        )
    )

    rows = []
    for cls, grp in work.groupby("route_class", sort=False):
        total_rows = len(grp)
        total_paths = int(grp["path_id"].nunique())
        counts = grp["hotspot_class"].value_counts()

        row = {
            "route_class": cls,
            "n_rows": total_rows,
            "n_paths": total_paths,
            "row_share_anisotropy_only": counts.get("anisotropy_only", 0) / max(total_rows, 1),
            "row_share_relational_only": counts.get("relational_only", 0) / max(total_rows, 1),
            "row_share_shared": counts.get("shared", 0) / max(total_rows, 1),
            "row_share_non_hotspot": counts.get("non_hotspot", 0) / max(total_rows, 1),
            "mean_distance_to_seam": safe_mean(grp["distance_to_seam"]),
            "mean_anisotropy": safe_mean(grp["sym_traceless_norm"]),
            "mean_relational": safe_mean(grp["neighbor_direction_mismatch_mean"]),
        }

        touch = path_touch[path_touch["route_class"] == cls]
        row["path_touch_anisotropy_only"] = safe_mean(touch["touches_anisotropy_only"])
        row["path_touch_relational_only"] = safe_mean(touch["touches_relational_only"])
        row["path_touch_shared"] = safe_mean(touch["touches_shared"])
        row["path_touch_any_hotspot"] = safe_mean(touch["touches_any_hotspot"])

        hotspot_counts = np.array(
            [
                counts.get("anisotropy_only", 0),
                counts.get("relational_only", 0),
                counts.get("shared", 0),
            ],
            dtype=float,
        )
        row["hotspot_entropy"] = entropy_from_counts(hotspot_counts)
        row["hotspot_herfindahl"] = herfindahl_from_counts(hotspot_counts)

        rows.append(row)

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_summary(summary_df: pd.DataFrame) -> str:
    lines = [
        "=== OBS-026 Family Two-Field Occupancy Summary ===",
        "",
    ]

    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"{row['route_class']} (n_paths={int(row['n_paths'])}, n_rows={int(row['n_rows'])})",
                f"  row_share_anisotropy_only = {float(row['row_share_anisotropy_only']):.4f}",
                f"  row_share_relational_only = {float(row['row_share_relational_only']):.4f}",
                f"  row_share_shared          = {float(row['row_share_shared']):.4f}",
                f"  row_share_non_hotspot     = {float(row['row_share_non_hotspot']):.4f}",
                f"  path_touch_anisotropy_only= {float(row['path_touch_anisotropy_only']):.4f}",
                f"  path_touch_relational_only= {float(row['path_touch_relational_only']):.4f}",
                f"  path_touch_shared         = {float(row['path_touch_shared']):.4f}",
                f"  path_touch_any_hotspot    = {float(row['path_touch_any_hotspot']):.4f}",
                f"  mean_distance_to_seam     = {float(row['mean_distance_to_seam']):.4f}",
                f"  mean_anisotropy           = {float(row['mean_anisotropy']):.6f}",
                f"  mean_relational           = {float(row['mean_relational']):.6f}",
                f"  hotspot_entropy           = {float(row['hotspot_entropy']):.4f}",
                f"  hotspot_herfindahl        = {float(row['hotspot_herfindahl']):.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive guide",
            "- anisotropy-only occupancy indicates preference for response-side symmetry-breaking regions",
            "- relational-only occupancy indicates preference for transport-obstruction regions",
            "- shared occupancy indicates convergence on the seam-core where both fields intensify",
            "- entropy/herfindahl summarize whether hotspot usage is distributed or concentrated",
        ]
    )
    return "\n".join(lines)


def render_figure(summary_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)

    classes = summary_df["route_class"].tolist()
    x = np.arange(len(classes))
    width = 0.24

    # Row-share composition
    ax = axes[0, 0]
    ax.bar(x - width, summary_df["row_share_anisotropy_only"], width, label="anisotropy-only")
    ax.bar(x, summary_df["row_share_relational_only"], width, label="relational-only")
    ax.bar(x + width, summary_df["row_share_shared"], width, label="shared")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=12)
    ax.set_ylabel("row share")
    ax.set_title("Hotspot regime occupancy", fontsize=14, pad=8)
    ax.grid(alpha=0.15, axis="y")
    ax.legend()

    # Path-touch summary
    ax = axes[0, 1]
    ax.bar(x - width, summary_df["path_touch_anisotropy_only"], width, label="anisotropy-only")
    ax.bar(x, summary_df["path_touch_relational_only"], width, label="relational-only")
    ax.bar(x + width, summary_df["path_touch_shared"], width, label="shared")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=12)
    ax.set_ylabel("fraction of paths touching class")
    ax.set_title("Path-touch profile", fontsize=14, pad=8)
    ax.grid(alpha=0.15, axis="y")

    # Mean field values
    ax = axes[1, 0]
    ax.bar(x - width / 2, summary_df["mean_anisotropy"], width, label="mean anisotropy")
    ax.bar(x + width / 2, summary_df["mean_relational"], width, label="mean relational")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=12)
    ax.set_title("Mean field exposure", fontsize=14, pad=8)
    ax.grid(alpha=0.15, axis="y")
    ax.legend()

    # Concentration / spread
    ax = axes[1, 1]
    ax.bar(x - width / 2, summary_df["hotspot_entropy"], width, label="entropy")
    ax.bar(x + width / 2, summary_df["hotspot_herfindahl"], width, label="herfindahl")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=12)
    ax.set_title("Hotspot usage structure", fontsize=14, pad=8)
    ax.grid(alpha=0.15, axis="y")
    ax.legend()

    fig.suptitle("PAM Observatory — OBS-026 family occupancy on two-field seam structure", fontsize=18)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare family occupancy on the two-field seam structure.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--include-other", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        routes_csv=args.routes_csv,
        outdir=args.outdir,
        include_other=args.include_other,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, routes = load_inputs(cfg)
    node_usage = compute_node_usage(routes, cfg.include_other)
    summary_df = compute_class_summary(routes, cfg.include_other)

    node_csv = outdir / "family_two_field_node_usage.csv"
    class_csv = outdir / "family_two_field_class_summary.csv"
    txt_path = outdir / "obs026_family_two_field_occupancy_summary.txt"
    png_path = outdir / "obs026_family_two_field_occupancy_figure.png"

    node_usage.to_csv(node_csv, index=False)
    summary_df.to_csv(class_csv, index=False)
    txt_path.write_text(build_summary(summary_df), encoding="utf-8")
    render_figure(summary_df, png_path)

    print(node_csv)
    print(class_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
