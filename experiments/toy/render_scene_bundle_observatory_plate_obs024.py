#!/usr/bin/env python3
"""
render_scene_bundle_observatory_plate_obs024.py

OBS-024-upgraded 2D observatory plate.

Adds relational seam-obstruction hotspots to the canonical bundle-rendered plate.

Inputs
------
<bundle_dir>/
  scene_nodes.csv
  scene_edges.csv
  scene_seam.csv
  scene_hubs.csv
  scene_routes.csv
  scene_glyphs.csv   (optional)

Additional input
----------------
outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv

Outputs
-------
<outdir>/
  scene_bundle_observatory_plate_obs024.png
  scene_bundle_observatory_plate_obs024_summary.txt
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
    bundle_dir: str = "outputs/obs022_scene_bundle"
    mismatch_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    outdir: str = "outputs/obs022_scene_bundle_plate_obs024"
    top_hubs_to_draw: int = 12
    max_rep_paths_per_family: int = 5
    hotspot_quantile: float = 0.85
    top_hotspot_labels: int = 8


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",
    "reorganization_heavy": "#B23A48",
    "settled_distant": "#5C6B73",
    "off_seam_reorganizing": "#2A9D8F",
}
BRANCH_COLOR = "#7FDBFF"

FAMILY_ORDER = [
    "stable_seam_corridor",
    "reorganization_heavy",
    "settled_distant",
    "off_seam_reorganizing",
]


def read_bundle_csv(bundle_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(bundle_dir) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_bundle(bundle_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    nodes = read_bundle_csv(bundle_dir, "scene_nodes.csv")
    edges = read_bundle_csv(bundle_dir, "scene_edges.csv")
    seam = read_bundle_csv(bundle_dir, "scene_seam.csv")
    hubs = read_bundle_csv(bundle_dir, "scene_hubs.csv")
    routes = read_bundle_csv(bundle_dir, "scene_routes.csv")

    glyph_path = Path(bundle_dir) / "scene_glyphs.csv"
    glyphs = pd.read_csv(glyph_path) if glyph_path.exists() else None

    if "is_branch_away" not in routes.columns:
        routes["is_branch_away"] = 0

    return nodes, edges, seam, hubs, routes, glyphs


def load_relational_mismatch(nodes: pd.DataFrame, mismatch_csv: str | Path) -> pd.DataFrame:
    mm = pd.read_csv(mismatch_csv).copy()

    keep = [
        c for c in [
            "node_id",
            "neighbor_direction_mismatch_deg",
            "local_direction_mismatch_deg",
            "transport_align_mean_deg",
            "distance_to_seam",
        ]
        if c in mm.columns
    ]
    mm = mm[keep].copy()

    for c in keep:
        if c != "node_id":
            mm[c] = pd.to_numeric(mm[c], errors="coerce")
    if "node_id" in mm.columns:
        mm["node_id"] = pd.to_numeric(mm["node_id"], errors="coerce")

    out = nodes.merge(mm, on="node_id", how="left", suffixes=("", "_mm"))

    if "neighbor_direction_mismatch_deg" not in out.columns:
        out["neighbor_direction_mismatch_deg"] = np.nan

    thr = float(pd.to_numeric(out["neighbor_direction_mismatch_deg"], errors="coerce").quantile(0.85))
    out["relational_hotspot"] = (
        pd.to_numeric(out["neighbor_direction_mismatch_deg"], errors="coerce") >= thr
    ).astype(int)
    return out


def summarize_families(routes: pd.DataFrame) -> pd.DataFrame:
    fam = (
        routes[["path_id", "path_family"]]
        .drop_duplicates()
        .groupby("path_family", as_index=False)
        .agg(n_paths=("path_id", "count"))
    )
    total = max(int(fam["n_paths"].sum()), 1)
    fam["share_paths"] = fam["n_paths"] / total
    fam["order"] = fam["path_family"].map({k: i for i, k in enumerate(FAMILY_ORDER)}).fillna(999)
    return fam.sort_values("order").reset_index(drop=True)


def summarize_routes(routes: pd.DataFrame) -> pd.DataFrame:
    reps = routes[(routes["is_representative"] == 1) | (routes["is_branch_away"] == 1)].copy()
    if len(reps) == 0:
        return pd.DataFrame()

    out = (
        reps.groupby(["path_id", "path_family", "is_representative", "is_branch_away"], as_index=False)
        .agg(
            n_steps=("step", "max"),
            mean_lazarus=("lazarus_score", "mean"),
            mean_distance_to_seam=("distance_to_seam", "mean"),
            mean_response=("response_strength", "mean"),
        )
    )
    out["route_class"] = np.where(out["is_branch_away"] == 1, "branch_away", "family_representative")
    out["order"] = out["path_family"].map({k: i for i, k in enumerate(FAMILY_ORDER)}).fillna(999)
    return out.sort_values(["route_class", "order", "mean_lazarus", "n_steps"], ascending=[True, True, False, False]).reset_index(drop=True)


def write_summary_text(
    outpath: Path,
    fam_summary: pd.DataFrame,
    route_summary: pd.DataFrame,
    hubs: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seam: pd.DataFrame,
) -> None:
    hotspot_mask = pd.to_numeric(nodes["relational_hotspot"], errors="coerce") == 1
    lines = [
        "=== OBS-024 Upgraded Scene Bundle Observatory Plate Summary ===",
        "",
        "Bundle counts",
        f"n_nodes={len(nodes)}",
        f"n_edges={len(edges)}",
        f"n_seam_points={len(seam)}",
        f"n_hubs={len(hubs)}",
        f"n_hotspots={int(hotspot_mask.sum())}",
        "",
        "Relational obstruction",
        f"mean_neighbor_direction_mismatch_deg={float(pd.to_numeric(nodes['neighbor_direction_mismatch_deg'], errors='coerce').mean()):.4f}",
        f"mean_hotspot_neighbor_direction_mismatch_deg={float(pd.to_numeric(nodes.loc[hotspot_mask, 'neighbor_direction_mismatch_deg'], errors='coerce').mean()):.4f}",
        "",
        "Family shares",
    ]
    for _, row in fam_summary.iterrows():
        lines.append(
            f"  {row['path_family']}: n_paths={int(row['n_paths'])}, share={float(row['share_paths']):.4f}"
        )

    if len(route_summary):
        lines.extend(["", "Representative route summary"])
        for _, row in route_summary.iterrows():
            lines.append(
                f"  {row['route_class']} | {row['path_family']} | {row['path_id']}: "
                f"steps={int(row['n_steps'])}, "
                f"mean_lazarus={float(row['mean_lazarus']):.4f}, "
                f"mean_distance_to_seam={float(row['mean_distance_to_seam']):.4f}, "
                f"mean_response={float(row['mean_response']) if pd.notna(row['mean_response']) else float('nan'):.4f}"
            )

    outpath.write_text("\n".join(lines), encoding="utf-8")


def draw_main_panel(
    ax,
    fig,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seam: pd.DataFrame,
    hubs: pd.DataFrame,
    routes: pd.DataFrame,
    top_hubs_to_draw: int,
    max_rep_paths_per_family: int,
    top_hotspot_labels: int,
) -> None:
    # phase-colored substrate nodes
    sc = ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["signed_phase"], errors="coerce"),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=52,
        alpha=0.30,
        linewidths=0,
        zorder=1,
    )

    # phase web
    for _, row in edges.iterrows():
        arr = [
            row.get("src_mds1", np.nan),
            row.get("src_mds2", np.nan),
            row.get("dst_mds1", np.nan),
            row.get("dst_mds2", np.nan),
            row.get("edge_phase_mid", np.nan),
        ]
        if not np.isfinite(arr).all():
            continue
        ax.plot(
            [row["src_mds1"], row["dst_mds1"]],
            [row["src_mds2"], row["dst_mds2"]],
            color=plt.cm.coolwarm((row["edge_phase_mid"] + 1.0) / 2.0),
            alpha=0.32,
            linewidth=1.0,
            zorder=2,
        )

    # seam
    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.6, alpha=0.60, zorder=3)
        ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.7, alpha=0.95, zorder=4)

    # family representatives
    reps = routes[routes["is_representative"] == 1].copy()
    if len(reps):
        for fam in ["reorganization_heavy", "stable_seam_corridor"]:
            fam_df = reps[reps["path_family"] == fam].copy()
            keep_ids = list(fam_df["path_id"].drop_duplicates()[:max_rep_paths_per_family])
            fam_df = fam_df[fam_df["path_id"].isin(keep_ids)]

            for _, grp in fam_df.groupby("path_id", sort=False):
                grp = grp.sort_values("step")
                ax.plot(
                    grp["mds1"],
                    grp["mds2"],
                    color=FAMILY_COLORS[fam],
                    linewidth=3.0 if fam == "stable_seam_corridor" else 1.9,
                    alpha=0.95 if fam == "stable_seam_corridor" else 0.62,
                    zorder=5 if fam == "stable_seam_corridor" else 4.6,
                )

    # branch-away paths
    branch = routes[routes["is_branch_away"] == 1].copy()
    if len(branch):
        for _, grp in branch.groupby("path_id", sort=False):
            grp = grp.sort_values("step").copy()
            if len(grp) < 2:
                continue

            d2s = pd.to_numeric(grp["distance_to_seam"], errors="coerce").to_numpy(dtype=float)
            x = pd.to_numeric(grp["mds1"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp["mds2"], errors="coerce").to_numpy(dtype=float)

            if np.isfinite(d2s).any():
                dmin = np.nanmin(d2s)
                dmax = np.nanmax(d2s)
                denom = max(dmax - dmin, 1e-9)
                t = (d2s - dmin) / denom
            else:
                t = np.zeros(len(d2s), dtype=float)

            for i in range(len(x) - 1):
                c = (0.85 - 0.35 * t[i], 0.97 - 0.10 * t[i], 1.0)
                ax.plot(
                    [x[i], x[i + 1]],
                    [y[i], y[i + 1]],
                    color=c,
                    linewidth=1.9,
                    alpha=0.88,
                    zorder=4.85,
                )

    # obstruction hotspots: charcoal dashed halos
    hotspots = nodes[pd.to_numeric(nodes["relational_hotspot"], errors="coerce") == 1].copy()
    if len(hotspots):
        hs_rel = pd.to_numeric(hotspots["neighbor_direction_mismatch_deg"], errors="coerce")
        hs_min = float(hs_rel.min())
        hs_max = float(hs_rel.max())
        denom = max(hs_max - hs_min, 1e-9)
        hs_scaled = (hs_rel - hs_min) / denom
        hs_sizes = 180 + 280 * hs_scaled.fillna(0.0)

        ax.scatter(
            hotspots["mds1"],
            hotspots["mds2"],
            s=hs_sizes,
            facecolors="none",
            edgecolors="#333333",
            linewidths=1.6,
            linestyles="dashed",
            alpha=0.9,
            zorder=6,
        )

        top_hotspots = hotspots.sort_values("neighbor_direction_mismatch_deg", ascending=False).head(top_hotspot_labels)
        for _, row in top_hotspots.iterrows():
            ax.scatter(
                [row["mds1"]],
                [row["mds2"]],
                s=155,
                facecolors="none",
                edgecolors="#FFD166",
                linewidths=1.8,
                zorder=7,
            )
            ax.text(
                float(row["mds1"]) + 0.05,
                float(row["mds2"]) + 0.05,
                f"{int(row['node_id'])}",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
                zorder=8,
            )

    # hubs: solid black rings
    hubs_draw = hubs.head(top_hubs_to_draw).copy()
    if len(hubs_draw):
        max_occ = max(float(pd.to_numeric(hubs_draw["n_unique_paths"], errors="coerce").max()), 1.0)
        sizes = 120 + 240 * (pd.to_numeric(hubs_draw["n_unique_paths"], errors="coerce") / max_occ)
        ax.scatter(
            hubs_draw["mds1"],
            hubs_draw["mds2"],
            s=sizes,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            zorder=6.8,
        )
        ax.scatter(hubs_draw["mds1"], hubs_draw["mds2"], s=18, c="black", zorder=7.8)

    ax.text(
        0.02,
        0.97,
        "phase manifold\nsolid black circles = routing hubs\ndashed charcoal halos = obstruction hotspots",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )
    ax.text(
        0.02,
        0.08,
        "black seam = detected phase boundary\nyellow labels = top relational-mismatch nodes",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )

    ax.set_title("Parameter manifold", fontsize=15, pad=10)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.08)
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("signed phase")


def draw_family_panel(ax, fam_summary: pd.DataFrame) -> None:
    y = np.arange(len(fam_summary))
    vals = pd.to_numeric(fam_summary["share_paths"], errors="coerce").to_numpy(dtype=float)
    colors = [FAMILY_COLORS.get(f, "#777777") for f in fam_summary["path_family"]]

    ax.barh(y, vals, color=colors, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(fam_summary["path_family"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("share")
    ax.set_title("Trajectory families", fontsize=14, pad=10)
    ax.grid(alpha=0.12, axis="x")


def draw_obstruction_panel(ax, nodes: pd.DataFrame) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= 0.15
    rel = pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce")

    vals = [
        float(rel[seam_mask].mean()),
        float(rel[~seam_mask].mean()),
    ]
    errs = [
        float(rel[seam_mask].std(ddof=1) / max(np.sqrt(int(rel[seam_mask].notna().sum())), 1.0)) if int(rel[seam_mask].notna().sum()) > 1 else 0.0,
        float(rel[~seam_mask].std(ddof=1) / max(np.sqrt(int(rel[~seam_mask].notna().sum())), 1.0)) if int(rel[~seam_mask].notna().sum()) > 1 else 0.0,
    ]

    ax.bar(["seam-band", "off-seam"], vals, yerr=errs, alpha=0.92)
    ax.set_ylabel("mean relational mismatch (deg)")
    ax.set_title("OBS-024 obstruction", fontsize=14, pad=10)
    ax.grid(alpha=0.12, axis="y")


def draw_diag_panel(
    ax,
    fam_summary: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seam: pd.DataFrame,
    hubs: pd.DataFrame,
    glyphs: pd.DataFrame | None,
    routes: pd.DataFrame,
) -> None:
    ax.axis("off")

    dominant_family = fam_summary.sort_values("share_paths", ascending=False).iloc[0]["path_family"] if len(fam_summary) else "n/a"
    n_glyphs = len(glyphs) if glyphs is not None else 0
    n_branch = int(routes[routes["is_branch_away"] == 1]["path_id"].nunique()) if "is_branch_away" in routes.columns else 0
    n_hotspots = int((pd.to_numeric(nodes["relational_hotspot"], errors="coerce") == 1).sum())
    mean_rel = float(pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce").mean())

    text = (
        "OBS-024 diagnostics\n\n"
        f"nodes: {len(nodes)}\n"
        f"edges: {len(edges)}\n"
        f"seam points: {len(seam)}\n"
        f"hubs: {len(hubs)}\n"
        f"glyphs: {n_glyphs}\n"
        f"branch-away: {n_branch}\n"
        f"hotspots: {n_hotspots}\n"
        f"mean rel. mismatch: {mean_rel:.2f}\n\n"
        f"dominant: {dominant_family}\n\n"
        "gold    stable corridor\n"
        "crimson reorg-heavy\n"
        "cyan    branch exit\n"
        "black   seam / hubs\n"
        "charcoal obstruction"
    )

    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=9.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )


def draw_status_strip(ax, nodes: pd.DataFrame, edges: pd.DataFrame, seam: pd.DataFrame, fam_summary: pd.DataFrame) -> None:
    ax.axis("off")
    dominant = fam_summary.sort_values("share_paths", ascending=False).iloc[0]["path_family"] if len(fam_summary) else "n/a"
    mean_phase = float(pd.to_numeric(nodes["signed_phase"], errors="coerce").mean())
    seam_detected = "yes" if len(seam) > 0 else "no"

    status = (
        f"OBS-024 plate   •   nodes {len(nodes)}   •   edges {len(edges)}   •   "
        f"seam detected {seam_detected}   •   dominant family {dominant}   •   "
        f"mean signed phase {mean_phase:.3f}"
    )
    ax.text(0.01, 0.5, status, va="center", ha="left", fontsize=11.5, family="monospace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OBS-024-upgraded 2D observatory plate from scene bundle.")
    parser.add_argument("--bundle-dir", default=Config.bundle_dir)
    parser.add_argument("--mismatch-csv", default=Config.mismatch_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--top-hubs-to-draw", type=int, default=Config.top_hubs_to_draw)
    parser.add_argument("--max-rep-paths-per-family", type=int, default=Config.max_rep_paths_per_family)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    parser.add_argument("--top-hotspot-labels", type=int, default=Config.top_hotspot_labels)
    args = parser.parse_args()

    cfg = Config(
        bundle_dir=args.bundle_dir,
        mismatch_csv=args.mismatch_csv,
        outdir=args.outdir,
        top_hubs_to_draw=args.top_hubs_to_draw,
        max_rep_paths_per_family=args.max_rep_paths_per_family,
        hotspot_quantile=args.hotspot_quantile,
        top_hotspot_labels=args.top_hotspot_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, seam, hubs, routes, glyphs = load_bundle(cfg.bundle_dir)
    nodes = load_relational_mismatch(nodes, cfg.mismatch_csv)

    # use requested quantile
    thr = float(pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce").quantile(cfg.hotspot_quantile))
    nodes["relational_hotspot"] = (
        pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce") >= thr
    ).astype(int)

    fam_summary = summarize_families(routes)
    route_summary = summarize_routes(routes)

    fig = plt.figure(figsize=(16, 9.3), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.0, 1.0, 0.14],
        width_ratios=[4.9, 1.9],
        hspace=0.10,
        wspace=0.12,
    )

    ax_main = fig.add_subplot(gs[0:2, 0])
    side = gs[0:2, 1].subgridspec(3, 1, height_ratios=[0.58, 0.58, 1.15], hspace=0.12)
    ax_family = fig.add_subplot(side[0, 0])
    ax_obstruction = fig.add_subplot(side[1, 0])
    ax_diag = fig.add_subplot(side[2, 0])
    ax_status = fig.add_subplot(gs[2, :])

    draw_main_panel(
        ax_main,
        fig,
        nodes,
        edges,
        seam,
        hubs,
        routes,
        cfg.top_hubs_to_draw,
        cfg.max_rep_paths_per_family,
        cfg.top_hotspot_labels,
    )
    draw_family_panel(ax_family, fam_summary)
    draw_obstruction_panel(ax_obstruction, nodes)
    draw_diag_panel(ax_diag, fam_summary, nodes, edges, seam, hubs, glyphs, routes)
    draw_status_strip(ax_status, nodes, edges, seam, fam_summary)

    fig.suptitle("PAM Observatory — bundle-rendered OBS-024 plate", fontsize=20)

    png_path = outdir / "scene_bundle_observatory_plate_obs024.png"
    txt_path = outdir / "scene_bundle_observatory_plate_obs024_summary.txt"

    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    write_summary_text(txt_path, fam_summary, route_summary, hubs, nodes, edges, seam)

    print(png_path)
    print(txt_path)


if __name__ == "__main__":
    main()
