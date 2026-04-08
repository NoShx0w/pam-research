#!/usr/bin/env python3
"""
render_scene_bundle_lifted_web_obs024.py

OBS-024-upgraded 3D lifted web scene.

Scientific grammar
------------------
- z lift            = signed phase
- colored edge web  = phase substrate
- black seam        = detected phase boundary
- gold/crimson      = seam-contact family routes
- cyan              = branch-exit routes
- black hub rings   = routing hubs
- charcoal hotspot rings = relational obstruction hotspots

Inputs
------
outputs/obs022_scene_bundle/
  scene_nodes.csv
  scene_edges.csv
  scene_seam.csv
  scene_hubs.csv
  scene_routes.csv

outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv

Outputs
-------
outputs/obs022_scene_bundle_lifted_web_obs024/
  scene_bundle_lifted_web_obs024.png
  scene_bundle_lifted_web_obs024_summary.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass(frozen=True)
class Config:
    bundle_dir: str = "outputs/obs022_scene_bundle"
    mismatch_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    outdir: str = "outputs/obs022_scene_bundle_lifted_web_obs024"
    hotspot_quantile: float = 0.85
    top_hotspot_labels: int = 8
    top_hubs_to_draw: int = 12
    max_rep_paths_per_family: int = 5
    elev: float = 24.0
    azim: float = -62.0
    z_scale: float = 1.55


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",
    "reorganization_heavy": "#B23A48",
}
BRANCH_COLOR = "#7FDBFF"


def read_bundle_csv(bundle_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(bundle_dir) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_bundle(bundle_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = read_bundle_csv(bundle_dir, "scene_nodes.csv")
    edges = read_bundle_csv(bundle_dir, "scene_edges.csv")
    seam = read_bundle_csv(bundle_dir, "scene_seam.csv")
    hubs = read_bundle_csv(bundle_dir, "scene_hubs.csv")
    routes = read_bundle_csv(bundle_dir, "scene_routes.csv")

    if "is_branch_away" not in routes.columns:
        routes["is_branch_away"] = 0
    if "is_representative" not in routes.columns:
        routes["is_representative"] = 0

    return nodes, edges, seam, hubs, routes


def merge_relational_mismatch(nodes: pd.DataFrame, mismatch_csv: str | Path, hotspot_quantile: float) -> pd.DataFrame:
    mm = pd.read_csv(mismatch_csv).copy()
    keep = [c for c in [
        "node_id",
        "neighbor_direction_mismatch_deg",
        "local_direction_mismatch_deg",
        "transport_align_mean_deg",
    ] if c in mm.columns]
    mm = mm[keep].copy()

    if "node_id" in mm.columns:
        mm["node_id"] = pd.to_numeric(mm["node_id"], errors="coerce")
    for c in keep:
        if c != "node_id":
            mm[c] = pd.to_numeric(mm[c], errors="coerce")

    out = nodes.merge(mm, on="node_id", how="left")

    rel = pd.to_numeric(out["neighbor_direction_mismatch_deg"], errors="coerce")
    thr = float(rel.quantile(hotspot_quantile))
    out["relational_hotspot"] = (rel >= thr).astype(int)
    out["z_lift"] = pd.to_numeric(out["signed_phase"], errors="coerce") * 1.0
    return out


def write_summary(outpath: Path, nodes: pd.DataFrame, edges: pd.DataFrame, seam: pd.DataFrame, hubs: pd.DataFrame, routes: pd.DataFrame) -> None:
    hotspot_mask = pd.to_numeric(nodes["relational_hotspot"], errors="coerce") == 1
    lines = [
        "=== OBS-024 Lifted Web Summary ===",
        "",
        f"n_nodes={len(nodes)}",
        f"n_edges={len(edges)}",
        f"n_seam_points={len(seam)}",
        f"n_hubs={len(hubs)}",
        f"n_hotspots={int(hotspot_mask.sum())}",
        f"mean_signed_phase={float(pd.to_numeric(nodes['signed_phase'], errors='coerce').mean()):.4f}",
        f"mean_relational_mismatch={float(pd.to_numeric(nodes['neighbor_direction_mismatch_deg'], errors='coerce').mean()):.4f}",
        f"mean_hotspot_relational_mismatch={float(pd.to_numeric(nodes.loc[hotspot_mask, 'neighbor_direction_mismatch_deg'], errors='coerce').mean()):.4f}",
        "",
        "Visual grammar",
        "- z lift = signed phase",
        "- black seam = detected phase boundary",
        "- gold = stable seam corridor",
        "- crimson = reorganization heavy",
        "- cyan = branch exit",
        "- black rings = routing hubs",
        "- charcoal rings = relational obstruction hotspots",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def draw_edge_web(ax, edges: pd.DataFrame, z_scale: float) -> None:
    for _, row in edges.iterrows():
        vals = [
            row.get("src_mds1", np.nan),
            row.get("src_mds2", np.nan),
            row.get("dst_mds1", np.nan),
            row.get("dst_mds2", np.nan),
            row.get("src_signed_phase", np.nan),
            row.get("dst_signed_phase", np.nan),
            row.get("edge_phase_mid", np.nan),
        ]
        if not np.isfinite(vals).all():
            continue

        x = [row["src_mds1"], row["dst_mds1"]]
        y = [row["src_mds2"], row["dst_mds2"]]
        z = [z_scale * row["src_signed_phase"], z_scale * row["dst_signed_phase"]]
        c = plt.cm.coolwarm((float(row["edge_phase_mid"]) + 1.0) / 2.0)

        ax.plot(x, y, z, color=c, alpha=0.34, linewidth=1.0, zorder=1)


def draw_nodes(ax, nodes: pd.DataFrame, z_scale: float) -> None:
    ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        z_scale * pd.to_numeric(nodes["signed_phase"], errors="coerce"),
        c=pd.to_numeric(nodes["signed_phase"], errors="coerce"),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=28,
        alpha=0.38,
        depthshade=False,
        linewidths=0,
        zorder=2,
    )


def draw_seam(ax, seam: pd.DataFrame, z_scale: float) -> None:
    seam_draw = seam.dropna(subset=["mds1", "mds2", "signed_phase"]).sort_values("mds1")
    if len(seam_draw) == 0:
        return

    x = seam_draw["mds1"].to_numpy(dtype=float)
    y = seam_draw["mds2"].to_numpy(dtype=float)
    z = z_scale * pd.to_numeric(seam_draw["signed_phase"], errors="coerce").to_numpy(dtype=float)

    ax.plot(x, y, z, color="white", alpha=0.55, linewidth=5.5, zorder=3)
    ax.plot(x, y, z, color="black", alpha=0.97, linewidth=2.9, zorder=4)


def draw_family_routes(ax, routes: pd.DataFrame, z_scale: float, max_rep_paths_per_family: int) -> None:
    reps = routes[routes["is_representative"] == 1].copy()
    if len(reps):
        for fam in ["reorganization_heavy", "stable_seam_corridor"]:
            fam_df = reps[reps["path_family"] == fam].copy()
            keep_ids = list(fam_df["path_id"].drop_duplicates()[:max_rep_paths_per_family])
            fam_df = fam_df[fam_df["path_id"].isin(keep_ids)]

            for _, grp in fam_df.groupby("path_id", sort=False):
                grp = grp.sort_values("step")
                x = pd.to_numeric(grp["mds1"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(grp["mds2"], errors="coerce").to_numpy(dtype=float)
                z = z_scale * pd.to_numeric(grp["signed_phase"], errors="coerce").to_numpy(dtype=float)

                ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x, y, z = x[ok], y[ok], z[ok]
                if len(x) < 2:
                    continue

                ax.plot(
                    x, y, z,
                    color=FAMILY_COLORS[fam],
                    linewidth=3.0 if fam == "stable_seam_corridor" else 1.8,
                    alpha=0.95 if fam == "stable_seam_corridor" else 0.62,
                    zorder=5 if fam == "stable_seam_corridor" else 4.6,
                )

    branch = routes[routes["is_branch_away"] == 1].copy()
    if len(branch):
        for _, grp in branch.groupby("path_id", sort=False):
            grp = grp.sort_values("step")
            x = pd.to_numeric(grp["mds1"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp["mds2"], errors="coerce").to_numpy(dtype=float)
            z = z_scale * pd.to_numeric(grp["signed_phase"], errors="coerce").to_numpy(dtype=float)
            d2s = pd.to_numeric(grp["distance_to_seam"], errors="coerce").to_numpy(dtype=float)

            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[ok], y[ok], z[ok]
            d2s = d2s[ok]
            if len(x) < 2:
                continue

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
                    [z[i], z[i + 1]],
                    color=c,
                    linewidth=1.85,
                    alpha=0.88,
                    zorder=4.85,
                )


def draw_hubs(ax, hubs: pd.DataFrame, nodes: pd.DataFrame, z_scale: float, top_hubs_to_draw: int) -> None:
    hubs_draw = hubs.head(top_hubs_to_draw).copy()
    if len(hubs_draw) == 0:
        return

    max_occ = max(float(pd.to_numeric(hubs_draw["n_unique_paths"], errors="coerce").max()), 1.0)
    sizes = 120 + 240 * (pd.to_numeric(hubs_draw["n_unique_paths"], errors="coerce") / max_occ)

    z = z_scale * pd.to_numeric(hubs_draw["signed_phase"], errors="coerce")

    ax.scatter(
        hubs_draw["mds1"],
        hubs_draw["mds2"],
        z,
        s=sizes,
        facecolors="none",
        edgecolors="black",
        linewidths=2.0,
        depthshade=False,
        zorder=7,
    )
    ax.scatter(
        hubs_draw["mds1"],
        hubs_draw["mds2"],
        z,
        s=18,
        c="black",
        depthshade=False,
        zorder=8,
    )


def draw_hotspots(ax, nodes: pd.DataFrame, z_scale: float, top_hotspot_labels: int) -> None:
    hotspots = nodes[pd.to_numeric(nodes["relational_hotspot"], errors="coerce") == 1].copy()
    if len(hotspots) == 0:
        return

    hs_rel = pd.to_numeric(hotspots["neighbor_direction_mismatch_deg"], errors="coerce")
    hs_min = float(hs_rel.min())
    hs_max = float(hs_rel.max())
    denom = max(hs_max - hs_min, 1e-9)
    hs_scaled = (hs_rel - hs_min) / denom
    hs_sizes = 150 + 260 * hs_scaled.fillna(0.0)
    z = z_scale * pd.to_numeric(hotspots["signed_phase"], errors="coerce")

    # charcoal hotspot rings
    ax.scatter(
        hotspots["mds1"],
        hotspots["mds2"],
        z,
        s=hs_sizes,
        facecolors="none",
        edgecolors="#333333",
        linewidths=1.5,
        depthshade=False,
        zorder=6,
    )

    # top hotspot labels
    top_hotspots = hotspots.sort_values("neighbor_direction_mismatch_deg", ascending=False).head(top_hotspot_labels)
    ax.scatter(
        top_hotspots["mds1"],
        top_hotspots["mds2"],
        z_scale * pd.to_numeric(top_hotspots["signed_phase"], errors="coerce"),
        s=155,
        facecolors="none",
        edgecolors="#FFD166",
        linewidths=1.8,
        depthshade=False,
        zorder=8.5,
    )

    for _, row in top_hotspots.iterrows():
        x = float(row["mds1"])
        y = float(row["mds2"])
        zt = float(z_scale * row["signed_phase"])
        ax.text(x + 0.05, y + 0.03, zt + 0.03, f"{int(row['node_id'])}", fontsize=8.5, zorder=9)


def style_axes(ax) -> None:
    ax.set_xlabel("MDS 1", labelpad=10)
    ax.set_ylabel("MDS 2", labelpad=10)
    ax.set_zlabel("signed phase", labelpad=10)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # lighten axis lines a bit
    try:
        ax.xaxis.line.set_alpha(0.25)
        ax.yaxis.line.set_alpha(0.25)
        ax.zaxis.line.set_alpha(0.25)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OBS-024-upgraded 3D lifted web scene.")
    parser.add_argument("--bundle-dir", default=Config.bundle_dir)
    parser.add_argument("--mismatch-csv", default=Config.mismatch_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    parser.add_argument("--top-hotspot-labels", type=int, default=Config.top_hotspot_labels)
    parser.add_argument("--top-hubs-to-draw", type=int, default=Config.top_hubs_to_draw)
    parser.add_argument("--max-rep-paths-per-family", type=int, default=Config.max_rep_paths_per_family)
    parser.add_argument("--elev", type=float, default=Config.elev)
    parser.add_argument("--azim", type=float, default=Config.azim)
    parser.add_argument("--z-scale", type=float, default=Config.z_scale)
    args = parser.parse_args()

    cfg = Config(
        bundle_dir=args.bundle_dir,
        mismatch_csv=args.mismatch_csv,
        outdir=args.outdir,
        hotspot_quantile=args.hotspot_quantile,
        top_hotspot_labels=args.top_hotspot_labels,
        top_hubs_to_draw=args.top_hubs_to_draw,
        max_rep_paths_per_family=args.max_rep_paths_per_family,
        elev=args.elev,
        azim=args.azim,
        z_scale=args.z_scale,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, seam, hubs, routes = load_bundle(cfg.bundle_dir)
    nodes = merge_relational_mismatch(nodes, cfg.mismatch_csv, cfg.hotspot_quantile)

    # propagate signed phase onto routes if needed
    if "signed_phase" not in routes.columns and {"node_id"}.issubset(routes.columns):
        routes = routes.merge(
            nodes[["node_id", "signed_phase"]].drop_duplicates("node_id"),
            on="node_id",
            how="left",
        )

    fig = plt.figure(figsize=(15.5, 9.5))
    ax = fig.add_subplot(111, projection="3d")

    draw_edge_web(ax, edges, cfg.z_scale)
    draw_nodes(ax, nodes, cfg.z_scale)
    draw_seam(ax, seam, cfg.z_scale)
    draw_family_routes(ax, routes, cfg.z_scale, cfg.max_rep_paths_per_family)
    draw_hotspots(ax, nodes, cfg.z_scale, cfg.top_hotspot_labels)
    draw_hubs(ax, hubs, nodes, cfg.z_scale, cfg.top_hubs_to_draw)

    style_axes(ax)
    ax.view_init(elev=cfg.elev, azim=cfg.azim)
    ax.set_title("OBS-024 — lifted relational seam-obstruction web", pad=18, fontsize=18)

    # add a phase colorbar
    mappable = plt.cm.ScalarMappable(cmap="coolwarm")
    mappable.set_clim(-1, 1)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.028, pad=0.04)
    cbar.set_label("signed phase")

    png_path = outdir / "scene_bundle_lifted_web_obs024.png"
    txt_path = outdir / "scene_bundle_lifted_web_obs024_summary.txt"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_summary(txt_path, nodes, edges, seam, hubs, routes)

    print(png_path)
    print(txt_path)


if __name__ == "__main__":
    main()
