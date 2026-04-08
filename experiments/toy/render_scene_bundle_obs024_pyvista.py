#!/usr/bin/env python3
"""
render_scene_bundle_obs024_pyvista.py

OBS-024-upgraded PyVista render for the PAM Observatory.

Visual grammar
--------------
- signed phase = z-lifted substrate
- seam = black organizer
- family routes = gold / crimson
- branch exits = cyan
- relational obstruction hotspots = charcoal spheres with warm accent shells
- routing hubs = smaller black spheres

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
outputs/obs022_scene_bundle_obs024_pyvista/
  scene_bundle_obs024_pyvista.png
  scene_bundle_obs024_pyvista_summary.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


@dataclass(frozen=True)
class Config:
    bundle_dir: str = "outputs/obs022_scene_bundle"
    mismatch_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    outdir: str = "outputs/obs022_scene_bundle_obs024_pyvista"
    hotspot_quantile: float = 0.85
    max_rep_paths_per_family: int = 5
    z_scale: float = 2.2

    substrate_node_size: float = 8.0
    hotspot_core_size: float = 24.0
    hotspot_shell_size: float = 31.0
    hub_size: float = 15.0

    seam_radius: float = 0.050
    edge_radius: float = 0.0055
    route_radius_corridor: float = 0.020
    route_radius_reorg: float = 0.013
    branch_radius: float = 0.010

    off_screen: bool = True
    window_size: tuple[int, int] = (1800, 1100)


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


def _numericize_known_columns(df: pd.DataFrame, skip: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    skip = skip or set()
    for col in out.columns:
        if col in skip:
            continue
        try:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            # If everything became NaN but original data had non-nulls, keep original column.
            if out[col].isna().all() and df[col].notna().any():
                out[col] = df[col]
        except Exception:
            out[col] = df[col]
    return out


def load_bundle(bundle_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = read_bundle_csv(bundle_dir, "scene_nodes.csv")
    edges = read_bundle_csv(bundle_dir, "scene_edges.csv")
    seam = read_bundle_csv(bundle_dir, "scene_seam.csv")
    hubs = read_bundle_csv(bundle_dir, "scene_hubs.csv")
    routes = read_bundle_csv(bundle_dir, "scene_routes.csv")

    nodes = _numericize_known_columns(nodes, skip={"path_id", "path_family"})
    edges = _numericize_known_columns(edges, skip={"path_id", "path_family"})
    seam = _numericize_known_columns(seam, skip={"path_id", "path_family"})
    hubs = _numericize_known_columns(hubs, skip={"path_id", "path_family"})
    routes = _numericize_known_columns(routes, skip={"path_id", "path_family"})

    if "is_branch_away" not in routes.columns:
        routes["is_branch_away"] = 0
    if "is_representative" not in routes.columns:
        routes["is_representative"] = 0

    return nodes, edges, seam, hubs, routes


def merge_relational_mismatch(nodes: pd.DataFrame, mismatch_csv: str | Path, hotspot_quantile: float) -> pd.DataFrame:
    mm = pd.read_csv(mismatch_csv).copy()
    keep = [c for c in ["node_id", "neighbor_direction_mismatch_deg"] if c in mm.columns]
    if "node_id" not in keep:
        raise ValueError("Mismatch csv must contain node_id")

    mm = mm[keep].copy()
    mm["node_id"] = pd.to_numeric(mm["node_id"], errors="coerce")
    mm["neighbor_direction_mismatch_deg"] = pd.to_numeric(mm["neighbor_direction_mismatch_deg"], errors="coerce")

    out = nodes.merge(mm, on="node_id", how="left")
    out["neighbor_direction_mismatch_deg"] = pd.to_numeric(
        out["neighbor_direction_mismatch_deg"], errors="coerce"
    )
    threshold = float(out["neighbor_direction_mismatch_deg"].quantile(hotspot_quantile))
    out["relational_hotspot"] = (out["neighbor_direction_mismatch_deg"] >= threshold).astype(int)
    return out


def node_xyz(df: pd.DataFrame, z_scale: float) -> np.ndarray:
    x = pd.to_numeric(df["mds1"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds2"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df["signed_phase"], errors="coerce").to_numpy(dtype=float) * z_scale
    return np.c_[x, y, z]


def polyline_from_points(points: np.ndarray) -> pv.PolyData:
    poly = pv.PolyData()
    poly.points = points
    n = len(points)
    poly.lines = np.concatenate(([n], np.arange(n, dtype=np.int32)))
    return poly


def add_tube(plotter: pv.Plotter, points: np.ndarray, color, radius: float, opacity: float = 1.0) -> None:
    if len(points) < 2:
        return
    if not np.isfinite(points).all():
        return
    poly = polyline_from_points(points)
    tube = poly.tube(radius=radius)
    plotter.add_mesh(tube, color=color, opacity=opacity, smooth_shading=True)


def _coolwarm_color_from_phase(mid_phase: float) -> tuple[float, float, float]:
    import matplotlib.cm as cm

    val = max(0.0, min(1.0, 0.5 * (mid_phase + 1.0)))
    rgba = cm.get_cmap("coolwarm")(val)
    return tuple(float(x) for x in rgba[:3])


def add_edges(plotter: pv.Plotter, edges: pd.DataFrame, nodes: pd.DataFrame, z_scale: float, radius: float) -> None:
    node_lookup = nodes.set_index("node_id")

    for _, row in edges.iterrows():
        try:
            src_id = int(row["src_id"])
            dst_id = int(row["dst_id"])
            a = node_lookup.loc[src_id]
            b = node_lookup.loc[dst_id]
        except Exception:
            continue

        pts = np.array(
            [
                [float(a["mds1"]), float(a["mds2"]), float(a["signed_phase"]) * z_scale],
                [float(b["mds1"]), float(b["mds2"]), float(b["signed_phase"]) * z_scale],
            ],
            dtype=float,
        )

        mid_phase = (
            float(row["edge_phase_mid"])
            if "edge_phase_mid" in row and pd.notna(row["edge_phase_mid"])
            else 0.0
        )
        color = _coolwarm_color_from_phase(mid_phase)
        add_tube(plotter, pts, color=color, radius=radius, opacity=0.18)


def add_routes(
    plotter: pv.Plotter,
    routes: pd.DataFrame,
    z_scale: float,
    max_rep_paths_per_family: int,
    radius_corridor: float,
    radius_reorg: float,
    radius_branch: float,
) -> None:
    reps = routes[pd.to_numeric(routes.get("is_representative", 0), errors="coerce") == 1].copy()
    if len(reps):
        for fam in ["reorganization_heavy", "stable_seam_corridor"]:
            fam_df = reps[reps["path_family"] == fam].copy()
            keep_ids = list(fam_df["path_id"].drop_duplicates()[:max_rep_paths_per_family])
            fam_df = fam_df[fam_df["path_id"].isin(keep_ids)]

            for _, grp in fam_df.groupby("path_id", sort=False):
                grp = grp.sort_values("step")
                pts = np.c_[
                    pd.to_numeric(grp["mds1"], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(grp["mds2"], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(grp["signed_phase"], errors="coerce").to_numpy(dtype=float) * z_scale,
                ]
                add_tube(
                    plotter,
                    pts,
                    color=FAMILY_COLORS[fam],
                    radius=radius_corridor if fam == "stable_seam_corridor" else radius_reorg,
                    opacity=0.97 if fam == "stable_seam_corridor" else 0.68,
                )

    branch = routes[pd.to_numeric(routes.get("is_branch_away", 0), errors="coerce") == 1].copy()
    if len(branch):
        for _, grp in branch.groupby("path_id", sort=False):
            grp = grp.sort_values("step")
            pts = np.c_[
                pd.to_numeric(grp["mds1"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(grp["mds2"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(grp["signed_phase"], errors="coerce").to_numpy(dtype=float) * z_scale,
            ]
            add_tube(plotter, pts, color=BRANCH_COLOR, radius=radius_branch, opacity=0.76)


def add_response_filaments(
    plotter: pv.Plotter,
    nodes: pd.DataFrame,
    z_scale: float,
    length_scale: float = 0.22,
    radius: float = 0.004,
    max_filaments: int = 24,
) -> None:
    work = nodes.copy()

    # keep only nodes with usable response direction
    work["rsp_theta"] = pd.to_numeric(work.get("rsp_theta"), errors="coerce")
    work["response_strength"] = pd.to_numeric(work.get("response_strength"), errors="coerce")
    work["rsp_anisotropy"] = pd.to_numeric(work.get("rsp_anisotropy"), errors="coerce")
    work["distance_to_seam"] = pd.to_numeric(work.get("distance_to_seam"), errors="coerce")
    work["neighbor_direction_mismatch_deg"] = pd.to_numeric(
        work.get("neighbor_direction_mismatch_deg"), errors="coerce"
    )

    work = work[work["rsp_theta"].notna()].copy()
    if len(work) == 0:
        return

    # prioritize seam-near + anisotropic + response-active + hotspot-like
    score_parts = []

    if work["response_strength"].notna().any():
        score_parts.append(work["response_strength"].rank(pct=True))
    if work["rsp_anisotropy"].notna().any():
        score_parts.append(work["rsp_anisotropy"].rank(pct=True))
    if work["neighbor_direction_mismatch_deg"].notna().any():
        score_parts.append(work["neighbor_direction_mismatch_deg"].rank(pct=True))
    if work["distance_to_seam"].notna().any():
        score_parts.append((-work["distance_to_seam"]).rank(pct=True))

    if score_parts:
        work["filament_score"] = sum(score_parts) / len(score_parts)
    else:
        work["filament_score"] = 0.0

    work = work.sort_values("filament_score", ascending=False).head(max_filaments).copy()

    for _, row in work.iterrows():
        x = float(row["mds1"])
        y = float(row["mds2"])
        z = float(row["signed_phase"]) * z_scale
        theta = float(row["rsp_theta"])

        response_strength = float(row["response_strength"]) if pd.notna(row["response_strength"]) else 0.0
        filament_len = length_scale * (0.65 + 0.55 * max(response_strength, 0.0))

        dx = 0.5 * filament_len * np.cos(theta)
        dy = 0.5 * filament_len * np.sin(theta)

        p0 = np.array([x - dx, y - dy, z], dtype=float)
        p1 = np.array([x + dx, y + dy, z], dtype=float)

        pts = np.vstack([p0, p1])

        add_tube(
            plotter,
            pts,
            color="#4B4B5A",   # muted dark gray-violet
            radius=radius,
            opacity=0.55,
        )


def write_summary(
    outpath: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seam: pd.DataFrame,
    hubs: pd.DataFrame,
    routes: pd.DataFrame,
    hotspot_quantile: float,
) -> None:
    hotspot_mask = pd.to_numeric(nodes["relational_hotspot"], errors="coerce") == 1
    rel = pd.to_numeric(nodes["neighbor_direction_mismatch_deg"], errors="coerce")
    branch_paths = (
        int(routes[pd.to_numeric(routes["is_branch_away"], errors="coerce") == 1]["path_id"].nunique())
        if "is_branch_away" in routes.columns
        else 0
    )

    lines = [
        "=== OBS-024 PyVista Render Summary ===",
        "",
        f"n_nodes={len(nodes)}",
        f"n_edges={len(edges)}",
        f"n_seam_points={len(seam)}",
        f"n_hubs={len(hubs)}",
        f"n_branch_paths={branch_paths}",
        f"n_hotspots={int(hotspot_mask.sum())}",
        f"hotspot_quantile={hotspot_quantile:.4f}",
        f"mean_relational_mismatch={float(rel.mean()):.4f}",
        f"mean_hotspot_relational_mismatch={float(rel[hotspot_mask].mean()):.4f}",
        "",
        "Visual grammar",
        "- substrate lift = signed phase",
        "- seam = black tube",
        "- gold = stable seam corridor",
        "- crimson = reorganization heavy",
        "- cyan = branch exit",
        "- black spheres = routing hubs",
        "- charcoal spheres + warm shell = relational obstruction hotspots",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OBS-024 upgraded PyVista observatory scene.")
    parser.add_argument("--bundle-dir", default=Config.bundle_dir)
    parser.add_argument("--mismatch-csv", default=Config.mismatch_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    parser.add_argument("--max-rep-paths-per-family", type=int, default=Config.max_rep_paths_per_family)
    parser.add_argument("--z-scale", type=float, default=Config.z_scale)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive PyVista window instead of off-screen rendering.",
    )
    parser.add_argument("--window-width", type=int, default=1800)
    parser.add_argument("--window-height", type=int, default=1100)
    args = parser.parse_args()

    cfg = Config(
        bundle_dir=args.bundle_dir,
        mismatch_csv=args.mismatch_csv,
        outdir=args.outdir,
        hotspot_quantile=args.hotspot_quantile,
        max_rep_paths_per_family=args.max_rep_paths_per_family,
        z_scale=args.z_scale,
        window_size=(args.window_width, args.window_height),
        off_screen=not args.interactive,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, seam, hubs, routes = load_bundle(cfg.bundle_dir)
    nodes = merge_relational_mismatch(nodes, cfg.mismatch_csv, cfg.hotspot_quantile)

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=cfg.off_screen, window_size=cfg.window_size)
    plotter.set_background("white")

    # Substrate nodes: demoted phase scaffold
    pts = node_xyz(nodes, cfg.z_scale)
    pdata = pv.PolyData(pts)
    pdata["signed_phase"] = pd.to_numeric(nodes["signed_phase"], errors="coerce").to_numpy(dtype=float)
    plotter.add_mesh(
        pdata,
        scalars="signed_phase",
        cmap="coolwarm",
        point_size=cfg.substrate_node_size,
        render_points_as_spheres=True,
        opacity=0.28,
        clim=[-1, 1],
        show_scalar_bar=True,
        scalar_bar_args={"title": "signed phase"},
    )

    # Background web
    add_edges(plotter, edges, nodes, cfg.z_scale, cfg.edge_radius)

    # Response filaments
    add_response_filaments(
        plotter,
        nodes,
        cfg.z_scale,
        length_scale=0.30,
        radius=0.0038,
        max_filaments=24,
    )

    # Seam
    seam_pts = np.c_[
        pd.to_numeric(seam["mds1"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(seam["mds2"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(seam["signed_phase"], errors="coerce").fillna(0.0).to_numpy(dtype=float) * cfg.z_scale,
    ]
    seam_pts = seam_pts[np.isfinite(seam_pts).all(axis=1)]
    if len(seam_pts) >= 2:
        add_tube(plotter, seam_pts, color="black", radius=cfg.seam_radius, opacity=0.96)

    # Family and branch routes
    add_routes(
        plotter,
        routes,
        cfg.z_scale,
        cfg.max_rep_paths_per_family,
        cfg.route_radius_corridor,
        cfg.route_radius_reorg,
        cfg.branch_radius,
    )

    # Routing hubs: smaller black spheres
    hubs_draw = hubs.head(12).copy()
    if len(hubs_draw):
        hub_pts = np.c_[
            pd.to_numeric(hubs_draw["mds1"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(hubs_draw["mds2"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(hubs_draw["signed_phase"], errors="coerce").to_numpy(dtype=float) * cfg.z_scale,
        ]
        valid = np.isfinite(hub_pts).all(axis=1)
        hub_pts = hub_pts[valid]
        if len(hub_pts):
            hub_mesh = pv.PolyData(hub_pts)
            plotter.add_mesh(
                hub_mesh,
                color="black",
                point_size=cfg.hub_size,
                render_points_as_spheres=True,
                opacity=0.95,
            )

    # Hotspots: charcoal core + warm accent shell
    hotspots = nodes[pd.to_numeric(nodes["relational_hotspot"], errors="coerce") == 1].copy()
    if len(hotspots):
        hs_pts = np.c_[
            pd.to_numeric(hotspots["mds1"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(hotspots["mds2"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(hotspots["signed_phase"], errors="coerce").to_numpy(dtype=float) * cfg.z_scale,
        ]
        valid = np.isfinite(hs_pts).all(axis=1)
        hs_pts = hs_pts[valid]
        hotspots = hotspots.loc[valid].copy()

        if len(hs_pts):
            # warm shell
            hs_shell = pv.PolyData(hs_pts)
            plotter.add_mesh(
                hs_shell,
                color="#D8A84E",
                point_size=cfg.hotspot_shell_size,
                render_points_as_spheres=True,
                opacity=0.28,
            )

            # charcoal core
            hs_core = pv.PolyData(hs_pts)
            plotter.add_mesh(
                hs_core,
                color="#2F2F2F",
                point_size=cfg.hotspot_core_size,
                render_points_as_spheres=True,
                opacity=0.97,
            )

            # fewer top labels with z offsets to reduce overlap
            top = hotspots.sort_values("neighbor_direction_mismatch_deg", ascending=False).head(5).copy()
            label_pts = np.c_[
                pd.to_numeric(top["mds1"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(top["mds2"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(top["signed_phase"], errors="coerce").to_numpy(dtype=float) * cfg.z_scale,
            ]
            if len(label_pts):
                offsets = np.array(
                    [
                        [0.00, 0.00, 0.18],
                        [0.00, 0.00, 0.24],
                        [0.00, 0.00, 0.30],
                        [0.00, 0.00, 0.36],
                        [0.00, 0.00, 0.42],
                    ],
                    dtype=float,
                )[: len(label_pts)]
                label_pts = label_pts + offsets

                labels = [
                    str(int(x))
                    for x in pd.to_numeric(top["node_id"], errors="coerce").fillna(-1)
                ]
                plotter.add_point_labels(
                    label_pts,
                    labels,
                    font_size=15,
                    point_size=0,
                    shape_opacity=0.55,
                    always_visible=True,
                    text_color="black",
                )

    plotter.add_text(
        "PAM Observatory — OBS-024 lifted obstruction scene",
        position="upper_edge",
        font_size=18,
    )

    # Camera: opens seam-core cluster more than previous version
    plotter.camera_position = [
        (6.9, -7.4, 5.8),
        (-0.2, 0.05, 0.10),
        (0.0, 0.0, 1.0),
    ]

    png_path = outdir / "scene_bundle_obs024_pyvista.png"
    txt_path = outdir / "scene_bundle_obs024_pyvista_summary.txt"

    if cfg.off_screen:
        plotter.show(screenshot=str(png_path), auto_close=True)
    else:
        plotter.show(auto_close=True)
        try:
            plotter.screenshot(str(png_path))
        except Exception:
            pass

    write_summary(txt_path, nodes, edges, seam, hubs, routes, cfg.hotspot_quantile)

    print(png_path)
    print(txt_path)


if __name__ == "__main__":
    main()