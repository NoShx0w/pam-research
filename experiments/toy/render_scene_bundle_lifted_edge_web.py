#!/usr/bin/env python3
"""
render_scene_bundle_lifted_edge_web.py

Render the signed-phase lifted edge web using only the canonical OBS-022 scene bundle.

Inputs
------
<bundle_dir>/
  scene_nodes.csv
  scene_edges.csv
  scene_seam.csv
  scene_hubs.csv
  scene_routes.csv

Output
------
<outdir>/scene_bundle_lifted_edge_web.png
<outdir>/scene_bundle_lifted_edge_web.html   (best effort)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "PyVista is required. Install with: .venv/bin/pip install pyvista"
    ) from exc


@dataclass(frozen=True)
class Config:
    bundle_dir: str = "outputs/obs022_scene_bundle"
    outdir: str = "outputs/obs022_scene_bundle_render"
    z_scale: float = 2.2
    edge_radius: float = 0.0045
    seam_radius: float = 0.018
    path_radius_corridor: float = 0.018
    path_radius_reorg: float = 0.012
    path_radius_branch: float = 0.010
    hub_scale: float = 0.050
    off_screen: bool = True


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",
    "reorganization_heavy": "#B23A48",
}
BACKGROUND_COLOR = "#F3F1ED"
SEAM_COLOR = "#111111"
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

    for df, cols in [
        (nodes, ["node_id", "mds1", "mds2", "signed_phase", "distance_to_seam", "lazarus_score"]),
        (edges, ["src_id", "dst_id", "src_mds1", "src_mds2", "dst_mds1", "dst_mds2",
                 "src_signed_phase", "dst_signed_phase", "edge_phase_mid"]),
        (seam, ["mds1", "mds2", "signed_phase", "distance_to_seam"]),
        (hubs, ["node_id", "mds1", "mds2", "signed_phase", "lazarus_score", "n_unique_paths", "path_occupancy"]),
        (routes, ["path_id", "step", "mds1", "mds2", "signed_phase", "distance_to_seam",
                  "path_family", "is_representative", "is_branch_away"]),
    ]:
        for col in cols:
            if col in df.columns:
                if col not in {"path_id", "path_family"}:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_branch_away" not in routes.columns:
        routes["is_branch_away"] = 0

    return nodes, edges, seam, hubs, routes


def xyz_from_xyz_phase(x: np.ndarray, y: np.ndarray, phase: np.ndarray, z_scale: float) -> np.ndarray:
    return np.column_stack([x, y, phase * z_scale])


def xyz_from_df(df: pd.DataFrame, z_scale: float) -> np.ndarray:
    x = pd.to_numeric(df["mds1"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds2"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df["signed_phase"], errors="coerce").to_numpy(dtype=float)
    return xyz_from_xyz_phase(x, y, z, z_scale)


def make_polyline(points: np.ndarray) -> pv.PolyData:
    poly = pv.PolyData(points)
    n = len(points)
    if n >= 2:
        poly.lines = np.hstack([[n], np.arange(n, dtype=np.int32)])
    return poly


def add_floor_shadow(plotter: pv.Plotter, nodes: pd.DataFrame) -> None:
    pts = np.column_stack([
        pd.to_numeric(nodes["mds1"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(nodes["mds2"], errors="coerce").to_numpy(dtype=float),
        np.full(len(nodes), -2.6, dtype=float),
    ])
    shadow = pv.PolyData(pts)
    plotter.add_mesh(
        shadow,
        color="#BEB8AF",
        render_points_as_spheres=True,
        point_size=3.5,
        opacity=0.08,
    )


def add_node_points(plotter: pv.Plotter, nodes: pd.DataFrame, z_scale: float) -> None:
    pts = xyz_from_df(nodes, z_scale)
    mesh = pv.PolyData(pts)
    mesh["signed_phase"] = pd.to_numeric(nodes["signed_phase"], errors="coerce").to_numpy(dtype=float)

    plotter.add_mesh(
        mesh,
        scalars="signed_phase",
        cmap="coolwarm",
        clim=[-1, 1],
        render_points_as_spheres=True,
        point_size=12,
        opacity=0.22,
        show_scalar_bar=True,
        scalar_bar_args={"title": "signed phase"},
    )


def add_edge_web(plotter: pv.Plotter, edges: pd.DataFrame, z_scale: float, edge_radius: float) -> None:
    vals = pd.to_numeric(edges["edge_phase_mid"], errors="coerce").to_numpy(dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    for _, row in edges.iterrows():
        arr = [
            row["src_mds1"], row["src_mds2"], row["src_signed_phase"],
            row["dst_mds1"], row["dst_mds2"], row["dst_signed_phase"],
            row["edge_phase_mid"],
        ]
        if not np.isfinite(arr).all():
            continue

        pts = np.array([
            [float(row["src_mds1"]), float(row["src_mds2"]), float(row["src_signed_phase"]) * z_scale],
            [float(row["dst_mds1"]), float(row["dst_mds2"]), float(row["dst_signed_phase"]) * z_scale],
        ])
        line = make_polyline(pts)
        tube = line.tube(radius=edge_radius)
        tube["edge_phase"] = np.full(tube.n_points, float(row["edge_phase_mid"]), dtype=float)

        plotter.add_mesh(
            tube,
            scalars="edge_phase",
            cmap="coolwarm",
            clim=[vmin, vmax],
            opacity=0.42,
            smooth_shading=True,
            show_scalar_bar=False,
        )


def add_seam(plotter: pv.Plotter, seam: pd.DataFrame, z_scale: float, seam_radius: float) -> None:
    seam = seam.copy().dropna(subset=["mds1", "mds2", "signed_phase"])
    if len(seam) < 2:
        return

    seam = seam.sort_values("mds1").reset_index(drop=True)
    pts = xyz_from_df(seam, z_scale)
    line = make_polyline(pts)
    tube = line.tube(radius=seam_radius)
    plotter.add_mesh(tube, color=SEAM_COLOR, smooth_shading=True, opacity=0.98)


def add_hubs(plotter: pv.Plotter, hubs: pd.DataFrame, z_scale: float, hub_scale: float) -> None:
    if len(hubs) == 0:
        return

    max_occ = max(float(pd.to_numeric(hubs["n_unique_paths"], errors="coerce").max()), 1.0)

    for _, row in hubs.iterrows():
        arr = [row["mds1"], row["mds2"], row["signed_phase"], row["n_unique_paths"]]
        if not np.isfinite(arr).all():
            continue

        x = float(row["mds1"])
        y = float(row["mds2"])
        z = float(row["signed_phase"]) * z_scale
        occ = float(row["n_unique_paths"]) / max_occ
        radius = 0.028 + hub_scale * occ

        sphere = pv.Sphere(radius=radius, center=(x, y, z))
        plotter.add_mesh(
            sphere,
            color="#111111",
            smooth_shading=True,
            opacity=0.90,
        )


def add_representative_routes(
    plotter: pv.Plotter,
    routes: pd.DataFrame,
    z_scale: float,
    path_radius_corridor: float,
    path_radius_reorg: float,
) -> None:
    reps = routes[routes["is_representative"] == 1].copy()
    if len(reps) == 0:
        return

    for fam, fam_df in reps.groupby("path_family", sort=False):
        if fam not in FAMILY_COLORS:
            continue
        color = FAMILY_COLORS[fam]
        radius = path_radius_corridor if fam == "stable_seam_corridor" else path_radius_reorg
        opacity = 0.96 if fam == "stable_seam_corridor" else 0.78

        for _, grp in fam_df.groupby("path_id", sort=False):
            grp = grp.sort_values("step").dropna(subset=["mds1", "mds2", "signed_phase"])
            if len(grp) < 2:
                continue

            pts = xyz_from_df(grp, z_scale)
            line = make_polyline(pts)
            tube = line.tube(radius=radius)
            plotter.add_mesh(
                tube,
                color=color,
                smooth_shading=True,
                opacity=opacity,
            )


def add_branch_away_routes(
    plotter: pv.Plotter,
    routes: pd.DataFrame,
    z_scale: float,
    path_radius_branch: float,
) -> None:
    branch = routes[routes["is_branch_away"] == 1].copy()
    if len(branch) == 0:
        return

    for _, grp in branch.groupby("path_id", sort=False):
        grp = grp.sort_values("step").dropna(subset=["mds1", "mds2", "signed_phase"])
        if len(grp) < 2:
            continue

        pts = xyz_from_df(grp, z_scale)
        line = make_polyline(pts)
        tube = line.tube(radius=path_radius_branch)
        plotter.add_mesh(
            tube,
            color=BRANCH_COLOR,
            smooth_shading=True,
            opacity=0.82,
        )


def set_camera(plotter: pv.Plotter) -> None:
    plotter.camera_position = [
        (7.6, -7.8, 4.2),
        (0.8, 0.15, 0.0),
        (0.0, 0.0, 1.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render lifted edge web from OBS-022 scene bundle.")
    parser.add_argument("--bundle-dir", default=Config.bundle_dir)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--z-scale", type=float, default=Config.z_scale)
    parser.add_argument("--edge-radius", type=float, default=Config.edge_radius)
    parser.add_argument("--seam-radius", type=float, default=Config.seam_radius)
    parser.add_argument("--path-radius-corridor", type=float, default=Config.path_radius_corridor)
    parser.add_argument("--path-radius-reorg", type=float, default=Config.path_radius_reorg)
    parser.add_argument("--path-radius-branch", type=float, default=Config.path_radius_branch)
    parser.add_argument("--hub-scale", type=float, default=Config.hub_scale)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        bundle_dir=args.bundle_dir,
        outdir=args.outdir,
        z_scale=args.z_scale,
        edge_radius=args.edge_radius,
        seam_radius=args.seam_radius,
        path_radius_corridor=args.path_radius_corridor,
        path_radius_reorg=args.path_radius_reorg,
        path_radius_branch=args.path_radius_branch,
        hub_scale=args.hub_scale,
        off_screen=not args.interactive,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, seam, hubs, routes = load_bundle(cfg.bundle_dir)

    plotter = pv.Plotter(off_screen=cfg.off_screen, window_size=(1800, 1300))
    plotter.set_background(BACKGROUND_COLOR)

    add_floor_shadow(plotter, nodes)
    add_node_points(plotter, nodes, cfg.z_scale)
    add_edge_web(plotter, edges, cfg.z_scale, cfg.edge_radius)
    add_seam(plotter, seam, cfg.z_scale, cfg.seam_radius)
    add_hubs(plotter, hubs, cfg.z_scale, cfg.hub_scale)
    add_representative_routes(
        plotter,
        routes,
        cfg.z_scale,
        cfg.path_radius_corridor,
        cfg.path_radius_reorg,
    )
    add_branch_away_routes(
        plotter,
        routes,
        cfg.z_scale,
        cfg.path_radius_branch,
    )

    plotter.add_title("Signed-phase lifted edge web from OBS-022 scene bundle", font_size=16)
    set_camera(plotter)

    png_path = outdir / "scene_bundle_lifted_edge_web.png"
    html_path = outdir / "scene_bundle_lifted_edge_web.html"

    if cfg.off_screen:
        plotter.show(screenshot=str(png_path))
    else:
        plotter.show()

    try:
        plotter.export_html(str(html_path))
    except Exception:
        pass

    print(png_path)
    print(html_path)


if __name__ == "__main__":
    main()