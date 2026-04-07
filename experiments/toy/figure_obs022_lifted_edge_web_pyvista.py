#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "PyVista is required for this script. Install with: .venv/bin/pip install pyvista"
    ) from exc


@dataclass(frozen=True)
class Config:
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    seam_csv: str = "outputs/fim_phase/phase_boundary_mds_backprojected.csv"
    edges_csv: str = "outputs/fim_distance/fisher_edges.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    outdir: str = "outputs/figure_obs022_lifted_edge_web"
    reps_per_family: int = 5
    top_hubs: int = 10
    z_scale: float = 2.2
    edge_radius: float = 0.0035
    seam_radius: float = 0.020
    path_radius_corridor: float = 0.016
    path_radius_reorg: float = 0.011
    hub_scale: float = 0.060
    off_screen: bool = True


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",
    "reorganization_heavy": "#B23A48",
}
BACKGROUND_COLOR = "#F3F1ED"
EDGE_COLOR = "#2F3437"
SEAM_COLOR = "#111111"


def load_nodes(phase_csv: str, lazarus_csv: str) -> pd.DataFrame:
    phase = pd.read_csv(phase_csv).copy()
    laz = pd.read_csv(lazarus_csv).copy()

    keep_phase = [
        c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase"]
        if c in phase.columns
    ]
    keep_laz = [
        c for c in ["node_id", "r", "alpha", "distance_to_seam", "lazarus_score"]
        if c in laz.columns
    ]

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in phase.columns and c in laz.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    nodes = phase[keep_phase].merge(laz[keep_laz], on=join_cols, how="left")

    for col in ["mds1", "mds2", "signed_phase", "lazarus_score", "distance_to_seam"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

    if "node_id" not in nodes.columns:
        nodes = nodes.reset_index(drop=True)
        nodes["node_id"] = nodes.index.astype(int)

    return nodes


def load_edges(edges_csv: str) -> pd.DataFrame:
    edges = pd.read_csv(edges_csv).copy()

    src_candidates = ["src", "u", "src_id"]
    dst_candidates = ["dst", "v", "dst_id"]

    src_col = next((c for c in src_candidates if c in edges.columns), None)
    dst_col = next((c for c in dst_candidates if c in edges.columns), None)

    if src_col is None or dst_col is None:
        raise ValueError(
            f"Could not find edge endpoint columns in {edges_csv}. "
            f"Expected one of {src_candidates} and one of {dst_candidates}."
        )

    edges = edges.rename(columns={src_col: "src_id", dst_col: "dst_id"})
    edges["src_id"] = pd.to_numeric(edges["src_id"], errors="coerce").astype("Int64")
    edges["dst_id"] = pd.to_numeric(edges["dst_id"], errors="coerce").astype("Int64")
    edges = edges.dropna(subset=["src_id", "dst_id"]).copy()
    edges["src_id"] = edges["src_id"].astype(int)
    edges["dst_id"] = edges["dst_id"].astype(int)
    return edges


def load_paths(path_nodes_csv: str, family_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    paths = pd.read_csv(path_nodes_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in paths.columns and c in nodes.columns]
    extra_cols = [c for c in ["signed_phase", "lazarus_score", "distance_to_seam"] if c in nodes.columns and c not in paths.columns]
    if join_cols and extra_cols:
        paths = paths.merge(nodes[join_cols + extra_cols], on=join_cols, how="left")

    for col in ["mds1", "mds2", "signed_phase", "lazarus_score", "distance_to_seam"]:
        if col in paths.columns:
            paths[col] = pd.to_numeric(paths[col], errors="coerce")

    return paths


def load_seam(seam_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    seam = pd.read_csv(seam_csv).copy()

    if {"mds1", "mds2"}.issubset(seam.columns):
        if "signed_phase" not in seam.columns:
            join_cols = [c for c in ["r", "alpha"] if c in seam.columns and c in nodes.columns]
            if join_cols:
                seam = seam.merge(nodes[join_cols + ["signed_phase"]], on=join_cols, how="left")
        return seam

    join_cols = [c for c in ["r", "alpha"] if c in seam.columns and c in nodes.columns]
    if join_cols:
        seam = seam.merge(nodes[join_cols + ["mds1", "mds2", "signed_phase"]], on=join_cols, how="left")
    return seam


def pick_hubs(paths: pd.DataFrame, nodes: pd.DataFrame, top_k: int) -> pd.DataFrame:
    traffic = (
        paths.groupby(["node_id", "r", "alpha"], as_index=False)
        .agg(n_unique_paths=("path_id", "nunique"))
        .sort_values("n_unique_paths", ascending=False)
        .reset_index(drop=True)
    )

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in traffic.columns and c in nodes.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    hubs = traffic.merge(
        nodes[[c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase", "lazarus_score"] if c in nodes.columns]],
        on=join_cols,
        how="left",
    )
    return hubs.head(top_k).copy()


def pick_representative_paths(paths: pd.DataFrame, reps_per_family: int) -> pd.DataFrame:
    grouped = (
        paths.groupby(["path_id", "path_family"], as_index=False)
        .agg(
            n_steps=("step", "max"),
            mean_lazarus=("lazarus_score", "mean"),
            mean_distance_to_seam=("distance_to_seam", "mean"),
        )
    )

    keep_ids: list[str] = []
    for fam in ["stable_seam_corridor", "reorganization_heavy"]:
        sub = grouped[grouped["path_family"] == fam].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(
            ["mean_lazarus", "n_steps", "mean_distance_to_seam"],
            ascending=[False, False, True],
        )
        keep_ids.extend(sub.head(reps_per_family)["path_id"].tolist())

    return paths[paths["path_id"].isin(keep_ids)].copy()


def xyz_from_df(df: pd.DataFrame, z_scale: float) -> np.ndarray:
    x = pd.to_numeric(df["mds1"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds2"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df["signed_phase"], errors="coerce").to_numpy(dtype=float) * z_scale
    return np.column_stack([x, y, z])


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
        opacity=0.24,
        show_scalar_bar=True,
        scalar_bar_args={"title": "signed phase"},
    )


def add_edge_web(plotter: pv.Plotter, edges: pd.DataFrame, nodes: pd.DataFrame, z_scale: float, edge_radius: float) -> None:
    lookup = nodes.set_index("node_id")[["mds1", "mds2", "signed_phase"]]

    vals = pd.to_numeric(nodes["signed_phase"], errors="coerce").to_numpy(dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    for _, row in edges.iterrows():
        s = int(row["src_id"])
        t = int(row["dst_id"])
        if s not in lookup.index or t not in lookup.index:
            continue

        a = lookup.loc[s]
        b = lookup.loc[t]

        arr = [a["mds1"], a["mds2"], a["signed_phase"], b["mds1"], b["mds2"], b["signed_phase"]]
        if not np.isfinite(arr).all():
            continue

        pts = np.array([
            [float(a["mds1"]), float(a["mds2"]), float(a["signed_phase"]) * z_scale],
            [float(b["mds1"]), float(b["mds2"]), float(b["signed_phase"]) * z_scale],
        ])

        line = make_polyline(pts)
        tube = line.tube(radius=edge_radius)

        midpoint_phase = 0.5 * (float(a["signed_phase"]) + float(b["signed_phase"]))
        tube["edge_phase"] = np.full(tube.n_points, midpoint_phase, dtype=float)

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
    seam = seam.copy().dropna(subset=["mds1", "mds2"])
    if len(seam) < 2:
        return

    if "signed_phase" not in seam.columns or seam["signed_phase"].isna().all():
        seam["signed_phase"] = 0.0

    seam = seam.sort_values("mds1").reset_index(drop=True)
    pts = xyz_from_df(seam, z_scale)
    line = make_polyline(pts)
    tube = line.tube(radius=seam_radius)
    plotter.add_mesh(tube, color=SEAM_COLOR, smooth_shading=True, opacity=0.98)


def add_hubs(plotter: pv.Plotter, hubs: pd.DataFrame, z_scale: float, hub_scale: float) -> None:
    if len(hubs) == 0:
        return

    max_occ = max(float(hubs["n_unique_paths"].max()), 1.0)

    for _, row in hubs.iterrows():
        x = float(row["mds1"])
        y = float(row["mds2"])
        z = float(row["signed_phase"]) * z_scale
        occ = float(row["n_unique_paths"]) / max_occ
        radius = 0.030 + hub_scale * occ
        sphere = pv.Sphere(radius=radius, center=(x, y, z))
        plotter.add_mesh(
            sphere,
            color="#111111",
            smooth_shading=True,
            opacity=0.95,
        )


def add_family_tubes(
    plotter: pv.Plotter,
    reps: pd.DataFrame,
    z_scale: float,
    path_radius_corridor: float,
    path_radius_reorg: float,
) -> None:
    for fam, fam_df in reps.groupby("path_family", sort=False):
        color = FAMILY_COLORS.get(str(fam), "#666666")
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


def set_camera(plotter: pv.Plotter) -> None:
    plotter.camera_position = [
        (7.6, -7.8, 4.2),
        (0.8, 0.15, 0.0),
        (0.0, 0.0, 1.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a signed-phase lifted 3D edge-web scene for OBS-022.")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--outdir", default="outputs/figure_obs022_lifted_edge_web")
    parser.add_argument("--reps-per-family", type=int, default=5)
    parser.add_argument("--top-hubs", type=int, default=10)
    parser.add_argument("--z-scale", type=float, default=2.2)
    parser.add_argument("--edge-radius", type=float, default=0.0035)
    parser.add_argument("--seam-radius", type=float, default=0.020)
    parser.add_argument("--path-radius-corridor", type=float, default=0.016)
    parser.add_argument("--path-radius-reorg", type=float, default=0.011)
    parser.add_argument("--hub-scale", type=float, default=0.060)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        phase_csv=args.phase_csv,
        seam_csv=args.seam_csv,
        edges_csv=args.edges_csv,
        lazarus_csv=args.lazarus_csv,
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        outdir=args.outdir,
        reps_per_family=args.reps_per_family,
        top_hubs=args.top_hubs,
        z_scale=args.z_scale,
        edge_radius=args.edge_radius,
        seam_radius=args.seam_radius,
        path_radius_corridor=args.path_radius_corridor,
        path_radius_reorg=args.path_radius_reorg,
        hub_scale=args.hub_scale,
        off_screen=not args.interactive,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(cfg.phase_csv, cfg.lazarus_csv)
    edges = load_edges(cfg.edges_csv)
    paths = load_paths(cfg.path_nodes_csv, cfg.family_csv, nodes)
    seam = load_seam(cfg.seam_csv, nodes)
    hubs = pick_hubs(paths, nodes, cfg.top_hubs)
    reps = pick_representative_paths(paths, cfg.reps_per_family)

    plotter = pv.Plotter(off_screen=cfg.off_screen, window_size=(1800, 1300))
    plotter.set_background(BACKGROUND_COLOR)

    add_floor_shadow(plotter, nodes)
    add_node_points(plotter, nodes, cfg.z_scale)
    add_edge_web(plotter, edges, nodes, cfg.z_scale, cfg.edge_radius)
    add_seam(plotter, seam, cfg.z_scale, cfg.seam_radius)
    add_hubs(plotter, hubs, cfg.z_scale, cfg.hub_scale)
    add_family_tubes(
        plotter,
        reps,
        cfg.z_scale,
        cfg.path_radius_corridor,
        cfg.path_radius_reorg,
    )

    plotter.add_title("Signed-phase lifted edge web on the PAM manifold", font_size=16)
    set_camera(plotter)

    png_path = outdir / "figure_obs022_lifted_edge_web.png"
    html_path = outdir / "figure_obs022_lifted_edge_web.html"

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
