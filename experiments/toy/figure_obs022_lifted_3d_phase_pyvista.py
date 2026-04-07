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
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    outdir: str = "outputs/figure_obs022_lifted_3d_phase"
    reps_per_family: int = 5
    top_hubs: int = 10
    z_scale: float = 2.2
    point_size: float = 15.0
    hub_scale: float = 0.055
    off_screen: bool = True


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",
    "reorganization_heavy": "#B23A48",
}

BACKGROUND_COLOR = "#F3F1ED"
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

    df = phase[keep_phase].merge(laz[keep_laz], on=join_cols, how="left")
    for col in ["mds1", "mds2", "signed_phase", "lazarus_score", "distance_to_seam"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_paths(path_nodes_csv: str, family_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    paths = pd.read_csv(path_nodes_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in paths.columns and c in nodes.columns]
    extra = [c for c in ["signed_phase", "lazarus_score", "distance_to_seam"] if c in nodes.columns and c not in paths.columns]
    if join_cols and extra:
        paths = paths.merge(nodes[join_cols + extra], on=join_cols, how="left")

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


def xyz_from_df(df: pd.DataFrame, z_col: str, z_scale: float) -> np.ndarray:
    x = pd.to_numeric(df["mds1"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds2"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float) * z_scale
    return np.column_stack([x, y, z])


def make_polyline(points: np.ndarray) -> pv.PolyData:
    poly = pv.PolyData(points)
    n = len(points)
    if n >= 2:
        poly.lines = np.hstack([[n], np.arange(n, dtype=np.int32)])
    return poly


def add_manifold_points(plotter: pv.Plotter, nodes: pd.DataFrame, z_scale: float, point_size: float) -> None:
    pts = xyz_from_df(nodes, "signed_phase", z_scale)
    mesh = pv.PolyData(pts)
    mesh["signed_phase"] = pd.to_numeric(nodes["signed_phase"], errors="coerce").to_numpy(dtype=float)

    plotter.add_mesh(
        mesh,
        scalars="signed_phase",
        cmap="coolwarm",
        clim=[-1, 1],
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.82,
        show_scalar_bar=True,
        scalar_bar_args={"title": "signed phase"},
    )


def add_seam(plotter: pv.Plotter, seam: pd.DataFrame, z_scale: float) -> None:
    seam = seam.copy().dropna(subset=["mds1", "mds2"])
    if len(seam) < 2:
        return
    if "signed_phase" not in seam.columns or seam["signed_phase"].isna().all():
        seam["signed_phase"] = 0.0

    seam = seam.sort_values("mds1").reset_index(drop=True)
    pts = xyz_from_df(seam, "signed_phase", z_scale)
    line = make_polyline(pts)
    tube = line.tube(radius=0.024)
    plotter.add_mesh(tube, color=SEAM_COLOR, smooth_shading=True)


def add_hubs(plotter: pv.Plotter, hubs: pd.DataFrame, z_scale: float, hub_scale: float) -> None:
    if len(hubs) == 0:
        return

    max_occ = max(float(hubs["n_unique_paths"].max()), 1.0)
    laz = pd.to_numeric(hubs["lazarus_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    for i, row in hubs.reset_index(drop=True).iterrows():
        x = float(row["mds1"])
        y = float(row["mds2"])
        z = float(row["signed_phase"]) * z_scale
        occ = float(row["n_unique_paths"]) / max_occ
        radius = 0.035 + hub_scale * occ

        sphere = pv.Sphere(radius=radius, center=(x, y, z))
        sphere["lazarus_score"] = np.full(sphere.n_points, laz[i], dtype=float)
        plotter.add_mesh(
            sphere,
            scalars="lazarus_score",
            cmap="magma",
            clim=[float(np.nanmin(laz)), float(np.nanmax(laz))] if np.isfinite(laz).any() else [0, 1],
            smooth_shading=True,
            opacity=0.95,
            show_scalar_bar=False,
        )


def add_family_tubes(plotter: pv.Plotter, reps: pd.DataFrame, z_scale: float) -> None:
    for fam, fam_df in reps.groupby("path_family", sort=False):
        color = FAMILY_COLORS.get(str(fam), "#666666")
        for _, grp in fam_df.groupby("path_id", sort=False):
            grp = grp.sort_values("step").dropna(subset=["mds1", "mds2", "signed_phase"])
            if len(grp) < 2:
                continue
            pts = xyz_from_df(grp, "signed_phase", z_scale)
            line = make_polyline(pts)
            tube = line.tube(radius=0.026 if fam == "stable_seam_corridor" else 0.020)
            plotter.add_mesh(
                tube,
                color=color,
                smooth_shading=True,
                opacity=0.98 if fam == "stable_seam_corridor" else 0.78,
            )


def add_floor_shadow(plotter: pv.Plotter, nodes: pd.DataFrame) -> None:
    pts = np.column_stack([
        pd.to_numeric(nodes["mds1"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(nodes["mds2"], errors="coerce").to_numpy(dtype=float),
        np.full(len(nodes), -2.8, dtype=float),
    ])
    shadow = pv.PolyData(pts)
    plotter.add_mesh(
        shadow,
        color="#BEB8AF",
        render_points_as_spheres=True,
        point_size=4.0,
        opacity=0.08,
    )


def set_camera(plotter: pv.Plotter) -> None:
    plotter.camera_position = [
        (5.8, -6.6, 3.4),
        (0.2, -0.1, -0.05),
        (0.0, 0.0, 1.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a signed-phase lifted 3D PyVista scene for OBS-022.")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--outdir", default="outputs/figure_obs022_lifted_3d_phase")
    parser.add_argument("--reps-per-family", type=int, default=5)
    parser.add_argument("--top-hubs", type=int, default=10)
    parser.add_argument("--z-scale", type=float, default=2.2)
    parser.add_argument("--point-size", type=float, default=15.0)
    parser.add_argument("--hub-scale", type=float, default=0.055)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        phase_csv=args.phase_csv,
        seam_csv=args.seam_csv,
        lazarus_csv=args.lazarus_csv,
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        outdir=args.outdir,
        reps_per_family=args.reps_per_family,
        top_hubs=args.top_hubs,
        z_scale=args.z_scale,
        point_size=args.point_size,
        hub_scale=args.hub_scale,
        off_screen=not args.interactive,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(cfg.phase_csv, cfg.lazarus_csv)
    paths = load_paths(cfg.path_nodes_csv, cfg.family_csv, nodes)
    seam = load_seam(cfg.seam_csv, nodes)
    hubs = pick_hubs(paths, nodes, cfg.top_hubs)
    reps = pick_representative_paths(paths, cfg.reps_per_family)

    plotter = pv.Plotter(off_screen=cfg.off_screen, window_size=(1800, 1300))
    plotter.set_background(BACKGROUND_COLOR)

    add_floor_shadow(plotter, nodes)
    add_manifold_points(plotter, nodes, cfg.z_scale, cfg.point_size)
    add_seam(plotter, seam, cfg.z_scale)
    add_hubs(plotter, hubs, cfg.z_scale, cfg.hub_scale)
    add_family_tubes(plotter, reps, cfg.z_scale)

    plotter.add_title("Signed-phase lifted seam-corridor geometry on the PAM manifold", font_size=16)
    set_camera(plotter)

    png_path = outdir / "figure_obs022_lifted_3d_phase.png"
    html_path = outdir / "figure_obs022_lifted_3d_phase.html"

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