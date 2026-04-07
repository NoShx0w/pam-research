#!/usr/bin/env python3
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
        "PyVista is required for this script. Install with: pip install pyvista"
    ) from exc


@dataclass(frozen=True)
class Config:
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    seam_csv: str = "outputs/fim_phase/phase_boundary_mds_backprojected.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    outdir: str = "outputs/figure_obs022_lifted_3d"
    reps_per_family: int = 5
    top_hubs: int = 12
    z_scale: float = 3.0
    point_size: float = 16.0
    off_screen: bool = True


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",
    "reorganization_heavy": "#B23A48",
}

BACKGROUND_COLOR = "#F3F1ED"
MANIFOLD_POINT_COLOR = "#B8C0CC"
SEAM_COLOR = "#111111"
HUB_COLOR = "#111111"


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

    for col in ["mds1", "mds2", "signed_phase", "lazarus_score"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

    return nodes


def load_paths(path_nodes_csv: str, family_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    paths = pd.read_csv(path_nodes_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in paths.columns and c in nodes.columns]
    extra_cols = [c for c in ["lazarus_score", "distance_to_seam"] if c in nodes.columns]
    if join_cols and extra_cols:
        missing = [c for c in extra_cols if c not in paths.columns]
        if missing:
            paths = paths.merge(nodes[join_cols + missing], on=join_cols, how="left")

    for col in ["mds1", "mds2", "lazarus_score", "distance_to_seam"]:
        if col in paths.columns:
            paths[col] = pd.to_numeric(paths[col], errors="coerce")

    return paths


def load_seam(seam_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    seam = pd.read_csv(seam_csv).copy()
    if {"mds1", "mds2"}.issubset(seam.columns):
        return seam

    join_cols = [c for c in ["r", "alpha"] if c in seam.columns and c in nodes.columns]
    if join_cols:
        seam = seam.merge(nodes[join_cols + ["mds1", "mds2", "lazarus_score"]], on=join_cols, how="left")
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
        nodes[[c for c in ["node_id", "r", "alpha", "mds1", "mds2", "lazarus_score"] if c in nodes.columns]],
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
    z = pd.to_numeric(df["lazarus_score"], errors="coerce").to_numpy(dtype=float) * z_scale
    return np.column_stack([x, y, z])


def make_polyline(points: np.ndarray) -> pv.PolyData:
    n = len(points)
    poly = pv.PolyData(points)
    if n >= 2:
        cells = np.hstack([[n], np.arange(n, dtype=np.int32)])
        poly.lines = cells
    return poly


def add_manifold_points(plotter: pv.Plotter, nodes: pd.DataFrame, z_scale: float, point_size: float) -> None:
    pts = xyz_from_df(nodes, z_scale)
    mesh = pv.PolyData(pts)
    mesh["signed_phase"] = pd.to_numeric(nodes["signed_phase"], errors="coerce").to_numpy(dtype=float)

    plotter.add_mesh(
        mesh,
        scalars="signed_phase",
        cmap="coolwarm",
        clim=[-1, 1],
        render_points_as_spheres=True,
        point_size=point_size,
        opacity=0.80,
        show_scalar_bar=True,
        scalar_bar_args={"title": "signed phase"},
    )


def add_seam(plotter: pv.Plotter, seam: pd.DataFrame, z_scale: float) -> None:
    seam = seam.copy().dropna(subset=["mds1", "mds2"])
    if len(seam) < 2:
        return

    if "lazarus_score" not in seam.columns or seam["lazarus_score"].isna().all():
        seam["lazarus_score"] = 0.0

    seam = seam.sort_values("mds1").reset_index(drop=True)
    pts = xyz_from_df(seam.rename(columns={"lazarus_score": "lazarus_score"}), z_scale)
    line = make_polyline(pts)
    tube = line.tube(radius=0.035)
    plotter.add_mesh(tube, color=SEAM_COLOR, smooth_shading=True)


def add_hubs(plotter: pv.Plotter, hubs: pd.DataFrame, z_scale: float) -> None:
    if len(hubs) == 0:
        return

    max_occ = max(float(hubs["n_unique_paths"].max()), 1.0)
    for _, row in hubs.iterrows():
        x = float(row["mds1"])
        y = float(row["mds2"])
        z = float(row["lazarus_score"]) * z_scale
        occ = float(row["n_unique_paths"]) / max_occ
        radius = 0.045 + 0.08 * occ
        sphere = pv.Sphere(radius=radius, center=(x, y, z))
        plotter.add_mesh(sphere, color=HUB_COLOR, smooth_shading=True, opacity=0.95)


def add_family_tubes(plotter: pv.Plotter, reps: pd.DataFrame, z_scale: float) -> None:
    for fam, fam_df in reps.groupby("path_family", sort=False):
        color = FAMILY_COLORS.get(str(fam), "#666666")
        for _, grp in fam_df.groupby("path_id", sort=False):
            grp = grp.sort_values("step").dropna(subset=["mds1", "mds2", "lazarus_score"])
            if len(grp) < 2:
                continue
            pts = xyz_from_df(grp, z_scale)
            line = make_polyline(pts)
            tube = line.tube(radius=0.020 if fam == "stable_seam_corridor" else 0.014)
            plotter.add_mesh(
                tube,
                color=color,
                smooth_shading=True,
                opacity=0.95 if fam == "stable_seam_corridor" else 0.70,
            )


def set_camera(plotter: pv.Plotter) -> None:
    plotter.camera_position = [
        (7.5, -8.0, 5.6),
        (0.7, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lifted 3D PyVista scene for OBS-022.")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--outdir", default="outputs/figure_obs022_lifted_3d")
    parser.add_argument("--reps-per-family", type=int, default=5)
    parser.add_argument("--top-hubs", type=int, default=12)
    parser.add_argument("--z-scale", type=float, default=3.0)
    parser.add_argument("--point-size", type=float, default=16.0)
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

    add_manifold_points(plotter, nodes, cfg.z_scale, cfg.point_size)
    add_seam(plotter, seam, cfg.z_scale)
    add_hubs(plotter, hubs, cfg.z_scale)
    add_family_tubes(plotter, reps, cfg.z_scale)

    plotter.add_title("Lifted seam-corridor geometry on the PAM manifold", font_size=16)
    set_camera(plotter)

    png_path = outdir / "figure_obs022_lifted_3d.png"
    html_path = outdir / "figure_obs022_lifted_3d.html"

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
