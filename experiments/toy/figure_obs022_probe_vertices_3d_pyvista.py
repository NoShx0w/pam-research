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
        "PyVista is required for this script. Install with: .venv/bin/pip install pyvista"
    ) from exc


@dataclass(frozen=True)
class Config:
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    seam_csv: str = "outputs/fim_phase/phase_boundary_mds_backprojected.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    outdir: str = "outputs/figure_obs022_probe_vertices_3d"
    max_paths: int = 12
    z_scale: float = 2.2
    seam_threshold: float = 0.15
    edge_radius: float = 0.010
    vertex_radius: float = 0.028
    seam_halo_radius: float = 0.040
    transition_radius: float = 0.050
    point_size: float = 10.0
    off_screen: bool = True


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",
    "reorganization_heavy": "#B23A48",
    "settled_distant": "#5C6B73",
    "off_seam_reorganizing": "#2A9D8F",
}

BACKGROUND_COLOR = "#F3F1ED"
SEAM_COLOR = "#111111"
TRANSITION_COLOR = "#F28E2B"


def load_nodes(phase_csv: str, lazarus_csv: str) -> pd.DataFrame:
    phase = pd.read_csv(phase_csv).copy()
    laz = pd.read_csv(lazarus_csv).copy()

    keep_phase = [c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase"] if c in phase.columns]
    keep_laz = [c for c in ["node_id", "r", "alpha", "distance_to_seam", "lazarus_score"] if c in laz.columns]

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in phase.columns and c in laz.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    df = phase[keep_phase].merge(laz[keep_laz], on=join_cols, how="left")
    for col in ["mds1", "mds2", "signed_phase", "distance_to_seam", "lazarus_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_paths(path_nodes_csv: str, family_csv: str, nodes: pd.DataFrame, max_paths: int) -> pd.DataFrame:
    paths = pd.read_csv(path_nodes_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in paths.columns and c in nodes.columns]
    extras = [c for c in ["signed_phase", "distance_to_seam", "lazarus_score"] if c in nodes.columns and c not in paths.columns]
    if join_cols and extras:
        paths = paths.merge(nodes[join_cols + extras], on=join_cols, how="left")

    for col in ["mds1", "mds2", "signed_phase", "distance_to_seam", "lazarus_score"]:
        if col in paths.columns:
            paths[col] = pd.to_numeric(paths[col], errors="coerce")

    # choose a balanced subset
    chosen_ids: list[str] = []
    grouped = (
        paths.groupby(["path_id", "path_family"], as_index=False)
        .agg(
            n_steps=("step", "max"),
            mean_lazarus=("lazarus_score", "mean"),
        )
    )

    fam_order = [
        "stable_seam_corridor",
        "reorganization_heavy",
        "off_seam_reorganizing",
        "settled_distant",
    ]
    per_family = max(1, max_paths // max(1, len(fam_order)))
    for fam_name in fam_order:
        sub = grouped[grouped["path_family"] == fam_name].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["mean_lazarus", "n_steps"], ascending=[False, False]).head(per_family)
        chosen_ids.extend(sub["path_id"].tolist())

    chosen_ids = chosen_ids[:max_paths]
    return paths[paths["path_id"].isin(chosen_ids)].copy()


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


def add_background(plotter: pv.Plotter, nodes: pd.DataFrame, z_scale: float, point_size: float) -> None:
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
        opacity=0.35,
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
    pts = xyz_from_df(seam, z_scale)
    line = make_polyline(pts)
    tube = line.tube(radius=0.022)
    plotter.add_mesh(tube, color=SEAM_COLOR, smooth_shading=True, opacity=0.95)


def add_path_edges(plotter: pv.Plotter, grp: pd.DataFrame, family: str, z_scale: float, radius: float) -> None:
    pts = xyz_from_df(grp, z_scale)
    if len(pts) < 2:
        return
    line = make_polyline(pts)
    tube = line.tube(radius=radius)
    plotter.add_mesh(
        tube,
        color=FAMILY_COLORS.get(family, "#777777"),
        smooth_shading=True,
        opacity=0.78,
    )


def add_vertex_sphere(
    plotter: pv.Plotter,
    center: tuple[float, float, float],
    radius: float,
    value: float,
    clim: tuple[float, float],
    opacity: float = 0.98,
) -> None:
    sphere = pv.Sphere(radius=radius, center=center)
    sphere["lazarus_score"] = np.full(sphere.n_points, value, dtype=float)
    plotter.add_mesh(
        sphere,
        scalars="lazarus_score",
        cmap="plasma",
        clim=list(clim),
        smooth_shading=True,
        opacity=opacity,
        show_scalar_bar=False,
    )


def add_transition_marker(plotter: pv.Plotter, center: tuple[float, float, float], radius: float) -> None:
    cube = pv.Cube(center=center, x_length=radius, y_length=radius, z_length=radius)
    plotter.add_mesh(cube, color=TRANSITION_COLOR, smooth_shading=True, opacity=0.98)


def first_transition_index(grp: pd.DataFrame) -> int | None:
    if "signed_phase" not in grp.columns:
        return None
    prev = 0
    for i, (_, row) in enumerate(grp.iterrows()):
        val = float(row["signed_phase"])
        sign = -1 if val < 0 else (1 if val > 0 else 0)
        if sign == 0:
            continue
        if prev != 0 and sign != prev:
            return i
        prev = sign
    return None


def add_paths_and_vertices(
    plotter: pv.Plotter,
    paths: pd.DataFrame,
    z_scale: float,
    edge_radius: float,
    vertex_radius: float,
    seam_halo_radius: float,
    transition_radius: float,
    seam_threshold: float,
) -> None:
    laz_vals = pd.to_numeric(paths["lazarus_score"], errors="coerce")
    laz_min = float(np.nanmin(laz_vals)) if np.isfinite(laz_vals).any() else 0.0
    laz_max = float(np.nanmax(laz_vals)) if np.isfinite(laz_vals).any() else 1.0
    clim = (laz_min, laz_max)

    for path_id, grp in paths.groupby("path_id", sort=False):
        grp = grp.sort_values("step").dropna(subset=["mds1", "mds2", "signed_phase"]).reset_index(drop=True)
        if len(grp) == 0:
            continue

        family = str(grp["path_family"].iloc[0])
        add_path_edges(plotter, grp, family, z_scale, edge_radius)

        t_idx = first_transition_index(grp)

        for i, row in grp.iterrows():
            x = float(row["mds1"])
            y = float(row["mds2"])
            z = float(row["signed_phase"]) * z_scale
            laz = float(row["lazarus_score"]) if pd.notna(row["lazarus_score"]) else laz_min

            # seam-contact halo
            d = float(row["distance_to_seam"]) if "distance_to_seam" in row.index and pd.notna(row["distance_to_seam"]) else np.inf
            if d <= seam_threshold:
                outer = pv.Sphere(radius=seam_halo_radius, center=(x, y, z))
                plotter.add_mesh(outer, color="black", smooth_shading=True, opacity=0.95)

            # main vertex
            add_vertex_sphere(plotter, (x, y, z), vertex_radius, laz, clim)

            # transition marker
            if t_idx is not None and i == t_idx:
                add_transition_marker(plotter, (x, y, z), transition_radius)


def add_floor_shadow(plotter: pv.Plotter, nodes: pd.DataFrame) -> None:
    pts = np.column_stack([
        pd.to_numeric(nodes["mds1"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(nodes["mds2"], errors="coerce").to_numpy(dtype=float),
        np.full(len(nodes), -2.9, dtype=float),
    ])
    shadow = pv.PolyData(pts)
    plotter.add_mesh(
        shadow,
        color="#BEB8AF",
        render_points_as_spheres=True,
        point_size=3.0,
        opacity=0.08,
    )


def set_camera(plotter: pv.Plotter) -> None:
    plotter.camera_position = [
        (5.9, -6.7, 3.5),
        (0.35, -0.10, -0.05),
        (0.0, 0.0, 1.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 3D vertex-emphasized probe geometry scene for OBS-022.")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--outdir", default="outputs/figure_obs022_probe_vertices_3d")
    parser.add_argument("--max-paths", type=int, default=12)
    parser.add_argument("--z-scale", type=float, default=2.2)
    parser.add_argument("--seam-threshold", type=float, default=0.15)
    parser.add_argument("--edge-radius", type=float, default=0.010)
    parser.add_argument("--vertex-radius", type=float, default=0.028)
    parser.add_argument("--seam-halo-radius", type=float, default=0.040)
    parser.add_argument("--transition-radius", type=float, default=0.050)
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        phase_csv=args.phase_csv,
        seam_csv=args.seam_csv,
        lazarus_csv=args.lazarus_csv,
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        outdir=args.outdir,
        max_paths=args.max_paths,
        z_scale=args.z_scale,
        seam_threshold=args.seam_threshold,
        edge_radius=args.edge_radius,
        vertex_radius=args.vertex_radius,
        seam_halo_radius=args.seam_halo_radius,
        transition_radius=args.transition_radius,
        point_size=args.point_size,
        off_screen=not args.interactive,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(cfg.phase_csv, cfg.lazarus_csv)
    seam = load_seam(cfg.seam_csv, nodes)
    paths = load_paths(cfg.path_nodes_csv, cfg.family_csv, nodes, cfg.max_paths)

    plotter = pv.Plotter(off_screen=cfg.off_screen, window_size=(1800, 1300))
    plotter.set_background(BACKGROUND_COLOR)

    add_floor_shadow(plotter, nodes)
    add_background(plotter, nodes, cfg.z_scale, cfg.point_size)
    add_seam(plotter, seam, cfg.z_scale)
    add_paths_and_vertices(
        plotter,
        paths,
        cfg.z_scale,
        cfg.edge_radius,
        cfg.vertex_radius,
        cfg.seam_halo_radius,
        cfg.transition_radius,
        cfg.seam_threshold,
    )

    plotter.add_title("3D probe flow and transition geometry on the PAM manifold", font_size=16)
    set_camera(plotter)

    png_path = outdir / "figure_obs022_probe_vertices_3d.png"
    html_path = outdir / "figure_obs022_probe_vertices_3d.html"

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
