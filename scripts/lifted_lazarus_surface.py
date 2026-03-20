from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RBFInterpolator


ROOT = Path("outputs")
OUTDIR = ROOT / "fim_report"
OUTDIR.mkdir(parents=True, exist_ok=True)

MDS_CSV = ROOT / "fim_mds" / "mds_coords.csv"
SEAM_CSV = ROOT / "fim_phase" / "phase_boundary_mds_backprojected.csv"
LAZ_CSV = ROOT / "fim_lazarus" / "lazarus_scores.csv"
CURV_CSV = ROOT / "fim_curvature" / "curvature_surface.csv"
TRAJ_DIRS = [
    ROOT / "fim_geodesics",
    ROOT / "fim_phase",
]

OUTFILE_SURFACE = OUTDIR / "lifted_lazarus_surface.png"
OUTFILE_PATHS = OUTDIR / "lifted_lazarus_surface_paths.png"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def infer_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Could not find any of {candidates} in columns {list(df.columns)}")


def normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def rename_alpha(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"α": "alpha"}) if "α" in df.columns else df


def candidate_traj_files():
    files = []
    for directory in TRAJ_DIRS:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob("geodesic_*.csv")))
        files.extend(sorted(directory.glob("path_*.csv")))
    seen = set()
    unique = []
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def load_trajectory_segments(max_paths: int = 8):
    segments = []
    for path in candidate_traj_files():
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        try:
            xcol = infer_column(df, ["mds1", "MDS1", "x", "mds_x"])
            ycol = infer_column(df, ["mds2", "MDS2", "y", "mds_y"])
        except KeyError:
            continue

        pts = df[[xcol, ycol]].dropna().to_numpy(dtype=float)
        if len(pts) >= 2:
            segments.append(pts)
        if len(segments) >= max_paths:
            break
    return segments


def build_base_tables():
    mds_df = rename_alpha(load_csv(MDS_CSV))
    laz_df = rename_alpha(load_csv(LAZ_CSV))
    seam_df = load_csv(SEAM_CSV)

    xcol = infer_column(mds_df, ["mds1", "MDS1", "x", "mds_x"])
    ycol = infer_column(mds_df, ["mds2", "MDS2", "y", "mds_y"])
    lcol = infer_column(laz_df, ["lazarus_score", "lazarus", "compression", "constraint_strength"])

    merged = mds_df.merge(
        laz_df,
        on=["r", "alpha"],
        how="inner",
        suffixes=("_mds", "_laz"),
    ).copy()

    x_mds = f"{xcol}_mds" if f"{xcol}_mds" in merged.columns else xcol
    y_mds = f"{ycol}_mds" if f"{ycol}_mds" in merged.columns else ycol
    l_laz = f"{lcol}_laz" if f"{lcol}_laz" in merged.columns else lcol

    merged["_x"] = merged[x_mds].astype(float)
    merged["_y"] = merged[y_mds].astype(float)
    merged["_lazarus_raw"] = merged[l_laz].astype(float)

    seam_pts = seam_df[[xcol, ycol]].dropna().to_numpy(dtype=float)
    return merged, seam_pts


def build_interpolator(points: np.ndarray, values: np.ndarray, smoothing: float = 0.001):
    return RBFInterpolator(
        points,
        values,
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )


def evaluate_grid(points: np.ndarray, interpolator, grid_res: int = 240):
    x = points[:, 0]
    y = points[:, 1]
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = interpolator(XY).reshape(X.shape)
    return X, Y, Z


def maybe_load_curvature_overlay(mds_df: pd.DataFrame):
    if not CURV_CSV.exists():
        return None

    curv_df = rename_alpha(load_csv(CURV_CSV))
    try:
        ccol = infer_column(curv_df, ["curvature", "scalar_curvature", "abs_curvature", "log10_abs_curvature", "K"])
    except KeyError:
        return None

    xcol = infer_column(mds_df, ["mds1", "MDS1", "x", "mds_x"])
    ycol = infer_column(mds_df, ["mds2", "MDS2", "y", "mds_y"])

    merged = mds_df.merge(
        curv_df,
        on=["r", "alpha"],
        how="inner",
        suffixes=("_mds", "_curv"),
    ).copy()

    x_mds = f"{xcol}_mds" if f"{xcol}_mds" in merged.columns else xcol
    y_mds = f"{ycol}_mds" if f"{ycol}_mds" in merged.columns else ycol
    c_curv = f"{ccol}_curv" if f"{ccol}_curv" in merged.columns else ccol

    merged["_x"] = merged[x_mds].astype(float)
    merged["_y"] = merged[y_mds].astype(float)
    merged["_curv"] = np.abs(merged[c_curv].astype(float))
    return merged


def style_axis(ax):
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_zlabel("Lazarus score")
    ax.view_init(elev=28, azim=-58)
    ax.grid(False)


def plot_lazarus_surface(with_paths: bool, outpath: Path):
    merged, seam_pts = build_base_tables()
    points = merged[["_x", "_y"]].to_numpy(dtype=float)

    laz_raw = merged["_lazarus_raw"].to_numpy(dtype=float)
    laz_norm = normalize_01(laz_raw)
    merged["_lazarus"] = laz_norm

    laz_interp = build_interpolator(points, laz_norm, smoothing=0.001)
    X, Y, Z = evaluate_grid(points, laz_interp, grid_res=240)

    # Soft seam band for visual tension: distance to seam in embedding space
    seam_band = None
    if len(seam_pts) > 1:
        XY = np.column_stack([X.ravel(), Y.ravel()])
        dmin = np.full(len(XY), np.inf, dtype=float)
        for j in range(len(seam_pts) - 1):
            a = seam_pts[j]
            b = seam_pts[j + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 <= 0:
                continue
            AP = XY - a
            t = np.clip((AP @ ab) / ab2, 0.0, 1.0)
            Q = a + np.outer(t, ab)
            d = np.linalg.norm(XY - Q, axis=1)
            dmin = np.minimum(dmin, d)
        seam_band = 1.0 - normalize_01(dmin).reshape(X.shape)

    fig = plt.figure(figsize=(11, 8), dpi=180)
    ax = fig.add_subplot(projection="3d")

    face = plt.cm.magma(Z)
    if seam_band is not None:
        # brighten a narrow seam-adjacent band to make "pressure" visible
        boost = np.clip(seam_band ** 3, 0, 1)
        face[..., :3] = np.clip(face[..., :3] + boost[..., None] * 0.18, 0, 1)

    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=face,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.98,
    )
    sm = ScalarMappable(norm=Normalize(vmin=float(np.nanmin(laz_raw)), vmax=float(np.nanmax(laz_raw))), cmap="magma")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("Lazarus score")

    # seam projected into Lazarus space
    if len(seam_pts) > 1:
        seam_z = laz_interp(seam_pts)
        ax.plot(
            seam_pts[:, 0],
            seam_pts[:, 1],
            seam_z,
            color="white",
            linewidth=2.0,
            alpha=0.95,
            zorder=20,
        )

    # trajectories lifted onto Lazarus surface
    if with_paths:
        for seg in load_trajectory_segments(max_paths=8):
            zseg = laz_interp(seg)
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                zseg,
                color="cyan",
                linewidth=1.1,
                alpha=0.8,
                zorder=25,
            )

    # measured support points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        laz_norm,
        c="white",
        s=5,
        alpha=0.15,
        depthshade=False,
    )

    # optional curvature hotspots as sparse overlay
    try:
        mds_df = rename_alpha(load_csv(MDS_CSV))
        curv = maybe_load_curvature_overlay(mds_df)
        if curv is not None and len(curv) > 3:
            cvals = normalize_01(curv["_curv"].to_numpy(dtype=float))
            mask = cvals > 0.8
            if np.any(mask):
                cpts = curv.loc[mask, ["_x", "_y"]].to_numpy(dtype=float)
                cz = laz_interp(cpts)
                ax.scatter(
                    cpts[:, 0],
                    cpts[:, 1],
                    cz + 0.01,
                    c="yellow",
                    s=18,
                    alpha=0.75,
                    depthshade=False,
                    zorder=30,
                )
    except Exception:
        pass

    title = "Lifted Lazarus Surface" if not with_paths else "Lifted Lazarus Surface with Paths"
    ax.set_title(title)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_lazarus_surface(with_paths=False, outpath=OUTFILE_SURFACE)
    plot_lazarus_surface(with_paths=True, outpath=OUTFILE_PATHS)
    print(f"Saved: {OUTFILE_SURFACE}")
    print(f"Saved: {OUTFILE_PATHS}")


if __name__ == "__main__":
    main()
