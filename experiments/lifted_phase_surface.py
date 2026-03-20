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
DIST_CSV = ROOT / "fim_phase" / "phase_distance_to_seam.csv"
LAZ_CSV = ROOT / "fim_phase" / "lazarus_surface.csv"
TRAJ_DIRS = [
    ROOT / "fim_geodesics",
    ROOT / "fim_phase",
]

OUTFILE_SURFACE = OUTDIR / "lifted_phase_surface.png"
OUTFILE_PATHS = OUTDIR / "lifted_phase_surface_paths.png"


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


def assign_sign(points: np.ndarray, seam: np.ndarray) -> np.ndarray:
    if len(seam) < 2:
        return np.ones(len(points), dtype=float)

    signs = np.ones(len(points), dtype=float)

    for i, p in enumerate(points):
        best_dist = np.inf
        best_sign = 1.0

        for j in range(len(seam) - 1):
            a = seam[j]
            b = seam[j + 1]

            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 <= 0:
                continue

            ap = p - a
            t = np.clip(np.dot(ap, ab) / ab2, 0.0, 1.0)
            q = a + t * ab
            dist = np.linalg.norm(p - q)

            if dist < best_dist:
                best_dist = dist
                offset = p - q
                cross = ab[0] * offset[1] - ab[1] * offset[0]
                best_sign = 1.0 if cross >= 0 else -1.0

        signs[i] = best_sign

    return signs


def build_signed_phase_field(
    mds_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    seam_df: pd.DataFrame,
):
    xcol = infer_column(mds_df, ["mds1", "MDS1", "x", "mds_x"])
    ycol = infer_column(mds_df, ["mds2", "MDS2", "y", "mds_y"])

    if "α" in dist_df.columns:
        dist_df = dist_df.rename(columns={"α": "alpha"})
    if "α" in mds_df.columns:
        mds_df = mds_df.rename(columns={"α": "alpha"})

    dcol = infer_column(dist_df, ["distance_to_seam", "fisher_distance_to_seam", "dist_to_seam"])

    merged = mds_df.merge(dist_df, on=["r", "alpha"], how="inner").copy()

    points = merged[[xcol, ycol]].to_numpy(dtype=float)
    seam_pts = seam_df[[xcol, ycol]].dropna().to_numpy(dtype=float)

    signs = assign_sign(points, seam_pts)
    phase_unsigned = normalize_01(merged[dcol].to_numpy(dtype=float))
    phase_signed = signs * phase_unsigned

    merged["signed_phase"] = phase_signed
    merged["_x"] = merged[xcol].astype(float)
    merged["_y"] = merged[ycol].astype(float)

    return merged, points, seam_pts


def build_interpolator(points: np.ndarray, values: np.ndarray, seam_pts: np.ndarray | None = None, seam_value: float = 0.0):
    if seam_pts is not None and len(seam_pts) > 0:
        fit_points = np.vstack([points, seam_pts])
        fit_values = np.concatenate([values, np.full(len(seam_pts), seam_value, dtype=float)])
    else:
        fit_points = points
        fit_values = values

    return RBFInterpolator(
        fit_points,
        fit_values,
        kernel="thin_plate_spline",
        smoothing=0.01,
    )


def evaluate_grid(points: np.ndarray, interpolator, grid_res: int = 220):
    x = points[:, 0]
    y = points[:, 1]
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = interpolator(XY).reshape(X.shape)
    return X, Y, Z


def load_lazarus_field(mds_df: pd.DataFrame):
    if not LAZ_CSV.exists():
        return None
    laz_df = pd.read_csv(LAZ_CSV)
    if "α" in laz_df.columns:
        laz_df = laz_df.rename(columns={"α": "alpha"})
    if "α" in mds_df.columns:
        mds_df = mds_df.rename(columns={"α": "alpha"})

    try:
        lcol = infer_column(laz_df, ["lazarus_score", "lazarus", "compression", "constraint_strength"])
    except KeyError:
        return None

    xcol = infer_column(mds_df, ["mds1", "MDS1", "x", "mds_x"])
    ycol = infer_column(mds_df, ["mds2", "MDS2", "y", "mds_y"])

    merged = mds_df.merge(laz_df, on=["r", "alpha"], how="inner").copy()
    merged["lazarus_score"] = merged[lcol].astype(float)
    merged["_x"] = merged[xcol].astype(float)
    merged["_y"] = merged[ycol].astype(float)
    return merged


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


def load_trajectory_segments(max_paths: int = 6):
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


def style_axis(ax):
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_zlabel("Signed phase")
    ax.view_init(elev=25, azim=-55)
    ax.grid(False)


def plot_surface(merged_phase: pd.DataFrame, seam_pts: np.ndarray, lazarus_df, with_paths: bool, outpath: Path):
    phase_points = merged_phase[["_x", "_y"]].to_numpy(dtype=float)

    phase_interp = build_interpolator(
        phase_points,
        merged_phase["signed_phase"].to_numpy(dtype=float),
        seam_pts=seam_pts,
        seam_value=0.0,
    )
    X, Y, Z = evaluate_grid(phase_points, phase_interp, grid_res=220)

    fig = plt.figure(figsize=(11, 8), dpi=180)
    ax = fig.add_subplot(projection="3d")

    if lazarus_df is not None and len(lazarus_df) > 5:
        laz_points = lazarus_df[["_x", "_y"]].to_numpy(dtype=float)
        laz_values = lazarus_df["lazarus_score"].to_numpy(dtype=float)
        laz_interp = build_interpolator(laz_points, laz_values)
        _, _, L = evaluate_grid(laz_points, laz_interp, grid_res=220)
        Lnorm = normalize_01(L)
        facecolors = plt.cm.magma(Lnorm)
        ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.96,
        )
        sm = ScalarMappable(norm=Normalize(vmin=float(np.nanmin(laz_values)), vmax=float(np.nanmax(laz_values))), cmap="magma")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08)
        cbar.set_label("Lazarus score")
    else:
        surf = ax.plot_surface(
            X, Y, Z,
            cmap="coolwarm",
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        cbar = fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.08)
        cbar.set_label("Signed phase")

    if len(seam_pts) > 1:
        ax.plot(
            seam_pts[:, 0],
            seam_pts[:, 1],
            np.zeros(len(seam_pts)),
            color="black",
            linewidth=2.2,
            zorder=10,
        )

    if with_paths:
        for seg in load_trajectory_segments(max_paths=6):
            zseg = phase_interp(seg)
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                zseg,
                color="white",
                linewidth=1.2,
                alpha=0.9,
                zorder=20,
            )

    ax.scatter(
        phase_points[:, 0],
        phase_points[:, 1],
        merged_phase["signed_phase"].to_numpy(dtype=float),
        c="white",
        s=6,
        alpha=0.18,
        depthshade=False,
    )

    title = "Lifted Phase Surface" if not with_paths else "Lifted Phase Surface with Paths"
    ax.set_title(title)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    mds_df = load_csv(MDS_CSV)
    seam_df = load_csv(SEAM_CSV)
    dist_df = load_csv(DIST_CSV)

    merged_phase, points, seam_pts = build_signed_phase_field(mds_df, dist_df, seam_df)
    lazarus_df = load_lazarus_field(mds_df)

    plot_surface(merged_phase, seam_pts, lazarus_df, with_paths=False, outpath=OUTFILE_SURFACE)
    plot_surface(merged_phase, seam_pts, lazarus_df, with_paths=True, outpath=OUTFILE_PATHS)

    print(f"Saved: {OUTFILE_SURFACE}")
    print(f"Saved: {OUTFILE_PATHS}")


if __name__ == "__main__":
    main()
