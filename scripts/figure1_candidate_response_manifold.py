from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RBFInterpolator


ROOT = Path("outputs")
OUTDIR = ROOT / "fim_report"
OUTDIR.mkdir(parents=True, exist_ok=True)

MDS_CSV = ROOT / "fim_mds" / "mds_coords.csv"
DIST_CSV = ROOT / "fim_phase" / "phase_distance_to_seam.csv"
SEAM_CSV = ROOT / "fim_phase" / "phase_boundary_mds_backprojected.csv"
LAZ_CSV = ROOT / "fim_lazarus" / "lazarus_scores.csv"
TRAJ_DIRS = [
    ROOT / "fim_geodesics",
    ROOT / "fim_phase",
]

OUTFILE = OUTDIR / "figure1_candidate_response_manifold.png"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def rename_alpha(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"α": "alpha"}) if "α" in df.columns else df


def infer_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Could not find any of {candidates} in columns {list(df.columns)}")


def normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    if not finite.any():
        return out
    v = values[finite]
    vmin = np.nanmin(v)
    vmax = np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        out[finite] = 0.0
        return out
    out[finite] = (v - vmin) / (vmax - vmin)
    return out


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


def build_fields():
    mds = rename_alpha(load_csv(MDS_CSV))
    dist = rename_alpha(load_csv(DIST_CSV))
    laz = rename_alpha(load_csv(LAZ_CSV))
    seam_df = load_csv(SEAM_CSV)

    xcol = infer_column(mds, ["mds1", "MDS1", "x", "mds_x"])
    ycol = infer_column(mds, ["mds2", "MDS2", "y", "mds_y"])
    dcol = infer_column(dist, ["distance_to_seam", "fisher_distance_to_seam", "dist_to_seam"])
    lcol = infer_column(laz, ["lazarus_score", "lazarus", "compression", "constraint_strength"])

    mds_dist = mds.merge(dist, on=["r", "alpha"], how="inner", suffixes=("_mds", "_dist")).copy()
    seam_pts = seam_df[[xcol, ycol]].dropna().to_numpy(dtype=float)

    x_mds = f"{xcol}_mds" if f"{xcol}_mds" in mds_dist.columns else xcol
    y_mds = f"{ycol}_mds" if f"{ycol}_mds" in mds_dist.columns else ycol
    pts = mds_dist[[x_mds, y_mds]].to_numpy(dtype=float)

    signs = assign_sign(pts, seam_pts)
    phase_unsigned = normalize_01(pd.to_numeric(mds_dist[dcol], errors="coerce").to_numpy(dtype=float))
    mds_dist["signed_phase"] = signs * phase_unsigned

    merged = mds_dist.merge(laz, on=["r", "alpha"], how="inner", suffixes=("", "_laz")).copy()
    l_laz = f"{lcol}_laz" if f"{lcol}_laz" in merged.columns else lcol

    merged["_x"] = pd.to_numeric(merged[x_mds], errors="coerce").astype(float)
    merged["_y"] = pd.to_numeric(merged[y_mds], errors="coerce").astype(float)
    merged["_phase"] = pd.to_numeric(merged["signed_phase"], errors="coerce").astype(float)
    merged["_lazarus_raw"] = pd.to_numeric(merged[l_laz], errors="coerce").astype(float)
    merged["_lazarus"] = normalize_01(merged["_lazarus_raw"].to_numpy(dtype=float))

    valid = (
        np.isfinite(merged["_x"].to_numpy()) &
        np.isfinite(merged["_y"].to_numpy()) &
        np.isfinite(merged["_phase"].to_numpy()) &
        np.isfinite(merged["_lazarus"].to_numpy())
    )
    return merged.loc[valid].copy(), seam_pts


def build_interpolator(points: np.ndarray, values: np.ndarray, smoothing: float = 0.001):
    return RBFInterpolator(
        points,
        values,
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )


def evaluate_grid(points: np.ndarray, interpolator, grid_res: int = 220):
    x = points[:, 0]
    y = points[:, 1]
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = interpolator(XY).reshape(X.shape)
    return X, Y, Z, xi, yi


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


def load_best_trajectory():
    best = None
    best_score = -np.inf
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
        if len(pts) < 4:
            continue
        score = np.linalg.norm(pts[-1] - pts[0])
        if score > best_score:
            best_score = score
            best = pts
    return best


def seam_contact_index(path_xy: np.ndarray, seam_pts: np.ndarray) -> int:
    if len(seam_pts) < 2:
        return int(len(path_xy) // 2)
    dmin = np.full(len(path_xy), np.inf)
    for j in range(len(seam_pts) - 1):
        a = seam_pts[j]
        b = seam_pts[j + 1]
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 <= 0:
            continue
        AP = path_xy - a
        t = np.clip((AP @ ab) / ab2, 0.0, 1.0)
        Q = a + np.outer(t, ab)
        d = np.linalg.norm(path_xy - Q, axis=1)
        dmin = np.minimum(dmin, d)
    return int(np.argmin(dmin))


def compute_response_grid(phase_Z: np.ndarray, laz_Z: np.ndarray, xi: np.ndarray, yi: np.ndarray, eps: float = 1e-12):
    dphidy, dphidx = np.gradient(phase_Z, yi, xi)
    dLdy, dLdx = np.gradient(laz_Z, yi, xi)
    phase_mag = np.sqrt(dphidx**2 + dphidy**2)
    laz_mag = np.sqrt(dLdx**2 + dLdy**2)
    dot = dLdx * dphidx + dLdy * dphidy
    alignment = dot / (phase_mag * laz_mag + eps)
    return {
        "dphidx": dphidx,
        "dphidy": dphidy,
        "laz_mag": laz_mag,
        "alignment": alignment,
    }


def nearest_grid_indices(x: np.ndarray, y: np.ndarray, xi: np.ndarray, yi: np.ndarray):
    ix = np.searchsorted(xi, x)
    iy = np.searchsorted(yi, y)
    ix = np.clip(ix, 1, len(xi) - 1)
    iy = np.clip(iy, 1, len(yi) - 1)
    ix = np.where(np.abs(xi[ix] - x) < np.abs(xi[ix - 1] - x), ix, ix - 1)
    iy = np.where(np.abs(yi[iy] - y) < np.abs(yi[iy - 1] - y), iy, iy - 1)
    return ix, iy


def add_tensor_glyphs(ax, X, Y, Z, xi, yi, resp, seam_pts, n_samples: int = 30):
    # bias glyphs toward seam but include a few controls
    dists = []
    pts = np.column_stack([X.ravel(), Y.ravel()])
    for p in pts:
        d = np.inf
        for j in range(len(seam_pts) - 1):
            a, b = seam_pts[j], seam_pts[j + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 <= 0:
                continue
            t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
            q = a + t * ab
            d = min(d, np.linalg.norm(p - q))
        dists.append(d)
    dists = np.array(dists).reshape(X.shape)

    score = resp["laz_mag"] / (np.nanmax(resp["laz_mag"]) + 1e-12)
    seam_weight = 1.0 - normalize_01(dists)
    priority = np.nan_to_num(score) * np.nan_to_num(seam_weight)

    flat_idx = np.argsort(priority.ravel())[::-1]
    chosen = []
    min_sep = 12
    for idx in flat_idx:
        i, j = np.unravel_index(idx, X.shape)
        if not np.isfinite(resp["alignment"][i, j]):
            continue
        if any((abs(i - ci) < min_sep and abs(j - cj) < min_sep) for ci, cj in chosen):
            continue
        chosen.append((i, j))
        if len(chosen) >= n_samples:
            break

    segments = []
    colors = []
    for i, j in chosen:
        x0, y0, z0 = X[i, j], Y[i, j], Z[i, j] + 0.015
        vx = resp["dphidx"][i, j]
        vy = resp["dphidy"][i, j]
        norm = np.hypot(vx, vy)
        if norm <= 1e-12:
            continue
        vx /= norm
        vy /= norm
        mag = resp["laz_mag"][i, j]
        L = 0.10 + 0.25 * (mag / (np.nanmax(resp["laz_mag"]) + 1e-12))
        x1 = x0 - vx * L
        y1 = y0 - vy * L
        x2 = x0 + vx * L
        y2 = y0 + vy * L
        z1 = z0
        z2 = z0
        segments.append([(x1, y1, z1), (x2, y2, z2)])
        colors.append(resp["alignment"][i, j])

    if segments:
        lc = Line3DCollection(segments, cmap="coolwarm", linewidths=1.8)
        lc.set_array(np.array(colors))
        lc.set_clim(-1, 1)
        ax.add_collection3d(lc)


def main():
    merged, seam_pts = build_fields()
    points = merged[["_x", "_y"]].to_numpy(dtype=float)

    phase_interp = build_interpolator(points, merged["_phase"].to_numpy(dtype=float), smoothing=0.001)
    laz_interp = build_interpolator(points, merged["_lazarus"].to_numpy(dtype=float), smoothing=0.001)

    X, Y, phase_Z, xi, yi = evaluate_grid(points, phase_interp, grid_res=220)
    _, _, laz_Z, _, _ = evaluate_grid(points, laz_interp, grid_res=220)

    resp = compute_response_grid(phase_Z, laz_Z, xi, yi)

    fig = plt.figure(figsize=(12, 8), dpi=180)
    ax = fig.add_subplot(projection="3d")

    face = plt.cm.magma(np.clip(laz_Z, 0, 1))
    ax.plot_surface(
        X, Y, phase_Z,
        facecolors=face,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.97,
    )

    # seam ridge
    seam_z = phase_interp(seam_pts) + 0.01
    ax.plot(seam_pts[:, 0], seam_pts[:, 1], seam_z, color="white", linewidth=2.0, alpha=0.95, zorder=20)

    # trajectory + event markers
    path_xy = load_best_trajectory()
    if path_xy is not None:
        path_z = phase_interp(path_xy) + 0.015
        ax.plot(path_xy[:, 0], path_xy[:, 1], path_z, color="cyan", linewidth=2.0, alpha=0.95, zorder=25)

        laz_along = laz_interp(path_xy)
        peak_idx = int(np.nanargmax(laz_along))
        contact_idx = seam_contact_index(path_xy, seam_pts)
        flip_idx = min(len(path_xy) - 1, contact_idx + max(1, (len(path_xy) - contact_idx) // 4))

        markers = [
            (peak_idx, "o", "gold", "L peak"),
            (contact_idx, "o", "white", "seam"),
            (flip_idx, "*", "deepskyblue", "flip"),
        ]
        for idx, marker, color, label in markers:
            x, y, z = path_xy[idx, 0], path_xy[idx, 1], path_z[idx]
            ax.scatter([x], [y], [z], c=color, s=55 if marker != "*" else 75, marker=marker, depthshade=False, zorder=30)
            ax.text(x, y, z + 0.04, label, color="white", fontsize=8)

    # tensor glyphs
    add_tensor_glyphs(ax, X, Y, phase_Z, xi, yi, resp, seam_pts, n_samples=28)

    # sparse support points
    sample = merged.sample(n=min(80, len(merged)), random_state=0)
    ax.scatter(sample["_x"], sample["_y"], sample["_phase"], c="white", s=4, alpha=0.10, depthshade=False)

    ax.set_title("Phase Geometry and Boundary-Activated Response")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_zlabel("Signed phase")
    ax.view_init(elev=27, azim=-58)
    ax.grid(False)

    # colorbar for Lazarus
    sm_laz = ScalarMappable(norm=Normalize(vmin=float(np.nanmin(merged["_lazarus"])), vmax=float(np.nanmax(merged["_lazarus"]))), cmap="magma")
    sm_laz.set_array([])
    cbar1 = fig.colorbar(sm_laz, ax=ax, fraction=0.03, pad=0.08)
    cbar1.set_label("Lazarus intensity")

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
