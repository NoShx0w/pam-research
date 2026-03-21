from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator


ROOT = Path("outputs")
OUTDIR = ROOT / "fim_report"
OUTDIR.mkdir(parents=True, exist_ok=True)

MDS_CSV = ROOT / "fim_mds" / "mds_coords.csv"
DIST_CSV = ROOT / "fim_phase" / "phase_distance_to_seam.csv"
SEAM_CSV = ROOT / "fim_phase" / "phase_boundary_mds_backprojected.csv"
LAZ_CSV = ROOT / "fim_lazarus" / "lazarus_scores.csv"

OUTFILE = OUTDIR / "figure1_candidate_operator_map.png"


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
    return RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=smoothing)


def evaluate_grid(points: np.ndarray, interpolator, grid_res: int = 260):
    x = points[:, 0]
    y = points[:, 1]
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = interpolator(XY).reshape(X.shape)
    return X, Y, Z, xi, yi


def seam_distance_grid(X: np.ndarray, Y: np.ndarray, seam_pts: np.ndarray) -> np.ndarray:
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
    return dmin.reshape(X.shape)


def compute_response_fields(phase_Z: np.ndarray, laz_Z: np.ndarray, xi: np.ndarray, yi: np.ndarray, eps: float = 1e-12):
    dphidy, dphidx = np.gradient(phase_Z, yi, xi)
    dLdy, dLdx = np.gradient(laz_Z, yi, xi)
    phase_mag = np.sqrt(dphidx**2 + dphidy**2)
    laz_mag = np.sqrt(dLdx**2 + dLdy**2)
    dot = dLdx * dphidx + dLdy * dphidy
    alignment = dot / (phase_mag * laz_mag + eps)
    response_strength = laz_mag * phase_mag
    return {
        "dphidx": dphidx,
        "dphidy": dphidy,
        "alignment": alignment,
        "response_strength": response_strength,
        "laz_mag": laz_mag,
    }


def candidate_traj_files():
    files = []
    for directory in [ROOT / "fim_geodesics", ROOT / "fim_phase"]:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob("geodesic_*.csv")))
        files.extend(sorted(directory.glob("path_*.csv")))
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


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


def draw_filaments(ax, X, Y, resp, seam_dist, n_samples: int = 40):
    score = normalize_01(resp["response_strength"])
    seam_weight = 1.0 - normalize_01(seam_dist)
    priority = np.nan_to_num(score) * (0.7 * np.nan_to_num(seam_weight) + 0.3)

    flat_idx = np.argsort(priority.ravel())[::-1]
    chosen = []
    min_sep = 10
    for idx in flat_idx:
        i, j = np.unravel_index(idx, X.shape)
        if not np.isfinite(resp["alignment"][i, j]):
            continue
        if any((abs(i - ci) < min_sep and abs(j - cj) < min_sep) for ci, cj in chosen):
            continue
        chosen.append((i, j))
        if len(chosen) >= n_samples:
            break

    cmap = plt.cm.coolwarm
    max_laz = np.nanmax(resp["laz_mag"]) + 1e-12
    max_strength = np.nanmax(resp["response_strength"]) + 1e-12

    for i, j in chosen:
        x0, y0 = X[i, j], Y[i, j]
        vx = resp["dphidx"][i, j]
        vy = resp["dphidy"][i, j]
        norm = np.hypot(vx, vy)
        if norm <= 1e-12:
            continue
        vx, vy = vx / norm, vy / norm

        mag = resp["laz_mag"][i, j]
        L = 0.08 + 0.22 * (mag / max_laz)
        alpha = 0.15 + 0.70 * (resp["response_strength"][i, j] / max_strength)
        color = cmap((resp["alignment"][i, j] + 1) / 2)

        x1, y1 = x0 - vx * L * 0.6, y0 - vy * L * 0.6
        x2, y2 = x0 + vx * L, y0 + vy * L
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1.8, solid_capstyle="round")


def main():
    merged, seam_pts = build_fields()
    points = merged[["_x", "_y"]].to_numpy(dtype=float)

    phase_interp = build_interpolator(points, merged["_phase"].to_numpy(dtype=float), smoothing=0.001)
    laz_interp = build_interpolator(points, merged["_lazarus"].to_numpy(dtype=float), smoothing=0.001)

    X, Y, phase_Z, xi, yi = evaluate_grid(points, phase_interp, grid_res=260)
    _, _, laz_Z, _, _ = evaluate_grid(points, laz_interp, grid_res=260)

    resp = compute_response_fields(phase_Z, laz_Z, xi, yi)
    seam_dist = seam_distance_grid(X, Y, seam_pts)

    strength = resp["response_strength"]
    strength_norm = normalize_01(strength)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)

    cf = ax.contourf(X, Y, strength_norm, levels=40, cmap="magma")
    ax.contour(
        X, Y, phase_Z,
        levels=[-0.75, -0.4, 0.0, 0.4, 0.75],
        colors="white",
        linewidths=[0.4, 0.5, 1.5, 0.5, 0.4],
        alpha=0.55
    )

    if len(seam_pts) > 1:
        ax.plot(seam_pts[:, 0], seam_pts[:, 1], color="white", linewidth=2.2, alpha=0.95)

    draw_filaments(ax, X, Y, resp, seam_dist, n_samples=42)

    path_xy = load_best_trajectory()
    if path_xy is not None:
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="cyan", linewidth=2.1, alpha=0.95)

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
            x, y = path_xy[idx, 0], path_xy[idx, 1]
            ax.scatter([x], [y], c=color, s=55 if marker != "*" else 85, marker=marker, zorder=8, edgecolors="black", linewidths=0.3)
            ax.text(x + 0.06, y + 0.04, label, color="white", fontsize=8, zorder=9)

    finite_strength = strength_norm[np.isfinite(strength_norm)]
    if finite_strength.size:
        quiet_mask = strength_norm < np.nanquantile(finite_strength, 0.35)
        if np.any(quiet_mask):
            veil = np.zeros((*quiet_mask.shape, 4))
            veil[..., :3] = 0.04
            veil[..., 3] = quiet_mask.astype(float) * 0.12
            ax.imshow(veil, extent=[X.min(), X.max(), Y.min(), Y.max()], origin="lower", aspect="auto")

    ax.set_title("Phase Geometry and Boundary-Activated Response")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_aspect("equal")

    cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("response strength")

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight")
    print(f"Saved: {OUTFILE}")


if __name__ == "__main__":
    main()
