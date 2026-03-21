from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator


ROOT = Path("outputs")
OUTDIR = ROOT / "fim_response"
OUTDIR.mkdir(parents=True, exist_ok=True)

MDS_CSV = ROOT / "fim_mds" / "mds_coords.csv"
DIST_CSV = ROOT / "fim_phase" / "phase_distance_to_seam.csv"
SEAM_CSV = ROOT / "fim_phase" / "phase_boundary_mds_backprojected.csv"
LAZ_CSV = ROOT / "fim_lazarus" / "lazarus_scores.csv"

ALIGN_PNG = OUTDIR / "alignment_map.png"
STRENGTH_PNG = OUTDIR / "response_strength_map.png"
COUPLING_PNG = OUTDIR / "signed_coupling_map.png"
SUMMARY_CSV = OUTDIR / "response_summary.csv"


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
    if not finite.any():
        return np.full_like(values, np.nan, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
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
    merged["_lazarus"] = normalize_01(pd.to_numeric(merged[l_laz], errors="coerce").to_numpy(dtype=float))

    valid = (
        np.isfinite(merged["_x"].to_numpy()) &
        np.isfinite(merged["_y"].to_numpy()) &
        np.isfinite(merged["_phase"].to_numpy()) &
        np.isfinite(merged["_lazarus"].to_numpy())
    )
    merged = merged.loc[valid].copy()
    return merged, seam_pts


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
    alignment = dot / (laz_mag * phase_mag + eps)
    response_strength = laz_mag * phase_mag
    signed_coupling = dot

    return {
        "alignment": alignment,
        "response_strength": response_strength,
        "signed_coupling": signed_coupling,
        "laz_mag": laz_mag,
    }


def plot_field(X, Y, Z, seam_pts, title: str, cmap: str, outfile: Path, vmin=None, vmax=None, cbar_label: str = ""):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    cf = ax.contourf(X, Y, Z, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    if len(seam_pts) > 1:
        ax.plot(seam_pts[:, 0], seam_pts[:, 1], color="white", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_aspect("equal")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def summarize(fields: dict, seam_dist: np.ndarray) -> pd.DataFrame:
    alignment = fields["alignment"]
    response_strength = fields["response_strength"]
    signed_coupling = fields["signed_coupling"]
    laz_mag = fields["laz_mag"]

    finite = np.isfinite(alignment) & np.isfinite(response_strength) & np.isfinite(signed_coupling)
    seam_thresh = np.nanmedian(seam_dist[finite])
    near = finite & (seam_dist <= seam_thresh)
    far = finite & (seam_dist > seam_thresh)

    laz_thresh = np.nanmedian(laz_mag[finite])
    high_laz = finite & (laz_mag >= laz_thresh)
    low_laz = finite & (laz_mag < laz_thresh)

    rows = [
        ("mean_alignment_all", float(np.nanmean(alignment[finite]))),
        ("median_alignment_all", float(np.nanmedian(alignment[finite]))),
        ("mean_alignment_high_lazarus", float(np.nanmean(alignment[high_laz]))),
        ("mean_alignment_low_lazarus", float(np.nanmean(alignment[low_laz]))),
        ("mean_abs_alignment_near_seam", float(np.nanmean(np.abs(alignment[near])))),
        ("mean_abs_alignment_far_from_seam", float(np.nanmean(np.abs(alignment[far])))),
        ("mean_response_strength_all", float(np.nanmean(response_strength[finite]))),
        ("mean_response_strength_near_seam", float(np.nanmean(response_strength[near]))),
        ("mean_response_strength_far_from_seam", float(np.nanmean(response_strength[far]))),
        ("mean_signed_coupling_all", float(np.nanmean(signed_coupling[finite]))),
        ("mean_signed_coupling_near_seam", float(np.nanmean(signed_coupling[near]))),
        ("mean_signed_coupling_far_from_seam", float(np.nanmean(signed_coupling[far]))),
        ("n_grid_finite", int(finite.sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main():
    merged, seam_pts = build_fields()
    points = merged[["_x", "_y"]].to_numpy(dtype=float)

    phase_interp = build_interpolator(points, merged["_phase"].to_numpy(dtype=float), smoothing=0.001)
    laz_interp = build_interpolator(points, merged["_lazarus"].to_numpy(dtype=float), smoothing=0.001)

    X, Y, phase_Z, xi, yi = evaluate_grid(points, phase_interp, grid_res=220)
    _, _, laz_Z, _, _ = evaluate_grid(points, laz_interp, grid_res=220)

    fields = compute_response_fields(phase_Z, laz_Z, xi, yi)
    seam_dist = seam_distance_grid(X, Y, seam_pts)
    summary = summarize(fields, seam_dist)
    summary.to_csv(SUMMARY_CSV, index=False)

    plot_field(
        X, Y, fields["alignment"], seam_pts,
        title="Alignment map: ∇L · ∇φ / (|∇L||∇φ|)",
        cmap="coolwarm",
        outfile=ALIGN_PNG,
        vmin=-1,
        vmax=1,
        cbar_label="cosine alignment",
    )
    plot_field(
        X, Y, fields["response_strength"], seam_pts,
        title="Response-strength map: ||∇L|| ||∇φ||",
        cmap="magma",
        outfile=STRENGTH_PNG,
        cbar_label="response strength",
    )
    plot_field(
        X, Y, fields["signed_coupling"], seam_pts,
        title="Signed-coupling map: ∇L · ∇φ",
        cmap="PiYG",
        outfile=COUPLING_PNG,
        cbar_label="signed coupling",
    )

    print(f"Saved: {ALIGN_PNG}")
    print(f"Saved: {STRENGTH_PNG}")
    print(f"Saved: {COUPLING_PNG}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
