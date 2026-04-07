#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    paths_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    seam_csv: str = "outputs/fim_phase/phase_boundary_mds_backprojected.csv"
    phase_csv: str = "outputs/fim_phase/phase_distance_to_seam.csv"
    outdir: str = "outputs/toy_seam_tangent_alignment"
    seam_band_quantile: float = 0.25
    seam_nn_k: int = 5


def wrap_angle_pi(theta: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def undirected_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(wrap_angle_pi(a - b))
    return np.minimum(d, np.pi - d)


def angle_to_alignment(diff: np.ndarray) -> np.ndarray:
    return np.abs(np.cos(diff))


def load_paths(paths_csv: str, family_csv: str) -> pd.DataFrame:
    paths = pd.read_csv(paths_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")
    return paths


def compute_path_tangents(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path_id, sub in paths.groupby("path_id", sort=False):
        sub = sub.sort_values("step").reset_index(drop=True)
        if len(sub) < 2:
            continue

        xs = pd.to_numeric(sub["mds1"], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(sub["mds2"], errors="coerce").to_numpy(dtype=float)

        for i in range(len(sub)):
            if i == 0:
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
            elif i == len(sub) - 1:
                dx = xs[i] - xs[i - 1]
                dy = ys[i] - ys[i - 1]
            else:
                dx = xs[i + 1] - xs[i - 1]
                dy = ys[i + 1] - ys[i - 1]

            theta = math.atan2(dy, dx) if np.isfinite(dx) and np.isfinite(dy) else np.nan

            row = sub.loc[i, ["path_id", "path_family", "step", "node_id", "r", "alpha", "mds1", "mds2"]].to_dict()
            row["path_theta"] = theta
            rows.append(row)

    return pd.DataFrame(rows)


def load_phase_distance(phase_csv: str) -> pd.DataFrame:
    df = pd.read_csv(phase_csv).copy()
    keep = [c for c in ["node_id", "r", "alpha", "distance_to_seam"] if c in df.columns]
    return df[keep]


def estimate_local_tangent(seam_df: pd.DataFrame, x: float, y: float, k: int) -> float:
    pts = seam_df[["mds1", "mds2"]].to_numpy(dtype=float)
    if len(pts) < 2:
        return np.nan

    d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
    idx = np.argsort(d2)[: max(2, min(k, len(pts)))]
    nbrs = pts[idx]

    center = nbrs.mean(axis=0)
    A = nbrs - center
    try:
        _, _, vh = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan

    v = vh[0]  # principal local seam direction
    return float(math.atan2(v[1], v[0]))


def annotate_seam_directions(points: pd.DataFrame, seam_df: pd.DataFrame, k: int) -> pd.DataFrame:
    thetas = []
    for _, row in points.iterrows():
        thetas.append(
            estimate_local_tangent(
                seam_df,
                float(row["mds1"]),
                float(row["mds2"]),
                k=k,
            )
        )
    out = points.copy()
    out["seam_tangent_theta"] = np.array(thetas, dtype=float)
    out["seam_normal_theta"] = wrap_angle_pi(out["seam_tangent_theta"].to_numpy(dtype=float) + np.pi / 2.0)
    return out


def summarize_family(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("path_family", dropna=False)
        .agg(
            n_points=("path_family", "size"),
            mean_align_tangent=("align_tangent", "mean"),
            mean_align_normal=("align_normal", "mean"),
            mean_diff_tangent_deg=("diff_tangent_deg", "mean"),
            mean_diff_normal_deg=("diff_normal_deg", "mean"),
            mean_distance_to_seam=("distance_to_seam", "mean"),
        )
        .reset_index()
        .sort_values("n_points", ascending=False)
        .reset_index(drop=True)
    )


def write_summary_text(global_summary: pd.DataFrame, seam_summary: pd.DataFrame, seam_thr: float, outpath: Path) -> None:
    lines = ["=== Seam Tangent Alignment Summary ===", ""]
    lines.append(f"seam_band_threshold = {seam_thr:.4f}")
    lines.append("")
    lines.append("Global family summary")
    for _, row in global_summary.iterrows():
        lines.append(
            f"  {row['path_family']}: "
            f"n_points={int(row['n_points'])}, "
            f"mean_align_tangent={row['mean_align_tangent']:.4f}, "
            f"mean_align_normal={row['mean_align_normal']:.4f}, "
            f"mean_diff_tangent_deg={row['mean_diff_tangent_deg']:.2f}, "
            f"mean_diff_normal_deg={row['mean_diff_normal_deg']:.2f}, "
            f"mean_distance_to_seam={row['mean_distance_to_seam']:.4f}"
        )
    lines.append("")
    lines.append("Seam-band family summary")
    for _, row in seam_summary.iterrows():
        lines.append(
            f"  {row['path_family']}: "
            f"n_points={int(row['n_points'])}, "
            f"mean_align_tangent={row['mean_align_tangent']:.4f}, "
            f"mean_align_normal={row['mean_align_normal']:.4f}, "
            f"mean_diff_tangent_deg={row['mean_diff_tangent_deg']:.2f}, "
            f"mean_diff_normal_deg={row['mean_diff_normal_deg']:.2f}, "
            f"mean_distance_to_seam={row['mean_distance_to_seam']:.4f}"
        )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(summary: pd.DataFrame, value_col: str, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    ax.bar(x, pd.to_numeric(summary[value_col], errors="coerce").to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(summary["path_family"], rotation=20, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare path-family alignment with local seam tangent vs seam normal.")
    parser.add_argument("--paths-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/toy_seam_tangent_alignment")
    parser.add_argument("--seam-band-quantile", type=float, default=0.25)
    parser.add_argument("--seam-nn-k", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(
        paths_csv=args.paths_csv,
        family_csv=args.family_csv,
        seam_csv=args.seam_csv,
        phase_csv=args.phase_csv,
        outdir=args.outdir,
        seam_band_quantile=args.seam_band_quantile,
        seam_nn_k=args.seam_nn_k,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(cfg.paths_csv, cfg.family_csv)
    tangents = compute_path_tangents(paths)

    phase = load_phase_distance(cfg.phase_csv)
    tangents = tangents.merge(phase, on=["node_id", "r", "alpha"], how="left")

    seam_df = pd.read_csv(cfg.seam_csv).copy()
    if not {"mds1", "mds2"}.issubset(seam_df.columns):
        raise ValueError("seam csv must contain mds1 and mds2")

    aligned = annotate_seam_directions(tangents, seam_df, cfg.seam_nn_k)

    diff_tangent = undirected_angle_diff(
        pd.to_numeric(aligned["path_theta"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(aligned["seam_tangent_theta"], errors="coerce").to_numpy(dtype=float),
    )
    diff_normal = undirected_angle_diff(
        pd.to_numeric(aligned["path_theta"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(aligned["seam_normal_theta"], errors="coerce").to_numpy(dtype=float),
    )

    aligned["diff_tangent_deg"] = np.degrees(diff_tangent)
    aligned["diff_normal_deg"] = np.degrees(diff_normal)
    aligned["align_tangent"] = angle_to_alignment(diff_tangent)
    aligned["align_normal"] = angle_to_alignment(diff_normal)

    seam_thr = float(pd.to_numeric(aligned["distance_to_seam"], errors="coerce").quantile(cfg.seam_band_quantile))
    seam_band = aligned[pd.to_numeric(aligned["distance_to_seam"], errors="coerce") <= seam_thr].copy()

    global_summary = summarize_family(aligned)
    seam_summary = summarize_family(seam_band)

    aligned.to_csv(outdir / "seam_tangent_alignment_points.csv", index=False)
    global_summary.to_csv(outdir / "seam_tangent_alignment_global_summary.csv", index=False)
    seam_summary.to_csv(outdir / "seam_tangent_alignment_seam_band_summary.csv", index=False)
    write_summary_text(global_summary, seam_summary, seam_thr, outdir / "seam_tangent_alignment_summary.txt")

    plot_metric(
        seam_summary,
        "mean_align_tangent",
        "Seam-band tangent alignment by family",
        outdir / "seam_tangent_alignment_seam_band.png",
    )
    plot_metric(
        seam_summary,
        "mean_align_normal",
        "Seam-band normal alignment by family",
        outdir / "seam_normal_alignment_seam_band.png",
    )

    print(outdir / "seam_tangent_alignment_points.csv")
    print(outdir / "seam_tangent_alignment_global_summary.csv")
    print(outdir / "seam_tangent_alignment_seam_band_summary.csv")
    print(outdir / "seam_tangent_alignment_summary.txt")
    print(outdir / "seam_tangent_alignment_seam_band.png")
    print(outdir / "seam_normal_alignment_seam_band.png")


if __name__ == "__main__":
    main()
