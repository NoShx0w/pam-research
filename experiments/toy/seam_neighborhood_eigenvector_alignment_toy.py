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
    fim_csv: str = "outputs/fim/fim_surface.csv"
    response_csv: str = "outputs/fim_response_operator/response_operator_nodes.csv"
    phase_csv: str = "outputs/fim_phase/phase_distance_to_seam.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    outdir: str = "outputs/toy_seam_neighborhood_eigenvector_alignment"
    seam_threshold: float = 0.15


def wrap_angle_pi(theta: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def undirected_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(wrap_angle_pi(a - b))
    return np.minimum(d, np.pi - d)


def angle_to_alignment(diff: np.ndarray) -> np.ndarray:
    return np.abs(np.cos(diff))


def principal_response_eig(
    t_xx: np.ndarray,
    t_xy: np.ndarray,
    t_yx: np.ndarray,
    t_yy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t_xx)
    eig1 = np.full(n, np.nan)
    eig2 = np.full(n, np.nan)
    theta = np.full(n, np.nan)

    for i in range(n):
        M = np.array([[t_xx[i], t_xy[i]], [t_yx[i], t_yy[i]]], dtype=float)
        if not np.isfinite(M).all():
            continue
        vals, vecs = np.linalg.eig(M)
        order = np.argsort(np.abs(vals))[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        v = np.real(vecs[:, 0])
        eig1[i] = float(np.real(vals[0]))
        eig2[i] = float(np.real(vals[1]))
        theta[i] = float(math.atan2(v[1], v[0]))

    return eig1, eig2, theta


def load_node_field(fim_csv: str, response_csv: str, phase_csv: str) -> pd.DataFrame:
    fim = pd.read_csv(fim_csv).copy()
    rsp = pd.read_csv(response_csv).copy()
    phase = pd.read_csv(phase_csv).copy()

    keep_fim = [c for c in ["r", "alpha", "fim_eig1", "fim_eig2", "fim_cond", "fim_theta"] if c in fim.columns]
    keep_rsp = [c for c in ["r", "alpha", "mds1", "mds2", "T_xx", "T_xy", "T_yx", "T_yy", "response_strength", "lazarus_score"] if c in rsp.columns]
    keep_phase = [c for c in ["node_id", "r", "alpha", "distance_to_seam", "signed_phase"] if c in phase.columns]

    df = rsp[keep_rsp].merge(fim[keep_fim], on=["r", "alpha"], how="left")
    df = df.merge(phase[keep_phase], on=["r", "alpha"], how="left")

    eig1, eig2, theta = principal_response_eig(
        pd.to_numeric(df["T_xx"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["T_xy"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["T_yx"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["T_yy"], errors="coerce").to_numpy(dtype=float),
    )
    df["rsp_eig1"] = eig1
    df["rsp_eig2"] = eig2
    df["rsp_theta"] = theta
    df["rsp_cond_like"] = np.where(
        np.abs(df["rsp_eig2"]) > 1e-12,
        np.abs(df["rsp_eig1"]) / np.abs(df["rsp_eig2"]),
        np.nan,
    )
    return df


def compute_path_tangents(path_nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path_id, sub in path_nodes.groupby("path_id", sort=False):
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
            rows.append(
                {
                    "path_id": path_id,
                    "step": int(sub.loc[i, "step"]),
                    "node_id": int(sub.loc[i, "node_id"]) if "node_id" in sub.columns else np.nan,
                    "r": float(sub.loc[i, "r"]),
                    "alpha": float(sub.loc[i, "alpha"]),
                    "mds1": float(sub.loc[i, "mds1"]),
                    "mds2": float(sub.loc[i, "mds2"]),
                    "path_theta": theta,
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("path_family", dropna=False)
        .agg(
            n_points=("path_family", "size"),
            mean_align_fim=("align_fim", "mean"),
            mean_align_rsp=("align_rsp", "mean"),
            mean_diff_fim_deg=("diff_fim_deg", "mean"),
            mean_diff_rsp_deg=("diff_rsp_deg", "mean"),
            mean_distance_to_seam=("distance_to_seam", "mean"),
            mean_lazarus=("lazarus_score", "mean"),
            mean_response_strength=("response_strength", "mean"),
        )
        .reset_index()
        .sort_values("n_points", ascending=False)
        .reset_index(drop=True)
    )


def write_summary_text(global_summary: pd.DataFrame, seam_summary: pd.DataFrame, seam_threshold: float, outpath: Path) -> None:
    lines = ["=== Seam-Neighborhood Eigenvector Alignment Summary ===", ""]
    lines.append(f"seam_threshold = {seam_threshold:.4f}")
    lines.append("")
    lines.append("Global family summary")
    for _, row in global_summary.iterrows():
        lines.append(
            f"  {row['path_family']}: "
            f"n_points={int(row['n_points'])}, "
            f"mean_align_fim={row['mean_align_fim']:.4f}, "
            f"mean_align_rsp={row['mean_align_rsp']:.4f}, "
            f"mean_diff_fim_deg={row['mean_diff_fim_deg']:.2f}, "
            f"mean_diff_rsp_deg={row['mean_diff_rsp_deg']:.2f}, "
            f"mean_distance_to_seam={row['mean_distance_to_seam']:.4f}, "
            f"mean_lazarus={row['mean_lazarus']:.4f}, "
            f"mean_response_strength={row['mean_response_strength']:.4f}"
        )
    lines.append("")
    lines.append("Seam-neighborhood family summary")
    for _, row in seam_summary.iterrows():
        lines.append(
            f"  {row['path_family']}: "
            f"n_points={int(row['n_points'])}, "
            f"mean_align_fim={row['mean_align_fim']:.4f}, "
            f"mean_align_rsp={row['mean_align_rsp']:.4f}, "
            f"mean_diff_fim_deg={row['mean_diff_fim_deg']:.2f}, "
            f"mean_diff_rsp_deg={row['mean_diff_rsp_deg']:.2f}, "
            f"mean_distance_to_seam={row['mean_distance_to_seam']:.4f}, "
            f"mean_lazarus={row['mean_lazarus']:.4f}, "
            f"mean_response_strength={row['mean_response_strength']:.4f}"
        )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(summary: pd.DataFrame, value_col: str, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    vals = pd.to_numeric(summary[value_col], errors="coerce").to_numpy(dtype=float)
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["path_family"], rotation=20, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Fisher/response eigenvector alignment inside the seam neighborhood.")
    parser.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    parser.add_argument("--response-csv", default="outputs/fim_response_operator/response_operator_nodes.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--outdir", default="outputs/toy_seam_neighborhood_eigenvector_alignment")
    parser.add_argument("--seam-threshold", type=float, default=0.15)
    args = parser.parse_args()

    cfg = Config(
        fim_csv=args.fim_csv,
        response_csv=args.response_csv,
        phase_csv=args.phase_csv,
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    node_df = load_node_field(cfg.fim_csv, cfg.response_csv, cfg.phase_csv)
    fam = pd.read_csv(cfg.family_csv).copy()
    path_nodes = pd.read_csv(cfg.path_nodes_csv).copy()

    tangents = compute_path_tangents(path_nodes)
    aligned = tangents.merge(
        fam[["path_id", "path_family"]],
        on="path_id",
        how="left",
    ).merge(
        node_df,
        on=["r", "alpha"],
        how="left",
    )

    diff_fim = undirected_angle_diff(
        pd.to_numeric(aligned["path_theta"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(aligned["fim_theta"], errors="coerce").to_numpy(dtype=float),
    )
    diff_rsp = undirected_angle_diff(
        pd.to_numeric(aligned["path_theta"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(aligned["rsp_theta"], errors="coerce").to_numpy(dtype=float),
    )

    aligned["diff_fim_deg"] = np.degrees(diff_fim)
    aligned["diff_rsp_deg"] = np.degrees(diff_rsp)
    aligned["align_fim"] = angle_to_alignment(diff_fim)
    aligned["align_rsp"] = angle_to_alignment(diff_rsp)

    seam_df = aligned[pd.to_numeric(aligned["distance_to_seam"], errors="coerce") <= cfg.seam_threshold].copy()

    global_summary = summarize(aligned)
    seam_summary = summarize(seam_df)

    aligned.to_csv(outdir / "seam_neighborhood_eigenvector_alignment_points.csv", index=False)
    global_summary.to_csv(outdir / "seam_neighborhood_eigenvector_alignment_global_summary.csv", index=False)
    seam_summary.to_csv(outdir / "seam_neighborhood_eigenvector_alignment_seam_summary.csv", index=False)
    write_summary_text(global_summary, seam_summary, cfg.seam_threshold, outdir / "seam_neighborhood_eigenvector_alignment_summary.txt")

    plot_metric(
        seam_summary,
        "mean_align_fim",
        "Seam-neighborhood Fisher alignment by family",
        outdir / "seam_neighborhood_fim_alignment.png",
    )
    plot_metric(
        seam_summary,
        "mean_align_rsp",
        "Seam-neighborhood response alignment by family",
        outdir / "seam_neighborhood_response_alignment.png",
    )

    print(outdir / "seam_neighborhood_eigenvector_alignment_points.csv")
    print(outdir / "seam_neighborhood_eigenvector_alignment_global_summary.csv")
    print(outdir / "seam_neighborhood_eigenvector_alignment_seam_summary.csv")
    print(outdir / "seam_neighborhood_eigenvector_alignment_summary.txt")
    print(outdir / "seam_neighborhood_fim_alignment.png")
    print(outdir / "seam_neighborhood_response_alignment.png")


if __name__ == "__main__":
    main()
