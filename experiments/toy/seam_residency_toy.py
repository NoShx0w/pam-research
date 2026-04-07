#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    paths_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    phase_csv: str = "outputs/fim_phase/phase_distance_to_seam.csv"
    outdir: str = "outputs/toy_seam_residency"
    seam_threshold: float = 0.15


def load_paths(paths_csv: str, family_csv: str, phase_csv: str) -> pd.DataFrame:
    paths = pd.read_csv(paths_csv).copy()
    fam = pd.read_csv(family_csv).copy()
    phase = pd.read_csv(phase_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    keep_phase = [c for c in ["node_id", "r", "alpha", "distance_to_seam"] if c in phase.columns]
    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")
    paths = paths.merge(phase[keep_phase], on=["node_id", "r", "alpha"], how="left")
    return paths


def seam_run_lengths(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    cur = 0
    for x in mask:
        if x:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def summarize_paths(paths: pd.DataFrame, seam_threshold: float) -> pd.DataFrame:
    rows = []

    for path_id, sub in paths.groupby("path_id", sort=False):
        sub = sub.sort_values("step").reset_index(drop=True)
        fam = str(sub["path_family"].iloc[0]) if len(sub) else ""
        d = pd.to_numeric(sub["distance_to_seam"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(d) & (d <= seam_threshold)

        runs = seam_run_lengths(mask.tolist())
        n_points = len(sub)
        n_seam_points = int(mask.sum())

        rows.append(
            {
                "path_id": path_id,
                "path_family": fam,
                "n_points": n_points,
                "n_seam_points": n_seam_points,
                "seam_fraction": float(n_seam_points / n_points) if n_points > 0 else np.nan,
                "n_seam_episodes": int(len(runs)),
                "mean_seam_run_length": float(np.mean(runs)) if runs else 0.0,
                "max_seam_run_length": int(max(runs)) if runs else 0,
                "ever_seam": int(n_seam_points > 0),
                "mean_distance_to_seam": float(np.nanmean(d)) if len(d) else np.nan,
                "min_distance_to_seam": float(np.nanmin(d)) if len(d) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def summarize_families(path_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        path_summary.groupby("path_family", dropna=False)
        .agg(
            n_paths=("path_id", "count"),
            seam_contact_rate=("ever_seam", "mean"),
            mean_seam_fraction=("seam_fraction", "mean"),
            mean_n_seam_episodes=("n_seam_episodes", "mean"),
            mean_seam_run_length=("mean_seam_run_length", "mean"),
            mean_max_seam_run_length=("max_seam_run_length", "mean"),
            mean_distance_to_seam=("mean_distance_to_seam", "mean"),
            mean_min_distance_to_seam=("min_distance_to_seam", "mean"),
        )
        .reset_index()
        .sort_values("n_paths", ascending=False)
        .reset_index(drop=True)
    )


def write_summary_text(summary: pd.DataFrame, seam_threshold: float, outpath: Path) -> None:
    lines = ["=== Seam Residency Summary ===", ""]
    lines.append(f"seam_threshold = {seam_threshold:.4f}")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(
            f"{row['path_family']}: "
            f"n_paths={int(row['n_paths'])}, "
            f"seam_contact_rate={row['seam_contact_rate']:.4f}, "
            f"mean_seam_fraction={row['mean_seam_fraction']:.4f}, "
            f"mean_n_seam_episodes={row['mean_n_seam_episodes']:.4f}, "
            f"mean_seam_run_length={row['mean_seam_run_length']:.4f}, "
            f"mean_max_seam_run_length={row['mean_max_seam_run_length']:.4f}, "
            f"mean_distance_to_seam={row['mean_distance_to_seam']:.4f}, "
            f"mean_min_distance_to_seam={row['mean_min_distance_to_seam']:.4f}"
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
    parser = argparse.ArgumentParser(description="Measure seam residency by canonical route family.")
    parser.add_argument("--paths-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/toy_seam_residency")
    parser.add_argument("--seam-threshold", type=float, default=0.15)
    args = parser.parse_args()

    cfg = Config(
        paths_csv=args.paths_csv,
        family_csv=args.family_csv,
        phase_csv=args.phase_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(cfg.paths_csv, cfg.family_csv, cfg.phase_csv)
    path_summary = summarize_paths(paths, cfg.seam_threshold)
    family_summary = summarize_families(path_summary)

    path_summary.to_csv(outdir / "seam_residency_path_summary.csv", index=False)
    family_summary.to_csv(outdir / "seam_residency_family_summary.csv", index=False)
    write_summary_text(family_summary, cfg.seam_threshold, outdir / "seam_residency_summary.txt")

    plot_metric(
        family_summary,
        "mean_seam_fraction",
        "Mean seam fraction by family",
        outdir / "seam_residency_fraction.png",
    )
    plot_metric(
        family_summary,
        "mean_seam_run_length",
        "Mean seam run length by family",
        outdir / "seam_residency_run_length.png",
    )
    plot_metric(
        family_summary,
        "mean_max_seam_run_length",
        "Mean max seam run length by family",
        outdir / "seam_residency_max_run_length.png",
    )

    print(outdir / "seam_residency_path_summary.csv")
    print(outdir / "seam_residency_family_summary.csv")
    print(outdir / "seam_residency_summary.txt")
    print(outdir / "seam_residency_fraction.png")
    print(outdir / "seam_residency_run_length.png")
    print(outdir / "seam_residency_max_run_length.png")


if __name__ == "__main__":
    main()
