#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_path_family_stratification.py

Stratify geodesic paths into interpretable route families.

Input
-----
CSV produced by geodesic_path_diagnostics.py, expected to contain:
- path_id
- n_nodes
- min_distance_to_seam
- near_fraction
- mean_criticality
- max_criticality
- mean_unsigned_obstruction
- mean_absolute_holonomy
- mean_angle_jump_deg
- max_angle_jump_deg
- n_sector_changes

Outputs
-------
- geodesic_path_family_assignments.csv
- geodesic_path_family_summary.csv
- geodesic_path_family_summary.txt
- geodesic_family_near_vs_roughness.png
- geodesic_family_load_vs_criticality.png
"""

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    path_csv: str
    outdir: str = "outputs/toy_geodesic_path_families"
    seam_contact_threshold: float = 0.50
    roughness_quantile: float = 0.67
    sector_change_quantile: float = 0.67


def _safe_quantile(series: pd.Series, q: float) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    return float(x.quantile(q))


def classify_paths(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = df.copy()

    rough_thr = _safe_quantile(work["mean_angle_jump_deg"], cfg.roughness_quantile)
    change_thr = _safe_quantile(work["n_sector_changes"], cfg.sector_change_quantile)

    def classify(row: pd.Series) -> str:
        near_fraction = pd.to_numeric(row.get("near_fraction"), errors="coerce")
        rough = pd.to_numeric(row.get("mean_angle_jump_deg"), errors="coerce")
        changes = pd.to_numeric(row.get("n_sector_changes"), errors="coerce")

        seam_heavy = pd.notna(near_fraction) and near_fraction >= cfg.seam_contact_threshold
        rough_high = pd.notna(rough) and pd.notna(rough_thr) and rough >= rough_thr
        change_high = pd.notna(changes) and pd.notna(change_thr) and changes >= change_thr

        if seam_heavy and (rough_high or change_high):
            return "reorganization_heavy"
        if seam_heavy:
            return "stable_seam_corridor"
        if rough_high or change_high:
            return "off_seam_reorganizing"
        return "settled_distant"

    work["path_family"] = work.apply(classify, axis=1)
    work["roughness_threshold"] = rough_thr
    work["sector_change_threshold"] = change_thr
    return work


def build_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "path_family",
        "n_nodes",
        "min_distance_to_seam",
        "near_fraction",
        "mean_criticality",
        "max_criticality",
        "mean_unsigned_obstruction",
        "mean_absolute_holonomy",
        "mean_angle_jump_deg",
        "max_angle_jump_deg",
        "n_sector_changes",
    ]
    work = df[cols].copy()
    summary = (
        work.groupby("path_family", dropna=False)
        .agg(
            n_paths=("path_family", "size"),
            mean_length=("n_nodes", "mean"),
            mean_min_seam=("min_distance_to_seam", "mean"),
            mean_near_fraction=("near_fraction", "mean"),
            mean_criticality=("mean_criticality", "mean"),
            mean_max_criticality=("max_criticality", "mean"),
            mean_unsigned_obstruction=("mean_unsigned_obstruction", "mean"),
            mean_absolute_holonomy=("mean_absolute_holonomy", "mean"),
            mean_angle_jump=("mean_angle_jump_deg", "mean"),
            mean_max_angle_jump=("max_angle_jump_deg", "mean"),
            mean_sector_changes=("n_sector_changes", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("n_paths", ascending=False).reset_index(drop=True)


def summarize_text(df: pd.DataFrame, fam: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Geodesic Path Family Stratification Summary ===")
    lines.append("")
    lines.append(f"n_paths = {len(df)}")
    if len(df):
        lines.append(
            f"roughness_threshold = {pd.to_numeric(df['roughness_threshold'], errors='coerce').iloc[0]:.4f}"
        )
        lines.append(
            f"sector_change_threshold = {pd.to_numeric(df['sector_change_threshold'], errors='coerce').iloc[0]:.4f}"
        )
    lines.append("")

    for _, row in fam.iterrows():
        lines.append(
            f"{row['path_family']}: "
            f"n_paths={int(row['n_paths'])}, "
            f"mean_length={row['mean_length']:.2f}, "
            f"mean_min_seam={row['mean_min_seam']:.4f}, "
            f"mean_near_fraction={row['mean_near_fraction']:.4f}, "
            f"mean_criticality={row['mean_criticality']:.4f}, "
            f"mean_max_criticality={row['mean_max_criticality']:.4f}, "
            f"mean_unsigned_obstruction={row['mean_unsigned_obstruction']:.4f}, "
            f"mean_abs_holonomy={row['mean_absolute_holonomy']:.4f}, "
            f"mean_angle_jump={row['mean_angle_jump']:.4f}, "
            f"mean_max_angle_jump={row['mean_max_angle_jump']:.4f}, "
            f"mean_sector_changes={row['mean_sector_changes']:.4f}"
        )
    return "\n".join(lines)


def plot_family_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str) -> None:
    plot_df = df[[x, y, "path_family"]].dropna().copy()
    fig, ax = plt.subplots(figsize=(8, 6))

    for fam, sub in plot_df.groupby("path_family", dropna=False):
        ax.scatter(sub[x], sub[y], s=45, label=str(fam))

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify geodesic paths into route families.")
    parser.add_argument("--path-csv", required=True)
    parser.add_argument("--outdir", default="outputs/toy_geodesic_path_families")
    parser.add_argument("--seam-contact-threshold", type=float, default=0.50)
    parser.add_argument("--roughness-quantile", type=float, default=0.67)
    parser.add_argument("--sector-change-quantile", type=float, default=0.67)
    args = parser.parse_args()

    cfg = Config(
        path_csv=args.path_csv,
        outdir=args.outdir,
        seam_contact_threshold=args.seam_contact_threshold,
        roughness_quantile=args.roughness_quantile,
        sector_change_quantile=args.sector_change_quantile,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.path_csv).copy()
    df = classify_paths(df, cfg)
    fam = build_family_summary(df)

    df.to_csv(outdir / "geodesic_path_family_assignments.csv", index=False)
    fam.to_csv(outdir / "geodesic_path_family_summary.csv", index=False)
    (outdir / "geodesic_path_family_summary.txt").write_text(
        summarize_text(df, fam),
        encoding="utf-8",
    )

    plot_family_scatter(
        df,
        "near_fraction",
        "mean_angle_jump_deg",
        outdir / "geodesic_family_near_vs_roughness.png",
        "Path families: seam contact vs roughness",
    )
    plot_family_scatter(
        df,
        "mean_unsigned_obstruction",
        "mean_criticality",
        outdir / "geodesic_family_load_vs_criticality.png",
        "Path families: load vs criticality",
    )

    print(outdir / "geodesic_path_family_assignments.csv")
    print(outdir / "geodesic_path_family_summary.csv")
    print(outdir / "geodesic_path_family_summary.txt")
    print(outdir / "geodesic_family_near_vs_roughness.png")
    print(outdir / "geodesic_family_load_vs_criticality.png")


if __name__ == "__main__":
    main()
