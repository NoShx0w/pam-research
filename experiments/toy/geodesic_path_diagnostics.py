#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_path_diagnostics.py

Annotate existing geodesic paths with seam-, transport-, topology-, and
identity-related diagnostics.

Goal
----
Take geodesics as primary again, and summarize how they interact with:

- phase seam proximity
- criticality
- transport load
- identity-angle roughness
- identity sector changes

Expected path input
-------------------
A CSV with one row per path-node membership, containing at least:

- path_id
- step
- r
- alpha

Optional:
- source_id / target_id
- node_id

This script joins node-level fields from canonical outputs and emits:

- path_diagnostics.csv
- path_node_diagnostics.csv
- seam_class_summary.csv
- geodesic_path_diagnostics_summary.txt
- geodesic_min_seam_vs_length.png
- geodesic_load_vs_roughness.png
- geodesic_criticality_vs_seam_fraction.png
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BEST_BASIS = (
    "absolute_holonomy_node",
    "obstruction_mean_abs_holonomy",
    "obstruction_signed_sum_holonomy",
)


@dataclass(frozen=True)
class Config:
    paths_csv: str
    outputs_root: str = "outputs"
    outdir: str = "outputs/toy_geodesic_path_diagnostics"
    eps: float = 1e-12
    near_q: float = 0.33
    mid_q: float = 0.66


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).copy() if path.exists() else pd.DataFrame()


def _zscore(series: pd.Series, eps: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean(skipna=True)) if len(x.dropna()) else 0.0
    sd = float(x.std(skipna=True)) if len(x.dropna()) else 0.0
    return (x - mu) / max(sd, eps)


def _corr(df: pd.DataFrame, x: str, y: str) -> float:
    if x not in df.columns or y not in df.columns:
        return float("nan")
    work = df[[x, y]].dropna()
    if len(work) < 3:
        return float("nan")
    return float(work[x].corr(work[y], method="spearman"))


def circ_abs_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def load_node_fields(outputs_root: Path) -> pd.DataFrame:
    identity_csv = outputs_root / "fim_identity" / "identity_field_nodes.csv"
    obstruction_csv = outputs_root / "fim_identity_obstruction" / "identity_obstruction_nodes.csv"
    obstruction_signed_csv = outputs_root / "fim_identity_obstruction" / "identity_obstruction_signed_nodes.csv"
    phase_csv = outputs_root / "fim_phase" / "signed_phase_coords.csv"
    phase_distance_csv = outputs_root / "fim_phase" / "phase_distance_to_seam.csv"
    criticality_csv = outputs_root / "fim_critical" / "criticality_surface.csv"

    identity = _safe_read_csv(identity_csv)
    obstruction = _safe_read_csv(obstruction_csv)
    obstruction_signed = _safe_read_csv(obstruction_signed_csv)
    phase = _safe_read_csv(phase_csv)
    phase_distance = _safe_read_csv(phase_distance_csv)
    criticality = _safe_read_csv(criticality_csv)

    if identity.empty:
        raise FileNotFoundError(f"Required input missing or empty: {identity_csv}")

    base_cols = [
        c for c in ["node_id", "i", "j", "r", "alpha", "identity_magnitude", "identity_spin"]
        if c in identity.columns
    ]
    merged = identity[base_cols].copy()

    if not obstruction.empty:
        keep = [
            c for c in [
                "node_id",
                "obstruction_mean_holonomy",
                "obstruction_mean_abs_holonomy",
                "obstruction_max_abs_holonomy",
            ] if c in obstruction.columns
        ]
        if keep:
            merged = merged.merge(obstruction[keep], on="node_id", how="left")

    if not obstruction_signed.empty:
        keep = [
            c for c in [
                "node_id",
                "obstruction_signed_sum_holonomy",
                "obstruction_signed_weighted_holonomy",
            ] if c in obstruction_signed.columns
        ]
        if keep:
            merged = merged.merge(obstruction_signed[keep], on="node_id", how="left")

    if "obstruction_mean_abs_holonomy" in merged.columns:
        merged["absolute_holonomy_node"] = merged["obstruction_mean_abs_holonomy"]
    else:
        merged["absolute_holonomy_node"] = np.nan

    if not phase.empty:
        keep = [c for c in ["r", "alpha", "signed_phase"] if c in phase.columns]
        if keep:
            for c in keep:
                phase[c] = pd.to_numeric(phase[c], errors="coerce")
            merged = merged.merge(phase[keep], on=["r", "alpha"], how="left")

    if not phase_distance.empty:
        keep = [c for c in ["r", "alpha", "distance_to_seam"] if c in phase_distance.columns]
        if keep:
            for c in keep:
                phase_distance[c] = pd.to_numeric(phase_distance[c], errors="coerce")
            merged = merged.merge(phase_distance[keep], on=["r", "alpha"], how="left")

    if not criticality.empty:
        keep = [c for c in ["r", "alpha", "criticality"] if c in criticality.columns]
        if keep:
            for c in keep:
                criticality[c] = pd.to_numeric(criticality[c], errors="coerce")
            merged = merged.merge(criticality[keep], on=["r", "alpha"], how="left")

    for col in merged.columns:
        if col != "node_id":
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return build_angle_fields(merged).sort_values(["r", "alpha"]).reset_index(drop=True)


def build_angle_fields(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    z1 = _zscore(work[BEST_BASIS[0]], 1e-12)
    z2 = _zscore(work[BEST_BASIS[1]], 1e-12)
    z3 = _zscore(work[BEST_BASIS[2]], 1e-12)

    mean_z = (z1 + z2 + z3) / 3.0
    d1 = z1 - mean_z
    d2 = z2 - mean_z
    d3 = z3 - mean_z

    x = 0.5 * (2.0 * d1 - d2 - d3)
    y = (math.sqrt(3.0) / 2.0) * (d2 - d3)

    angle_deg = np.degrees(np.arctan2(y, x))
    work["dew_angle_deg"] = angle_deg
    work["dew_sector"] = pd.cut(
        angle_deg,
        bins=[-180, -120, -60, 0, 60, 120, 180],
        labels=["S1", "S2", "S3", "S4", "S5", "S6"],
        include_lowest=True,
    )
    return work


def assign_seam_bins(df: pd.DataFrame, near_q: float, mid_q: float) -> pd.DataFrame:
    work = df.copy()
    seam = pd.to_numeric(work["distance_to_seam"], errors="coerce")
    q1 = float(seam.quantile(near_q))
    q2 = float(seam.quantile(mid_q))

    def _bin(v: float) -> str | None:
        if pd.isna(v):
            return None
        if v <= q1:
            return "near"
        if v <= q2:
            return "mid"
        return "far"

    work["seam_bin"] = seam.map(_bin)
    return work


def load_paths(paths_csv: Path) -> pd.DataFrame:
    paths = pd.read_csv(paths_csv).copy()
    required = {"path_id", "step", "r", "alpha"}
    missing = required - set(paths.columns)
    if missing:
        raise ValueError(f"Path CSV missing required columns: {sorted(missing)}")

    for col in ["step", "r", "alpha"]:
        paths[col] = pd.to_numeric(paths[col], errors="coerce")
    return paths.sort_values(["path_id", "step"]).reset_index(drop=True)


def annotate_path_nodes(paths: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c for c in [
            "node_id",
            "r",
            "alpha",
            "distance_to_seam",
            "seam_bin",
            "criticality",
            "signed_phase",
            "obstruction_mean_abs_holonomy",
            "obstruction_signed_sum_holonomy",
            "absolute_holonomy_node",
            "dew_angle_deg",
            "dew_sector",
        ]
        if c in nodes.columns
    ]
    ann = paths.merge(nodes[keep], on=["r", "alpha"], how="left")
    ann = ann.sort_values(["path_id", "step"]).reset_index(drop=True)

    jump_vals = []
    sector_change_vals = []
    prev_by_path: dict[object, tuple[float | None, object | None]] = {}

    for row in ann.itertuples(index=False):
        pid = row.path_id
        angle = getattr(row, "dew_angle_deg", np.nan)
        sector = getattr(row, "dew_sector", None)

        if pid not in prev_by_path:
            jump_vals.append(np.nan)
            sector_change_vals.append(0)
        else:
            prev_angle, prev_sector = prev_by_path[pid]
            if pd.isna(angle) or prev_angle is None or pd.isna(prev_angle):
                jump_vals.append(np.nan)
            else:
                jump_vals.append(circ_abs_diff_deg(float(angle), float(prev_angle)))
            sector_change_vals.append(int(str(sector) != str(prev_sector)))

        prev_by_path[pid] = (angle, sector)

    ann["path_angle_jump_deg"] = jump_vals
    ann["path_sector_change"] = sector_change_vals
    return ann


def summarize_paths(path_nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for pid, sub in path_nodes.groupby("path_id", dropna=False):
        seam_bins = sub["seam_bin"].dropna().astype(str)
        n = len(sub)

        row = {
            "path_id": pid,
            "n_nodes": int(n),
            "n_steps": int(max(0, n - 1)),
            "min_distance_to_seam": float(pd.to_numeric(sub["distance_to_seam"], errors="coerce").min()),
            "mean_distance_to_seam": float(pd.to_numeric(sub["distance_to_seam"], errors="coerce").mean()),
            "near_fraction": float((seam_bins == "near").mean()) if len(seam_bins) else np.nan,
            "mid_fraction": float((seam_bins == "mid").mean()) if len(seam_bins) else np.nan,
            "far_fraction": float((seam_bins == "far").mean()) if len(seam_bins) else np.nan,
            "mean_criticality": float(pd.to_numeric(sub["criticality"], errors="coerce").mean()),
            "max_criticality": float(pd.to_numeric(sub["criticality"], errors="coerce").max()),
            "mean_unsigned_obstruction": float(pd.to_numeric(sub["obstruction_mean_abs_holonomy"], errors="coerce").mean()),
            "max_unsigned_obstruction": float(pd.to_numeric(sub["obstruction_mean_abs_holonomy"], errors="coerce").max()),
            "mean_absolute_holonomy": float(pd.to_numeric(sub["absolute_holonomy_node"], errors="coerce").mean()),
            "mean_angle_jump_deg": float(pd.to_numeric(sub["path_angle_jump_deg"], errors="coerce").mean()),
            "max_angle_jump_deg": float(pd.to_numeric(sub["path_angle_jump_deg"], errors="coerce").max()),
            "n_sector_changes": int(pd.to_numeric(sub["path_sector_change"], errors="coerce").fillna(0).sum()),
        }

        min_seam = row["min_distance_to_seam"]
        if pd.isna(min_seam):
            seam_class = "unknown"
        elif row["near_fraction"] > 0:
            seam_class = "seam_contact"
        elif row["mid_fraction"] > 0:
            seam_class = "seam_grazing"
        else:
            seam_class = "seam_distant"

        row["seam_class"] = seam_class
        rows.append(row)

    return pd.DataFrame(rows)


def build_class_summary(path_summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seam_class",
        "min_distance_to_seam",
        "mean_criticality",
        "max_criticality",
        "mean_unsigned_obstruction",
        "mean_absolute_holonomy",
        "mean_angle_jump_deg",
        "max_angle_jump_deg",
        "n_sector_changes",
    ]
    work = path_summary[cols].copy()
    grouped = work.groupby("seam_class", dropna=False).agg(
        n_paths=("seam_class", "size"),
        mean_min_seam=("min_distance_to_seam", "mean"),
        mean_criticality=("mean_criticality", "mean"),
        mean_max_criticality=("max_criticality", "mean"),
        mean_unsigned_obstruction=("mean_unsigned_obstruction", "mean"),
        mean_absolute_holonomy=("mean_absolute_holonomy", "mean"),
        mean_angle_jump=("mean_angle_jump_deg", "mean"),
        mean_max_angle_jump=("max_angle_jump_deg", "mean"),
        mean_sector_changes=("n_sector_changes", "mean"),
    )
    return grouped.reset_index()


def summarize_text(path_summary: pd.DataFrame, class_summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Geodesic Path Diagnostics Summary ===")
    lines.append("")
    lines.append(f"n_paths = {len(path_summary)}")
    lines.append("")
    lines.append("Path-level correlations")
    lines.append(
        f"  corr(min_seam, mean_unsigned_obstruction) = "
        f"{_corr(path_summary, 'min_distance_to_seam', 'mean_unsigned_obstruction'):.4f}"
    )
    lines.append(
        f"  corr(min_seam, mean_angle_jump) = "
        f"{_corr(path_summary, 'min_distance_to_seam', 'mean_angle_jump_deg'):.4f}"
    )
    lines.append(
        f"  corr(min_seam, mean_criticality) = "
        f"{_corr(path_summary, 'min_distance_to_seam', 'mean_criticality'):.4f}"
    )
    lines.append(
        f"  corr(mean_angle_jump, mean_unsigned_obstruction) = "
        f"{_corr(path_summary, 'mean_angle_jump_deg', 'mean_unsigned_obstruction'):.4f}"
    )
    lines.append("")
    lines.append("Seam-class summary")
    for _, row in class_summary.iterrows():
        lines.append(
            f"  {row['seam_class']}: "
            f"n_paths={int(row['n_paths'])}, "
            f"mean_min_seam={row['mean_min_seam']:.4f}, "
            f"mean_criticality={row['mean_criticality']:.4f}, "
            f"mean_max_criticality={row['mean_max_criticality']:.4f}, "
            f"mean_unsigned_obstruction={row['mean_unsigned_obstruction']:.4f}, "
            f"mean_abs_holonomy={row['mean_absolute_holonomy']:.4f}, "
            f"mean_angle_jump={row['mean_angle_jump']:.4f}, "
            f"mean_max_angle_jump={row['mean_max_angle_jump']:.4f}, "
            f"mean_sector_changes={row['mean_sector_changes']:.4f}"
        )
    return "\n".join(lines)


def plot_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str) -> None:
    plot_df = df[[x, y]].dropna().copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df[x], plot_df[y], s=40)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate geodesic paths with seam/load/identity diagnostics.")
    parser.add_argument("--paths-csv", required=True, help="CSV with path_id, step, r, alpha")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="outputs/toy_geodesic_path_diagnostics")
    parser.add_argument("--near-q", type=float, default=0.33)
    parser.add_argument("--mid-q", type=float, default=0.66)
    args = parser.parse_args()

    cfg = Config(
        paths_csv=args.paths_csv,
        outputs_root=args.outputs_root,
        outdir=args.outdir,
        near_q=args.near_q,
        mid_q=args.mid_q,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_node_fields(Path(cfg.outputs_root))
    nodes = assign_seam_bins(nodes, cfg.near_q, cfg.mid_q)
    paths = load_paths(Path(cfg.paths_csv))
    path_nodes = annotate_path_nodes(paths, nodes)
    path_summary = summarize_paths(path_nodes)
    class_summary = build_class_summary(path_summary)

    path_nodes.to_csv(outdir / "path_node_diagnostics.csv", index=False)
    path_summary.to_csv(outdir / "path_diagnostics.csv", index=False)
    class_summary.to_csv(outdir / "seam_class_summary.csv", index=False)
    (outdir / "geodesic_path_diagnostics_summary.txt").write_text(
        summarize_text(path_summary, class_summary),
        encoding="utf-8",
    )

    plot_scatter(
        path_summary,
        "min_distance_to_seam",
        "n_nodes",
        outdir / "geodesic_min_seam_vs_length.png",
        "Path length vs minimum seam distance",
    )
    plot_scatter(
        path_summary,
        "mean_unsigned_obstruction",
        "mean_angle_jump_deg",
        outdir / "geodesic_load_vs_roughness.png",
        "Path load vs identity-angle roughness",
    )
    plot_scatter(
        path_summary,
        "near_fraction",
        "mean_criticality",
        outdir / "geodesic_criticality_vs_seam_fraction.png",
        "Criticality vs seam-contact fraction",
    )

    print("=== Geodesic Path Diagnostics ===")
    for name in [
        "path_node_diagnostics.csv",
        "path_diagnostics.csv",
        "seam_class_summary.csv",
        "geodesic_path_diagnostics_summary.txt",
        "geodesic_min_seam_vs_length.png",
        "geodesic_load_vs_roughness.png",
        "geodesic_criticality_vs_seam_fraction.png",
    ]:
        print(outdir / name)


if __name__ == "__main__":
    main()
