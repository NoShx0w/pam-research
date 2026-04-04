#!/usr/bin/env python3
from __future__ import annotations

"""
identity_dew_toy.py

Minimal DEW-style interpretive toy for PAM identity structure.

Current focus
-------------
This version fixes the DEW basis to the current best basis:

    c1 = absolute_holonomy_node
    c2 = obstruction_mean_abs_holonomy
    c3 = obstruction_signed_sum_holonomy

and shifts from basis search to interpretation.

Outputs
-------
- DEW node table with angle + two radius variants
- sector summary table
- top-node tables for both radius variants
- scatter plots colored by signed obstruction and sized by each radius
- summary text describing the current DEW readout

This remains a toy / study script, not a canonical pipeline stage.
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
class DewConfig:
    comp1: str = BEST_BASIS[0]
    comp2: str = BEST_BASIS[1]
    comp3: str = BEST_BASIS[2]
    outputs_root: str = "outputs"
    outdir: str = "outputs/toy_identity_dew_best_basis"
    eps: float = 1e-12
    annotate_top_n: int = 8
    top_n: int = 12


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).copy() if path.exists() else pd.DataFrame()


def _zscore(series: pd.Series, eps: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean(skipna=True)) if len(x.dropna()) else 0.0
    sd = float(x.std(skipna=True)) if len(x.dropna()) else 0.0
    return (x - mu) / max(sd, eps)


def load_inputs(outputs_root: Path) -> pd.DataFrame:
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

    for df in (identity, obstruction, obstruction_signed):
        if not df.empty and "node_id" in df.columns:
            df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("Int64").astype(str)

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
            ]
            if c in obstruction.columns
        ]
        if keep:
            merged = merged.merge(obstruction[keep], on="node_id", how="left")

    if not obstruction_signed.empty:
        keep = [
            c for c in [
                "node_id",
                "obstruction_signed_sum_holonomy",
                "obstruction_signed_weighted_holonomy",
            ]
            if c in obstruction_signed.columns
        ]
        if keep:
            merged = merged.merge(obstruction_signed[keep], on="node_id", how="left")

    if "obstruction_mean_abs_holonomy" in merged.columns:
        merged["absolute_holonomy_node"] = merged["obstruction_mean_abs_holonomy"]
    else:
        merged["absolute_holonomy_node"] = np.nan

    if not phase.empty:
        for col in ("r", "alpha", "signed_phase"):
            if col in phase.columns:
                phase[col] = pd.to_numeric(phase[col], errors="coerce")
        keep = [c for c in ["r", "alpha", "signed_phase"] if c in phase.columns]
        if keep:
            merged = merged.merge(phase[keep], on=["r", "alpha"], how="left")

    if not phase_distance.empty:
        for col in ("r", "alpha", "distance_to_seam"):
            if col in phase_distance.columns:
                phase_distance[col] = pd.to_numeric(phase_distance[col], errors="coerce")
        keep = [c for c in ["r", "alpha", "distance_to_seam"] if c in phase_distance.columns]
        if keep:
            merged = merged.merge(phase_distance[keep], on=["r", "alpha"], how="left")

    if not criticality.empty:
        for col in ("r", "alpha", "criticality"):
            if col in criticality.columns:
                criticality[col] = pd.to_numeric(criticality[col], errors="coerce")
        keep = [c for c in ["r", "alpha", "criticality"] if c in criticality.columns]
        if keep:
            merged = merged.merge(criticality[keep], on=["r", "alpha"], how="left")

    for col in merged.columns:
        if col != "node_id":
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged.sort_values(["r", "alpha"]).reset_index(drop=True)


def build_dew_projection(df: pd.DataFrame, cfg: DewConfig) -> pd.DataFrame:
    required = [cfg.comp1, cfg.comp2, cfg.comp3]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required DEW component columns: {missing}")

    work = df.copy()

    z1 = _zscore(work[cfg.comp1], cfg.eps)
    z2 = _zscore(work[cfg.comp2], cfg.eps)
    z3 = _zscore(work[cfg.comp3], cfg.eps)

    mean_z = (z1 + z2 + z3) / 3.0
    d1 = z1 - mean_z
    d2 = z2 - mean_z
    d3 = z3 - mean_z

    x = 0.5 * (2.0 * d1 - d2 - d3)
    y = (math.sqrt(3.0) / 2.0) * (d2 - d3)

    radius_dev = np.sqrt((d1**2 + d2**2 + d3**2) / 2.0)
    radius_raw = np.sqrt(z1**2 + z2**2 + z3**2)
    radius_magaware = np.sqrt(z1**2 + z2**2 + np.abs(z3)**2)

    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)

    work["dew_comp1_name"] = cfg.comp1
    work["dew_comp2_name"] = cfg.comp2
    work["dew_comp3_name"] = cfg.comp3
    work["dew_comp1_z"] = z1
    work["dew_comp2_z"] = z2
    work["dew_comp3_z"] = z3
    work["dew_dev1"] = d1
    work["dew_dev2"] = d2
    work["dew_dev3"] = d3
    work["dew_x"] = x
    work["dew_y"] = y
    work["dew_radius_dev"] = radius_dev
    work["dew_radius_raw"] = radius_raw
    work["dew_radius_magaware"] = radius_magaware
    work["dew_angle_rad"] = angle_rad
    work["dew_angle_deg"] = angle_deg
    work["dew_abs_signed_phase"] = np.abs(pd.to_numeric(work.get("signed_phase"), errors="coerce"))
    work["dew_sector"] = pd.cut(
        angle_deg,
        bins=[-180, -120, -60, 0, 60, 120, 180],
        labels=["S1", "S2", "S3", "S4", "S5", "S6"],
        include_lowest=True,
    )
    return work


def _corr(df: pd.DataFrame, x: str, y: str) -> float:
    if x not in df.columns or y not in df.columns:
        return float("nan")
    work = df[[x, y]].dropna()
    if len(work) < 3:
        return float("nan")
    return float(work[x].corr(work[y], method="spearman"))


def compute_summary_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"n_nodes": int(len(df))}
    metrics["corr_angle_signed_phase_spearman"] = _corr(df, "dew_angle_deg", "signed_phase")
    metrics["corr_angle_signed_obstruction_spearman"] = _corr(df, "dew_angle_deg", "obstruction_signed_sum_holonomy")

    for radius_col in ["dew_radius_dev", "dew_radius_raw", "dew_radius_magaware"]:
        metrics[f"corr_{radius_col}_criticality_spearman"] = _corr(df, radius_col, "criticality")
        metrics[f"corr_{radius_col}_distance_to_seam_spearman"] = _corr(df, radius_col, "distance_to_seam")
        metrics[f"corr_{radius_col}_abs_signed_phase_spearman"] = _corr(df, radius_col, "dew_abs_signed_phase")
        metrics[f"corr_{radius_col}_obs_abs_spearman"] = _corr(df, radius_col, "obstruction_mean_abs_holonomy")
    return metrics


def build_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    cols = [
        "dew_sector",
        "obstruction_signed_sum_holonomy",
        "signed_phase",
        "criticality",
        "dew_radius_dev",
        "dew_radius_raw",
    ]
    work = work[cols].copy()
    grouped = work.groupby("dew_sector", dropna=False).agg(
        n_nodes=("dew_sector", "size"),
        mean_signed_obstruction=("obstruction_signed_sum_holonomy", "mean"),
        mean_signed_phase=("signed_phase", "mean"),
        mean_criticality=("criticality", "mean"),
        mean_radius_dev=("dew_radius_dev", "mean"),
        mean_radius_raw=("dew_radius_raw", "mean"),
    )
    return grouped.reset_index()


def build_top_table(df: pd.DataFrame, value_col: str, top_n: int) -> pd.DataFrame:
    cols = [c for c in [
        "node_id", "r", "alpha", value_col, "criticality", "distance_to_seam",
        "signed_phase", "obstruction_mean_abs_holonomy", "obstruction_signed_sum_holonomy"
    ] if c in df.columns]
    work = df[cols].dropna(subset=[value_col]).copy()
    work = work.sort_values(value_col, ascending=False).head(top_n).reset_index(drop=True)
    work.insert(0, "rank", np.arange(1, len(work) + 1))
    return work


def _annotate_top(ax, df: pd.DataFrame, value_col: str, n: int, label_col: str = "node_id") -> None:
    if value_col not in df.columns or label_col not in df.columns:
        return
    sub = df.dropna(subset=["dew_x", "dew_y", value_col, label_col]).copy()
    if sub.empty:
        return
    sub = sub.sort_values(value_col, ascending=False).head(n)
    for _, row in sub.iterrows():
        ax.annotate(
            str(row[label_col]),
            (row["dew_x"], row["dew_y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )


def plot_obstruction_with_radius(df: pd.DataFrame, outpath: Path, radius_col: str, title: str, annotate_top_n: int = 0) -> None:
    plot_df = df.dropna(subset=["dew_x", "dew_y", radius_col, "obstruction_signed_sum_holonomy"]).copy()
    fig, ax = plt.subplots(figsize=(8, 7))

    cvals = pd.to_numeric(plot_df["obstruction_signed_sum_holonomy"], errors="coerce")
    rvals = pd.to_numeric(plot_df[radius_col], errors="coerce")
    rmin = float(rvals.min()) if len(rvals) else 0.0
    rmax = float(rvals.max()) if len(rvals) else 1.0
    span = max(rmax - rmin, 1e-12)
    sizes = 30.0 + 120.0 * ((rvals - rmin) / span)

    sc = ax.scatter(plot_df["dew_x"], plot_df["dew_y"], c=cvals, s=sizes)
    fig.colorbar(sc, ax=ax, shrink=0.85)

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1.0, alpha=0.35)

    if annotate_top_n > 0:
        _annotate_top(ax, plot_df, radius_col, annotate_top_n)

    ax.set_title(title)
    ax.set_xlabel("DEW x")
    ax.set_ylabel("DEW y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def summarize(df: pd.DataFrame, cfg: DewConfig) -> str:
    metrics = compute_summary_metrics(df)
    lines: list[str] = []
    lines.append("=== Identity DEW Best-Basis Summary ===")
    lines.append("")
    lines.append("Basis")
    lines.append(f"  c1 = {cfg.comp1}")
    lines.append(f"  c2 = {cfg.comp2}")
    lines.append(f"  c3 = {cfg.comp3}")
    lines.append("")
    lines.append(f"n_nodes = {metrics['n_nodes']}")
    lines.append(
        f"corr(dew_angle_deg, signed_obstruction) = {metrics['corr_angle_signed_obstruction_spearman']:.4f} (spearman)"
    )
    lines.append(
        f"corr(dew_angle_deg, signed_phase) = {metrics['corr_angle_signed_phase_spearman']:.4f} (spearman)"
    )
    lines.append("")
    lines.append("Radius correlations")
    for radius_col in ["dew_radius_dev", "dew_radius_raw", "dew_radius_magaware"]:
        lines.append(f"  {radius_col}")
        lines.append(f"    criticality = {metrics[f'corr_{radius_col}_criticality_spearman']:.4f}")
        lines.append(f"    distance_to_seam = {metrics[f'corr_{radius_col}_distance_to_seam_spearman']:.4f}")
        lines.append(f"    abs(signed_phase) = {metrics[f'corr_{radius_col}_abs_signed_phase_spearman']:.4f}")
        lines.append(f"    obstruction_mean_abs_holonomy = {metrics[f'corr_{radius_col}_obs_abs_spearman']:.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpretive DEW toy for the current best PAM identity basis.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="outputs/toy_identity_dew_best_basis")
    parser.add_argument("--annotate-top-n", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    cfg = DewConfig(
        outputs_root=args.outputs_root,
        outdir=args.outdir,
        annotate_top_n=args.annotate_top_n,
        top_n=args.top_n,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(Path(cfg.outputs_root))
    dew = build_dew_projection(df, cfg)

    dew.to_csv(outdir / "identity_dew_nodes.csv", index=False)
    build_sector_summary(dew).to_csv(outdir / "identity_dew_sector_summary.csv", index=False)
    build_top_table(dew, "dew_radius_dev", cfg.top_n).to_csv(outdir / "identity_dew_top_radius_dev.csv", index=False)
    build_top_table(dew, "dew_radius_raw", cfg.top_n).to_csv(outdir / "identity_dew_top_radius_raw.csv", index=False)

    plot_obstruction_with_radius(
        dew,
        outdir / "identity_dew_signed_obstruction_size_radius_dev.png",
        "dew_radius_dev",
        "Identity DEW — signed obstruction, size = deviatoric radius",
        annotate_top_n=cfg.annotate_top_n,
    )
    plot_obstruction_with_radius(
        dew,
        outdir / "identity_dew_signed_obstruction_size_radius_raw.png",
        "dew_radius_raw",
        "Identity DEW — signed obstruction, size = raw radius",
        annotate_top_n=cfg.annotate_top_n,
    )

    (outdir / "identity_dew_summary.txt").write_text(summarize(dew, cfg), encoding="utf-8")

    print("=== Identity DEW Best-Basis Pass ===")
    print(outdir / "identity_dew_summary.txt")
    print(outdir / "identity_dew_sector_summary.csv")
    print(outdir / "identity_dew_top_radius_dev.csv")
    print(outdir / "identity_dew_top_radius_raw.csv")
    print(outdir / "identity_dew_signed_obstruction_size_radius_dev.png")
    print(outdir / "identity_dew_signed_obstruction_size_radius_raw.png")


if __name__ == "__main__":
    main()
