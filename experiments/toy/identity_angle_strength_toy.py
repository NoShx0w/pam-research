#!/usr/bin/env python3
from __future__ import annotations

"""
identity_angle_strength_toy.py

Minimal toy for PAM identity structure using:

- x-axis: DEW angle from the current best transport-centered basis
- y-axis: independently chosen strength axis

Goal
----
Keep the one compressed coordinate that has clearly worked (DEW angle),
and replace the ambiguous DEW radius with explicit strength variables.

Current best basis
------------------
- absolute_holonomy_node
- obstruction_mean_abs_holonomy
- obstruction_signed_sum_holonomy

Outputs
-------
- identity_angle_strength_nodes.csv
- identity_angle_strength_summary.txt
- identity_angle_strength_obstruction_abs.png
- identity_angle_strength_absolute_holonomy.png
- identity_angle_strength_criticality.png
- identity_angle_strength_distance_to_seam.png
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

STRENGTH_AXES = {
    "obstruction_abs": "obstruction_mean_abs_holonomy",
    "absolute_holonomy": "absolute_holonomy_node",
    "criticality": "criticality",
    "distance_to_seam": "distance_to_seam",
}


@dataclass(frozen=True)
class Config:
    comp1: str = BEST_BASIS[0]
    comp2: str = BEST_BASIS[1]
    comp3: str = BEST_BASIS[2]
    outputs_root: str = "outputs"
    outdir: str = "outputs/toy_identity_angle_strength"
    eps: float = 1e-12
    annotate_top_n: int = 10


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


def build_angle_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    required = [cfg.comp1, cfg.comp2, cfg.comp3]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required DEW basis columns: {missing}")

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

    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)

    work["dew_comp1_name"] = cfg.comp1
    work["dew_comp2_name"] = cfg.comp2
    work["dew_comp3_name"] = cfg.comp3
    work["dew_comp1_z"] = z1
    work["dew_comp2_z"] = z2
    work["dew_comp3_z"] = z3
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


def build_summary_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"n_nodes": int(len(df))}
    metrics["corr_angle_signed_obstruction_spearman"] = _corr(df, "dew_angle_deg", "obstruction_signed_sum_holonomy")
    metrics["corr_angle_signed_phase_spearman"] = _corr(df, "dew_angle_deg", "signed_phase")

    for label, col in STRENGTH_AXES.items():
        metrics[f"corr_angle_{label}_spearman"] = _corr(df, "dew_angle_deg", col)
        metrics[f"corr_{label}_signed_obstruction_spearman"] = _corr(df, col, "obstruction_signed_sum_holonomy")
        metrics[f"corr_{label}_signed_phase_spearman"] = _corr(df, col, "signed_phase")

    return metrics


def build_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dew_sector",
        "signed_phase",
        "criticality",
        "distance_to_seam",
        "absolute_holonomy_node",
        "obstruction_mean_abs_holonomy",
        "obstruction_signed_sum_holonomy",
    ]
    work = df[cols].copy()
    grouped = work.groupby("dew_sector", dropna=False).agg(
        n_nodes=("dew_sector", "size"),
        mean_signed_phase=("signed_phase", "mean"),
        mean_criticality=("criticality", "mean"),
        mean_distance_to_seam=("distance_to_seam", "mean"),
        mean_absolute_holonomy=("absolute_holonomy_node", "mean"),
        mean_unsigned_obstruction=("obstruction_mean_abs_holonomy", "mean"),
        mean_signed_obstruction=("obstruction_signed_sum_holonomy", "mean"),
    )
    return grouped.reset_index()


def _annotate_top(ax, df: pd.DataFrame, value_col: str, n: int, label_col: str = "node_id") -> None:
    sub = df.dropna(subset=["dew_angle_deg", value_col, label_col]).copy()
    if sub.empty:
        return
    sub = sub.assign(_abs=np.abs(pd.to_numeric(sub[value_col], errors="coerce")))
    sub = sub.sort_values("_abs", ascending=False).head(n)
    for _, row in sub.iterrows():
        ax.annotate(
            str(row[label_col]),
            (row["dew_angle_deg"], row[value_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )


def plot_angle_strength(
    df: pd.DataFrame,
    outpath: Path,
    strength_col: str,
    title: str,
    color_col: str = "obstruction_signed_sum_holonomy",
    annotate_top_n: int = 0,
) -> None:
    plot_df = df.dropna(subset=["dew_angle_deg", strength_col]).copy()
    fig, ax = plt.subplots(figsize=(9, 6))

    if color_col in plot_df.columns:
        cvals = pd.to_numeric(plot_df[color_col], errors="coerce")
        valid = cvals.notna()
        if valid.any():
            sc = ax.scatter(
                plot_df.loc[valid, "dew_angle_deg"],
                plot_df.loc[valid, strength_col],
                c=cvals.loc[valid],
                s=45,
            )
            fig.colorbar(sc, ax=ax, shrink=0.85)
        else:
            ax.scatter(plot_df["dew_angle_deg"], plot_df[strength_col], s=45)
    else:
        ax.scatter(plot_df["dew_angle_deg"], plot_df[strength_col], s=45)

    for deg in (-120, -60, 0, 60, 120):
        ax.axvline(deg, linewidth=1.0, alpha=0.35)

    if annotate_top_n > 0:
        _annotate_top(ax, plot_df, strength_col, annotate_top_n)

    ax.set_title(title)
    ax.set_xlabel("DEW angle (deg)")
    ax.set_ylabel(strength_col)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def summarize(df: pd.DataFrame) -> str:
    metrics = build_summary_metrics(df)
    sector = build_sector_summary(df)

    lines: list[str] = []
    lines.append("=== Identity Angle + Strength Toy Summary ===")
    lines.append("")
    lines.append("Best basis")
    lines.append(f"  c1 = {BEST_BASIS[0]}")
    lines.append(f"  c2 = {BEST_BASIS[1]}")
    lines.append(f"  c3 = {BEST_BASIS[2]}")
    lines.append("")
    lines.append(f"n_nodes = {metrics['n_nodes']}")
    lines.append(f"corr(angle, signed_obstruction) = {metrics['corr_angle_signed_obstruction_spearman']:.4f} (spearman)")
    lines.append(f"corr(angle, signed_phase) = {metrics['corr_angle_signed_phase_spearman']:.4f} (spearman)")
    lines.append("")
    lines.append("Angle vs strength-axis correlations")
    for label in STRENGTH_AXES:
        lines.append(
            f"  angle vs {label} = {metrics[f'corr_angle_{label}_spearman']:.4f}"
        )
    lines.append("")
    lines.append("Strength-axis alignment summaries")
    for label in STRENGTH_AXES:
        lines.append(
            f"  {label} vs signed_obstruction = "
            f"{metrics[f'corr_{label}_signed_obstruction_spearman']:.4f}"
        )
        lines.append(
            f"  {label} vs signed_phase = "
            f"{metrics[f'corr_{label}_signed_phase_spearman']:.4f}"
        )
    lines.append("")
    lines.append("Sector summary")
    for _, row in sector.iterrows():
        lines.append(
            f"  {row['dew_sector']}: "
            f"n={int(row['n_nodes'])}, "
            f"mean_signed_phase={float(row['mean_signed_phase']):.4f}, "
            f"mean_criticality={float(row['mean_criticality']):.4f}, "
            f"mean_distance_to_seam={float(row['mean_distance_to_seam']):.4f}, "
            f"mean_absolute_holonomy={float(row['mean_absolute_holonomy']):.4f}, "
            f"mean_unsigned_obstruction={float(row['mean_unsigned_obstruction']):.4f}, "
            f"mean_signed_obstruction={float(row['mean_signed_obstruction']):.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal angle + independent strength toy for PAM identity structure.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="outputs/toy_identity_angle_strength")
    parser.add_argument("--annotate-top-n", type=int, default=10)
    args = parser.parse_args()

    cfg = Config(outputs_root=args.outputs_root, outdir=args.outdir, annotate_top_n=args.annotate_top_n)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(Path(cfg.outputs_root))
    angle_df = build_angle_table(df, cfg)

    angle_df.to_csv(outdir / "identity_angle_strength_nodes.csv", index=False)
    build_sector_summary(angle_df).to_csv(outdir / "identity_angle_strength_sector_summary.csv", index=False)
    (outdir / "identity_angle_strength_summary.txt").write_text(summarize(angle_df), encoding="utf-8")

    plot_angle_strength(
        angle_df,
        outdir / "identity_angle_strength_obstruction_abs.png",
        "obstruction_mean_abs_holonomy",
        "Identity angle + strength — obstruction magnitude",
        annotate_top_n=cfg.annotate_top_n,
    )
    plot_angle_strength(
        angle_df,
        outdir / "identity_angle_strength_absolute_holonomy.png",
        "absolute_holonomy_node",
        "Identity angle + strength — absolute holonomy",
        annotate_top_n=cfg.annotate_top_n,
    )
    plot_angle_strength(
        angle_df,
        outdir / "identity_angle_strength_criticality.png",
        "criticality",
        "Identity angle + strength — criticality",
        annotate_top_n=0,
    )
    plot_angle_strength(
        angle_df,
        outdir / "identity_angle_strength_distance_to_seam.png",
        "distance_to_seam",
        "Identity angle + strength — distance to seam",
        annotate_top_n=0,
    )

    print("=== Identity Angle + Strength Toy ===")
    print(outdir / "identity_angle_strength_summary.txt")
    print(outdir / "identity_angle_strength_sector_summary.csv")
    print(outdir / "identity_angle_strength_obstruction_abs.png")
    print(outdir / "identity_angle_strength_absolute_holonomy.png")
    print(outdir / "identity_angle_strength_criticality.png")
    print(outdir / "identity_angle_strength_distance_to_seam.png")


if __name__ == "__main__":
    main()
