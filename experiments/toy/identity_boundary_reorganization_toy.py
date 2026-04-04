#!/usr/bin/env python3
from __future__ import annotations

"""
identity_boundary_reorganization_toy.py

Boundary-conditioned identity reorganization toy.

Goal
----
Test whether identity angle / sector structure reorganizes as the phase
boundary (seam) is approached.

Core objects
------------
- DEW angle from the current best transport-centered basis
- seam distance from phase layer
- sector occupancy by seam-distance bin
- angle-strength coupling by seam-distance bin
- local neighbor angle-jump roughness by seam-distance bin

Current best basis
------------------
- absolute_holonomy_node
- obstruction_mean_abs_holonomy
- obstruction_signed_sum_holonomy

Outputs
-------
- identity_boundary_nodes.csv
- identity_boundary_bin_summary.csv
- identity_boundary_sector_counts.csv
- identity_boundary_correlation_summary.csv
- identity_boundary_neighbor_jumps.csv
- identity_boundary_summary.txt
- identity_boundary_angle_histograms.png
- identity_boundary_sector_stacks.png
- identity_boundary_neighbor_jump_boxplot.png

Notes
-----
This is an exploratory study script, not a canonical pipeline stage.
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

ANGLE_AUX_COLS = [
    "obstruction_mean_abs_holonomy",
    "absolute_holonomy_node",
    "criticality",
    "obstruction_signed_sum_holonomy",
    "distance_to_seam",
]


@dataclass(frozen=True)
class Config:
    comp1: str = BEST_BASIS[0]
    comp2: str = BEST_BASIS[1]
    comp3: str = BEST_BASIS[2]
    outputs_root: str = "outputs"
    outdir: str = "outputs/toy_identity_boundary_reorganization"
    eps: float = 1e-12
    n_bins: int = 3
    seam_col: str = "distance_to_seam"


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
            df["node_id"] = (
                pd.to_numeric(df["node_id"], errors="coerce")
                .astype("Int64")
                .astype(str)
            )

    base_cols = [
        c
        for c in ["node_id", "i", "j", "r", "alpha", "identity_magnitude", "identity_spin"]
        if c in identity.columns
    ]
    merged = identity[base_cols].copy()

    if not obstruction.empty:
        keep = [
            c
            for c in [
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
            c
            for c in [
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


def build_angle_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    required = [cfg.comp1, cfg.comp2, cfg.comp3]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required basis columns: {missing}")

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

    work["dew_angle_deg"] = angle_deg
    work["dew_angle_rad"] = angle_rad
    work["dew_sector"] = pd.cut(
        angle_deg,
        bins=[-180, -120, -60, 0, 60, 120, 180],
        labels=["S1", "S2", "S3", "S4", "S5", "S6"],
        include_lowest=True,
    )
    work["dew_abs_signed_phase"] = np.abs(pd.to_numeric(work.get("signed_phase"), errors="coerce"))
    return work


def add_seam_bins(df: pd.DataFrame, seam_col: str, n_bins: int) -> pd.DataFrame:
    work = df.copy()
    seam = pd.to_numeric(work[seam_col], errors="coerce")
    valid = seam.dropna()

    if len(valid) < n_bins:
        work["seam_bin"] = pd.Series(["all"] * len(work), index=work.index, dtype="object")
        return work

    # Quantile-based bins for balanced groups.
    labels = ["near", "mid", "far"] if n_bins == 3 else [f"bin_{i+1}" for i in range(n_bins)]
    try:
        work["seam_bin"] = pd.qcut(seam, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        work["seam_bin"] = pd.Series(["all"] * len(work), index=work.index, dtype="object")

    # Ensure near means smallest seam distance.
    if n_bins == 3 and work["seam_bin"].dtype.name == "category":
        cats = list(work["seam_bin"].cat.categories)
        if cats != labels[: len(cats)]:
            mapping = {old: new for old, new in zip(cats, labels[: len(cats)])}
            work["seam_bin"] = work["seam_bin"].astype(str).map(mapping)

    return work


def circular_resultant_length_deg(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    theta = np.deg2rad(x.to_numpy())
    c = np.mean(np.cos(theta))
    s = np.mean(np.sin(theta))
    return float(np.sqrt(c * c + s * s))


def build_bin_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bin_name, sub in df.groupby("seam_bin", dropna=False):
        row = {
            "seam_bin": str(bin_name),
            "n_nodes": int(len(sub)),
            "mean_distance_to_seam": float(pd.to_numeric(sub["distance_to_seam"], errors="coerce").mean()),
            "mean_angle_deg": float(pd.to_numeric(sub["dew_angle_deg"], errors="coerce").mean()),
            "std_angle_deg": float(pd.to_numeric(sub["dew_angle_deg"], errors="coerce").std()),
            "angle_resultant_length": circular_resultant_length_deg(sub["dew_angle_deg"]),
            "mean_unsigned_obstruction": float(pd.to_numeric(sub["obstruction_mean_abs_holonomy"], errors="coerce").mean()),
            "mean_absolute_holonomy": float(pd.to_numeric(sub["absolute_holonomy_node"], errors="coerce").mean()),
            "mean_signed_obstruction": float(pd.to_numeric(sub["obstruction_signed_sum_holonomy"], errors="coerce").mean()),
            "mean_criticality": float(pd.to_numeric(sub["criticality"], errors="coerce").mean()),
            "mean_signed_phase": float(pd.to_numeric(sub["signed_phase"], errors="coerce").mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_sector_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["seam_bin", "dew_sector"], dropna=False)
        .size()
        .rename("n_nodes")
        .reset_index()
    )
    return counts


def build_correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bin_name, sub in df.groupby("seam_bin", dropna=False):
        row = {
            "seam_bin": str(bin_name),
            "n_nodes": int(len(sub)),
            "corr_angle_signed_obstruction": _corr(sub, "dew_angle_deg", "obstruction_signed_sum_holonomy"),
            "corr_angle_signed_phase": _corr(sub, "dew_angle_deg", "signed_phase"),
        }
        for col in ANGLE_AUX_COLS:
            row[f"corr_angle_{col}"] = _corr(sub, "dew_angle_deg", col)
        rows.append(row)
    return pd.DataFrame(rows)


def build_neighbor_jumps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neighbor-jump roughness over the lattice.

    For each node, compare angle to right/down neighbors where available.
    Assign jump records to the source node seam bin.
    """
    work = df.copy()
    key_cols = ["i", "j", "dew_angle_deg", "seam_bin", "r", "alpha", "distance_to_seam"]
    work = work[key_cols].dropna(subset=["i", "j", "dew_angle_deg"]).copy()
    work["i"] = pd.to_numeric(work["i"], errors="coerce").astype(int)
    work["j"] = pd.to_numeric(work["j"], errors="coerce").astype(int)

    lookup = {
        (int(row.i), int(row.j)): row
        for row in work.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []

    def circ_abs_diff_deg(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    for (i, j), row in lookup.items():
        for di, dj, direction in [(1, 0, "down"), (0, 1, "right")]:
            nbr = lookup.get((i + di, j + dj))
            if nbr is None:
                continue
            jump = circ_abs_diff_deg(float(row.dew_angle_deg), float(nbr.dew_angle_deg))
            rows.append(
                {
                    "src_i": i,
                    "src_j": j,
                    "direction": direction,
                    "seam_bin": str(row.seam_bin),
                    "src_distance_to_seam": float(row.distance_to_seam) if pd.notna(row.distance_to_seam) else np.nan,
                    "angle_jump_deg": jump,
                }
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, bin_summary: pd.DataFrame, corr_summary: pd.DataFrame, neighbor_jumps: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Identity Boundary Reorganization Summary ===")
    lines.append("")
    lines.append("Best basis")
    lines.append(f"  c1 = {BEST_BASIS[0]}")
    lines.append(f"  c2 = {BEST_BASIS[1]}")
    lines.append(f"  c3 = {BEST_BASIS[2]}")
    lines.append("")
    lines.append(f"n_nodes = {len(df)}")
    lines.append("")
    lines.append("Global correlations")
    lines.append(f"  corr(angle, signed_obstruction) = {_corr(df, 'dew_angle_deg', 'obstruction_signed_sum_holonomy'):.4f}")
    lines.append(f"  corr(angle, signed_phase) = {_corr(df, 'dew_angle_deg', 'signed_phase'):.4f}")
    lines.append(f"  corr(angle, distance_to_seam) = {_corr(df, 'dew_angle_deg', 'distance_to_seam'):.4f}")
    lines.append("")
    lines.append("Seam-bin summaries")
    for _, row in bin_summary.iterrows():
        lines.append(
            f"  {row['seam_bin']}: "
            f"n={int(row['n_nodes'])}, "
            f"mean_d={row['mean_distance_to_seam']:.4f}, "
            f"mean_angle={row['mean_angle_deg']:.4f}, "
            f"std_angle={row['std_angle_deg']:.4f}, "
            f"resultant={row['angle_resultant_length']:.4f}, "
            f"mean_unsigned_obs={row['mean_unsigned_obstruction']:.4f}, "
            f"mean_abs_hol={row['mean_absolute_holonomy']:.4f}, "
            f"mean_signed_obs={row['mean_signed_obstruction']:.4f}, "
            f"mean_criticality={row['mean_criticality']:.4f}, "
            f"mean_signed_phase={row['mean_signed_phase']:.4f}"
        )
    lines.append("")
    lines.append("Coupling by seam bin")
    for _, row in corr_summary.iterrows():
        def _fmt(name: str) -> str:
            value = row.get(name, float("nan"))
            try:
                return f"{float(value):.4f}"
            except Exception:
                return "nan"

        lines.append(
            f"  {row['seam_bin']}: "
            f"corr(angle,signed_obs)={_fmt('corr_angle_signed_obstruction')}, "
            f"corr(angle,signed_phase)={_fmt('corr_angle_signed_phase')}, "
            f"corr(angle,unsigned_obs)={_fmt('corr_angle_obstruction_mean_abs_holonomy')}, "
            f"corr(angle,abs_hol)={_fmt('corr_angle_absolute_holonomy_node')}, "
            f"corr(angle,criticality)={_fmt('corr_angle_criticality')}, "
            f"corr(angle,distance_to_seam)={_fmt('corr_angle_distance_to_seam')}"
        )
    lines.append("")
    if not neighbor_jumps.empty:
        jump_stats = (
            neighbor_jumps.groupby("seam_bin")["angle_jump_deg"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )
        lines.append("Neighbor angle-jump roughness")
        for _, row in jump_stats.iterrows():
            lines.append(
                f"  {row['seam_bin']}: "
                f"n={int(row['count'])}, "
                f"mean_jump={row['mean']:.4f}, "
                f"median_jump={row['median']:.4f}, "
                f"std_jump={row['std']:.4f}"
            )
    return "\n".join(lines)


def plot_angle_histograms(df: pd.DataFrame, outpath: Path) -> None:
    bins = np.linspace(-180, 180, 19)
    fig, ax = plt.subplots(figsize=(9, 6))
    for bin_name, sub in df.groupby("seam_bin", dropna=False):
        vals = pd.to_numeric(sub["dew_angle_deg"], errors="coerce").dropna()
        if len(vals):
            ax.hist(vals, bins=bins, alpha=0.5, label=str(bin_name))
    ax.set_title("Identity angle histograms by seam-distance bin")
    ax.set_xlabel("DEW angle (deg)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_sector_stacks(sector_counts: pd.DataFrame, outpath: Path) -> None:
    pivot = sector_counts.pivot(index="seam_bin", columns="dew_sector", values="n_nodes").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for col in pivot.columns:
        vals = pivot[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=str(col))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pivot.index])
    ax.set_title("Sector occupancy by seam-distance bin")
    ax.set_xlabel("seam bin")
    ax.set_ylabel("n nodes")
    ax.legend(title="sector")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_neighbor_jump_boxplot(neighbor_jumps: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    groups = []
    labels = []
    for bin_name, sub in neighbor_jumps.groupby("seam_bin", dropna=False):
        vals = pd.to_numeric(sub["angle_jump_deg"], errors="coerce").dropna()
        if len(vals):
            groups.append(vals.to_numpy())
            labels.append(str(bin_name))
    if groups:
        ax.boxplot(groups, labels=labels, showfliers=True)
    ax.set_title("Neighbor angle jumps by seam-distance bin")
    ax.set_xlabel("seam bin")
    ax.set_ylabel("abs circular angle jump (deg)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary-conditioned identity reorganization toy.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="outputs/toy_identity_boundary_reorganization")
    parser.add_argument("--n-bins", type=int, default=3)
    args = parser.parse_args()

    cfg = Config(outputs_root=args.outputs_root, outdir=args.outdir, n_bins=args.n_bins)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(Path(cfg.outputs_root))
    angle_df = build_angle_table(df, cfg)
    angle_df = add_seam_bins(angle_df, cfg.seam_col, cfg.n_bins)

    bin_summary = build_bin_summary(angle_df)
    sector_counts = build_sector_counts(angle_df)
    corr_summary = build_correlation_summary(angle_df)
    neighbor_jumps = build_neighbor_jumps(angle_df)

    angle_df.to_csv(outdir / "identity_boundary_nodes.csv", index=False)
    bin_summary.to_csv(outdir / "identity_boundary_bin_summary.csv", index=False)
    sector_counts.to_csv(outdir / "identity_boundary_sector_counts.csv", index=False)
    corr_summary.to_csv(outdir / "identity_boundary_correlation_summary.csv", index=False)
    neighbor_jumps.to_csv(outdir / "identity_boundary_neighbor_jumps.csv", index=False)

    summary_text = summarize(angle_df, bin_summary, corr_summary, neighbor_jumps)
    (outdir / "identity_boundary_summary.txt").write_text(summary_text, encoding="utf-8")

    plot_angle_histograms(angle_df, outdir / "identity_boundary_angle_histograms.png")
    plot_sector_stacks(sector_counts, outdir / "identity_boundary_sector_stacks.png")
    if not neighbor_jumps.empty:
        plot_neighbor_jump_boxplot(neighbor_jumps, outdir / "identity_boundary_neighbor_jump_boxplot.png")

    print("=== Identity Boundary Reorganization Toy ===")
    for name in [
        "identity_boundary_summary.txt",
        "identity_boundary_bin_summary.csv",
        "identity_boundary_sector_counts.csv",
        "identity_boundary_correlation_summary.csv",
        "identity_boundary_neighbor_jumps.csv",
        "identity_boundary_angle_histograms.png",
        "identity_boundary_sector_stacks.png",
        "identity_boundary_neighbor_jump_boxplot.png",
    ]:
        path = outdir / name
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
