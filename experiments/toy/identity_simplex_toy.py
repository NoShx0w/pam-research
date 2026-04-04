#!/usr/bin/env python3
from __future__ import annotations

"""
identity_simplex_toy.py

Minimal simplex / ternary-style toy for PAM identity structure.

Purpose
-------
Test whether the current best transport-centered identity basis is more
naturally read as a 3-way balance structure than as a wheel with a single
radius.

Current basis
-------------
- absolute_holonomy_node
- obstruction_mean_abs_holonomy
- abs(obstruction_signed_sum_holonomy)

This first pass deliberately uses the magnitude of signed obstruction so the
simplex coordinates remain directly compositional.

Outputs
-------
- identity_simplex_nodes.csv
- identity_simplex_summary.txt
- identity_simplex_top_vertex1.csv
- identity_simplex_top_vertex2.csv
- identity_simplex_top_vertex3.csv
- identity_simplex_by_signed_phase.png
- identity_simplex_by_criticality.png
- identity_simplex_by_signed_obstruction.png

Interpretation
--------------
The simplex view asks whether nodes cluster near:
- vertex 1: absolute holonomy dominance
- vertex 2: unsigned obstruction dominance
- vertex 3: signed-obstruction magnitude dominance

This is a toy / study script, not a canonical pipeline stage.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimplexConfig:
    comp1: str = "absolute_holonomy_node"
    comp2: str = "obstruction_mean_abs_holonomy"
    comp3: str = "obstruction_signed_sum_holonomy"
    outputs_root: str = "outputs"
    outdir: str = "outputs/toy_identity_simplex"
    eps: float = 1e-12
    top_n: int = 12


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).copy() if path.exists() else pd.DataFrame()


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


def build_simplex_projection(df: pd.DataFrame, cfg: SimplexConfig) -> pd.DataFrame:
    required = [cfg.comp1, cfg.comp2, cfg.comp3]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required simplex component columns: {missing}")

    work = df.copy()

    c1 = pd.to_numeric(work[cfg.comp1], errors="coerce")
    c2 = pd.to_numeric(work[cfg.comp2], errors="coerce")
    c3_signed = pd.to_numeric(work[cfg.comp3], errors="coerce")
    c3 = np.abs(c3_signed)

    total = c1 + c2 + c3
    total = total.where(total > cfg.eps, np.nan)

    p1 = c1 / total
    p2 = c2 / total
    p3 = c3 / total

    # Equilateral simplex vertices:
    # v1=(0,0), v2=(1,0), v3=(0.5, sqrt(3)/2)
    x = p2 + 0.5 * p3
    y = (math.sqrt(3.0) / 2.0) * p3

    dominant = np.select(
        [
            (p1 >= p2) & (p1 >= p3),
            (p2 >= p1) & (p2 >= p3),
            (p3 >= p1) & (p3 >= p2),
        ],
        ["holonomy", "unsigned_obstruction", "signed_obstruction_mag"],
        default="mixed",
    )

    balance_entropy = -(p1 * np.log(np.clip(p1, cfg.eps, None)) + p2 * np.log(np.clip(p2, cfg.eps, None)) + p3 * np.log(np.clip(p3, cfg.eps, None)))

    work["simplex_comp1"] = c1
    work["simplex_comp2"] = c2
    work["simplex_comp3_abs"] = c3
    work["simplex_total"] = total
    work["simplex_p1_holonomy"] = p1
    work["simplex_p2_unsigned_obstruction"] = p2
    work["simplex_p3_signed_obstruction_mag"] = p3
    work["simplex_x"] = x
    work["simplex_y"] = y
    work["simplex_dominant_component"] = dominant
    work["simplex_balance_entropy"] = balance_entropy
    work["simplex_abs_signed_phase"] = np.abs(pd.to_numeric(work.get("signed_phase"), errors="coerce"))

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
    metrics["corr_p1_signed_phase_spearman"] = _corr(df, "simplex_p1_holonomy", "signed_phase")
    metrics["corr_p2_criticality_spearman"] = _corr(df, "simplex_p2_unsigned_obstruction", "criticality")
    metrics["corr_p3_signed_obstruction_spearman"] = _corr(df, "simplex_p3_signed_obstruction_mag", "obstruction_signed_sum_holonomy")
    metrics["corr_entropy_criticality_spearman"] = _corr(df, "simplex_balance_entropy", "criticality")
    metrics["corr_entropy_distance_to_seam_spearman"] = _corr(df, "simplex_balance_entropy", "distance_to_seam")
    return metrics


def build_dominance_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "simplex_dominant_component",
        "signed_phase",
        "criticality",
        "distance_to_seam",
        "obstruction_mean_abs_holonomy",
        "obstruction_signed_sum_holonomy",
    ]
    work = df[cols].copy()
    grouped = work.groupby("simplex_dominant_component", dropna=False).agg(
        n_nodes=("simplex_dominant_component", "size"),
        mean_signed_phase=("signed_phase", "mean"),
        mean_criticality=("criticality", "mean"),
        mean_distance_to_seam=("distance_to_seam", "mean"),
        mean_unsigned_obstruction=("obstruction_mean_abs_holonomy", "mean"),
        mean_signed_obstruction=("obstruction_signed_sum_holonomy", "mean"),
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


def _draw_simplex_outline(ax) -> None:
    verts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, math.sqrt(3.0) / 2.0],
        [0.0, 0.0],
    ])
    ax.plot(verts[:, 0], verts[:, 1], linewidth=1.2)
    ax.text(-0.03, -0.04, "Holonomy", fontsize=10)
    ax.text(0.95, -0.04, "Unsigned Obs.", fontsize=10, ha="right")
    ax.text(0.5, math.sqrt(3.0) / 2.0 + 0.03, "|Signed Obs.|", fontsize=10, ha="center")


def plot_simplex(df: pd.DataFrame, outpath: Path, color_col: str, title: str) -> None:
    plot_df = df.dropna(subset=["simplex_x", "simplex_y"]).copy()
    fig, ax = plt.subplots(figsize=(8, 7))

    if color_col in plot_df.columns:
        cvals = pd.to_numeric(plot_df[color_col], errors="coerce")
        valid = cvals.notna()
        if valid.any():
            sc = ax.scatter(plot_df.loc[valid, "simplex_x"], plot_df.loc[valid, "simplex_y"], c=cvals.loc[valid], s=45)
            fig.colorbar(sc, ax=ax, shrink=0.85)
        else:
            ax.scatter(plot_df["simplex_x"], plot_df["simplex_y"], s=45)
    else:
        ax.scatter(plot_df["simplex_x"], plot_df["simplex_y"], s=45)

    _draw_simplex_outline(ax)
    ax.set_title(title)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3.0) / 2.0 + 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def summarize(df: pd.DataFrame, cfg: SimplexConfig) -> str:
    metrics = compute_summary_metrics(df)
    dom = build_dominance_summary(df)

    lines: list[str] = []
    lines.append("=== Identity Simplex Toy Summary ===")
    lines.append("")
    lines.append("Basis")
    lines.append(f"  c1 = {cfg.comp1}")
    lines.append(f"  c2 = {cfg.comp2}")
    lines.append(f"  c3 = abs({cfg.comp3})")
    lines.append("")
    lines.append(f"n_nodes = {metrics['n_nodes']}")
    lines.append(f"corr(p1_holonomy, signed_phase) = {metrics['corr_p1_signed_phase_spearman']:.4f} (spearman)")
    lines.append(f"corr(p2_unsigned_obstruction, criticality) = {metrics['corr_p2_criticality_spearman']:.4f} (spearman)")
    lines.append(f"corr(p3_signed_obstruction_mag, signed_obstruction) = {metrics['corr_p3_signed_obstruction_spearman']:.4f} (spearman)")
    lines.append(f"corr(balance_entropy, criticality) = {metrics['corr_entropy_criticality_spearman']:.4f} (spearman)")
    lines.append(f"corr(balance_entropy, distance_to_seam) = {metrics['corr_entropy_distance_to_seam_spearman']:.4f} (spearman)")
    lines.append("")
    lines.append("Dominance summary")
    for _, row in dom.iterrows():
        lines.append(
            f"  {row['simplex_dominant_component']}: n={int(row['n_nodes'])}, "
            f"mean_signed_phase={float(row['mean_signed_phase']):.4f}, "
            f"mean_criticality={float(row['mean_criticality']):.4f}, "
            f"mean_distance_to_seam={float(row['mean_distance_to_seam']):.4f}, "
            f"mean_unsigned_obstruction={float(row['mean_unsigned_obstruction']):.4f}, "
            f"mean_signed_obstruction={float(row['mean_signed_obstruction']):.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal simplex / ternary-style toy for PAM identity structure.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="outputs/toy_identity_simplex")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    cfg = SimplexConfig(outputs_root=args.outputs_root, outdir=args.outdir, top_n=args.top_n)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(Path(cfg.outputs_root))
    simplex = build_simplex_projection(df, cfg)

    simplex.to_csv(outdir / "identity_simplex_nodes.csv", index=False)
    build_dominance_summary(simplex).to_csv(outdir / "identity_simplex_dominance_summary.csv", index=False)
    build_top_table(simplex, "simplex_p1_holonomy", cfg.top_n).to_csv(outdir / "identity_simplex_top_vertex1.csv", index=False)
    build_top_table(simplex, "simplex_p2_unsigned_obstruction", cfg.top_n).to_csv(outdir / "identity_simplex_top_vertex2.csv", index=False)
    build_top_table(simplex, "simplex_p3_signed_obstruction_mag", cfg.top_n).to_csv(outdir / "identity_simplex_top_vertex3.csv", index=False)

    plot_simplex(simplex, outdir / "identity_simplex_by_signed_phase.png", "signed_phase", "Identity Simplex — colored by signed phase")
    plot_simplex(simplex, outdir / "identity_simplex_by_criticality.png", "criticality", "Identity Simplex — colored by criticality")
    plot_simplex(simplex, outdir / "identity_simplex_by_signed_obstruction.png", "obstruction_signed_sum_holonomy", "Identity Simplex — colored by signed obstruction")

    (outdir / "identity_simplex_summary.txt").write_text(summarize(simplex, cfg), encoding="utf-8")

    print("=== Identity Simplex Toy ===")
    print(outdir / "identity_simplex_summary.txt")
    print(outdir / "identity_simplex_dominance_summary.csv")
    print(outdir / "identity_simplex_top_vertex1.csv")
    print(outdir / "identity_simplex_top_vertex2.csv")
    print(outdir / "identity_simplex_top_vertex3.csv")
    print(outdir / "identity_simplex_by_signed_phase.png")
    print(outdir / "identity_simplex_by_criticality.png")
    print(outdir / "identity_simplex_by_signed_obstruction.png")


if __name__ == "__main__":
    main()

