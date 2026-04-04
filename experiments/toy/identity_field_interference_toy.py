#!/usr/bin/env python3
from __future__ import annotations

"""
identity_field_interference_toy.py

Cross-layer interference toy for PAM.

Goal
----
Construct a simple local interference score from:

- identity-angle change
- seam-distance change
- transport magnitude

and test whether high-interference regions align with:
- criticality
- seam proximity
- local angle roughness

Interpretation
--------------
This is a first operational pass on the "moiré-like" / interference idea.

The score is intentionally simple:
    I = z(local_angle_jump) + z(local_seam_jump) + z(local_transport_load)

This is not yet a canonical field-theoretic object.
It is a low-friction diagnostic for cross-layer mismatch.

Current best identity basis
---------------------------
- absolute_holonomy_node
- obstruction_mean_abs_holonomy
- obstruction_signed_sum_holonomy

Outputs
-------
- identity_interference_nodes.csv
- identity_interference_edges.csv
- identity_interference_summary.txt
- identity_interference_top_nodes.csv
- identity_interference_vs_criticality.png
- identity_interference_vs_distance_to_seam.png
- identity_interference_vs_angle_jump.png
- identity_interference_lattice.png

Optional follow-up
------------------
If this produces a real signal, it becomes the bridge back to geodesics:
geodesics through high-interference zones.
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
    comp1: str = BEST_BASIS[0]
    comp2: str = BEST_BASIS[1]
    comp3: str = BEST_BASIS[2]
    outputs_root: str = "outputs"
    outdir: str = "outputs/toy_identity_field_interference"
    eps: float = 1e-12
    top_n: int = 15


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
    return work


def circ_abs_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def build_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build right/down neighbor edge table with local mismatch quantities.
    """
    work = df.copy()
    key_cols = [
        "node_id",
        "i",
        "j",
        "r",
        "alpha",
        "dew_angle_deg",
        "distance_to_seam",
        "obstruction_mean_abs_holonomy",
        "criticality",
        "signed_phase",
    ]
    work = work[key_cols].dropna(subset=["i", "j", "dew_angle_deg"]).copy()
    work["i"] = pd.to_numeric(work["i"], errors="coerce").astype(int)
    work["j"] = pd.to_numeric(work["j"], errors="coerce").astype(int)

    lookup = {(int(r.i), int(r.j)): r for r in work.itertuples(index=False)}
    rows: list[dict[str, object]] = []

    for (i, j), row in lookup.items():
        for di, dj, direction in [(1, 0, "down"), (0, 1, "right")]:
            nbr = lookup.get((i + di, j + dj))
            if nbr is None:
                continue

            angle_jump = circ_abs_diff_deg(float(row.dew_angle_deg), float(nbr.dew_angle_deg))

            seam_a = float(row.distance_to_seam) if pd.notna(row.distance_to_seam) else np.nan
            seam_b = float(nbr.distance_to_seam) if pd.notna(nbr.distance_to_seam) else np.nan
            seam_jump = abs(seam_a - seam_b) if pd.notna(seam_a) and pd.notna(seam_b) else np.nan

            obs_a = float(row.obstruction_mean_abs_holonomy) if pd.notna(row.obstruction_mean_abs_holonomy) else np.nan
            obs_b = float(nbr.obstruction_mean_abs_holonomy) if pd.notna(nbr.obstruction_mean_abs_holonomy) else np.nan
            obs_local = np.nanmean([obs_a, obs_b]) if pd.notna(obs_a) or pd.notna(obs_b) else np.nan

            crit_a = float(row.criticality) if pd.notna(row.criticality) else np.nan
            crit_b = float(nbr.criticality) if pd.notna(nbr.criticality) else np.nan
            crit_local = np.nanmean([crit_a, crit_b]) if pd.notna(crit_a) or pd.notna(crit_b) else np.nan

            rows.append(
                {
                    "src_node_id": row.node_id,
                    "src_i": i,
                    "src_j": j,
                    "direction": direction,
                    "angle_jump_deg": angle_jump,
                    "seam_jump": seam_jump,
                    "obs_local": obs_local,
                    "crit_local": crit_local,
                }
            )

    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges

    edges["z_angle_jump"] = _zscore(edges["angle_jump_deg"], 1e-12)
    edges["z_seam_jump"] = _zscore(edges["seam_jump"], 1e-12)
    edges["z_obs_local"] = _zscore(edges["obs_local"], 1e-12)

    # Main simple interference score.
    edges["interference_score"] = (
        edges["z_angle_jump"].fillna(0.0)
        + edges["z_seam_jump"].fillna(0.0)
        + edges["z_obs_local"].fillna(0.0)
    )

    # Optional multiplicative variant for comparison.
    edges["interference_product"] = (
        pd.to_numeric(edges["angle_jump_deg"], errors="coerce").fillna(0.0)
        * pd.to_numeric(edges["seam_jump"], errors="coerce").fillna(0.0)
        * (1.0 + pd.to_numeric(edges["obs_local"], errors="coerce").fillna(0.0))
    )

    return edges


def aggregate_node_interference(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if edges.empty:
        work["mean_angle_jump"] = np.nan
        work["mean_seam_jump"] = np.nan
        work["mean_obs_local"] = np.nan
        work["mean_interference_score"] = np.nan
        work["mean_interference_product"] = np.nan
        return work

    agg = (
        edges.groupby("src_node_id", dropna=False)
        .agg(
            mean_angle_jump=("angle_jump_deg", "mean"),
            mean_seam_jump=("seam_jump", "mean"),
            mean_obs_local=("obs_local", "mean"),
            mean_crit_local=("crit_local", "mean"),
            mean_interference_score=("interference_score", "mean"),
            mean_interference_product=("interference_product", "mean"),
            n_edges=("src_node_id", "size"),
        )
        .reset_index()
        .rename(columns={"src_node_id": "node_id"})
    )

    merged = work.merge(agg, on="node_id", how="left")
    return merged


def build_top_nodes(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = [
        c for c in [
            "node_id",
            "r",
            "alpha",
            "mean_interference_score",
            "mean_angle_jump",
            "mean_seam_jump",
            "mean_obs_local",
            "criticality",
            "distance_to_seam",
            "obstruction_mean_abs_holonomy",
            "obstruction_signed_sum_holonomy",
            "signed_phase",
        ] if c in df.columns
    ]
    work = df[cols].dropna(subset=["mean_interference_score"]).copy()
    work = work.sort_values("mean_interference_score", ascending=False).head(top_n).reset_index(drop=True)
    work.insert(0, "rank", np.arange(1, len(work) + 1))
    return work


def summarize(nodes: pd.DataFrame, edges: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Identity Field Interference Summary ===")
    lines.append("")
    lines.append("Best basis")
    lines.append(f"  c1 = {BEST_BASIS[0]}")
    lines.append(f"  c2 = {BEST_BASIS[1]}")
    lines.append(f"  c3 = {BEST_BASIS[2]}")
    lines.append("")
    lines.append(f"n_nodes = {len(nodes)}")
    lines.append(f"n_edges = {len(edges)}")
    lines.append("")
    lines.append("Global baseline")
    lines.append(f"  corr(angle, signed_obstruction) = {_corr(nodes, 'dew_angle_deg', 'obstruction_signed_sum_holonomy'):.4f}")
    lines.append(f"  corr(angle, distance_to_seam) = {_corr(nodes, 'dew_angle_deg', 'distance_to_seam'):.4f}")
    lines.append("")
    lines.append("Interference score alignment")
    lines.append(f"  corr(interference, criticality) = {_corr(nodes, 'mean_interference_score', 'criticality'):.4f}")
    lines.append(f"  corr(interference, distance_to_seam) = {_corr(nodes, 'mean_interference_score', 'distance_to_seam'):.4f}")
    lines.append(f"  corr(interference, mean_angle_jump) = {_corr(nodes, 'mean_interference_score', 'mean_angle_jump'):.4f}")
    lines.append(f"  corr(interference, obstruction_mean_abs_holonomy) = {_corr(nodes, 'mean_interference_score', 'obstruction_mean_abs_holonomy'):.4f}")
    lines.append("")
    lines.append("Component correlations")
    lines.append(f"  corr(mean_angle_jump, criticality) = {_corr(nodes, 'mean_angle_jump', 'criticality'):.4f}")
    lines.append(f"  corr(mean_seam_jump, criticality) = {_corr(nodes, 'mean_seam_jump', 'criticality'):.4f}")
    lines.append(f"  corr(mean_obs_local, criticality) = {_corr(nodes, 'mean_obs_local', 'criticality'):.4f}")
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


def plot_lattice(df: pd.DataFrame, outpath: Path, value_col: str = "mean_interference_score") -> None:
    plot_df = df.dropna(subset=["r", "alpha", value_col]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        plot_df["alpha"],
        plot_df["r"],
        c=plot_df[value_col],
        s=80,
    )
    fig.colorbar(sc, ax=ax, shrink=0.85)
    ax.set_title("Interference score over lattice")
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-layer identity field interference toy.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="outputs/toy_identity_field_interference")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    cfg = Config(outputs_root=args.outputs_root, outdir=args.outdir, top_n=args.top_n)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(Path(cfg.outputs_root))
    angle_df = build_angle_table(df, cfg)
    edges = build_edge_table(angle_df)
    nodes = aggregate_node_interference(angle_df, edges)
    top_nodes = build_top_nodes(nodes, cfg.top_n)

    nodes.to_csv(outdir / "identity_interference_nodes.csv", index=False)
    edges.to_csv(outdir / "identity_interference_edges.csv", index=False)
    top_nodes.to_csv(outdir / "identity_interference_top_nodes.csv", index=False)
    (outdir / "identity_interference_summary.txt").write_text(summarize(nodes, edges), encoding="utf-8")

    plot_scatter(
        nodes,
        "mean_interference_score",
        "criticality",
        outdir / "identity_interference_vs_criticality.png",
        "Interference vs criticality",
    )
    plot_scatter(
        nodes,
        "mean_interference_score",
        "distance_to_seam",
        outdir / "identity_interference_vs_distance_to_seam.png",
        "Interference vs seam distance",
    )
    plot_scatter(
        nodes,
        "mean_interference_score",
        "mean_angle_jump",
        outdir / "identity_interference_vs_angle_jump.png",
        "Interference vs mean angle jump",
    )
    plot_lattice(
        nodes,
        outdir / "identity_interference_lattice.png",
        value_col="mean_interference_score",
    )

    print("=== Identity Field Interference Toy ===")
    for name in [
        "identity_interference_summary.txt",
        "identity_interference_top_nodes.csv",
        "identity_interference_nodes.csv",
        "identity_interference_edges.csv",
        "identity_interference_vs_criticality.png",
        "identity_interference_vs_distance_to_seam.png",
        "identity_interference_vs_angle_jump.png",
        "identity_interference_lattice.png",
    ]:
        print(outdir / name)


if __name__ == "__main__":
    main()
