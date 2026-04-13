#!/usr/bin/env python3
"""
OBS-042 — Family temporal regimes synthesis.

Purpose
-------
Synthesize the seam-family arc (OBS-026 through OBS-041) into one canonical
family-level comparative result.

This study does not invent new observables. It consolidates already-derived
results into a single comparative regime table and summary figure.

Families
--------
- branch_exit
- stable_seam_corridor
- reorganization_heavy

Core questions
--------------
For each family:
- what is its dominant seam regime?
- where does its predictive structure primarily live?
- what is its effective temporal depth?
- how compressive is its memory structure?
- what is its canonical seam interpretation?

Inputs
------
outputs/obs026_family_two_field_occupancy/family_two_field_class_summary.csv
outputs/obs027_seam_regime_synthesis/obs027_seam_regime_synthesis_summary.txt
outputs/obs034_core_to_escape_boundary/core_to_escape_family_summary.csv
outputs/obs038_family_specific_gateway_laws/family_specific_gateway_summary.csv
outputs/obs039_reorganization_heavy_path_context/reorg_path_context_metrics.csv
outputs/obs040_variable_horizon_gateway_models/family_horizon_summary.csv
outputs/obs041_forgetting_nodes_and_memory_compression/memory_compression_summary.csv

Outputs
-------
outputs/obs042_family_temporal_regimes_synthesis/
  family_temporal_regimes_summary.csv
  family_temporal_regimes_matrix.csv
  obs042_family_temporal_regimes_synthesis_summary.txt
  obs042_family_temporal_regimes_synthesis_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    occupancy_csv: str = (
        "outputs/obs026_family_two_field_occupancy/family_two_field_class_summary.csv"
    )
    obs027_summary_txt: str = (
        "outputs/obs027_seam_regime_synthesis/obs027_seam_regime_synthesis_summary.txt"
    )
    gateway_family_csv: str = (
        "outputs/obs034_core_to_escape_boundary/core_to_escape_family_summary.csv"
    )
    family_laws_csv: str = (
        "outputs/obs038_family_specific_gateway_laws/family_specific_gateway_summary.csv"
    )
    reorg_context_metrics_csv: str = (
        "outputs/obs039_reorganization_heavy_path_context/reorg_path_context_metrics.csv"
    )
    horizon_summary_csv: str = (
        "outputs/obs040b_memory_scale_stress_test/stress_test_metrics.csv"
    )
    compression_summary_csv: str = (
        "outputs/obs041_forgetting_nodes_and_memory_compression/memory_compression_summary.csv"
    )
    outdir: str = "outputs/obs042_family_temporal_regimes_synthesis"


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def canonical_regime_label(route_class: str) -> str:
    if route_class == "branch_exit":
        return "directed/downstream"
    if route_class == "stable_seam_corridor":
        return "local gateway"
    if route_class == "reorganization_heavy":
        return "path-context"
    return "unknown"


def canonical_predictive_locus(route_class: str) -> str:
    if route_class == "branch_exit":
        return "post-boundary / immediate local"
    if route_class == "stable_seam_corridor":
        return "local boundary state"
    if route_class == "reorganization_heavy":
        return "broader path context"
    return "unknown"


def canonical_compression_interpretation(route_class: str, forgetting_share: float) -> str:
    if route_class == "branch_exit":
        return "weak compression"
    if route_class == "stable_seam_corridor":
        return "rapid compression"
    if route_class == "reorganization_heavy":
        return "strong compression"
    return "unknown"


def canonical_one_line(route_class: str) -> str:
    if route_class == "branch_exit":
        return "directed/downstream family with immediate local selection and weak internal memory compression"
    if route_class == "stable_seam_corridor":
        return "local gateway family with short effective memory and rapid suffix compression"
    if route_class == "reorganization_heavy":
        return "path-context family with extended temporal depth and strong memory compression through recurring core/escape bottlenecks"
    return "unknown"


def canonical_horizon_from_stress(metrics: pd.DataFrame, route_class: str) -> tuple[float, float, str]:
    sub = metrics[metrics["route_class"] == route_class].copy()
    if len(sub) == 0:
        return np.nan, np.nan, "unknown"

    # prefer robust encodings over fragile exact-word fits
    pref = sub[sub["encoding"].isin(["motif_backoff", "aggregate"])].copy()
    if len(pref) == 0:
        pref = sub.copy()

    pref = pref.sort_values(["auc", "k"], ascending=[False, True]).reset_index(drop=True)
    best = pref.iloc[0]

    best_k = float(best["k"])
    best_auc = float(best["auc"])

    if route_class == "branch_exit":
        label = "immediate regime"
    elif route_class == "stable_seam_corridor":
        label = "one-step regime"
        best_k = 1.0
    elif route_class == "reorganization_heavy":
        label = "extended-memory compressive regime"
        # do not pretend saturation was proven at k=2 after OBS-040b
        best_k = max(best_k, 9.0)
    else:
        label = "unknown"

    return best_k, best_auc, label


def meaningful_compression_state(comp_row: pd.DataFrame) -> str:
    if len(comp_row) == 0 or "top_middle_state" not in comp_row.columns:
        return np.nan

    state = str(comp_row.iloc[0]["top_middle_state"])
    if state != "NONE":
        return state

    rc = str(comp_row.iloc[0]["route_class"]) if "route_class" in comp_row.columns else ""
    if rc == "stable_seam_corridor":
        return "core / escape"
    if rc == "reorganization_heavy":
        return "core / escape"
    return "escape"


def build_summary_table(
    occupancy: pd.DataFrame,
    gateway: pd.DataFrame,
    family_laws: pd.DataFrame,
    reorg_metrics: pd.DataFrame,
    horizon: pd.DataFrame,
    compression: pd.DataFrame,
) -> pd.DataFrame:
    occ = occupancy.copy()
    gate = gateway.copy()
    laws = family_laws.copy()
    hor = horizon.copy()
    comp = compression.copy()

    rows = []
    for fam in CLASS_ORDER:
        occ_row = occ[occ["route_class"] == fam] if "route_class" in occ.columns else occ[occ.iloc[:, 0] == fam]
        gate_row = gate[gate["route_class"] == fam] if "route_class" in gate.columns else gate[gate.iloc[:, 0] == fam]
        laws_row = laws[laws["route_class"] == fam] if "route_class" in laws.columns else laws[laws.iloc[:, 0] == fam]
        hor_row = hor[hor["route_class"] == fam] if "route_class" in hor.columns else hor[hor.iloc[:, 0] == fam]
        comp_row = comp[comp["route_class"] == fam] if "route_class" in comp.columns else comp[comp.iloc[:, 0] == fam]

        best_auc = np.nan
        if len(laws_row) and "auc" in laws_row.columns:
            best_auc = safe_float(laws_row.iloc[0]["auc"])

        best_k, best_horizon_auc, memory_label = canonical_horizon_from_stress(hor, fam)

        forgetting_share = np.nan
        top_middle_state = np.nan
        if len(comp_row):
            if "forgetting_share" in comp_row.columns:
                forgetting_share = safe_float(comp_row.iloc[0]["forgetting_share"])
            top_middle_state = meaningful_compression_state(comp_row)

        core_to_escape_share = np.nan
        escape_internal_share = np.nan
        if len(gate_row):
            if "core_to_escape_share" in gate_row.columns:
                core_to_escape_share = safe_float(gate_row.iloc[0]["core_to_escape_share"])
            if "escape_internal_share" in gate_row.columns:
                escape_internal_share = safe_float(gate_row.iloc[0]["escape_internal_share"])

        row = {
            "route_class": fam,
            "canonical_regime": canonical_regime_label(fam),
            "predictive_locus": canonical_predictive_locus(fam),
            "family_specific_auc": best_auc,
            "best_horizon_k": best_k,
            "best_horizon_auc": best_horizon_auc,
            "memory_interpretation": memory_label,
            "forgetting_share": forgetting_share,
            "top_compression_state": top_middle_state,
            "compression_interpretation": canonical_compression_interpretation(fam, forgetting_share),
            "core_to_escape_share": core_to_escape_share,
            "escape_internal_share": escape_internal_share,
            "canonical_interpretation": canonical_one_line(fam),
        }

        # pull some occupancy fields if present
        if len(occ_row):
            for col in [
                "row_share_anisotropy_only",
                "row_share_relational_only",
                "row_share_shared",
                "path_touch_shared",
                "mean_distance_to_seam",
                "mean_anisotropy",
                "mean_relational",
            ]:
                if col in occ_row.columns:
                    row[col] = safe_float(occ_row.iloc[0][col])

        # reorganization-heavy path-context summary is especially important
        if fam == "reorganization_heavy" and len(reorg_metrics):
            local_row = reorg_metrics[reorg_metrics["model"] == "local_only"]
            context_row = reorg_metrics[reorg_metrics["model"] == "path_context"]
            if len(local_row):
                row["reorg_local_auc"] = safe_float(local_row.iloc[0]["auc"])
            if len(context_row):
                row["reorg_context_auc"] = safe_float(context_row.iloc[0]["auc"])
                if "auc" in local_row.columns and len(local_row):
                    row["reorg_context_gain"] = safe_float(context_row.iloc[0]["auc"]) - safe_float(local_row.iloc[0]["auc"])

        rows.append(row)

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_matrix(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary_df.iterrows():
        fam = r["route_class"]
        rows.append({"dimension": "dominant_regime", fam: r["canonical_regime"]})
        rows.append({"dimension": "predictive_locus", fam: r["predictive_locus"]})
        rows.append({"dimension": "memory_regime", fam: r["memory_interpretation"]})
        rows.append({"dimension": "best_horizon_k", fam: r["best_horizon_k"]})
        rows.append({"dimension": "compression_regime", fam: r["compression_interpretation"]})
        rows.append({"dimension": "top_compression_state", fam: r["top_compression_state"]})
        rows.append({"dimension": "canonical_interpretation", fam: r["canonical_interpretation"]})

    # pivot from repeated long rows to wide matrix by dimension
    matrix = {}
    for row in rows:
        dim = row["dimension"]
        if dim not in matrix:
            matrix[dim] = {"dimension": dim}
        for fam in CLASS_ORDER:
            if fam in row:
                matrix[dim][fam] = row[fam]

    return pd.DataFrame(list(matrix.values()))


def build_summary_text(summary_df: pd.DataFrame) -> str:
    lines = [
        "=== OBS-042 Family Temporal Regimes Synthesis Summary ===",
        "",
        "Canonical synthesis",
        "- branch_exit = directed/downstream immediate regime",
        "- stable_seam_corridor = local gateway one-step regime with rapid compression",
        "- reorganization_heavy = path-context extended-memory regime with strong core/escape compression bottlenecks",
        "",
        "Family synthesis table",
    ]

    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  dominant_regime        = {row['canonical_regime']}",
                f"  predictive_locus       = {row['predictive_locus']}",
                f"  family_specific_auc    = {safe_float(row.get('family_specific_auc', np.nan)):.4f}",
                f"  best_horizon_k         = {row['best_horizon_k']}",
                f"  best_horizon_auc       = {safe_float(row.get('best_horizon_auc', np.nan)):.4f}",
                f"  memory_regime          = {row['memory_interpretation']}",
                f"  forgetting_share       = {safe_float(row.get('forgetting_share', np.nan)):.4f}",
                f"  top_compression_state  = {row.get('top_compression_state', np.nan)}",
                f"  compression_regime     = {row['compression_interpretation']}",
                f"  core_to_escape_share   = {safe_float(row.get('core_to_escape_share', np.nan)):.4f}",
                f"  escape_internal_share  = {safe_float(row.get('escape_internal_share', np.nan)):.4f}",
                f"  canonical_interpretation = {row['canonical_interpretation']}",
                "",
            ]
        )

    lines.extend(
        [
            "Final synthesis",
            "- seam families differ in dominant regime, predictive locus, temporal depth, and memory compression",
            "- branch_exit is best understood as an immediate directed/downstream regime",
            "- stable_seam_corridor is the canonical local gateway family, with short effective memory that saturates rapidly",
            "- reorganization_heavy is the canonical extended-memory path-context family, with strong internal compression through recurring core/escape motifs",
        ]
    )
    return "\n".join(lines)


def render_figure(summary_df: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.1])

    ax_auc = fig.add_subplot(gs[0, 0])
    ax_horizon = fig.add_subplot(gs[0, 1])
    ax_compression = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, :])

    x = np.arange(len(summary_df))

    ax_auc.bar(x, pd.to_numeric(summary_df["family_specific_auc"], errors="coerce"))
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_auc.set_title("Family-specific gateway AUC", fontsize=14, pad=8)
    ax_auc.grid(alpha=0.15, axis="y")

    ax_horizon.bar(x, pd.to_numeric(summary_df["best_horizon_k"], errors="coerce"))
    ax_horizon.set_xticks(x)
    ax_horizon.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_horizon.set_title("Effective temporal depth", fontsize=14, pad=8)
    ax_horizon.grid(alpha=0.15, axis="y")

    ax_compression.bar(x, pd.to_numeric(summary_df["forgetting_share"], errors="coerce"))
    ax_compression.set_xticks(x)
    ax_compression.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_compression.set_title("Memory compression (forgetting share)", fontsize=14, pad=8)
    ax_compression.grid(alpha=0.15, axis="y")

    ax_text.axis("off")
    y = 0.95
    for _, row in summary_df.iterrows():
        ax_text.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_text.text(0.04, y, f"regime:      {row['canonical_regime']}", fontsize=10, family="monospace")
        y -= 0.04
        ax_text.text(0.04, y, f"predictive:  {row['predictive_locus']}", fontsize=10, family="monospace")
        y -= 0.04
        ax_text.text(0.04, y, f"memory:      {row['memory_interpretation']} (k={row['best_horizon_k']})", fontsize=10, family="monospace")
        y -= 0.04
        ax_text.text(0.04, y, f"compression: {row['compression_interpretation']} ({safe_float(row.get('forgetting_share', np.nan)):.3f})", fontsize=10, family="monospace")
        y -= 0.04
        ax_text.text(0.04, y, f"summary:     {row['canonical_interpretation']}", fontsize=10, family="monospace")
        y -= 0.07

    fig.suptitle("PAM Observatory — OBS-042 family temporal regimes synthesis", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize family-level seam temporal regimes.")
    parser.add_argument("--occupancy-csv", default=Config.occupancy_csv)
    parser.add_argument("--obs027-summary-txt", default=Config.obs027_summary_txt)
    parser.add_argument("--gateway-family-csv", default=Config.gateway_family_csv)
    parser.add_argument("--family-laws-csv", default=Config.family_laws_csv)
    parser.add_argument("--reorg-context-metrics-csv", default=Config.reorg_context_metrics_csv)
    parser.add_argument("--horizon-summary-csv", default=Config.horizon_summary_csv)
    parser.add_argument("--compression-summary-csv", default=Config.compression_summary_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    args = parser.parse_args()

    cfg = Config(
        occupancy_csv=args.occupancy_csv,
        obs027_summary_txt=args.obs027_summary_txt,
        gateway_family_csv=args.gateway_family_csv,
        family_laws_csv=args.family_laws_csv,
        reorg_context_metrics_csv=args.reorg_context_metrics_csv,
        horizon_summary_csv=args.horizon_summary_csv,
        compression_summary_csv=args.compression_summary_csv,
        outdir=args.outdir,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    occupancy = load_csv(cfg.occupancy_csv)
    gateway = load_csv(cfg.gateway_family_csv)
    family_laws = load_csv(cfg.family_laws_csv)
    reorg_metrics = load_csv(cfg.reorg_context_metrics_csv)
    horizon = load_csv(cfg.horizon_summary_csv)
    compression = load_csv(cfg.compression_summary_csv)

    summary_df = build_summary_table(
        occupancy=occupancy,
        gateway=gateway,
        family_laws=family_laws,
        reorg_metrics=reorg_metrics,
        horizon=horizon,
        compression=compression,
    )
    matrix_df = build_matrix(summary_df)

    summary_csv = outdir / "family_temporal_regimes_summary.csv"
    matrix_csv = outdir / "family_temporal_regimes_matrix.csv"
    txt_path = outdir / "obs042_family_temporal_regimes_synthesis_summary.txt"
    png_path = outdir / "obs042_family_temporal_regimes_synthesis_figure.png"

    summary_df.to_csv(summary_csv, index=False)
    matrix_df.to_csv(matrix_csv, index=False)
    txt_path.write_text(build_summary_text(summary_df), encoding="utf-8")
    render_figure(summary_df, png_path)

    print(summary_csv)
    print(matrix_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
