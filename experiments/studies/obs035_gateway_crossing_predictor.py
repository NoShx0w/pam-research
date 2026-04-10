#!/usr/bin/env python3
"""
OBS-035 — Gateway crossing predictor.

Purpose
-------
Turn the core-to-escape gateway from OBS-034 into a predictive law.

This study predicts whether a composition is a core->escape crossing using
local pre-gateway information:

- launch object/state
- relational obstruction
- anisotropy
- distance to seam
- launch generator
- family label

Core outcome
------------
y = 1 if composition is classified as core_to_escape
y = 0 if composition is classified as core_internal

We intentionally compare only boundary-adjacent cases from the reversible core:
- core_internal  : stayed inside reversible core
- core_to_escape : crossed into directed escape

Inputs
------
outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv

Outputs
-------
outputs/obs035_gateway_crossing_predictor/
  gateway_crossing_dataset.csv
  gateway_crossing_coefficients.csv
  gateway_crossing_family_summary.csv
  obs035_gateway_crossing_predictor_summary.txt
  obs035_gateway_crossing_predictor_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Config:
    crossings_csv: str = (
        "outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv"
    )
    assignments_csv: str = (
        "outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv"
    )
    outdir: str = "outputs/obs035_gateway_crossing_predictor"
    min_rows_per_family: int = 6
    random_state: int = 42


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_cols = {
        "route_class",
        "generator_1",
        "generator_2",
        "src1",
        "tgt1",
        "src2",
        "tgt2",
        "sector_1",
        "sector_2",
        "crossing_type",
        "composition_typed",
        "motif",
        "motif_class",
        "state_a",
        "state_b",
        "state_c",
        "state_a_red",
        "state_b_red",
        "state_c_red",
        "generator_word",
        "generator_completed",
        "path_family",
    }
    for col in df.columns:
        if col not in text_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def safe_mean(s: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def build_assignment_feature_map(assignments: pd.DataFrame) -> pd.DataFrame:
    a = assignments.copy()
    a["route_class"] = a["route_class"].astype(str)
    a["generator_completed"] = a["generator_completed"].astype(str)

    # Representative local feature profile per family+generator
    feat = (
        a.groupby(["route_class", "generator_completed"], as_index=False)
        .agg(
            n_rows=("generator_completed", "size"),
            launch_state=("state_a_red", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
            target_state=("state_c_red", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
            mean_relational=("relational_a", "mean") if "relational_a" in a.columns else ("distance_a", "size"),
            mean_anisotropy=("anisotropy_a", "mean") if "anisotropy_a" in a.columns else ("distance_a", "size"),
            mean_distance=("distance_a", "mean") if "distance_a" in a.columns else ("n_rows", "size"),
        )
    )

    # If fallback tuples above created wrong values due to missing columns, patch them
    if "relational_a" not in a.columns:
        feat["mean_relational"] = np.nan
    if "anisotropy_a" not in a.columns:
        feat["mean_anisotropy"] = np.nan
    if "distance_a" not in a.columns:
        feat["mean_distance"] = np.nan

    return feat


def build_dataset(crossings: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    # Only compare reversible-core launches:
    #   core_internal    -> negative
    #   core_to_escape   -> positive
    use = crossings[crossings["crossing_type"].isin(["core_internal", "core_to_escape"])].copy()
    use["y_cross"] = (use["crossing_type"] == "core_to_escape").astype(int)

    feat = build_assignment_feature_map(assignments)

    # Join launch-generator features using generator_1 as pre-gateway local context
    ds = use.merge(
        feat,
        left_on=["route_class", "generator_1"],
        right_on=["route_class", "generator_completed"],
        how="left",
        suffixes=("", "_feat"),
    )

    ds["launch_generator"] = ds["generator_1"].astype(str)
    ds["next_generator"] = ds["generator_2"].astype(str)

    # Keep simple interpretable fields
    keep = [
        "route_class",
        "crossing_type",
        "y_cross",
        "n_compositions",
        "composition_share",
        "launch_generator",
        "next_generator",
        "launch_state",
        "target_state",
        "mean_relational",
        "mean_anisotropy",
        "mean_distance",
        "composition_typed",
    ]
    ds = ds[keep].copy()
    return ds


def fit_logistic(ds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    # Build interpretable design matrix
    X = ds.copy()

    # numeric
    for col in ["mean_relational", "mean_anisotropy", "mean_distance"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)

    # categoricals
    design = pd.get_dummies(
        X[[
            "route_class",
            "launch_generator",
            "next_generator",
            "launch_state",
            "target_state",
            "mean_relational",
            "mean_anisotropy",
            "mean_distance",
        ]],
        columns=["route_class", "launch_generator", "next_generator", "launch_state", "target_state"],
        drop_first=False,
        dtype=float,
    )

    y = ds["y_cross"].to_numpy(dtype=int)
    sample_weight = pd.to_numeric(ds["n_compositions"], errors="coerce").fillna(1.0).to_numpy(dtype=float)

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        random_state=42,
        max_iter=2000,
    )
    model.fit(design.to_numpy(dtype=float), y, sample_weight=sample_weight)

    probs = model.predict_proba(design.to_numpy(dtype=float))[:, 1]
    auc = float(roc_auc_score(y, probs, sample_weight=sample_weight)) if len(np.unique(y)) > 1 else float("nan")

    coef = pd.DataFrame(
        {
            "feature": design.columns,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc,
        "intercept": float(model.intercept_[0]),
        "n_rows": float(len(ds)),
        "positive_rate": float(ds["y_cross"].mean()),
    }
    return coef, metrics


def build_family_summary(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in CLASS_ORDER:
        sub = ds[ds["route_class"] == cls].copy()
        if len(sub) == 0:
            rows.append(
                {
                    "route_class": cls,
                    "n_rows": 0,
                    "crossing_rate": np.nan,
                    "mean_relational_cross": np.nan,
                    "mean_relational_internal": np.nan,
                    "mean_anisotropy_cross": np.nan,
                    "mean_anisotropy_internal": np.nan,
                    "mean_distance_cross": np.nan,
                    "mean_distance_internal": np.nan,
                    "top_launch_generator_cross": np.nan,
                    "top_launch_generator_internal": np.nan,
                }
            )
            continue

        pos = sub[sub["y_cross"] == 1]
        neg = sub[sub["y_cross"] == 0]

        rows.append(
            {
                "route_class": cls,
                "n_rows": int(len(sub)),
                "crossing_rate": float(sub["y_cross"].mean()),
                "mean_relational_cross": safe_mean(pos["mean_relational"]),
                "mean_relational_internal": safe_mean(neg["mean_relational"]),
                "mean_anisotropy_cross": safe_mean(pos["mean_anisotropy"]),
                "mean_anisotropy_internal": safe_mean(neg["mean_anisotropy"]),
                "mean_distance_cross": safe_mean(pos["mean_distance"]),
                "mean_distance_internal": safe_mean(neg["mean_distance"]),
                "top_launch_generator_cross": pos["launch_generator"].value_counts().index[0] if len(pos) else np.nan,
                "top_launch_generator_internal": neg["launch_generator"].value_counts().index[0] if len(neg) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_summary(ds: pd.DataFrame, coef: pd.DataFrame, fam: pd.DataFrame, metrics: dict[str, float]) -> str:
    lines = [
        "=== OBS-035 Gateway Crossing Predictor Summary ===",
        "",
        f"n_rows = {int(metrics['n_rows'])}",
        f"positive_rate = {metrics['positive_rate']:.4f}",
        f"auc = {metrics['auc']:.4f}",
        f"intercept = {metrics['intercept']:.6f}",
        "",
        "Interpretive guide",
        "- positive class = core_to_escape crossing",
        "- negative class = core_internal continuation",
        "- positive coefficients increase crossing propensity",
        "- negative coefficients favor remaining in the reversible core",
        "",
        "Top predictive features",
    ]

    for _, row in coef.head(12).iterrows():
        lines.append(
            f"  {row['feature']}: coef={row['coefficient']:.6f}"
        )

    lines.extend(["", "Family summaries"])
    for _, row in fam.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_rows                    = {int(row['n_rows'])}",
                f"  crossing_rate             = {float(row['crossing_rate']):.4f}",
                f"  mean_relational_cross     = {float(row['mean_relational_cross']):.4f}",
                f"  mean_relational_internal  = {float(row['mean_relational_internal']):.4f}",
                f"  mean_anisotropy_cross     = {float(row['mean_anisotropy_cross']):.4f}",
                f"  mean_anisotropy_internal  = {float(row['mean_anisotropy_internal']):.4f}",
                f"  mean_distance_cross       = {float(row['mean_distance_cross']):.4f}",
                f"  mean_distance_internal    = {float(row['mean_distance_internal']):.4f}",
                f"  top_launch_generator_cross= {row['top_launch_generator_cross']}",
                f"  top_launch_generator_internal= {row['top_launch_generator_internal']}",
                "",
            ]
        )

    return "\n".join(lines)


def render_figure(ds: pd.DataFrame, coef: pd.DataFrame, fam: pd.DataFrame, metrics: dict[str, float], outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_coef = fig.add_subplot(gs[0, 0])
    ax_rates = fig.add_subplot(gs[0, 1])
    ax_fields = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    top = coef.head(10).iloc[::-1]
    ax_coef.barh(np.arange(len(top)), top["coefficient"].to_numpy(dtype=float))
    ax_coef.set_yticks(np.arange(len(top)))
    ax_coef.set_yticklabels(top["feature"].tolist(), fontsize=8)
    ax_coef.set_title("Top predictor coefficients", fontsize=14, pad=8)
    ax_coef.grid(alpha=0.15, axis="x")

    x = np.arange(len(fam))
    ax_rates.bar(x, fam["crossing_rate"])
    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(fam["route_class"], rotation=12)
    ax_rates.set_title("Core→escape crossing rate", fontsize=14, pad=8)
    ax_rates.grid(alpha=0.15, axis="y")

    width = 0.34
    ax_fields.bar(x - width / 2, fam["mean_relational_cross"] - fam["mean_relational_internal"], width, label="Δ relational")
    ax_fields.bar(x + width / 2, fam["mean_anisotropy_cross"] - fam["mean_anisotropy_internal"], width, label="Δ anisotropy")
    ax_fields.set_xticks(x)
    ax_fields.set_xticklabels(fam["route_class"], rotation=12)
    ax_fields.set_title("Crossing minus internal field means", fontsize=14, pad=8)
    ax_fields.grid(alpha=0.15, axis="y")
    ax_fields.legend()

    ax_top.axis("off")
    y = 0.95
    for _, row in fam.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"cross gen: {row['top_launch_generator_cross']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"internal gen: {row['top_launch_generator_internal']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"Δrel: {row['mean_relational_cross'] - row['mean_relational_internal']:.4f}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"Δaniso: {row['mean_anisotropy_cross'] - row['mean_anisotropy_internal']:.4f}", fontsize=10, family="monospace")
        y -= 0.07
    ax_top.set_title("Family gateway contrast", fontsize=14, pad=8)

    ax_diag.axis("off")
    text = (
        "OBS-035 diagnostics\n\n"
        f"AUC:\n{metrics['auc']:.3f}\n\n"
        f"positive rate:\n{metrics['positive_rate']:.3f}\n\n"
        "Interpretation:\n"
        "this tests whether local\n"
        "pre-gateway state predicts\n"
        "conversion from reversible\n"
        "core into directed escape."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-035 gateway crossing predictor", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict core-to-escape gateway crossing from local state.")
    parser.add_argument("--crossings-csv", default=Config.crossings_csv)
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--min-rows-per-family", type=int, default=Config.min_rows_per_family)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    args = parser.parse_args()

    cfg = Config(
        crossings_csv=args.crossings_csv,
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        min_rows_per_family=args.min_rows_per_family,
        random_state=args.random_state,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    crossings = load_csv(cfg.crossings_csv)
    assignments = load_csv(cfg.assignments_csv)

    ds = build_dataset(crossings, assignments)
    coef, metrics = fit_logistic(ds)
    fam = build_family_summary(ds)

    ds_csv = outdir / "gateway_crossing_dataset.csv"
    coef_csv = outdir / "gateway_crossing_coefficients.csv"
    fam_csv = outdir / "gateway_crossing_family_summary.csv"
    txt_path = outdir / "obs035_gateway_crossing_predictor_summary.txt"
    png_path = outdir / "obs035_gateway_crossing_predictor_figure.png"

    ds.to_csv(ds_csv, index=False)
    coef.to_csv(coef_csv, index=False)
    fam.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(ds, coef, fam, metrics), encoding="utf-8")
    render_figure(ds, coef, fam, metrics, png_path)

    print(ds_csv)
    print(coef_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
