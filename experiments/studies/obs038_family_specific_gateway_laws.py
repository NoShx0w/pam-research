#!/usr/bin/env python3
"""
OBS-038 — Family-specific gateway laws.

Purpose
-------
Test whether the weak pooled gateway signal is masking stronger
family-specific laws.

OBS-035c and OBS-037b suggested:
- modest pooled predictive signal
- little benefit from refined symbolic states
- little benefit from one-step history
- stable_seam_corridor appears qualitatively different from
  branch_exit and reorganization_heavy

This study fits separate family-level gateway predictors and compares them
against the pooled model.

Prediction task
---------------
For each family independently:

    y = 1  core_to_escape
    y = 0  core_internal

using only valid pre-second-step information.

Inputs
------
outputs/obs037b_pre_second_step_gateway_predictor/gateway_crossing_pre_second_step_dataset.csv

Outputs
-------
outputs/obs038_family_specific_gateway_laws/
  family_specific_gateway_coefficients.csv
  family_specific_gateway_metrics.csv
  family_specific_gateway_summary.csv
  obs038_family_specific_gateway_laws_summary.txt
  obs038_family_specific_gateway_laws_figure.png
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
    dataset_csv: str = (
        "outputs/obs037b_pre_second_step_gateway_predictor/"
        "gateway_crossing_pre_second_step_dataset.csv"
    )
    outdir: str = "outputs/obs038_family_specific_gateway_laws"
    min_rows: int = 8
    random_state: int = 42


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

TEXT_COLS = {
    "route_class",
    "crossing_type",
    "prev_generator",
    "prev_state",
    "prev_target",
    "prelaunch_word",
    "composition_typed",
}


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in TEXT_COLS:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def safe_mean(s: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def fit_family_model(df: pd.DataFrame, family_name: str) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.copy()

    for col in ["mean_relational", "mean_anisotropy", "mean_distance"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median() if work[col].notna().any() else 0.0)

    # Keep only columns available before second step
    design = pd.get_dummies(
        work[
            [
                "prev_generator",
                "prev_state",
                "prev_target",
                "prelaunch_word",
                "mean_relational",
                "mean_anisotropy",
                "mean_distance",
            ]
        ],
        columns=["prev_generator", "prev_state", "prev_target", "prelaunch_word"],
        drop_first=False,
        dtype=float,
    )

    y = work["y_cross"].to_numpy(dtype=int)

    # Need both classes to fit meaningful model
    if len(work) == 0 or len(np.unique(y)) < 2:
        return (
            pd.DataFrame(columns=["route_class", "feature", "coefficient", "abs_coefficient"]),
            {
                "route_class": family_name,
                "n_rows": float(len(work)),
                "positive_rate": float(work["y_cross"].mean()) if len(work) else float("nan"),
                "auc": float("nan"),
                "intercept": float("nan"),
                "n_features": 0.0,
            },
        )

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        random_state=42,
        max_iter=4000,
    )
    model.fit(design.to_numpy(dtype=float), y)

    probs = model.predict_proba(design.to_numpy(dtype=float))[:, 1]
    auc = float(roc_auc_score(y, probs))

    coef = pd.DataFrame(
        {
            "route_class": family_name,
            "feature": design.columns,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    metrics = {
        "route_class": family_name,
        "n_rows": float(len(work)),
        "positive_rate": float(work["y_cross"].mean()),
        "auc": auc,
        "intercept": float(model.intercept_[0]),
        "n_features": float(design.shape[1]),
    }
    return coef, metrics


def build_summary_table(ds: pd.DataFrame, coef_all: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for fam in CLASS_ORDER:
        sub = ds[ds["route_class"] == fam].copy()
        pos = sub[sub["y_cross"] == 1]
        neg = sub[sub["y_cross"] == 0]
        coef = coef_all[coef_all["route_class"] == fam].copy()
        met = metrics_df[metrics_df["route_class"] == fam]

        auc = float(met["auc"].iloc[0]) if len(met) else float("nan")

        rows.append(
            {
                "route_class": fam,
                "n_rows": int(len(sub)),
                "crossing_rate": float(sub["y_cross"].mean()) if len(sub) else float("nan"),
                "auc": auc,
                "mean_relational_cross": safe_mean(pos["mean_relational"]),
                "mean_relational_internal": safe_mean(neg["mean_relational"]),
                "mean_anisotropy_cross": safe_mean(pos["mean_anisotropy"]),
                "mean_anisotropy_internal": safe_mean(neg["mean_anisotropy"]),
                "top_prev_generator_cross": pos["prev_generator"].value_counts().index[0] if len(pos) else np.nan,
                "top_prev_generator_internal": neg["prev_generator"].value_counts().index[0] if len(neg) else np.nan,
                "top_feature_1": coef.iloc[0]["feature"] if len(coef) > 0 else np.nan,
                "top_feature_2": coef.iloc[1]["feature"] if len(coef) > 1 else np.nan,
                "top_feature_3": coef.iloc[2]["feature"] if len(coef) > 2 else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_summary(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, coef_all: pd.DataFrame) -> str:
    pooled_reference = 0.6381  # from OBS-037b

    lines = [
        "=== OBS-038 Family-Specific Gateway Laws Summary ===",
        "",
        f"pooled_reference_auc = {pooled_reference:.4f}",
        "",
        "Interpretive guide",
        "- family-specific AUC tests whether pooled modeling washed out stronger per-family structure",
        "- top features indicate which pre-second-step variables matter within each family",
        "- family AUC materially above pooled reference suggests a family-specific law",
        "",
        "Family metrics",
    ]

    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_rows                     = {int(row['n_rows'])}",
                f"  crossing_rate              = {float(row['crossing_rate']):.4f}",
                f"  auc                        = {float(row['auc']):.4f}",
                f"  mean_relational_cross      = {float(row['mean_relational_cross']):.4f}",
                f"  mean_relational_internal   = {float(row['mean_relational_internal']):.4f}",
                f"  mean_anisotropy_cross      = {float(row['mean_anisotropy_cross']):.4f}",
                f"  mean_anisotropy_internal   = {float(row['mean_anisotropy_internal']):.4f}",
                f"  top_prev_generator_cross   = {row['top_prev_generator_cross']}",
                f"  top_prev_generator_internal= {row['top_prev_generator_internal']}",
                f"  top_feature_1              = {row['top_feature_1']}",
                f"  top_feature_2              = {row['top_feature_2']}",
                f"  top_feature_3              = {row['top_feature_3']}",
                "",
            ]
        )

    lines.append("Top family-specific coefficients")
    for fam in CLASS_ORDER:
        fam_coef = coef_all[coef_all["route_class"] == fam].head(5)
        lines.append(f"  {fam}")
        if len(fam_coef) == 0:
            lines.append("    none")
        else:
            for _, row in fam_coef.iterrows():
                lines.append(f"    {row['feature']}: coef={row['coefficient']:.6f}")

    return "\n".join(lines)


def render_figure(summary_df: pd.DataFrame, coef_all: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_auc = fig.add_subplot(gs[0, 0])
    ax_fields = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    x = np.arange(len(summary_df))
    ax_auc.bar(x, summary_df["auc"])
    ax_auc.axhline(0.6381, linestyle="--")
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_auc.set_title("Family-specific AUC", fontsize=14, pad=8)
    ax_auc.grid(alpha=0.15, axis="y")

    width = 0.34
    ax_fields.bar(
        x - width / 2,
        summary_df["mean_anisotropy_cross"] - summary_df["mean_anisotropy_internal"],
        width,
        label="Δ anisotropy",
    )
    ax_fields.bar(
        x + width / 2,
        summary_df["mean_relational_cross"] - summary_df["mean_relational_internal"],
        width,
        label="Δ relational",
    )
    ax_fields.set_xticks(x)
    ax_fields.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_fields.set_title("Crossing minus internal field means", fontsize=14, pad=8)
    ax_fields.grid(alpha=0.15, axis="y")
    ax_fields.legend()

    ax_top.axis("off")
    y = 0.95
    for _, row in summary_df.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"f1: {row['top_feature_1']}", fontsize=9.5, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"f2: {row['top_feature_2']}", fontsize=9.5, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"f3: {row['top_feature_3']}", fontsize=9.5, family="monospace")
        y -= 0.07
    ax_top.set_title("Top family features", fontsize=14, pad=8)

    ax_text.axis("off")
    y = 0.95
    for fam in CLASS_ORDER:
        fam_coef = coef_all[coef_all["route_class"] == fam].head(4)
        ax_text.text(0.02, y, fam, fontsize=12, fontweight="bold")
        y -= 0.06
        if len(fam_coef) == 0:
            ax_text.text(0.04, y, "none", fontsize=10, family="monospace")
            y -= 0.05
        else:
            for _, row in fam_coef.iterrows():
                ax_text.text(0.04, y, f"{row['feature']}: {row['coefficient']:.4f}", fontsize=9.5, family="monospace")
                y -= 0.045
        y -= 0.04
    ax_text.set_title("Coefficient detail", fontsize=14, pad=8)

    best = summary_df.sort_values("auc", ascending=False).iloc[0]
    ax_diag.axis("off")
    text = (
        "OBS-038 diagnostics\n\n"
        "pooled reference AUC:\n0.638\n\n"
        f"best family AUC:\n{best['route_class']} = {best['auc']:.3f}\n\n"
        "Interpretation:\n"
        "tests whether each family\n"
        "has its own gateway law,\n"
        "instead of one pooled law."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-038 family-specific gateway laws", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit family-specific gateway laws.")
    parser.add_argument("--dataset-csv", default=Config.dataset_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--min-rows", type=int, default=Config.min_rows)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    args = parser.parse_args()

    cfg = Config(
        dataset_csv=args.dataset_csv,
        outdir=args.outdir,
        min_rows=args.min_rows,
        random_state=args.random_state,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_csv(cfg.dataset_csv)

    coef_frames = []
    metric_rows = []
    for fam in CLASS_ORDER:
        fam_df = ds[ds["route_class"] == fam].copy()
        coef, metrics = fit_family_model(fam_df, fam)
        coef_frames.append(coef)
        metric_rows.append(metrics)

    coef_all = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    summary_df = build_summary_table(ds, coef_all, metrics_df)

    coef_csv = outdir / "family_specific_gateway_coefficients.csv"
    metrics_csv = outdir / "family_specific_gateway_metrics.csv"
    summary_csv = outdir / "family_specific_gateway_summary.csv"
    txt_path = outdir / "obs038_family_specific_gateway_laws_summary.txt"
    png_path = outdir / "obs038_family_specific_gateway_laws_figure.png"

    coef_all.to_csv(coef_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    txt_path.write_text(build_summary(metrics_df, summary_df, coef_all), encoding="utf-8")
    render_figure(summary_df, coef_all, png_path)

    print(coef_csv)
    print(metrics_csv)
    print(summary_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
