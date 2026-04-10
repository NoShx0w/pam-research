#!/usr/bin/env python3
"""
OBS-036 — Gateway state refinement.

Purpose
-------
Refine the launch-side state alphabet near the reversible-core / directed-escape
boundary.

OBS-035c showed that gateway crossing is weakly predictable from:
- launch generator / state type
- modest anisotropy signal

This suggests the current reduced state alphabet {R, A, L} is still too coarse
near the gateway.

This study builds a refined launch-state partition by splitting launch instances
using local continuous fields, then tests whether the refined state labels
improve boundary prediction.

Core idea
---------
Starting from instance-level launch rows:
- split the low state L into low_aniso / low_rel / low_balanced
- split flank states by local anisotropy intensity and relational intensity
- compare predictive performance of:
    1. coarse state alphabet
    2. refined state alphabet

Inputs
------
outputs/obs035c_instance_level_gateway_predictor/gateway_crossing_instance_dataset.csv

Outputs
-------
outputs/obs036_gateway_state_refinement/
  gateway_state_refined_dataset.csv
  gateway_state_refined_coefficients.csv
  gateway_state_refined_family_summary.csv
  obs036_gateway_state_refinement_summary.txt
  obs036_gateway_state_refinement_figure.png
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
        "outputs/obs035c_instance_level_gateway_predictor/"
        "gateway_crossing_instance_dataset.csv"
    )
    outdir: str = "outputs/obs036_gateway_state_refinement"
    random_state: int = 42
    aniso_quantile: float = 0.67
    rel_quantile: float = 0.67


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

TEXT_COLS = {
    "route_class",
    "crossing_type",
    "launch_generator",
    "launch_state",
    "launch_target",
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


def build_refined_state(
    launch_state: str,
    aniso: float,
    rel: float,
    aniso_hi: float,
    rel_hi: float,
) -> str:
    if pd.isna(aniso):
        aniso = 0.0
    if pd.isna(rel):
        rel = 0.0

    state = str(launch_state)

    if state == "L":
        if aniso >= aniso_hi and rel < rel_hi:
            return "L_aniso"
        if rel >= rel_hi and aniso < aniso_hi:
            return "L_rel"
        if aniso >= aniso_hi and rel >= rel_hi:
            return "L_hot"
        return "L_balanced"

    if state == "A":
        if aniso >= aniso_hi:
            return "A_hot"
        if rel >= rel_hi:
            return "A_rel_tilt"
        return "A_base"

    if state == "R":
        if rel >= rel_hi:
            return "R_hot"
        if aniso >= aniso_hi:
            return "R_aniso_tilt"
        return "R_base"

    return state


def add_refined_states(ds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = ds.copy()

    aniso = pd.to_numeric(out["mean_anisotropy"], errors="coerce")
    rel = pd.to_numeric(out["mean_relational"], errors="coerce")

    aniso_hi = float(aniso.quantile(cfg.aniso_quantile)) if aniso.notna().any() else 0.0
    rel_hi = float(rel.quantile(cfg.rel_quantile)) if rel.notna().any() else 0.0

    out["launch_state_refined"] = [
        build_refined_state(s, a, r, aniso_hi, rel_hi)
        for s, a, r in zip(out["launch_state"], aniso, rel)
    ]
    out["launch_target_refined"] = [
        build_refined_state(s, a, r, aniso_hi, rel_hi)
        for s, a, r in zip(out["launch_target"], aniso, rel)
    ]
    out["aniso_high"] = (aniso >= aniso_hi).astype(int)
    out["rel_high"] = (rel >= rel_hi).astype(int)
    out["aniso_hi_threshold"] = aniso_hi
    out["rel_hi_threshold"] = rel_hi
    return out


def fit_model(
    ds: pd.DataFrame,
    *,
    use_refined_states: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    X = ds.copy()

    for col in ["mean_relational", "mean_anisotropy", "mean_distance"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)

    state_col = "launch_state_refined" if use_refined_states else "launch_state"
    target_col = "launch_target_refined" if use_refined_states else "launch_target"

    design = pd.get_dummies(
        X[
            [
                "route_class",
                "launch_generator",
                state_col,
                target_col,
                "aniso_high",
                "rel_high",
                "mean_relational",
                "mean_anisotropy",
                "mean_distance",
            ]
        ],
        columns=["route_class", "launch_generator", state_col, target_col],
        drop_first=False,
        dtype=float,
    )

    y = X["y_cross"].to_numpy(dtype=int)

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        random_state=42,
        max_iter=3000,
    )
    model.fit(design.to_numpy(dtype=float), y)

    probs = model.predict_proba(design.to_numpy(dtype=float))[:, 1]
    auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan")

    coef = pd.DataFrame(
        {
            "feature": design.columns,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
            "model": "refined" if use_refined_states else "coarse",
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc,
        "intercept": float(model.intercept_[0]),
        "n_rows": float(len(ds)),
        "positive_rate": float(ds["y_cross"].mean()),
        "model": "refined" if use_refined_states else "coarse",
    }
    return coef, metrics


def build_family_summary(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in CLASS_ORDER:
        sub = ds[ds["route_class"] == cls].copy()
        pos = sub[sub["y_cross"] == 1]
        neg = sub[sub["y_cross"] == 0]

        rows.append(
            {
                "route_class": cls,
                "n_rows": int(len(sub)),
                "crossing_rate": float(sub["y_cross"].mean()) if len(sub) else float("nan"),
                "top_refined_state_cross": pos["launch_state_refined"].value_counts().index[0] if len(pos) else np.nan,
                "top_refined_state_internal": neg["launch_state_refined"].value_counts().index[0] if len(neg) else np.nan,
                "mean_relational_cross": safe_mean(pos["mean_relational"]),
                "mean_relational_internal": safe_mean(neg["mean_relational"]),
                "mean_anisotropy_cross": safe_mean(pos["mean_anisotropy"]),
                "mean_anisotropy_internal": safe_mean(neg["mean_anisotropy"]),
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(
    ds: pd.DataFrame,
    coef_all: pd.DataFrame,
    fam: pd.DataFrame,
    coarse_metrics: dict[str, float],
    refined_metrics: dict[str, float],
) -> str:
    delta_auc = refined_metrics["auc"] - coarse_metrics["auc"]

    lines = [
        "=== OBS-036 Gateway State Refinement Summary ===",
        "",
        f"n_rows = {int(refined_metrics['n_rows'])}",
        f"positive_rate = {refined_metrics['positive_rate']:.4f}",
        "",
        "Model comparison",
        f"  coarse_auc  = {coarse_metrics['auc']:.4f}",
        f"  refined_auc = {refined_metrics['auc']:.4f}",
        f"  delta_auc   = {delta_auc:.4f}",
        "",
        "Interpretive guide",
        "- coarse model uses original launch_state / launch_target",
        "- refined model splits low/flank states by anisotropy / relational intensity",
        "- positive delta_auc means refined boundary states carry extra predictive signal",
        "",
        "Top refined-model features",
    ]

    refined_coef = coef_all[coef_all["model"] == "refined"].head(12)
    for _, row in refined_coef.iterrows():
        lines.append(f"  {row['feature']}: coef={row['coefficient']:.6f}")

    lines.extend(["", "Family refined-state summaries"])
    for _, row in fam.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_rows                     = {int(row['n_rows'])}",
                f"  crossing_rate              = {float(row['crossing_rate']):.4f}",
                f"  top_refined_state_cross    = {row['top_refined_state_cross']}",
                f"  top_refined_state_internal = {row['top_refined_state_internal']}",
                f"  mean_relational_cross      = {float(row['mean_relational_cross']):.4f}",
                f"  mean_relational_internal   = {float(row['mean_relational_internal']):.4f}",
                f"  mean_anisotropy_cross      = {float(row['mean_anisotropy_cross']):.4f}",
                f"  mean_anisotropy_internal   = {float(row['mean_anisotropy_internal']):.4f}",
                "",
            ]
        )

    return "\n".join(lines)


def render_figure(
    coef_all: pd.DataFrame,
    fam: pd.DataFrame,
    coarse_metrics: dict[str, float],
    refined_metrics: dict[str, float],
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_auc = fig.add_subplot(gs[0, 0])
    ax_coef = fig.add_subplot(gs[0, 1])
    ax_states = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    # AUC comparison
    auc_vals = [coarse_metrics["auc"], refined_metrics["auc"]]
    ax_auc.bar([0, 1], auc_vals)
    ax_auc.set_xticks([0, 1])
    ax_auc.set_xticklabels(["coarse", "refined"])
    ax_auc.set_title("Boundary prediction AUC", fontsize=14, pad=8)
    ax_auc.grid(alpha=0.15, axis="y")

    # Refined coefficients
    refined_coef = coef_all[coef_all["model"] == "refined"].head(10).iloc[::-1]
    ax_coef.barh(np.arange(len(refined_coef)), refined_coef["coefficient"].to_numpy(dtype=float))
    ax_coef.set_yticks(np.arange(len(refined_coef)))
    ax_coef.set_yticklabels(refined_coef["feature"].tolist(), fontsize=8)
    ax_coef.set_title("Top refined-state coefficients", fontsize=14, pad=8)
    ax_coef.grid(alpha=0.15, axis="x")

    # Family crossing rates
    x = np.arange(len(fam))
    ax_states.bar(x, fam["crossing_rate"])
    ax_states.set_xticks(x)
    ax_states.set_xticklabels(fam["route_class"], rotation=12)
    ax_states.set_title("Crossing rate by family", fontsize=14, pad=8)
    ax_states.grid(alpha=0.15, axis="y")

    # Family state contrasts
    ax_top.axis("off")
    y = 0.95
    for _, row in fam.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"cross:    {row['top_refined_state_cross']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"internal: {row['top_refined_state_internal']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(
            0.04,
            y,
            f"Δaniso: {row['mean_anisotropy_cross'] - row['mean_anisotropy_internal']:.4f}",
            fontsize=10,
            family="monospace",
        )
        y -= 0.045
        ax_top.text(
            0.04,
            y,
            f"Δrel:   {row['mean_relational_cross'] - row['mean_relational_internal']:.4f}",
            fontsize=10,
            family="monospace",
        )
        y -= 0.07
    ax_top.set_title("Refined launch-state contrast", fontsize=14, pad=8)

    delta_auc = refined_metrics["auc"] - coarse_metrics["auc"]
    ax_diag.axis("off")
    text = (
        "OBS-036 diagnostics\n\n"
        f"coarse AUC:\n{coarse_metrics['auc']:.3f}\n\n"
        f"refined AUC:\n{refined_metrics['auc']:.3f}\n\n"
        f"delta AUC:\n{delta_auc:.3f}\n\n"
        "Interpretation:\n"
        "tests whether the gateway\n"
        "law is limited by coarse\n"
        "state resolution rather\n"
        "than by lack of structure."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-036 gateway state refinement", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine launch states near the gateway boundary.")
    parser.add_argument("--dataset-csv", default=Config.dataset_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--aniso-quantile", type=float, default=Config.aniso_quantile)
    parser.add_argument("--rel-quantile", type=float, default=Config.rel_quantile)
    args = parser.parse_args()

    cfg = Config(
        dataset_csv=args.dataset_csv,
        outdir=args.outdir,
        random_state=args.random_state,
        aniso_quantile=args.aniso_quantile,
        rel_quantile=args.rel_quantile,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_csv(cfg.dataset_csv)
    ds = add_refined_states(ds, cfg)

    coarse_coef, coarse_metrics = fit_model(ds, use_refined_states=False)
    refined_coef, refined_metrics = fit_model(ds, use_refined_states=True)

    coef_all = pd.concat([coarse_coef, refined_coef], ignore_index=True)
    fam = build_family_summary(ds)

    ds_csv = outdir / "gateway_state_refined_dataset.csv"
    coef_csv = outdir / "gateway_state_refined_coefficients.csv"
    fam_csv = outdir / "gateway_state_refined_family_summary.csv"
    txt_path = outdir / "obs036_gateway_state_refinement_summary.txt"
    png_path = outdir / "obs036_gateway_state_refinement_figure.png"

    ds.to_csv(ds_csv, index=False)
    coef_all.to_csv(coef_csv, index=False)
    fam.to_csv(fam_csv, index=False)
    txt_path.write_text(
        build_summary(ds, coef_all, fam, coarse_metrics, refined_metrics),
        encoding="utf-8",
    )
    render_figure(coef_all, fam, coarse_metrics, refined_metrics, png_path)

    print(ds_csv)
    print(coef_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
