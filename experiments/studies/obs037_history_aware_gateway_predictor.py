#!/usr/bin/env python3
"""
OBS-037 — History-aware gateway predictor.

Purpose
-------
Test whether core->escape crossing is governed more by short transition history
than by static launch-side state alone.

This extends OBS-035c by adding one-step history context:
- previous generator
- previous source/target states
- previous local fields (approximate, from generator-instance pool)
- a short history word

Prediction task
---------------
y = 1 if composition is classified as core_to_escape
y = 0 if composition is classified as core_internal

Inputs
------
outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv

Outputs
-------
outputs/obs037_history_aware_gateway_predictor/
  gateway_crossing_history_dataset.csv
  gateway_crossing_history_coefficients.csv
  gateway_crossing_history_family_summary.csv
  obs037_history_aware_gateway_predictor_summary.txt
  obs037_history_aware_gateway_predictor_figure.png
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
    outdir: str = "outputs/obs037_history_aware_gateway_predictor"
    random_state: int = 42


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

TEXT_COLS = {
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


def build_instance_pool(assignments: pd.DataFrame) -> pd.DataFrame:
    a = assignments.copy()
    a["route_class"] = a["route_class"].astype(str)
    a["generator_completed"] = a["generator_completed"].astype(str)

    out = pd.DataFrame(
        {
            "route_class": a["route_class"],
            "generator_completed": a["generator_completed"],
            "state_a_red": a["state_a_red"].astype(str),
            "state_c_red": a["state_c_red"].astype(str),
            "relational_a": pd.to_numeric(a.get("relational_a", np.nan), errors="coerce"),
            "anisotropy_a": pd.to_numeric(a.get("anisotropy_a", np.nan), errors="coerce"),
            "distance_a": pd.to_numeric(a.get("distance_a", np.nan), errors="coerce"),
        }
    )
    return out


def sample_launch_instances(
    row: pd.Series,
    pool: pd.DataFrame,
    *,
    random_state: int,
) -> pd.DataFrame:
    cls = str(row["route_class"])
    g1 = str(row["generator_1"])
    src1 = str(row["src1"])
    tgt1 = str(row["tgt1"])

    sub = pool[
        (pool["route_class"] == cls)
        & (pool["generator_completed"] == g1)
        & (pool["state_a_red"] == src1)
        & (pool["state_c_red"] == tgt1)
    ].copy()

    if len(sub) == 0:
        sub = pool[
            (pool["route_class"] == cls)
            & (pool["generator_completed"] == g1)
        ].copy()

    if len(sub) == 0:
        return sub

    n_rep = int(round(float(row["n_compositions"]))) if pd.notna(row["n_compositions"]) else 1
    n_rep = max(n_rep, 1)

    return sub.sample(
        n=n_rep,
        replace=(len(sub) < n_rep),
        random_state=random_state,
    ).reset_index(drop=True)


def build_dataset(crossings: pd.DataFrame, assignments: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    use = crossings[crossings["crossing_type"].isin(["core_internal", "core_to_escape"])].copy()
    use["y_cross"] = (use["crossing_type"] == "core_to_escape").astype(int)

    pool = build_instance_pool(assignments)

    rows = []
    for idx, row in use.reset_index(drop=True).iterrows():
        sampled = sample_launch_instances(row, pool, random_state=cfg.random_state + idx)
        if len(sampled) == 0:
            continue

        for _, s in sampled.iterrows():
            prev_generator = str(row["generator_1"])
            launch_generator = str(row["generator_2"])

            prev_src = str(row["src1"])
            prev_tgt = str(row["tgt1"])
            launch_src = str(row["src2"])
            launch_tgt = str(row["tgt2"])

            history_word = f"{prev_src}->{prev_tgt};{launch_src}->{launch_tgt}"
            generator_word = f"{prev_generator};{launch_generator}"

            rows.append(
                {
                    "route_class": str(row["route_class"]),
                    "crossing_type": str(row["crossing_type"]),
                    "y_cross": int(row["y_cross"]),
                    "prev_generator": prev_generator,
                    "launch_generator": launch_generator,
                    "prev_state": prev_src,
                    "prev_target": prev_tgt,
                    "launch_state": launch_src,
                    "launch_target": launch_tgt,
                    "history_word": history_word,
                    "generator_word": generator_word,
                    # approximate prelaunch local fields from the sampled first-generator instance
                    "mean_relational": s["relational_a"],
                    "mean_anisotropy": s["anisotropy_a"],
                    "mean_distance": s["distance_a"],
                    "composition_typed": row.get("composition_typed", np.nan),
                }
            )

    return pd.DataFrame(rows)


def fit_model(ds: pd.DataFrame, *, use_history: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    X = ds.copy()

    for col in ["mean_relational", "mean_anisotropy", "mean_distance"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)

    features = [
        "route_class",
        "launch_generator",
        "launch_state",
        "launch_target",
        "mean_relational",
        "mean_anisotropy",
        "mean_distance",
    ]
    if use_history:
        features = [
            "route_class",
            "prev_generator",
            "launch_generator",
            "prev_state",
            "prev_target",
            "launch_state",
            "launch_target",
            "history_word",
            "generator_word",
            "mean_relational",
            "mean_anisotropy",
            "mean_distance",
        ]

    cat_cols = [c for c in features if c not in {"mean_relational", "mean_anisotropy", "mean_distance"}]

    design = pd.get_dummies(
        X[features],
        columns=cat_cols,
        drop_first=False,
        dtype=float,
    )

    y = X["y_cross"].to_numpy(dtype=int)

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        random_state=42,
        max_iter=4000,
    )
    model.fit(design.to_numpy(dtype=float), y)

    probs = model.predict_proba(design.to_numpy(dtype=float))[:, 1]
    auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan")

    coef = pd.DataFrame(
        {
            "feature": design.columns,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
            "model": "history" if use_history else "launch_only",
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc,
        "intercept": float(model.intercept_[0]),
        "n_rows": float(len(ds)),
        "positive_rate": float(ds["y_cross"].mean()),
        "model": "history" if use_history else "launch_only",
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
                "top_history_cross": pos["history_word"].value_counts().index[0] if len(pos) else np.nan,
                "top_history_internal": neg["history_word"].value_counts().index[0] if len(neg) else np.nan,
                "top_generator_word_cross": pos["generator_word"].value_counts().index[0] if len(pos) else np.nan,
                "top_generator_word_internal": neg["generator_word"].value_counts().index[0] if len(neg) else np.nan,
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
    coef_all: pd.DataFrame,
    fam: pd.DataFrame,
    launch_metrics: dict[str, float],
    history_metrics: dict[str, float],
) -> str:
    delta_auc = history_metrics["auc"] - launch_metrics["auc"]

    lines = [
        "=== OBS-037 History-Aware Gateway Predictor Summary ===",
        "",
        f"n_rows = {int(history_metrics['n_rows'])}",
        f"positive_rate = {history_metrics['positive_rate']:.4f}",
        "",
        "Model comparison",
        f"  launch_only_auc = {launch_metrics['auc']:.4f}",
        f"  history_auc     = {history_metrics['auc']:.4f}",
        f"  delta_auc       = {delta_auc:.4f}",
        "",
        "Interpretive guide",
        "- launch_only uses current launch-side information only",
        "- history model adds previous generator/state and short history words",
        "- positive delta_auc means short transition memory improves gateway prediction",
        "",
        "Top history-model features",
    ]

    hist_coef = coef_all[coef_all["model"] == "history"].head(12)
    for _, row in hist_coef.iterrows():
        lines.append(f"  {row['feature']}: coef={row['coefficient']:.6f}")

    lines.extend(["", "Family history summaries"])
    for _, row in fam.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_rows                    = {int(row['n_rows'])}",
                f"  crossing_rate             = {float(row['crossing_rate']):.4f}",
                f"  top_history_cross         = {row['top_history_cross']}",
                f"  top_history_internal      = {row['top_history_internal']}",
                f"  top_generator_word_cross  = {row['top_generator_word_cross']}",
                f"  top_generator_word_internal = {row['top_generator_word_internal']}",
                f"  mean_relational_cross     = {float(row['mean_relational_cross']):.4f}",
                f"  mean_relational_internal  = {float(row['mean_relational_internal']):.4f}",
                f"  mean_anisotropy_cross     = {float(row['mean_anisotropy_cross']):.4f}",
                f"  mean_anisotropy_internal  = {float(row['mean_anisotropy_internal']):.4f}",
                "",
            ]
        )

    return "\n".join(lines)


def render_figure(
    coef_all: pd.DataFrame,
    fam: pd.DataFrame,
    launch_metrics: dict[str, float],
    history_metrics: dict[str, float],
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_auc = fig.add_subplot(gs[0, 0])
    ax_coef = fig.add_subplot(gs[0, 1])
    ax_rates = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    auc_vals = [launch_metrics["auc"], history_metrics["auc"]]
    ax_auc.bar([0, 1], auc_vals)
    ax_auc.set_xticks([0, 1])
    ax_auc.set_xticklabels(["launch_only", "history"])
    ax_auc.set_title("Gateway prediction AUC", fontsize=14, pad=8)
    ax_auc.grid(alpha=0.15, axis="y")

    hist_coef = coef_all[coef_all["model"] == "history"].head(10).iloc[::-1]
    ax_coef.barh(np.arange(len(hist_coef)), hist_coef["coefficient"].to_numpy(dtype=float))
    ax_coef.set_yticks(np.arange(len(hist_coef)))
    ax_coef.set_yticklabels(hist_coef["feature"].tolist(), fontsize=8)
    ax_coef.set_title("Top history-aware coefficients", fontsize=14, pad=8)
    ax_coef.grid(alpha=0.15, axis="x")

    x = np.arange(len(fam))
    ax_rates.bar(x, fam["crossing_rate"])
    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(fam["route_class"], rotation=12)
    ax_rates.set_title("Crossing rate by family", fontsize=14, pad=8)
    ax_rates.grid(alpha=0.15, axis="y")

    ax_top.axis("off")
    y = 0.95
    for _, row in fam.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"cross hist: {row['top_history_cross']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"int hist:   {row['top_history_internal']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"cross gen:  {row['top_generator_word_cross']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"int gen:    {row['top_generator_word_internal']}", fontsize=10, family="monospace")
        y -= 0.07
    ax_top.set_title("Family short-history contrast", fontsize=14, pad=8)

    delta_auc = history_metrics["auc"] - launch_metrics["auc"]
    ax_diag.axis("off")
    text = (
        "OBS-037 diagnostics\n\n"
        f"launch-only AUC:\n{launch_metrics['auc']:.3f}\n\n"
        f"history AUC:\n{history_metrics['auc']:.3f}\n\n"
        f"delta AUC:\n{delta_auc:.3f}\n\n"
        "Interpretation:\n"
        "tests whether gateway\n"
        "crossing depends on short\n"
        "transition memory rather\n"
        "than static launch state."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-037 history-aware gateway predictor", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict gateway crossing using short transition history.")
    parser.add_argument("--crossings-csv", default=Config.crossings_csv)
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    args = parser.parse_args()

    cfg = Config(
        crossings_csv=args.crossings_csv,
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        random_state=args.random_state,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    crossings = load_csv(cfg.crossings_csv)
    assignments = load_csv(cfg.assignments_csv)

    ds = build_dataset(crossings, assignments, cfg)
    launch_coef, launch_metrics = fit_model(ds, use_history=False)
    history_coef, history_metrics = fit_model(ds, use_history=True)
    coef_all = pd.concat([launch_coef, history_coef], ignore_index=True)
    fam = build_family_summary(ds)

    ds_csv = outdir / "gateway_crossing_history_dataset.csv"
    coef_csv = outdir / "gateway_crossing_history_coefficients.csv"
    fam_csv = outdir / "gateway_crossing_history_family_summary.csv"
    txt_path = outdir / "obs037_history_aware_gateway_predictor_summary.txt"
    png_path = outdir / "obs037_history_aware_gateway_predictor_figure.png"

    ds.to_csv(ds_csv, index=False)
    coef_all.to_csv(coef_csv, index=False)
    fam.to_csv(fam_csv, index=False)
    txt_path.write_text(
        build_summary(coef_all, fam, launch_metrics, history_metrics),
        encoding="utf-8",
    )
    render_figure(coef_all, fam, launch_metrics, history_metrics, png_path)

    print(ds_csv)
    print(coef_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
