#!/usr/bin/env python3
"""
OBS-037b — Pre-second-step gateway predictor.

Purpose
-------
Test whether gateway crossing can be predicted BEFORE the second generator is
known.

This corrects OBS-037 by removing leakage from:
- generator_2
- src2 / tgt2
- generator_word including second step
- history_word including second step

Prediction task
---------------
Given only information available before the second step fires, predict whether
the next move will be:

    y = 1  core_to_escape
    y = 0  core_internal

Inputs
------
outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv

Outputs
-------
outputs/obs037b_pre_second_step_gateway_predictor/
  gateway_crossing_pre_second_step_dataset.csv
  gateway_crossing_pre_second_step_coefficients.csv
  gateway_crossing_pre_second_step_family_summary.csv
  obs037b_pre_second_step_gateway_predictor_summary.txt
  obs037b_pre_second_step_gateway_predictor_figure.png
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
    outdir: str = "outputs/obs037b_pre_second_step_gateway_predictor"
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
    "prev_generator",
    "prev_state",
    "prev_target",
    "prelaunch_word",
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
    a["state_a_red"] = a["state_a_red"].astype(str)
    a["state_c_red"] = a["state_c_red"].astype(str)

    out = pd.DataFrame(
        {
            "route_class": a["route_class"],
            "generator_completed": a["generator_completed"],
            "state_a_red": a["state_a_red"],
            "state_c_red": a["state_c_red"],
            "relational_a": pd.to_numeric(a.get("relational_a", np.nan), errors="coerce"),
            "anisotropy_a": pd.to_numeric(a.get("anisotropy_a", np.nan), errors="coerce"),
            "distance_a": pd.to_numeric(a.get("distance_a", np.nan), errors="coerce"),
        }
    )
    return out


def sample_pre_second_step_instances(
    row: pd.Series,
    pool: pd.DataFrame,
    *,
    random_state: int,
) -> pd.DataFrame:
    cls = str(row["route_class"])
    prev_gen = str(row["generator_1"])
    prev_src = str(row["src1"])
    prev_tgt = str(row["tgt1"])

    sub = pool[
        (pool["route_class"] == cls)
        & (pool["generator_completed"] == prev_gen)
        & (pool["state_a_red"] == prev_src)
        & (pool["state_c_red"] == prev_tgt)
    ].copy()

    if len(sub) == 0:
        sub = pool[
            (pool["route_class"] == cls)
            & (pool["generator_completed"] == prev_gen)
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
        sampled = sample_pre_second_step_instances(row, pool, random_state=cfg.random_state + idx)
        if len(sampled) == 0:
            continue

        for _, s in sampled.iterrows():
            prev_generator = str(row["generator_1"])
            prev_state = str(row["src1"]) if pd.notna(row["src1"]) else str(s["state_a_red"])
            prev_target = str(row["tgt1"]) if pd.notna(row["tgt1"]) else str(s["state_c_red"])
            prelaunch_word = f"{prev_state}->{prev_target}"

            rows.append(
                {
                    "route_class": str(row["route_class"]),
                    "crossing_type": str(row["crossing_type"]),
                    "y_cross": int(row["y_cross"]),
                    "prev_generator": prev_generator,
                    "prev_state": prev_state,
                    "prev_target": prev_target,
                    "prelaunch_word": prelaunch_word,
                    "mean_relational": s["relational_a"],
                    "mean_anisotropy": s["anisotropy_a"],
                    "mean_distance": s["distance_a"],
                    "composition_typed": row.get("composition_typed", np.nan),
                }
            )

    return pd.DataFrame(rows)


def fit_model(ds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    X = ds.copy()

    for col in ["mean_relational", "mean_anisotropy", "mean_distance"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)

    design = pd.get_dummies(
        X[
            [
                "route_class",
                "prev_generator",
                "prev_state",
                "prev_target",
                "prelaunch_word",
                "mean_relational",
                "mean_anisotropy",
                "mean_distance",
            ]
        ],
        columns=["route_class", "prev_generator", "prev_state", "prev_target", "prelaunch_word"],
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
        pos = sub[sub["y_cross"] == 1]
        neg = sub[sub["y_cross"] == 0]

        rows.append(
            {
                "route_class": cls,
                "n_rows": int(len(sub)),
                "crossing_rate": float(sub["y_cross"].mean()) if len(sub) else float("nan"),
                "top_prelaunch_cross": pos["prelaunch_word"].value_counts().index[0] if len(pos) else np.nan,
                "top_prelaunch_internal": neg["prelaunch_word"].value_counts().index[0] if len(neg) else np.nan,
                "top_prev_generator_cross": pos["prev_generator"].value_counts().index[0] if len(pos) else np.nan,
                "top_prev_generator_internal": neg["prev_generator"].value_counts().index[0] if len(neg) else np.nan,
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


def build_summary(coef: pd.DataFrame, fam: pd.DataFrame, metrics: dict[str, float]) -> str:
    lines = [
        "=== OBS-037b Pre-Second-Step Gateway Predictor Summary ===",
        "",
        f"n_rows = {int(metrics['n_rows'])}",
        f"positive_rate = {metrics['positive_rate']:.4f}",
        f"auc = {metrics['auc']:.4f}",
        f"intercept = {metrics['intercept']:.6f}",
        "",
        "Discipline",
        "- positive class = core_to_escape crossing",
        "- negative class = core_internal continuation",
        "- uses only information available before generator_2 is known",
        "- excludes generator_2, src2, tgt2, and any two-step word containing the second step",
        "",
        "Top pre-second-step features",
    ]

    for _, row in coef.head(12).iterrows():
        lines.append(f"  {row['feature']}: coef={row['coefficient']:.6f}")

    lines.extend(["", "Family summaries"])
    for _, row in fam.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_rows                    = {int(row['n_rows'])}",
                f"  crossing_rate             = {float(row['crossing_rate']):.4f}",
                f"  top_prelaunch_cross       = {row['top_prelaunch_cross']}",
                f"  top_prelaunch_internal    = {row['top_prelaunch_internal']}",
                f"  top_prev_generator_cross  = {row['top_prev_generator_cross']}",
                f"  top_prev_generator_internal = {row['top_prev_generator_internal']}",
                f"  mean_relational_cross     = {float(row['mean_relational_cross']):.4f}",
                f"  mean_relational_internal  = {float(row['mean_relational_internal']):.4f}",
                f"  mean_anisotropy_cross     = {float(row['mean_anisotropy_cross']):.4f}",
                f"  mean_anisotropy_internal  = {float(row['mean_anisotropy_internal']):.4f}",
                "",
            ]
        )

    return "\n".join(lines)


def render_figure(coef: pd.DataFrame, fam: pd.DataFrame, metrics: dict[str, float], outpath: Path) -> None:
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
    ax_coef.set_title("Top pre-second-step coefficients", fontsize=14, pad=8)
    ax_coef.grid(alpha=0.15, axis="x")

    x = np.arange(len(fam))
    ax_rates.bar(x, fam["crossing_rate"])
    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(fam["route_class"], rotation=12)
    ax_rates.set_title("Crossing rate by family", fontsize=14, pad=8)
    ax_rates.grid(alpha=0.15, axis="y")

    width = 0.34
    ax_fields.bar(x - width / 2, fam["mean_relational_cross"] - fam["mean_relational_internal"], width, label="Δ relational")
    ax_fields.bar(x + width / 2, fam["mean_anisotropy_cross"] - fam["mean_anisotropy_internal"], width, label="Δ anisotropy")
    ax_fields.set_xticks(x)
    ax_fields.set_xticklabels(fam["route_class"], rotation=12)
    ax_fields.set_title("Crossing minus internal prelaunch fields", fontsize=14, pad=8)
    ax_fields.grid(alpha=0.15, axis="y")
    ax_fields.legend()

    ax_top.axis("off")
    y = 0.95
    for _, row in fam.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"cross state: {row['top_prelaunch_cross']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"int state:   {row['top_prelaunch_internal']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"cross gen:   {row['top_prev_generator_cross']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"int gen:     {row['top_prev_generator_internal']}", fontsize=10, family="monospace")
        y -= 0.07
    ax_top.set_title("Family pre-second-step contrast", fontsize=14, pad=8)

    ax_diag.axis("off")
    text = (
        "OBS-037b diagnostics\n\n"
        f"AUC:\n{metrics['auc']:.3f}\n\n"
        f"positive rate:\n{metrics['positive_rate']:.3f}\n\n"
        "Interpretation:\n"
        "this is the first valid\n"
        "history-aware test that\n"
        "stops before the gateway\n"
        "decision itself is known."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-037b pre-second-step gateway predictor", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict gateway crossing before the second step is known.")
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
    coef, metrics = fit_model(ds)
    fam = build_family_summary(ds)

    ds_csv = outdir / "gateway_crossing_pre_second_step_dataset.csv"
    coef_csv = outdir / "gateway_crossing_pre_second_step_coefficients.csv"
    fam_csv = outdir / "gateway_crossing_pre_second_step_family_summary.csv"
    txt_path = outdir / "obs037b_pre_second_step_gateway_predictor_summary.txt"
    png_path = outdir / "obs037b_pre_second_step_gateway_predictor_figure.png"

    ds.to_csv(ds_csv, index=False)
    coef.to_csv(coef_csv, index=False)
    fam.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(coef, fam, metrics), encoding="utf-8")
    render_figure(coef, fam, metrics, png_path)

    print(ds_csv)
    print(coef_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
