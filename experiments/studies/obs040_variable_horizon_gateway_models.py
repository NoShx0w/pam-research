#!/usr/bin/env python3
"""
OBS-040 — Variable-horizon gateway models.

Purpose
-------
Treat temporal depth as a first-class variable and test whether predictive
power saturates at a finite history horizon.

This study asks, for each family:
- does gateway prediction improve as history length k increases?
- if so, at what horizon does performance saturate?
- do different families have different effective memory scales?

Prediction task
---------------
For each family independently:

    y = 1  core_to_escape
    y = 0  core_internal

using features built from the previous k steps of path history.

Families
--------
- stable_seam_corridor
- reorganization_heavy
- branch_exit

Inputs
------
outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv
outputs/obs029_seam_escape_channels/seam_escape_steps.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv

Outputs
-------
outputs/obs040_variable_horizon_gateway_models/
  horizon_metrics.csv
  horizon_coefficients.csv
  family_horizon_summary.csv
  obs040_variable_horizon_gateway_models_summary.txt
  obs040_variable_horizon_gateway_models_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    steps_csv: str = (
        "outputs/obs029_seam_escape_channels/seam_escape_steps.csv"
    )
    assignments_csv: str = (
        "outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv"
    )
    outdir: str = "outputs/obs040_variable_horizon_gateway_models"
    max_k: int = 5
    recent_window: int = 5
    random_state: int = 42
    saturation_tol: float = 0.02


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

TEXT_COLS = {
    "route_class",
    "crossing_type",
    "generator_1",
    "generator_2",
    "src1",
    "tgt1",
    "src2",
    "tgt2",
    "sector_1",
    "sector_2",
    "composition_typed",
    "path_family",
    "path_id",
    "route_id",
    "family",
    "from_hotspot_class",
    "theta_bin",
    "generator_completed",
    "state_a_red",
    "state_c_red",
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


def safe_mean(s: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(s)), errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def shannon_entropy(labels: pd.Series) -> float:
    if len(labels) == 0:
        return 0.0
    p = labels.value_counts(normalize=True, dropna=True).to_numpy(dtype=float)
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p + 1e-12)).sum())


def detect_id_col(df: pd.DataFrame) -> str:
    for col in ["path_id", "route_id", "trajectory_id", "id"]:
        if col in df.columns:
            return col
    raise ValueError("Could not detect path identifier column.")


def detect_step_col(df: pd.DataFrame) -> str:
    for col in ["step", "t", "time", "idx"]:
        if col in df.columns:
            return col
    raise ValueError("Could not detect step/time column.")


def infer_sector(row: pd.Series, seam_threshold: float = 0.15) -> str:
    hotspot = str(row.get("from_hotspot_class", ""))
    if hotspot in {"anisotropy_only", "relational_only", "shared"}:
        return "core"
    if hotspot == "non_hotspot":
        return "escape"

    d2s = pd.to_numeric(row.get("from_distance_to_seam", np.nan), errors="coerce")
    if pd.notna(d2s):
        return "core" if d2s <= seam_threshold else "escape"

    committed = pd.to_numeric(row.get("is_committed_escape", np.nan), errors="coerce")
    escaped = pd.to_numeric(row.get("is_escape_step", np.nan), errors="coerce")
    if pd.notna(committed) and committed == 1:
        return "escape"
    if pd.notna(escaped) and escaped == 1:
        return "escape"

    return "core"


def build_launch_pool(assignments: pd.DataFrame) -> pd.DataFrame:
    a = assignments.copy()
    return pd.DataFrame(
        {
            "route_class": a["route_class"].astype(str),
            "generator_completed": a["generator_completed"].astype(str),
            "state_a_red": a["state_a_red"].astype(str),
            "state_c_red": a["state_c_red"].astype(str),
            "relational_a": pd.to_numeric(a.get("relational_a", np.nan), errors="coerce"),
            "anisotropy_a": pd.to_numeric(a.get("anisotropy_a", np.nan), errors="coerce"),
            "distance_a": pd.to_numeric(a.get("distance_a", np.nan), errors="coerce"),
        }
    )


def build_step_context(steps: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    pid = detect_id_col(steps)
    step_col = detect_step_col(steps)

    s = steps.copy()
    s["route_class"] = s.get("route_class", s.get("path_family", s.get("family", "unknown"))).astype(str)
    s["sector_est"] = s.apply(infer_sector, axis=1)
    s["hotspot_est"] = s.get("from_hotspot_class", pd.Series([""] * len(s))).astype(str)
    s["theta_bin"] = s.get("theta_bin", pd.Series([""] * len(s))).astype(str)

    out_rows = []
    for path_value, grp in s.groupby(pid, sort=False):
        grp = grp.sort_values(step_col).reset_index(drop=True)

        grp["core_flag"] = (grp["sector_est"] == "core").astype(int)
        grp["escape_flag"] = (grp["sector_est"] == "escape").astype(int)

        grp["cum_core_before"] = grp["core_flag"].cumsum().shift(fill_value=0)
        grp["cum_escape_before"] = grp["escape_flag"].cumsum().shift(fill_value=0)
        grp["core_share_before"] = grp["cum_core_before"] / np.maximum(np.arange(len(grp)), 1)
        grp["escape_share_before"] = grp["cum_escape_before"] / np.maximum(np.arange(len(grp)), 1)
        grp["escape_touched_before"] = (grp["cum_escape_before"] > 0).astype(int)

        grp["recent_sector_entropy"] = 0.0
        grp["recent_hotspot_entropy"] = 0.0
        grp["recent_theta_entropy"] = 0.0

        for i in range(len(grp)):
            lo = max(0, i - cfg.recent_window)
            hist = grp.iloc[lo:i]
            grp.loc[i, "recent_sector_entropy"] = shannon_entropy(hist["sector_est"])
            grp.loc[i, "recent_hotspot_entropy"] = shannon_entropy(hist["hotspot_est"])
            grp.loc[i, "recent_theta_entropy"] = shannon_entropy(hist["theta_bin"])

        grp["runlen_core_before"] = 0
        grp["runlen_escape_before"] = 0
        core_run = 0
        esc_run = 0
        prev_sector = None
        for i in range(len(grp)):
            grp.loc[i, "runlen_core_before"] = core_run
            grp.loc[i, "runlen_escape_before"] = esc_run
            cur = grp.loc[i, "sector_est"]
            if cur == prev_sector == "core":
                core_run += 1
            elif cur == "core":
                core_run = 1
                esc_run = 0
            elif cur == prev_sector == "escape":
                esc_run += 1
            else:
                esc_run = 1
                core_run = 0
            prev_sector = cur

        keep_cols = [
            pid,
            step_col,
            "route_class",
            "sector_est",
            "hotspot_est",
            "theta_bin",
            "from_relational",
            "from_anisotropy",
            "from_distance_to_seam",
            "cum_core_before",
            "cum_escape_before",
            "core_share_before",
            "escape_share_before",
            "escape_touched_before",
            "recent_sector_entropy",
            "recent_hotspot_entropy",
            "recent_theta_entropy",
            "runlen_core_before",
            "runlen_escape_before",
        ]
        out_rows.append(grp[keep_cols])

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def sample_launch_candidates(
    crossing_row: pd.Series,
    launch_pool: pd.DataFrame,
    n_rep: int,
    random_state: int,
) -> pd.DataFrame:
    fam = str(crossing_row["route_class"])
    gen = str(crossing_row["generator_1"])
    src = str(crossing_row["src1"])
    tgt = str(crossing_row["tgt1"])

    sub = launch_pool[
        (launch_pool["route_class"] == fam)
        & (launch_pool["generator_completed"] == gen)
        & (launch_pool["state_a_red"] == src)
        & (launch_pool["state_c_red"] == tgt)
    ].copy()

    if len(sub) == 0:
        sub = launch_pool[
            (launch_pool["route_class"] == fam)
            & (launch_pool["generator_completed"] == gen)
        ].copy()

    if len(sub) == 0:
        return sub

    return sub.sample(
        n=n_rep,
        replace=(len(sub) < n_rep),
        random_state=random_state,
    ).reset_index(drop=True)


def horizon_words(hist: pd.DataFrame, k: int) -> dict[str, str | float]:
    rows = {}
    if len(hist) == 0:
        for i in range(1, k + 1):
            rows[f"h{i}_sector"] = "NONE"
            rows[f"h{i}_hotspot"] = "NONE"
            rows[f"h{i}_theta"] = "NONE"
        rows["history_sector_word"] = "NONE"
        rows["history_theta_word"] = "NONE"
        return rows

    tail = hist.tail(k).reset_index(drop=True)
    missing = k - len(tail)

    sectors = ["NONE"] * missing + tail["sector_est"].astype(str).tolist()
    hotspots = ["NONE"] * missing + tail["hotspot_est"].astype(str).tolist()
    thetas = ["NONE"] * missing + tail["theta_bin"].astype(str).tolist()

    for i in range(1, k + 1):
        rows[f"h{i}_sector"] = sectors[-i]
        rows[f"h{i}_hotspot"] = hotspots[-i]
        rows[f"h{i}_theta"] = thetas[-i]

    rows["history_sector_word"] = "|".join(sectors)
    rows["history_theta_word"] = "|".join(thetas)
    return rows


def build_dataset_for_family(
    family: str,
    crossings: pd.DataFrame,
    step_context: pd.DataFrame,
    launch_pool: pd.DataFrame,
    raw_steps: pd.DataFrame,
    k: int,
    cfg: Config,
) -> pd.DataFrame:
    pid = detect_id_col(raw_steps)
    step_col = detect_step_col(raw_steps)

    use = crossings[
        (crossings["route_class"].astype(str) == family)
        & (crossings["crossing_type"].isin(["core_internal", "core_to_escape"]))
    ].copy()
    use["y_cross"] = (use["crossing_type"] == "core_to_escape").astype(int)

    step_context = step_context[step_context["route_class"] == family].copy()
    raw_steps = raw_steps[raw_steps["route_class"].astype(str) == family].copy()

    rows = []
    for i, row in use.reset_index(drop=True).iterrows():
        n_rep = int(round(float(row["n_compositions"]))) if pd.notna(row["n_compositions"]) else 1
        n_rep = max(n_rep, 1)

        sampled_launch = sample_launch_candidates(row, launch_pool, n_rep, cfg.random_state + i)
        if len(sampled_launch) == 0:
            continue

        context_candidates = step_context.copy()
        if pid in row.index and pd.notna(row[pid]):
            context_candidates = context_candidates[context_candidates[pid].astype(str) == str(row[pid])]

        if len(context_candidates) == 0:
            continue

        sampled_context = context_candidates.sample(
            n=n_rep,
            replace=(len(context_candidates) < n_rep),
            random_state=cfg.random_state + 1000 + i,
        ).reset_index(drop=True)

        for j in range(n_rep):
            l = sampled_launch.iloc[j]
            c = sampled_context.iloc[j]

            path_value = c[pid]
            current_step = c[step_col]
            hist = step_context[
                (step_context[pid] == path_value)
                & (step_context[step_col] < current_step)
            ].sort_values(step_col)

            hwords = horizon_words(hist, k)

            rows.append(
                {
                    "route_class": family,
                    "k": k,
                    "crossing_type": str(row["crossing_type"]),
                    "y_cross": int(row["y_cross"]),
                    "prev_generator": str(row["generator_1"]),
                    "prev_state": str(row["src1"]) if pd.notna(row["src1"]) else str(l["state_a_red"]),
                    "prev_target": str(row["tgt1"]) if pd.notna(row["tgt1"]) else str(l["state_c_red"]),
                    "mean_relational": pd.to_numeric(c.get("from_relational", l["relational_a"]), errors="coerce"),
                    "mean_anisotropy": pd.to_numeric(c.get("from_anisotropy", l["anisotropy_a"]), errors="coerce"),
                    "mean_distance": pd.to_numeric(c.get("from_distance_to_seam", l["distance_a"]), errors="coerce"),
                    "cum_core_before": c["cum_core_before"],
                    "cum_escape_before": c["cum_escape_before"],
                    "core_share_before": c["core_share_before"],
                    "escape_share_before": c["escape_share_before"],
                    "escape_touched_before": c["escape_touched_before"],
                    "recent_sector_entropy": c["recent_sector_entropy"],
                    "recent_hotspot_entropy": c["recent_hotspot_entropy"],
                    "recent_theta_entropy": c["recent_theta_entropy"],
                    "runlen_core_before": c["runlen_core_before"],
                    "runlen_escape_before": c["runlen_escape_before"],
                    **hwords,
                }
            )

    return pd.DataFrame(rows)


def fit_family_horizon_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.copy()
    numeric_cols = [
        "mean_relational",
        "mean_anisotropy",
        "mean_distance",
        "cum_core_before",
        "cum_escape_before",
        "core_share_before",
        "escape_share_before",
        "escape_touched_before",
        "recent_sector_entropy",
        "recent_hotspot_entropy",
        "recent_theta_entropy",
        "runlen_core_before",
        "runlen_escape_before",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median() if work[col].notna().any() else 0.0)

    cat_cols = [
        c for c in work.columns
        if c.startswith("h") and (c.endswith("_sector") or c.endswith("_hotspot") or c.endswith("_theta"))
    ] + [
        "history_sector_word",
        "history_theta_word",
        "prev_generator",
        "prev_state",
        "prev_target",
    ]

    X = pd.get_dummies(
        work[numeric_cols + cat_cols],
        columns=cat_cols,
        drop_first=False,
        dtype=float,
    )

    y = work["y_cross"].to_numpy(dtype=int)
    if len(work) == 0 or len(np.unique(y)) < 2:
        coef = pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
        metrics = {"auc": float("nan"), "intercept": float("nan"), "n_rows": float(len(work))}
        return coef, metrics

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        random_state=42,
        max_iter=6000,
    )
    model.fit(X.to_numpy(dtype=float), y)

    probs = model.predict_proba(X.to_numpy(dtype=float))[:, 1]
    auc = float(roc_auc_score(y, probs))

    coef = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc,
        "intercept": float(model.intercept_[0]),
        "n_rows": float(len(work)),
        "positive_rate": float(work["y_cross"].mean()),
    }
    return coef, metrics


def summarize_family_horizons(metrics_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for fam in CLASS_ORDER:
        sub = metrics_df[metrics_df["route_class"] == fam].sort_values("k").copy()
        if len(sub) == 0 or sub["auc"].notna().sum() == 0:
            rows.append(
                {
                    "route_class": fam,
                    "best_k": np.nan,
                    "best_auc": np.nan,
                    "k0_auc": np.nan,
                    "delta_best_vs_k0": np.nan,
                    "saturation_k": np.nan,
                }
            )
            continue

        best_idx = sub["auc"].idxmax()
        best_row = sub.loc[best_idx]
        k0 = sub[sub["k"] == 0]["auc"]
        k0_auc = float(k0.iloc[0]) if len(k0) else float("nan")
        best_auc = float(best_row["auc"])
        best_k = int(best_row["k"])

        saturation_k = np.nan
        for _, r in sub.iterrows():
            if pd.notna(r["auc"]) and (best_auc - float(r["auc"]) <= cfg.saturation_tol):
                saturation_k = int(r["k"])
                break

        rows.append(
            {
                "route_class": fam,
                "best_k": best_k,
                "best_auc": best_auc,
                "k0_auc": k0_auc,
                "delta_best_vs_k0": best_auc - k0_auc if pd.notna(k0_auc) else np.nan,
                "saturation_k": saturation_k,
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(
    metrics_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    family_summary: pd.DataFrame,
    cfg: Config,
) -> str:
    lines = [
        "=== OBS-040 Variable-Horizon Gateway Models Summary ===",
        "",
        "Interpretive guide",
        "- k=0 is the local/context baseline with no explicit history-word depth",
        "- increasing k adds longer temporal depth via prior sector/hotspot/theta words",
        "- saturation_k is the smallest horizon within tolerance of best AUC",
        "",
        "Family horizon summaries",
    ]

    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  k0_auc           = {float(row['k0_auc']):.4f}",
                f"  best_k           = {int(row['best_k']) if pd.notna(row['best_k']) else 'nan'}",
                f"  best_auc         = {float(row['best_auc']):.4f}",
                f"  delta_best_vs_k0 = {float(row['delta_best_vs_k0']):.4f}",
                f"  saturation_k     = {int(row['saturation_k']) if pd.notna(row['saturation_k']) else 'nan'}",
                "",
            ]
        )

    lines.append("Per-family AUC by horizon")
    for fam in CLASS_ORDER:
        sub = metrics_df[metrics_df["route_class"] == fam].sort_values("k")
        vals = ", ".join(f"k={int(r.k)}:{float(r.auc):.4f}" for r in sub.itertuples(index=False) if pd.notna(r.auc))
        lines.append(f"  {fam}: {vals}")

    lines.append("")
    lines.append("Top best-horizon features")
    for fam in CLASS_ORDER:
        best = family_summary[family_summary["route_class"] == fam]
        if len(best) == 0 or pd.isna(best["best_k"].iloc[0]):
            continue
        k = int(best["best_k"].iloc[0])
        sub = coef_df[(coef_df["route_class"] == fam) & (coef_df["k"] == k)].head(6)
        lines.append(f"  {fam} (k={k})")
        for _, r in sub.iterrows():
            lines.append(f"    {r['feature']}: coef={r['coefficient']:.6f}")

    return "\n".join(lines)


def render_figure(
    metrics_df: pd.DataFrame,
    family_summary: pd.DataFrame,
    coef_df: pd.DataFrame,
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_curve = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1])
    ax_sat = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    for fam in CLASS_ORDER:
        sub = metrics_df[metrics_df["route_class"] == fam].sort_values("k")
        ax_curve.plot(sub["k"], sub["auc"], marker="o", label=fam)
    ax_curve.set_xlabel("History horizon k")
    ax_curve.set_ylabel("AUC")
    ax_curve.set_title("AUC vs temporal horizon", fontsize=14, pad=8)
    ax_curve.grid(alpha=0.2)
    ax_curve.legend()

    x = np.arange(len(family_summary))
    ax_gain.bar(x, family_summary["delta_best_vs_k0"])
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_gain.set_title("Best gain over k=0", fontsize=14, pad=8)
    ax_gain.grid(alpha=0.15, axis="y")

    ax_sat.bar(x, family_summary["saturation_k"])
    ax_sat.set_xticks(x)
    ax_sat.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_sat.set_title("Saturation horizon", fontsize=14, pad=8)
    ax_sat.grid(alpha=0.15, axis="y")

    ax_text.axis("off")
    y = 0.95
    for fam in CLASS_ORDER:
        best = family_summary[family_summary["route_class"] == fam]
        if len(best) == 0 or pd.isna(best["best_k"].iloc[0]):
            continue
        k = int(best["best_k"].iloc[0])
        sub = coef_df[(coef_df["route_class"] == fam) & (coef_df["k"] == k)].head(4)
        ax_text.text(0.02, y, f"{fam} (best k={k})", fontsize=12, fontweight="bold")
        y -= 0.06
        for _, r in sub.iterrows():
            ax_text.text(0.04, y, f"{r['feature']}: {r['coefficient']:.4f}", fontsize=9.5, family="monospace")
            y -= 0.045
        y -= 0.04
    ax_text.set_title("Best-horizon features", fontsize=14, pad=8)

    best_overall = family_summary.sort_values("best_auc", ascending=False).iloc[0]
    ax_diag.axis("off")
    text = (
        "OBS-040 diagnostics\n\n"
        f"best family:\n{best_overall['route_class']}\n"
        f"AUC={best_overall['best_auc']:.3f}\n\n"
        f"best horizon:\nk={int(best_overall['best_k']) if pd.notna(best_overall['best_k']) else 'nan'}\n\n"
        "Interpretation:\n"
        "tests whether each family\n"
        "has its own finite memory\n"
        "scale for gateway prediction."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-040 variable-horizon gateway models", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit variable-horizon gateway models by family.")
    parser.add_argument("--crossings-csv", default=Config.crossings_csv)
    parser.add_argument("--steps-csv", default=Config.steps_csv)
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--max-k", type=int, default=Config.max_k)
    parser.add_argument("--recent-window", type=int, default=Config.recent_window)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--saturation-tol", type=float, default=Config.saturation_tol)
    args = parser.parse_args()

    cfg = Config(
        crossings_csv=args.crossings_csv,
        steps_csv=args.steps_csv,
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        max_k=args.max_k,
        recent_window=args.recent_window,
        random_state=args.random_state,
        saturation_tol=args.saturation_tol,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    crossings = load_csv(cfg.crossings_csv)
    raw_steps = load_csv(cfg.steps_csv)
    assignments = load_csv(cfg.assignments_csv)

    step_context = build_step_context(raw_steps, cfg)
    launch_pool = build_launch_pool(assignments)
    raw_steps["route_class"] = raw_steps.get("route_class", raw_steps.get("path_family", "unknown")).astype(str)

    metric_rows = []
    coef_rows = []

    for fam in CLASS_ORDER:
        for k in range(cfg.max_k + 1):
            ds = build_dataset_for_family(
                fam, crossings, step_context, launch_pool, raw_steps, k, cfg
            )
            coef, metrics = fit_family_horizon_model(ds)
            metrics["route_class"] = fam
            metrics["k"] = k
            metric_rows.append(metrics)

            if len(coef):
                coef = coef.copy()
                coef["route_class"] = fam
                coef["k"] = k
                coef_rows.append(coef)

    metrics_df = pd.DataFrame(metric_rows)
    coef_df = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
    family_summary = summarize_family_horizons(metrics_df, cfg)

    metrics_csv = outdir / "horizon_metrics.csv"
    coef_csv = outdir / "horizon_coefficients.csv"
    fam_csv = outdir / "family_horizon_summary.csv"
    txt_path = outdir / "obs040_variable_horizon_gateway_models_summary.txt"
    png_path = outdir / "obs040_variable_horizon_gateway_models_figure.png"

    metrics_df.to_csv(metrics_csv, index=False)
    coef_df.to_csv(coef_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(metrics_df, coef_df, family_summary, cfg), encoding="utf-8")
    render_figure(metrics_df, family_summary, coef_df, png_path)

    print(metrics_csv)
    print(coef_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
