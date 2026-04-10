#!/usr/bin/env python3
"""
OBS-039 — Reorganization-heavy path context.

Purpose
-------
Test whether reorganization-heavy gateway behavior is governed more by broader
path context than by local launch state alone.

Motivation
----------
OBS-038 showed:
- stable_seam_corridor has a clearer local gateway law
- reorganization_heavy is poorly captured by the same local predictor class

This study focuses only on reorganization_heavy and asks whether crossing vs
non-crossing is better explained by accumulated path context, such as:
- prior time in reversible core
- prior time in directed sector
- prior escape-return contact
- prior shuttle persistence
- recent motif diversity

Prediction task
---------------
Within reorganization_heavy only:

    y = 1  core_to_escape
    y = 0  core_internal

using path-context features computed from the route BEFORE the launch event.

Inputs
------
outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv
outputs/obs026_family_two_field_occupancy/family_two_field_paths.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv

Outputs
-------
outputs/obs039_reorganization_heavy_path_context/
  reorg_path_context_dataset.csv
  reorg_path_context_coefficients.csv
  reorg_path_context_metrics.csv
  obs039_reorganization_heavy_path_context_summary.txt
  obs039_reorganization_heavy_path_context_figure.png
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
    paths_csv: str = (
        "outputs/obs029_seam_escape_channels/seam_escape_steps.csv"
    )
    assignments_csv: str = (
        "outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv"
    )
    outdir: str = "outputs/obs039_reorganization_heavy_path_context"
    random_state: int = 42
    recent_window: int = 5


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
    "route_id",
    "path_id",
    "family",
    "hotspot_class",
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


def safe_mean(s: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(s), errors="coerce")
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


def build_launch_feature_pool(assignments: pd.DataFrame) -> pd.DataFrame:
    a = assignments.copy()
    out = pd.DataFrame({
        "route_class": a["route_class"].astype(str),
        "generator_completed": a["generator_completed"].astype(str),
        "state_a_red": a["state_a_red"].astype(str),
        "state_c_red": a["state_c_red"].astype(str),
        "relational_a": pd.to_numeric(a.get("relational_a", np.nan), errors="coerce"),
        "anisotropy_a": pd.to_numeric(a.get("anisotropy_a", np.nan), errors="coerce"),
        "distance_a": pd.to_numeric(a.get("distance_a", np.nan), errors="coerce"),
    })
    return out


def infer_sector_from_hotspot(row: pd.Series) -> str:
    hotspot = str(row.get("from_hotspot_class", ""))
    if hotspot in {"anisotropy_only", "relational_only", "shared"}:
        return "core"
    if hotspot == "non_hotspot":
        return "escape"

    d2s = pd.to_numeric(row.get("from_distance_to_seam", np.nan), errors="coerce")
    if pd.notna(d2s):
        return "core" if d2s <= 0.15 else "escape"

    committed = pd.to_numeric(row.get("is_committed_escape", np.nan), errors="coerce")
    escaped = pd.to_numeric(row.get("is_escape_step", np.nan), errors="coerce")
    if pd.notna(committed) and committed == 1:
        return "escape"
    if pd.notna(escaped) and escaped == 1:
        return "escape"

    return "core"


def build_path_context_table(paths: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    pid = detect_id_col(paths)
    step_col = detect_step_col(paths)

    p = paths.copy()
    p["route_class"] = p.get("route_class", p.get("path_family", p.get("family", "unknown"))).astype(str)
    p["sector_est"] = p.apply(infer_sector_from_hotspot, axis=1)

    rows = []
    for path_value, grp in p.groupby(pid, sort=False):
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
        grp["recent_shared_count"] = 0
        grp["recent_non_hotspot_count"] = 0

        hotspot_col = "from_hotspot_class"

        for i in range(len(grp)):
            lo = max(0, i - cfg.recent_window)
            hist = grp.iloc[lo:i]
            grp.loc[i, "recent_sector_entropy"] = shannon_entropy(hist["sector_est"])
            grp.loc[i, "recent_hotspot_entropy"] = shannon_entropy(hist[hotspot_col].astype(str))
            grp.loc[i, "recent_shared_count"] = int((hist[hotspot_col] == "shared").sum())
            grp.loc[i, "recent_non_hotspot_count"] = int((hist[hotspot_col] == "non_hotspot").sum())

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
            "cum_core_before",
            "cum_escape_before",
            "core_share_before",
            "escape_share_before",
            "escape_touched_before",
            "recent_sector_entropy",
            "recent_hotspot_entropy",
            "recent_shared_count",
            "recent_non_hotspot_count",
            "runlen_core_before",
            "runlen_escape_before",
        ]
        rows.append(grp[keep_cols])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_dataset(
    crossings: pd.DataFrame,
    path_context: pd.DataFrame,
    launch_pool: pd.DataFrame,
    paths: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    use = crossings[
        (crossings["route_class"].astype(str) == "reorganization_heavy")
        & (crossings["crossing_type"].isin(["core_internal", "core_to_escape"]))
    ].copy()
    use["y_cross"] = (use["crossing_type"] == "core_to_escape").astype(int)

    pid = detect_id_col(paths)
    step_col = detect_step_col(paths)

    path_steps = paths.copy()
    path_steps["route_class"] = path_steps.get("route_class", path_steps.get("path_family", "unknown")).astype(str)

    rows = []
    for i, row in use.reset_index(drop=True).iterrows():
        context_candidates = path_context[path_context["route_class"] == "reorganization_heavy"].copy()
        step_candidates = path_steps[path_steps["route_class"] == "reorganization_heavy"].copy()

        if pid in row.index and pd.notna(row[pid]):
            context_candidates = context_candidates[context_candidates[pid].astype(str) == str(row[pid])]
            step_candidates = step_candidates[step_candidates[pid].astype(str) == str(row[pid])]

        if len(context_candidates) == 0 or len(step_candidates) == 0:
            continue

        n_rep = int(round(float(row["n_compositions"]))) if pd.notna(row["n_compositions"]) else 1
        n_rep = max(n_rep, 1)

        sampled_context = context_candidates.sample(
            n=n_rep,
            replace=(len(context_candidates) < n_rep),
            random_state=cfg.random_state + i,
        ).reset_index(drop=True)

        sampled_steps = step_candidates.sample(
            n=n_rep,
            replace=(len(step_candidates) < n_rep),
            random_state=cfg.random_state + 1000 + i,
        ).reset_index(drop=True)

        for j in range(n_rep):
            c = sampled_context.iloc[j]
            s = sampled_steps.iloc[j]

            rows.append(
                {
                    "route_class": "reorganization_heavy",
                    "crossing_type": str(row["crossing_type"]),
                    "y_cross": int(row["y_cross"]),
                    "prev_generator": str(row["generator_1"]),
                    "prev_state": str(row["src1"]),
                    "prev_target": str(row["tgt1"]),
                    "mean_relational": pd.to_numeric(s.get("from_relational", np.nan), errors="coerce"),
                    "mean_anisotropy": pd.to_numeric(s.get("from_anisotropy", np.nan), errors="coerce"),
                    "mean_distance": pd.to_numeric(s.get("from_distance_to_seam", np.nan), errors="coerce"),
                    "cum_core_before": c["cum_core_before"],
                    "cum_escape_before": c["cum_escape_before"],
                    "core_share_before": c["core_share_before"],
                    "escape_share_before": c["escape_share_before"],
                    "escape_touched_before": c["escape_touched_before"],
                    "recent_sector_entropy": c["recent_sector_entropy"],
                    "recent_hotspot_entropy": c["recent_hotspot_entropy"],
                    "recent_shared_count": c["recent_shared_count"],
                    "recent_non_hotspot_count": c["recent_non_hotspot_count"],
                    "runlen_core_before": c["runlen_core_before"],
                    "runlen_escape_before": c["runlen_escape_before"],
                }
            )

    return pd.DataFrame(rows)


def fit_models(ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    work = ds.copy()

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
        "recent_shared_count",
        "recent_non_hotspot_count",
        "runlen_core_before",
        "runlen_escape_before",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median() if work[col].notna().any() else 0.0)

    # Baseline local model
    X_local = pd.get_dummies(
        work[
            [
                "prev_generator",
                "prev_state",
                "prev_target",
                "mean_relational",
                "mean_anisotropy",
                "mean_distance",
            ]
        ],
        columns=["prev_generator", "prev_state", "prev_target"],
        drop_first=False,
        dtype=float,
    )

    # Context model
    X_context = pd.get_dummies(
        work[
            [
                "prev_generator",
                "prev_state",
                "prev_target",
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
                "recent_shared_count",
                "recent_non_hotspot_count",
                "runlen_core_before",
                "runlen_escape_before",
            ]
        ],
        columns=["prev_generator", "prev_state", "prev_target"],
        drop_first=False,
        dtype=float,
    )

    y = work["y_cross"].to_numpy(dtype=int)

    def fit_one(X: pd.DataFrame, label: str):
        if len(np.unique(y)) < 2:
            coef = pd.DataFrame(columns=["model", "feature", "coefficient", "abs_coefficient"])
            metrics = {"model": label, "auc": float("nan"), "intercept": float("nan"), "n_rows": float(len(work))}
            return coef, metrics

        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            random_state=42,
            max_iter=5000,
        )
        model.fit(X.to_numpy(dtype=float), y)
        probs = model.predict_proba(X.to_numpy(dtype=float))[:, 1]
        auc = float(roc_auc_score(y, probs))
        coef = pd.DataFrame(
            {
                "model": label,
                "feature": X.columns,
                "coefficient": model.coef_[0],
                "abs_coefficient": np.abs(model.coef_[0]),
            }
        ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        metrics = {
            "model": label,
            "auc": auc,
            "intercept": float(model.intercept_[0]),
            "n_rows": float(len(work)),
            "positive_rate": float(work["y_cross"].mean()),
        }
        return coef, metrics

    coef_local, metrics_local = fit_one(X_local, "local_only")
    coef_context, metrics_context = fit_one(X_context, "path_context")
    return coef_local, coef_context, metrics_local, metrics_context


def build_summary(ds: pd.DataFrame, coef_all: pd.DataFrame, metrics_local: dict[str, float], metrics_context: dict[str, float]) -> str:
    delta_auc = metrics_context["auc"] - metrics_local["auc"]

    lines = [
        "=== OBS-039 Reorganization-Heavy Path Context Summary ===",
        "",
        f"n_rows = {int(metrics_context['n_rows'])}",
        f"positive_rate = {metrics_context['positive_rate']:.4f}",
        "",
        "Model comparison",
        f"  local_only_auc  = {metrics_local['auc']:.4f}",
        f"  path_context_auc = {metrics_context['auc']:.4f}",
        f"  delta_auc        = {delta_auc:.4f}",
        "",
        "Interpretive guide",
        "- local_only uses only pre-second-step local state",
        "- path_context adds accumulated prior-sector and prior-pattern features",
        "- positive delta_auc means reorganization-heavy depends on broader path context",
        "",
        "Top path-context features",
    ]

    top = coef_all[coef_all["model"] == "path_context"].head(15)
    for _, row in top.iterrows():
        lines.append(f"  {row['feature']}: coef={row['coefficient']:.6f}")

    lines.extend(
        [
            "",
            "Dataset means",
            f"  mean_relational_cross     = {safe_mean(ds.loc[ds['y_cross'] == 1, 'mean_relational']):.4f}",
            f"  mean_relational_internal  = {safe_mean(ds.loc[ds['y_cross'] == 0, 'mean_relational']):.4f}",
            f"  mean_anisotropy_cross     = {safe_mean(ds.loc[ds['y_cross'] == 1, 'mean_anisotropy']):.4f}",
            f"  mean_anisotropy_internal  = {safe_mean(ds.loc[ds['y_cross'] == 0, 'mean_anisotropy']):.4f}",
            f"  mean_core_share_cross     = {safe_mean(ds.loc[ds['y_cross'] == 1, 'core_share_before']):.4f}",
            f"  mean_core_share_internal  = {safe_mean(ds.loc[ds['y_cross'] == 0, 'core_share_before']):.4f}",
            f"  mean_escape_share_cross   = {safe_mean(ds.loc[ds['y_cross'] == 1, 'escape_share_before']):.4f}",
            f"  mean_escape_share_internal= {safe_mean(ds.loc[ds['y_cross'] == 0, 'escape_share_before']):.4f}",
            f"  mean_recent_entropy_cross = {safe_mean(ds.loc[ds['y_cross'] == 1, 'recent_hotspot_entropy']):.4f}",
            f"  mean_recent_entropy_internal = {safe_mean(ds.loc[ds['y_cross'] == 0, 'recent_hotspot_entropy']):.4f}",
        ]
    )
    return "\n".join(lines)


def render_figure(coef_all: pd.DataFrame, metrics_local: dict[str, float], metrics_context: dict[str, float], ds: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_auc = fig.add_subplot(gs[0, 0])
    ax_coef = fig.add_subplot(gs[0, 1])
    ax_means = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    ax_auc.bar([0, 1], [metrics_local["auc"], metrics_context["auc"]])
    ax_auc.set_xticks([0, 1])
    ax_auc.set_xticklabels(["local_only", "path_context"])
    ax_auc.set_title("Reorganization-heavy AUC", fontsize=14, pad=8)
    ax_auc.grid(alpha=0.15, axis="y")

    top = coef_all[coef_all["model"] == "path_context"].head(10).iloc[::-1]
    ax_coef.barh(np.arange(len(top)), top["coefficient"].to_numpy(dtype=float))
    ax_coef.set_yticks(np.arange(len(top)))
    ax_coef.set_yticklabels(top["feature"].tolist(), fontsize=8)
    ax_coef.set_title("Top path-context coefficients", fontsize=14, pad=8)
    ax_coef.grid(alpha=0.15, axis="x")

    means = pd.DataFrame({
        "feature": ["core_share_before", "escape_share_before", "recent_hotspot_entropy"],
        "cross": [
            safe_mean(ds.loc[ds["y_cross"] == 1, "core_share_before"]),
            safe_mean(ds.loc[ds["y_cross"] == 1, "escape_share_before"]),
            safe_mean(ds.loc[ds["y_cross"] == 1, "recent_hotspot_entropy"]),
        ],
        "internal": [
            safe_mean(ds.loc[ds["y_cross"] == 0, "core_share_before"]),
            safe_mean(ds.loc[ds["y_cross"] == 0, "escape_share_before"]),
            safe_mean(ds.loc[ds["y_cross"] == 0, "recent_hotspot_entropy"]),
        ],
    })
    x = np.arange(len(means))
    width = 0.34
    ax_means.bar(x - width / 2, means["cross"], width, label="cross")
    ax_means.bar(x + width / 2, means["internal"], width, label="internal")
    ax_means.set_xticks(x)
    ax_means.set_xticklabels(means["feature"], rotation=12)
    ax_means.set_title("Path-context mean contrast", fontsize=14, pad=8)
    ax_means.grid(alpha=0.15, axis="y")
    ax_means.legend()

    ax_text.axis("off")
    y = 0.95
    for _, row in coef_all[coef_all["model"] == "path_context"].head(8).iterrows():
        ax_text.text(0.02, y, f"{row['feature']}: {row['coefficient']:.5f}", fontsize=10, family="monospace")
        y -= 0.055
    ax_text.set_title("Coefficient detail", fontsize=14, pad=8)

    delta_auc = metrics_context["auc"] - metrics_local["auc"]
    ax_diag.axis("off")
    text = (
        "OBS-039 diagnostics\n\n"
        f"local-only AUC:\n{metrics_local['auc']:.3f}\n\n"
        f"path-context AUC:\n{metrics_context['auc']:.3f}\n\n"
        f"delta AUC:\n{delta_auc:.3f}\n\n"
        "Interpretation:\n"
        "tests whether the unresolved\n"
        "reorganization-heavy family\n"
        "is governed by broader path\n"
        "context rather than local state."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-039 reorganization-heavy path context", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test reorganization-heavy path-context gateway law.")
    parser.add_argument("--crossings-csv", default=Config.crossings_csv)
    parser.add_argument("--paths-csv", default=Config.paths_csv)
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--recent-window", type=int, default=Config.recent_window)
    args = parser.parse_args()

    cfg = Config(
        crossings_csv=args.crossings_csv,
        paths_csv=args.paths_csv,
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        random_state=args.random_state,
        recent_window=args.recent_window,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    crossings = load_csv(cfg.crossings_csv)
    paths = load_csv(cfg.paths_csv)
    assignments = load_csv(cfg.assignments_csv)

    path_context = build_path_context_table(paths, cfg)
    launch_pool = build_launch_feature_pool(assignments)
    ds = build_dataset(crossings, path_context, launch_pool, paths, cfg)
    coef_local, coef_context, metrics_local, metrics_context = fit_models(ds)
    coef_all = pd.concat([coef_local, coef_context], ignore_index=True)

    ds_csv = outdir / "reorg_path_context_dataset.csv"
    coef_csv = outdir / "reorg_path_context_coefficients.csv"
    metrics_csv = outdir / "reorg_path_context_metrics.csv"
    txt_path = outdir / "obs039_reorganization_heavy_path_context_summary.txt"
    png_path = outdir / "obs039_reorganization_heavy_path_context_figure.png"

    ds.to_csv(ds_csv, index=False)
    coef_all.to_csv(coef_csv, index=False)
    pd.DataFrame([metrics_local, metrics_context]).to_csv(metrics_csv, index=False)
    txt_path.write_text(build_summary(ds, coef_all, metrics_local, metrics_context), encoding="utf-8")
    render_figure(coef_all, metrics_local, metrics_context, ds, png_path)

    print(ds_csv)
    print(coef_csv)
    print(metrics_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
