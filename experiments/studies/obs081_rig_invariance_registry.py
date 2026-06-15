#!/usr/bin/env python3
"""
obs081_rig_invariance_registry.py

OBS-081 — Reusable Invariance Registry.

v2 patch
--------
Changes from v1:
1. Adds explicit carrier_role.
2. Adds weak_redundant_carrier status.
3. Separates task-level geometry-needed diagnosis from carrier-level role.
4. Recalibrates non-core carrier status using both OBS-080c and OBS-080d evidence.
5. Avoids calling marginal but evidenced carriers "insufficient_evidence".
6. Makes the registry more suitable as substrate for a future RIG Navigator / TUI.

Purpose
-------
OBS-081 converts OBS-080 contract-sensitivity results into an explicit
registry of reusable invariants, failure localizations, geometry-needed levels,
carrier roles, and repair annotations.

OBS-080 established contract stability of the OBS-078/079 local stability core
across:

    OBS-080a:
      numeric transform contracts

    OBS-080b:
      scale-band contracts

    OBS-080c:
      feature-family contracts

    OBS-080d:
      structural-resampling contracts

OBS-081 asks:

    Can those results be represented as relation-level reusable invariance
    records?

The registry is intended as the bridge from observatory science to a later
RIG Navigator / TUI.

Default output
--------------
    outputs/rig_registry/

Core artifacts
--------------
    rig_relation_registry.csv
    rig_survival_matrix.csv
    rig_failure_localization.csv
    rig_geometry_needed_ladder.csv
    rig_repair_recommendations.csv
    rig_registry_report.md

Scientific guardrail
--------------------
OBS-081 is a synthesis/registry layer over existing OBS-080 artifacts.

It does not:
    introduce new raw data,
    establish external generalization,
    prove causality,
    perform intervention/control,
    or create formal topology.

It summarizes within-table reusable-invariance evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASKS = ["three_way", "C_vs_Cp2", "C_vs_Cp3", "Cp2_vs_Cp3"]

CORE_CONTRACT = "stability_core_3"
GEOMETRY_CONTRACT = "geometry_scores_only"
PATH_CONTRACT = "path_shares_only"
STABILITY_PLUS_GEOMETRY = "stability_plus_geometry"
STRICT_NUMERIC = "strict_numeric_all"
NO_WINDOW = "no_window"

PRIMARY_CONTRACTS = [
    CORE_CONTRACT,
    GEOMETRY_CONTRACT,
    PATH_CONTRACT,
    STABILITY_PLUS_GEOMETRY,
    NO_WINDOW,
    STRICT_NUMERIC,
]

TASK_THRESHOLDS = {
    "three_way": 0.80,
    "C_vs_Cp2": 0.90,
    "C_vs_Cp3": 0.90,
    "Cp2_vs_Cp3": 0.75,
}

TASK_BASELINES = {
    "three_way": 1.0 / 3.0,
    "C_vs_Cp2": 0.50,
    "C_vs_Cp3": 0.50,
    "Cp2_vs_Cp3": 0.50,
}

RIG_STATUS_ORDER = {
    "stable_reusable_invariant": 6,
    "redundant_reusable_invariant": 5,
    "weak_redundant_carrier": 4,
    "context_sensitive_reusable_invariant": 3,
    "fragile_candidate": 2,
    "accidental_relation": 1,
    "insufficient_evidence": 0,
}

CARRIER_ROLES = {
    CORE_CONTRACT: "compact_core_carrier",
    GEOMETRY_CONTRACT: "geometry_sharpening_carrier",
    PATH_CONTRACT: "path_support_carrier",
    STABILITY_PLUS_GEOMETRY: "enriched_geometry_carrier",
    NO_WINDOW: "non_window_redundant_carrier",
    STRICT_NUMERIC: "strict_numeric_reference_carrier",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def join_notes(parts: list[str]) -> str:
    return " ".join([p.strip() for p in parts if isinstance(p, str) and p.strip()])


def best_row(
    df: pd.DataFrame,
    group_cols: list[str],
    score_col: str,
) -> pd.DataFrame:
    if df.empty or score_col not in df.columns:
        return pd.DataFrame()
    return (
        df.sort_values(score_col, ascending=False)
        .groupby(group_cols, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def normalize_score_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Return dataframe with an available balanced-accuracy-like score column name.
    Does not rename in-place unless needed.
    """
    if df.empty:
        return df, None

    score_candidates = [
        "best_mean_balanced_accuracy",
        "balanced_accuracy",
        "mean_balanced_accuracy",
        "actual_balanced_accuracy",
        "best_balanced_accuracy",
    ]

    for col in score_candidates:
        if col in df.columns:
            return df, col

    return df, None


def status_from_score(
    score: float,
    threshold: float,
    ci_low: float | None = None,
    p_above: float | None = None,
) -> str:
    if not np.isfinite(score):
        return "missing"

    if ci_low is not None and np.isfinite(ci_low):
        if ci_low >= threshold:
            return "pass_strong"
        if score >= threshold:
            return "pass_with_lower_tail"
        if score >= threshold - 0.03:
            return "borderline"
        return "weak"

    if p_above is not None and np.isfinite(p_above):
        if score >= threshold and p_above >= 0.95:
            return "pass_strong"
        if score >= threshold and p_above >= 0.75:
            return "pass"
        if score >= threshold:
            return "pass_unstable"
        if score >= threshold - 0.03:
            return "borderline"
        return "weak"

    if score >= threshold:
        return "pass"
    if score >= threshold - 0.03:
        return "borderline"
    return "weak"


def relation_id(task: str, carrier: str) -> str:
    return f"{task}__{carrier}"


def carrier_role(carrier: str) -> str:
    return CARRIER_ROLES.get(carrier, "unknown_carrier")


# -----------------------------------------------------------------------------
# Load OBS summaries
# -----------------------------------------------------------------------------

def load_obs080a(obs080a_dir: Path) -> pd.DataFrame:
    """
    Expected preferred files from OBS-080a:
      obs080a_transform_stability_matrix.csv OR
      obs080a_stability_matrix.csv OR
      obs080a_panel_scores.csv

    Returns normalized survival rows.
    """
    candidates = [
        obs080a_dir / "obs080a_transform_stability_matrix.csv",
        obs080a_dir / "obs080a_stability_matrix.csv",
        obs080a_dir / "obs080a_panel_scores.csv",
    ]

    src = pd.DataFrame()
    src_path = None
    for path in candidates:
        src = read_csv_if_exists(path)
        if not src.empty:
            src_path = path
            break

    if src.empty:
        return pd.DataFrame()

    df = src.copy()

    rename = {}
    if "transform" in df.columns and "transform_contract" not in df.columns:
        rename["transform"] = "transform_contract"
    if "contract" in df.columns and "transform_contract" not in df.columns:
        rename["contract"] = "transform_contract"
    df = df.rename(columns=rename)

    df, score_col = normalize_score_column(df)
    if score_col is None or "task" not in df.columns:
        return pd.DataFrame()

    if "feature_contract" not in df.columns:
        df["feature_contract"] = CORE_CONTRACT

    if "transform_contract" not in df.columns:
        if "numeric_transform" in df.columns:
            df["transform_contract"] = df["numeric_transform"]
        else:
            df["transform_contract"] = "unknown_transform"

    best = best_row(df, ["task", "feature_contract", "transform_contract"], score_col)
    if best.empty:
        return pd.DataFrame()

    best["obs"] = "OBS-080a"
    best["contract_family"] = "numeric_transform"
    best["contract_name"] = best["transform_contract"].astype(str)
    best["score"] = pd.to_numeric(best[score_col], errors="coerce")
    best["threshold"] = best["task"].map(TASK_THRESHOLDS)
    best["status"] = [
        status_from_score(safe_float(s), safe_float(t))
        for s, t in zip(best["score"], best["threshold"])
    ]
    best["source_path"] = str(src_path) if src_path else ""

    keep = [
        "obs",
        "contract_family",
        "contract_name",
        "task",
        "feature_contract",
        "score",
        "threshold",
        "status",
        "source_path",
    ]
    for extra in ["model", "scheme", "best_model", "best_scheme"]:
        if extra in best.columns:
            keep.append(extra)

    return best[[c for c in keep if c in best.columns]]


def load_obs080b(obs080b_dir: Path) -> pd.DataFrame:
    candidates = [
        obs080b_dir / "obs080b_scale_band_stability_matrix.csv",
        obs080b_dir / "obs080b_stability_matrix.csv",
        obs080b_dir / "obs080b_scale_band_scores.csv",
        obs080b_dir / "obs080b_panel_scores.csv",
    ]

    src = pd.DataFrame()
    src_path = None
    for path in candidates:
        src = read_csv_if_exists(path)
        if not src.empty:
            src_path = path
            break

    if src.empty:
        return pd.DataFrame()

    df = src.copy()

    rename = {}
    if "scale_band" in df.columns and "scale_band_contract" not in df.columns:
        rename["scale_band"] = "scale_band_contract"
    if "band" in df.columns and "scale_band_contract" not in df.columns:
        rename["band"] = "scale_band_contract"
    if "row_contract" in df.columns and "scale_band_contract" not in df.columns:
        rename["row_contract"] = "scale_band_contract"
    df = df.rename(columns=rename)

    df, score_col = normalize_score_column(df)
    if score_col is None or "task" not in df.columns:
        return pd.DataFrame()

    if "feature_contract" not in df.columns:
        df["feature_contract"] = CORE_CONTRACT

    if "scale_band_contract" not in df.columns:
        df["scale_band_contract"] = "unknown_scale_band"

    best = best_row(df, ["task", "feature_contract", "scale_band_contract"], score_col)
    if best.empty:
        return pd.DataFrame()

    best["obs"] = "OBS-080b"
    best["contract_family"] = "scale_band"
    best["contract_name"] = best["scale_band_contract"].astype(str)
    best["score"] = pd.to_numeric(best[score_col], errors="coerce")
    best["threshold"] = best["task"].map(TASK_THRESHOLDS)
    best["status"] = [
        status_from_score(safe_float(s), safe_float(t))
        for s, t in zip(best["score"], best["threshold"])
    ]
    best["source_path"] = str(src_path) if src_path else ""

    keep = [
        "obs",
        "contract_family",
        "contract_name",
        "task",
        "feature_contract",
        "score",
        "threshold",
        "status",
        "source_path",
    ]
    for extra in ["model", "scheme", "best_model", "best_scheme"]:
        if extra in best.columns:
            keep.append(extra)

    return best[[c for c in keep if c in best.columns]]


def load_obs080c(obs080c_dir: Path) -> pd.DataFrame:
    path = obs080c_dir / "obs080c_feature_contract_stability_matrix.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame()

    src = df.copy()
    src, score_col = normalize_score_column(src)

    if (
        score_col is None
        or "task" not in src.columns
        or "feature_contract" not in src.columns
    ):
        return pd.DataFrame()

    out = src.copy()
    out["obs"] = "OBS-080c"
    out["contract_family"] = "feature_family"
    out["contract_name"] = out["feature_contract"].astype(str)
    out["score"] = pd.to_numeric(out[score_col], errors="coerce")
    out["threshold"] = out["task"].map(TASK_THRESHOLDS)
    out["ci_low"] = np.nan
    out["p_above_threshold"] = np.nan
    out["status"] = [
        status_from_score(safe_float(s), safe_float(t))
        for s, t in zip(out["score"], out["threshold"])
    ]
    out["source_path"] = str(path)

    keep = [
        "obs",
        "contract_family",
        "contract_name",
        "task",
        "feature_contract",
        "score",
        "threshold",
        "status",
        "source_path",
    ]
    for extra in [
        "best_model",
        "best_scheme",
        "stratified_cv_best_ba",
        "leave_object_out_best_ba",
        "leave_cohort_out_best_ba",
        "leave_transition_out_best_ba",
    ]:
        if extra in out.columns:
            keep.append(extra)

    return out[[c for c in keep if c in out.columns]]


def load_obs080d(obs080d_dir: Path) -> pd.DataFrame:
    path = obs080d_dir / "obs080d_structural_stability_matrix.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return pd.DataFrame()

    required = [
        "task",
        "feature_contract",
        "resampling_contract",
        "best_mean_balanced_accuracy",
        "best_ci95_low_balanced_accuracy",
        "p_above_threshold",
    ]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    out = df.copy()
    out["obs"] = "OBS-080d"
    out["contract_family"] = "structural_resampling"
    out["contract_name"] = out["resampling_contract"].astype(str)
    out["score"] = pd.to_numeric(out["best_mean_balanced_accuracy"], errors="coerce")
    out["threshold"] = out["task"].map(TASK_THRESHOLDS)
    out["ci_low"] = pd.to_numeric(out["best_ci95_low_balanced_accuracy"], errors="coerce")
    out["p_above_threshold"] = pd.to_numeric(out["p_above_threshold"], errors="coerce")
    out["status"] = [
        status_from_score(safe_float(s), safe_float(t), safe_float(ci), safe_float(p))
        for s, t, ci, p in zip(
            out["score"],
            out["threshold"],
            out["ci_low"],
            out["p_above_threshold"],
        )
    ]
    out["source_path"] = str(path)

    keep = [
        "obs",
        "contract_family",
        "contract_name",
        "task",
        "feature_contract",
        "score",
        "threshold",
        "ci_low",
        "p_above_threshold",
        "status",
        "source_path",
    ]
    for extra in ["best_model", "failure_rate", "n_success", "n_fail"]:
        if extra in out.columns:
            keep.append(extra)

    return out[[c for c in keep if c in out.columns]]


# -----------------------------------------------------------------------------
# Survival matrix
# -----------------------------------------------------------------------------

def build_survival_matrix(
    obs080a: pd.DataFrame,
    obs080b: pd.DataFrame,
    obs080c: pd.DataFrame,
    obs080d: pd.DataFrame,
) -> pd.DataFrame:
    parts = [df for df in [obs080a, obs080b, obs080c, obs080d] if not df.empty]

    if not parts:
        return pd.DataFrame(
            columns=[
                "relation_id",
                "obs",
                "contract_family",
                "contract_name",
                "task",
                "feature_contract",
                "score",
                "threshold",
                "status",
                "source_path",
            ]
        )

    out = pd.concat(parts, ignore_index=True, sort=False)
    out["relation_id"] = [
        relation_id(t, c)
        for t, c in zip(out["task"].astype(str), out["feature_contract"].astype(str))
    ]
    out["carrier_role"] = out["feature_contract"].map(carrier_role)

    cols_first = [
        "relation_id",
        "obs",
        "contract_family",
        "contract_name",
        "task",
        "feature_contract",
        "carrier_role",
        "score",
        "threshold",
        "status",
    ]
    rest = [c for c in out.columns if c not in cols_first]
    return out[cols_first + rest]


# -----------------------------------------------------------------------------
# Score helpers
# -----------------------------------------------------------------------------

def feature_contract_score(obs080c: pd.DataFrame, task: str, contract: str) -> float:
    if obs080c.empty:
        return np.nan
    q = obs080c[
        (obs080c["task"] == task)
        & (obs080c["feature_contract"] == contract)
    ]
    if q.empty:
        return np.nan
    return float(pd.to_numeric(q["score"], errors="coerce").max())


def structural_score(obs080d: pd.DataFrame, task: str, contract: str) -> tuple[float, float, float]:
    if obs080d.empty:
        return (np.nan, np.nan, np.nan)
    q = obs080d[
        (obs080d["task"] == task)
        & (obs080d["feature_contract"] == contract)
    ]
    if q.empty:
        return (np.nan, np.nan, np.nan)

    score = float(pd.to_numeric(q["score"], errors="coerce").mean())

    ci_low_series = pd.to_numeric(q.get("ci_low", pd.Series(dtype=float)), errors="coerce")
    ci_low = float(ci_low_series.min()) if not ci_low_series.empty else np.nan

    p_series = pd.to_numeric(q.get("p_above_threshold", pd.Series(dtype=float)), errors="coerce")
    p_above = float(p_series.min()) if not p_series.empty else np.nan

    return score, ci_low, p_above


def survival_summary_for_relation(survival: pd.DataFrame, task: str, carrier: str) -> dict[str, Any]:
    q = survival[
        (survival["task"] == task)
        & (survival["feature_contract"] == carrier)
    ].copy()

    out: dict[str, Any] = {
        "n_survival_rows": int(len(q)),
        "mean_survival_score": np.nan,
        "min_survival_score": np.nan,
    }

    for fam in ["numeric_transform", "scale_band", "feature_family", "structural_resampling"]:
        fq = q[q["contract_family"] == fam]
        if fq.empty:
            out[f"{fam}_n"] = 0
            out[f"{fam}_mean_score"] = np.nan
            out[f"{fam}_min_score"] = np.nan
            out[f"{fam}_status"] = "missing"
            continue

        scores = pd.to_numeric(fq["score"], errors="coerce")
        statuses = fq["status"].astype(str).tolist()
        threshold = TASK_THRESHOLDS[task]

        out[f"{fam}_n"] = int(len(fq))
        out[f"{fam}_mean_score"] = float(scores.mean())
        out[f"{fam}_min_score"] = float(scores.min())

        if all(s in {"pass_strong", "pass"} for s in statuses):
            out[f"{fam}_status"] = "pass"
        elif any(s in {"pass_strong", "pass", "pass_with_lower_tail", "pass_unstable"} for s in statuses):
            out[f"{fam}_status"] = "mixed"
        elif scores.mean() >= threshold:
            out[f"{fam}_status"] = "pass"
        elif scores.mean() >= threshold - 0.03:
            out[f"{fam}_status"] = "borderline"
        else:
            out[f"{fam}_status"] = "weak"

    if not q.empty:
        scores = pd.to_numeric(q["score"], errors="coerce")
        out["mean_survival_score"] = float(scores.mean())
        out["min_survival_score"] = float(scores.min())

    return out


# -----------------------------------------------------------------------------
# Geometry-needed and carrier-role logic
# -----------------------------------------------------------------------------

def infer_task_geometry_needed(task: str, obs080c: pd.DataFrame, obs080d: pd.DataFrame) -> dict[str, Any]:
    """
    Task-level geometry-needed assessment.

    This intentionally describes how much geometry the task/relation needs,
    not the role of each carrier row.
    """
    core_c = feature_contract_score(obs080c, task, CORE_CONTRACT)
    geom_c = feature_contract_score(obs080c, task, GEOMETRY_CONTRACT)
    spg_c = feature_contract_score(obs080c, task, STABILITY_PLUS_GEOMETRY)
    paths_c = feature_contract_score(obs080c, task, PATH_CONTRACT)
    strict_c = feature_contract_score(obs080c, task, STRICT_NUMERIC)
    no_window_c = feature_contract_score(obs080c, task, NO_WINDOW)

    core_d_mean, core_d_ci_low, core_d_p = structural_score(obs080d, task, CORE_CONTRACT)
    threshold = TASK_THRESHOLDS[task]

    level = "unknown"
    label = "insufficient evidence"
    rationale_parts = []

    core_pass = np.isfinite(core_c) and core_c >= threshold
    structural_pass = (
        np.isfinite(core_d_mean)
        and core_d_mean >= threshold
        and (not np.isfinite(core_d_p) or core_d_p >= 0.90)
    )

    if core_pass and structural_pass:
        level = "Level 1"
        label = "compact core sufficient"
        rationale_parts.append("stability_core_3 exceeds threshold and survives structural resampling")
    elif core_pass:
        level = "Level 1/2"
        label = "compact core sufficient but structurally qualified"
        rationale_parts.append("stability_core_3 exceeds threshold but structural evidence is weaker or missing")
    elif np.isfinite(geom_c) and geom_c >= threshold:
        level = "Level 3"
        label = "geometry carrier required"
        rationale_parts.append("geometry_scores_only reaches threshold when compact core does not")
    elif np.isfinite(paths_c) and paths_c >= threshold:
        level = "Level 4"
        label = "path/cohort carrier required"
        rationale_parts.append("path_shares_only reaches threshold when compact core does not")
    elif np.isfinite(strict_c) and strict_c >= threshold:
        level = "Level 5"
        label = "full strict feature geometry required"
        rationale_parts.append("strict_numeric_all reaches threshold when compact contracts do not")
    else:
        level = "Level 6"
        label = "new data or new experiment required"
        rationale_parts.append("tested contracts do not provide stable reusable separation")

    sharpeners = []
    if np.isfinite(geom_c) and np.isfinite(core_c) and geom_c - core_c >= 0.05:
        sharpeners.append("geometry scores sharpen relation")
    if np.isfinite(spg_c) and np.isfinite(core_c) and spg_c - core_c >= 0.05:
        sharpeners.append("stability+geometry sharpens relation")
    if np.isfinite(paths_c) and np.isfinite(core_c) and paths_c - core_c >= 0.03:
        sharpeners.append("path shares sharpen relation")
    if np.isfinite(no_window_c) and np.isfinite(core_c) and no_window_c - core_c >= 0.05:
        sharpeners.append("non-window contracts carry stronger redundant support")
    if np.isfinite(strict_c) and np.isfinite(core_c) and strict_c - core_c >= 0.05:
        sharpeners.append("strict numeric geometry carries stronger redundant support")

    if sharpeners:
        rationale_parts.append("; ".join(sharpeners))

    return {
        "task_geometry_needed_level": level,
        "task_geometry_needed_label": label,
        "task_geometry_needed_rationale": join_notes(rationale_parts),
        "level_1_stability_core_ba": core_c,
        "level_3_geometry_ba": geom_c,
        "level_4_paths_ba": paths_c,
        "level_3_stability_plus_geometry_ba": spg_c,
        "level_5_no_window_ba": no_window_c,
        "level_5_strict_numeric_ba": strict_c,
        "obs080d_core_mean_ba": core_d_mean,
        "obs080d_core_min_ci95_low": core_d_ci_low,
        "obs080d_core_min_p_above_threshold": core_d_p,
    }


def infer_carrier_status(
    task: str,
    carrier: str,
    obs080c: pd.DataFrame,
    obs080d: pd.DataFrame,
    survival_info: dict[str, Any],
    task_geometry: dict[str, Any],
) -> str:
    threshold = TASK_THRESHOLDS[task]
    baseline = TASK_BASELINES[task]

    c_score = feature_contract_score(obs080c, task, carrier)
    d_mean, d_ci_low, d_p = structural_score(obs080d, task, carrier)

    core_c = safe_float(task_geometry.get("level_1_stability_core_ba"))
    core_d = safe_float(task_geometry.get("obs080d_core_mean_ba"))
    core_p = safe_float(task_geometry.get("obs080d_core_min_p_above_threshold"))

    if carrier == CORE_CONTRACT:
        if not np.isfinite(core_c) and not np.isfinite(core_d):
            return "insufficient_evidence"

        if core_c < threshold and core_d < threshold:
            return "fragile_candidate" if max(core_c, core_d) > baseline + 0.10 else "accidental_relation"

        if task in {"C_vs_Cp2", "C_vs_Cp3"}:
            if core_c >= threshold and core_d >= threshold and (not np.isfinite(core_p) or core_p >= 0.95):
                return "stable_reusable_invariant"

        if task in {"Cp2_vs_Cp3", "three_way"}:
            if core_c >= threshold and core_d >= threshold:
                return "context_sensitive_reusable_invariant"

        return "context_sensitive_reusable_invariant"

    # Non-core carrier calibration.
    has_c = np.isfinite(c_score)
    has_d = np.isfinite(d_mean)

    if not has_c and not has_d:
        return "insufficient_evidence"

    # Strong redundant carrier: both feature-family and structural evidence support it,
    # or one support is extremely strong and the other is absent.
    if (
        (has_c and c_score >= threshold and has_d and d_mean >= threshold and (not np.isfinite(d_p) or d_p >= 0.95))
        or (has_c and c_score >= threshold + 0.05 and not has_d)
        or (has_d and d_mean >= threshold + 0.05 and not has_c)
    ):
        return "redundant_reusable_invariant"

    # Weak redundant carrier: evidence exists, near/above threshold in at least one layer,
    # but lower-tail structural support or one family is marginal.
    if (
        (has_c and c_score >= threshold - 0.03)
        or (has_d and d_mean >= threshold - 0.03)
        or (has_c and c_score >= baseline + 0.25)
        or (has_d and d_mean >= baseline + 0.25)
    ):
        return "weak_redundant_carrier"

    # Fragile but non-trivial.
    if (
        (has_c and c_score >= baseline + 0.15)
        or (has_d and d_mean >= baseline + 0.15)
    ):
        return "fragile_candidate"

    return "accidental_relation"


def infer_carrier_role_detail(
    task: str,
    carrier: str,
    obs080c: pd.DataFrame,
    obs080d: pd.DataFrame,
    task_geometry: dict[str, Any],
) -> str:
    c_score = feature_contract_score(obs080c, task, carrier)
    d_mean, _, d_p = structural_score(obs080d, task, carrier)
    threshold = TASK_THRESHOLDS[task]

    if carrier == CORE_CONTRACT:
        if task in {"C_vs_Cp2", "C_vs_Cp3"}:
            return "primary compact carrier"
        return "compact carrier with contextual sensitivity"

    if carrier == GEOMETRY_CONTRACT:
        if np.isfinite(c_score) and c_score >= threshold:
            return "geometry carrier that preserves or sharpens relation"
        return "geometry carrier with partial support"

    if carrier == PATH_CONTRACT:
        if np.isfinite(c_score) and c_score >= threshold and np.isfinite(d_mean) and d_mean >= threshold:
            return "path/cohort support carrier"
        if np.isfinite(d_mean) and d_mean >= threshold:
            return "structurally supported path carrier with feature-family lower-tail sensitivity"
        return "weak path/cohort carrier"

    if carrier == STABILITY_PLUS_GEOMETRY:
        return "enriched stability-geometry carrier"

    if carrier == NO_WINDOW:
        return "non-window redundant carrier"

    if carrier == STRICT_NUMERIC:
        return "strict numeric reference carrier"

    return "unclassified carrier"


# -----------------------------------------------------------------------------
# Failure and repair logic
# -----------------------------------------------------------------------------

def infer_failure_type(contract_family: str, contract_name: str, task: str) -> str:
    if contract_family == "scale_band":
        return "scale_position_sensitivity"
    if contract_family == "feature_family":
        return "feature_projection_sensitivity"
    if contract_family == "structural_resampling":
        if "object" in contract_name:
            return "object_support_sensitivity"
        if "cohort" in contract_name:
            return "cohort_support_sensitivity"
        if "transition" in contract_name:
            return "transition_support_sensitivity"
        return "structural_recomposition_sensitivity"
    if contract_family == "numeric_transform":
        return "numeric_transform_sensitivity"
    if task == "Cp2_vs_Cp3":
        return "sensitive_pair"
    return "general_sensitivity"


def infer_failure_localization(
    task: str,
    carrier: str,
    survival: pd.DataFrame,
    task_geometry: dict[str, Any],
) -> dict[str, Any]:
    q = survival[
        (survival["task"] == task)
        & (survival["feature_contract"] == carrier)
    ].copy()

    rows = []
    if not q.empty:
        threshold = TASK_THRESHOLDS[task]
        q["margin"] = pd.to_numeric(q["score"], errors="coerce") - threshold
        q = q.sort_values("score", ascending=True)

        for _, r in q.head(12).iterrows():
            score = safe_float(r.get("score"))
            status = str(r.get("status", "unknown"))
            fam = str(r.get("contract_family", "unknown"))
            name = str(r.get("contract_name", "unknown"))

            if (
                score < threshold
                or status in {"borderline", "weak", "pass_with_lower_tail", "pass_unstable"}
            ):
                rows.append(
                    {
                        "task": task,
                        "feature_contract": carrier,
                        "carrier_role": carrier_role(carrier),
                        "contract_family": fam,
                        "contract_name": name,
                        "score": score,
                        "threshold": threshold,
                        "margin": score - threshold if np.isfinite(score) else np.nan,
                        "status": status,
                        "failure_type": infer_failure_type(fam, name, task),
                    }
                )

    notes = []
    if task == "Cp2_vs_Cp3":
        notes.append("Cp2_vs_Cp3 remains the sensitive diagnostic pair")
        if safe_float(task_geometry.get("level_3_geometry_ba")) > safe_float(task_geometry.get("level_1_stability_core_ba")) + 0.05:
            notes.append("geometry carrier sharply improves Cp2_vs_Cp3")
    if task == "three_way":
        notes.append("three-way compact core is reusable but structurally more sensitive than C-separating contrasts")
    if task in {"C_vs_Cp2", "C_vs_Cp3"}:
        notes.append("C-separating contrast is stable under compact core")

    return {
        "localized_failure_rows": rows,
        "failure_notes": join_notes(notes),
    }


def infer_repair_recommendation(
    task: str,
    carrier: str,
    status: str,
    task_geometry: dict[str, Any],
) -> dict[str, str]:
    actions = []
    rationale = []

    role = carrier_role(carrier)

    if status == "stable_reusable_invariant":
        actions.append("preserve compact core")
        actions.append("no repair needed")
        rationale.append("compact core is sufficient and structurally stable")

    elif status == "context_sensitive_reusable_invariant":
        actions.append("preserve compact core")
        if task == "Cp2_vs_Cp3":
            actions.append("add geometry-support annotation")
            actions.append("add scale-position sensitivity note")
            rationale.append("Cp2_vs_Cp3 survives under core but geometry contracts sharpen it substantially")
        elif task == "three_way":
            actions.append("prefer enriched geometry for high precision")
            actions.append("annotate structural sensitivity")
            rationale.append("three-way compact core survives but enriched geometry improves precision")
        else:
            actions.append("annotate context sensitivity")
            rationale.append("relation survives but shows qualified support")

    elif status == "redundant_reusable_invariant":
        actions.append("record as redundant carrier")
        actions.append("link to compact stability core")
        rationale.append("carrier preserves or sharpens relation outside the compact core")

    elif status == "weak_redundant_carrier":
        actions.append("record as weak redundant carrier")
        actions.append("do not promote as primary support")
        actions.append("link to localized failure rows")
        rationale.append("carrier has evidence but lower-tail or projection sensitivity remains")

    elif status == "fragile_candidate":
        actions.append("do not promote as reusable invariant")
        actions.append("localize weakest contracts")
        actions.append("test additional data or enriched geometry")
        rationale.append("candidate is above baseline in some layers but not contract-stable enough")

    elif status == "accidental_relation":
        actions.append("discard or quarantine relation")
        rationale.append("relation collapses under tested contracts")

    else:
        actions.append("collect more evidence")
        rationale.append("insufficient artifact coverage")

    return {
        "repair_recommendation": "; ".join(actions),
        "repair_rationale": join_notes(rationale),
    }


# -----------------------------------------------------------------------------
# Registry construction
# -----------------------------------------------------------------------------

def build_registry(
    survival: pd.DataFrame,
    obs080c: pd.DataFrame,
    obs080d: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry_rows = []
    failure_rows = []
    ladder_rows = []
    repair_rows = []

    carriers = [
        CORE_CONTRACT,
        GEOMETRY_CONTRACT,
        PATH_CONTRACT,
        STABILITY_PLUS_GEOMETRY,
        NO_WINDOW,
        STRICT_NUMERIC,
    ]

    for task in TASKS:
        task_geometry = infer_task_geometry_needed(task, obs080c, obs080d)

        for carrier in carriers:
            rid = relation_id(task, carrier)
            survival_info = survival_summary_for_relation(survival, task, carrier)

            status = infer_carrier_status(
                task=task,
                carrier=carrier,
                obs080c=obs080c,
                obs080d=obs080d,
                survival_info=survival_info,
                task_geometry=task_geometry,
            )

            failure_info = infer_failure_localization(
                task=task,
                carrier=carrier,
                survival=survival,
                task_geometry=task_geometry,
            )

            repair = infer_repair_recommendation(
                task=task,
                carrier=carrier,
                status=status,
                task_geometry=task_geometry,
            )

            c_score = feature_contract_score(obs080c, task, carrier)
            d_score, d_ci_low, d_p = structural_score(obs080d, task, carrier)

            role = carrier_role(carrier)
            role_detail = infer_carrier_role_detail(task, carrier, obs080c, obs080d, task_geometry)

            registry_rows.append(
                {
                    "relation_id": rid,
                    "task": task,
                    "carrier": carrier,
                    "carrier_role": role,
                    "carrier_role_detail": role_detail,
                    "rig_status": status,
                    "rig_status_rank": RIG_STATUS_ORDER.get(status, 0),
                    "threshold": TASK_THRESHOLDS[task],
                    "baseline": TASK_BASELINES[task],
                    "obs080c_carrier_ba": c_score,
                    "obs080d_carrier_mean_ba": d_score,
                    "obs080d_carrier_min_ci95_low": d_ci_low,
                    "obs080d_carrier_min_p_above_threshold": d_p,
                    "task_geometry_needed_level": task_geometry["task_geometry_needed_level"],
                    "task_geometry_needed_label": task_geometry["task_geometry_needed_label"],
                    "task_geometry_needed_rationale": task_geometry["task_geometry_needed_rationale"],
                    "failure_notes": failure_info["failure_notes"],
                    "repair_recommendation": repair["repair_recommendation"],
                    "repair_rationale": repair["repair_rationale"],
                    **survival_info,
                    **task_geometry,
                }
            )

            ladder_rows.append(
                {
                    "relation_id": rid,
                    "task": task,
                    "carrier": carrier,
                    "carrier_role": role,
                    "carrier_role_detail": role_detail,
                    "task_geometry_needed_level": task_geometry["task_geometry_needed_level"],
                    "task_geometry_needed_label": task_geometry["task_geometry_needed_label"],
                    "task_geometry_needed_rationale": task_geometry["task_geometry_needed_rationale"],
                    "level_1_stability_core_ba": task_geometry["level_1_stability_core_ba"],
                    "level_3_geometry_ba": task_geometry["level_3_geometry_ba"],
                    "level_4_paths_ba": task_geometry["level_4_paths_ba"],
                    "level_3_stability_plus_geometry_ba": task_geometry["level_3_stability_plus_geometry_ba"],
                    "level_5_no_window_ba": task_geometry["level_5_no_window_ba"],
                    "level_5_strict_numeric_ba": task_geometry["level_5_strict_numeric_ba"],
                }
            )

            repair_rows.append(
                {
                    "relation_id": rid,
                    "task": task,
                    "carrier": carrier,
                    "carrier_role": role,
                    "rig_status": status,
                    "repair_recommendation": repair["repair_recommendation"],
                    "repair_rationale": repair["repair_rationale"],
                    "failure_notes": failure_info["failure_notes"],
                }
            )

            for fr in failure_info["localized_failure_rows"]:
                failure_rows.append({"relation_id": rid, **fr})

    registry = pd.DataFrame(registry_rows)
    if not registry.empty:
        registry = registry.sort_values(
            ["task", "rig_status_rank", "carrier"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    failures = pd.DataFrame(failure_rows)
    ladder = pd.DataFrame(ladder_rows)
    repairs = pd.DataFrame(repair_rows)

    return registry, failures, ladder, repairs


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def build_input_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for key, path in paths.items():
        rows.append(
            {
                "artifact": key,
                "path": str(path),
                "status": "ok" if path.exists() else "missing",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    registry: pd.DataFrame,
    survival: pd.DataFrame,
    failures: pd.DataFrame,
    ladder: pd.DataFrame,
    repairs: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# OBS-081 — Reusable Invariance Registry")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-081 converts OBS-080 contract-sensitivity results into an explicit "
        "registry of reusable invariants, carrier roles, failure localizations, "
        "task-level geometry-needed levels, and repair annotations."
    )
    lines.append("")
    lines.append("OBS-081 is a synthesis layer. It does not introduce new raw data.")
    lines.append("")
    lines.append("## v2 patch")
    lines.append("")
    lines.append("```text")
    lines.append("1. Adds carrier_role.")
    lines.append("2. Adds weak_redundant_carrier status.")
    lines.append("3. Separates task_geometry_needed_* from carrier_role.")
    lines.append("4. Recalibrates non-core carrier status from OBS-080c and OBS-080d evidence.")
    lines.append("5. Avoids treating marginal but evidenced carriers as insufficient evidence.")
    lines.append("```")
    lines.append("")
    lines.append("## Input artifact manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest, max_rows=80))
    lines.append("")
    lines.append("## Registry status vocabulary")
    lines.append("")
    lines.append("```text")
    lines.append("stable_reusable_invariant:")
    lines.append("  compact relation survives broadly with little repair pressure")
    lines.append("")
    lines.append("context_sensitive_reusable_invariant:")
    lines.append("  relation survives but has localized sensitivity or needs annotation")
    lines.append("")
    lines.append("redundant_reusable_invariant:")
    lines.append("  alternate carrier preserves or sharpens a relation already carried by core")
    lines.append("")
    lines.append("weak_redundant_carrier:")
    lines.append("  alternate carrier has evidence but lower-tail or projection sensitivity remains")
    lines.append("")
    lines.append("fragile_candidate:")
    lines.append("  relation appears above baseline but is not contract-stable enough")
    lines.append("")
    lines.append("accidental_relation:")
    lines.append("  relation collapses under tested contracts")
    lines.append("")
    lines.append("insufficient_evidence:")
    lines.append("  artifact coverage is missing or unusable")
    lines.append("```")
    lines.append("")
    lines.append("## Carrier roles")
    lines.append("")
    lines.append("```text")
    lines.append("stability_core_3:")
    lines.append("  compact_core_carrier")
    lines.append("")
    lines.append("geometry_scores_only:")
    lines.append("  geometry_sharpening_carrier")
    lines.append("")
    lines.append("path_shares_only:")
    lines.append("  path_support_carrier")
    lines.append("")
    lines.append("stability_plus_geometry:")
    lines.append("  enriched_geometry_carrier")
    lines.append("")
    lines.append("no_window:")
    lines.append("  non_window_redundant_carrier")
    lines.append("")
    lines.append("strict_numeric_all:")
    lines.append("  strict_numeric_reference_carrier")
    lines.append("```")
    lines.append("")
    lines.append("## Relation registry")
    lines.append("")
    if registry.empty:
        lines.append("_No registry rows._")
    else:
        display_cols = [
            "relation_id",
            "task",
            "carrier",
            "carrier_role",
            "rig_status",
            "obs080c_carrier_ba",
            "obs080d_carrier_mean_ba",
            "task_geometry_needed_level",
            "task_geometry_needed_label",
            "repair_recommendation",
        ]
        lines.append(markdown_table(registry[[c for c in display_cols if c in registry.columns]], max_rows=160))
    lines.append("")
    lines.append("## Core relations")
    lines.append("")
    if registry.empty:
        lines.append("_No core rows._")
    else:
        core = registry[registry["carrier"] == CORE_CONTRACT]
        display_cols = [
            "relation_id",
            "task",
            "rig_status",
            "obs080c_carrier_ba",
            "obs080d_carrier_mean_ba",
            "obs080d_carrier_min_ci95_low",
            "obs080d_carrier_min_p_above_threshold",
            "task_geometry_needed_level",
            "task_geometry_needed_label",
            "failure_notes",
            "repair_recommendation",
        ]
        lines.append(markdown_table(core[[c for c in display_cols if c in core.columns]], max_rows=40))
    lines.append("")
    lines.append("## Survival matrix summary")
    lines.append("")
    if survival.empty:
        lines.append("_No survival rows._")
    else:
        sm = (
            survival.groupby(["obs", "contract_family", "task", "feature_contract"], dropna=False)
            .agg(
                n=("score", "size"),
                mean_score=("score", "mean"),
                min_score=("score", "min"),
                max_score=("score", "max"),
            )
            .reset_index()
            .sort_values(["obs", "task", "feature_contract"])
        )
        lines.append(markdown_table(sm, max_rows=200))
    lines.append("")
    lines.append("## Geometry-needed ladder")
    lines.append("")
    if ladder.empty:
        lines.append("_No ladder rows._")
    else:
        display_cols = [
            "relation_id",
            "task",
            "carrier",
            "carrier_role",
            "task_geometry_needed_level",
            "task_geometry_needed_label",
            "level_1_stability_core_ba",
            "level_3_geometry_ba",
            "level_4_paths_ba",
            "level_3_stability_plus_geometry_ba",
            "level_5_no_window_ba",
            "level_5_strict_numeric_ba",
        ]
        lines.append(markdown_table(ladder[[c for c in display_cols if c in ladder.columns]], max_rows=160))
    lines.append("")
    lines.append("## Failure localization")
    lines.append("")
    if failures.empty:
        lines.append("_No localized failure rows emitted._")
    else:
        lines.append(markdown_table(failures, max_rows=200))
    lines.append("")
    lines.append("## Repair recommendations")
    lines.append("")
    if repairs.empty:
        lines.append("_No repair rows._")
    else:
        lines.append(markdown_table(repairs, max_rows=160))
    lines.append("")
    lines.append("## Canonical OBS-081 interpretation")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-081 turns OBS-080 contract-sensitivity into relation-level reusable")
    lines.append("invariance records.")
    lines.append("")
    lines.append("C_vs_Cp2 and C_vs_Cp3 register as stable reusable invariants under")
    lines.append("the compact stability core.")
    lines.append("")
    lines.append("Cp2_vs_Cp3 registers as a context-sensitive reusable invariant:")
    lines.append("the compact core survives, but geometry and broader contracts sharpen")
    lines.append("the relation.")
    lines.append("")
    lines.append("three_way registers as reusable but structurally and geometrically enriched.")
    lines.append("```")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-081 summarizes within-table reusable-invariance evidence.")
    lines.append("It does not establish external generalization, intervention, or causal control.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("rig_input_manifest.csv")
    lines.append("rig_relation_registry.csv")
    lines.append("rig_survival_matrix.csv")
    lines.append("rig_failure_localization.csv")
    lines.append("rig_geometry_needed_ladder.csv")
    lines.append("rig_repair_recommendations.csv")
    lines.append("rig_registry_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-081")

    (outdir / "rig_registry_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-081 Reusable Invariance Registry.")
    ap.add_argument(
        "--obs080a-dir",
        default="outputs/comparisons/obs080a_stability_core_transform_sensitivity",
    )
    ap.add_argument(
        "--obs080b-dir",
        default="outputs/comparisons/obs080b_stability_core_scale_band_sensitivity",
    )
    ap.add_argument(
        "--obs080c-dir",
        default="outputs/comparisons/obs080c_feature_family_contract_sensitivity",
    )
    ap.add_argument(
        "--obs080d-dir",
        default="outputs/comparisons/obs080d_structural_resampling_contract_sensitivity",
    )
    ap.add_argument(
        "--outdir",
        default="outputs/rig_registry",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outdir = ensure_outdir(args.outdir)

    obs080a_dir = Path(args.obs080a_dir)
    obs080b_dir = Path(args.obs080b_dir)
    obs080c_dir = Path(args.obs080c_dir)
    obs080d_dir = Path(args.obs080d_dir)

    input_manifest = build_input_manifest(
        {
            "obs080a_dir": obs080a_dir,
            "obs080b_dir": obs080b_dir,
            "obs080c_dir": obs080c_dir,
            "obs080d_dir": obs080d_dir,
        }
    )

    obs080a = load_obs080a(obs080a_dir)
    obs080b = load_obs080b(obs080b_dir)
    obs080c = load_obs080c(obs080c_dir)
    obs080d = load_obs080d(obs080d_dir)

    survival = build_survival_matrix(obs080a, obs080b, obs080c, obs080d)

    registry, failures, ladder, repairs = build_registry(
        survival=survival,
        obs080c=obs080c,
        obs080d=obs080d,
    )

    write_csv(input_manifest, outdir / "rig_input_manifest.csv")
    write_csv(registry, outdir / "rig_relation_registry.csv")
    write_csv(survival, outdir / "rig_survival_matrix.csv")
    write_csv(failures, outdir / "rig_failure_localization.csv")
    write_csv(ladder, outdir / "rig_geometry_needed_ladder.csv")
    write_csv(repairs, outdir / "rig_repair_recommendations.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        registry=registry,
        survival=survival,
        failures=failures,
        ladder=ladder,
        repairs=repairs,
    )

    print(f"[OBS-081] wrote RIG registry to {outdir}")
    if survival.empty:
        print("[OBS-081] warning: survival matrix is empty; check OBS-080 input paths/file names.")


if __name__ == "__main__":
    main()
