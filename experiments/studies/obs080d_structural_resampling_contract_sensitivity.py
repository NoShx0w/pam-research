#!/usr/bin/env python3
"""
obs080d_structural_resampling_contract_sensitivity.py

OBS-080d — Structural Resampling Contract Sensitivity.

Purpose
-------
OBS-080d tests whether the OBS-078/079 stability signal survives
structural resampling of interpreted support units.

This differs from OBS-079b:

    OBS-079b:
      row bootstrap of stability coordinates

    OBS-080d:
      structural bootstrap of object / cohort / transition support units

The central question:

    Does the stability signal survive when object, cohort, transition,
    object×cohort, and object×transition support are resampled as units?

Input
-----
    outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/
      obs078a_feature_table.csv

Default output
--------------
    outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/

Feature contracts
-----------------
Focused subset from OBS-080c:

    stability_core_3
    geometry_scores_only
    path_shares_only
    stability_plus_geometry
    strict_numeric_all
    no_window

Tasks
-----
    three_way
    C_vs_Cp2
    C_vs_Cp3
    Cp2_vs_Cp3

Resampling contracts
--------------------
    row_bootstrap
    object_bootstrap
    cohort_bootstrap
    transition_bootstrap
    object_cohort_bootstrap
    object_transition_bootstrap

Each replicate:
    1. sample units with replacement
    2. rebuild resampled table
    3. preserve duplicated sampled units as duplicated rows
    4. evaluate stratified CV on the resampled table
    5. record balanced accuracy

Scientific guardrail
--------------------
OBS-080d is a within-table structural-resampling perturbation.

It is not:
    external validation
    new data validation
    causal proof
    model-independent generalization
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")


MODEL_RANDOM_STATE = 80004

TARGET_COL = "case"

STABILITY_FEATURES = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

TASKS = [
    "three_way",
    "C_vs_Cp2",
    "C_vs_Cp3",
    "Cp2_vs_Cp3",
]

PAIRWISE_CASES = {
    "C_vs_Cp2": ("C", "Cp2"),
    "C_vs_Cp3": ("C", "Cp3"),
    "Cp2_vs_Cp3": ("Cp2", "Cp3"),
}

FOCUSED_FEATURE_CONTRACTS = [
    "stability_core_3",
    "geometry_scores_only",
    "path_shares_only",
    "stability_plus_geometry",
    "strict_numeric_all",
    "no_window",
]

RESAMPLING_CONTRACTS = [
    "row_bootstrap",
    "object_bootstrap",
    "cohort_bootstrap",
    "transition_bootstrap",
    "object_cohort_bootstrap",
    "object_transition_bootstrap",
]

IDENTITY_AND_VALIDATION_COLS = {
    TARGET_COL,
    "row_id",
    "bootstrap_row_id",
    "bootstrap_replicate",
    "bootstrap_unit",
    "bootstrap_unit_sample_index",
    "object",
    "cohort",
    "transition",
    "source",
    "model",
    "case_id",
    "label",
    "candidate_id",
}

STRICT_EXCLUDE_EXACT = {
    "candidate_rank",
    "scale_index_from",
    "scale_index_to",
    "transition_delta",
    "n_paths",
    "n_windows",
    "n_rows",
    "row_count",
    "count",
}

STRICT_EXCLUDE_PATTERNS = [
    r".*_denom_paths$",
    r".*_n_paths$",
    r".*_global_path_share$",
    r".*_global_share$",
    r".*_global_count$",
    r".*_count$",
    r".*_counts$",
    r".*_id$",
    r"^id$",
]

WINDOW_MEAN_FEATURES = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

FAILURE_COLUMNS = [
    "task",
    "feature_contract",
    "resampling_contract",
    "model",
    "replicate",
    "reason",
    "n_rows",
    "n_classes",
    "class_counts",
    "n_units",
    "unit_counts",
]

SCORE_COLUMNS = [
    "task",
    "feature_contract",
    "resampling_contract",
    "model",
    "replicate",
    "n_rows",
    "n_classes",
    "class_counts",
    "n_features",
    "features",
    "n_cv_splits",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "threshold",
    "above_threshold",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    if df is None:
        df = pd.DataFrame(columns=columns or [])
    elif df.empty and columns is not None:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def class_counts_string(y: pd.Series) -> str:
    counts = Counter(pd.Series(y).astype(str))
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def n_classes(y: pd.Series) -> int:
    return int(pd.Series(y).astype(str).nunique(dropna=True))


def transition_label(row: pd.Series) -> str:
    return f"{row.get('scale_index_from')}→{row.get('scale_index_to')}"


def task_cases(task: str) -> tuple[str, ...]:
    if task == "three_way":
        return ("C", "Cp2", "Cp3")
    if task in PAIRWISE_CASES:
        return PAIRWISE_CASES[task]
    raise ValueError(f"Unknown task: {task}")


def labels_for_task(task: str) -> list[str]:
    return list(task_cases(task))


def task_threshold(task: str) -> float:
    if task == "three_way":
        return 0.80
    if task in {"C_vs_Cp2", "C_vs_Cp3"}:
        return 0.90
    if task == "Cp2_vs_Cp3":
        return 0.75
    raise ValueError(task)


def task_baseline(task: str) -> float:
    return 1.0 / len(task_cases(task))


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    require_columns(out, [TARGET_COL, "object", "cohort"] + STABILITY_FEATURES, "feature table")

    if "transition" not in out.columns:
        require_columns(out, ["scale_index_from", "scale_index_to"], "feature table")
        out["transition"] = out.apply(transition_label, axis=1)

    out = out.reset_index(drop=True).copy()
    out["row_id"] = np.arange(len(out))

    return out


def is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    converted = pd.to_numeric(s, errors="coerce")
    return converted.notna().sum() >= max(3, int(0.5 * len(s)))


def matches_any_pattern(name: str, patterns: list[str]) -> bool:
    return any(re.match(p, name) for p in patterns)


def is_strict_excluded_feature(col: str) -> bool:
    if col in IDENTITY_AND_VALIDATION_COLS:
        return True
    if col in STRICT_EXCLUDE_EXACT:
        return True
    if matches_any_pattern(col, STRICT_EXCLUDE_PATTERNS):
        return True
    return False


def strict_numeric_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if is_strict_excluded_feature(col):
            continue
        if is_numeric_series(df[col]):
            cols.append(col)
    return sorted(set(cols))


def existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# -----------------------------------------------------------------------------
# Feature contracts
# -----------------------------------------------------------------------------

def infer_feature_families(df: pd.DataFrame) -> dict[str, list[str]]:
    strict_all = strict_numeric_features(df)

    stability_core = existing(df, STABILITY_FEATURES)

    window_means = [
        c for c in strict_all
        if c in WINDOW_MEAN_FEATURES
        or (
            c.endswith("_mean")
            and any(token in c for token in ["lambda", "delta_d", "bounded"])
        )
    ]
    window_means = sorted(set(stability_core + window_means))

    window_z = [
        c for c in strict_all
        if c.endswith("_z")
        or "_z_" in c
        or c.endswith("_z_global")
        or c.endswith("_zscore")
        or "zscore" in c
    ]

    geometry_tokens = [
        "support_score",
        "overlap_score",
        "shape_score",
        "id_score",
        "pinch_score_total",
        "centroid",
        "drift",
        "jaccard",
        "anisotropy",
        "participation",
        "volume",
        "local_mle",
        "twonn",
        "two_nn",
        "intrinsic_dim",
    ]
    geometry = [
        c for c in strict_all
        if any(tok in c.lower() for tok in geometry_tokens)
    ]

    path_shares = [
        c for c in strict_all
        if c.endswith("_path_share")
        or "__path_share" in c
        or "path_share" in c
    ]

    window_family = sorted(set(window_means + window_z))

    no_window = [
        c for c in strict_all
        if c not in set(window_family)
        and c not in set(stability_core)
        and not any(tok in c for tok in ["lambda", "delta_d", "bounded"])
    ]

    panels = {
        "stability_core_3": stability_core,
        "geometry_scores_only": geometry,
        "path_shares_only": path_shares,
        "stability_plus_geometry": sorted(set(stability_core + geometry)),
        "strict_numeric_all": strict_all,
        "no_window": no_window,
    }

    return panels


def build_feature_contract_manifest(
    df: pd.DataFrame,
    panels: dict[str, list[str]],
    requested_contracts: list[str],
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    rows = []
    active: dict[str, list[str]] = {}

    for contract in requested_contracts:
        if contract not in panels:
            raise ValueError(f"Unknown feature contract: {contract}")

        features = existing(df, panels[contract])
        features = [f for f in features if not is_strict_excluded_feature(f)]

        active[contract] = features

        if not features:
            rows.append(
                {
                    "feature_contract": contract,
                    "feature": "",
                    "feature_index": -1,
                    "n_features": 0,
                    "status": "empty",
                }
            )
        else:
            for i, feature in enumerate(features):
                rows.append(
                    {
                        "feature_contract": contract,
                        "feature": feature,
                        "feature_index": i,
                        "n_features": len(features),
                        "status": "ok",
                    }
                )

    return active, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Models / scoring
# -----------------------------------------------------------------------------

def make_model(name: str) -> Any:
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if name == "logreg":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=MODEL_RANDOM_STATE,
        )

    if name == "tree_depth2":
        return DecisionTreeClassifier(
            max_depth=2,
            class_weight="balanced",
            random_state=MODEL_RANDOM_STATE,
        )

    if name == "rf_depth2":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {name}")


def make_pipeline(model_name: str) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", make_model(model_name)),
        ]
    )


def score_predictions(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    yt = pd.Series(y_true).astype(str)
    yp = pd.Series(y_pred).astype(str)

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(
            f1_score(
                yt,
                yp,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
    }


def confusion_rows(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    task: str,
    feature_contract: str,
    resampling_contract: str,
    model: str,
    replicate: int,
    labels: list[str],
) -> pd.DataFrame:
    cm = confusion_matrix(
        pd.Series(y_true).astype(str),
        pd.Series(y_pred).astype(str),
        labels=labels,
    )

    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            rows.append(
                {
                    "task": task,
                    "feature_contract": feature_contract,
                    "resampling_contract": resampling_contract,
                    "model": model,
                    "replicate": replicate,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(cm[i, j]),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Structural resampling
# -----------------------------------------------------------------------------

def add_structural_unit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_bootstrap_unit"] = out["row_id"].astype(str)
    out["object_bootstrap_unit"] = out["object"].astype(str)
    out["cohort_bootstrap_unit"] = out["cohort"].astype(str)
    out["transition_bootstrap_unit"] = out["transition"].astype(str)
    out["object_cohort_bootstrap_unit"] = (
        out["object"].astype(str) + "||" + out["cohort"].astype(str)
    )
    out["object_transition_bootstrap_unit"] = (
        out["object"].astype(str) + "||" + out["transition"].astype(str)
    )
    return out


def unit_col_for_resampling(contract: str) -> str:
    mapping = {
        "row_bootstrap": "row_bootstrap_unit",
        "object_bootstrap": "object_bootstrap_unit",
        "cohort_bootstrap": "cohort_bootstrap_unit",
        "transition_bootstrap": "transition_bootstrap_unit",
        "object_cohort_bootstrap": "object_cohort_bootstrap_unit",
        "object_transition_bootstrap": "object_transition_bootstrap_unit",
    }
    if contract not in mapping:
        raise ValueError(f"Unknown resampling contract: {contract}")
    return mapping[contract]


def bootstrap_resample(
    df: pd.DataFrame,
    resampling_contract: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    unit_col = unit_col_for_resampling(resampling_contract)
    units = sorted(df[unit_col].dropna().astype(str).unique())

    if not units:
        return pd.DataFrame(), {
            "n_units": 0,
            "unit_counts": "",
        }

    sampled_units = rng.choice(units, size=len(units), replace=True)

    parts = []
    sampled_counter = Counter(sampled_units)

    for sample_index, unit in enumerate(sampled_units):
        part = df[df[unit_col].astype(str) == str(unit)].copy()
        part["bootstrap_unit"] = str(unit)
        part["bootstrap_unit_sample_index"] = sample_index
        parts.append(part)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = out.reset_index(drop=True).copy()
    out["bootstrap_row_id"] = np.arange(len(out))

    meta = {
        "n_units": int(len(units)),
        "n_sampled_units": int(len(sampled_units)),
        "n_unique_sampled_units": int(len(sampled_counter)),
        "unit_counts": ";".join(f"{k}:{v}" for k, v in sorted(sampled_counter.items())),
    }

    return out, meta


def build_resampling_manifest(df: pd.DataFrame, contracts: list[str]) -> pd.DataFrame:
    rows = []

    for contract in contracts:
        unit_col = unit_col_for_resampling(contract)
        units = df[unit_col].dropna().astype(str)
        counts = units.value_counts().sort_index()

        rows.append(
            {
                "resampling_contract": contract,
                "unit_col": unit_col,
                "n_units": int(counts.size),
                "min_rows_per_unit": int(counts.min()) if len(counts) else 0,
                "median_rows_per_unit": float(counts.median()) if len(counts) else np.nan,
                "mean_rows_per_unit": float(counts.mean()) if len(counts) else np.nan,
                "max_rows_per_unit": int(counts.max()) if len(counts) else 0,
                "unit_row_counts": ";".join(f"{k}:{v}" for k, v in counts.items()),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_bootstrap_replicate(
    resampled_df: pd.DataFrame,
    task: str,
    feature_contract: str,
    features: list[str],
    resampling_contract: str,
    model_name: str,
    replicate: int,
    n_splits: int,
    meta: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = task_cases(task)
    labels = labels_for_task(task)
    threshold = task_threshold(task)

    sub = resampled_df[resampled_df[TARGET_COL].isin(cases)].copy()
    y = sub[TARGET_COL].astype(str)

    if not features:
        fail = {
            "task": task,
            "feature_contract": feature_contract,
            "resampling_contract": resampling_contract,
            "model": model_name,
            "replicate": replicate,
            "reason": "empty_feature_contract",
            "n_rows": int(len(sub)),
            "n_classes": n_classes(y) if len(y) else 0,
            "class_counts": class_counts_string(y) if len(y) else "",
            "n_units": meta.get("n_units", np.nan),
            "unit_counts": meta.get("unit_counts", ""),
        }
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([fail], columns=FAILURE_COLUMNS),
            pd.DataFrame(),
        )

    if sub.empty or n_classes(y) < len(cases):
        fail = {
            "task": task,
            "feature_contract": feature_contract,
            "resampling_contract": resampling_contract,
            "model": model_name,
            "replicate": replicate,
            "reason": "missing_task_classes",
            "n_rows": int(len(sub)),
            "n_classes": n_classes(y) if len(y) else 0,
            "class_counts": class_counts_string(y) if len(y) else "",
            "n_units": meta.get("n_units", np.nan),
            "unit_counts": meta.get("unit_counts", ""),
        }
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([fail], columns=FAILURE_COLUMNS),
            pd.DataFrame(),
        )

    min_class = min(Counter(y).values())
    splits = min(n_splits, min_class)

    if splits < 2:
        fail = {
            "task": task,
            "feature_contract": feature_contract,
            "resampling_contract": resampling_contract,
            "model": model_name,
            "replicate": replicate,
            "reason": "not_enough_class_support",
            "n_rows": int(len(sub)),
            "n_classes": n_classes(y),
            "class_counts": class_counts_string(y),
            "n_units": meta.get("n_units", np.nan),
            "unit_counts": meta.get("unit_counts", ""),
        }
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([fail], columns=FAILURE_COLUMNS),
            pd.DataFrame(),
        )

    X = sub[features].apply(pd.to_numeric, errors="coerce")

    cv = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=MODEL_RANDOM_STATE + replicate,
    )

    try:
        pipe = make_pipeline(model_name)
        pred = cross_val_predict(pipe, X, y, cv=cv)

        score = {
            "task": task,
            "feature_contract": feature_contract,
            "resampling_contract": resampling_contract,
            "model": model_name,
            "replicate": replicate,
            "n_rows": int(len(sub)),
            "n_classes": n_classes(y),
            "class_counts": class_counts_string(y),
            "n_features": len(features),
            "features": ",".join(features),
            "n_cv_splits": int(splits),
            **score_predictions(y, pred, labels=labels),
        }
        score["threshold"] = threshold
        score["above_threshold"] = bool(score["balanced_accuracy"] > threshold)

        pred_df = sub[
            [
                "row_id",
                "bootstrap_row_id",
                TARGET_COL,
                "object",
                "cohort",
                "transition",
                "bootstrap_unit",
                "bootstrap_unit_sample_index",
            ]
        ].copy()
        pred_df["task"] = task
        pred_df["feature_contract"] = feature_contract
        pred_df["resampling_contract"] = resampling_contract
        pred_df["model"] = model_name
        pred_df["replicate"] = replicate
        pred_df["prediction"] = pred
        pred_df["correct"] = pred_df[TARGET_COL].astype(str) == pred_df["prediction"].astype(str)

        cm = confusion_rows(
            y_true=y,
            y_pred=pred,
            task=task,
            feature_contract=feature_contract,
            resampling_contract=resampling_contract,
            model=model_name,
            replicate=replicate,
            labels=labels,
        )

        return (
            pd.DataFrame([score], columns=SCORE_COLUMNS),
            pred_df,
            pd.DataFrame(columns=FAILURE_COLUMNS),
            cm,
        )

    except Exception as e:
        fail = {
            "task": task,
            "feature_contract": feature_contract,
            "resampling_contract": resampling_contract,
            "model": model_name,
            "replicate": replicate,
            "reason": str(e),
            "n_rows": int(len(sub)),
            "n_classes": n_classes(y),
            "class_counts": class_counts_string(y),
            "n_units": meta.get("n_units", np.nan),
            "unit_counts": meta.get("unit_counts", ""),
        }
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([fail], columns=FAILURE_COLUMNS),
            pd.DataFrame(),
        )


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def ci_low(x: pd.Series, q: float = 0.025) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(q))


def ci_high(x: pd.Series, q: float = 0.975) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(q))


def summarize_bootstrap_scores(scores: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    keys = ["task", "feature_contract", "resampling_contract", "model"]
    rows = []

    fail_counts = (
        failures.groupby(keys, dropna=False)
        .size()
        .rename("n_fail")
        .reset_index()
        if not failures.empty
        else pd.DataFrame(columns=keys + ["n_fail"])
    )

    for key_vals, g in scores.groupby(keys, dropna=False):
        key = dict(zip(keys, key_vals))
        ba = pd.to_numeric(g["balanced_accuracy"], errors="coerce")
        acc = pd.to_numeric(g["accuracy"], errors="coerce")
        f1 = pd.to_numeric(g["macro_f1"], errors="coerce")
        threshold = float(g["threshold"].iloc[0])
        baseline = task_baseline(key["task"])

        fc = fail_counts.copy()
        for k, v in key.items():
            fc = fc[fc[k] == v]
        n_fail = int(fc["n_fail"].iloc[0]) if not fc.empty else 0
        n_success = int(len(g))
        total = n_success + n_fail

        rows.append(
            {
                **key,
                "baseline_balanced_accuracy": baseline,
                "threshold": threshold,
                "n_success": n_success,
                "n_fail": n_fail,
                "failure_rate": float(n_fail / total) if total else np.nan,
                "mean_accuracy": float(acc.mean()),
                "mean_balanced_accuracy": float(ba.mean()),
                "median_balanced_accuracy": float(ba.median()),
                "std_balanced_accuracy": float(ba.std(ddof=1)) if len(ba) > 1 else np.nan,
                "ci95_low_balanced_accuracy": ci_low(ba),
                "ci95_high_balanced_accuracy": ci_high(ba),
                "p05_balanced_accuracy": ci_low(ba, 0.05),
                "p95_balanced_accuracy": ci_high(ba, 0.95),
                "min_balanced_accuracy": float(ba.min()),
                "max_balanced_accuracy": float(ba.max()),
                "mean_macro_f1": float(f1.mean()),
                "p_above_threshold": float((ba > threshold).mean()),
                "p_above_baseline": float((ba > baseline).mean()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["task", "feature_contract", "resampling_contract", "mean_balanced_accuracy"],
            ascending=[True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    return out


def build_structural_stability_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    q = summary[summary["model"] != "dummy"].copy()
    rows = []

    for (task, feature_contract, resampling_contract), g in q.groupby(
        ["task", "feature_contract", "resampling_contract"],
        dropna=False,
    ):
        best = g.sort_values("mean_balanced_accuracy", ascending=False).head(1)
        rows.append(
            {
                "task": task,
                "feature_contract": feature_contract,
                "resampling_contract": resampling_contract,
                "best_model": best["model"].iloc[0],
                "baseline_balanced_accuracy": float(best["baseline_balanced_accuracy"].iloc[0]),
                "threshold": float(best["threshold"].iloc[0]),
                "n_success": int(best["n_success"].iloc[0]),
                "n_fail": int(best["n_fail"].iloc[0]),
                "failure_rate": float(best["failure_rate"].iloc[0]),
                "best_mean_balanced_accuracy": float(best["mean_balanced_accuracy"].iloc[0]),
                "best_median_balanced_accuracy": float(best["median_balanced_accuracy"].iloc[0]),
                "best_ci95_low_balanced_accuracy": float(best["ci95_low_balanced_accuracy"].iloc[0]),
                "best_ci95_high_balanced_accuracy": float(best["ci95_high_balanced_accuracy"].iloc[0]),
                "best_p05_balanced_accuracy": float(best["p05_balanced_accuracy"].iloc[0]),
                "best_p95_balanced_accuracy": float(best["p95_balanced_accuracy"].iloc[0]),
                "best_min_balanced_accuracy": float(best["min_balanced_accuracy"].iloc[0]),
                "best_max_balanced_accuracy": float(best["max_balanced_accuracy"].iloc[0]),
                "p_above_threshold": float(best["p_above_threshold"].iloc[0]),
                "p_above_baseline": float(best["p_above_baseline"].iloc[0]),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["task", "feature_contract", "best_mean_balanced_accuracy"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return out


def build_core_vs_reference_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()

    rows = []
    refs = [
        "geometry_scores_only",
        "path_shares_only",
        "stability_plus_geometry",
        "strict_numeric_all",
        "no_window",
    ]

    for (task, resampling_contract), g in matrix.groupby(["task", "resampling_contract"], dropna=False):
        core = g[g["feature_contract"] == "stability_core_3"]

        def ba(contract: str) -> float:
            frame = g[g["feature_contract"] == contract]
            if frame.empty:
                return np.nan
            return float(frame["best_mean_balanced_accuracy"].iloc[0])

        core_ba = ba("stability_core_3")

        row = {
            "task": task,
            "resampling_contract": resampling_contract,
            "stability_core_3_ba": core_ba,
            "stability_core_3_ci95_low": float(core["best_ci95_low_balanced_accuracy"].iloc[0])
            if not core.empty
            else np.nan,
            "stability_core_3_p_above_threshold": float(core["p_above_threshold"].iloc[0])
            if not core.empty
            else np.nan,
        }

        for ref in refs:
            ref_ba = ba(ref)
            row[f"{ref}_ba"] = ref_ba
            row[f"core_minus_{ref}_ba"] = (
                core_ba - ref_ba if np.isfinite(core_ba) and np.isfinite(ref_ba) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["task", "resampling_contract"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_input_manifest(outdir: Path, feature_table_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "artifact": "feature_table",
            "path": str(feature_table_path),
            "status": "ok" if feature_table_path.exists() else "missing",
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
        }
    ]
    out = pd.DataFrame(rows)
    write_csv(out, outdir / "obs080d_input_manifest.csv")
    return out


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    resampling_manifest: pd.DataFrame,
    scores_summary: pd.DataFrame,
    matrix: pd.DataFrame,
    core_reference: pd.DataFrame,
    failures: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    lines = []

    lines.append("# OBS-080d — Structural Resampling Contract Sensitivity")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-080d tests whether the OBS-078/079 stability signal survives "
        "structural resampling of interpreted support units."
    )
    lines.append("")
    lines.append("This is distinct from row bootstrap:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-079b:")
    lines.append("  row bootstrap of stability coordinates")
    lines.append("")
    lines.append("OBS-080d:")
    lines.append("  structural bootstrap of object / cohort / transition support units")
    lines.append("```")
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest))
    lines.append("")
    lines.append("## Feature-contract manifest summary")
    lines.append("")
    fm = (
        feature_manifest.groupby(["feature_contract", "status"], dropna=False)
        .agg(n_features=("feature", lambda x: int((pd.Series(x).astype(str) != "").sum())))
        .reset_index()
        .sort_values(["feature_contract", "status"])
    )
    lines.append(markdown_table(fm, max_rows=80))
    lines.append("")
    lines.append("## Resampling manifest")
    lines.append("")
    lines.append(markdown_table(resampling_manifest, max_rows=80))
    lines.append("")
    lines.append("## Bootstrap settings")
    lines.append("")
    lines.append("```text")
    lines.append(f"n_bootstrap = {n_bootstrap}")
    lines.append("evaluation = stratified_cv on each resampled table")
    lines.append("```")
    lines.append("")
    lines.append("## Structural stability matrix")
    lines.append("")
    if matrix.empty:
        lines.append("_No structural stability rows._")
    else:
        display_cols = [
            "task",
            "feature_contract",
            "resampling_contract",
            "best_model",
            "threshold",
            "n_success",
            "n_fail",
            "failure_rate",
            "best_mean_balanced_accuracy",
            "best_ci95_low_balanced_accuracy",
            "best_ci95_high_balanced_accuracy",
            "best_p05_balanced_accuracy",
            "best_p95_balanced_accuracy",
            "p_above_threshold",
        ]
        lines.append(markdown_table(matrix[display_cols], max_rows=220))
    lines.append("")
    lines.append("## Core vs reference summary")
    lines.append("")
    if core_reference.empty:
        lines.append("_No core/reference rows._")
    else:
        lines.append(markdown_table(core_reference, max_rows=160))
    lines.append("")
    lines.append("## Full bootstrap score summary")
    lines.append("")
    if scores_summary.empty:
        lines.append("_No bootstrap score summary rows._")
    else:
        display = scores_summary[
            [
                "task",
                "feature_contract",
                "resampling_contract",
                "model",
                "n_success",
                "n_fail",
                "failure_rate",
                "mean_balanced_accuracy",
                "median_balanced_accuracy",
                "ci95_low_balanced_accuracy",
                "ci95_high_balanced_accuracy",
                "p_above_threshold",
            ]
        ].sort_values(
            ["task", "feature_contract", "resampling_contract", "mean_balanced_accuracy"],
            ascending=[True, True, True, False],
        )
        lines.append(markdown_table(display, max_rows=260))
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    if failures.empty:
        lines.append("_No failures._")
    else:
        lines.append(markdown_table(failures, max_rows=160))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("Strong structural-resampling pass:")
    lines.append("")
    lines.append("```text")
    lines.append("p_above_threshold ≈ 1.0")
    lines.append("ci95_low remains high")
    lines.append("failure_rate ≈ 0")
    lines.append("stability_core_3 remains competitive with reference contracts")
    lines.append("```")
    lines.append("")
    lines.append("Expected nuanced pattern:")
    lines.append("")
    lines.append("```text")
    lines.append("C_vs_Cp2 and C_vs_Cp3 should be most stable.")
    lines.append("Cp2_vs_Cp3 should remain the sensitive diagnostic pair.")
    lines.append("geometry_scores_only / stability_plus_geometry may outperform the compact core.")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-080d is within-table structural resampling.")
    lines.append("It is not external validation or causal proof.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs080d_input_manifest.csv")
    lines.append("obs080d_feature_contract_manifest.csv")
    lines.append("obs080d_resampling_manifest.csv")
    lines.append("obs080d_bootstrap_scores.csv")
    lines.append("obs080d_bootstrap_predictions.csv")
    lines.append("obs080d_bootstrap_confusion_matrices.csv")
    lines.append("obs080d_bootstrap_failures.csv")
    lines.append("obs080d_bootstrap_summary.csv")
    lines.append("obs080d_structural_stability_matrix.csv")
    lines.append("obs080d_core_vs_reference_summary.csv")
    lines.append("obs080d_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-080d")

    (outdir / "obs080d_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-080d structural resampling contract sensitivity.")
    ap.add_argument(
        "--feature-table",
        default=(
            "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/"
            "obs078a_feature_table.csv"
        ),
    )
    ap.add_argument(
        "--outdir",
        default="outputs/comparisons/obs080d_structural_resampling_contract_sensitivity",
    )
    ap.add_argument(
        "--feature-contracts",
        default=",".join(FOCUSED_FEATURE_CONTRACTS),
    )
    ap.add_argument(
        "--resampling-contracts",
        default=",".join(RESAMPLING_CONTRACTS),
    )
    ap.add_argument(
        "--tasks",
        default=",".join(TASKS),
    )
    ap.add_argument(
        "--models",
        default="logreg,tree_depth2,rf_depth2,dummy",
    )
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    ap.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write per-row bootstrap predictions. This can be large.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outdir = ensure_outdir(args.outdir)
    feature_table_path = Path(args.feature_table)

    if not feature_table_path.exists():
        raise FileNotFoundError(feature_table_path)

    df = pd.read_csv(feature_table_path)
    df = prepare_columns(df)
    df = add_structural_unit_columns(df)

    numeric_cols = strict_numeric_features(df)
    df = safe_numeric(df, numeric_cols)

    requested_contracts = [x.strip() for x in args.feature_contracts.split(",") if x.strip()]
    requested_resampling = [x.strip() for x in args.resampling_contracts.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")

    unknown_resampling = sorted(set(requested_resampling) - set(RESAMPLING_CONTRACTS))
    if unknown_resampling:
        raise ValueError(f"Unknown resampling contracts: {unknown_resampling}")

    inferred_panels = infer_feature_families(df)
    feature_contracts, feature_manifest = build_feature_contract_manifest(
        df=df,
        panels=inferred_panels,
        requested_contracts=requested_contracts,
    )

    input_manifest = write_input_manifest(outdir, feature_table_path, df)
    write_csv(feature_manifest, outdir / "obs080d_feature_contract_manifest.csv")

    resampling_manifest = build_resampling_manifest(df, requested_resampling)
    write_csv(resampling_manifest, outdir / "obs080d_resampling_manifest.csv")

    score_parts = []
    pred_parts = []
    failure_parts = []
    cm_parts = []

    run_index = 0

    for resampling_contract in requested_resampling:
        for replicate in range(args.n_bootstrap):
            rng = np.random.default_rng(args.seed + 100000 * run_index + replicate)
            resampled_df, meta = bootstrap_resample(df, resampling_contract, rng)

            for feature_contract, features in feature_contracts.items():
                for task in tasks:
                    for model_name in models:
                        run_index += 1

                        scores, preds, fails, cms = evaluate_bootstrap_replicate(
                            resampled_df=resampled_df,
                            task=task,
                            feature_contract=feature_contract,
                            features=features,
                            resampling_contract=resampling_contract,
                            model_name=model_name,
                            replicate=replicate,
                            n_splits=args.n_splits,
                            meta=meta,
                        )

                        if not scores.empty:
                            score_parts.append(scores)
                        if args.write_predictions and not preds.empty:
                            pred_parts.append(preds)
                        if not fails.empty:
                            failure_parts.append(fails)
                        if not cms.empty:
                            cm_parts.append(cms)

    scores_df = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame(columns=SCORE_COLUMNS)
    failures_df = (
        pd.concat(failure_parts, ignore_index=True)
        if failure_parts
        else pd.DataFrame(columns=FAILURE_COLUMNS)
    )
    cm_df = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()
    preds_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    summary_df = summarize_bootstrap_scores(scores_df, failures_df)
    matrix_df = build_structural_stability_matrix(summary_df)
    core_ref_df = build_core_vs_reference_summary(matrix_df)

    write_csv(scores_df, outdir / "obs080d_bootstrap_scores.csv", columns=SCORE_COLUMNS)
    write_csv(failures_df, outdir / "obs080d_bootstrap_failures.csv", columns=FAILURE_COLUMNS)
    write_csv(summary_df, outdir / "obs080d_bootstrap_summary.csv")
    write_csv(matrix_df, outdir / "obs080d_structural_stability_matrix.csv")
    write_csv(core_ref_df, outdir / "obs080d_core_vs_reference_summary.csv")
    write_csv(cm_df, outdir / "obs080d_bootstrap_confusion_matrices.csv")

    if args.write_predictions:
        write_csv(preds_df, outdir / "obs080d_bootstrap_predictions.csv")
    else:
        write_csv(
            pd.DataFrame(
                [
                    {
                        "status": "not_written",
                        "reason": "rerun with --write-predictions to emit per-row predictions",
                    }
                ]
            ),
            outdir / "obs080d_bootstrap_predictions.csv",
        )

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        feature_manifest=feature_manifest,
        resampling_manifest=resampling_manifest,
        scores_summary=summary_df,
        matrix=matrix_df,
        core_reference=core_ref_df,
        failures=failures_df,
        n_bootstrap=args.n_bootstrap,
    )

    print(f"[OBS-080d] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
