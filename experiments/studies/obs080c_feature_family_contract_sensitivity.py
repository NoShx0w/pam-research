#!/usr/bin/env python3
"""
obs080c_feature_family_contract_sensitivity.py

OBS-080c — Feature-Family / Observable-Contract Sensitivity.

Purpose
-------
OBS-078 found that the C / Cp2 / Cp3 distinction compresses to a 3-feature
window-local stability core:

    mean_lambda_local_mean
    mean_delta_d_mean
    bounded_share_mean

OBS-079 showed that this core is structurally robust, bootstrap-stable,
and pairwise anatomized within the current feature-table contract.

OBS-080a showed transform-contract stability.
OBS-080b showed scale-band stability / scale-position sensitivity.

OBS-080c asks:

    Does the C / Cp2 / Cp3 distinction survive alternate feature-family
    contracts, and how competitive is the 3-feature stability core against
    broader OBS-077/078-derived feature families?

Feature contracts
-----------------
The script builds feature panels from the OBS-078a feature table, using
safe column-name inference with strict leakage exclusions.

Core panels include:

    stability_core_3
    window_means_only
    window_z_only
    geometry_scores_only
    path_shares_only
    stability_plus_geometry
    stability_plus_paths
    stability_plus_z
    no_window
    no_geometry
    no_paths
    strict_numeric_all

Classification tasks
--------------------
    three_way:
        C / Cp2 / Cp3

    pairwise:
        C vs Cp2
        C vs Cp3
        Cp2 vs Cp3

Validation schemes
------------------
    stratified_cv
    leave_object_out
    leave_cohort_out
    leave_transition_out

Outputs
-------
    obs080c_input_manifest.csv
    obs080c_feature_contract_manifest.csv
    obs080c_validation_groups.csv
    obs080c_feature_contract_scores.csv
    obs080c_feature_contract_predictions.csv
    obs080c_feature_contract_scheme_summary.csv
    obs080c_pairwise_feature_contract_summary.csv
    obs080c_feature_contract_stability_matrix.csv
    obs080c_permutation_scores.csv
    obs080c_permutation_summary.csv
    obs080c_confusion_matrices.csv
    obs080c_failures.csv
    obs080c_report.md

Scientific guardrail
--------------------
OBS-080c is a feature-contract sensitivity diagnostic, not causal proof.

The supported statement is:

    The local stability core is feature-contract sufficient if it remains
    competitive with broader feature contracts and survives held-out validation.
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


MODEL_RANDOM_STATE = 80003

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

VALIDATION_SCHEMES = [
    "stratified_cv",
    "leave_object_out",
    "leave_cohort_out",
    "leave_transition_out",
]

DEFAULT_FEATURE_CONTRACTS = [
    "stability_core_3",
    "window_means_only",
    "window_z_only",
    "geometry_scores_only",
    "path_shares_only",
    "stability_plus_geometry",
    "stability_plus_paths",
    "stability_plus_z",
    "no_window",
    "no_geometry",
    "no_paths",
    "strict_numeric_all",
]

IDENTITY_AND_VALIDATION_COLS = {
    TARGET_COL,
    "row_id",
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
    "scheme",
    "model",
    "heldout_group",
    "reason",
    "n_train",
    "n_test",
    "n_rows",
    "train_class_counts",
    "test_class_counts",
    "class_counts",
]


# -----------------------------------------------------------------------------
# Utilities
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


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    require_columns(out, [TARGET_COL, "object", "cohort"] + STABILITY_FEATURES, "feature table")

    if "transition" not in out.columns:
        require_columns(out, ["scale_index_from", "scale_index_to"], "feature table")
        out["transition"] = out.apply(transition_label, axis=1)

    out = out.reset_index(drop=True).copy()
    out["row_id"] = np.arange(len(out))

    return out


def scheme_group_col(scheme: str) -> str | None:
    if scheme == "stratified_cv":
        return None
    if scheme == "leave_object_out":
        return "object"
    if scheme == "leave_cohort_out":
        return "cohort"
    if scheme == "leave_transition_out":
        return "transition"
    raise ValueError(f"Unknown scheme: {scheme}")


def task_cases(task: str) -> tuple[str, ...]:
    if task == "three_way":
        return ("C", "Cp2", "Cp3")
    if task in PAIRWISE_CASES:
        return PAIRWISE_CASES[task]
    raise ValueError(f"Unknown task: {task}")


def labels_for_task(task: str) -> list[str]:
    return list(task_cases(task))


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

    # Keep the canonical 3-feature window core first.
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

    no_geometry = [c for c in strict_all if c not in set(geometry)]
    no_paths = [c for c in strict_all if c not in set(path_shares)]

    panels = {
        "stability_core_3": stability_core,
        "window_means_only": window_means,
        "window_z_only": window_z,
        "geometry_scores_only": geometry,
        "path_shares_only": path_shares,
        "stability_plus_geometry": sorted(set(stability_core + geometry)),
        "stability_plus_paths": sorted(set(stability_core + path_shares)),
        "stability_plus_z": sorted(set(stability_core + window_z)),
        "no_window": no_window,
        "no_geometry": no_geometry,
        "no_paths": no_paths,
        "strict_numeric_all": strict_all,
    }

    return panels


def build_feature_contract_manifest(
    df: pd.DataFrame,
    panels: dict[str, list[str]],
    requested_contracts: list[str],
    drop_empty: bool,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    rows = []
    active: dict[str, list[str]] = {}

    for contract in requested_contracts:
        if contract not in panels:
            raise ValueError(f"Unknown feature contract: {contract}")

        features = existing(df, panels[contract])
        features = [f for f in features if not is_strict_excluded_feature(f)]

        if drop_empty and not features:
            rows.append(
                {
                    "feature_contract": contract,
                    "feature": "",
                    "feature_index": -1,
                    "n_features": 0,
                    "status": "dropped_empty",
                }
            )
            continue

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
    scheme: str,
    model: str,
    heldout_group: str,
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
                    "scheme": scheme,
                    "model": model,
                    "heldout_group": heldout_group,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(cm[i, j]),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Validation inventory
# -----------------------------------------------------------------------------

def build_validation_groups(
    df: pd.DataFrame,
    contracts: dict[str, list[str]],
    tasks: list[str],
    schemes: list[str],
) -> pd.DataFrame:
    rows = []

    for contract, features in contracts.items():
        for task in tasks:
            cases = task_cases(task)
            sub = df[df[TARGET_COL].isin(cases)].copy()

            for scheme in schemes:
                group_col = scheme_group_col(scheme)

                if group_col is None:
                    rows.append(
                        {
                            "feature_contract": contract,
                            "n_features": len(features),
                            "task": task,
                            "scheme": scheme,
                            "group_col": "",
                            "heldout_group": "stratified_cv",
                            "n_rows": int(len(sub)),
                            "n_classes": n_classes(sub[TARGET_COL]),
                            "class_counts": class_counts_string(sub[TARGET_COL]),
                        }
                    )
                    continue

                for group_value, g in sub.groupby(group_col, dropna=False):
                    rows.append(
                        {
                            "feature_contract": contract,
                            "n_features": len(features),
                            "task": task,
                            "scheme": scheme,
                            "group_col": group_col,
                            "heldout_group": str(group_value),
                            "n_rows": int(len(g)),
                            "n_classes": n_classes(g[TARGET_COL]),
                            "class_counts": class_counts_string(g[TARGET_COL]),
                        }
                    )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_stratified_cv(
    df: pd.DataFrame,
    task: str,
    feature_contract: str,
    features: list[str],
    model_name: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = task_cases(task)
    labels = labels_for_task(task)

    sub = df[df[TARGET_COL].isin(cases)].copy()
    y = sub[TARGET_COL].astype(str)

    if not features:
        fail = pd.DataFrame(
            [
                {
                    "task": task,
                    "feature_contract": feature_contract,
                    "scheme": "stratified_cv",
                    "model": model_name,
                    "heldout_group": "stratified_cv",
                    "reason": "empty_feature_contract",
                    "n_rows": int(len(sub)),
                    "class_counts": class_counts_string(y),
                }
            ],
            columns=FAILURE_COLUMNS,
        )
        return pd.DataFrame(), pd.DataFrame(), fail, pd.DataFrame()

    X = sub[features].apply(pd.to_numeric, errors="coerce")

    if sub.empty or n_classes(y) < len(cases):
        fail = pd.DataFrame(
            [
                {
                    "task": task,
                    "feature_contract": feature_contract,
                    "scheme": "stratified_cv",
                    "model": model_name,
                    "heldout_group": "stratified_cv",
                    "reason": "missing_task_classes",
                    "n_rows": int(len(sub)),
                    "class_counts": class_counts_string(y) if len(y) else "",
                }
            ],
            columns=FAILURE_COLUMNS,
        )
        return pd.DataFrame(), pd.DataFrame(), fail, pd.DataFrame()

    min_class = min(Counter(y).values())
    splits = min(n_splits, min_class)

    if splits < 2:
        fail = pd.DataFrame(
            [
                {
                    "task": task,
                    "feature_contract": feature_contract,
                    "scheme": "stratified_cv",
                    "model": model_name,
                    "heldout_group": "stratified_cv",
                    "reason": "not_enough_class_support",
                    "n_rows": int(len(sub)),
                    "class_counts": class_counts_string(y),
                }
            ],
            columns=FAILURE_COLUMNS,
        )
        return pd.DataFrame(), pd.DataFrame(), fail, pd.DataFrame()

    cv = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=MODEL_RANDOM_STATE,
    )

    pipe = make_pipeline(model_name)
    pred = cross_val_predict(pipe, X, y, cv=cv)

    score = {
        "task": task,
        "feature_contract": feature_contract,
        "n_features": len(features),
        "features": ",".join(features),
        "scheme": "stratified_cv",
        "model": model_name,
        "heldout_group": "stratified_cv",
        "group_col": "",
        "n_train": int(len(sub)),
        "n_test": int(len(sub)),
        "n_train_classes": n_classes(y),
        "n_test_classes": n_classes(y),
        "train_class_counts": class_counts_string(y),
        "test_class_counts": class_counts_string(y),
        "valid_for_primary_summary": True,
        "note": "",
        **score_predictions(y, pred, labels=labels),
    }

    pred_df = sub[["row_id", TARGET_COL, "object", "cohort", "transition"]].copy()
    pred_df["task"] = task
    pred_df["feature_contract"] = feature_contract
    pred_df["scheme"] = "stratified_cv"
    pred_df["model"] = model_name
    pred_df["heldout_group"] = "stratified_cv"
    pred_df["prediction"] = pred
    pred_df["correct"] = pred_df[TARGET_COL].astype(str) == pred_df["prediction"].astype(str)

    cm = confusion_rows(
        y_true=y,
        y_pred=pred,
        task=task,
        feature_contract=feature_contract,
        scheme="stratified_cv",
        model=model_name,
        heldout_group="stratified_cv",
        labels=labels,
    )

    return pd.DataFrame([score]), pred_df, pd.DataFrame(columns=FAILURE_COLUMNS), cm


def evaluate_leave_group_out(
    df: pd.DataFrame,
    task: str,
    feature_contract: str,
    features: list[str],
    scheme: str,
    model_name: str,
    require_all_classes_in_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = task_cases(task)
    labels = labels_for_task(task)
    group_col = scheme_group_col(scheme)
    assert group_col is not None

    sub = df[df[TARGET_COL].isin(cases)].copy()
    all_classes = set(cases)

    score_rows = []
    pred_parts = []
    fail_rows = []
    cm_parts = []

    if not features:
        fail_rows.append(
            {
                "task": task,
                "feature_contract": feature_contract,
                "scheme": scheme,
                "model": model_name,
                "heldout_group": "",
                "reason": "empty_feature_contract",
                "n_rows": int(len(sub)),
                "class_counts": class_counts_string(sub[TARGET_COL]) if len(sub) else "",
            }
        )
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(fail_rows, columns=FAILURE_COLUMNS),
            pd.DataFrame(),
        )

    if sub.empty or n_classes(sub[TARGET_COL]) < len(cases):
        fail_rows.append(
            {
                "task": task,
                "feature_contract": feature_contract,
                "scheme": scheme,
                "model": model_name,
                "heldout_group": "",
                "reason": "missing_task_classes",
                "n_rows": int(len(sub)),
                "class_counts": class_counts_string(sub[TARGET_COL]) if len(sub) else "",
            }
        )
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(fail_rows, columns=FAILURE_COLUMNS),
            pd.DataFrame(),
        )

    for group_value, test in sub.groupby(group_col, dropna=False):
        train = sub[sub[group_col] != group_value]

        y_train = train[TARGET_COL].astype(str)
        y_test = test[TARGET_COL].astype(str)

        train_classes = set(y_train.unique())
        test_classes = set(y_test.unique())

        valid_primary = True
        notes = []

        if len(train) == 0 or len(test) == 0:
            fail_rows.append(
                {
                    "task": task,
                    "feature_contract": feature_contract,
                    "scheme": scheme,
                    "model": model_name,
                    "heldout_group": str(group_value),
                    "reason": "empty_train_or_test",
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_class_counts": class_counts_string(y_train),
                    "test_class_counts": class_counts_string(y_test),
                }
            )
            continue

        if len(train_classes) < 2:
            fail_rows.append(
                {
                    "task": task,
                    "feature_contract": feature_contract,
                    "scheme": scheme,
                    "model": model_name,
                    "heldout_group": str(group_value),
                    "reason": "train_has_less_than_two_classes",
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_class_counts": class_counts_string(y_train),
                    "test_class_counts": class_counts_string(y_test),
                }
            )
            continue

        if not set(train_classes).issuperset(all_classes):
            valid_primary = False
            notes.append("train_missing_task_class")

        if require_all_classes_in_test and set(test_classes) != all_classes:
            valid_primary = False
            notes.append("test_missing_task_class")

        if len(test_classes) < 2:
            valid_primary = False
            notes.append("test_has_less_than_two_classes")

        try:
            pipe = make_pipeline(model_name)
            pipe.fit(train[features].apply(pd.to_numeric, errors="coerce"), y_train)
            pred = pipe.predict(test[features].apply(pd.to_numeric, errors="coerce"))

            score = {
                "task": task,
                "feature_contract": feature_contract,
                "n_features": len(features),
                "features": ",".join(features),
                "scheme": scheme,
                "model": model_name,
                "heldout_group": str(group_value),
                "group_col": group_col,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_train_classes": int(len(train_classes)),
                "n_test_classes": int(len(test_classes)),
                "train_class_counts": class_counts_string(y_train),
                "test_class_counts": class_counts_string(y_test),
                "valid_for_primary_summary": bool(valid_primary),
                "note": ";".join(notes),
                **score_predictions(y_test, pred, labels=labels),
            }
            score_rows.append(score)

            pred_df = test[["row_id", TARGET_COL, "object", "cohort", "transition"]].copy()
            pred_df["task"] = task
            pred_df["feature_contract"] = feature_contract
            pred_df["scheme"] = scheme
            pred_df["model"] = model_name
            pred_df["heldout_group"] = str(group_value)
            pred_df["prediction"] = pred
            pred_df["correct"] = pred_df[TARGET_COL].astype(str) == pred_df["prediction"].astype(str)
            pred_parts.append(pred_df)

            cm_parts.append(
                confusion_rows(
                    y_true=y_test,
                    y_pred=pred,
                    task=task,
                    feature_contract=feature_contract,
                    scheme=scheme,
                    model=model_name,
                    heldout_group=str(group_value),
                    labels=labels,
                )
            )

        except Exception as e:
            fail_rows.append(
                {
                    "task": task,
                    "feature_contract": feature_contract,
                    "scheme": scheme,
                    "model": model_name,
                    "heldout_group": str(group_value),
                    "reason": str(e),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_class_counts": class_counts_string(y_train),
                    "test_class_counts": class_counts_string(y_test),
                }
            )

    scores = pd.DataFrame(score_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    fails = pd.DataFrame(fail_rows, columns=FAILURE_COLUMNS)
    cms = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()

    return scores, preds, fails, cms


# -----------------------------------------------------------------------------
# Permutations
# -----------------------------------------------------------------------------

def permutation_stratified_cv(
    df: pd.DataFrame,
    task: str,
    feature_contract: str,
    features: list[str],
    model_name: str,
    n_splits: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if model_name == "dummy" or not features:
        return pd.DataFrame()

    cases = task_cases(task)
    labels = labels_for_task(task)

    rng = np.random.default_rng(seed)
    sub = df[df[TARGET_COL].isin(cases)].copy()

    if sub.empty or n_classes(sub[TARGET_COL]) < len(cases):
        return pd.DataFrame()

    X = sub[features].apply(pd.to_numeric, errors="coerce")
    y0 = sub[TARGET_COL].astype(str).to_numpy()

    rows = []
    for i in range(n_permutations):
        y_perm = pd.Series(rng.permutation(y0))
        min_class = min(Counter(y_perm).values())
        splits = min(n_splits, min_class)
        if splits < 2:
            continue

        cv = StratifiedKFold(
            n_splits=splits,
            shuffle=True,
            random_state=MODEL_RANDOM_STATE + i,
        )

        pipe = make_pipeline(model_name)
        pred = cross_val_predict(pipe, X, y_perm, cv=cv)

        rows.append(
            {
                "task": task,
                "feature_contract": feature_contract,
                "scheme": "stratified_cv",
                "model": model_name,
                "permutation_index": i,
                **score_predictions(y_perm, pred, labels=labels),
            }
        )

    return pd.DataFrame(rows)


def summarize_permutations(scores: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    if perm.empty:
        return pd.DataFrame()

    keys = ["task", "feature_contract", "scheme", "model"]
    rows = []

    for key_vals, g in perm.groupby(keys, dropna=False):
        key = dict(zip(keys, key_vals))
        actual = scores.copy()
        for k, v in key.items():
            actual = actual[actual[k] == v]

        actual_ba = (
            float(pd.to_numeric(actual["balanced_accuracy"], errors="coerce").mean())
            if not actual.empty
            else np.nan
        )

        ba = pd.to_numeric(g["balanced_accuracy"], errors="coerce")

        rows.append(
            {
                **key,
                "actual_balanced_accuracy": actual_ba,
                "perm_mean_balanced_accuracy": float(ba.mean()),
                "perm_std_balanced_accuracy": float(ba.std(ddof=1)) if len(ba) > 1 else np.nan,
                "perm_p_ge_actual": float((ba >= actual_ba).mean()) if np.isfinite(actual_ba) else np.nan,
                "n_permutations": int(len(g)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["task", "actual_balanced_accuracy"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)

    return out


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def summarize_schemes(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    keys = ["task", "feature_contract", "scheme", "model"]
    rows = []

    for key_vals, g in scores.groupby(keys, dropna=False):
        key = dict(zip(keys, key_vals))
        primary = (
            g[g["valid_for_primary_summary"].astype(bool)]
            if "valid_for_primary_summary" in g.columns
            else g
        )

        for scope, gg in [("all_groups", g), ("primary_valid_groups", primary)]:
            if gg.empty:
                continue

            n_test = pd.to_numeric(gg["n_test"], errors="coerce")
            ba = pd.to_numeric(gg["balanced_accuracy"], errors="coerce")
            acc = pd.to_numeric(gg["accuracy"], errors="coerce")
            f1 = pd.to_numeric(gg["macro_f1"], errors="coerce")

            rows.append(
                {
                    **key,
                    "summary_scope": scope,
                    "n_groups": int(len(gg)),
                    "total_test_rows": int(n_test.sum()),
                    "mean_accuracy": float(acc.mean()),
                    "mean_balanced_accuracy": float(ba.mean()),
                    "mean_macro_f1": float(f1.mean()),
                    "weighted_balanced_accuracy_by_n_test": float(np.average(ba, weights=n_test))
                    if n_test.sum() > 0
                    else np.nan,
                    "min_balanced_accuracy": float(ba.min()),
                    "max_balanced_accuracy": float(ba.max()),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["summary_scope", "task", "feature_contract", "scheme", "mean_balanced_accuracy"],
            ascending=[True, True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    return out


def build_feature_contract_stability_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    q = summary[
        (summary["summary_scope"] == "primary_valid_groups")
        & (summary["model"] != "dummy")
    ].copy()

    if q.empty:
        return pd.DataFrame()

    rows = []

    for (task, feature_contract), g in q.groupby(["task", "feature_contract"]):
        best = g.sort_values("mean_balanced_accuracy", ascending=False).head(1)
        strat = g[g["scheme"] == "stratified_cv"].sort_values("mean_balanced_accuracy", ascending=False).head(1)
        loo = g[g["scheme"] == "leave_object_out"].sort_values("mean_balanced_accuracy", ascending=False).head(1)
        lco = g[g["scheme"] == "leave_cohort_out"].sort_values("mean_balanced_accuracy", ascending=False).head(1)
        lto = g[g["scheme"] == "leave_transition_out"].sort_values("mean_balanced_accuracy", ascending=False).head(1)

        def val(frame: pd.DataFrame, col: str) -> Any:
            if frame.empty:
                return np.nan
            return frame[col].iloc[0]

        rows.append(
            {
                "task": task,
                "feature_contract": feature_contract,
                "best_model": val(best, "model"),
                "best_scheme": val(best, "scheme"),
                "best_mean_balanced_accuracy": float(val(best, "mean_balanced_accuracy")),
                "stratified_cv_best_ba": float(val(strat, "mean_balanced_accuracy")) if not strat.empty else np.nan,
                "leave_object_out_best_ba": float(val(loo, "mean_balanced_accuracy")) if not loo.empty else np.nan,
                "leave_cohort_out_best_ba": float(val(lco, "mean_balanced_accuracy")) if not lco.empty else np.nan,
                "leave_transition_out_best_ba": float(val(lto, "mean_balanced_accuracy")) if not lto.empty else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["task", "best_mean_balanced_accuracy"],
            ascending=[True, False],
        ).reset_index(drop=True)

    return out


def build_pairwise_feature_contract_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()

    q = matrix[matrix["task"].isin(PAIRWISE_CASES.keys())].copy()
    return q.sort_values(
        ["task", "best_mean_balanced_accuracy"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_contract_contrast_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()

    rows = []
    for task, g in matrix.groupby("task"):
        core = g[g["feature_contract"] == "stability_core_3"]
        full = g[g["feature_contract"] == "strict_numeric_all"]
        no_window = g[g["feature_contract"] == "no_window"]
        no_geometry = g[g["feature_contract"] == "no_geometry"]
        no_paths = g[g["feature_contract"] == "no_paths"]

        def ba(frame: pd.DataFrame) -> float:
            if frame.empty:
                return np.nan
            return float(frame["best_mean_balanced_accuracy"].iloc[0])

        core_ba = ba(core)
        full_ba = ba(full)

        rows.append(
            {
                "task": task,
                "stability_core_3_ba": core_ba,
                "strict_numeric_all_ba": full_ba,
                "core_minus_full_ba": core_ba - full_ba if np.isfinite(core_ba) and np.isfinite(full_ba) else np.nan,
                "no_window_ba": ba(no_window),
                "no_geometry_ba": ba(no_geometry),
                "no_paths_ba": ba(no_paths),
                "core_minus_no_window_ba": core_ba - ba(no_window) if np.isfinite(core_ba) and np.isfinite(ba(no_window)) else np.nan,
                "core_minus_no_geometry_ba": core_ba - ba(no_geometry) if np.isfinite(core_ba) and np.isfinite(ba(no_geometry)) else np.nan,
                "core_minus_no_paths_ba": core_ba - ba(no_paths) if np.isfinite(core_ba) and np.isfinite(ba(no_paths)) else np.nan,
            }
        )

    return pd.DataFrame(rows)


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
    write_csv(out, outdir / "obs080c_input_manifest.csv")
    return out


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    validation_groups: pd.DataFrame,
    scheme_summary: pd.DataFrame,
    stability_matrix: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    perm_summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# OBS-080c — Feature-Family / Observable-Contract Sensitivity")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-080c tests whether the OBS-078/079 three-feature local stability core "
        "remains competitive under alternate feature-family contracts."
    )
    lines.append("")
    lines.append("Stability core:")
    lines.append("")
    lines.append("```text")
    for f in STABILITY_FEATURES:
        lines.append(f)
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
    lines.append("## Validation group inventory")
    lines.append("")
    if validation_groups.empty:
        lines.append("_No validation groups._")
    else:
        vg = (
            validation_groups.groupby(["feature_contract", "task", "scheme", "group_col"], dropna=False)
            .agg(
                n_groups=("heldout_group", "count"),
                n_features=("n_features", "first"),
                min_rows=("n_rows", "min"),
                mean_rows=("n_rows", "mean"),
                max_rows=("n_rows", "max"),
                min_classes=("n_classes", "min"),
                max_classes=("n_classes", "max"),
            )
            .reset_index()
            .sort_values(["feature_contract", "task", "scheme", "group_col"])
        )
        lines.append(markdown_table(vg, max_rows=160))
    lines.append("")
    lines.append("## Feature-contract stability matrix")
    lines.append("")
    if stability_matrix.empty:
        lines.append("_No feature-contract stability rows._")
    else:
        lines.append(markdown_table(stability_matrix, max_rows=160))
    lines.append("")
    lines.append("## Pairwise feature-contract summary")
    lines.append("")
    if pairwise_summary.empty:
        lines.append("_No pairwise summary rows._")
    else:
        lines.append(markdown_table(pairwise_summary, max_rows=160))
    lines.append("")
    lines.append("## Contract contrast summary")
    lines.append("")
    if contrast_summary.empty:
        lines.append("_No contract contrast rows._")
    else:
        lines.append(markdown_table(contrast_summary, max_rows=80))
    lines.append("")
    lines.append("## Scheme summary")
    lines.append("")
    if scheme_summary.empty:
        lines.append("_No scheme summary rows._")
    else:
        display = scheme_summary[
            [
                "task",
                "feature_contract",
                "scheme",
                "model",
                "summary_scope",
                "n_groups",
                "total_test_rows",
                "mean_balanced_accuracy",
                "weighted_balanced_accuracy_by_n_test",
                "min_balanced_accuracy",
                "max_balanced_accuracy",
            ]
        ].sort_values(
            ["summary_scope", "task", "feature_contract", "scheme", "mean_balanced_accuracy"],
            ascending=[True, True, True, True, False],
        )
        lines.append(markdown_table(display, max_rows=220))
    lines.append("")
    lines.append("## Permutation summary")
    lines.append("")
    if perm_summary.empty:
        lines.append("_No permutation summary rows._")
    else:
        display = perm_summary[
            [
                "task",
                "feature_contract",
                "model",
                "actual_balanced_accuracy",
                "perm_mean_balanced_accuracy",
                "perm_std_balanced_accuracy",
                "perm_p_ge_actual",
            ]
        ].sort_values(["task", "actual_balanced_accuracy"], ascending=[True, False])
        lines.append(markdown_table(display, max_rows=180))
    lines.append("")
    lines.append("## Failures / skipped rows")
    lines.append("")
    if failures.empty:
        lines.append("_No failures._")
    else:
        lines.append(markdown_table(failures, max_rows=120))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("Strong feature-contract sufficiency evidence:")
    lines.append("")
    lines.append("```text")
    lines.append("stability_core_3 remains > 0.80 three-way")
    lines.append("stability_core_3 remains > 0.90 for C vs Cp2 / C vs Cp3")
    lines.append("stability_core_3 remains above permutation for Cp2 vs Cp3")
    lines.append("broader panels improve but do not replace the core")
    lines.append("```")
    lines.append("")
    lines.append("Key diagnostic:")
    lines.append("")
    lines.append("```text")
    lines.append("If no_window drops strongly:")
    lines.append("  local window stability is central.")
    lines.append("")
    lines.append("If no_window remains strong:")
    lines.append("  the stability signal is redundant across feature families.")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-080c is a feature-contract sensitivity diagnostic, not causal proof.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs080c_input_manifest.csv")
    lines.append("obs080c_feature_contract_manifest.csv")
    lines.append("obs080c_validation_groups.csv")
    lines.append("obs080c_feature_contract_scores.csv")
    lines.append("obs080c_feature_contract_predictions.csv")
    lines.append("obs080c_feature_contract_scheme_summary.csv")
    lines.append("obs080c_pairwise_feature_contract_summary.csv")
    lines.append("obs080c_feature_contract_stability_matrix.csv")
    lines.append("obs080c_contract_contrast_summary.csv")
    lines.append("obs080c_permutation_scores.csv")
    lines.append("obs080c_permutation_summary.csv")
    lines.append("obs080c_confusion_matrices.csv")
    lines.append("obs080c_failures.csv")
    lines.append("obs080c_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-080c")

    (outdir / "obs080c_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-080c feature-family contract sensitivity.")
    ap.add_argument(
        "--feature-table",
        default=(
            "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/"
            "obs078a_feature_table.csv"
        ),
    )
    ap.add_argument(
        "--outdir",
        default="outputs/comparisons/obs080c_feature_family_contract_sensitivity",
    )
    ap.add_argument(
        "--feature-contracts",
        default=",".join(DEFAULT_FEATURE_CONTRACTS),
        help="Comma-separated feature contracts.",
    )
    ap.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help="Comma-separated tasks.",
    )
    ap.add_argument(
        "--schemes",
        default=",".join(VALIDATION_SCHEMES),
        help="Comma-separated validation schemes.",
    )
    ap.add_argument(
        "--models",
        default="logreg,tree_depth2,rf_depth2,dummy",
        help="Comma-separated model names.",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument(
        "--require-all-classes-in-test",
        action="store_true",
        help="Mark held-out groups missing task classes as not primary-valid.",
    )
    ap.add_argument(
        "--keep-empty-contracts",
        action="store_true",
        help="Keep empty feature contracts and record failures instead of dropping them.",
    )
    ap.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outdir = ensure_outdir(args.outdir)
    feature_table_path = Path(args.feature_table)

    if not feature_table_path.exists():
        raise FileNotFoundError(feature_table_path)

    df = pd.read_csv(feature_table_path)
    df = prepare_columns(df)

    # Ensure all strict numeric candidates can be read numerically.
    numeric_cols = strict_numeric_features(df)
    df = safe_numeric(df, numeric_cols)

    requested_contracts = [x.strip() for x in args.feature_contracts.split(",") if x.strip()]
    inferred_panels = infer_feature_families(df)
    contracts, feature_manifest = build_feature_contract_manifest(
        df=df,
        panels=inferred_panels,
        requested_contracts=requested_contracts,
        drop_empty=not args.keep_empty_contracts,
    )

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")

    schemes = [x.strip() for x in args.schemes.split(",") if x.strip()]
    unknown_schemes = sorted(set(schemes) - set(VALIDATION_SCHEMES))
    if unknown_schemes:
        raise ValueError(f"Unknown schemes: {unknown_schemes}")

    models = [x.strip() for x in args.models.split(",") if x.strip()]

    input_manifest = write_input_manifest(outdir, feature_table_path, df)
    write_csv(feature_manifest, outdir / "obs080c_feature_contract_manifest.csv")

    validation_groups = build_validation_groups(df, contracts, tasks, schemes)
    write_csv(validation_groups, outdir / "obs080c_validation_groups.csv")

    score_parts = []
    pred_parts = []
    fail_parts = []
    cm_parts = []
    perm_parts = []

    run_index = 0

    for feature_contract, features in contracts.items():
        for task in tasks:
            for scheme in schemes:
                for model_name in models:
                    run_index += 1

                    if scheme == "stratified_cv":
                        scores, preds, fails, cms = evaluate_stratified_cv(
                            df=df,
                            task=task,
                            feature_contract=feature_contract,
                            features=features,
                            model_name=model_name,
                            n_splits=args.n_splits,
                        )

                        if not scores.empty:
                            score_parts.append(scores)
                        if not preds.empty:
                            pred_parts.append(preds)
                        if not fails.empty:
                            fail_parts.append(fails)
                        if not cms.empty:
                            cm_parts.append(cms)

                        if args.n_permutations > 0 and model_name != "dummy":
                            perm = permutation_stratified_cv(
                                df=df,
                                task=task,
                                feature_contract=feature_contract,
                                features=features,
                                model_name=model_name,
                                n_splits=args.n_splits,
                                n_permutations=args.n_permutations,
                                seed=args.seed + run_index,
                            )
                            if not perm.empty:
                                perm_parts.append(perm)

                    else:
                        scores, preds, fails, cms = evaluate_leave_group_out(
                            df=df,
                            task=task,
                            feature_contract=feature_contract,
                            features=features,
                            scheme=scheme,
                            model_name=model_name,
                            require_all_classes_in_test=args.require_all_classes_in_test,
                        )

                        if not scores.empty:
                            score_parts.append(scores)
                        if not preds.empty:
                            pred_parts.append(preds)
                        if not fails.empty:
                            fail_parts.append(fails)
                        if not cms.empty:
                            cm_parts.append(cms)

    scores_df = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    if not scores_df.empty:
        scores_df = scores_df.sort_values(
            ["task", "feature_contract", "scheme", "model", "balanced_accuracy"],
            ascending=[True, True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    preds_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    failures_df = (
        pd.concat(fail_parts, ignore_index=True)
        if fail_parts
        else pd.DataFrame(columns=FAILURE_COLUMNS)
    )
    cm_df = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()
    perm_df = pd.concat(perm_parts, ignore_index=True) if perm_parts else pd.DataFrame()

    scheme_summary = summarize_schemes(scores_df)
    stability_matrix = build_feature_contract_stability_matrix(scheme_summary)
    pairwise_summary = build_pairwise_feature_contract_summary(stability_matrix)
    contrast_summary = build_contract_contrast_summary(stability_matrix)
    perm_summary = summarize_permutations(scores_df, perm_df)

    write_csv(scores_df, outdir / "obs080c_feature_contract_scores.csv")
    write_csv(preds_df, outdir / "obs080c_feature_contract_predictions.csv")
    write_csv(scheme_summary, outdir / "obs080c_feature_contract_scheme_summary.csv")
    write_csv(stability_matrix, outdir / "obs080c_feature_contract_stability_matrix.csv")
    write_csv(pairwise_summary, outdir / "obs080c_pairwise_feature_contract_summary.csv")
    write_csv(contrast_summary, outdir / "obs080c_contract_contrast_summary.csv")
    write_csv(perm_df, outdir / "obs080c_permutation_scores.csv")
    write_csv(perm_summary, outdir / "obs080c_permutation_summary.csv")
    write_csv(failures_df, outdir / "obs080c_failures.csv", columns=FAILURE_COLUMNS)
    write_csv(cm_df, outdir / "obs080c_confusion_matrices.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        feature_manifest=feature_manifest,
        validation_groups=validation_groups,
        scheme_summary=scheme_summary,
        stability_matrix=stability_matrix,
        pairwise_summary=pairwise_summary,
        contrast_summary=contrast_summary,
        perm_summary=perm_summary,
        failures=failures_df,
    )

    print(f"[OBS-080c] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
