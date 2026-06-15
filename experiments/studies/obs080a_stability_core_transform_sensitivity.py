#!/usr/bin/env python3
"""
obs080a_stability_core_transform_sensitivity.py

OBS-080a — Stability Core Transform Sensitivity.

Purpose
-------
OBS-078 found that the C / Cp2 / Cp3 distinction compresses to a 3-feature
window-local stability core:

    mean_lambda_local_mean
    mean_delta_d_mean
    bounded_share_mean

OBS-079 showed that this core is structurally robust, bootstrap-stable,
and pairwise anatomized within the current feature-table contract.

OBS-080a asks:

    Does the stability core survive alternate transform / normalization contracts?

Transforms tested
-----------------
    raw
    standard_z
    robust_median_iqr
    rank_percentile
    quantile_normal
    minmax
    signed_log1p_abs

Classification tasks
--------------------
    3-way:
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
    obs080a_input_manifest.csv
    obs080a_transform_manifest.csv
    obs080a_transformed_feature_table.csv
    obs080a_validation_groups.csv
    obs080a_transform_scores.csv
    obs080a_transform_predictions.csv
    obs080a_transform_scheme_summary.csv
    obs080a_pairwise_transform_summary.csv
    obs080a_transform_stability_matrix.csv
    obs080a_permutation_scores.csv
    obs080a_permutation_summary.csv
    obs080a_failures.csv
    obs080a_report.md

Scientific guardrail
--------------------
OBS-080a is a contract-sensitivity diagnostic, not causal proof.

The supported statement is:

    The local stability core is transform-stable if classification remains
    above baseline across alternate scaling / ranking contracts.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")


MODEL_RANDOM_STATE = 80001
EPS = 1e-12

TARGET_COL = "case"

STABILITY_FEATURES = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

TRANSFORMS = [
    "raw",
    "standard_z",
    "robust_median_iqr",
    "rank_percentile",
    "quantile_normal",
    "minmax",
    "signed_log1p_abs",
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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
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
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def class_counts_string(y: pd.Series) -> str:
    counts = Counter(pd.Series(y).astype(str))
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def n_classes(y: pd.Series) -> int:
    return int(pd.Series(y).astype(str).nunique(dropna=True))


def transition_label(row: pd.Series) -> str:
    return f"{row.get('scale_index_from')}→{row.get('scale_index_to')}"


def prepare_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    require_columns(out, ["object", "cohort"], "feature table")

    if "transition" not in out.columns:
        require_columns(out, ["scale_index_from", "scale_index_to"], "feature table")
        out["transition"] = out.apply(transition_label, axis=1)

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


def transformed_feature_names(transform: str) -> list[str]:
    return [f"{transform}__{f}" for f in STABILITY_FEATURES]


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

def _median_iqr_scale(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    med = vals.median()
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or abs(iqr) <= EPS:
        return vals * 0.0
    return (vals - med) / iqr


def _standard_z(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    mu = vals.mean()
    sd = vals.std(ddof=1)
    if not np.isfinite(sd) or abs(sd) <= EPS:
        return vals * 0.0
    return (vals - mu) / sd


def _rank_percentile(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    return vals.rank(method="average", pct=True)


def _minmax(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    lo = vals.min()
    hi = vals.max()
    span = hi - lo
    if not np.isfinite(span) or abs(span) <= EPS:
        return vals * 0.0
    return (vals - lo) / span


def _signed_log1p_abs(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    return np.sign(vals) * np.log1p(np.abs(vals))


def add_transforms(df: pd.DataFrame, transforms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    manifest_rows = []

    raw = out[STABILITY_FEATURES].copy()

    for transform in transforms:
        if transform == "raw":
            transformed = raw.copy()

        elif transform == "standard_z":
            transformed = raw.apply(_standard_z, axis=0)

        elif transform == "robust_median_iqr":
            transformed = raw.apply(_median_iqr_scale, axis=0)

        elif transform == "rank_percentile":
            transformed = raw.apply(_rank_percentile, axis=0)

        elif transform == "minmax":
            transformed = raw.apply(_minmax, axis=0)

        elif transform == "signed_log1p_abs":
            transformed = raw.apply(_signed_log1p_abs, axis=0)

        elif transform == "quantile_normal":
            # Quantile-normal is fitted globally as a contract transform.
            # n_quantiles cannot exceed n_samples.
            arr = raw.to_numpy(dtype=float)
            imputed = SimpleImputer(strategy="median").fit_transform(arr)
            qt = QuantileTransformer(
                n_quantiles=min(len(out), 1000),
                output_distribution="normal",
                random_state=MODEL_RANDOM_STATE,
            )
            qarr = qt.fit_transform(imputed)
            transformed = pd.DataFrame(qarr, columns=STABILITY_FEATURES, index=out.index)

        else:
            raise ValueError(f"Unknown transform: {transform}")

        for feature in STABILITY_FEATURES:
            new_col = f"{transform}__{feature}"
            out[new_col] = transformed[feature]

            vals = pd.to_numeric(out[new_col], errors="coerce")
            manifest_rows.append(
                {
                    "transform": transform,
                    "source_feature": feature,
                    "transformed_feature": new_col,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)),
                    "min": float(vals.min()),
                    "median": float(vals.median()),
                    "max": float(vals.max()),
                    "n_non_null": int(vals.notna().sum()),
                }
            )

    return out, pd.DataFrame(manifest_rows)


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
        )

    raise ValueError(f"Unknown model: {name}")


def make_pipeline(model_name: str) -> Pipeline:
    # Features are already transformed by OBS-080a contracts.
    # We still impute for safety, but intentionally do not add a scaler here.
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
    transform: str,
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
                    "transform": transform,
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

def build_validation_groups(df: pd.DataFrame, tasks: list[str], schemes: list[str]) -> pd.DataFrame:
    rows = []

    for task in tasks:
        cases = task_cases(task)
        sub = df[df[TARGET_COL].isin(cases)].copy()

        for scheme in schemes:
            group_col = scheme_group_col(scheme)

            if group_col is None:
                rows.append(
                    {
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
    transform: str,
    model_name: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = task_cases(task)
    labels = labels_for_task(task)
    features = transformed_feature_names(transform)

    sub = df[df[TARGET_COL].isin(cases)].copy()
    X = sub[features]
    y = sub[TARGET_COL].astype(str)

    min_class = min(Counter(y).values())
    splits = min(n_splits, min_class)

    if splits < 2:
        fail = pd.DataFrame(
            [
                {
                    "task": task,
                    "transform": transform,
                    "scheme": "stratified_cv",
                    "model": model_name,
                    "heldout_group": "stratified_cv",
                    "reason": "not_enough_class_support",
                    "n_rows": int(len(sub)),
                    "class_counts": class_counts_string(y),
                }
            ]
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
        "transform": transform,
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
    pred_df["transform"] = transform
    pred_df["scheme"] = "stratified_cv"
    pred_df["model"] = model_name
    pred_df["heldout_group"] = "stratified_cv"
    pred_df["prediction"] = pred
    pred_df["correct"] = pred_df[TARGET_COL].astype(str) == pred_df["prediction"].astype(str)

    cm = confusion_rows(
        y_true=y,
        y_pred=pred,
        task=task,
        transform=transform,
        scheme="stratified_cv",
        model=model_name,
        heldout_group="stratified_cv",
        labels=labels,
    )

    return pd.DataFrame([score]), pred_df, pd.DataFrame(), cm


def evaluate_leave_group_out(
    df: pd.DataFrame,
    task: str,
    transform: str,
    scheme: str,
    model_name: str,
    require_all_classes_in_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = task_cases(task)
    labels = labels_for_task(task)
    features = transformed_feature_names(transform)
    group_col = scheme_group_col(scheme)
    assert group_col is not None

    sub = df[df[TARGET_COL].isin(cases)].copy()
    all_classes = set(cases)

    score_rows = []
    pred_parts = []
    fail_rows = []
    cm_parts = []

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
                    "transform": transform,
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
                    "transform": transform,
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
            pipe.fit(train[features], y_train)
            pred = pipe.predict(test[features])

            score = {
                "task": task,
                "transform": transform,
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
            pred_df["transform"] = transform
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
                    transform=transform,
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
                    "transform": transform,
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
    fails = pd.DataFrame(fail_rows)
    cms = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()

    return scores, preds, fails, cms


# -----------------------------------------------------------------------------
# Permutations
# -----------------------------------------------------------------------------

def permutation_stratified_cv(
    df: pd.DataFrame,
    task: str,
    transform: str,
    model_name: str,
    n_splits: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if model_name == "dummy":
        return pd.DataFrame()

    cases = task_cases(task)
    labels = labels_for_task(task)
    features = transformed_feature_names(transform)

    rng = np.random.default_rng(seed)
    sub = df[df[TARGET_COL].isin(cases)].copy()

    X = sub[features]
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
                "transform": transform,
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

    keys = ["task", "transform", "scheme", "model"]
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

    keys = ["task", "transform", "scheme", "model"]
    rows = []

    for key_vals, g in scores.groupby(keys, dropna=False):
        key = dict(zip(keys, key_vals))
        primary = g[g["valid_for_primary_summary"].astype(bool)] if "valid_for_primary_summary" in g.columns else g

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
                    if n_test.sum() > 0 else np.nan,
                    "min_balanced_accuracy": float(ba.min()),
                    "max_balanced_accuracy": float(ba.max()),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["summary_scope", "task", "scheme", "mean_balanced_accuracy"],
            ascending=[True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    return out


def build_transform_stability_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """
    One row per task/transform summarizing best primary-valid performance across
    non-dummy models and schemes.
    """
    if summary.empty:
        return pd.DataFrame()

    q = summary[
        (summary["summary_scope"] == "primary_valid_groups")
        & (summary["model"] != "dummy")
    ].copy()

    if q.empty:
        return pd.DataFrame()

    rows = []
    for (task, transform), g in q.groupby(["task", "transform"]):
        best = g.sort_values("mean_balanced_accuracy", ascending=False).head(1)
        strat = g[g["scheme"] == "stratified_cv"].sort_values("mean_balanced_accuracy", ascending=False).head(1)
        loo = g[g["scheme"] == "leave_object_out"].sort_values("mean_balanced_accuracy", ascending=False).head(1)
        lco = g[g["scheme"] == "leave_cohort_out"].sort_values("mean_balanced_accuracy", ascending=False).head(1)
        lto = g[g["scheme"] == "leave_transition_out"].sort_values("mean_balanced_accuracy", ascending=False).head(1)

        def val(frame: pd.DataFrame, col: str) -> float | str:
            if frame.empty:
                return np.nan
            return frame[col].iloc[0]

        rows.append(
            {
                "task": task,
                "transform": transform,
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
        out = out.sort_values(["task", "best_mean_balanced_accuracy"], ascending=[True, False]).reset_index(drop=True)
    return out


def build_pairwise_transform_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    q = matrix[matrix["task"].isin(PAIRWISE_CASES.keys())].copy()
    return q.sort_values(["task", "best_mean_balanced_accuracy"], ascending=[True, False]).reset_index(drop=True)


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
    write_csv(out, outdir / "obs080a_input_manifest.csv")
    return out


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    transform_manifest: pd.DataFrame,
    validation_groups: pd.DataFrame,
    scheme_summary: pd.DataFrame,
    stability_matrix: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    perm_summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# OBS-080a — Stability Core Transform Sensitivity")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-080a tests whether the OBS-078/079 three-feature local stability core "
        "survives alternate normalization / transform contracts."
    )
    lines.append("")
    lines.append("Stability core:")
    lines.append("")
    lines.append("```text")
    for f in STABILITY_FEATURES:
        lines.append(f)
    lines.append("```")
    lines.append("")
    lines.append("Transforms:")
    lines.append("")
    lines.append("```text")
    for t in TRANSFORMS:
        lines.append(t)
    lines.append("```")
    lines.append("")
    lines.append("Tasks:")
    lines.append("")
    lines.append("```text")
    for t in TASKS:
        lines.append(t)
    lines.append("```")
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest))
    lines.append("")
    lines.append("## Transform manifest summary")
    lines.append("")
    tm = (
        transform_manifest.groupby("transform", as_index=False)
        .agg(n_features=("transformed_feature", "count"))
        .sort_values("transform")
    )
    lines.append(markdown_table(tm, max_rows=80))
    lines.append("")
    lines.append("## Validation group inventory")
    lines.append("")
    vg = (
        validation_groups.groupby(["task", "scheme", "group_col"], dropna=False)
        .agg(
            n_groups=("heldout_group", "count"),
            min_rows=("n_rows", "min"),
            mean_rows=("n_rows", "mean"),
            max_rows=("n_rows", "max"),
            min_classes=("n_classes", "min"),
            max_classes=("n_classes", "max"),
        )
        .reset_index()
        .sort_values(["task", "scheme", "group_col"])
    )
    lines.append(markdown_table(vg, max_rows=120))
    lines.append("")
    lines.append("## Transform stability matrix")
    lines.append("")
    if stability_matrix.empty:
        lines.append("_No transform stability rows._")
    else:
        lines.append(markdown_table(stability_matrix, max_rows=120))
    lines.append("")
    lines.append("## Pairwise transform summary")
    lines.append("")
    if pairwise_summary.empty:
        lines.append("_No pairwise summary rows._")
    else:
        lines.append(markdown_table(pairwise_summary, max_rows=120))
    lines.append("")
    lines.append("## Scheme summary")
    lines.append("")
    if scheme_summary.empty:
        lines.append("_No scheme summary rows._")
    else:
        display = scheme_summary[
            [
                "task",
                "transform",
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
            ["summary_scope", "task", "scheme", "mean_balanced_accuracy"],
            ascending=[True, True, True, False],
        )
        lines.append(markdown_table(display, max_rows=160))
    lines.append("")
    lines.append("## Permutation summary")
    lines.append("")
    if perm_summary.empty:
        lines.append("_No permutation summary rows._")
    else:
        display = perm_summary[
            [
                "task",
                "transform",
                "model",
                "actual_balanced_accuracy",
                "perm_mean_balanced_accuracy",
                "perm_std_balanced_accuracy",
                "perm_p_ge_actual",
            ]
        ].sort_values(["task", "actual_balanced_accuracy"], ascending=[True, False])
        lines.append(markdown_table(display, max_rows=120))
    lines.append("")
    lines.append("## Failures / skipped rows")
    lines.append("")
    if failures.empty:
        lines.append("_No failures._")
    else:
        lines.append(markdown_table(failures, max_rows=80))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("Strong transform stability evidence:")
    lines.append("")
    lines.append("```text")
    lines.append("three_way BA remains > 0.80 across most transforms")
    lines.append("C vs Cp2 and C vs Cp3 remain > 0.90 across most transforms")
    lines.append("Cp2 vs Cp3 remains > 0.75 across robust/rank-like transforms")
    lines.append("```")
    lines.append("")
    lines.append("Key transform:")
    lines.append("")
    lines.append("```text")
    lines.append("rank_percentile")
    lines.append("```")
    lines.append("")
    lines.append("If rank_percentile works, the stability core is not only calibrated by raw magnitudes; it preserves ordinal regime structure.")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-080a is a transform-contract sensitivity diagnostic, not causal proof.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs080a_input_manifest.csv")
    lines.append("obs080a_transform_manifest.csv")
    lines.append("obs080a_transformed_feature_table.csv")
    lines.append("obs080a_validation_groups.csv")
    lines.append("obs080a_transform_scores.csv")
    lines.append("obs080a_transform_predictions.csv")
    lines.append("obs080a_transform_scheme_summary.csv")
    lines.append("obs080a_pairwise_transform_summary.csv")
    lines.append("obs080a_transform_stability_matrix.csv")
    lines.append("obs080a_permutation_scores.csv")
    lines.append("obs080a_permutation_summary.csv")
    lines.append("obs080a_failures.csv")
    lines.append("obs080a_confusion_matrices.csv")
    lines.append("obs080a_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-080a")

    (outdir / "obs080a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-080a stability core transform sensitivity.")
    ap.add_argument(
        "--feature-table",
        default=(
            "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/"
            "obs078a_feature_table.csv"
        ),
    )
    ap.add_argument(
        "--outdir",
        default="outputs/comparisons/obs080a_stability_core_transform_sensitivity",
    )
    ap.add_argument(
        "--transforms",
        default=",".join(TRANSFORMS),
        help="Comma-separated transforms.",
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
    ap.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outdir = ensure_outdir(args.outdir)
    feature_table_path = Path(args.feature_table)

    if not feature_table_path.exists():
        raise FileNotFoundError(feature_table_path)

    df = pd.read_csv(feature_table_path)
    require_columns(df, [TARGET_COL] + STABILITY_FEATURES, "feature table")

    df = prepare_validation_columns(df)
    df = safe_numeric(df, STABILITY_FEATURES)
    df = df.reset_index(drop=True).copy()
    df["row_id"] = np.arange(len(df))

    transforms = [x.strip() for x in args.transforms.split(",") if x.strip()]
    unknown_transforms = sorted(set(transforms) - set(TRANSFORMS))
    if unknown_transforms:
        raise ValueError(f"Unknown transforms: {unknown_transforms}")

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

    tdf, transform_manifest = add_transforms(df, transforms)
    write_csv(tdf, outdir / "obs080a_transformed_feature_table.csv")
    write_csv(transform_manifest, outdir / "obs080a_transform_manifest.csv")

    validation_groups = build_validation_groups(tdf, tasks, schemes)
    write_csv(validation_groups, outdir / "obs080a_validation_groups.csv")

    score_parts = []
    pred_parts = []
    fail_parts = []
    cm_parts = []
    perm_parts = []

    run_index = 0

    for task in tasks:
        for transform in transforms:
            for scheme in schemes:
                for model_name in models:
                    run_index += 1

                    if scheme == "stratified_cv":
                        scores, preds, fails, cms = evaluate_stratified_cv(
                            df=tdf,
                            task=task,
                            transform=transform,
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
                                df=tdf,
                                task=task,
                                transform=transform,
                                model_name=model_name,
                                n_splits=args.n_splits,
                                n_permutations=args.n_permutations,
                                seed=args.seed + run_index,
                            )
                            if not perm.empty:
                                perm_parts.append(perm)

                    else:
                        scores, preds, fails, cms = evaluate_leave_group_out(
                            df=tdf,
                            task=task,
                            transform=transform,
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
            ["task", "transform", "scheme", "model", "balanced_accuracy"],
            ascending=[True, True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    preds_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    failures_df = pd.concat(fail_parts, ignore_index=True) if fail_parts else pd.DataFrame()
    cm_df = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()
    perm_df = pd.concat(perm_parts, ignore_index=True) if perm_parts else pd.DataFrame()

    scheme_summary = summarize_schemes(scores_df)
    stability_matrix = build_transform_stability_matrix(scheme_summary)
    pairwise_summary = build_pairwise_transform_summary(stability_matrix)
    perm_summary = summarize_permutations(scores_df, perm_df)

    write_csv(scores_df, outdir / "obs080a_transform_scores.csv")
    write_csv(preds_df, outdir / "obs080a_transform_predictions.csv")
    write_csv(scheme_summary, outdir / "obs080a_transform_scheme_summary.csv")
    write_csv(stability_matrix, outdir / "obs080a_transform_stability_matrix.csv")
    write_csv(pairwise_summary, outdir / "obs080a_pairwise_transform_summary.csv")
    write_csv(perm_df, outdir / "obs080a_permutation_scores.csv")
    write_csv(perm_summary, outdir / "obs080a_permutation_summary.csv")
    write_csv(failures_df, outdir / "obs080a_failures.csv")
    write_csv(cm_df, outdir / "obs080a_confusion_matrices.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        transform_manifest=transform_manifest,
        validation_groups=validation_groups,
        scheme_summary=scheme_summary,
        stability_matrix=stability_matrix,
        pairwise_summary=pairwise_summary,
        perm_summary=perm_summary,
        failures=failures_df,
    )

    print(f"[OBS-080a] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
