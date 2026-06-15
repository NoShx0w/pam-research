#!/usr/bin/env python3
"""
obs079c_pairwise_stability_classifiers.py

OBS-079c — Pairwise Stability Classifiers.

Purpose
-------
OBS-078 found that the C / Cp2 / Cp3 distinction compresses to a small
window-local stability signature:

    mean_lambda_local_mean
    mean_delta_d_mean
    bounded_share_mean

OBS-079a showed that this signature survives leave-structure-out validation.
OBS-079b showed that the measured stability coordinates are bootstrap-stable.

OBS-079c asks:

    Which pairwise distinctions are supported by which stability axes?

Pairwise tests
--------------
    C vs Cp2
    C vs Cp3
    Cp2 vs Cp3

Feature panels
--------------
    lambda_only
    delta_d_only
    bounded_only
    lambda_delta
    lambda_bounded
    delta_bounded
    full_3_feature_core

Validation schemes
------------------
    stratified_cv
    leave_object_out
    leave_cohort_out
    leave_transition_out

Outputs
-------
    obs079c_input_manifest.csv
    obs079c_feature_panels.csv
    obs079c_validation_groups.csv
    obs079c_pairwise_scores.csv
    obs079c_pairwise_predictions.csv
    obs079c_pairwise_permutation_scores.csv
    obs079c_pairwise_permutation_summary.csv
    obs079c_pairwise_scheme_summary.csv
    obs079c_feature_importance.csv
    obs079c_report.md

Scientific guardrail
--------------------
OBS-079c is a pairwise anatomy / robustness diagnostic. It does not establish
causality.

The key supported statement is:

    Different pairwise case distinctions may rely on different axes of the
    3-feature stability core.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_RANDOM_STATE = 79003

TARGET_COL = "case"

STABILITY_FEATURES = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

PAIRWISE_CASES = [
    ("C", "Cp2"),
    ("C", "Cp3"),
    ("Cp2", "Cp3"),
]

FEATURE_PANELS: dict[str, list[str]] = {
    "lambda_only": ["mean_lambda_local_mean"],
    "delta_d_only": ["mean_delta_d_mean"],
    "bounded_only": ["bounded_share_mean"],
    "lambda_delta": ["mean_lambda_local_mean", "mean_delta_d_mean"],
    "lambda_bounded": ["mean_lambda_local_mean", "bounded_share_mean"],
    "delta_bounded": ["mean_delta_d_mean", "bounded_share_mean"],
    "full_3_feature_core": [
        "mean_lambda_local_mean",
        "mean_delta_d_mean",
        "bounded_share_mean",
    ],
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
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def class_counts_string(y: pd.Series) -> str:
    counts = Counter(pd.Series(y).astype(str))
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def transition_label(row: pd.Series) -> str:
    return f"{row.get('scale_index_from')}→{row.get('scale_index_to')}"


def prepare_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    require_columns(out, ["object", "cohort"], "feature table")

    if "transition" not in out.columns:
        require_columns(out, ["scale_index_from", "scale_index_to"], "feature table")
        out["transition"] = out.apply(transition_label, axis=1)

    return out


def binary_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}_vs_{pair[1]}"


def n_classes(y: pd.Series) -> int:
    return int(pd.Series(y).astype(str).nunique(dropna=True))


def scheme_group_col(scheme: str) -> str | None:
    if scheme == "stratified_cv":
        return None
    if scheme == "leave_object_out":
        return "object"
    if scheme == "leave_cohort_out":
        return "cohort"
    if scheme == "leave_transition_out":
        return "transition"
    raise ValueError(f"Unknown validation scheme: {scheme}")


# -----------------------------------------------------------------------------
# Models
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
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", make_model(model_name)),
        ]
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def score_predictions(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    yt = pd.Series(y_true).astype(str)
    yp = pd.Series(y_pred).astype(str)

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def confusion_rows(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    pair: str,
    panel: str,
    scheme: str,
    model: str,
    heldout_group: str,
) -> pd.DataFrame:
    labels = sorted(pd.Series(y_true).astype(str).unique())
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
                    "pair": pair,
                    "feature_panel": panel,
                    "scheme": scheme,
                    "model": model,
                    "heldout_group": heldout_group,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(cm[i, j]),
                }
            )

    return pd.DataFrame(rows)


def feature_importance_rows(
    fitted: Pipeline,
    pair: str,
    panel: str,
    scheme: str,
    model_name: str,
    features: list[str],
    heldout_group: str = "all",
) -> pd.DataFrame:
    model = fitted.named_steps["model"]
    rows = []

    if hasattr(model, "feature_importances_"):
        for feature, value in zip(features, model.feature_importances_):
            rows.append(
                {
                    "pair": pair,
                    "feature_panel": panel,
                    "scheme": scheme,
                    "model": model_name,
                    "heldout_group": heldout_group,
                    "class": "",
                    "feature": feature,
                    "importance_type": "feature_importance",
                    "value": float(value),
                    "abs_value": float(abs(value)),
                }
            )

    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        classes = list(getattr(model, "classes_", []))

        # Binary LogisticRegression stores one coefficient vector for the
        # positive class: shape = (1, n_features). Treat that vector as the
        # separating axis instead of trying to index both classes.
        if coef.ndim == 2 and coef.shape[0] == 1:
            class_label = (
                f"{classes[1]}_vs_{classes[0]}"
                if len(classes) == 2
                else "binary_axis"
            )
            vals = coef[0]
            for feature, value in zip(features, vals):
                rows.append(
                    {
                        "pair": pair,
                        "feature_panel": panel,
                        "scheme": scheme,
                        "model": model_name,
                        "heldout_group": heldout_group,
                        "class": class_label,
                        "feature": feature,
                        "importance_type": "coefficient",
                        "value": float(value),
                        "abs_value": float(abs(value)),
                    }
                )

        else:
            for class_idx, cls in enumerate(classes):
                if class_idx >= coef.shape[0]:
                    continue
                vals = coef[class_idx]
                for feature, value in zip(features, vals):
                    rows.append(
                        {
                            "pair": pair,
                            "feature_panel": panel,
                            "scheme": scheme,
                            "model": model_name,
                            "heldout_group": heldout_group,
                            "class": cls,
                            "feature": feature,
                            "importance_type": "coefficient",
                            "value": float(value),
                            "abs_value": float(abs(value)),
                        }
                    )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Validation group inventory
# -----------------------------------------------------------------------------

def build_validation_groups(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    schemes: list[str],
) -> pd.DataFrame:
    rows = []

    for pair_tuple in pairs:
        pair_name = binary_label(pair_tuple)
        sub = df[df[TARGET_COL].isin(pair_tuple)].copy()

        for scheme in schemes:
            group_col = scheme_group_col(scheme)

            if group_col is None:
                rows.append(
                    {
                        "pair": pair_name,
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
                        "pair": pair_name,
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
    pair_tuple: tuple[str, str],
    panel_name: str,
    features: list[str],
    model_name: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_name = binary_label(pair_tuple)
    sub = df[df[TARGET_COL].isin(pair_tuple)].copy()

    X = sub[features]
    y = sub[TARGET_COL].astype(str)

    min_class = min(Counter(y).values())
    splits = min(n_splits, min_class)

    if splits < 2:
        fail = pd.DataFrame(
            [
                {
                    "pair": pair_name,
                    "feature_panel": panel_name,
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
        "pair": pair_name,
        "case_a": pair_tuple[0],
        "case_b": pair_tuple[1],
        "feature_panel": panel_name,
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
        **score_predictions(y, pred),
    }

    pred_df = sub[["row_id", TARGET_COL, "object", "cohort", "transition"]].copy()
    pred_df["pair"] = pair_name
    pred_df["feature_panel"] = panel_name
    pred_df["scheme"] = "stratified_cv"
    pred_df["model"] = model_name
    pred_df["heldout_group"] = "stratified_cv"
    pred_df["prediction"] = pred
    pred_df["correct"] = pred_df[TARGET_COL].astype(str) == pred_df["prediction"].astype(str)

    cm = confusion_rows(
        y_true=y,
        y_pred=pred,
        pair=pair_name,
        panel=panel_name,
        scheme="stratified_cv",
        model=model_name,
        heldout_group="stratified_cv",
    )

    # Fit once on all data for feature importance.
    fitted = make_pipeline(model_name)
    fitted.fit(X, y)
    imp = feature_importance_rows(
        fitted=fitted,
        pair=pair_name,
        panel=panel_name,
        scheme="stratified_cv",
        model_name=model_name,
        features=features,
        heldout_group="all",
    )

    return pd.DataFrame([score]), pred_df, pd.DataFrame(), pd.concat([cm, imp], ignore_index=True, sort=False)


def evaluate_leave_group_out(
    df: pd.DataFrame,
    pair_tuple: tuple[str, str],
    panel_name: str,
    features: list[str],
    scheme: str,
    model_name: str,
    require_both_classes_in_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_name = binary_label(pair_tuple)
    group_col = scheme_group_col(scheme)
    assert group_col is not None

    sub = df[df[TARGET_COL].isin(pair_tuple)].copy()
    all_classes = set(pair_tuple)

    score_rows = []
    pred_parts = []
    fail_rows = []
    cm_parts = []
    imp_parts = []

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
                    "pair": pair_name,
                    "feature_panel": panel_name,
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
                    "pair": pair_name,
                    "feature_panel": panel_name,
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

        if train_classes != all_classes:
            valid_primary = False
            notes.append("train_missing_pair_class")

        if require_both_classes_in_test and test_classes != all_classes:
            valid_primary = False
            notes.append("test_missing_pair_class")

        if len(test_classes) < 2:
            valid_primary = False
            notes.append("test_has_less_than_two_classes")

        try:
            pipe = make_pipeline(model_name)
            pipe.fit(train[features], y_train)
            pred = pipe.predict(test[features])

            score = {
                "pair": pair_name,
                "case_a": pair_tuple[0],
                "case_b": pair_tuple[1],
                "feature_panel": panel_name,
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
                **score_predictions(y_test, pred),
            }
            score_rows.append(score)

            pred_df = test[["row_id", TARGET_COL, "object", "cohort", "transition"]].copy()
            pred_df["pair"] = pair_name
            pred_df["feature_panel"] = panel_name
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
                    pair=pair_name,
                    panel=panel_name,
                    scheme=scheme,
                    model=model_name,
                    heldout_group=str(group_value),
                )
            )

            imp = feature_importance_rows(
                fitted=pipe,
                pair=pair_name,
                panel=panel_name,
                scheme=scheme,
                model_name=model_name,
                features=features,
                heldout_group=str(group_value),
            )
            if not imp.empty:
                imp_parts.append(imp)

        except Exception as e:
            fail_rows.append(
                {
                    "pair": pair_name,
                    "feature_panel": panel_name,
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
    imps = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()

    return scores, preds, fails, cms, imps


# -----------------------------------------------------------------------------
# Permutations
# -----------------------------------------------------------------------------

def permutation_stratified_cv(
    df: pd.DataFrame,
    pair_tuple: tuple[str, str],
    panel_name: str,
    features: list[str],
    model_name: str,
    n_splits: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if model_name == "dummy":
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    pair_name = binary_label(pair_tuple)
    sub = df[df[TARGET_COL].isin(pair_tuple)].copy()

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
                "pair": pair_name,
                "feature_panel": panel_name,
                "scheme": "stratified_cv",
                "model": model_name,
                "permutation_index": i,
                **score_predictions(y_perm, pred),
            }
        )

    return pd.DataFrame(rows)


def summarize_permutations(scores: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    if perm.empty:
        return pd.DataFrame()

    rows = []

    keys = ["pair", "feature_panel", "scheme", "model"]

    for key_vals, g in perm.groupby(keys, dropna=False):
        key = dict(zip(keys, key_vals))

        actual = scores.copy()
        for k, v in key.items():
            actual = actual[actual[k] == v]

        # For stratified CV there is one row. For safety, use mean if multiple.
        actual_ba = float(pd.to_numeric(actual["balanced_accuracy"], errors="coerce").mean()) if not actual.empty else np.nan
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
            ["pair", "actual_balanced_accuracy"],
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

    rows = []
    keys = ["pair", "feature_panel", "scheme", "model"]

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
            ["summary_scope", "pair", "scheme", "mean_balanced_accuracy"],
            ascending=[True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)
    return out


def build_pairwise_axis_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compact primary summary for feature-axis anatomy.

    Focuses on stratified CV and primary valid leave-structure rows.
    """
    if summary.empty:
        return pd.DataFrame()

    q = summary[summary["summary_scope"] == "primary_valid_groups"].copy()
    if q.empty:
        return pd.DataFrame()

    rows = []
    for pair, g in q.groupby("pair"):
        for scheme, sg in g.groupby("scheme"):
            best = sg.sort_values("mean_balanced_accuracy", ascending=False).head(1)
            if best.empty:
                continue

            rows.append(
                {
                    "pair": pair,
                    "scheme": scheme,
                    "best_feature_panel": best["feature_panel"].iloc[0],
                    "best_model": best["model"].iloc[0],
                    "best_mean_balanced_accuracy": float(best["mean_balanced_accuracy"].iloc[0]),
                    "best_weighted_balanced_accuracy": float(best["weighted_balanced_accuracy_by_n_test"].iloc[0]),
                    "n_groups": int(best["n_groups"].iloc[0]),
                    "total_test_rows": int(best["total_test_rows"].iloc[0]),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pair", "scheme"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Manifests / report
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
    write_csv(out, outdir / "obs079c_input_manifest.csv")
    return out


def write_feature_panels(outdir: Path, panels: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for panel, features in panels.items():
        for i, feature in enumerate(features):
            rows.append(
                {
                    "feature_panel": panel,
                    "feature_index": i,
                    "feature": feature,
                    "n_features": len(features),
                }
            )
    out = pd.DataFrame(rows)
    write_csv(out, outdir / "obs079c_feature_panels.csv")
    return out


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    feature_panels: pd.DataFrame,
    validation_groups: pd.DataFrame,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    axis_summary: pd.DataFrame,
    perm_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# OBS-079c — Pairwise Stability Classifiers")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-079c tests which axes of the OBS-078 three-feature stability core "
        "support each pairwise case distinction."
    )
    lines.append("")
    lines.append("Pairwise tests:")
    lines.append("")
    lines.append("```text")
    for a, b in PAIRWISE_CASES:
        lines.append(f"{a} vs {b}")
    lines.append("```")
    lines.append("")
    lines.append("Feature panels:")
    lines.append("")
    panel_summary = (
        feature_panels.groupby("feature_panel", as_index=False)
        .agg(n_features=("feature", "count"), features=("feature", lambda x: ",".join(x)))
        .sort_values(["n_features", "feature_panel"])
    )
    lines.append(markdown_table(panel_summary, max_rows=50))
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest))
    lines.append("")
    lines.append("## Validation group inventory")
    lines.append("")
    group_summary = (
        validation_groups.groupby(["pair", "scheme", "group_col"], dropna=False)
        .agg(
            n_groups=("heldout_group", "count"),
            min_rows=("n_rows", "min"),
            mean_rows=("n_rows", "mean"),
            max_rows=("n_rows", "max"),
            min_classes=("n_classes", "min"),
            max_classes=("n_classes", "max"),
        )
        .reset_index()
        .sort_values(["pair", "scheme", "group_col"])
    )
    lines.append(markdown_table(group_summary, max_rows=100))
    lines.append("")
    lines.append("## Pairwise axis summary")
    lines.append("")
    if axis_summary.empty:
        lines.append("_No axis summary rows._")
    else:
        lines.append(markdown_table(axis_summary, max_rows=80))
    lines.append("")
    lines.append("## Scheme summary")
    lines.append("")
    if summary.empty:
        lines.append("_No scheme summary rows._")
    else:
        display = summary[
            [
                "pair",
                "feature_panel",
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
            ["summary_scope", "pair", "scheme", "mean_balanced_accuracy"],
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
                "pair",
                "feature_panel",
                "scheme",
                "model",
                "actual_balanced_accuracy",
                "perm_mean_balanced_accuracy",
                "perm_std_balanced_accuracy",
                "perm_p_ge_actual",
                "n_permutations",
            ]
        ].sort_values(["pair", "actual_balanced_accuracy"], ascending=[True, False])
        lines.append(markdown_table(display, max_rows=120))
    lines.append("")
    lines.append("## Top feature importances / coefficients")
    lines.append("")
    if feature_importance.empty:
        lines.append("_No feature-importance rows._")
    else:
        top = feature_importance.sort_values("abs_value", ascending=False).head(80)
        cols = [
            "pair",
            "feature_panel",
            "scheme",
            "model",
            "heldout_group",
            "class",
            "feature",
            "importance_type",
            "value",
            "abs_value",
        ]
        lines.append(markdown_table(top[[c for c in cols if c in top.columns]], max_rows=80))
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
    lines.append("Expected pattern:")
    lines.append("")
    lines.append("```text")
    lines.append("C vs Cp2:")
    lines.append("  strong separation via boundedness + divergence")
    lines.append("")
    lines.append("C vs Cp3:")
    lines.append("  strongest separation via boundedness + displacement/instability")
    lines.append("")
    lines.append("Cp2 vs Cp3:")
    lines.append("  weak with lambda_only")
    lines.append("  stronger with delta_d / boundedness / full core")
    lines.append("```")
    lines.append("")
    lines.append("Canonical supported statement if confirmed:")
    lines.append("")
    lines.append("```text")
    lines.append("C separates from Cp2/Cp3 by the full bounded-stability regime.")
    lines.append("Cp2 and Cp3 separate mainly by displacement/boundedness,")
    lines.append("not by lambda expansion.")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-079c is a pairwise anatomy diagnostic, not causal proof.")
    lines.append("Held-out groups with incomplete class coverage should not be overinterpreted.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs079c_input_manifest.csv")
    lines.append("obs079c_feature_panels.csv")
    lines.append("obs079c_validation_groups.csv")
    lines.append("obs079c_pairwise_scores.csv")
    lines.append("obs079c_pairwise_predictions.csv")
    lines.append("obs079c_pairwise_permutation_scores.csv")
    lines.append("obs079c_pairwise_permutation_summary.csv")
    lines.append("obs079c_pairwise_scheme_summary.csv")
    lines.append("obs079c_pairwise_axis_summary.csv")
    lines.append("obs079c_feature_importance.csv")
    lines.append("obs079c_confusion_matrices.csv")
    lines.append("obs079c_failures.csv")
    lines.append("obs079c_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-079c")

    (outdir / "obs079c_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-079c pairwise stability classifiers.")
    ap.add_argument(
        "--feature-table",
        default=(
            "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/"
            "obs078a_feature_table.csv"
        ),
    )
    ap.add_argument(
        "--outdir",
        default="outputs/comparisons/obs079c_pairwise_stability_classifiers",
    )
    ap.add_argument(
        "--models",
        default="logreg,tree_depth2,rf_depth2,dummy",
        help="Comma-separated model names.",
    )
    ap.add_argument(
        "--schemes",
        default=",".join(VALIDATION_SCHEMES),
        help="Comma-separated validation schemes.",
    )
    ap.add_argument(
        "--feature-panels",
        default=",".join(FEATURE_PANELS.keys()),
        help="Comma-separated feature panels.",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument(
        "--require-both-classes-in-test",
        action="store_true",
        help="Mark held-out groups missing either pairwise class as not primary-valid.",
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

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    unknown_schemes = sorted(set(schemes) - set(VALIDATION_SCHEMES))
    if unknown_schemes:
        raise ValueError(f"Unknown validation schemes: {unknown_schemes}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    panel_names = [p.strip() for p in args.feature_panels.split(",") if p.strip()]
    unknown_panels = sorted(set(panel_names) - set(FEATURE_PANELS.keys()))
    if unknown_panels:
        raise ValueError(f"Unknown feature panels: {unknown_panels}")

    panels = {name: FEATURE_PANELS[name] for name in panel_names}

    input_manifest = write_input_manifest(outdir, feature_table_path, df)
    feature_panels_df = write_feature_panels(outdir, panels)

    validation_groups = build_validation_groups(df, PAIRWISE_CASES, schemes)
    write_csv(validation_groups, outdir / "obs079c_validation_groups.csv")

    score_parts = []
    pred_parts = []
    fail_parts = []
    cm_parts = []
    imp_parts = []
    perm_parts = []

    run_index = 0

    for pair_tuple in PAIRWISE_CASES:
        for panel_name, features in panels.items():
            for scheme in schemes:
                for model_name in model_names:
                    run_index += 1

                    if scheme == "stratified_cv":
                        scores, preds, fails, cm_or_imp = evaluate_stratified_cv(
                            df=df,
                            pair_tuple=pair_tuple,
                            panel_name=panel_name,
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

                        if not cm_or_imp.empty:
                            # This mixed dataframe contains confusion rows and possibly
                            # importance rows. Split by available columns.
                            if "actual" in cm_or_imp.columns and "predicted" in cm_or_imp.columns:
                                cm_parts.append(cm_or_imp[cm_or_imp["actual"].notna()].copy())
                            if "feature" in cm_or_imp.columns:
                                imp_parts.append(cm_or_imp[cm_or_imp["feature"].notna()].copy())

                        if args.n_permutations > 0 and model_name != "dummy":
                            perm = permutation_stratified_cv(
                                df=df,
                                pair_tuple=pair_tuple,
                                panel_name=panel_name,
                                features=features,
                                model_name=model_name,
                                n_splits=args.n_splits,
                                n_permutations=args.n_permutations,
                                seed=args.seed + run_index,
                            )
                            if not perm.empty:
                                perm_parts.append(perm)

                    else:
                        scores, preds, fails, cms, imps = evaluate_leave_group_out(
                            df=df,
                            pair_tuple=pair_tuple,
                            panel_name=panel_name,
                            features=features,
                            scheme=scheme,
                            model_name=model_name,
                            require_both_classes_in_test=args.require_both_classes_in_test,
                        )

                        if not scores.empty:
                            score_parts.append(scores)
                        if not preds.empty:
                            pred_parts.append(preds)
                        if not fails.empty:
                            fail_parts.append(fails)
                        if not cms.empty:
                            cm_parts.append(cms)
                        if not imps.empty:
                            imp_parts.append(imps)

    scores_df = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    if not scores_df.empty:
        scores_df = scores_df.sort_values(
            ["pair", "scheme", "feature_panel", "model", "balanced_accuracy"],
            ascending=[True, True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    preds_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    failures_df = pd.concat(fail_parts, ignore_index=True) if fail_parts else pd.DataFrame()
    cm_df = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()
    imp_df = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()
    perm_df = pd.concat(perm_parts, ignore_index=True) if perm_parts else pd.DataFrame()

    if not imp_df.empty:
        imp_df = imp_df.sort_values(
            ["pair", "feature_panel", "scheme", "model", "abs_value"],
            ascending=[True, True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    summary_df = summarize_schemes(scores_df)
    axis_summary_df = build_pairwise_axis_summary(summary_df)
    perm_summary_df = summarize_permutations(scores_df, perm_df)

    write_csv(scores_df, outdir / "obs079c_pairwise_scores.csv")
    write_csv(preds_df, outdir / "obs079c_pairwise_predictions.csv")
    write_csv(perm_df, outdir / "obs079c_pairwise_permutation_scores.csv")
    write_csv(perm_summary_df, outdir / "obs079c_pairwise_permutation_summary.csv")
    write_csv(summary_df, outdir / "obs079c_pairwise_scheme_summary.csv")
    write_csv(axis_summary_df, outdir / "obs079c_pairwise_axis_summary.csv")
    write_csv(imp_df, outdir / "obs079c_feature_importance.csv")
    write_csv(cm_df, outdir / "obs079c_confusion_matrices.csv")
    write_csv(failures_df, outdir / "obs079c_failures.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        feature_panels=feature_panels_df,
        validation_groups=validation_groups,
        scores=scores_df,
        summary=summary_df,
        axis_summary=axis_summary_df,
        perm_summary=perm_summary_df,
        feature_importance=imp_df,
        failures=failures_df,
    )

    print(f"[OBS-079c] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
