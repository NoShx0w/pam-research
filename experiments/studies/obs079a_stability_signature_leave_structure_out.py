#!/usr/bin/env python3
"""
obs079a_stability_signature_leave_structure_out.py

OBS-079a — Stability Signature Leave-Structure-Out Robustness.

Purpose
-------
OBS-078 found that the C / Cp2 / Cp3 distinction compresses to a small
window-local stability signature:

    mean_lambda_local_mean
    mean_delta_d_mean
    bounded_share_mean

OBS-079a asks:

    Does this 3-feature stability core survive structural perturbation?

Specifically, can it classify C / Cp2 / Cp3 when whole structural groups are
held out?

Validation schemes
------------------
  standard_stratified_cv
  leave_object_out
  leave_cohort_out
  leave_transition_out
  leave_object_cohort_out
  leave_object_transition_out

Inputs
------
Default:

  outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/
    obs078a_feature_table.csv

Outputs
-------
  obs079a_input_manifest.csv
  obs079a_feature_manifest.csv
  obs079a_validation_groups.csv
  obs079a_leave_structure_scores.csv
  obs079a_leave_structure_predictions.csv
  obs079a_leave_structure_failures.csv
  obs079a_scheme_summary.csv
  obs079a_report.md

Scientific guardrail
--------------------
OBS-079a is a robustness diagnostic. It does not establish causality.

The key supported statement is:

    The OBS-078 local stability signature generalizes across held-out
    structural partitions if performance remains above chance when objects,
    cohorts, or transitions are withheld.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_RANDOM_STATE = 79001

STABILITY_FEATURES = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

TARGET_COL = "case"

SCHEMES = [
    "standard_stratified_cv",
    "leave_object_out",
    "leave_cohort_out",
    "leave_transition_out",
    "leave_object_cohort_out",
    "leave_object_transition_out",
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
    counts = Counter(y.astype(str))
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def n_classes(y: pd.Series) -> int:
    return int(y.astype(str).nunique(dropna=True))


def transition_label(row: pd.Series) -> str:
    return f"{row.get('scale_index_from')}→{row.get('scale_index_to')}"


def object_cohort_label(row: pd.Series) -> str:
    return f"{row.get('object')}::{row.get('cohort')}"


def object_transition_label(row: pd.Series) -> str:
    return f"{row.get('object')}::{row.get('transition')}"


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
# Scoring
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
    scheme: str,
    model: str,
    heldout_group: str,
) -> pd.DataFrame:
    labels = sorted(pd.Series(y_true).astype(str).unique())
    cm = confusion_matrix(pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str), labels=labels)

    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            rows.append(
                {
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
# Validation group construction
# -----------------------------------------------------------------------------

def prepare_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    require_columns(out, ["object", "cohort"], "feature table")

    if "transition" not in out.columns:
        require_columns(out, ["scale_index_from", "scale_index_to"], "feature table")
        out["transition"] = out.apply(transition_label, axis=1)

    out["object_cohort"] = out.apply(object_cohort_label, axis=1)
    out["object_transition"] = out.apply(object_transition_label, axis=1)

    return out


def scheme_group_col(scheme: str) -> str | None:
    if scheme == "standard_stratified_cv":
        return None
    if scheme == "leave_object_out":
        return "object"
    if scheme == "leave_cohort_out":
        return "cohort"
    if scheme == "leave_transition_out":
        return "transition"
    if scheme == "leave_object_cohort_out":
        return "object_cohort"
    if scheme == "leave_object_transition_out":
        return "object_transition"
    raise ValueError(f"Unknown scheme: {scheme}")


def build_validation_groups(df: pd.DataFrame, schemes: list[str]) -> pd.DataFrame:
    rows = []

    for scheme in schemes:
        group_col = scheme_group_col(scheme)

        if group_col is None:
            rows.append(
                {
                    "scheme": scheme,
                    "group_col": "",
                    "heldout_group": "stratified_cv",
                    "n_rows": int(len(df)),
                    "n_classes": n_classes(df[TARGET_COL]),
                    "class_counts": class_counts_string(df[TARGET_COL]),
                }
            )
            continue

        for group_value, g in df.groupby(group_col, dropna=False):
            rows.append(
                {
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

def evaluate_standard_cv(
    df: pd.DataFrame,
    model_name: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df[STABILITY_FEATURES]
    y = df[TARGET_COL].astype(str)

    min_class = min(Counter(y).values())
    splits = min(n_splits, min_class)

    if splits < 2:
        fail = pd.DataFrame(
            [
                {
                    "scheme": "standard_stratified_cv",
                    "model": model_name,
                    "heldout_group": "stratified_cv",
                    "reason": "not_enough_class_support",
                    "n_rows": int(len(df)),
                    "class_counts": class_counts_string(y),
                }
            ]
        )
        return pd.DataFrame(), pd.DataFrame(), fail

    cv = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=MODEL_RANDOM_STATE,
    )

    pipe = make_pipeline(model_name)
    pred = cross_val_predict(pipe, X, y, cv=cv)

    score = {
        "scheme": "standard_stratified_cv",
        "model": model_name,
        "heldout_group": "stratified_cv",
        "group_col": "",
        "n_train": int(len(df)),
        "n_test": int(len(df)),
        "n_train_classes": n_classes(y),
        "n_test_classes": n_classes(y),
        "train_class_counts": class_counts_string(y),
        "test_class_counts": class_counts_string(y),
        "valid_for_primary_summary": True,
        "note": "",
        **score_predictions(y, pred),
    }

    pred_df = df[["row_id", TARGET_COL, "object", "cohort", "transition"]].copy()
    pred_df["scheme"] = "standard_stratified_cv"
    pred_df["model"] = model_name
    pred_df["heldout_group"] = "stratified_cv"
    pred_df["prediction"] = pred
    pred_df["correct"] = pred_df[TARGET_COL].astype(str) == pred_df["prediction"].astype(str)

    cm = confusion_rows(y, pred, "standard_stratified_cv", model_name, "stratified_cv")

    return pd.DataFrame([score]), pred_df, pd.DataFrame()


def evaluate_leave_group_out(
    df: pd.DataFrame,
    scheme: str,
    model_name: str,
    require_all_classes_in_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_col = scheme_group_col(scheme)
    assert group_col is not None

    score_rows = []
    pred_parts = []
    fail_rows = []
    cm_parts = []

    all_classes = sorted(df[TARGET_COL].astype(str).unique())

    for group_value, test in df.groupby(group_col, dropna=False):
        train = df[df[group_col] != group_value]

        y_train = train[TARGET_COL].astype(str)
        y_test = test[TARGET_COL].astype(str)

        train_classes = sorted(y_train.unique())
        test_classes = sorted(y_test.unique())

        valid_primary = True
        notes = []

        if len(train) == 0 or len(test) == 0:
            fail_rows.append(
                {
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

        missing_train_classes = sorted(set(all_classes) - set(train_classes))
        if missing_train_classes:
            valid_primary = False
            notes.append(f"missing_train_classes={','.join(missing_train_classes)}")

        if require_all_classes_in_test and set(test_classes) != set(all_classes):
            valid_primary = False
            notes.append("test_missing_some_classes")

        if len(test_classes) < 2:
            valid_primary = False
            notes.append("test_has_less_than_two_classes")

        try:
            pipe = make_pipeline(model_name)
            pipe.fit(train[STABILITY_FEATURES], y_train)
            pred = pipe.predict(test[STABILITY_FEATURES])

            score = {
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
            pred_df["scheme"] = scheme
            pred_df["model"] = model_name
            pred_df["heldout_group"] = str(group_value)
            pred_df["prediction"] = pred
            pred_df["correct"] = pred_df[TARGET_COL].astype(str) == pred_df["prediction"].astype(str)
            pred_parts.append(pred_df)

            cm_parts.append(confusion_rows(y_test, pred, scheme, model_name, str(group_value)))

        except Exception as e:
            fail_rows.append(
                {
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


def summarize_schemes(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    rows = []

    for (scheme, model), g in scores.groupby(["scheme", "model"], dropna=False):
        primary = g[g["valid_for_primary_summary"].astype(bool)] if "valid_for_primary_summary" in g.columns else g

        for label, gg in [("all_groups", g), ("primary_valid_groups", primary)]:
            if gg.empty:
                continue

            rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "summary_scope": label,
                    "n_groups": int(len(gg)),
                    "total_test_rows": int(pd.to_numeric(gg["n_test"], errors="coerce").sum()),
                    "mean_accuracy": float(pd.to_numeric(gg["accuracy"], errors="coerce").mean()),
                    "mean_balanced_accuracy": float(pd.to_numeric(gg["balanced_accuracy"], errors="coerce").mean()),
                    "mean_macro_f1": float(pd.to_numeric(gg["macro_f1"], errors="coerce").mean()),
                    "weighted_accuracy_by_n_test": float(
                        np.average(
                            pd.to_numeric(gg["accuracy"], errors="coerce"),
                            weights=pd.to_numeric(gg["n_test"], errors="coerce"),
                        )
                    ) if pd.to_numeric(gg["n_test"], errors="coerce").sum() > 0 else np.nan,
                    "weighted_balanced_accuracy_by_n_test": float(
                        np.average(
                            pd.to_numeric(gg["balanced_accuracy"], errors="coerce"),
                            weights=pd.to_numeric(gg["n_test"], errors="coerce"),
                        )
                    ) if pd.to_numeric(gg["n_test"], errors="coerce").sum() > 0 else np.nan,
                    "min_balanced_accuracy": float(pd.to_numeric(gg["balanced_accuracy"], errors="coerce").min()),
                    "max_balanced_accuracy": float(pd.to_numeric(gg["balanced_accuracy"], errors="coerce").max()),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["summary_scope", "mean_balanced_accuracy"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)

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
    write_csv(out, outdir / "obs079a_input_manifest.csv")
    return out


def write_feature_manifest(outdir: Path) -> pd.DataFrame:
    rows = [
        {
            "feature": "mean_lambda_local_mean",
            "role": "stability_core",
            "interpretation": "local divergence / expansion tendency",
        },
        {
            "feature": "mean_delta_d_mean",
            "role": "stability_core",
            "interpretation": "local distance-growth / displacement tendency",
        },
        {
            "feature": "bounded_share_mean",
            "role": "stability_core",
            "interpretation": "local boundedness / containment tendency",
        },
    ]
    out = pd.DataFrame(rows)
    write_csv(out, outdir / "obs079a_feature_manifest.csv")
    return out


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    validation_groups: pd.DataFrame,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# OBS-079a — Stability Signature Leave-Structure-Out Robustness")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-079a tests whether the OBS-078 three-feature local stability core "
        "survives held-out structural partitions."
    )
    lines.append("")
    lines.append("Stability core:")
    lines.append("")
    lines.append("```text")
    for f in STABILITY_FEATURES:
        lines.append(f)
    lines.append("```")
    lines.append("")
    lines.append("Validation schemes:")
    lines.append("")
    lines.append("```text")
    for s in SCHEMES:
        lines.append(s)
    lines.append("```")
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest))
    lines.append("")
    lines.append("## Validation group inventory")
    lines.append("")
    group_summary = (
        validation_groups.groupby(["scheme", "group_col"], dropna=False)
        .agg(
            n_groups=("heldout_group", "count"),
            min_rows=("n_rows", "min"),
            max_rows=("n_rows", "max"),
            mean_rows=("n_rows", "mean"),
            min_classes=("n_classes", "min"),
            max_classes=("n_classes", "max"),
        )
        .reset_index()
        .sort_values(["scheme", "group_col"])
    )
    lines.append(markdown_table(group_summary, max_rows=80))
    lines.append("")
    lines.append("## Scheme summary")
    lines.append("")
    if summary.empty:
        lines.append("_No scheme summary rows._")
    else:
        display = summary[
            [
                "scheme",
                "model",
                "summary_scope",
                "n_groups",
                "total_test_rows",
                "mean_accuracy",
                "mean_balanced_accuracy",
                "mean_macro_f1",
                "weighted_balanced_accuracy_by_n_test",
                "min_balanced_accuracy",
                "max_balanced_accuracy",
            ]
        ].sort_values(
            ["summary_scope", "scheme", "mean_balanced_accuracy"],
            ascending=[True, True, False],
        )
        lines.append(markdown_table(display, max_rows=120))
    lines.append("")
    lines.append("## Best rows")
    lines.append("")
    if scores.empty:
        lines.append("_No score rows._")
    else:
        best = scores.sort_values("balanced_accuracy", ascending=False, na_position="last")
        lines.append(markdown_table(best[
            [
                "scheme",
                "model",
                "heldout_group",
                "n_train",
                "n_test",
                "n_train_classes",
                "n_test_classes",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "valid_for_primary_summary",
                "note",
            ]
        ], max_rows=80))
    lines.append("")
    lines.append("## Failure / skipped rows")
    lines.append("")
    if failures.empty:
        lines.append("_No failures._")
    else:
        lines.append(markdown_table(failures, max_rows=80))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("Strong robustness evidence:")
    lines.append("")
    lines.append("```text")
    lines.append("leave_object_out:")
    lines.append("  mean balanced accuracy remains above chance")
    lines.append("")
    lines.append("leave_cohort_out:")
    lines.append("  stability signature does not depend on one cohort label")
    lines.append("")
    lines.append("leave_transition_out:")
    lines.append("  stability signature does not depend on one scale transition")
    lines.append("```")
    lines.append("")
    lines.append("Suggested thresholds:")
    lines.append("")
    lines.append("```text")
    lines.append("leave_object_out       BA > 0.60 = useful")
    lines.append("leave_cohort_out       BA > 0.70 = strong")
    lines.append("leave_transition_out   BA > 0.70 = strong")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-079a is a robustness diagnostic, not causal proof.")
    lines.append("Groups with incomplete test class coverage are recorded and should not be overinterpreted.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs079a_input_manifest.csv")
    lines.append("obs079a_feature_manifest.csv")
    lines.append("obs079a_validation_groups.csv")
    lines.append("obs079a_leave_structure_scores.csv")
    lines.append("obs079a_leave_structure_predictions.csv")
    lines.append("obs079a_leave_structure_failures.csv")
    lines.append("obs079a_confusion_matrices.csv")
    lines.append("obs079a_scheme_summary.csv")
    lines.append("obs079a_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-079a")

    (outdir / "obs079a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-079a stability signature leave-structure-out robustness.")
    ap.add_argument(
        "--feature-table",
        default=(
            "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/"
            "obs078a_feature_table.csv"
        ),
    )
    ap.add_argument(
        "--outdir",
        default="outputs/comparisons/obs079a_stability_signature_leave_structure_out",
    )
    ap.add_argument(
        "--models",
        default="logreg,tree_depth2,rf_depth2,dummy",
        help="Comma-separated model names.",
    )
    ap.add_argument(
        "--schemes",
        default=",".join(SCHEMES),
        help="Comma-separated validation schemes.",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument(
        "--require-all-classes-in-test",
        action="store_true",
        help=(
            "Mark held-out groups missing one or more target classes as not valid "
            "for primary summary. This is recommended for strict interpretation."
        ),
    )
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

    input_manifest = write_input_manifest(outdir, feature_table_path, df)
    write_feature_manifest(outdir)

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    unknown_schemes = sorted(set(schemes) - set(SCHEMES))
    if unknown_schemes:
        raise ValueError(f"Unknown schemes: {unknown_schemes}")

    validation_groups = build_validation_groups(df, schemes)
    write_csv(validation_groups, outdir / "obs079a_validation_groups.csv")

    score_parts = []
    pred_parts = []
    fail_parts = []
    cm_parts = []

    for scheme in schemes:
        for model_name in models:
            if scheme == "standard_stratified_cv":
                scores, preds, fails = evaluate_standard_cv(
                    df=df,
                    model_name=model_name,
                    n_splits=args.n_splits,
                )

                if not scores.empty:
                    score_parts.append(scores)
                    cm_parts.append(
                        confusion_rows(
                            y_true=df[TARGET_COL],
                            y_pred=preds["prediction"],
                            scheme=scheme,
                            model=model_name,
                            heldout_group="stratified_cv",
                        )
                    )
                if not preds.empty:
                    pred_parts.append(preds)
                if not fails.empty:
                    fail_parts.append(fails)

            else:
                scores, preds, fails, cms = evaluate_leave_group_out(
                    df=df,
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
            ["scheme", "model", "balanced_accuracy"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    preds_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    failures_df = pd.concat(fail_parts, ignore_index=True) if fail_parts else pd.DataFrame()
    cms_df = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()

    summary_df = summarize_schemes(scores_df)

    write_csv(scores_df, outdir / "obs079a_leave_structure_scores.csv")
    write_csv(preds_df, outdir / "obs079a_leave_structure_predictions.csv")
    write_csv(failures_df, outdir / "obs079a_leave_structure_failures.csv")
    write_csv(cms_df, outdir / "obs079a_confusion_matrices.csv")
    write_csv(summary_df, outdir / "obs079a_scheme_summary.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        validation_groups=validation_groups,
        scores=scores_df,
        summary=summary_df,
        failures=failures_df,
    )

    print(f"[OBS-079a] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
