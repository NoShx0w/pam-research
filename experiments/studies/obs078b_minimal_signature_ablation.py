#!/usr/bin/env python3
"""
obs078b_minimal_signature_ablation.py

OBS-078b — Minimal Signature Ablation.

Purpose
-------
OBS-078a established that C / Cp2 / Cp3 are separable from OBS-077-derived
mechanistic features under strict anti-leakage controls.

OBS-078b asks:

    What is the smallest strict feature subset that still recovers C / Cp2 / Cp3?

This script reuses the OBS-078a v2 feature table and feature manifest. It does
not rebuild artifacts from OBS-077. It performs feature-family and minimal-panel
ablations over already validated OBS-077-derived features.

Inputs
------
  --feature-table
      outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/
        obs078a_feature_table.csv

  --feature-manifest
      outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/
        obs078a_feature_manifest.csv

Outputs
-------
  obs078b_input_manifest.csv
  obs078b_feature_sets.csv
  obs078b_panel_scores.csv
  obs078b_permutation_scores.csv
  obs078b_permutation_summary.csv
  obs078b_feature_importance.csv
  obs078b_confusion_matrices.csv
  obs078b_minimal_signature_summary.md

Primary interpretation
----------------------
A useful minimal signature is one that:

  - uses strict object-blind features only
  - has few features, ideally <= 10
  - scores far above permutation baseline
  - uses features matching the OBS-077 mechanism:
        boundedness
        local divergence
        path recovery/nonrecovery composition
        path-family/seam composition
        transition geometry

Scientific guardrail
--------------------
OBS-078b is a separability/minimality diagnostic, not causal proof.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_RANDOM_STATE = 78078


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


def parse_bool(s: str) -> bool:
    x = str(s).strip().lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool: {s}")


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if "strict_excluded" in out.columns:
        out["strict_excluded"] = out["strict_excluded"].map(parse_bool)
    else:
        out["strict_excluded"] = False

    for col in ["feature", "family", "role"]:
        if col not in out.columns:
            raise ValueError(f"Feature manifest missing required column: {col}")

    return out


def feature_exists(table: pd.DataFrame, feature: str) -> bool:
    return feature in table.columns


def existing_features(table: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in table.columns]


def strict_allowed_features(manifest: pd.DataFrame) -> set[str]:
    q = manifest[
        (manifest["role"] == "numeric")
        & (~manifest["strict_excluded"])
        & (~manifest["family"].isin(["target", "key", "identity"]))
    ]
    return set(q["feature"].astype(str))


def strict_numeric_features_by_family(manifest: pd.DataFrame, families: list[str]) -> list[str]:
    q = manifest[
        (manifest["role"] == "numeric")
        & (~manifest["strict_excluded"])
        & (manifest["family"].isin(families))
    ]
    return list(q["feature"].astype(str))


def features_matching(
    table: pd.DataFrame,
    allowed: set[str],
    include_substrings: list[str] | None = None,
    include_suffixes: list[str] | None = None,
    include_prefixes: list[str] | None = None,
    exact: list[str] | None = None,
) -> list[str]:
    include_substrings = include_substrings or []
    include_suffixes = include_suffixes or []
    include_prefixes = include_prefixes or []
    exact = exact or []

    cols: list[str] = []

    for f in exact:
        if f in table.columns and f in allowed:
            cols.append(f)

    for col in table.columns:
        if col not in allowed:
            continue
        if any(s in col for s in include_substrings):
            cols.append(col)
            continue
        if any(col.endswith(s) for s in include_suffixes):
            cols.append(col)
            continue
        if any(col.startswith(s) for s in include_prefixes):
            cols.append(col)
            continue

    return list(dict.fromkeys(cols))


# -----------------------------------------------------------------------------
# Feature set definitions
# -----------------------------------------------------------------------------

def build_feature_sets(table: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, list[str]]:
    """
    Build strict object-blind feature sets.

    All returned features:
      - must exist in feature table
      - must be numeric
      - must not be strict_excluded
      - must not be identity/key/target
    """
    allowed = strict_allowed_features(manifest)

    geometry_all = strict_numeric_features_by_family(manifest, ["geometry"])
    path_all = strict_numeric_features_by_family(manifest, ["path_labels"])
    window_all = strict_numeric_features_by_family(manifest, ["window_divergence"])

    # Core exact features from OBS-077 / OBS-078a v2.
    window_means = existing_features(
        table,
        [
            "mean_lambda_local_mean",
            "mean_delta_d_mean",
            "bounded_share_mean",
        ],
    )
    window_means = [f for f in window_means if f in allowed]

    window_z = existing_features(
        table,
        [
            "mean_lambda_local_z",
            "mean_delta_d_z",
            "bounded_share_z",
            "divergence_z_sum",
        ],
    )
    window_z = [f for f in window_z if f in allowed]

    window_no_counts = [f for f in window_all if f in table.columns and f in allowed]

    path_shares = features_matching(
        table,
        allowed,
        include_suffixes=["__path_share"],
    )

    path_enrichment = features_matching(
        table,
        allowed,
        include_suffixes=[
            "__path_enrichment",
            "__path_log2_enrichment",
            "__enrichment",
            "__log2_enrichment",
        ],
    )

    # Avoid accidentally including non-strict global/count metrics.
    path_enrichment = [
        f for f in path_enrichment
        if "global" not in f and "__n_paths" not in f and "__denom_paths" not in f
    ]

    outcome_shares = features_matching(
        table,
        allowed,
        include_prefixes=["outcome__"],
        include_suffixes=["__path_share"],
    )

    path_family_shares = features_matching(
        table,
        allowed,
        include_prefixes=["path_family__"],
        include_suffixes=["__path_share"],
    )

    seam_shares = features_matching(
        table,
        allowed,
        include_prefixes=["seam__"],
        include_suffixes=["__path_share"],
    )

    projection_enrichment = features_matching(
        table,
        allowed,
        include_prefixes=["proj__"],
        include_suffixes=["__enrichment", "__log2_enrichment"],
    )

    geometry_core = existing_features(
        table,
        [
            "support_score",
            "shape_score",
            "id_score",
            "centroid_drift",
        ],
    )
    geometry_core = [f for f in geometry_core if f in allowed]

    geometry_support = existing_features(
        table,
        [
            "support_score",
            "jaccard_loss",
            "centroid_drift",
            "support_size_delta",
        ],
    )
    geometry_support = [f for f in geometry_support if f in allowed]

    geometry_shape_id = existing_features(
        table,
        [
            "shape_score",
            "id_score",
            "overlap_score",
            "overlap_delta_max",
        ],
    )
    geometry_shape_id = [f for f in geometry_shape_id if f in allowed]

    geometry_scores_only = existing_features(
        table,
        [
            "support_score",
            "overlap_score",
            "shape_score",
            "id_score",
        ],
    )
    geometry_scores_only = [f for f in geometry_scores_only if f in allowed]

    minimal_obs077_candidate = existing_features(
        table,
        [
            "bounded_share_mean",
            "mean_lambda_local_mean",
            "mean_delta_d_mean",
            "outcome__recovering__path_share",
            "outcome__nonrecovering__path_share",
            "path_family__stable_seam_corridor__path_share",
            "path_family__off_seam_reorganizing__path_share",
            "support_score",
            "id_score",
        ],
    )
    minimal_obs077_candidate = [f for f in minimal_obs077_candidate if f in allowed]

    minimal_window_path = existing_features(
        table,
        [
            "bounded_share_mean",
            "mean_lambda_local_mean",
            "mean_delta_d_mean",
            "outcome__recovering__path_share",
            "outcome__nonrecovering__path_share",
            "path_family__stable_seam_corridor__path_share",
            "path_family__off_seam_reorganizing__path_share",
        ],
    )
    minimal_window_path = [f for f in minimal_window_path if f in allowed]

    minimal_window_geometry = existing_features(
        table,
        [
            "bounded_share_mean",
            "mean_lambda_local_mean",
            "mean_delta_d_mean",
            "support_score",
            "id_score",
            "centroid_drift",
        ],
    )
    minimal_window_geometry = [f for f in minimal_window_geometry if f in allowed]

    feature_sets: dict[str, list[str]] = {
        "window_means_only": window_means,
        "window_z_only": window_z,
        "window_all_no_counts": window_no_counts,
        "path_shares_only": path_shares,
        "path_enrichment_only": path_enrichment,
        "outcome_shares_only": outcome_shares,
        "path_family_shares_only": path_family_shares,
        "seam_shares_only": seam_shares,
        "projection_enrichment_only": projection_enrichment,
        "geometry_core_only": geometry_core,
        "geometry_support_only": geometry_support,
        "geometry_shape_id_only": geometry_shape_id,
        "geometry_scores_only": geometry_scores_only,
        "no_path_labels": list(dict.fromkeys(geometry_core + window_no_counts)),
        "no_geometry": list(dict.fromkeys(path_shares + path_enrichment + window_no_counts)),
        "no_windows": list(dict.fromkeys(geometry_core + path_shares + path_enrichment)),
        "paths_plus_windows_minimal": list(dict.fromkeys(path_shares + window_no_counts)),
        "geometry_plus_windows_minimal": list(dict.fromkeys(geometry_core + window_no_counts)),
        "geometry_plus_paths_minimal": list(dict.fromkeys(geometry_core + path_shares + path_enrichment)),
        "minimal_obs077_candidate": minimal_obs077_candidate,
        "minimal_window_path": minimal_window_path,
        "minimal_window_geometry": minimal_window_geometry,
        "full_strict_object_blind": list(dict.fromkeys(geometry_all + path_all + window_all)),
    }

    # Keep non-empty only and remove duplicates.
    cleaned: dict[str, list[str]] = {}
    for name, feats in feature_sets.items():
        feats2 = list(dict.fromkeys([f for f in feats if f in table.columns and f in allowed]))
        if feats2:
            cleaned[name] = feats2

    return cleaned


def write_feature_sets(outdir: Path, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, feats in feature_sets.items():
        for i, feat in enumerate(feats):
            rows.append(
                {
                    "feature_set": name,
                    "feature_index": i,
                    "feature": feat,
                    "n_features": len(feats),
                }
            )
    df = pd.DataFrame(rows)
    write_csv(df, outdir / "obs078b_feature_sets.csv")
    return df


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def make_preprocessor(features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )


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


def make_pipeline(model_name: str, features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            ("model", make_model(model_name)),
        ]
    )


def choose_cv(y: pd.Series, cv_mode: str) -> Any:
    counts = Counter(y.astype(str))
    min_class = min(counts.values())

    if cv_mode == "loo":
        return LeaveOneOut()

    if cv_mode in {"auto", "stratified"}:
        n_splits = min(5, min_class)
        if n_splits >= 2:
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=MODEL_RANDOM_STATE,
            )
        return LeaveOneOut()

    raise ValueError(f"Unknown cv mode: {cv_mode}")


def score_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_feature_set(
    table: pd.DataFrame,
    feature_set: str,
    features: list[str],
    model_name: str,
    cv_mode: str,
) -> tuple[dict[str, Any], np.ndarray, Pipeline]:
    X = table[features].copy()
    y = table["case"].astype(str)

    pipe = make_pipeline(model_name, features)
    cv = choose_cv(y, cv_mode)

    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    fitted = make_pipeline(model_name, features)
    fitted.fit(X, y)

    result = {
        "feature_set": feature_set,
        "model": model_name,
        "cv_mode": cv_mode,
        "n_rows": int(len(table)),
        "n_features": int(len(features)),
        "features": ",".join(features),
        **score_predictions(y, y_pred),
    }

    return result, y_pred, fitted


def permutation_scores(
    table: pd.DataFrame,
    feature_set: str,
    features: list[str],
    model_name: str,
    cv_mode: str,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    X = table[features].copy()
    y0 = table["case"].astype(str).to_numpy()

    rows = []
    for i in range(n_permutations):
        y_perm = pd.Series(rng.permutation(y0))

        pipe = make_pipeline(model_name, features)
        cv = choose_cv(y_perm, cv_mode)
        y_pred = cross_val_predict(pipe, X, y_perm, cv=cv)

        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "permutation_index": i,
                **score_predictions(y_perm, y_pred),
            }
        )

    return pd.DataFrame(rows)


def summarize_permutations(panel_scores: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    if perm.empty:
        return pd.DataFrame()

    rows = []

    for (feature_set, model), g in perm.groupby(["feature_set", "model"]):
        actual = panel_scores[
            (panel_scores["feature_set"] == feature_set)
            & (panel_scores["model"] == model)
        ]

        actual_ba = float(actual["balanced_accuracy"].iloc[0]) if not actual.empty else np.nan

        ba = pd.to_numeric(g["balanced_accuracy"], errors="coerce")
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "actual_balanced_accuracy": actual_ba,
                "perm_mean_balanced_accuracy": float(ba.mean()),
                "perm_std_balanced_accuracy": float(ba.std(ddof=1)) if len(ba) > 1 else np.nan,
                "perm_p_ge_actual": float((ba >= actual_ba).mean()) if np.isfinite(actual_ba) else np.nan,
                "n_permutations": int(len(g)),
            }
        )

    return pd.DataFrame(rows)


def confusion_rows(
    y_true: pd.Series,
    y_pred: np.ndarray,
    feature_set: str,
    model: str,
) -> pd.DataFrame:
    labels = sorted(y_true.astype(str).unique())
    cm = confusion_matrix(y_true.astype(str), y_pred, labels=labels)

    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(cm[i, j]),
                }
            )
    return pd.DataFrame(rows)


def feature_importance_rows(
    fitted: Pipeline,
    feature_set: str,
    model_name: str,
    features: list[str],
) -> pd.DataFrame:
    model = fitted.named_steps["model"]
    rows = []

    if hasattr(model, "feature_importances_"):
        for feature, value in zip(features, model.feature_importances_):
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "class": "",
                    "feature": feature,
                    "importance_type": "feature_importance",
                    "value": float(value),
                    "abs_value": float(abs(value)),
                }
            )

    elif hasattr(model, "coef_"):
        coef = model.coef_
        classes = list(getattr(model, "classes_", []))

        for class_idx, cls in enumerate(classes):
            vals = coef[class_idx] if coef.ndim > 1 else coef
            for feature, value in zip(features, vals):
                rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "class": cls,
                        "feature": feature,
                        "importance_type": "coefficient",
                        "value": float(value),
                        "abs_value": float(abs(value)),
                    }
                )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Input manifest / report
# -----------------------------------------------------------------------------

def write_input_manifest(
    outdir: Path,
    feature_table_path: Path,
    feature_manifest_path: Path,
    table: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "artifact": "feature_table",
            "path": str(feature_table_path),
            "status": "ok" if feature_table_path.exists() else "missing",
            "rows": int(len(table)),
            "cols": int(len(table.columns)),
        },
        {
            "artifact": "feature_manifest",
            "path": str(feature_manifest_path),
            "status": "ok" if feature_manifest_path.exists() else "missing",
            "rows": int(len(manifest)),
            "cols": int(len(manifest.columns)),
        },
    ]

    df = pd.DataFrame(rows)
    write_csv(df, outdir / "obs078b_input_manifest.csv")
    return df


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    table: pd.DataFrame,
    feature_sets_df: pd.DataFrame,
    panel_scores: pd.DataFrame,
    perm_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    min_ba_threshold: float,
    max_features_threshold: int,
) -> None:
    case_counts = table["case"].value_counts().rename_axis("case").reset_index(name="n_rows")

    scores_sorted = panel_scores.sort_values(
        ["balanced_accuracy", "macro_f1", "accuracy"],
        ascending=False,
        na_position="last",
    )

    compact_candidates = scores_sorted[
        (scores_sorted["n_features"] <= max_features_threshold)
        & (scores_sorted["balanced_accuracy"] >= min_ba_threshold)
        & (scores_sorted["model"] != "dummy")
    ].sort_values(["n_features", "balanced_accuracy"], ascending=[True, False])

    best_by_set = (
        panel_scores[panel_scores["model"] != "dummy"]
        .sort_values(["feature_set", "balanced_accuracy"], ascending=[True, False])
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values(["n_features", "balanced_accuracy"], ascending=[True, False])
    )

    top_imp = pd.DataFrame()
    if not feature_importance.empty:
        top_imp = feature_importance.sort_values("abs_value", ascending=False).head(40)

    lines = []
    lines.append("# OBS-078b — Minimal Signature Ablation")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-078b asks what the smallest strict OBS-077-derived feature subset is "
        "that still recovers C / Cp2 / Cp3 above permutation baseline."
    )
    lines.append("")
    lines.append("Unlike OBS-078a, this script does not rebuild OBS-077 artifacts. It consumes the validated OBS-078a v2 feature table and manifest.")
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest))
    lines.append("")
    lines.append("## Feature table")
    lines.append("")
    lines.append(f"Rows: `{len(table)}`")
    lines.append("")
    lines.append(f"Columns: `{len(table.columns)}`")
    lines.append("")
    lines.append("### Rows by case")
    lines.append("")
    lines.append(markdown_table(case_counts))
    lines.append("")
    lines.append("## Feature sets")
    lines.append("")
    fs_summary = (
        feature_sets_df.groupby("feature_set", as_index=False)
        .agg(n_features=("feature", "count"))
        .sort_values(["n_features", "feature_set"])
    )
    lines.append(markdown_table(fs_summary, max_rows=80))
    lines.append("")
    lines.append("## Best score by feature set")
    lines.append("")
    lines.append(markdown_table(best_by_set[[
        "feature_set",
        "model",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]], max_rows=80))
    lines.append("")
    lines.append("## Compact candidates")
    lines.append("")
    lines.append(
        f"Criteria: `n_features <= {max_features_threshold}` and `balanced_accuracy >= {min_ba_threshold}`."
    )
    lines.append("")
    if compact_candidates.empty:
        lines.append("_No compact candidates met the configured criteria._")
    else:
        lines.append(markdown_table(compact_candidates[[
            "feature_set",
            "model",
            "n_features",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]], max_rows=80))
    lines.append("")
    lines.append("## Permutation summary")
    lines.append("")
    if perm_summary.empty:
        lines.append("_No permutation summary._")
    else:
        ps = perm_summary.sort_values("actual_balanced_accuracy", ascending=False)
        lines.append(markdown_table(ps, max_rows=80))
    lines.append("")
    lines.append("## Top feature importances / coefficients")
    lines.append("")
    if top_imp.empty:
        lines.append("_No feature importance rows._")
    else:
        lines.append(markdown_table(top_imp[[
            "feature_set",
            "model",
            "class",
            "feature",
            "importance_type",
            "value",
            "abs_value",
        ]], max_rows=40))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("A strong OBS-078b result is:")
    lines.append("")
    lines.append("```text")
    lines.append("a small strict feature set, ideally <= 10 features,")
    lines.append("with balanced_accuracy >= 0.80 and permutation baseline near 0.33–0.34")
    lines.append("```")
    lines.append("")
    lines.append("Candidate canonical reads:")
    lines.append("")
    lines.append("```text")
    lines.append("window_means_only succeeds:")
    lines.append("  boundedness / local divergence alone carries the signature")
    lines.append("")
    lines.append("minimal_obs077_candidate succeeds:")
    lines.append("  a compact mix of boundedness, divergence, path composition,")
    lines.append("  and transition geometry is sufficient")
    lines.append("")
    lines.append("no_path_labels succeeds:")
    lines.append("  path labels are not required once geometry and windows are present")
    lines.append("")
    lines.append("no_windows succeeds:")
    lines.append("  scale-transition geometry and path composition are sufficient")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-078b is a separability/minimality diagnostic, not causal proof.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs078b_input_manifest.csv")
    lines.append("obs078b_feature_sets.csv")
    lines.append("obs078b_panel_scores.csv")
    lines.append("obs078b_permutation_scores.csv")
    lines.append("obs078b_permutation_summary.csv")
    lines.append("obs078b_feature_importance.csv")
    lines.append("obs078b_confusion_matrices.csv")
    lines.append("obs078b_minimal_signature_summary.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-078b")

    (outdir / "obs078b_minimal_signature_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-078b minimal signature ablation.")
    ap.add_argument("--feature-table", required=True)
    ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cv", default="auto", choices=["auto", "stratified", "loo"])
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument(
        "--models",
        default="logreg,tree_depth2,rf_depth2,dummy",
        help="Comma-separated models.",
    )
    ap.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    ap.add_argument(
        "--min-ba-threshold",
        type=float,
        default=0.80,
        help="Compact candidate balanced-accuracy threshold.",
    )
    ap.add_argument(
        "--max-features-threshold",
        type=int,
        default=10,
        help="Compact candidate max feature count.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = ensure_outdir(args.outdir)

    feature_table_path = Path(args.feature_table)
    feature_manifest_path = Path(args.feature_manifest)

    if not feature_table_path.exists():
        raise FileNotFoundError(feature_table_path)
    if not feature_manifest_path.exists():
        raise FileNotFoundError(feature_manifest_path)

    table = pd.read_csv(feature_table_path)
    manifest = normalize_manifest(pd.read_csv(feature_manifest_path))

    if "case" not in table.columns:
        raise ValueError("Feature table missing target column: case")

    input_manifest = write_input_manifest(
        outdir=outdir,
        feature_table_path=feature_table_path,
        feature_manifest_path=feature_manifest_path,
        table=table,
        manifest=manifest,
    )

    feature_sets = build_feature_sets(table, manifest)
    if not feature_sets:
        raise RuntimeError("No non-empty feature sets were constructed.")

    feature_sets_df = write_feature_sets(outdir, feature_sets)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    score_rows = []
    perm_parts = []
    confusion_parts = []
    importance_parts = []

    y_true = table["case"].astype(str)

    run_index = 0

    for feature_set, features in feature_sets.items():
        for model_name in models:
            run_index += 1
            try:
                result, y_pred, fitted = evaluate_feature_set(
                    table=table,
                    feature_set=feature_set,
                    features=features,
                    model_name=model_name,
                    cv_mode=args.cv,
                )
                score_rows.append(result)

                confusion_parts.append(
                    confusion_rows(
                        y_true=y_true,
                        y_pred=y_pred,
                        feature_set=feature_set,
                        model=model_name,
                    )
                )

                if model_name != "dummy":
                    imp = feature_importance_rows(
                        fitted=fitted,
                        feature_set=feature_set,
                        model_name=model_name,
                        features=features,
                    )
                    if not imp.empty:
                        importance_parts.append(imp)

                if args.n_permutations > 0 and model_name != "dummy":
                    perm = permutation_scores(
                        table=table,
                        feature_set=feature_set,
                        features=features,
                        model_name=model_name,
                        cv_mode=args.cv,
                        n_permutations=args.n_permutations,
                        seed=args.seed + run_index,
                    )
                    if not perm.empty:
                        perm_parts.append(perm)

            except Exception as e:
                score_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "cv_mode": args.cv,
                        "n_rows": int(len(table)),
                        "n_features": int(len(features)),
                        "features": ",".join(features),
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "macro_f1": np.nan,
                        "error": str(e),
                    }
                )

    panel_scores = pd.DataFrame(score_rows)
    panel_scores = panel_scores.sort_values(
        ["balanced_accuracy", "macro_f1", "accuracy"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    write_csv(panel_scores, outdir / "obs078b_panel_scores.csv")

    perm_df = pd.concat(perm_parts, ignore_index=True) if perm_parts else pd.DataFrame()
    write_csv(perm_df, outdir / "obs078b_permutation_scores.csv")

    perm_summary = summarize_permutations(panel_scores, perm_df)
    if not perm_summary.empty:
        perm_summary = perm_summary.sort_values(
            "actual_balanced_accuracy",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    write_csv(perm_summary, outdir / "obs078b_permutation_summary.csv")

    confusion_df = pd.concat(confusion_parts, ignore_index=True) if confusion_parts else pd.DataFrame()
    write_csv(confusion_df, outdir / "obs078b_confusion_matrices.csv")

    importance_df = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    if not importance_df.empty:
        importance_df = importance_df.sort_values(
            ["feature_set", "model", "abs_value"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    write_csv(importance_df, outdir / "obs078b_feature_importance.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        table=table,
        feature_sets_df=feature_sets_df,
        panel_scores=panel_scores,
        perm_summary=perm_summary,
        feature_importance=importance_df,
        min_ba_threshold=args.min_ba_threshold,
        max_features_threshold=args.max_features_threshold,
    )

    print(f"[OBS-078b] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
