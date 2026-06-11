#!/usr/bin/env python3
"""
obs078a_mechanistic_signature_classifier.py

OBS-078a — Mechanistic Signature Classifier, v2.

Purpose
-------
Test whether the C / Cp2 / Cp3 distinction can be recovered from only
OBS-077-derived mechanistic features:

  - OBS-077a pinch transition geometry
  - OBS-077b / OBS-077c path-label cohort enrichment
  - OBS-077c window-local divergence / boundedness contrasts

v2 patch
--------
The v1 run showed strong separability, but also exposed leakage-prone features:

  - *_global_path_share
  - *_global_share
  - *_denom_paths
  - *_n_paths
  - n_paths
  - n_windows
  - candidate_rank
  - scale_index_from
  - scale_index_to
  - transition_delta

These can encode corpus-level base rates, support-size artifacts, or transition
identity. v2 adds strict feature filtering.

v2 also fixes logistic regression for multiclass classification.

Core question
-------------
Can C / Cp2 / Cp3 be separated using only derived mechanism features?

Inputs
------
Each --case points at a corpus pipeline directory, for example:

  C=outputs/corpora/C/campaigns/canonical_legacy/pipeline
  Cp2=outputs/corpora/Cp2/campaigns/full_v2/pipeline
  Cp3=outputs/corpora/Cp3/campaigns/full_v1/pipeline

Expected artifacts under each pipeline dir:

  obs077a_pinch_point_geometry_shared14_mds_pilot_v2/
    obs077a_pinch_point_candidates.csv

  obs077b_path_label_projection_shared14_mds_pilot/
    obs077b_pinch_label_projection.csv

  obs077c_window_coupling_bridge_shared14_mds_pilot_v2/
    obs077c_pinch_cohort_numeric_contrast.csv
    obs077c_pinch_cohort_seam_band_enrichment.csv
    obs077c_pinch_cohort_path_family_enrichment.csv
    obs077c_pinch_cohort_outcome_group_enrichment.csv

Outputs
-------
  obs078a_input_manifest.csv
  obs078a_feature_manifest.csv
  obs078a_feature_table.csv
  obs078a_panel_scores.csv
  obs078a_permutation_scores.csv
  obs078a_permutation_summary.csv
  obs078a_feature_importance.csv
  obs078a_confusion_matrices.csv
  obs078a_report.md

Scientific guardrail
--------------------
This is a separability diagnostic, not causal proof.

No raw path IDs, node IDs, raw observables, embeddings, or file paths are used
as model features.

v2 strict mode further excludes corpus-global base-rate/count/rank/scale-identity
features.
"""

from __future__ import annotations

import argparse
import random
import re
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_RANDOM_STATE = 77078
EPS = 1e-12


# -----------------------------------------------------------------------------
# Expected directories/files
# -----------------------------------------------------------------------------

PINCH_DIR = "obs077a_pinch_point_geometry_shared14_mds_pilot_v2"
PINCH_FILE = "obs077a_pinch_point_candidates.csv"

PROJ_DIR = "obs077b_path_label_projection_shared14_mds_pilot"
PROJ_FILE = "obs077b_pinch_label_projection.csv"

WINDOW_DIR = "obs077c_window_coupling_bridge_shared14_mds_pilot_v2"
NUMERIC_FILE = "obs077c_pinch_cohort_numeric_contrast.csv"
SEAM_FILE = "obs077c_pinch_cohort_seam_band_enrichment.csv"
PATH_FAMILY_FILE = "obs077c_pinch_cohort_path_family_enrichment.csv"
OUTCOME_FILE = "obs077c_pinch_cohort_outcome_group_enrichment.csv"


KEY_COLS = [
    "case",
    "candidate_rank",
    "object",
    "scale_index_from",
    "scale_index_to",
    "cohort",
]

PINCH_KEYS = [
    "case",
    "candidate_rank",
    "object",
    "scale_index_from",
    "scale_index_to",
]

PINCH_NUMERIC_FEATURES = [
    "pinch_score_total",
    "support_score",
    "overlap_score",
    "shape_score",
    "id_score",
    "jaccard_prev",
    "jaccard_loss",
    "centroid_drift",
    "support_size_delta",
    "overlap_delta_max",
    "participation_ratio_delta_abs",
    "anisotropy_delta_abs",
    "log_volume_delta_abs",
    "twonn_id_delta_abs",
    "local_mle_id_delta_abs",
]

WINDOW_NUMERIC_FEATURES = [
    "n_paths",
    "n_windows",
    "mean_lambda_local_mean",
    "mean_lambda_local_z",
    "mean_delta_d_mean",
    "mean_delta_d_z",
    "bounded_share_mean",
    "bounded_share_z",
    "divergence_z_sum",
]

IDENTITY_CATEGORICAL_FEATURES = [
    "object",
    "cohort",
    "dominant_family",
    "dominant_reason",
]

IDENTITY_NUMERIC_FEATURES = [
    "candidate_rank",
    "scale_index_from",
    "scale_index_to",
    "transition_delta",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def parse_case_arg(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise ValueError(f"--case must be NAME=PATH, got: {s}")
    name, path = s.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty case name in --case {s}")
    return name, Path(path)


def safe_string(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"nan": "NA", "None": "NA"}).fillna("NA")


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["candidate_rank", "scale_index_from", "scale_index_to"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in ["object", "cohort", "dominant_family", "dominant_reason"]:
        if col in out.columns:
            out[col] = safe_string(out[col])

    return out


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def safe_label(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_") or "NA"


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            cols.append(c)
        else:
            seen[c] += 1
            cols.append(f"{c}__dup{seen[c]}")
    out = df.copy()
    out.columns = cols
    return out


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


# -----------------------------------------------------------------------------
# Artifact loading
# -----------------------------------------------------------------------------

def artifact_paths(base: Path) -> dict[str, Path]:
    return {
        "pinch": base / PINCH_DIR / PINCH_FILE,
        "projection": base / PROJ_DIR / PROJ_FILE,
        "numeric": base / WINDOW_DIR / NUMERIC_FILE,
        "seam_band": base / WINDOW_DIR / SEAM_FILE,
        "path_family": base / WINDOW_DIR / PATH_FAMILY_FILE,
        "outcome_group": base / WINDOW_DIR / OUTCOME_FILE,
    }


def load_case_artifacts(case: str, base: Path) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in artifact_paths(base).items():
        df = read_csv_if_exists(path)
        if not df.empty:
            df = normalize_keys(df)
            df["case"] = case
        loaded[name] = df
    return loaded


def write_input_manifest(
    outdir: Path,
    cases: dict[str, Path],
    loaded: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for case, base in cases.items():
        for name, path in artifact_paths(base).items():
            df = loaded.get(case, {}).get(name, pd.DataFrame())
            rows.append(
                {
                    "case": case,
                    "artifact": name,
                    "status": "ok" if path.exists() else "missing",
                    "rows": int(len(df)),
                    "cols": int(len(df.columns)) if not df.empty else 0,
                    "path": str(path),
                }
            )
    out = pd.DataFrame(rows)
    write_csv(out, outdir / "obs078a_input_manifest.csv")
    return out


# -----------------------------------------------------------------------------
# Feature preparation
# -----------------------------------------------------------------------------

def prepare_pinch_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = normalize_keys(df)

    # OBS-077a v2 may not persist candidate_rank. Reconstruct from score order.
    if "candidate_rank" not in out.columns:
        if "pinch_score_total" in out.columns:
            out = out.sort_values("pinch_score_total", ascending=False).reset_index(drop=True)
        else:
            out = out.reset_index(drop=True)
        out["candidate_rank"] = np.arange(1, len(out) + 1)

    out["candidate_rank"] = pd.to_numeric(out["candidate_rank"], errors="coerce").astype("Int64")

    required = {"case", "candidate_rank", "object", "scale_index_from", "scale_index_to"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Pinch file missing columns: {sorted(missing)}")

    out["transition_delta"] = (
        pd.to_numeric(out["scale_index_to"], errors="coerce")
        - pd.to_numeric(out["scale_index_from"], errors="coerce")
    )

    for col in ["dominant_family", "dominant_reason"]:
        if col not in out.columns:
            out[col] = "NA"

    keep = (
        PINCH_KEYS
        + ["transition_delta", "dominant_family", "dominant_reason"]
        + [c for c in PINCH_NUMERIC_FEATURES if c in out.columns]
    )

    out = out[keep].drop_duplicates(subset=PINCH_KEYS)
    out = safe_numeric(out, [c for c in PINCH_NUMERIC_FEATURES + ["transition_delta"] if c in out.columns])
    return out


def prepare_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = normalize_keys(df)
    required = set(KEY_COLS)
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Numeric contrast file missing columns: {sorted(missing)}")

    if "dominant_family" not in out.columns:
        out["dominant_family"] = "NA"

    keep = KEY_COLS + ["dominant_family"] + [c for c in WINDOW_NUMERIC_FEATURES if c in out.columns]
    out = out[keep].drop_duplicates(subset=KEY_COLS)
    out = safe_numeric(out, [c for c in WINDOW_NUMERIC_FEATURES if c in out.columns])
    return out


def enrichment_pivot(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Convert a categorical enrichment table to wide features keyed by KEY_COLS.

    Expected input columns include:
      label_value
      path_enrichment
      path_log2_enrichment
      path_share
      global_path_share
      n_paths
      denom_paths

    v2 keeps all features in the raw feature table but marks leakage-prone
    features in the manifest and excludes them from strict panels.
    """
    if df.empty:
        return pd.DataFrame()

    out = normalize_keys(df)

    required = set(KEY_COLS + ["label_value"])
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Enrichment table missing columns: {sorted(missing)}")

    metric_cols = [
        c
        for c in [
            "path_enrichment",
            "path_log2_enrichment",
            "path_share",
            "global_path_share",
            "n_paths",
            "denom_paths",
        ]
        if c in out.columns
    ]

    if not metric_cols:
        return out[KEY_COLS].drop_duplicates()

    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["label_value_safe"] = out["label_value"].map(safe_label)

    wide_parts = []
    for metric in metric_cols:
        piv = out.pivot_table(
            index=KEY_COLS,
            columns="label_value_safe",
            values=metric,
            aggfunc="max",
        ).reset_index()

        piv.columns = [
            c if c in KEY_COLS else f"{prefix}__{c}__{metric}"
            for c in piv.columns
        ]
        wide_parts.append(piv)

    merged = wide_parts[0]
    for part in wide_parts[1:]:
        merged = merged.merge(part, on=KEY_COLS, how="outer")

    return merged


def prepare_projection_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional OBS-077b projection artifact.

    This table often has columns:
      label_col, label_value, n_paths, global_share, enrichment, log2_enrichment

    It is useful for non-strict panels, but v2 strict panels remove global/count
    features and keep enrichment/log2_enrichment-like quantities only.
    """
    if df.empty:
        return pd.DataFrame()

    out = normalize_keys(df)

    if not set(KEY_COLS).issubset(out.columns):
        return pd.DataFrame()

    label_col = None
    for cand in ["label_col", "label", "category"]:
        if cand in out.columns:
            label_col = cand
            break

    value_col = None
    for cand in ["label_value", "value", "category_value"]:
        if cand in out.columns:
            value_col = cand
            break

    if label_col is None or value_col is None:
        return pd.DataFrame()

    metric_cols = [
        c
        for c in [
            "n_paths",
            "denom_paths",
            "path_share",
            "global_share",
            "object_share",
            "global_path_share",
            "path_enrichment",
            "enrichment",
            "path_log2_enrichment",
            "log2_enrichment",
        ]
        if c in out.columns
    ]

    if not metric_cols:
        return pd.DataFrame()

    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["feature_name"] = (
        "proj__"
        + out[label_col].map(safe_label)
        + "__"
        + out[value_col].map(safe_label)
    )

    wide_parts = []
    for metric in metric_cols:
        piv = out.pivot_table(
            index=KEY_COLS,
            columns="feature_name",
            values=metric,
            aggfunc="max",
        ).reset_index()

        piv.columns = [
            c if c in KEY_COLS else f"{c}__{metric}"
            for c in piv.columns
        ]
        wide_parts.append(piv)

    merged = wide_parts[0]
    for part in wide_parts[1:]:
        merged = merged.merge(part, on=KEY_COLS, how="outer")

    return merged


def build_case_feature_table(case: str, artifacts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pinch = prepare_pinch_features(artifacts.get("pinch", pd.DataFrame()))
    numeric = prepare_numeric_features(artifacts.get("numeric", pd.DataFrame()))

    if numeric.empty:
        return pd.DataFrame()

    table = numeric.copy()

    if not pinch.empty:
        # Prefer exact join with dominant_family when available.
        exact_keys = PINCH_KEYS + ["dominant_family"]
        if set(exact_keys).issubset(table.columns) and set(exact_keys).issubset(pinch.columns):
            table = table.merge(pinch, on=exact_keys, how="left", suffixes=("", "_pinch"))
        else:
            table = table.merge(pinch, on=PINCH_KEYS, how="left", suffixes=("", "_pinch"))

        # Fallback softer join if dominant_family mismatch left geometry empty.
        if "pinch_score_total" in table.columns and table["pinch_score_total"].isna().all():
            drop_cols = [
                c for c in pinch.columns
                if c in table.columns and c not in PINCH_KEYS
            ]
            table = table.drop(columns=drop_cols, errors="ignore")
            table = table.merge(
                pinch.drop(columns=["dominant_family"], errors="ignore").drop_duplicates(subset=PINCH_KEYS),
                on=PINCH_KEYS,
                how="left",
            )

    for artifact, prefix in [
        ("seam_band", "seam"),
        ("path_family", "path_family"),
        ("outcome_group", "outcome"),
    ]:
        wide = enrichment_pivot(artifacts.get(artifact, pd.DataFrame()), prefix=prefix)
        if not wide.empty:
            table = table.merge(wide, on=KEY_COLS, how="left")

    proj = prepare_projection_features(artifacts.get("projection", pd.DataFrame()))
    if not proj.empty:
        table = table.merge(proj, on=KEY_COLS, how="left")

    if "transition_delta" not in table.columns:
        table["transition_delta"] = (
            pd.to_numeric(table["scale_index_to"], errors="coerce")
            - pd.to_numeric(table["scale_index_from"], errors="coerce")
        )

    table = dedupe_columns(table)
    return table


def build_feature_table(loaded: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    pieces = []
    for case, artifacts in loaded.items():
        ft = build_case_feature_table(case, artifacts)
        if not ft.empty:
            pieces.append(ft)

    if not pieces:
        return pd.DataFrame()

    table = pd.concat(pieces, ignore_index=True, sort=False)
    table = normalize_keys(table)

    forbidden_raw = [c for c in table.columns if c.lower() in {"path_id", "node_id"}]
    table = table.drop(columns=forbidden_raw, errors="ignore")

    return table


# -----------------------------------------------------------------------------
# Feature manifest / leakage filter
# -----------------------------------------------------------------------------

def feature_family(col: str, dtype: Any) -> tuple[str, str]:
    if col == "case":
        return "target", "target"

    if col in KEY_COLS:
        return "key", "identifier"

    if col in IDENTITY_CATEGORICAL_FEATURES:
        return "identity", "categorical"

    if col in IDENTITY_NUMERIC_FEATURES:
        return "identity", "numeric"

    if col in PINCH_NUMERIC_FEATURES:
        return "geometry", "numeric"

    if col in WINDOW_NUMERIC_FEATURES:
        return "window_divergence", "numeric"

    if col.startswith("seam__"):
        return "path_labels", "numeric"

    if col.startswith("path_family__"):
        return "path_labels", "numeric"

    if col.startswith("outcome__"):
        return "path_labels", "numeric"

    if col.startswith("proj__"):
        return "path_labels", "numeric"

    if pd.api.types.is_numeric_dtype(dtype):
        return "other_numeric", "numeric"

    return "other_categorical", "categorical"


def is_strict_leakage_feature(col: str) -> bool:
    """
    Strict OBS-078a v2 exclusion rules.

    Remove:
      - global/base-rate features
      - counts/denominators
      - candidate rank and scale identity
      - object/cohort/dominant categorical identity in object_blind mode elsewhere
    """
    if col in {"n_paths", "n_windows"}:
        return True

    if col in {"candidate_rank", "scale_index_from", "scale_index_to", "transition_delta"}:
        return True

    leakage_suffixes = [
        "__global_path_share",
        "__global_share",
        "__denom_paths",
        "__n_paths",
        "_global_path_share",
        "_global_share",
        "_denom_paths",
        "_n_paths",
    ]

    if any(col.endswith(suf) for suf in leakage_suffixes):
        return True

    if "__global_path_share" in col:
        return True
    if "__global_share" in col:
        return True
    if "__denom_paths" in col:
        return True
    if "__n_paths" in col:
        return True

    return False


def write_feature_manifest(outdir: Path, table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in table.columns:
        family, role = feature_family(col, table[col].dtype)
        rows.append(
            {
                "feature": col,
                "family": family,
                "role": role,
                "strict_excluded": bool(is_strict_leakage_feature(col)),
                "n_non_null": int(table[col].notna().sum()),
                "n_unique": int(table[col].nunique(dropna=True)),
                "dtype": str(table[col].dtype),
            }
        )

    manifest = pd.DataFrame(rows)
    write_csv(manifest, outdir / "obs078a_feature_manifest.csv")
    return manifest


# -----------------------------------------------------------------------------
# Panels / models
# -----------------------------------------------------------------------------

def get_feature_cols(
    feature_manifest: pd.DataFrame,
    panel: str,
    object_mode: str,
    strict: bool,
) -> tuple[list[str], list[str]]:
    if panel == "geometry_only":
        families = ["geometry"]
    elif panel == "path_labels_only":
        families = ["path_labels"]
    elif panel == "window_divergence_only":
        families = ["window_divergence"]
    elif panel == "geometry_plus_paths":
        families = ["geometry", "path_labels"]
    elif panel == "paths_plus_windows":
        families = ["path_labels", "window_divergence"]
    elif panel == "full_obs077_signature":
        families = ["geometry", "path_labels", "window_divergence"]
    else:
        raise ValueError(f"Unknown panel: {panel}")

    fm = feature_manifest.copy()
    fm = fm[fm["family"].isin(families)]

    if strict:
        fm = fm[~fm["strict_excluded"]]

    numeric_cols = list(fm[fm["role"] == "numeric"]["feature"])
    categorical_cols: list[str] = []

    if object_mode == "object_aware":
        ident = feature_manifest[
            (feature_manifest["family"] == "identity")
            & (feature_manifest["role"] == "categorical")
        ].copy()

        if strict:
            # Strict object-aware may still include object/cohort/dominant labels.
            # It does not include transition rank/scale numeric identity.
            pass

        categorical_cols = list(ident["feature"])

        # Non-strict object-aware gets transition numeric identity too.
        if not strict:
            for col in IDENTITY_NUMERIC_FEATURES:
                if col in list(feature_manifest["feature"]):
                    numeric_cols.append(col)

    elif object_mode == "object_blind":
        categorical_cols = []
    else:
        raise ValueError(f"Unknown object_mode: {object_mode}")

    numeric_cols = list(dict.fromkeys([c for c in numeric_cols if c != "case"]))
    categorical_cols = list(dict.fromkeys([c for c in categorical_cols if c != "case"]))

    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No features selected.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


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


def make_pipeline(model_name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
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


def evaluate_panel(
    table: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    panel: str,
    object_mode: str,
    strict: bool,
    model_name: str,
    cv_mode: str,
) -> tuple[dict[str, Any], np.ndarray, Pipeline, list[str], list[str]]:
    numeric_cols, categorical_cols = get_feature_cols(
        feature_manifest=feature_manifest,
        panel=panel,
        object_mode=object_mode,
        strict=strict,
    )

    if not numeric_cols and not categorical_cols:
        raise ValueError("No features selected.")

    X = table[numeric_cols + categorical_cols].copy()
    y = table["case"].astype(str)

    pipe = make_pipeline(model_name, numeric_cols, categorical_cols)
    cv = choose_cv(y, cv_mode)

    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    fitted = make_pipeline(model_name, numeric_cols, categorical_cols)
    fitted.fit(X, y)

    scores = score_predictions(y, y_pred)

    result = {
        "panel": panel,
        "object_mode": object_mode,
        "strict": bool(strict),
        "model": model_name,
        "cv_mode": cv_mode,
        "n_rows": int(len(table)),
        "n_features_raw": int(len(numeric_cols) + len(categorical_cols)),
        "numeric_cols": ",".join(numeric_cols),
        "categorical_cols": ",".join(categorical_cols),
        **scores,
    }

    return result, y_pred, fitted, numeric_cols, categorical_cols


def permutation_scores(
    table: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    panel: str,
    object_mode: str,
    strict: bool,
    model_name: str,
    cv_mode: str,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric_cols, categorical_cols = get_feature_cols(
        feature_manifest=feature_manifest,
        panel=panel,
        object_mode=object_mode,
        strict=strict,
    )

    X = table[numeric_cols + categorical_cols].copy()
    y0 = table["case"].astype(str).to_numpy()

    rows = []
    for i in range(n_permutations):
        y_perm = pd.Series(rng.permutation(y0))
        pipe = make_pipeline(model_name, numeric_cols, categorical_cols)
        cv = choose_cv(y_perm, cv_mode)
        y_pred = cross_val_predict(pipe, X, y_perm, cv=cv)
        scores = score_predictions(y_perm, y_pred)
        rows.append(
            {
                "panel": panel,
                "object_mode": object_mode,
                "strict": bool(strict),
                "model": model_name,
                "permutation_index": i,
                **scores,
            }
        )

    return pd.DataFrame(rows)


def confusion_rows(
    y_true: pd.Series,
    y_pred: np.ndarray,
    panel: str,
    object_mode: str,
    strict: bool,
    model: str,
) -> pd.DataFrame:
    labels = sorted(y_true.astype(str).unique())
    cm = confusion_matrix(y_true.astype(str), y_pred, labels=labels)

    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            rows.append(
                {
                    "panel": panel,
                    "object_mode": object_mode,
                    "strict": bool(strict),
                    "model": model,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(cm[i, j]),
                }
            )

    return pd.DataFrame(rows)


def transformed_feature_names(
    pipe: Pipeline,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    names: list[str] = []
    if numeric_cols:
        names.extend(numeric_cols)

    if categorical_cols:
        try:
            cat_pipe = pipe.named_steps["preprocess"].named_transformers_["cat"]
            ohe = cat_pipe.named_steps["onehot"]
            names.extend(list(ohe.get_feature_names_out(categorical_cols)))
        except Exception:
            names.extend(categorical_cols)

    return names


def feature_importance_rows(
    pipe: Pipeline,
    panel: str,
    object_mode: str,
    strict: bool,
    model_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    names = transformed_feature_names(pipe, numeric_cols, categorical_cols)

    rows = []

    if hasattr(model, "feature_importances_"):
        for name, value in zip(names, model.feature_importances_):
            rows.append(
                {
                    "panel": panel,
                    "object_mode": object_mode,
                    "strict": bool(strict),
                    "model": model_name,
                    "class": "",
                    "feature": name,
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
            for name, value in zip(names, vals):
                rows.append(
                    {
                        "panel": panel,
                        "object_mode": object_mode,
                        "strict": bool(strict),
                        "model": model_name,
                        "class": cls,
                        "feature": name,
                        "importance_type": "coefficient",
                        "value": float(value),
                        "abs_value": float(abs(value)),
                    }
                )

    return pd.DataFrame(rows)


def summarize_permutations(panel_scores: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    if perm.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["panel", "object_mode", "strict", "model"]

    for keys, g in perm.groupby(group_cols):
        panel, object_mode, strict, model = keys
        actual = panel_scores[
            (panel_scores["panel"] == panel)
            & (panel_scores["object_mode"] == object_mode)
            & (panel_scores["strict"] == strict)
            & (panel_scores["model"] == model)
        ]

        actual_ba = float(actual["balanced_accuracy"].iloc[0]) if not actual.empty else np.nan
        ba = pd.to_numeric(g["balanced_accuracy"], errors="coerce")
        p_ge = float((ba >= actual_ba).mean()) if np.isfinite(actual_ba) else np.nan

        rows.append(
            {
                "panel": panel,
                "object_mode": object_mode,
                "strict": bool(strict),
                "model": model,
                "actual_balanced_accuracy": actual_ba,
                "perm_mean_balanced_accuracy": float(ba.mean()),
                "perm_std_balanced_accuracy": float(ba.std(ddof=1)) if len(ba) > 1 else np.nan,
                "perm_p_ge_actual": p_ge,
                "n_permutations": int(len(g)),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(
    outdir: Path,
    feature_table: pd.DataFrame,
    input_manifest: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    panel_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    scores_sorted = panel_scores.sort_values(
        ["strict", "balanced_accuracy", "macro_f1"],
        ascending=[False, False, False],
        na_position="last",
    )

    strict_scores = panel_scores[panel_scores["strict"] == True].sort_values(
        ["balanced_accuracy", "macro_f1"],
        ascending=False,
        na_position="last",
    )

    nonstrict_scores = panel_scores[panel_scores["strict"] == False].sort_values(
        ["balanced_accuracy", "macro_f1"],
        ascending=False,
        na_position="last",
    )

    case_counts = feature_table["case"].value_counts().rename_axis("case").reset_index(name="n_rows")

    leak_counts = (
        feature_manifest.groupby(["family", "strict_excluded"])
        .size()
        .reset_index(name="n_features")
        .sort_values(["family", "strict_excluded"])
    )

    strict_perm = permutation_summary[permutation_summary["strict"] == True].sort_values(
        "actual_balanced_accuracy",
        ascending=False,
    ) if not permutation_summary.empty else pd.DataFrame()

    top_imp = pd.DataFrame()
    if not feature_importance.empty:
        top_imp = feature_importance.sort_values("abs_value", ascending=False).head(40)

    lines = []
    lines.append("# OBS-078a — Mechanistic Signature Classifier v2")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-078a tests whether C/Cp2/Cp3 can be recovered from OBS-077-derived "
        "mechanistic features."
    )
    lines.append("")
    lines.append("v2 adds strict anti-leakage controls.")
    lines.append("")
    lines.append("## Strict-mode exclusions")
    lines.append("")
    lines.append("Strict panels exclude:")
    lines.append("")
    lines.append("```text")
    lines.append("*_global_path_share")
    lines.append("*_global_share")
    lines.append("*_denom_paths")
    lines.append("*_n_paths")
    lines.append("n_paths")
    lines.append("n_windows")
    lines.append("candidate_rank")
    lines.append("scale_index_from")
    lines.append("scale_index_to")
    lines.append("transition_delta")
    lines.append("```")
    lines.append("")
    lines.append("Strict object-blind panels also exclude object/cohort/dominant categorical identity.")
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest, max_rows=30))
    lines.append("")
    lines.append("## Feature table")
    lines.append("")
    lines.append(f"Rows: `{len(feature_table)}`")
    lines.append("")
    lines.append(f"Columns: `{len(feature_table.columns)}`")
    lines.append("")
    lines.append("### Rows by case")
    lines.append("")
    lines.append(markdown_table(case_counts))
    lines.append("")
    lines.append("### Feature leakage audit")
    lines.append("")
    lines.append(markdown_table(leak_counts, max_rows=40))
    lines.append("")
    lines.append("## Strict panel scores")
    lines.append("")
    lines.append(markdown_table(strict_scores, max_rows=40))
    lines.append("")
    lines.append("## Non-strict panel scores")
    lines.append("")
    lines.append(markdown_table(nonstrict_scores, max_rows=30))
    lines.append("")
    lines.append("## Strict permutation summary")
    lines.append("")
    lines.append(markdown_table(strict_perm, max_rows=40))
    lines.append("")
    lines.append("## Top feature importances / coefficients")
    lines.append("")
    lines.append(markdown_table(top_imp, max_rows=40))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("Strong evidence:")
    lines.append("")
    lines.append("```text")
    lines.append("strict full_obs077_signature above permutation")
    lines.append("strict paths_plus_windows above permutation")
    lines.append("strict window_divergence_only object_blind above permutation")
    lines.append("strict geometry_only object_blind above permutation")
    lines.append("```")
    lines.append("")
    lines.append("Most important strict result:")
    lines.append("")
    lines.append("```text")
    lines.append("object_blind strict performance above permutation")
    lines.append("```")
    lines.append("")
    lines.append("because this means the separability is not merely named support identity or corpus-global base rates.")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("This remains a separability diagnostic, not causal proof.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs078a_input_manifest.csv")
    lines.append("obs078a_feature_manifest.csv")
    lines.append("obs078a_feature_table.csv")
    lines.append("obs078a_panel_scores.csv")
    lines.append("obs078a_permutation_scores.csv")
    lines.append("obs078a_permutation_summary.csv")
    lines.append("obs078a_feature_importance.csv")
    lines.append("obs078a_confusion_matrices.csv")
    lines.append("obs078a_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-078a v2")

    (outdir / "obs078a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="OBS-078a v2 mechanistic signature classifier."
    )
    ap.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case mapping NAME=PIPELINE_DIR. Repeat for C, Cp2, Cp3.",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--cv",
        default="auto",
        choices=["auto", "stratified", "loo"],
    )
    ap.add_argument(
        "--n-permutations",
        type=int,
        default=100,
    )
    ap.add_argument(
        "--models",
        default="logreg,tree_depth2,rf_depth2,dummy",
    )
    ap.add_argument(
        "--panels",
        default="geometry_only,path_labels_only,window_divergence_only,geometry_plus_paths,paths_plus_windows,full_obs077_signature",
    )
    ap.add_argument(
        "--object-modes",
        default="object_aware,object_blind",
    )
    ap.add_argument(
        "--strict-modes",
        default="true,false",
        help="Comma-separated strict modes: true,false or true only.",
    )
    ap.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    return ap.parse_args()


def parse_bool_list(s: str) -> list[bool]:
    vals = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p in {"true", "1", "yes", "strict"}:
            vals.append(True)
        elif p in {"false", "0", "no", "nonstrict"}:
            vals.append(False)
        else:
            raise ValueError(f"Invalid strict mode: {part}")
    return vals


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = ensure_outdir(args.outdir)

    cases: dict[str, Path] = {}
    for item in args.case:
        name, path = parse_case_arg(item)
        cases[name] = path

    loaded = {case: load_case_artifacts(case, base) for case, base in cases.items()}
    input_manifest = write_input_manifest(outdir, cases, loaded)

    feature_table = build_feature_table(loaded)
    if feature_table.empty:
        raise RuntimeError("Feature table is empty. Check input artifacts.")

    write_csv(feature_table, outdir / "obs078a_feature_table.csv")
    feature_manifest = write_feature_manifest(outdir, feature_table)

    panels = [x.strip() for x in args.panels.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    object_modes = [x.strip() for x in args.object_modes.split(",") if x.strip()]
    strict_modes = parse_bool_list(args.strict_modes)

    score_rows = []
    perm_parts = []
    confusion_parts = []
    importance_parts = []

    y_true = feature_table["case"].astype(str)

    run_index = 0

    for strict in strict_modes:
        for panel in panels:
            for object_mode in object_modes:
                for model_name in models:
                    run_index += 1
                    try:
                        result, y_pred, fitted, numeric_cols, categorical_cols = evaluate_panel(
                            table=feature_table,
                            feature_manifest=feature_manifest,
                            panel=panel,
                            object_mode=object_mode,
                            strict=strict,
                            model_name=model_name,
                            cv_mode=args.cv,
                        )

                        score_rows.append(result)

                        confusion_parts.append(
                            confusion_rows(
                                y_true=y_true,
                                y_pred=y_pred,
                                panel=panel,
                                object_mode=object_mode,
                                strict=strict,
                                model=model_name,
                            )
                        )

                        if model_name != "dummy":
                            imp = feature_importance_rows(
                                pipe=fitted,
                                panel=panel,
                                object_mode=object_mode,
                                strict=strict,
                                model_name=model_name,
                                numeric_cols=numeric_cols,
                                categorical_cols=categorical_cols,
                            )
                            if not imp.empty:
                                importance_parts.append(imp)

                        if args.n_permutations > 0 and model_name != "dummy":
                            perm = permutation_scores(
                                table=feature_table,
                                feature_manifest=feature_manifest,
                                panel=panel,
                                object_mode=object_mode,
                                strict=strict,
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
                                "panel": panel,
                                "object_mode": object_mode,
                                "strict": bool(strict),
                                "model": model_name,
                                "cv_mode": args.cv,
                                "n_rows": int(len(feature_table)),
                                "n_features_raw": 0,
                                "accuracy": np.nan,
                                "balanced_accuracy": np.nan,
                                "macro_f1": np.nan,
                                "error": str(e),
                                "numeric_cols": "",
                                "categorical_cols": "",
                            }
                        )

    panel_scores = pd.DataFrame(score_rows)
    panel_scores = panel_scores.sort_values(
        ["strict", "balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    write_csv(panel_scores, outdir / "obs078a_panel_scores.csv")

    permutation_df = pd.concat(perm_parts, ignore_index=True) if perm_parts else pd.DataFrame()
    write_csv(permutation_df, outdir / "obs078a_permutation_scores.csv")

    permutation_summary = summarize_permutations(panel_scores, permutation_df)
    permutation_summary = permutation_summary.sort_values(
        ["strict", "actual_balanced_accuracy"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True) if not permutation_summary.empty else permutation_summary
    write_csv(permutation_summary, outdir / "obs078a_permutation_summary.csv")

    confusion_df = pd.concat(confusion_parts, ignore_index=True) if confusion_parts else pd.DataFrame()
    write_csv(confusion_df, outdir / "obs078a_confusion_matrices.csv")

    importance_df = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    if not importance_df.empty:
        importance_df = importance_df.sort_values(
            ["strict", "panel", "object_mode", "model", "abs_value"],
            ascending=[False, True, True, True, False],
        ).reset_index(drop=True)
    write_csv(importance_df, outdir / "obs078a_feature_importance.csv")

    write_report(
        outdir=outdir,
        feature_table=feature_table,
        input_manifest=input_manifest,
        feature_manifest=feature_manifest,
        panel_scores=panel_scores,
        permutation_summary=permutation_summary,
        feature_importance=importance_df,
    )

    print(f"[OBS-078a v2] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
