#!/usr/bin/env python3
"""
OBS-075b — Cp3 endpoint / velocity ablation v2

Purpose
-------
Recompute Cp3 directional-transfer asymmetry after stricter feature blinding,
using already-produced OBS-073 feature tables as the primary input.

This v2 patch fixes the v1 problem where coupling / coupled-outcome / recovery
targets were unavailable because OBS-075b tried to reconstruct labels from
lower-level corpus-root artifacts. OBS-073 already contains the validated target
columns and feature table, so this script now consumes those tables directly.

Scientific question
-------------------
Does the Cp3 directional asymmetry seen in OBS-075 survive after removing:

1. direct seam / proximity features,
2. absolute grid / location features,
3. endpoint / velocity / path-length proxies,
4. tortuosity / turning / angle-jump proxies?

It compares:

- Cp2 ↔ Cp3
- Cp  ↔ Cp3
- C   ↔ Cp3

and reports specificity against both controls:

- specificity_vs_Cp = asym(Cp3,Cp2) - asym(Cp3,Cp)
- specificity_vs_C  = asym(Cp3,Cp2) - asym(Cp3,C)

Outputs
-------
- obs075b_model_scores.csv
- obs075b_asymmetry_specificity.csv
- obs075b_pair_asymmetry.csv
- obs075b_feature_manifest.csv
- obs075b_target_class_counts.csv
- obs075b_summary.md

Notes
-----
- This script does not recompute geometry, path families, coupling, outcomes,
  or recovery labels.
- It recomputes transfer models from existing OBS-073 feature tables.
- `insufficient_classes` means the target slice is unavailable or too small,
  not that the scientific effect collapsed.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


TARGET_COLS: dict[str, str] = {
    "path_family_no_direct_seam_no_grid": "target_path_family",
    "outcome_group_no_direct_seam_no_grid": "target_outcome_group",
    "coupling_class_no_direct_seam_no_grid": "target_coupling_class",
    "coupled_outcome_group_no_direct_seam_no_grid": "target_coupled_outcome_group",
    "recovery_channel_no_direct_seam_no_grid": "target_recovery_channel_structural",
    "recovery_channel_boundedness_strict": "target_recovery_channel_boundedness_strict",
}


FOCUS_ORDER: dict[str, int] = {
    "coupled_outcome_group_no_direct_seam_no_grid": 1,
    "recovery_channel_no_direct_seam_no_grid": 2,
    "recovery_channel_boundedness_strict": 3,
    "coupling_class_no_direct_seam_no_grid": 4,
    "outcome_group_no_direct_seam_no_grid": 5,
    "path_family_no_direct_seam_no_grid": 6,
}


FEATURE_SETS: tuple[str, ...] = (
    "no_direct_seam_no_grid",
    "no_direct_seam_no_grid_no_endpoint_velocity",
    "no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity",
    "holonomy_criticality_shape_only",
)


ALWAYS_EXCLUDE_COLS: set[str] = {
    "path_id",
    "path_family",
    "corpus",
    "source_root",
    "scale",
    "tortuosity_coordinate_basis",
}


ALWAYS_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "target_",
    "obs050_",
    "obs051_",
    "obs072_",
)


DIRECT_SEAM_PATTERNS: tuple[str, ...] = (
    "distance_to_seam",
    "near_fraction",
    "mid_fraction",
    "far_fraction",
    "seam",
)


GRID_LOCATION_PATTERNS: tuple[str, ...] = (
    "node_id",
    "pn_r_",
    "pn_alpha_",
    "pn_mds1_",
    "pn_mds2_",
)


ENDPOINT_VELOCITY_PATTERNS: tuple[str, ...] = (
    "last_minus_first",
    "net_displacement",
    "path_arclength",
    "path_chord_length",
    "pd_n_steps",
    "pn_n_steps",
    "pd_n_nodes",
)


TORTUOSITY_PATTERNS: tuple[str, ...] = (
    "tortuosity",
    "turning_",
    "angle_jump",
    "sector_change",
    "dew_angle",
)


HOLONOMY_CRITICALITY_SHAPE_KEEP_PATTERNS: tuple[str, ...] = (
    "criticality",
    "holonomy",
    "obstruction",
    "signed_phase",
    "path_sector_change",
    "n_sector_changes",
)


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    name: str
    table_path: Path
    baseline_corpus: str
    cp3_corpus: str = "Cp3"


@dataclass(frozen=True)
class RunConfig:
    cp2_cp3_feature_table: Path
    cp_cp3_feature_table: Path
    c_cp3_feature_table: Path | None
    outdir: Path
    model_family: str
    n_estimators: int
    max_depth: int | None
    random_state: int
    min_class_count: int
    warn_class_count: int
    targets: tuple[str, ...]
    feature_sets: tuple[str, ...]


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def parse_max_depth(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"", "none", "null", "nan"}:
        return None
    return int(s)


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def fmt4(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.4f}"
    except Exception:
        return "NA"


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "\n_No rows._\n"

    work = df.copy()
    if columns is not None:
        keep = [c for c in columns if c in work.columns]
        work = work[keep]
    if max_rows is not None:
        work = work.head(max_rows)

    def cell(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    headers = list(work.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in headers) + " |")
    return "\n".join(lines) + "\n"


def has_any_pattern(col: str, patterns: Iterable[str]) -> bool:
    c = col.lower()
    return any(p.lower() in c for p in patterns)


def starts_with_any(col: str, prefixes: Iterable[str]) -> bool:
    return any(col.startswith(p) for p in prefixes)


def classify_feature(col: str) -> str:
    if col in ALWAYS_EXCLUDE_COLS or starts_with_any(col, ALWAYS_EXCLUDE_PREFIXES):
        return "target_or_metadata"

    if has_any_pattern(col, DIRECT_SEAM_PATTERNS):
        return "direct_seam"

    if has_any_pattern(col, GRID_LOCATION_PATTERNS):
        return "grid_location"

    if has_any_pattern(col, ENDPOINT_VELOCITY_PATTERNS):
        return "endpoint_velocity"

    if has_any_pattern(col, TORTUOSITY_PATTERNS):
        return "tortuosity"

    if "criticality" in col.lower():
        return "criticality"

    if "holonomy" in col.lower() or "obstruction" in col.lower():
        return "holonomy_obstruction"

    if "signed_phase" in col.lower():
        return "signed_phase"

    if col.startswith("pd_"):
        return "path_diagnostic_continuous"

    if col.startswith("pn_"):
        return "path_node_continuous"

    return "field_other"


def is_numeric_predictor(df: pd.DataFrame, col: str) -> bool:
    if col in ALWAYS_EXCLUDE_COLS:
        return False
    if starts_with_any(col, ALWAYS_EXCLUDE_PREFIXES):
        return False
    return pd.api.types.is_numeric_dtype(df[col])


def allowed_feature(col: str, feature_set: str) -> bool:
    if col in ALWAYS_EXCLUDE_COLS or starts_with_any(col, ALWAYS_EXCLUDE_PREFIXES):
        return False

    if has_any_pattern(col, DIRECT_SEAM_PATTERNS):
        return False

    if has_any_pattern(col, GRID_LOCATION_PATTERNS):
        return False

    if feature_set in {
        "no_direct_seam_no_grid_no_endpoint_velocity",
        "no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity",
        "holonomy_criticality_shape_only",
    }:
        if has_any_pattern(col, ENDPOINT_VELOCITY_PATTERNS):
            return False

    if feature_set in {
        "no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity",
        "holonomy_criticality_shape_only",
    }:
        if has_any_pattern(col, TORTUOSITY_PATTERNS):
            return False

    if feature_set == "holonomy_criticality_shape_only":
        return has_any_pattern(col, HOLONOMY_CRITICALITY_SHAPE_KEEP_PATTERNS)

    return True


def make_model(cfg: RunConfig):
    if cfg.model_family == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    if cfg.model_family == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model_family: {cfg.model_family}")


def read_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "corpus" not in df.columns:
        raise ValueError(f"Missing required corpus column in {path}")
    return df


# ---------------------------------------------------------------------
# Audits and manifests
# ---------------------------------------------------------------------


def build_target_class_counts(
    pair: PairSpec,
    df: pd.DataFrame,
    targets: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for target in targets:
        target_col = TARGET_COLS[target]
        if target_col not in df.columns:
            rows.append(
                {
                    "pair": pair.name,
                    "target": target,
                    "target_col": target_col,
                    "corpus": pd.NA,
                    "class_label": pd.NA,
                    "n": 0,
                    "status": "missing_target_column",
                }
            )
            continue

        for corpus, sub in df.groupby("corpus", dropna=False):
            counts = sub[target_col].value_counts(dropna=False)
            if counts.empty:
                rows.append(
                    {
                        "pair": pair.name,
                        "target": target,
                        "target_col": target_col,
                        "corpus": corpus,
                        "class_label": pd.NA,
                        "n": 0,
                        "status": "no_rows",
                    }
                )
            else:
                for label, n in counts.items():
                    rows.append(
                        {
                            "pair": pair.name,
                            "target": target,
                            "target_col": target_col,
                            "corpus": corpus,
                            "class_label": label if not pd.isna(label) else "NaN",
                            "n": int(n),
                            "status": "ok",
                        }
                    )
    return pd.DataFrame(rows)


def build_feature_manifest(
    df_by_pair: dict[str, pd.DataFrame],
    targets: Iterable[str],
    feature_sets: Iterable[str],
) -> pd.DataFrame:
    rows = []

    # Use union of columns so C/Cp/Cp2 small schema differences are visible.
    all_cols: list[str] = sorted({c for df in df_by_pair.values() for c in df.columns})

    for target in targets:
        target_col = TARGET_COLS[target]
        for feature_set in feature_sets:
            for col in all_cols:
                provenance = classify_feature(col)
                numeric_somewhere = any(
                    col in df.columns and pd.api.types.is_numeric_dtype(df[col])
                    for df in df_by_pair.values()
                )
                allowed = bool(numeric_somewhere and allowed_feature(col, feature_set))
                rows.append(
                    {
                        "target": target,
                        "target_col": target_col,
                        "feature_set": feature_set,
                        "feature": col,
                        "provenance_or_exclusion": provenance,
                        "numeric_somewhere": int(numeric_somewhere),
                        "allowed": int(allowed),
                    }
                )

    return pd.DataFrame(rows)


def feature_columns_for_pair(df: pd.DataFrame, feature_set: str) -> list[str]:
    cols = []
    for col in df.columns:
        if not is_numeric_predictor(df, col):
            continue
        if not allowed_feature(col, feature_set):
            continue
        cols.append(col)
    return cols


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------


def class_status(y: pd.Series, min_class_count: int, warn_class_count: int) -> tuple[str, dict[str, int], str]:
    y_clean = y.dropna().astype(str)
    counts = y_clean.value_counts().to_dict()

    if len(counts) < 2:
        return "insufficient_classes", counts, "fewer_than_two_classes"

    min_count = min(counts.values())
    if min_count < min_class_count:
        return "insufficient_classes", counts, f"min_class_count_below_{min_class_count}"

    if min_count < warn_class_count:
        return "ok_warn_small_class", counts, f"min_class_count_below_warn_{warn_class_count}"

    return "ok", counts, "ok"


def transfer_run(
    *,
    pair: PairSpec,
    df: pd.DataFrame,
    target: str,
    feature_set: str,
    train_corpus: str,
    test_corpus: str,
    cfg: RunConfig,
) -> dict:
    target_col = TARGET_COLS[target]

    base = {
        "run_type": "cross_corpus_transfer",
        "pair": pair.name,
        "target": target,
        "target_col": target_col,
        "feature_set": feature_set,
        "train_corpus": train_corpus,
        "test_corpus": test_corpus,
        "model_family": cfg.model_family,
        "max_depth": cfg.max_depth if cfg.max_depth is not None else pd.NA,
        "n_estimators": cfg.n_estimators if cfg.model_family == "rf" else pd.NA,
    }

    if target_col not in df.columns:
        return {
            **base,
            "status": "missing_target_column",
            "status_detail": "missing_target_column",
            "n_train": 0,
            "n_test": 0,
            "feature_count": 0,
            "accuracy": pd.NA,
            "balanced_accuracy": pd.NA,
            "macro_f1": pd.NA,
            "weighted_f1": pd.NA,
            "train_class_counts": "{}",
            "test_class_counts": "{}",
        }

    feature_cols = feature_columns_for_pair(df, feature_set)
    if not feature_cols:
        return {
            **base,
            "status": "no_features",
            "status_detail": "no_features_after_mask",
            "n_train": 0,
            "n_test": 0,
            "feature_count": 0,
            "accuracy": pd.NA,
            "balanced_accuracy": pd.NA,
            "macro_f1": pd.NA,
            "weighted_f1": pd.NA,
            "train_class_counts": "{}",
            "test_class_counts": "{}",
        }

    train = df[df["corpus"].astype(str).eq(train_corpus)].copy()
    test = df[df["corpus"].astype(str).eq(test_corpus)].copy()

    train = train.dropna(subset=[target_col]).copy()
    test = test.dropna(subset=[target_col]).copy()

    train_status, train_counts, train_detail = class_status(
        train[target_col],
        cfg.min_class_count,
        cfg.warn_class_count,
    )
    test_status, test_counts, test_detail = class_status(
        test[target_col],
        cfg.min_class_count,
        cfg.warn_class_count,
    )

    hard_bad = {"insufficient_classes"}

    if train_status in hard_bad or test_status in hard_bad:
        return {
            **base,
            "status": "insufficient_classes",
            "status_detail": f"train={train_detail};test={test_detail}",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "feature_count": int(len(feature_cols)),
            "accuracy": pd.NA,
            "balanced_accuracy": pd.NA,
            "macro_f1": pd.NA,
            "weighted_f1": pd.NA,
            "train_class_counts": repr(train_counts),
            "test_class_counts": repr(test_counts),
        }

    status = "ok"
    status_detail = "ok"
    if train_status == "ok_warn_small_class" or test_status == "ok_warn_small_class":
        status = "ok_warn_small_class"
        status_detail = f"train={train_detail};test={test_detail}"

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train[target_col].astype(str)
    y_test = test[target_col].astype(str)

    if cfg.model_family == "rf":
        # RandomForestClassifier can handle neither NaNs nor infs.
        imputer = SimpleImputer(strategy="median")
        X_train_arr = imputer.fit_transform(X_train)
        X_test_arr = imputer.transform(X_test)
        model = make_model(cfg)
        model.fit(X_train_arr, y_train)
        pred = model.predict(X_test_arr)
    else:
        model = make_model(cfg)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    return {
        **base,
        "status": status,
        "status_detail": status_detail,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "train_class_counts": repr(train_counts),
        "test_class_counts": repr(test_counts),
    }


def run_pair_models(pair: PairSpec, df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for target in cfg.targets:
        for feature_set in cfg.feature_sets:
            rows.append(
                transfer_run(
                    pair=pair,
                    df=df,
                    target=target,
                    feature_set=feature_set,
                    train_corpus=pair.baseline_corpus,
                    test_corpus=pair.cp3_corpus,
                    cfg=cfg,
                )
            )
            rows.append(
                transfer_run(
                    pair=pair,
                    df=df,
                    target=target,
                    feature_set=feature_set,
                    train_corpus=pair.cp3_corpus,
                    test_corpus=pair.baseline_corpus,
                    cfg=cfg,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Asymmetry
# ---------------------------------------------------------------------


def summarize_pair_asymmetry(scores: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    rows = []

    for (target, feature_set), sub in scores.groupby(["target", "feature_set"], dropna=False):
        a = sub[
            sub["train_corpus"].astype(str).eq(pair.baseline_corpus)
            & sub["test_corpus"].astype(str).eq(pair.cp3_corpus)
        ]
        b = sub[
            sub["train_corpus"].astype(str).eq(pair.cp3_corpus)
            & sub["test_corpus"].astype(str).eq(pair.baseline_corpus)
        ]

        row = {
            "pair": pair.name,
            "baseline_corpus": pair.baseline_corpus,
            "cp3_corpus": pair.cp3_corpus,
            "target": target,
            "target_col": TARGET_COLS.get(str(target), pd.NA),
            "feature_set": feature_set,
            "ba_baseline_to_cp3": pd.NA,
            "ba_cp3_to_baseline": pd.NA,
            "asymmetry_cp3_minus_baseline": pd.NA,
            "macro_f1_baseline_to_cp3": pd.NA,
            "macro_f1_cp3_to_baseline": pd.NA,
            "n_train_baseline_to_cp3": pd.NA,
            "n_test_baseline_to_cp3": pd.NA,
            "n_train_cp3_to_baseline": pd.NA,
            "n_test_cp3_to_baseline": pd.NA,
            "feature_count": pd.NA,
            "status_baseline_to_cp3": "missing_direction",
            "status_cp3_to_baseline": "missing_direction",
            "status_detail_baseline_to_cp3": pd.NA,
            "status_detail_cp3_to_baseline": pd.NA,
        }

        if len(a):
            ar = a.iloc[0]
            row.update(
                {
                    "ba_baseline_to_cp3": ar["balanced_accuracy"],
                    "macro_f1_baseline_to_cp3": ar["macro_f1"],
                    "n_train_baseline_to_cp3": ar["n_train"],
                    "n_test_baseline_to_cp3": ar["n_test"],
                    "feature_count": ar["feature_count"],
                    "status_baseline_to_cp3": ar["status"],
                    "status_detail_baseline_to_cp3": ar["status_detail"],
                }
            )

        if len(b):
            br = b.iloc[0]
            row.update(
                {
                    "ba_cp3_to_baseline": br["balanced_accuracy"],
                    "macro_f1_cp3_to_baseline": br["macro_f1"],
                    "n_train_cp3_to_baseline": br["n_train"],
                    "n_test_cp3_to_baseline": br["n_test"],
                    "feature_count": br["feature_count"],
                    "status_cp3_to_baseline": br["status"],
                    "status_detail_cp3_to_baseline": br["status_detail"],
                }
            )

        ba_a = safe_float(row["ba_baseline_to_cp3"])
        ba_b = safe_float(row["ba_cp3_to_baseline"])
        if math.isfinite(ba_a) and math.isfinite(ba_b):
            row["asymmetry_cp3_minus_baseline"] = ba_b - ba_a

        rows.append(row)

    return pd.DataFrame(rows)


def build_specificity(
    cp2_asym: pd.DataFrame,
    cp_asym: pd.DataFrame,
    c_asym: pd.DataFrame | None,
) -> pd.DataFrame:
    left = cp2_asym.copy()
    left = left.rename(
        columns={
            "pair": "pair_cp2",
            "ba_baseline_to_cp3": "ba_cp2_to_cp3",
            "ba_cp3_to_baseline": "ba_cp3_to_cp2",
            "asymmetry_cp3_minus_baseline": "asymmetry_cp3_minus_cp2",
            "macro_f1_baseline_to_cp3": "macro_f1_cp2_to_cp3",
            "macro_f1_cp3_to_baseline": "macro_f1_cp3_to_cp2",
            "n_train_baseline_to_cp3": "n_train_cp2_to_cp3",
            "n_test_baseline_to_cp3": "n_test_cp2_to_cp3",
            "n_train_cp3_to_baseline": "n_train_cp3_to_cp2",
            "n_test_cp3_to_baseline": "n_test_cp3_to_cp2",
            "status_baseline_to_cp3": "status_cp2_to_cp3",
            "status_cp3_to_baseline": "status_cp3_to_cp2",
            "status_detail_baseline_to_cp3": "status_detail_cp2_to_cp3",
            "status_detail_cp3_to_baseline": "status_detail_cp3_to_cp2",
        }
    )

    cp = cp_asym.copy().rename(
        columns={
            "pair": "pair_cp",
            "ba_baseline_to_cp3": "ba_cp_to_cp3",
            "ba_cp3_to_baseline": "ba_cp3_to_cp",
            "asymmetry_cp3_minus_baseline": "asymmetry_cp3_minus_cp",
            "macro_f1_baseline_to_cp3": "macro_f1_cp_to_cp3",
            "macro_f1_cp3_to_baseline": "macro_f1_cp3_to_cp",
            "n_train_baseline_to_cp3": "n_train_cp_to_cp3",
            "n_test_baseline_to_cp3": "n_test_cp_to_cp3",
            "n_train_cp3_to_baseline": "n_train_cp3_to_cp",
            "n_test_cp3_to_baseline": "n_test_cp3_to_cp",
            "status_baseline_to_cp3": "status_cp_to_cp3",
            "status_cp3_to_baseline": "status_cp3_to_cp",
            "status_detail_baseline_to_cp3": "status_detail_cp_to_cp3",
            "status_detail_cp3_to_baseline": "status_detail_cp3_to_cp",
        }
    )

    out = left.merge(
        cp[
            [
                "target",
                "feature_set",
                "pair_cp",
                "ba_cp_to_cp3",
                "ba_cp3_to_cp",
                "asymmetry_cp3_minus_cp",
                "macro_f1_cp_to_cp3",
                "macro_f1_cp3_to_cp",
                "n_train_cp_to_cp3",
                "n_test_cp_to_cp3",
                "n_train_cp3_to_cp",
                "n_test_cp3_to_cp",
                "status_cp_to_cp3",
                "status_cp3_to_cp",
                "status_detail_cp_to_cp3",
                "status_detail_cp3_to_cp",
            ]
        ],
        on=["target", "feature_set"],
        how="left",
    )

    if c_asym is not None and not c_asym.empty:
        c = c_asym.copy().rename(
            columns={
                "pair": "pair_c",
                "ba_baseline_to_cp3": "ba_c_to_cp3",
                "ba_cp3_to_baseline": "ba_cp3_to_c",
                "asymmetry_cp3_minus_baseline": "asymmetry_cp3_minus_c",
                "macro_f1_baseline_to_cp3": "macro_f1_c_to_cp3",
                "macro_f1_cp3_to_baseline": "macro_f1_cp3_to_c",
                "n_train_baseline_to_cp3": "n_train_c_to_cp3",
                "n_test_baseline_to_cp3": "n_test_c_to_cp3",
                "n_train_cp3_to_baseline": "n_train_cp3_to_c",
                "n_test_cp3_to_baseline": "n_test_cp3_to_c",
                "status_baseline_to_cp3": "status_c_to_cp3",
                "status_cp3_to_baseline": "status_cp3_to_c",
                "status_detail_baseline_to_cp3": "status_detail_c_to_cp3",
                "status_detail_cp3_to_baseline": "status_detail_cp3_to_c",
            }
        )
        out = out.merge(
            c[
                [
                    "target",
                    "feature_set",
                    "pair_c",
                    "ba_c_to_cp3",
                    "ba_cp3_to_c",
                    "asymmetry_cp3_minus_c",
                    "macro_f1_c_to_cp3",
                    "macro_f1_cp3_to_c",
                    "n_train_c_to_cp3",
                    "n_test_c_to_cp3",
                    "n_train_cp3_to_c",
                    "n_test_cp3_to_c",
                    "status_c_to_cp3",
                    "status_cp3_to_c",
                    "status_detail_c_to_cp3",
                    "status_detail_cp3_to_c",
                ]
            ],
            on=["target", "feature_set"],
            how="left",
        )
    else:
        out["pair_c"] = pd.NA
        out["ba_c_to_cp3"] = pd.NA
        out["ba_cp3_to_c"] = pd.NA
        out["asymmetry_cp3_minus_c"] = pd.NA
        out["macro_f1_c_to_cp3"] = pd.NA
        out["macro_f1_cp3_to_c"] = pd.NA
        out["n_train_c_to_cp3"] = pd.NA
        out["n_test_c_to_cp3"] = pd.NA
        out["n_train_cp3_to_c"] = pd.NA
        out["n_test_cp3_to_c"] = pd.NA
        out["status_c_to_cp3"] = pd.NA
        out["status_cp3_to_c"] = pd.NA
        out["status_detail_c_to_cp3"] = pd.NA
        out["status_detail_cp3_to_c"] = pd.NA

    asym_cp2 = pd.to_numeric(out["asymmetry_cp3_minus_cp2"], errors="coerce")
    asym_cp = pd.to_numeric(out["asymmetry_cp3_minus_cp"], errors="coerce")

    out["specificity_vs_cp"] = asym_cp2 - asym_cp

    if "asymmetry_cp3_minus_c" in out.columns:
        asym_c = pd.to_numeric(out["asymmetry_cp3_minus_c"], errors="coerce")
        out["specificity_vs_c"] = asym_cp2 - asym_c
    else:
        out["specificity_vs_c"] = np.nan

    out["abs_specificity_vs_cp"] = pd.to_numeric(out["specificity_vs_cp"], errors="coerce").abs()
    out["abs_specificity_vs_c"] = pd.to_numeric(out["specificity_vs_c"], errors="coerce").abs()

    out["focus_priority"] = out["target"].map(FOCUS_ORDER).fillna(999).astype(int)

    # Useful status rollup.
    status_cols = [
        "status_cp2_to_cp3",
        "status_cp3_to_cp2",
        "status_cp_to_cp3",
        "status_cp3_to_cp",
        "status_c_to_cp3",
        "status_cp3_to_c",
    ]
    present = [c for c in status_cols if c in out.columns]

    def rollup(row):
        vals = [str(row[c]) for c in present if not pd.isna(row[c])]
        if not vals:
            return "missing"
        if all(v.startswith("ok") for v in vals):
            if any(v == "ok_warn_small_class" for v in vals):
                return "ok_warn_small_class"
            return "ok"
        if any(v == "insufficient_classes" for v in vals):
            return "partial_or_insufficient_classes"
        return "partial"

    out["specificity_status"] = out.apply(rollup, axis=1)

    sort_cols = ["focus_priority", "feature_set"]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def write_summary(
    *,
    cfg: RunConfig,
    pair_specs: list[PairSpec],
    corpus_manifest: pd.DataFrame,
    target_counts: pd.DataFrame,
    model_scores: pd.DataFrame,
    pair_asymmetry: pd.DataFrame,
    specificity: pd.DataFrame,
    feature_manifest: pd.DataFrame,
) -> None:
    out = cfg.outdir / "obs075b_summary.md"

    valid_specificity = specificity[
        specificity["specificity_status"].astype(str).str.startswith("ok")
    ].copy()

    anti_cols = [
        "target",
        "feature_set",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "asymmetry_cp3_minus_cp2",
        "ba_cp_to_cp3",
        "ba_cp3_to_cp",
        "asymmetry_cp3_minus_cp",
        "specificity_vs_cp",
        "ba_c_to_cp3",
        "ba_cp3_to_c",
        "asymmetry_cp3_minus_c",
        "specificity_vs_c",
        "specificity_status",
    ]

    focus = specificity[
        specificity["target"].isin(
            [
                "coupled_outcome_group_no_direct_seam_no_grid",
                "recovery_channel_no_direct_seam_no_grid",
                "recovery_channel_boundedness_strict",
                "coupling_class_no_direct_seam_no_grid",
                "outcome_group_no_direct_seam_no_grid",
                "path_family_no_direct_seam_no_grid",
            ]
        )
    ].copy()

    top_vs_cp = (
        valid_specificity.assign(
            specificity_vs_cp_num=pd.to_numeric(valid_specificity["specificity_vs_cp"], errors="coerce")
        )
        .sort_values("specificity_vs_cp_num", ascending=False, na_position="last")
        .drop(columns=["specificity_vs_cp_num"])
        .head(20)
    )

    top_vs_c = (
        valid_specificity.assign(
            specificity_vs_c_num=pd.to_numeric(valid_specificity["specificity_vs_c"], errors="coerce")
        )
        .sort_values("specificity_vs_c_num", ascending=False, na_position="last")
        .drop(columns=["specificity_vs_c_num"])
        .head(20)
    )

    unavailable = specificity[
        ~specificity["specificity_status"].astype(str).str.startswith("ok")
    ].copy()

    feature_summary = (
        feature_manifest.groupby(["feature_set", "provenance_or_exclusion"])["allowed"]
        .agg(n_features="count", n_allowed="sum")
        .reset_index()
        .sort_values(["feature_set", "provenance_or_exclusion"])
    )

    lines = []
    lines.append("# OBS-075b — Cp3 endpoint / velocity ablation v2")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "OBS-075b recomputes Cp3 directional transfer after removing direct seam, "
        "absolute grid-location, endpoint/velocity, path-length, and tortuosity proxies."
    )
    lines.append("")
    lines.append(
        "v2 uses already-produced OBS-073 feature tables as the primary input, so coupled, "
        "coupled-outcome, and recovery-channel labels are inherited from the validated OBS-073 bridge."
    )
    lines.append("")
    lines.append("This script does not recompute geometry, path families, coupling, outcomes, or recovery channels.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for spec in pair_specs:
        lines.append(f"- `{spec.name}`: `{spec.table_path}`; baseline `{spec.baseline_corpus}`; Cp3 `{spec.cp3_corpus}`")
    lines.append("")
    lines.append("## Model configuration")
    lines.append("")
    lines.append(f"- model family: `{cfg.model_family}`")
    if cfg.model_family == "rf":
        lines.append(f"- n estimators: `{cfg.n_estimators}`")
        lines.append(f"- max depth: `{cfg.max_depth if cfg.max_depth is not None else 'None'}`")
    lines.append(f"- min class count: `{cfg.min_class_count}`")
    lines.append(f"- warning class count: `{cfg.warn_class_count}`")
    lines.append(f"- random state: `{cfg.random_state}`")
    lines.append("")
    lines.append("## Feature sets")
    lines.append("")
    lines.append("- `no_direct_seam_no_grid`: removes seam/proximity and absolute grid/location features.")
    lines.append("- `no_direct_seam_no_grid_no_endpoint_velocity`: additionally removes endpoint, delta, displacement, path-length, and n-step proxies.")
    lines.append("- `no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity`: additionally removes tortuosity, turning, angle-jump, sector-change, and DEW-angle proxies.")
    lines.append("- `holonomy_criticality_shape_only`: keeps only holonomy/obstruction, criticality, signed-phase, and path-shape features.")
    lines.append("")
    lines.append("## Corpus manifest")
    lines.append("")
    lines.append(markdown_table(corpus_manifest))
    lines.append("")
    lines.append("## Target class-count audit")
    lines.append("")
    lines.append(
        "Rows with small minority classes are allowed only above `min_class_count`; "
        "`ok_warn_small_class` rows should be interpreted cautiously."
    )
    lines.append("")
    target_count_pivot = target_counts.copy()
    lines.append(markdown_table(target_count_pivot, max_rows=80))
    lines.append("")
    lines.append("## Specificity table")
    lines.append("")
    lines.append(markdown_table(specificity, columns=anti_cols))
    lines.append("")
    lines.append("## Focus targets")
    lines.append("")
    lines.append(markdown_table(focus, columns=anti_cols))
    lines.append("")
    lines.append("## Highest specificity versus Cp control")
    lines.append("")
    lines.append(markdown_table(top_vs_cp, columns=anti_cols))
    lines.append("")
    lines.append("## Highest specificity versus C control")
    lines.append("")
    lines.append(markdown_table(top_vs_c, columns=anti_cols))
    lines.append("")
    lines.append("## Unavailable / partial rows")
    lines.append("")
    if unavailable.empty:
        lines.append("_No unavailable rows under the configured class-count thresholds._")
    else:
        lines.append(
            markdown_table(
                unavailable,
                columns=[
                    "target",
                    "feature_set",
                    "specificity_status",
                    "status_cp2_to_cp3",
                    "status_cp3_to_cp2",
                    "status_cp_to_cp3",
                    "status_cp3_to_cp",
                    "status_c_to_cp3",
                    "status_cp3_to_c",
                ],
            )
        )
    lines.append("")
    lines.append("## Model scores")
    lines.append("")
    lines.append(
        markdown_table(
            model_scores,
            columns=[
                "pair",
                "target",
                "feature_set",
                "train_corpus",
                "test_corpus",
                "model_family",
                "max_depth",
                "status",
                "status_detail",
                "n_train",
                "n_test",
                "feature_count",
                "balanced_accuracy",
                "macro_f1",
            ],
            max_rows=120,
        )
    )
    lines.append("")
    lines.append("## Feature-manifest summary")
    lines.append("")
    lines.append(markdown_table(feature_summary))
    lines.append("")
    lines.append("## Provisional read")
    lines.append("")

    # Conservative automatic read.
    ok_coupled = specificity[
        specificity["target"].eq("coupled_outcome_group_no_direct_seam_no_grid")
        & specificity["specificity_status"].astype(str).str.startswith("ok")
    ]
    ok_recovery = specificity[
        specificity["target"].eq("recovery_channel_no_direct_seam_no_grid")
        & specificity["specificity_status"].astype(str).str.startswith("ok")
    ]
    ok_outcome = specificity[
        specificity["target"].eq("outcome_group_no_direct_seam_no_grid")
        & specificity["specificity_status"].astype(str).str.startswith("ok")
    ]

    if not ok_coupled.empty or not ok_recovery.empty:
        lines.append(
            "Coupled/recovery targets are available in OBS-073 feature-table mode. "
            "Rows that remain positive after endpoint/velocity and tortuosity removal are stronger "
            "than OBS-075's original transfer-asymmetry read; rows that collapse should be interpreted "
            "as terminal-trajectory or path-proxy mediated."
        )
    elif not ok_outcome.empty:
        lines.append(
            "Coupled/recovery targets are unavailable under the configured thresholds, but outcome-group "
            "targets remain testable. Any surviving effect should be narrowed to outcome-level transfer, "
            "not the original coupled/recovery OBS-075 claim."
        )
    else:
        lines.append(
            "No primary target is available under the configured thresholds. This is an input/class-count "
            "availability result, not a scientific collapse."
        )

    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("")
    lines.append("- `insufficient_classes` means the slice is unavailable or below threshold, not that the effect collapsed.")
    lines.append("- C-control recovery rows may be fragile because C has a very small false-recovery class.")
    lines.append("- Positive specificity versus Cp but not versus C suggests a corpus-family interaction, not strict Cp2 specificity.")
    lines.append("- Positive specificity after endpoint/velocity and tortuosity removal is stronger evidence against the broad-boundary critique.")
    lines.append("- Random forests should be compared with `--model-family logreg` and shallow forests via `--max-depth 2` or `--max-depth 4`.")
    lines.append("- Path-level lexical controls remain unavailable unless corpus JSON responses can be joined to path IDs.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def build_corpus_manifest(pair_specs: list[PairSpec], df_by_pair: dict[str, pd.DataFrame], targets: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for spec in pair_specs:
        df = df_by_pair[spec.name]
        for corpus, sub in df.groupby("corpus", dropna=False):
            row = {
                "pair": spec.name,
                "corpus": corpus,
                "n_rows": int(len(sub)),
                "n_columns": int(sub.shape[1]),
            }
            for target in targets:
                col = TARGET_COLS[target]
                if col in sub.columns:
                    row[f"n_{col}"] = int(sub[col].notna().sum())
                    row[f"n_classes_{col}"] = int(sub[col].dropna().astype(str).nunique())
                else:
                    row[f"n_{col}"] = 0
                    row[f"n_classes_{col}"] = 0
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="OBS-075b v2: recompute Cp3 endpoint/velocity ablations from OBS-073 feature tables."
    )

    parser.add_argument(
        "--cp2-cp3-feature-table",
        default="outputs/comparisons/obs073_Cp2_vs_Cp3_v5_smoke/obs073_feature_table.csv",
        help="OBS-073 feature table for Cp2 vs Cp3.",
    )
    parser.add_argument(
        "--cp-cp3-feature-table",
        default="outputs/comparisons/obs073_Cp_vs_Cp3_v5_smoke/obs073_feature_table.csv",
        help="OBS-073 feature table for Cp vs Cp3.",
    )
    parser.add_argument(
        "--c-cp3-feature-table",
        default="outputs/comparisons/obs073_C_vs_Cp3_v5_smoke/obs073_feature_table.csv",
        help="OBS-073 feature table for C vs Cp3. Use empty string to disable.",
    )

    parser.add_argument(
        "--outdir",
        default="outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_v2",
        help="Output directory.",
    )

    parser.add_argument(
        "--model-family",
        choices=["rf", "logreg"],
        default="rf",
        help="Model family.",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--max-depth",
        default=None,
        help="Random forest max_depth. Use empty/None for unlimited.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-class-count", type=int, default=25)
    parser.add_argument("--warn-class-count", type=int, default=100)

    parser.add_argument(
        "--targets",
        nargs="*",
        default=list(TARGET_COLS.keys()),
        choices=list(TARGET_COLS.keys()),
        help="Targets to run.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=list(FEATURE_SETS),
        choices=list(FEATURE_SETS),
        help="Feature sets to run.",
    )

    args = parser.parse_args()

    c_path = Path(args.c_cp3_feature_table) if str(args.c_cp3_feature_table).strip() else None

    return RunConfig(
        cp2_cp3_feature_table=Path(args.cp2_cp3_feature_table),
        cp_cp3_feature_table=Path(args.cp_cp3_feature_table),
        c_cp3_feature_table=c_path,
        outdir=Path(args.outdir),
        model_family=args.model_family,
        n_estimators=int(args.n_estimators),
        max_depth=parse_max_depth(args.max_depth),
        random_state=int(args.random_state),
        min_class_count=int(args.min_class_count),
        warn_class_count=int(args.warn_class_count),
        targets=tuple(args.targets),
        feature_sets=tuple(args.feature_sets),
    )


def main() -> None:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    pair_specs = [
        PairSpec(
            name="Cp2_vs_Cp3",
            table_path=cfg.cp2_cp3_feature_table,
            baseline_corpus="Cp2",
        ),
        PairSpec(
            name="Cp_vs_Cp3",
            table_path=cfg.cp_cp3_feature_table,
            baseline_corpus="Cp",
        ),
    ]
    if cfg.c_cp3_feature_table is not None:
        pair_specs.append(
            PairSpec(
                name="C_vs_Cp3",
                table_path=cfg.c_cp3_feature_table,
                baseline_corpus="C",
            )
        )

    df_by_pair: dict[str, pd.DataFrame] = {}
    for spec in pair_specs:
        df_by_pair[spec.name] = read_feature_table(spec.table_path)

    corpus_manifest = build_corpus_manifest(pair_specs, df_by_pair, cfg.targets)

    target_counts = pd.concat(
        [
            build_target_class_counts(spec, df_by_pair[spec.name], cfg.targets)
            for spec in pair_specs
        ],
        ignore_index=True,
    )

    feature_manifest = build_feature_manifest(df_by_pair, cfg.targets, cfg.feature_sets)

    model_scores = pd.concat(
        [
            run_pair_models(spec, df_by_pair[spec.name], cfg)
            for spec in pair_specs
        ],
        ignore_index=True,
    )

    pair_asymmetry_parts = []
    for spec in pair_specs:
        pair_scores = model_scores[model_scores["pair"].eq(spec.name)].copy()
        pair_asymmetry_parts.append(summarize_pair_asymmetry(pair_scores, spec))
    pair_asymmetry = pd.concat(pair_asymmetry_parts, ignore_index=True)

    cp2_asym = pair_asymmetry[pair_asymmetry["pair"].eq("Cp2_vs_Cp3")].copy()
    cp_asym = pair_asymmetry[pair_asymmetry["pair"].eq("Cp_vs_Cp3")].copy()
    c_asym = pair_asymmetry[pair_asymmetry["pair"].eq("C_vs_Cp3")].copy()
    if c_asym.empty:
        c_asym = None

    specificity = build_specificity(cp2_asym, cp_asym, c_asym)

    # Write outputs.
    corpus_manifest.to_csv(cfg.outdir / "obs075b_corpus_manifest.csv", index=False)
    target_counts.to_csv(cfg.outdir / "obs075b_target_class_counts.csv", index=False)
    feature_manifest.to_csv(cfg.outdir / "obs075b_feature_manifest.csv", index=False)
    model_scores.to_csv(cfg.outdir / "obs075b_model_scores.csv", index=False)
    pair_asymmetry.to_csv(cfg.outdir / "obs075b_pair_asymmetry.csv", index=False)
    specificity.to_csv(cfg.outdir / "obs075b_asymmetry_specificity.csv", index=False)

    write_summary(
        cfg=cfg,
        pair_specs=pair_specs,
        corpus_manifest=corpus_manifest,
        target_counts=target_counts,
        model_scores=model_scores,
        pair_asymmetry=pair_asymmetry,
        specificity=specificity,
        feature_manifest=feature_manifest,
    )

    print(cfg.outdir / "obs075b_corpus_manifest.csv")
    print(cfg.outdir / "obs075b_target_class_counts.csv")
    print(cfg.outdir / "obs075b_feature_manifest.csv")
    print(cfg.outdir / "obs075b_model_scores.csv")
    print(cfg.outdir / "obs075b_pair_asymmetry.csv")
    print(cfg.outdir / "obs075b_asymmetry_specificity.csv")
    print(cfg.outdir / "obs075b_summary.md")


if __name__ == "__main__":
    main()
