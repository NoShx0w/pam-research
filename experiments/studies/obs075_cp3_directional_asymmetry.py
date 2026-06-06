#!/usr/bin/env python3
"""
OBS-075 — Cp3 directional asymmetry v2.

Purpose
-------
Summarize whether Cp3 transfer asymmetry is specific to Cp2, or whether it is a
generic artifact of Cp3 acting as a noisy / smeared / broad-boundary corpus.

v2 adds:
1. Explicit anti-shortcut specificity table.
2. Coupled/recovery specificity focus table.
3. Optional endpoint/velocity proxy audit from OBS-073 permutation-importance
   CSVs, if provided.
4. Stronger interpretation guardrails distinguishing:
   - observed directional asymmetry
   - anti-shortcut asymmetry
   - feature-proxy risk
   - still-missing model-recompute ablations.

This script consumes existing OBS-073 v5 artifacts. It does not recompute
geometry, path families, coupling, outcomes, recovery channels, or transfer
models.

Primary question
----------------
Is Cp3's directional transfer asymmetry strongest against Cp2, especially for
coupled-outcome and recovery-channel targets, or does the same asymmetry appear
against Cp?

Key interpretation
------------------
If Cp3 -> Cp2 greatly exceeds Cp2 -> Cp3 for coupled/recovery targets, while
Cp3 -> Cp does not similarly exceed Cp -> Cp3, then the asymmetry is
relation-specific rather than a generic "blurred Cp3 transfers everywhere"
artifact.

Outputs
-------
obs075_directional_asymmetry.csv
    Per pair / target / variant directional transfer rows.

obs075_asymmetry_specificity.csv
    Shared target variants aligned across Cp2<->Cp3 and Cp<->Cp3, with
    Cp2-specific asymmetry contrast.

obs075_anti_shortcut_specificity.csv
    no_direct_seam_no_grid rows only, foregrounding the most defensible
    anti-shortcut evidence.

obs075_target_focus.csv
    Compact focus table for coupled / recovery / route / outcome targets.

obs075_endpoint_velocity_proxy_audit.csv
    Optional feature-proxy audit if permutation-importance inputs are present.

obs075_pair_audit.csv
    Pair completeness audit.

obs075_summary.md
    Human-readable OBS-075 summary.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    name: str
    scores_csv: Path
    baseline_corpus: str
    cp3_corpus: str = "Cp3"
    permutation_csv: Path | None = None


@dataclass(frozen=True)
class Config:
    cp2_pair_scores: Path
    cp_pair_scores: Path
    outdir: Path

    cp2_pair_name: str = "Cp2_vs_Cp3"
    cp_pair_name: str = "Cp_vs_Cp3"
    cp2_baseline: str = "Cp2"
    cp_baseline: str = "Cp"
    cp3_corpus: str = "Cp3"

    cp2_pair_permutation: Path | None = None
    cp_pair_permutation: Path | None = None

    top_n_specificity: int = 25
    top_n_focus: int = 30
    top_n_proxy_audit: int = 20


FOCUS_TARGET_PATTERNS = (
    "coupled_outcome_group",
    "recovery_channel",
    "coupling_class",
    "outcome_group",
    "path_family",
)


ANTI_SHORTCUT_VARIANTS = {"no_direct_seam_no_grid"}


ENDPOINT_VELOCITY_PATTERNS = (
    "last_minus_first",
    "n_steps",
    "path_length",
    "arclength",
    "arc_length",
    "chord",
    "net_displacement",
    "tortuosity",
    "turning",
    "step_count",
    "path_n",
)


DIRECT_SEAM_PATTERNS = (
    "distance_to_seam",
    "near_fraction",
    "mid_fraction",
    "far_fraction",
    "seam",
)


GRID_LOCATION_PATTERNS = (
    "node_id",
    "mds1",
    "mds2",
    "alpha",
    "r_",
    "_r",
)


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required {label} exists but is empty: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def safe_int_flag(x: Any) -> int:
    v = safe_float(x)
    if not np.isfinite(v):
        return 0
    return int(v != 0)


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def first_present_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------
# Target normalization
# ---------------------------------------------------------------------


def infer_variant(target: str) -> str:
    """
    Convert OBS-073 target names into a shared target variant.

    Examples
    --------
    path_family_no_direct_seam_no_grid -> no_direct_seam_no_grid
    path_family                         -> full
    recovery_channel_structural         -> full_structural
    recovery_channel_boundedness_strict -> boundedness_strict
    """
    t = str(target)

    if t == "recovery_channel_structural":
        return "full_structural"
    if t == "recovery_channel_boundedness_strict":
        return "boundedness_strict"

    suffixes = [
        "_no_direct_seam_no_grid",
        "_no_direct_seam",
        "_no_grid",
    ]
    for suffix in suffixes:
        if t.endswith(suffix):
            return suffix.removeprefix("_")

    return "full"


def infer_base_target(target: str) -> str:
    """
    Strip OBS-073 blinding suffixes while preserving recovery-channel structure.
    """
    t = str(target)

    if t == "recovery_channel_structural":
        return "recovery_channel"
    if t == "recovery_channel_boundedness_strict":
        return "recovery_channel_boundedness_strict"
    if t.startswith("recovery_channel_"):
        return "recovery_channel"

    for suffix in [
        "_no_direct_seam_no_grid",
        "_no_direct_seam",
        "_no_grid",
    ]:
        if t.endswith(suffix):
            return t[: -len(suffix)]

    return t


def is_focus_target(base_target: str) -> bool:
    b = str(base_target)
    return any(p in b for p in FOCUS_TARGET_PATTERNS)


def focus_priority(base_target: str) -> int:
    b = str(base_target)
    if "coupled_outcome" in b:
        return 0
    if b == "recovery_channel":
        return 1
    if "recovery_channel_boundedness" in b:
        return 2
    if "coupling_class" in b:
        return 3
    if "outcome_group" in b:
        return 4
    if "path_family" in b:
        return 5
    return 9


# ---------------------------------------------------------------------
# Directional asymmetry construction
# ---------------------------------------------------------------------


def get_first_ok_transfer(
    scores: pd.DataFrame,
    *,
    target: str,
    train_corpus: str,
    test_corpus: str,
) -> pd.Series | None:
    rows = scores[
        scores.get("run_type", "").astype(str).eq("cross_corpus_transfer")
        & scores.get("status", "").astype(str).eq("ok")
        & scores.get("target", "").astype(str).eq(target)
        & scores.get("train_corpus", "").astype(str).eq(train_corpus)
        & scores.get("test_corpus", "").astype(str).eq(test_corpus)
    ].copy()

    if rows.empty:
        return None

    rows["_n_test_sort"] = numeric_col(rows, "n_test")
    rows["_ba_sort"] = numeric_col(rows, "balanced_accuracy")
    rows = rows.sort_values(["_n_test_sort", "_ba_sort"], ascending=[False, False])
    return rows.iloc[0]


def row_value(row: pd.Series | None, col: str) -> Any:
    if row is None:
        return np.nan
    return row.get(col, np.nan)


def build_directional_for_pair(pair: PairSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = read_csv_required(pair.scores_csv, f"{pair.name} OBS-073 model scores")

    required = {"run_type", "status", "target", "train_corpus", "test_corpus", "balanced_accuracy"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"{pair.scores_csv} is missing required columns: {missing}")

    cross = scores[
        scores["run_type"].astype(str).eq("cross_corpus_transfer")
        & scores["status"].astype(str).eq("ok")
    ].copy()

    if cross.empty:
        return pd.DataFrame(), pd.DataFrame()

    targets = sorted(cross["target"].astype(str).unique().tolist())

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for target in targets:
        base_target = infer_base_target(target)
        variant = infer_variant(target)

        baseline_to_cp3 = get_first_ok_transfer(
            scores,
            target=target,
            train_corpus=pair.baseline_corpus,
            test_corpus=pair.cp3_corpus,
        )
        cp3_to_baseline = get_first_ok_transfer(
            scores,
            target=target,
            train_corpus=pair.cp3_corpus,
            test_corpus=pair.baseline_corpus,
        )

        audit_rows.append(
            {
                "pair": pair.name,
                "target": target,
                "base_target": base_target,
                "variant": variant,
                "baseline_corpus": pair.baseline_corpus,
                "cp3_corpus": pair.cp3_corpus,
                "has_baseline_to_cp3": int(baseline_to_cp3 is not None),
                "has_cp3_to_baseline": int(cp3_to_baseline is not None),
            }
        )

        if baseline_to_cp3 is None and cp3_to_baseline is None:
            continue

        ba_baseline_to_cp3 = safe_float(row_value(baseline_to_cp3, "balanced_accuracy"))
        ba_cp3_to_baseline = safe_float(row_value(cp3_to_baseline, "balanced_accuracy"))

        asym = (
            ba_cp3_to_baseline - ba_baseline_to_cp3
            if np.isfinite(ba_cp3_to_baseline) and np.isfinite(ba_baseline_to_cp3)
            else float("nan")
        )

        no_direct_seam = max(
            safe_int_flag(row_value(baseline_to_cp3, "no_direct_seam")),
            safe_int_flag(row_value(cp3_to_baseline, "no_direct_seam")),
        )
        no_grid_location = max(
            safe_int_flag(row_value(baseline_to_cp3, "no_grid_location")),
            safe_int_flag(row_value(cp3_to_baseline, "no_grid_location")),
        )

        rows.append(
            {
                "pair": pair.name,
                "target": target,
                "base_target": base_target,
                "variant": variant,
                "baseline_corpus": pair.baseline_corpus,
                "cp3_corpus": pair.cp3_corpus,

                "ba_baseline_to_cp3": ba_baseline_to_cp3,
                "ba_cp3_to_baseline": ba_cp3_to_baseline,
                "asymmetry_cp3_minus_baseline": asym,

                "macro_f1_baseline_to_cp3": safe_float(row_value(baseline_to_cp3, "macro_f1")),
                "macro_f1_cp3_to_baseline": safe_float(row_value(cp3_to_baseline, "macro_f1")),
                "accuracy_baseline_to_cp3": safe_float(row_value(baseline_to_cp3, "accuracy")),
                "accuracy_cp3_to_baseline": safe_float(row_value(cp3_to_baseline, "accuracy")),
                "weighted_f1_baseline_to_cp3": safe_float(row_value(baseline_to_cp3, "weighted_f1")),
                "weighted_f1_cp3_to_baseline": safe_float(row_value(cp3_to_baseline, "weighted_f1")),

                "n_train_baseline_to_cp3": safe_float(row_value(baseline_to_cp3, "n_train")),
                "n_test_baseline_to_cp3": safe_float(row_value(baseline_to_cp3, "n_test")),
                "n_train_cp3_to_baseline": safe_float(row_value(cp3_to_baseline, "n_train")),
                "n_test_cp3_to_baseline": safe_float(row_value(cp3_to_baseline, "n_test")),

                "feature_count_baseline_to_cp3": safe_float(row_value(baseline_to_cp3, "feature_count")),
                "feature_count_cp3_to_baseline": safe_float(row_value(cp3_to_baseline, "feature_count")),

                "no_direct_seam": no_direct_seam,
                "no_grid_location": no_grid_location,
                "anti_shortcut": int(no_direct_seam == 1 and no_grid_location == 1),
                "allow_obs051_features": max(
                    safe_int_flag(row_value(baseline_to_cp3, "allow_obs051_features")),
                    safe_int_flag(row_value(cp3_to_baseline, "allow_obs051_features")),
                ),
                "within_corpus_only": max(
                    safe_int_flag(row_value(baseline_to_cp3, "within_corpus_only")),
                    safe_int_flag(row_value(cp3_to_baseline, "within_corpus_only")),
                ),
                "obs072_locus_target": max(
                    safe_int_flag(row_value(baseline_to_cp3, "obs072_locus_target")),
                    safe_int_flag(row_value(cp3_to_baseline, "obs072_locus_target")),
                ),
                "shortcut_risk_baseline_to_cp3": row_value(baseline_to_cp3, "shortcut_risk"),
                "shortcut_risk_cp3_to_baseline": row_value(cp3_to_baseline, "shortcut_risk"),
                "is_focus_target": int(is_focus_target(base_target)),
                "focus_priority": focus_priority(base_target),
            }
        )

    directional = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)

    if not directional.empty:
        directional = directional.sort_values(
            ["focus_priority", "base_target", "variant", "pair"],
            kind="stable",
        ).reset_index(drop=True)

    return directional, audit


def build_specificity(
    cp2_directional: pd.DataFrame,
    cp_directional: pd.DataFrame,
    *,
    cp2_pair_name: str,
    cp_pair_name: str,
) -> pd.DataFrame:
    if cp2_directional.empty or cp_directional.empty:
        return pd.DataFrame()

    keep = [
        "target",
        "base_target",
        "variant",
        "ba_baseline_to_cp3",
        "ba_cp3_to_baseline",
        "asymmetry_cp3_minus_baseline",
        "macro_f1_baseline_to_cp3",
        "macro_f1_cp3_to_baseline",
        "accuracy_baseline_to_cp3",
        "accuracy_cp3_to_baseline",
        "weighted_f1_baseline_to_cp3",
        "weighted_f1_cp3_to_baseline",
        "n_train_baseline_to_cp3",
        "n_test_baseline_to_cp3",
        "n_train_cp3_to_baseline",
        "n_test_cp3_to_baseline",
        "feature_count_baseline_to_cp3",
        "feature_count_cp3_to_baseline",
        "no_direct_seam",
        "no_grid_location",
        "anti_shortcut",
        "allow_obs051_features",
        "is_focus_target",
        "focus_priority",
    ]

    keep = [c for c in keep if c in cp2_directional.columns and c in cp_directional.columns]

    left = cp2_directional[keep].copy()
    right = cp_directional[keep].copy()

    left = left.rename(
        columns={
            "ba_baseline_to_cp3": "ba_cp2_to_cp3",
            "ba_cp3_to_baseline": "ba_cp3_to_cp2",
            "asymmetry_cp3_minus_baseline": "asymmetry_cp3_minus_cp2",
            "macro_f1_baseline_to_cp3": "macro_f1_cp2_to_cp3",
            "macro_f1_cp3_to_baseline": "macro_f1_cp3_to_cp2",
            "accuracy_baseline_to_cp3": "accuracy_cp2_to_cp3",
            "accuracy_cp3_to_baseline": "accuracy_cp3_to_cp2",
            "weighted_f1_baseline_to_cp3": "weighted_f1_cp2_to_cp3",
            "weighted_f1_cp3_to_baseline": "weighted_f1_cp3_to_cp2",
            "n_train_baseline_to_cp3": "n_train_cp2_to_cp3",
            "n_test_baseline_to_cp3": "n_test_cp2_to_cp3",
            "n_train_cp3_to_baseline": "n_train_cp3_to_cp2",
            "n_test_cp3_to_baseline": "n_test_cp3_to_cp2",
            "feature_count_baseline_to_cp3": "feature_count_cp2_to_cp3",
            "feature_count_cp3_to_baseline": "feature_count_cp3_to_cp2",
        }
    )

    right = right.rename(
        columns={
            "ba_baseline_to_cp3": "ba_cp_to_cp3",
            "ba_cp3_to_baseline": "ba_cp3_to_cp",
            "asymmetry_cp3_minus_baseline": "asymmetry_cp3_minus_cp",
            "macro_f1_baseline_to_cp3": "macro_f1_cp_to_cp3",
            "macro_f1_cp3_to_baseline": "macro_f1_cp3_to_cp",
            "accuracy_baseline_to_cp3": "accuracy_cp_to_cp3",
            "accuracy_cp3_to_baseline": "accuracy_cp3_to_cp",
            "weighted_f1_baseline_to_cp3": "weighted_f1_cp_to_cp3",
            "weighted_f1_cp3_to_baseline": "weighted_f1_cp3_to_cp",
            "n_train_baseline_to_cp3": "n_train_cp_to_cp3",
            "n_test_baseline_to_cp3": "n_test_cp_to_cp3",
            "n_train_cp3_to_baseline": "n_train_cp3_to_cp",
            "n_test_cp3_to_baseline": "n_test_cp3_to_cp",
            "feature_count_baseline_to_cp3": "feature_count_cp_to_cp3",
            "feature_count_cp3_to_baseline": "feature_count_cp3_to_cp",
        }
    )

    joined = left.merge(
        right,
        on=[
            "target",
            "base_target",
            "variant",
            "no_direct_seam",
            "no_grid_location",
            "anti_shortcut",
            "allow_obs051_features",
            "is_focus_target",
            "focus_priority",
        ],
        how="inner",
    )

    if joined.empty:
        return joined

    joined["asymmetry_specificity_cp2_minus_cp"] = (
        pd.to_numeric(joined["asymmetry_cp3_minus_cp2"], errors="coerce")
        - pd.to_numeric(joined["asymmetry_cp3_minus_cp"], errors="coerce")
    )
    joined["abs_asymmetry_specificity"] = (
        pd.to_numeric(joined["asymmetry_specificity_cp2_minus_cp"], errors="coerce").abs()
    )

    joined["cp2_pair_name"] = cp2_pair_name
    joined["cp_pair_name"] = cp_pair_name

    joined = joined.sort_values(
        ["abs_asymmetry_specificity", "focus_priority", "base_target", "variant"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return joined


def build_target_focus(specificity: pd.DataFrame) -> pd.DataFrame:
    if specificity.empty:
        return pd.DataFrame()

    focus = specificity[specificity["is_focus_target"].astype(int).eq(1)].copy()
    if focus.empty:
        return focus

    focus = focus.sort_values(
        ["focus_priority", "abs_asymmetry_specificity"],
        ascending=[True, False],
        kind="stable",
    )

    return focus.reset_index(drop=True)


def build_anti_shortcut_specificity(specificity: pd.DataFrame) -> pd.DataFrame:
    if specificity.empty:
        return pd.DataFrame()

    anti = specificity[
        specificity["anti_shortcut"].astype(int).eq(1)
        | specificity["variant"].astype(str).isin(ANTI_SHORTCUT_VARIANTS)
    ].copy()

    if anti.empty:
        return anti

    anti = anti.sort_values(
        ["abs_asymmetry_specificity", "focus_priority", "base_target"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return anti


# ---------------------------------------------------------------------
# Optional endpoint/velocity proxy audit
# ---------------------------------------------------------------------


def classify_feature_name(feature: str) -> str:
    f = str(feature).lower()

    if any(p in f for p in ENDPOINT_VELOCITY_PATTERNS):
        return "endpoint_or_velocity_proxy"

    if any(p in f for p in DIRECT_SEAM_PATTERNS):
        return "direct_seam_or_seam_proximity"

    # Avoid classifying signed_phase as grid just because it may contain r as a char.
    grid_tokens = [
        "node_id",
        "mds1",
        "mds2",
        "alpha",
        "pn_r_",
        "field_pn_r_",
        "pn_r",
        "field_pn_r",
    ]
    if any(p in f for p in grid_tokens):
        return "grid_or_absolute_location"

    if "criticality" in f:
        return "criticality"
    if "holonomy" in f or "obstruction" in f:
        return "holonomy_obstruction"
    if "signed_phase" in f or "angle" in f or "sector" in f:
        return "path_field_shape"
    if "lex_" in f:
        return "lexical"

    return "other"


def normalize_importance_table(perm: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    if perm.empty:
        return pd.DataFrame()

    feature_col = first_present_column(
        perm,
        ["feature", "feature_name", "name", "column"],
    )
    target_col = first_present_column(
        perm,
        ["target", "target_name"],
    )
    run_type_col = first_present_column(
        perm,
        ["run_type"],
    )
    train_col = first_present_column(
        perm,
        ["train_corpus"],
    )
    test_col = first_present_column(
        perm,
        ["test_corpus"],
    )
    importance_col = first_present_column(
        perm,
        [
            "importance_mean",
            "permutation_importance_mean",
            "mean_importance",
            "importance",
            "score_delta_mean",
        ],
    )
    std_col = first_present_column(
        perm,
        [
            "importance_std",
            "permutation_importance_std",
            "std_importance",
            "score_delta_std",
        ],
    )

    if feature_col is None or target_col is None or importance_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["pair"] = pair_name
    out["target"] = perm[target_col].astype(str)
    out["base_target"] = out["target"].map(infer_base_target)
    out["variant"] = out["target"].map(infer_variant)
    out["feature"] = perm[feature_col].astype(str)
    out["feature_class"] = out["feature"].map(classify_feature_name)
    out["importance_mean"] = pd.to_numeric(perm[importance_col], errors="coerce")
    out["importance_std"] = pd.to_numeric(perm[std_col], errors="coerce") if std_col else np.nan

    out["run_type"] = perm[run_type_col].astype(str) if run_type_col else ""
    out["train_corpus"] = perm[train_col].astype(str) if train_col else ""
    out["test_corpus"] = perm[test_col].astype(str) if test_col else ""

    return out


def build_proxy_audit_for_pair(pair: PairSpec, top_n: int) -> pd.DataFrame:
    perm = read_csv_optional(pair.permutation_csv)
    norm = normalize_importance_table(perm, pair.name)
    if norm.empty:
        return pd.DataFrame(
            [
                {
                    "pair": pair.name,
                    "status": "missing_or_unusable_permutation_importance",
                    "target": "",
                    "base_target": "",
                    "variant": "",
                    "top_n": top_n,
                    "endpoint_velocity_importance_share": np.nan,
                    "endpoint_velocity_features_in_top_n": np.nan,
                    "direct_seam_importance_share": np.nan,
                    "grid_location_importance_share": np.nan,
                    "top_endpoint_velocity_features": "",
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    group_cols = ["target", "base_target", "variant"]

    for keys, g in norm.groupby(group_cols, dropna=False):
        target, base_target, variant = keys
        gg = g.copy()
        gg = gg[np.isfinite(pd.to_numeric(gg["importance_mean"], errors="coerce"))]
        gg = gg.sort_values("importance_mean", ascending=False).head(top_n)

        total = float(gg["importance_mean"].clip(lower=0).sum())
        endpoint = gg[gg["feature_class"].eq("endpoint_or_velocity_proxy")]
        seam = gg[gg["feature_class"].eq("direct_seam_or_seam_proximity")]
        grid = gg[gg["feature_class"].eq("grid_or_absolute_location")]

        endpoint_sum = float(endpoint["importance_mean"].clip(lower=0).sum())
        seam_sum = float(seam["importance_mean"].clip(lower=0).sum())
        grid_sum = float(grid["importance_mean"].clip(lower=0).sum())

        top_endpoint_features = "; ".join(
            endpoint.sort_values("importance_mean", ascending=False)["feature"].head(8).tolist()
        )

        rows.append(
            {
                "pair": pair.name,
                "status": "ok",
                "target": target,
                "base_target": base_target,
                "variant": variant,
                "top_n": top_n,
                "top_n_total_positive_importance": total,
                "endpoint_velocity_importance_sum": endpoint_sum,
                "endpoint_velocity_importance_share": endpoint_sum / total if total > 0 else 0.0,
                "endpoint_velocity_features_in_top_n": int(len(endpoint)),
                "direct_seam_importance_sum": seam_sum,
                "direct_seam_importance_share": seam_sum / total if total > 0 else 0.0,
                "grid_location_importance_sum": grid_sum,
                "grid_location_importance_share": grid_sum / total if total > 0 else 0.0,
                "top_endpoint_velocity_features": top_endpoint_features,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["endpoint_velocity_importance_share", "endpoint_velocity_features_in_top_n"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)

    return out


def build_proxy_audit(cp2_pair: PairSpec, cp_pair: PairSpec, top_n: int) -> pd.DataFrame:
    return pd.concat(
        [
            build_proxy_audit_for_pair(cp2_pair, top_n=top_n),
            build_proxy_audit_for_pair(cp_pair, top_n=top_n),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> list[str]:
    if df.empty:
        return ["No rows."]

    use = [c for c in cols if c in df.columns]
    if max_rows is not None:
        df = df.head(max_rows)

    lines = [
        "| " + " | ".join(use) + " |",
        "| " + " | ".join(["---"] * len(use)) + " |",
    ]

    for _, row in df.iterrows():
        vals: list[str] = []
        for c in use:
            val = row.get(c, "")
            if isinstance(val, (float, int, np.floating, np.integer)):
                vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")

    return lines


def summarize_pair_directional(pair_df: pd.DataFrame, pair_name: str) -> list[str]:
    lines: list[str] = []
    if pair_df.empty:
        return [f"No directional rows for `{pair_name}`.", ""]

    lines.append(f"### {pair_name}")
    lines.append("")

    focus = pair_df[
        pair_df["base_target"].astype(str).isin(
            [
                "path_family",
                "coupling_class",
                "coupling_class_full",
                "outcome_group",
                "coupled_outcome_group",
                "recovery_channel",
                "recovery_channel_boundedness_strict",
            ]
        )
    ].copy()

    focus = focus[
        focus["variant"].isin(
            ["full", "full_structural", "boundedness_strict", "no_direct_seam_no_grid"]
        )
    ]

    cols = [
        "base_target",
        "variant",
        "ba_baseline_to_cp3",
        "ba_cp3_to_baseline",
        "asymmetry_cp3_minus_baseline",
        "anti_shortcut",
    ]
    lines.extend(markdown_table(focus.sort_values(["focus_priority", "variant"]), cols))
    lines.append("")

    return lines


def infer_provisional_read(
    specificity: pd.DataFrame,
    anti_shortcut: pd.DataFrame,
    proxy_audit: pd.DataFrame,
) -> str:
    if specificity.empty:
        return "No asymmetry-specificity table was produced."

    coupled_recovery = specificity[
        specificity["base_target"].astype(str).isin(
            ["coupled_outcome_group", "recovery_channel"]
        )
    ].copy()

    anti_coupled_recovery = anti_shortcut[
        anti_shortcut["base_target"].astype(str).isin(
            ["coupled_outcome_group", "recovery_channel"]
        )
    ].copy()

    max_full = pd.to_numeric(
        coupled_recovery["asymmetry_specificity_cp2_minus_cp"], errors="coerce"
    ).max() if not coupled_recovery.empty else np.nan

    max_anti = pd.to_numeric(
        anti_coupled_recovery["asymmetry_specificity_cp2_minus_cp"], errors="coerce"
    ).max() if not anti_coupled_recovery.empty else np.nan

    proxy_warning = ""
    if not proxy_audit.empty and "endpoint_velocity_importance_share" in proxy_audit.columns:
        ok = proxy_audit[proxy_audit.get("status", "").astype(str).eq("ok")].copy()
        if not ok.empty:
            top_proxy = pd.to_numeric(ok["endpoint_velocity_importance_share"], errors="coerce").max()
            if np.isfinite(top_proxy) and top_proxy >= 0.25:
                proxy_warning = (
                    " However, endpoint/velocity proxy audit shows nontrivial proxy reliance in at least "
                    "one target, so a true recompute ablation is still required."
                )

    if np.isfinite(max_anti) and max_anti >= 0.20:
        return (
            "Cp3 directional asymmetry is strongest against Cp2 for coupled/recovery targets, "
            "and a reduced version survives the no_direct_seam_no_grid anti-shortcut slice. "
            "This weakens the simple broad-boundary/noisy-Cp3 explanation, because the same "
            "Cp3-transfer advantage does not appear uniformly against Cp."
            + proxy_warning
        )

    if np.isfinite(max_full) and max_full >= 0.25:
        return (
            "Cp3 directional asymmetry is strong against Cp2 for coupled/recovery targets, but the "
            "anti-shortcut evidence is weak or unavailable. Treat the result as relation-specific "
            "but still shortcut-sensitive."
            + proxy_warning
        )

    return (
        "Cp3 asymmetry specificity is weak or mixed. Treat the coupled-but-unresolved read as "
        "provisional pending feature-family and model-complexity stress tests."
        + proxy_warning
    )


def write_summary(
    cfg: Config,
    cp2_directional: pd.DataFrame,
    cp_directional: pd.DataFrame,
    specificity: pd.DataFrame,
    anti_shortcut: pd.DataFrame,
    focus: pd.DataFrame,
    proxy_audit: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# OBS-075 — Cp3 directional asymmetry v2",
        "",
        "## Scope",
        "",
        "OBS-075 tests whether Cp3 transfer asymmetry is specific to Cp2, or whether it is a generic artifact of Cp3 acting as a noisy / broad-boundary corpus.",
        "",
        "This script is artifact-first. It consumes existing OBS-073 model-score CSVs and does not recompute geometry, path families, coupling, outcomes, or recovery channels.",
        "",
        "v2 adds an explicit anti-shortcut specificity table and an optional endpoint/velocity proxy audit from permutation-importance artifacts.",
        "",
        "## Inputs",
        "",
        f"- Cp2/Cp3 OBS-073 scores: `{cfg.cp2_pair_scores}`",
        f"- Cp/Cp3 OBS-073 scores: `{cfg.cp_pair_scores}`",
        f"- Cp2/Cp3 permutation importances: `{cfg.cp2_pair_permutation}`",
        f"- Cp/Cp3 permutation importances: `{cfg.cp_pair_permutation}`",
        "",
        "## Directional asymmetry definition",
        "",
        "For each pair, the baseline corpus is compared against Cp3:",
        "",
        "- `baseline→Cp3`: model trained on baseline corpus and tested on Cp3.",
        "- `Cp3→baseline`: model trained on Cp3 and tested on baseline corpus.",
        "- `asymmetry Cp3-baseline = BA(Cp3→baseline) - BA(baseline→Cp3)`.",
        "",
        "Cp2-specific asymmetry is:",
        "",
        "`asymmetry_specificity = asymmetry(Cp3→Cp2 minus Cp2→Cp3) - asymmetry(Cp3→Cp minus Cp→Cp3)`",
        "",
        "Positive values mean Cp3's transfer advantage is stronger against Cp2 than against Cp.",
        "",
        "## Pair summaries",
        "",
    ]

    lines.extend(summarize_pair_directional(cp2_directional, cfg.cp2_pair_name))
    lines.extend(summarize_pair_directional(cp_directional, cfg.cp_pair_name))

    lines.extend(
        [
            "## Anti-shortcut specificity read",
            "",
            "This table keeps only `no_direct_seam_no_grid` rows. It is the most conservative directional-asymmetry slice available from OBS-073 model-score artifacts.",
            "",
        ]
    )

    anti_cols = [
        "target",
        "base_target",
        "variant",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "asymmetry_cp3_minus_cp2",
        "ba_cp_to_cp3",
        "ba_cp3_to_cp",
        "asymmetry_cp3_minus_cp",
        "asymmetry_specificity_cp2_minus_cp",
        "macro_f1_cp2_to_cp3",
        "macro_f1_cp3_to_cp2",
        "macro_f1_cp_to_cp3",
        "macro_f1_cp3_to_cp",
    ]
    lines.extend(markdown_table(anti_shortcut, anti_cols, max_rows=cfg.top_n_specificity))
    lines.append("")

    lines.extend(
        [
            "## Highest asymmetry-specificity rows",
            "",
        ]
    )

    spec_cols = [
        "target",
        "base_target",
        "variant",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "asymmetry_cp3_minus_cp2",
        "ba_cp_to_cp3",
        "ba_cp3_to_cp",
        "asymmetry_cp3_minus_cp",
        "asymmetry_specificity_cp2_minus_cp",
        "anti_shortcut",
        "no_direct_seam",
        "no_grid_location",
    ]
    lines.extend(markdown_table(specificity, spec_cols, max_rows=cfg.top_n_specificity))
    lines.append("")

    lines.extend(
        [
            "## Coupled / recovery focus",
            "",
        ]
    )

    focus_cols = [
        "target",
        "base_target",
        "variant",
        "asymmetry_cp3_minus_cp2",
        "asymmetry_cp3_minus_cp",
        "asymmetry_specificity_cp2_minus_cp",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "ba_cp_to_cp3",
        "ba_cp3_to_cp",
        "n_train_cp3_to_cp2",
        "n_train_cp3_to_cp",
        "anti_shortcut",
    ]
    lines.extend(markdown_table(focus, focus_cols, max_rows=cfg.top_n_focus))
    lines.append("")

    lines.extend(
        [
            "## Endpoint / velocity proxy audit",
            "",
            "This audit uses permutation-importance artifacts when available. It does not remove features or rerun models. It only checks whether top reported features are dominated by endpoint/velocity-like proxies.",
            "",
        ]
    )

    proxy_cols = [
        "pair",
        "target",
        "base_target",
        "variant",
        "status",
        "endpoint_velocity_importance_share",
        "endpoint_velocity_features_in_top_n",
        "direct_seam_importance_share",
        "grid_location_importance_share",
        "top_endpoint_velocity_features",
    ]
    lines.extend(markdown_table(proxy_audit, proxy_cols, max_rows=cfg.top_n_proxy_audit))
    lines.append("")

    lines.extend(
        [
            "## Provisional read",
            "",
            infer_provisional_read(specificity, anti_shortcut, proxy_audit),
            "",
            "## Audit",
            "",
        ]
    )

    if audit.empty:
        lines.append("No audit rows.")
    else:
        audit_summary = (
            audit.groupby(["pair"], as_index=False)
            .agg(
                n_targets=("target", "count"),
                n_baseline_to_cp3=("has_baseline_to_cp3", "sum"),
                n_cp3_to_baseline=("has_cp3_to_baseline", "sum"),
            )
        )
        lines.extend(
            markdown_table(
                audit_summary,
                ["pair", "n_targets", "n_baseline_to_cp3", "n_cp3_to_baseline"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This script summarizes directional transfer asymmetry; it does not prove causal mechanism.",
            "- Strong Cp2-specific asymmetry weakens, but does not eliminate, broad-boundary and distribution-smearing critiques.",
            "- The anti-shortcut table is stronger than the full/unblinded table, but it can still contain endpoint, velocity, path-length, and delta-style proxies.",
            "- The endpoint/velocity proxy audit is not a substitute for a true feature-removal transfer recomputation.",
            "- Boundedness-strict rows may have small train/test slices and should be interpreted with class-count caution.",
            "- The next defense layer should recompute transfer after removing `last_minus_first`, path length, n-step, arclength, chord, net-displacement, and tortuosity features.",
            "- The next defense layer should compare Random Forests against lower-complexity models such as shallow trees and logistic regression.",
            "",
        ]
    )

    (cfg.outdir / "obs075_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS-075 Cp3 directional asymmetry v2.")

    p.add_argument(
        "--cp2-pair-scores",
        default="outputs/comparisons/obs073_Cp2_vs_Cp3_v5_smoke/obs073_model_scores.csv",
        help="OBS-073 model_scores CSV for Cp2 vs Cp3.",
    )
    p.add_argument(
        "--cp-pair-scores",
        default="outputs/comparisons/obs073_Cp_vs_Cp3_v5_smoke/obs073_model_scores.csv",
        help="OBS-073 model_scores CSV for Cp vs Cp3.",
    )
    p.add_argument(
        "--cp2-pair-permutation",
        default="outputs/comparisons/obs073_Cp2_vs_Cp3_v5_smoke/obs073_feature_importance_permutation.csv",
        help="Optional OBS-073 permutation-importance CSV for Cp2 vs Cp3.",
    )
    p.add_argument(
        "--cp-pair-permutation",
        default="outputs/comparisons/obs073_Cp_vs_Cp3_v5_smoke/obs073_feature_importance_permutation.csv",
        help="Optional OBS-073 permutation-importance CSV for Cp vs Cp3.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/comparisons/obs075_cp3_directional_asymmetry_v2",
        help="Output directory.",
    )

    p.add_argument("--cp2-pair-name", default="Cp2_vs_Cp3")
    p.add_argument("--cp-pair-name", default="Cp_vs_Cp3")
    p.add_argument("--cp2-baseline", default="Cp2")
    p.add_argument("--cp-baseline", default="Cp")
    p.add_argument("--cp3-corpus", default="Cp3")

    p.add_argument("--top-n-specificity", type=int, default=25)
    p.add_argument("--top-n-focus", type=int, default=30)
    p.add_argument("--top-n-proxy-audit", type=int, default=20)

    return p.parse_args()


def none_if_blank(value: str | None) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"none", "null", "na"}:
        return None
    return Path(raw)


def main() -> None:
    args = parse_args()

    cfg = Config(
        cp2_pair_scores=Path(args.cp2_pair_scores),
        cp_pair_scores=Path(args.cp_pair_scores),
        outdir=Path(args.outdir),
        cp2_pair_name=args.cp2_pair_name,
        cp_pair_name=args.cp_pair_name,
        cp2_baseline=args.cp2_baseline,
        cp_baseline=args.cp_baseline,
        cp3_corpus=args.cp3_corpus,
        cp2_pair_permutation=none_if_blank(args.cp2_pair_permutation),
        cp_pair_permutation=none_if_blank(args.cp_pair_permutation),
        top_n_specificity=int(args.top_n_specificity),
        top_n_focus=int(args.top_n_focus),
        top_n_proxy_audit=int(args.top_n_proxy_audit),
    )

    ensure_dir(cfg.outdir)

    cp2_pair = PairSpec(
        name=cfg.cp2_pair_name,
        scores_csv=cfg.cp2_pair_scores,
        baseline_corpus=cfg.cp2_baseline,
        cp3_corpus=cfg.cp3_corpus,
        permutation_csv=cfg.cp2_pair_permutation,
    )
    cp_pair = PairSpec(
        name=cfg.cp_pair_name,
        scores_csv=cfg.cp_pair_scores,
        baseline_corpus=cfg.cp_baseline,
        cp3_corpus=cfg.cp3_corpus,
        permutation_csv=cfg.cp_pair_permutation,
    )

    cp2_directional, cp2_audit = build_directional_for_pair(cp2_pair)
    cp_directional, cp_audit = build_directional_for_pair(cp_pair)

    directional = pd.concat([cp2_directional, cp_directional], ignore_index=True)
    audit = pd.concat([cp2_audit, cp_audit], ignore_index=True)

    specificity = build_specificity(
        cp2_directional,
        cp_directional,
        cp2_pair_name=cfg.cp2_pair_name,
        cp_pair_name=cfg.cp_pair_name,
    )
    focus = build_target_focus(specificity)
    anti_shortcut = build_anti_shortcut_specificity(specificity)
    proxy_audit = build_proxy_audit(
        cp2_pair,
        cp_pair,
        top_n=cfg.top_n_proxy_audit,
    )

    directional.to_csv(cfg.outdir / "obs075_directional_asymmetry.csv", index=False)
    specificity.to_csv(cfg.outdir / "obs075_asymmetry_specificity.csv", index=False)
    anti_shortcut.to_csv(cfg.outdir / "obs075_anti_shortcut_specificity.csv", index=False)
    focus.to_csv(cfg.outdir / "obs075_target_focus.csv", index=False)
    proxy_audit.to_csv(cfg.outdir / "obs075_endpoint_velocity_proxy_audit.csv", index=False)
    audit.to_csv(cfg.outdir / "obs075_pair_audit.csv", index=False)

    write_summary(
        cfg=cfg,
        cp2_directional=cp2_directional,
        cp_directional=cp_directional,
        specificity=specificity,
        anti_shortcut=anti_shortcut,
        focus=focus,
        proxy_audit=proxy_audit,
        audit=audit,
    )

    print(cfg.outdir / "obs075_directional_asymmetry.csv")
    print(cfg.outdir / "obs075_asymmetry_specificity.csv")
    print(cfg.outdir / "obs075_anti_shortcut_specificity.csv")
    print(cfg.outdir / "obs075_target_focus.csv")
    print(cfg.outdir / "obs075_endpoint_velocity_proxy_audit.csv")
    print(cfg.outdir / "obs075_pair_audit.csv")
    print(cfg.outdir / "obs075_summary.md")


if __name__ == "__main__":
    main()
