#!/usr/bin/env python3
"""
obs082_rig_intervention_readiness_audit.py

OBS-082 — RIG Intervention-Readiness Audit

Purpose
-------
Audit OBS-081 relation × carrier records and determine which records are mature
enough to define conservative, testable intervention hypotheses within the
current PAM artifact lineage.

This script does NOT perform interventions.
This script does NOT establish control.
This script does NOT establish causality.
This script does NOT establish external generalization.

It computes an artifact-scoped readiness score from six dimensions:
    D1. Invariance strength
    D2. Failure localization
    D3. Repair specificity
    D4. Geometry sufficiency
    D5. Carrier convergence
    D6. Negative-control contrast

Design constraints
------------------
- File-first behavior.
- No invented demo data.
- Clear missing-artifact and missing-column diagnostics.
- Generated CSVs plus a markdown report.
- Normalize comparison / relation_name / task_name into canonical column: task.
- Do not merely restate rig_status.
- If invariance evidence comes only from categorical fallback over rig_status,
  readiness_class cannot exceed B.

Expected default inputs
-----------------------
outputs/rig_registry/rig_relation_registry.csv
outputs/rig_registry/rig_survival_matrix.csv
outputs/rig_registry/rig_failure_localization.csv
outputs/rig_registry/rig_geometry_needed_ladder.csv
outputs/rig_registry/rig_repair_recommendations.csv

Default outputs
---------------
outputs/rig_registry/obs082_intervention_readiness/obs082_input_manifest.csv
outputs/rig_registry/obs082_intervention_readiness/obs082_relation_readiness_scores.csv
outputs/rig_registry/obs082_intervention_readiness/obs082_candidate_intervention_hypotheses.csv
outputs/rig_registry/obs082_intervention_readiness/obs082_negative_control_contrasts.csv
outputs/rig_registry/obs082_intervention_readiness/obs082_failure_mode_inventory.csv
outputs/rig_registry/obs082_intervention_readiness/obs082_blockers.csv
outputs/rig_registry/obs082_intervention_readiness/obs082_report.md
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

WEIGHTS = {
    "invariance_strength_score": 0.25,
    "failure_localization_score": 0.15,
    "repair_specificity_score": 0.15,
    "geometry_sufficiency_score": 0.15,
    "carrier_convergence_score": 0.20,
    "negative_control_contrast_score": 0.10,
}

SCOPE_GUARDRAILS = [
    "within OBS-081 relation × carrier registry records",
    "within OBS-080-tested contract families",
    "within OBS-078a / OBS-079 / OBS-080 stability-core lineage",
    "within C / Cp2 / Cp3 regime comparisons where present",
    "within current repo-generated artifacts",
]

NON_CLAIMS = [
    "OBS-082 does not perform interventions.",
    "OBS-082 does not establish control.",
    "OBS-082 does not establish causality.",
    "OBS-082 does not establish external generalization.",
    "OBS-082 audits whether registry records are mature enough to define testable intervention hypotheses.",
]

STATUS_FALLBACK_MAP = {
    "stable_reusable_invariant": 0.90,
    "stable_reusable": 0.90,
    "stable": 0.85,
    "reusable_invariant": 0.80,
    "context_sensitive_reusable_invariant": 0.60,
    "context_sensitive": 0.55,
    "weak_or_failed_candidate": 0.25,
    "weak_failed": 0.25,
    "failed": 0.15,
    "weak": 0.25,
}

GEOMETRY_SCORE_MAP = {
    "level_1_compact_core_sufficient": 1.00,
    "compact_core_sufficient": 1.00,
    "level_1": 1.00,
    "level1": 1.00,
    "level_2_enriched_geometry_needed": 0.70,
    "enriched_geometry_needed": 0.70,
    "level_2": 0.70,
    "level2": 0.70,
    "level_3_contextual_geometry_needed": 0.45,
    "contextual_geometry_needed": 0.45,
    "level_3": 0.45,
    "level3": 0.45,
    "unresolved_geometry_need": 0.20,
    "unresolved": 0.20,
}

READINESS_OUTPUT_COLUMNS = [
    "relation_id", "task", "carrier", "feature_family", "rig_status",
    "readiness_score", "readiness_class", "readiness_blockers", "readiness_limiter", "score_basis",
    "invariance_strength_score", "failure_localization_score", "repair_specificity_score",
    "geometry_sufficiency_score", "carrier_convergence_score", "negative_control_contrast_score",
    "survival_rate", "tested_count", "survival_count", "failure_count",
    "n_carriers_tested", "n_carriers_survived", "carrier_survival_rate",
    "dominant_failure_mode", "failure_localization_basis",
    "repair_recommendation", "repair_type", "repair_specificity_basis",
    "geometry_level", "minimal_sufficient_geometry", "geometry_sufficiency_basis",
    "negative_control_group", "negative_control_reference_score", "negative_control_contrast",
    "audit_notes",
]

CANDIDATE_OUTPUT_COLUMNS = [
    "hypothesis_id", "relation_id", "task", "carrier", "feature_family",
    "readiness_class", "readiness_score", "readiness_limiter", "candidate_hypothesis",
    "intervention_axis", "expected_direction", "required_geometry_level",
    "required_carrier_evidence", "supporting_dimensions", "limiting_dimensions",
    "required_next_test", "falsification_condition", "scope_condition",
]

NEGATIVE_CONTROL_OUTPUT_COLUMNS = [
    "relation_id", "task", "carrier", "feature_family", "negative_control_type",
    "negative_control_relation_id", "record_score", "control_score", "contrast",
    "contrast_passed", "contrast_basis",
]

FAILURE_OUTPUT_COLUMNS = [
    "relation_id", "task", "carrier", "feature_family", "failure_mode",
    "failure_location", "failure_feature_family", "failure_scale_band", "failure_transform",
    "localized_failure_count", "diffuse_failure_count", "failure_entropy", "readiness_impact",
]

MANIFEST_COLUMNS = [
    "input_name", "input_path", "exists", "rows", "columns",
    "required_columns_present", "missing_columns", "used_for_dimensions", "notes",
]

BLOCKER_OUTPUT_COLUMNS = [
    "relation_id", "task", "carrier", "feature_family",
    "readiness_class", "readiness_score", "blocker", "blocker_type", "readiness_impact",
]

@dataclass
class InputSpec:
    name: str
    path: Path
    required_any: tuple[str, ...]
    required_for_full: tuple[str, ...]
    used_for: str
    optional: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBS-082 — RIG Intervention-Readiness Audit")
    parser.add_argument("--registry", default="outputs/rig_registry/rig_relation_registry.csv")
    parser.add_argument("--survival", default="outputs/rig_registry/rig_survival_matrix.csv")
    parser.add_argument("--failure-localization", default="outputs/rig_registry/rig_failure_localization.csv")
    parser.add_argument("--geometry-ladder", default="outputs/rig_registry/rig_geometry_needed_ladder.csv")
    parser.add_argument("--repair-recommendations", default="outputs/rig_registry/rig_repair_recommendations.csv")
    parser.add_argument("--outdir", default="outputs/rig_registry/obs082_intervention_readiness")
    parser.add_argument("--adequate-tested-count", type=int, default=3)
    parser.add_argument("--contrast-scale", type=float, default=0.50)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def clean_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def norm_token(value: Any) -> str:
    text = clean_name(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def as_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return default if math.isnan(out) else out
    except Exception:
        return default


def clip01(value: Any) -> float:
    val = as_float(value)
    if math.isnan(val):
        return np.nan
    return float(max(0.0, min(1.0, val)))


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def ensure_task_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "task" in df.columns:
        df["task"] = df["task"].map(clean_name)
        return df
    src = first_existing_col(df, ["comparison", "relation_name", "task_name", "pairwise_task", "regime_comparison", "label_task"])
    df["task"] = df[src].map(clean_name) if src else ""
    return df


def ensure_relation_id(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_task_column(df).copy()
    if "relation_id" in df.columns:
        df["relation_id"] = df["relation_id"].map(clean_name)
        return df
    parts = [c for c in ["task", "relation_name", "carrier", "feature_family", "contract_family"] if c in df.columns]
    if not parts:
        df["relation_id"] = [f"relation_{i:04d}" for i in range(len(df))]
        return df
    def build(row: pd.Series) -> str:
        vals = [norm_token(row.get(c, "")) for c in parts]
        vals = [v for v in vals if v]
        return "__".join(vals) if vals else ""
    df["relation_id"] = df.apply(build, axis=1)
    empty = df["relation_id"].eq("")
    if empty.any():
        df.loc[empty, "relation_id"] = [f"relation_{i:04d}" for i in df.index[empty]]
    return df


def normalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_relation_id(df).copy()
    alias_map = {
        "carrier": ["carrier", "carrier_family", "artifact_carrier"],
        "feature_family": [
            "feature_family", "feature_contract", "feature_set", "feature_group",
            "carrier_feature_family", "carrier",
        ],
        "rig_status": ["rig_status", "status", "registry_status"],
        "survival_rate": ["survival_rate", "pass_rate", "contract_survival_rate"],
        "tested_count": ["tested_count", "n_tested", "num_tests", "contract_count", "n_contracts", "n_survival_rows"],
        "survival_count": ["survival_count", "n_survived", "pass_count", "passed_count"],
        "failure_count": ["failure_count", "n_failed", "fail_count", "failed_count"],
        "repair_recommendation": ["repair_recommendation", "recommendation", "repair", "next_repair"],
        "repair_type": ["repair_type", "recommendation_type"],
        "geometry_level": [
            "geometry_level", "minimal_geometry_level", "geometry_needed",
            "task_geometry_needed_level", "task_geometry_needed_label",
        ],
        "minimal_sufficient_geometry": [
            "minimal_sufficient_geometry", "geometry_needed", "minimal_geometry",
            "task_geometry_needed_label",
        ],
        "failure_mode": ["failure_mode", "dominant_failure_mode", "failure_type"],
        "failure_location": ["failure_location", "contract_family", "contract_name"],
    }
    for canonical, aliases in alias_map.items():
        if canonical not in df.columns:
            src = first_existing_col(df, aliases)
            if src:
                df[canonical] = df[src]
    for col in ["carrier", "feature_family", "rig_status"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(clean_name)
    return df


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return normalize_common_columns(pd.read_csv(path))


def manifest_row(spec: InputSpec, df: pd.DataFrame | None) -> dict[str, Any]:
    cols = list(df.columns) if df is not None else []
    present = [c for c in spec.required_any if c in cols]
    missing = [c for c in spec.required_for_full if c not in cols]
    return {
        "input_name": spec.name,
        "input_path": str(spec.path),
        "exists": bool(df is not None),
        "rows": int(len(df)) if df is not None else 0,
        "columns": ";".join(cols),
        "required_columns_present": ";".join(present),
        "missing_columns": ";".join(missing),
        "used_for_dimensions": spec.used_for,
        "notes": "" if df is not None else "missing_artifact",
    }


def aggregate_by_relation(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["relation_id"])
    df = normalize_common_columns(df)
    numeric = [c for c in df.columns if c != "relation_id" and pd.api.types.is_numeric_dtype(df[c])]
    agg: dict[str, Any] = {c: "mean" for c in numeric}
    for c in df.columns:
        if c != "relation_id" and c not in numeric:
            agg[c] = lambda s: next((clean_name(x) for x in s if clean_name(x)), "")
    out = normalize_common_columns(df.groupby("relation_id", dropna=False).agg(agg).reset_index())

    # OBS-081 v2 failure-localization rows often expose failure_type/status/score
    # rather than localized/diffuse counts. Preserve concentration diagnostics so
    # D2 can score relation-level rows without inventing failure boundaries.
    fcol = first_existing_col(df, ["failure_type", "failure_mode"])
    if fcol:
        def dominant(series: pd.Series) -> str:
            vals = [clean_name(x) for x in series if clean_name(x)]
            if not vals:
                return ""
            return pd.Series(vals).value_counts().index[0]
        conc = df.groupby("relation_id", dropna=False)[fcol].agg(
            n_distinct_failure_types=lambda s: len({clean_name(x) for x in s if clean_name(x)}),
            dominant_failure_mode=dominant,
        ).reset_index()
        out = out.merge(conc, on="relation_id", how="left", suffixes=("", "_conc"))
        if "dominant_failure_mode_conc" in out.columns:
            out["dominant_failure_mode"] = out["dominant_failure_mode"].where(
                out["dominant_failure_mode"].map(clean_name).ne(""),
                out["dominant_failure_mode_conc"],
            )
            out = out.drop(columns=["dominant_failure_mode_conc"])
    return normalize_common_columns(out)


def merge_artifacts(registry: pd.DataFrame, survival: pd.DataFrame | None, failure: pd.DataFrame | None, geometry: pd.DataFrame | None, repair: pd.DataFrame | None) -> pd.DataFrame:
    base = normalize_common_columns(registry).copy().loc[:, lambda d: ~d.columns.duplicated()]
    for prefix, extra in [
        ("surv", aggregate_by_relation(survival)),
        ("fail", aggregate_by_relation(failure)),
        ("geom", aggregate_by_relation(geometry)),
        ("repair", aggregate_by_relation(repair)),
    ]:
        if extra.empty or "relation_id" not in extra.columns:
            continue
        extra = extra.copy().loc[:, lambda d: ~d.columns.duplicated()]
        # Prefix every joined artifact column to preserve provenance. This avoids
        # mistaking OBS-081 failure-localization score/threshold columns for
        # survival-matrix evidence in D1.
        extra = extra.rename(columns={c: f"{prefix}_{c}" for c in extra.columns if c != "relation_id"})
        base = base.merge(extra, on="relation_id", how="left")
    return normalize_common_columns(base)


def row_first(row: pd.Series, candidates: Iterable[str], default: Any = "") -> Any:
    for col in candidates:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and clean_name(val) != "":
                return val
    return default


def append_blocker(existing: Any, blocker: str) -> str:
    vals = [v for v in clean_name(existing).split(";") if v]
    if blocker not in vals:
        vals.append(blocker)
    return ";".join(vals)


def has_blocker(blockers: Any, names: Iterable[str]) -> bool:
    vals = set(clean_name(blockers).split(";"))
    return any(n in vals for n in names)


def compute_invariance(row: pd.Series) -> tuple[float, str, float, float, float, list[str]]:
    """Compute D1 with OBS-081 v2 evidence before rig_status fallback.

    Fallback order:
    1. survival_rate
    2. survival_count / tested_count
    3. 1 - failure_count / tested_count
    4. obs080d_carrier_mean_ba
    5. obs080c_carrier_ba
    6. mean_survival_score
    7. min_survival_score
    8. aggregate survival_matrix score/pass rows if available
    9. rig_status categorical_fallback
    """
    blockers: list[str] = []
    survival_rate = as_float(row_first(row, ["survival_rate", "surv_survival_rate"]))
    tested_count = as_float(row_first(row, ["tested_count", "surv_tested_count", "n_survival_rows", "surv_n_survival_rows"]))
    survival_count = as_float(row_first(row, ["survival_count", "surv_survival_count"]))
    failure_count = as_float(row_first(row, ["failure_count", "surv_failure_count"]))

    if not math.isnan(survival_rate):
        rate = clip01(survival_rate)
        if math.isnan(failure_count) and not math.isnan(tested_count):
            failure_count = max(0.0, tested_count * (1.0 - rate))
        return rate, "survival_rate", rate, tested_count, failure_count, blockers
    if not math.isnan(survival_count) and not math.isnan(tested_count) and tested_count > 0:
        rate = clip01(survival_count / tested_count)
        if math.isnan(failure_count):
            failure_count = max(0.0, tested_count - survival_count)
        return rate, "survival_count_over_tested_count", rate, tested_count, failure_count, blockers
    if not math.isnan(failure_count) and not math.isnan(tested_count) and tested_count > 0:
        rate = clip01(1.0 - failure_count / tested_count)
        return rate, "one_minus_failure_count_over_tested_count", rate, tested_count, failure_count, blockers

    for col in [
        "obs080d_carrier_mean_ba", "surv_obs080d_carrier_mean_ba",
        "obs080c_carrier_ba", "surv_obs080c_carrier_ba",
        "mean_survival_score", "surv_mean_survival_score",
        "min_survival_score", "surv_min_survival_score",
    ]:
        val = as_float(row.get(col, np.nan))
        if not math.isnan(val):
            return clip01(val), col.replace("surv_", ""), clip01(val), tested_count, failure_count, blockers

    # Aggregated survival_matrix fallback. aggregate_by_relation keeps numeric
    # score/pass columns as relation means, so a mean pass indicator or mean score
    # can serve as direct evidence before categorical status is used.
    for col in [
        "surv_passed", "surv_pass", "surv_passed_flag", "surv_passed_bool",
        "surv_score", "surv_ba", "surv_balanced_accuracy", "surv_metric_score",
    ]:
        val = as_float(row.get(col, np.nan))
        if not math.isnan(val):
            return clip01(val), f"aggregate_survival_matrix_{col.replace('surv_', '')}", clip01(val), tested_count, failure_count, blockers

    status = norm_token(row_first(row, ["rig_status", "surv_rig_status", "status"], ""))
    score = STATUS_FALLBACK_MAP.get(status, np.nan)
    if math.isnan(score):
        blockers.append("missing_invariance_basis")
        return np.nan, "missing", np.nan, tested_count, failure_count, blockers
    return score, "categorical_fallback", score, tested_count, failure_count, blockers


def compute_failure_localization(row: pd.Series, tested_count: float, failure_count: float, adequate_tested_count: int) -> tuple[float, str, str, list[str]]:
    blockers: list[str] = []
    localized = as_float(row_first(row, ["localized_failure_count", "fail_localized_failure_count"]))
    diffuse = as_float(row_first(row, ["diffuse_failure_count", "fail_diffuse_failure_count"]))
    entropy = as_float(row_first(row, ["failure_entropy", "fail_failure_entropy"]))
    dominant = clean_name(row_first(row, [
        "dominant_failure_mode", "fail_dominant_failure_mode",
        "failure_mode", "fail_failure_mode", "failure_type", "fail_failure_type",
    ], ""))
    if not math.isnan(localized) and not math.isnan(diffuse) and localized + diffuse > 0:
        return clip01(localized / (localized + diffuse)), "localized_over_total_failure_count", dominant, blockers
    if not math.isnan(entropy):
        return (clip01(1.0 - entropy) if 0 <= entropy <= 1 else clip01(1.0 / (1.0 + entropy))), "entropy_fallback", dominant, blockers
    if not math.isnan(failure_count) and failure_count == 0 and not math.isnan(tested_count) and tested_count >= adequate_tested_count:
        return 0.70, "no_observed_failures_robustness_without_boundary_evidence", dominant or "no_observed_failures", blockers

    # OBS-081 v2 failure localization can expose rows with failure_type,
    # contract_family/contract_name, status, score, threshold, and margin rather
    # than explicit localized/diffuse counts. Use distinct failure-type
    # concentration as a conservative localization proxy.
    n_types = as_float(row_first(row, ["n_distinct_failure_types", "fail_n_distinct_failure_types"]))
    if not math.isnan(n_types) and n_types > 0:
        if n_types == 1:
            return 0.80, "distinct_failure_type_concentration_one_dominant_type", dominant, blockers
        if 2 <= n_types <= 3:
            return 0.60, "distinct_failure_type_concentration_2_to_3_types", dominant, blockers
        return 0.35, "distinct_failure_type_concentration_more_than_3_types", dominant, blockers

    failure_mode = clean_name(row_first(row, ["failure_mode", "fail_failure_mode", "failure_type", "fail_failure_type"], ""))
    failure_location = clean_name(row_first(row, ["failure_location", "fail_failure_location", "contract_family", "fail_contract_family", "contract_name", "fail_contract_name"], ""))
    status = norm_token(row_first(row, ["fail_status", "status"], ""))
    score = as_float(row_first(row, ["fail_score", "score"]))
    threshold = as_float(row_first(row, ["fail_threshold", "threshold"]))
    margin = as_float(row_first(row, ["fail_margin", "margin"]))
    has_failure_row_evidence = bool(failure_mode or failure_location or status or not math.isnan(score) or not math.isnan(threshold) or not math.isnan(margin))
    if has_failure_row_evidence:
        if failure_mode:
            return 0.80, "single_failure_type_or_mode_fallback", failure_mode, blockers
        if failure_location:
            return 0.60, "contract_location_failure_evidence_fallback", failure_location, blockers
        return 0.35, "untyped_failure_row_evidence_fallback", dominant, blockers

    blockers.append("missing_failure_localization")
    return np.nan, "missing", dominant, blockers


def compute_repair_specificity(row: pd.Series) -> tuple[float, str, str, str, list[str]]:
    blockers: list[str] = []
    rec = clean_name(row_first(row, ["repair_recommendation", "repair_repair_recommendation", "recommendation", "repair_recommendation_text"], ""))
    typ = clean_name(row_first(row, ["repair_type", "repair_repair_type"], ""))
    specificity = as_float(row_first(row, ["repair_specificity", "repair_repair_specificity"]))
    confidence = as_float(row_first(row, ["repair_confidence", "repair_repair_confidence"]))
    specific_fields = [
        "recommended_feature_family", "repair_recommended_feature_family",
        "recommended_geometry_layer", "repair_recommended_geometry_layer",
        "recommended_scale_band", "repair_recommended_scale_band",
        "recommended_carrier_extension", "repair_recommended_carrier_extension",
    ]
    has_specific = any(clean_name(row.get(c, "")) for c in specific_fields if c in row.index)
    if not math.isnan(specificity):
        return clip01(specificity), "repair_specificity_column", rec, typ, blockers
    if has_specific and rec and not math.isnan(confidence):
        return 1.00, "specific_repair_with_confidence", rec, typ, blockers
    if has_specific and rec:
        return 0.80, "specific_repair_without_confidence", rec, typ, blockers
    if rec:
        if any(tok in norm_token(rec) for tok in ["inspect", "further", "review", "unknown", "generic"]):
            return 0.25, "generic_repair_recommendation", rec, typ, blockers
        return 0.50, "general_repair_recommendation", rec, typ, blockers
    blockers.append("missing_repair_specificity")
    return 0.00, "missing", rec, typ, blockers


def compute_geometry_sufficiency(row: pd.Series) -> tuple[float, str, str, str, list[str]]:
    blockers: list[str] = []
    compact = row_first(row, ["compact_core_sufficient", "geom_compact_core_sufficient", "level_1_sufficient", "geom_level_1_sufficient"], "")
    if clean_name(compact).lower() in {"true", "1", "yes", "y"} or compact is True:
        return 1.00, "compact_core_sufficient_boolean", "level_1_compact_core_sufficient", "compact_core", blockers
    level = clean_name(row_first(row, ["geometry_level", "geom_geometry_level", "geometry_needed", "geom_geometry_needed"], ""))
    minimal = clean_name(row_first(row, ["minimal_sufficient_geometry", "geom_minimal_sufficient_geometry", "minimal_geometry", "geom_minimal_geometry"], ""))
    token = norm_token(level or minimal)
    if token in GEOMETRY_SCORE_MAP:
        return GEOMETRY_SCORE_MAP[token], "geometry_level_mapping", level, minimal, blockers
    gain = as_float(row_first(row, ["geometry_gain", "geom_geometry_gain"]))
    if not math.isnan(gain):
        return 1.0 - clip01(gain), "one_minus_geometry_gain_fallback", level, minimal, blockers
    blockers.append("missing_geometry_sufficiency")
    return np.nan, "missing", level, minimal, blockers


def compute_carrier_convergence(df: pd.DataFrame, row: pd.Series) -> tuple[float, int, int, float, list[str]]:
    blockers: list[str] = []
    n_tested = as_float(row_first(row, ["n_carriers_tested", "surv_n_carriers_tested"]))
    n_survived = as_float(row_first(row, ["n_carriers_survived", "surv_n_carriers_survived"]))
    carrier_rate = as_float(row_first(row, ["carrier_survival_rate", "surv_carrier_survival_rate", "carrier_agreement_score", "surv_carrier_agreement_score"]))
    if not math.isnan(carrier_rate):
        score = clip01(carrier_rate)
        if not math.isnan(n_tested) and n_tested == 1:
            score = min(score, 0.55)
        return score, int(n_tested) if not math.isnan(n_tested) else 0, int(n_survived) if not math.isnan(n_survived) else 0, carrier_rate, blockers
    if not math.isnan(n_tested) and not math.isnan(n_survived) and n_tested > 0:
        rate = clip01(n_survived / n_tested)
        return min(rate, 0.55) if n_tested == 1 else rate, int(n_tested), int(n_survived), rate, blockers
    task = clean_name(row.get("task", ""))
    carrier = clean_name(row.get("carrier", ""))
    if "carrier" in df.columns and carrier:
        peers = df[df["task"].map(clean_name) == task] if task else df[df["relation_id"] == row.get("relation_id")]
        carriers = sorted({clean_name(x) for x in peers.get("carrier", pd.Series(dtype=str)) if clean_name(x)})
        n = len(carriers)
        if n > 1:
            survived = sum(1 for _, prow in peers.iterrows() if as_float(compute_invariance(prow)[0]) >= 0.60)
            score = clip01(survived / n)
            return score, n, survived, score, blockers
        if n == 1:
            inv = compute_invariance(row)[0]
            score = min(clip01(inv) if not math.isnan(inv) else 0.30, 0.55)
            return score, 1, int(score >= 0.50), score, blockers
    blockers.append("missing_carrier_basis")
    return np.nan, 0, 0, np.nan, blockers


def compute_readiness_score(row: pd.Series) -> float:
    weighted = 0.0
    total = 0.0
    missing = 0
    for col, weight in WEIGHTS.items():
        val = as_float(row.get(col))
        if math.isnan(val):
            missing += 1
            continue
        weighted += weight * clip01(val)
        total += weight
    if total == 0:
        return np.nan
    return clip01((weighted / total) * max(0.0, 1.0 - 0.08 * missing))


def identify_negative_controls(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=NEGATIVE_CONTROL_OUTPUT_COLUMNS)
    weak_mask = (scored["invariance_strength_score"].fillna(0) < 0.45) | scored["rig_status"].map(lambda x: any(t in norm_token(x) for t in ["weak", "failed"]))
    context_mask = scored["rig_status"].map(lambda x: "context" in norm_token(x))
    sparse_mask = scored["n_carriers_tested"].fillna(0) <= 1
    geometry_mask = scored["geometry_sufficiency_score"].fillna(0) < 0.50
    rows = []
    for _, row in scored.iterrows():
        record_score = as_float(row.get("invariance_strength_score"))
        for typ, mask in [
            ("weak_status_or_low_survival", weak_mask),
            ("context_sensitive", context_mask),
            ("carrier_sparsity", sparse_mask),
            ("geometry_ambiguity", geometry_mask),
        ]:
            controls = scored[mask & (scored["relation_id"] != row["relation_id"])]
            if controls.empty or math.isnan(record_score) or not controls["invariance_strength_score"].notna().any():
                continue
            control_score = float(controls["invariance_strength_score"].dropna().median())
            contrast = record_score - control_score
            rows.append({
                "relation_id": row.get("relation_id", ""),
                "task": row.get("task", ""),
                "carrier": row.get("carrier", ""),
                "feature_family": row.get("feature_family", ""),
                "negative_control_type": typ,
                "negative_control_relation_id": "median_control_group",
                "record_score": record_score,
                "control_score": control_score,
                "contrast": contrast,
                "contrast_passed": bool(contrast >= 0.25),
                "contrast_basis": "record_invariance_minus_control_group_median",
            })
    out = pd.DataFrame(rows)
    return out[NEGATIVE_CONTROL_OUTPUT_COLUMNS] if not out.empty else pd.DataFrame(columns=NEGATIVE_CONTROL_OUTPUT_COLUMNS)


def add_negative_control_scores(scored: pd.DataFrame, contrasts: pd.DataFrame, contrast_scale: float) -> pd.DataFrame:
    scored = scored.copy()
    for col in ["negative_control_group", "negative_control_reference_score", "negative_control_contrast", "negative_control_contrast_score"]:
        scored[col] = np.nan if col != "negative_control_group" else ""
    if contrasts.empty:
        scored["readiness_blockers"] = scored["readiness_blockers"].map(lambda x: append_blocker(x, "missing_negative_control_basis"))
        return scored
    grouped = contrasts.groupby("relation_id", dropna=False).agg(
        negative_control_group=("negative_control_type", lambda s: ";".join(sorted(set(map(clean_name, s))))),
        negative_control_reference_score=("control_score", "median"),
        negative_control_contrast=("contrast", "max"),
    ).reset_index()
    scored = scored.merge(grouped, on="relation_id", how="left", suffixes=("", "_nc"))
    for col in ["negative_control_group", "negative_control_reference_score", "negative_control_contrast"]:
        alt = f"{col}_nc"
        if alt in scored.columns:
            scored[col] = scored[alt].combine_first(scored[col])
            scored = scored.drop(columns=[alt])
    scored["negative_control_contrast_score"] = scored["negative_control_contrast"].map(lambda x: clip01(as_float(x) / contrast_scale) if not math.isnan(as_float(x)) and contrast_scale > 0 else np.nan)
    missing = scored["negative_control_contrast_score"].isna()
    scored.loc[missing, "readiness_blockers"] = scored.loc[missing, "readiness_blockers"].map(lambda x: append_blocker(x, "missing_negative_control_basis"))
    return scored


def derive_readiness_limiters(row: pd.Series) -> str:
    """Return present-but-insufficient readiness limiters.

    Blockers are reserved for missing, schema, or unjoinable evidence. Limiters
    are computed from available dimension scores and describe why a scorable
    record does not reach a stronger readiness class.
    """
    limiters: list[str] = []
    inv = as_float(row.get("invariance_strength_score"))
    carrier = as_float(row.get("carrier_convergence_score"))
    neg = as_float(row.get("negative_control_contrast_score"))
    fail = as_float(row.get("failure_localization_score"))
    repair = as_float(row.get("repair_specificity_score"))

    if not math.isnan(inv) and inv < 0.60:
        limiters.append("low_invariance_strength")
    if not math.isnan(carrier) and carrier < 0.65:
        limiters.append("low_carrier_convergence")
    if not math.isnan(neg) and neg < 0.50:
        limiters.append("weak_negative_control_contrast")
    if not math.isnan(fail) and fail < 0.60:
        limiters.append("diffuse_failure_localization")
    if not math.isnan(repair) and repair < 0.60:
        limiters.append("generic_repair_specificity")
    return ";".join(sorted(set(limiters)))


def assign_readiness_class(row: pd.Series) -> str:
    score = as_float(row.get("readiness_score"))
    inv = as_float(row.get("invariance_strength_score"))
    carrier = as_float(row.get("carrier_convergence_score"))
    neg = as_float(row.get("negative_control_contrast_score"))
    fail = as_float(row.get("failure_localization_score"))
    repair = as_float(row.get("repair_specificity_score"))
    blockers = clean_name(row.get("readiness_blockers", ""))

    fatal_blockers = [
        "missing_required_registry",
        "unjoinable_relation_keys",
        "no_survival_or_failure_evidence",
        "missing_invariance_basis",
    ]
    if has_blocker(blockers, fatal_blockers) or math.isnan(score):
        return "X: blocked / insufficient artifact support"

    has_fatal = has_blocker(blockers, fatal_blockers)
    class_a = (
        score >= 0.80
        and inv >= 0.75
        and carrier >= 0.65
        and neg >= 0.50
        and not has_fatal
        and (fail >= 0.60 or repair >= 0.60)
    )
    class_b = (
        score >= 0.65
        and inv >= 0.60
        and carrier >= 0.65
        and neg >= 0.25
        and not has_fatal
    )

    if class_a:
        klass = "A: hypothesis-ready"
    elif class_b:
        klass = "B: candidate-ready"
    elif score >= 0.45 or (inv >= 0.75 and (math.isnan(neg) or neg < 0.25)):
        klass = "C: diagnostic-only"
    elif score < 0.45:
        klass = "D: registry-only"
    else:
        klass = "X: blocked / insufficient artifact support"

    if clean_name(row.get("score_basis", "")) == "categorical_fallback" and klass.startswith("A:"):
        klass = "B: candidate-ready"
    return klass


def score_records(df: pd.DataFrame, adequate_tested_count: int, contrast_scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in df.iterrows():
        blockers: list[str] = []
        inv, inv_basis, survival_rate, tested_count, failure_count, bs = compute_invariance(row); blockers += bs
        fail, fail_basis, dominant, bs = compute_failure_localization(row, tested_count, failure_count, adequate_tested_count); blockers += bs
        repair, repair_basis, repair_rec, repair_type, bs = compute_repair_specificity(row); blockers += bs
        geom, geom_basis, geom_level, minimal_geom, bs = compute_geometry_sufficiency(row); blockers += bs
        carrier, nct, ncs, carrier_rate, bs = compute_carrier_convergence(df, row); blockers += bs
        survival_count = as_float(row_first(row, ["survival_count", "surv_survival_count"]))
        if math.isnan(survival_count) and not math.isnan(survival_rate) and not math.isnan(tested_count):
            survival_count = survival_rate * tested_count
        if math.isnan(failure_count) and not math.isnan(tested_count) and not math.isnan(survival_count):
            failure_count = tested_count - survival_count
        rows.append({
            "relation_id": clean_name(row.get("relation_id", "")),
            "task": clean_name(row.get("task", "")),
            "carrier": clean_name(row.get("carrier", "")),
            "feature_family": clean_name(row.get("feature_family", "")),
            "rig_status": clean_name(row.get("rig_status", "")),
            "readiness_score": np.nan,
            "readiness_class": "",
            "readiness_blockers": ";".join(sorted(set(blockers))),
            "readiness_limiter": "",
            "score_basis": inv_basis,
            "invariance_strength_score": inv,
            "failure_localization_score": fail,
            "repair_specificity_score": repair,
            "geometry_sufficiency_score": geom,
            "carrier_convergence_score": carrier,
            "negative_control_contrast_score": np.nan,
            "survival_rate": survival_rate,
            "tested_count": tested_count,
            "survival_count": survival_count,
            "failure_count": failure_count,
            "n_carriers_tested": nct,
            "n_carriers_survived": ncs,
            "carrier_survival_rate": carrier_rate,
            "dominant_failure_mode": dominant,
            "failure_localization_basis": fail_basis,
            "repair_recommendation": repair_rec,
            "repair_type": repair_type,
            "repair_specificity_basis": repair_basis,
            "geometry_level": geom_level,
            "minimal_sufficient_geometry": minimal_geom,
            "geometry_sufficiency_basis": geom_basis,
            "negative_control_group": "",
            "negative_control_reference_score": np.nan,
            "negative_control_contrast": np.nan,
            "audit_notes": "artifact_scoped_intervention_readiness_audit",
        })
    scored = pd.DataFrame(rows)
    if scored.empty:
        return pd.DataFrame(columns=READINESS_OUTPUT_COLUMNS), pd.DataFrame(columns=NEGATIVE_CONTROL_OUTPUT_COLUMNS)
    scored["readiness_score"] = scored.apply(compute_readiness_score, axis=1)
    contrasts = identify_negative_controls(scored)
    scored = add_negative_control_scores(scored, contrasts, contrast_scale)
    scored["readiness_score"] = scored.apply(compute_readiness_score, axis=1)
    scored["readiness_limiter"] = scored.apply(derive_readiness_limiters, axis=1)
    scored["readiness_class"] = scored.apply(assign_readiness_class, axis=1)
    for col in READINESS_OUTPUT_COLUMNS:
        if col not in scored.columns:
            scored[col] = np.nan if any(k in col for k in ["score", "count", "rate"]) else ""
    return scored[READINESS_OUTPUT_COLUMNS], contrasts


def build_failure_inventory(failure: pd.DataFrame | None, scored: pd.DataFrame) -> pd.DataFrame:
    if failure is None or failure.empty:
        return pd.DataFrame(columns=FAILURE_OUTPUT_COLUMNS)
    df = normalize_common_columns(failure).copy()
    out = pd.DataFrame({
        "relation_id": df.get("relation_id", ""),
        "task": df.get("task", ""),
        "carrier": df.get("carrier", ""),
        "feature_family": df.get("feature_family", ""),
        "failure_mode": df.get("failure_mode", df.get("failure_type", df.get("dominant_failure_mode", ""))),
        "failure_location": df.get("failure_location", df.get("contract_family", df.get("contract_name", ""))),
        "failure_feature_family": df.get("failure_feature_family", ""),
        "failure_scale_band": df.get("failure_scale_band", ""),
        "failure_transform": df.get("failure_transform", ""),
        "localized_failure_count": df.get("localized_failure_count", np.nan),
        "diffuse_failure_count": df.get("diffuse_failure_count", np.nan),
        "failure_entropy": df.get("failure_entropy", np.nan),
    })
    impact = scored[["relation_id", "failure_localization_score", "failure_localization_basis"]].copy()
    impact["readiness_impact"] = impact.apply(lambda r: f"failure_localization_score={r['failure_localization_score']};basis={r['failure_localization_basis']}", axis=1)
    out = out.merge(impact[["relation_id", "readiness_impact"]], on="relation_id", how="left")
    for col in FAILURE_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[FAILURE_OUTPUT_COLUMNS]


def blocker_type(blocker: str) -> str:
    token = norm_token(blocker)
    if "missing" in token:
        return "missing_evidence"
    if "unjoinable" in token or "schema" in token:
        return "schema_or_join"
    if "no_survival" in token or "no_tested" in token:
        return "insufficient_test_evidence"
    return "readiness_constraint"


def blocker_impact(blocker: str) -> str:
    token = norm_token(blocker)
    if token in {"missing_invariance_basis", "missing_required_registry", "unjoinable_relation_keys", "no_survival_or_failure_evidence"}:
        return "fatal_or_class_x"
    if token in {"missing_negative_control_basis", "missing_carrier_basis"}:
        return "prevents_class_a"
    if token in {"missing_failure_localization", "missing_geometry_sufficiency"}:
        return "reduces_dimension_score_and_may_limit_class"
    if token == "missing_repair_specificity":
        return "sets_repair_specificity_to_zero_and_may_limit_class"
    return "documented_blocker"


def build_blockers(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scored.empty:
        return pd.DataFrame(columns=BLOCKER_OUTPUT_COLUMNS)
    for _, row in scored.iterrows():
        blockers = [b for b in clean_name(row.get("readiness_blockers", "")).split(";") if b]
        for blocker in blockers:
            rows.append({
                "relation_id": row.get("relation_id", ""),
                "task": row.get("task", ""),
                "carrier": row.get("carrier", ""),
                "feature_family": row.get("feature_family", ""),
                "readiness_class": row.get("readiness_class", ""),
                "readiness_score": row.get("readiness_score", np.nan),
                "blocker": blocker,
                "blocker_type": blocker_type(blocker),
                "readiness_impact": blocker_impact(blocker),
            })
    out = pd.DataFrame(rows)
    return out[BLOCKER_OUTPUT_COLUMNS] if not out.empty else pd.DataFrame(columns=BLOCKER_OUTPUT_COLUMNS)


def build_candidate_hypotheses(scored: pd.DataFrame) -> pd.DataFrame:
    candidates = scored[scored["readiness_class"].str.startswith(("A:", "B:"), na=False)].copy()
    rows = []
    for i, row in candidates.sort_values("readiness_score", ascending=False).reset_index(drop=True).iterrows():
        task = clean_name(row.get("task", "")) or "observed task"
        carrier = clean_name(row.get("carrier", "")) or "observed carrier"
        feature = clean_name(row.get("feature_family", "")) or "observed feature family"
        geometry = clean_name(row.get("minimal_sufficient_geometry", "")) or clean_name(row.get("geometry_level", "")) or "current registry geometry"
        dims = list(WEIGHTS.keys())
        supporting = [d for d in dims if as_float(row.get(d)) >= 0.65]
        limiting = [d for d in dims if math.isnan(as_float(row.get(d))) or as_float(row.get(d)) < 0.50]
        rows.append({
            "hypothesis_id": f"OBS082-H{i+1:03d}",
            "relation_id": row.get("relation_id", ""),
            "task": task,
            "carrier": carrier,
            "feature_family": feature,
            "readiness_class": row.get("readiness_class", ""),
            "readiness_score": row.get("readiness_score", np.nan),
            "readiness_limiter": row.get("readiness_limiter", ""),
            "candidate_hypothesis": f"Within the OBS-081 registry lineage, if the {task} relation depends on {feature} evidence carried by {carrier}, then targeted perturbation or withholding of that evidence should reduce relation survival under tested OBS-080-style contracts.",
            "intervention_axis": feature,
            "expected_direction": "reduced relation survival under targeted perturbation/withholding",
            "required_geometry_level": geometry,
            "required_carrier_evidence": f"carrier convergence evidence for {carrier}",
            "supporting_dimensions": ";".join(supporting),
            "limiting_dimensions": ";".join(limiting),
            "required_next_test": "Run an explicit perturbation/withholding probe against matched negative controls within the same artifact lineage.",
            "falsification_condition": "The hypothesis is weakened if relation survival does not decrease under the targeted perturbation, or if matched negative controls show comparable survival.",
            "scope_condition": "; ".join(SCOPE_GUARDRAILS),
        })
    out = pd.DataFrame(rows)
    return out[CANDIDATE_OUTPUT_COLUMNS] if not out.empty else pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)


def md_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "\n_No rows._\n"
    try:
        return "\n" + df.head(max_rows).to_markdown(index=False) + "\n"
    except Exception:
        return "\n" + df.head(max_rows).to_string(index=False) + "\n"


def write_report(outdir: Path, manifest: pd.DataFrame, scored: pd.DataFrame, candidates: pd.DataFrame, contrasts: pd.DataFrame, failure_inventory: pd.DataFrame) -> None:
    class_counts = scored["readiness_class"].value_counts(dropna=False).reset_index() if not scored.empty else pd.DataFrame(columns=["readiness_class", "count"])
    class_counts.columns = ["readiness_class", "n"] if len(class_counts.columns) == 2 else class_counts.columns
    blockers: list[str] = []
    limiters: list[str] = []
    if not scored.empty:
        for item in scored["readiness_blockers"].fillna(""):
            blockers += [x for x in str(item).split(";") if x]
        if "readiness_limiter" in scored.columns:
            for item in scored["readiness_limiter"].fillna(""):
                limiters += [x for x in str(item).split(";") if x]
    blocker_counts = pd.Series(blockers).value_counts().reset_index() if blockers else pd.DataFrame(columns=["blocker", "n"])
    limiter_counts = pd.Series(limiters).value_counts().reset_index() if limiters else pd.DataFrame(columns=["readiness_limiter", "n"])
    if not blocker_counts.empty:
        blocker_counts.columns = ["blocker", "n"]
    if not limiter_counts.empty:
        limiter_counts.columns = ["readiness_limiter", "n"]
    top_cols = ["relation_id", "task", "carrier", "feature_family", "readiness_score", "readiness_class", "readiness_limiter", "readiness_blockers"]
    top = scored.sort_values("readiness_score", ascending=False)[top_cols] if not scored.empty else pd.DataFrame(columns=top_cols)
    report = []
    report.append("# OBS-082 — RIG Intervention-Readiness Audit\n\n")
    report.append("## 1. Scope\n\n")
    report.extend(f"- {x}\n" for x in SCOPE_GUARDRAILS)
    report.append("\n")
    report.extend(f"- {x}\n" for x in NON_CLAIMS)
    report.append("\n## 2. Inputs and artifact lineage\n")
    report.append(md_table(manifest, 20))
    report.append("\n## 3. Scoring schema\n\n")
    report.append("Readiness is computed from invariance strength, failure localization, repair specificity, geometry sufficiency, carrier convergence, and negative-control contrast. `rig_status` is metadata; it is used only as a categorical fallback when direct invariance evidence is unavailable. If invariance evidence comes only from categorical fallback, readiness class cannot exceed B.\n")
    report.append(md_table(pd.DataFrame([{"dimension": k, "weight": v} for k, v in WEIGHTS.items()])))
    report.append("\n## 4. Readiness class summary\n")
    report.append(md_table(class_counts))
    report.append("\n## 5. Readiness limiter summary\n")
    report.append(md_table(limiter_counts))
    report.append("\n## 6. Dimension-level results\n")
    dim_cols = ["relation_id", "task"] + list(WEIGHTS.keys()) + ["readiness_score", "readiness_class"]
    report.append(md_table(scored[dim_cols].sort_values("readiness_score", ascending=False) if not scored.empty else pd.DataFrame(columns=dim_cols)))
    report.append("\n## 7. Candidate intervention-hypothesis records\n")
    cand_cols = ["hypothesis_id", "relation_id", "task", "readiness_class", "readiness_score", "readiness_limiter", "required_next_test"]
    report.append(md_table(candidates[cand_cols] if not candidates.empty else pd.DataFrame(columns=cand_cols)))
    report.append("\n## 8. Negative-control contrasts\n")
    report.append(md_table(contrasts, 20))
    report.append("\n## 9. Failure localization and repair structure\n")
    fail_cols = ["relation_id", "failure_mode", "failure_location", "readiness_impact"]
    report.append(md_table(failure_inventory[fail_cols] if not failure_inventory.empty else pd.DataFrame(columns=fail_cols)))
    report.append("\n## 10. Geometry sufficiency ladder\n")
    geom_cols = ["relation_id", "task", "geometry_level", "minimal_sufficient_geometry", "geometry_sufficiency_score", "geometry_sufficiency_basis"]
    report.append(md_table(scored[geom_cols] if not scored.empty else pd.DataFrame(columns=geom_cols)))
    report.append("\n## 11. Blocked / insufficient records\n")
    report.append(md_table(blocker_counts))
    report.append("\n## 12. Interpretation\n\n")
    report.append("Class A and B records are candidates for defining future testable intervention hypotheses. They are not evidence that interventions have been performed or that the system is controllable. Records in Class C/D/X remain useful as diagnostics, registry evidence, or missing-artifact signals.\n")
    report.append("\n## 13. What this does not show\n")
    report.extend(f"- {x}\n" for x in NON_CLAIMS[:-1])
    report.append("\n## 14. Recommended next tests\n\n")
    report.append("For Class A/B records, design a follow-up perturbation or withholding probe that targets the record's intervention axis, evaluates relation survival under matched OBS-080-style contracts, and compares against explicit negative controls.\n")
    report.append("\n## 15. Top readiness records\n")
    report.append(md_table(top, 20))
    (outdir / "obs082_report.md").write_text("".join(report), encoding="utf-8")


def write_empty_outputs(outdir: Path, manifest: pd.DataFrame) -> None:
    pd.DataFrame(columns=READINESS_OUTPUT_COLUMNS).to_csv(outdir / "obs082_relation_readiness_scores.csv", index=False)
    pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS).to_csv(outdir / "obs082_candidate_intervention_hypotheses.csv", index=False)
    pd.DataFrame(columns=NEGATIVE_CONTROL_OUTPUT_COLUMNS).to_csv(outdir / "obs082_negative_control_contrasts.csv", index=False)
    pd.DataFrame(columns=FAILURE_OUTPUT_COLUMNS).to_csv(outdir / "obs082_failure_mode_inventory.csv", index=False)
    pd.DataFrame(columns=BLOCKER_OUTPUT_COLUMNS).to_csv(outdir / "obs082_blockers.csv", index=False)
    write_report(outdir, manifest, pd.DataFrame(columns=READINESS_OUTPUT_COLUMNS), pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS), pd.DataFrame(columns=NEGATIVE_CONTROL_OUTPUT_COLUMNS), pd.DataFrame(columns=FAILURE_OUTPUT_COLUMNS))


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    specs = [
        InputSpec("rig_relation_registry", Path(args.registry), ("relation_id", "task", "comparison", "relation_name", "task_name"), ("relation_id", "task", "carrier", "feature_family", "rig_status"), "base records; task normalization; OBS-081 v2 carrier/survival evidence; rig_status metadata"),
        InputSpec("rig_survival_matrix", Path(args.survival), ("relation_id", "task", "comparison", "relation_name", "task_name"), ("relation_id", "score", "passed"), "optional D1 aggregate survival-matrix fallback; tested/survival/failure counts", optional=True),
        InputSpec("rig_failure_localization", Path(args.failure_localization), ("relation_id", "task", "comparison", "relation_name", "task_name"), ("relation_id", "failure_type", "failure_mode"), "D2 failure localization; failure inventory", optional=True),
        InputSpec("rig_geometry_needed_ladder", Path(args.geometry_ladder), ("relation_id", "task", "comparison", "relation_name", "task_name"), ("relation_id", "geometry_needed", "task_geometry_needed_level", "task_geometry_needed_label"), "D4 geometry sufficiency", optional=True),
        InputSpec("rig_repair_recommendations", Path(args.repair_recommendations), ("relation_id", "task", "comparison", "relation_name", "task_name"), ("relation_id", "repair_recommendation"), "D3 repair specificity", optional=True),
    ]
    loaded: dict[str, pd.DataFrame | None] = {}
    manifest_rows = []
    for spec in specs:
        df = read_csv_if_exists(spec.path)
        loaded[spec.name] = df
        manifest_rows.append(manifest_row(spec, df))
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(outdir / "obs082_input_manifest.csv", index=False)
    registry = loaded["rig_relation_registry"]
    survival = loaded["rig_survival_matrix"]
    failure = loaded["rig_failure_localization"]
    geometry = loaded["rig_geometry_needed_ladder"]
    repair = loaded["rig_repair_recommendations"]
    if registry is None or registry.empty:
        write_empty_outputs(outdir, manifest)
        raise SystemExit(f"ERROR: Missing required registry artifact: {args.registry}. Wrote manifest and empty schema outputs to {outdir}.")
    merged = merge_artifacts(registry, survival, failure, geometry, repair)
    if "relation_id" not in merged.columns or merged["relation_id"].map(clean_name).eq("").all():
        raise SystemExit("ERROR: Could not construct usable relation_id keys from input artifacts.")
    scored, contrasts = score_records(merged, args.adequate_tested_count, args.contrast_scale)
    failure_inventory = build_failure_inventory(failure, scored)
    blockers = build_blockers(scored)
    candidates = build_candidate_hypotheses(scored)
    scored.to_csv(outdir / "obs082_relation_readiness_scores.csv", index=False)
    candidates.to_csv(outdir / "obs082_candidate_intervention_hypotheses.csv", index=False)
    contrasts.to_csv(outdir / "obs082_negative_control_contrasts.csv", index=False)
    failure_inventory.to_csv(outdir / "obs082_failure_mode_inventory.csv", index=False)
    blockers.to_csv(outdir / "obs082_blockers.csv", index=False)
    write_report(outdir, manifest, scored, candidates, contrasts, failure_inventory)
    print("OBS-082 — RIG Intervention-Readiness Audit complete")
    print(f"Output directory: {outdir}")
    print("Wrote:")
    for name in [
        "obs082_input_manifest.csv",
        "obs082_relation_readiness_scores.csv",
        "obs082_candidate_intervention_hypotheses.csv",
        "obs082_negative_control_contrasts.csv",
        "obs082_failure_mode_inventory.csv",
        "obs082_blockers.csv",
    "obs082_report.md",
    ]:
        print(f"  - {outdir / name}")
    if not scored.empty:
        print("\nReadiness classes:")
        print(scored["readiness_class"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
