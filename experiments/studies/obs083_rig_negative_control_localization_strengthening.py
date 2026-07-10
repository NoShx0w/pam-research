#!/usr/bin/env python3
"""
obs083_rig_negative_control_localization_strengthening.py

OBS-083 — RIG Negative-Control and Failure-Localization Strengthening

Version: v2 conservative prior-aware audit

Purpose
-------
Refine the OBS-082 diagnostic-only readiness result for the 24 OBS-081
Reusable Invariance Registry records. This script strengthens the audit layer by
constructing matched negative-control designs, relation/carrier/contract/
geometry-needed contrast tables, a failure-localization matrix, repair-specificity
sharpening table, and diagnostic subclass assignments.

Scope and guardrails
--------------------
- Reads existing OBS-081/OBS-082 artifacts from outputs/rig_registry/.
- Writes outputs/rig_registry/obs083_negative_control_localization/.
- Does not perform interventions.
- Does not claim causality, control, actionability, universal invariance, or
  formal topology.
- Treats C4 as diagnostic-only / promising next-test candidate, NOT
  candidate-ready or actionable.
- Never treats missing evidence as positive evidence. Missing optional OBS-080
  contract columns are carried as missing/provenance limitations.

Expected inputs
---------------
OBS-081, under --registry-dir:
    rig_input_manifest.csv
    rig_relation_registry.csv
    rig_survival_matrix.csv
    rig_failure_localization.csv
    rig_geometry_needed_ladder.csv
    rig_repair_recommendations.csv
    rig_registry_report.md

OBS-082, under --registry-dir/obs082_intervention_readiness:
    obs082_input_manifest.csv
    obs082_relation_readiness_scores.csv
    obs082_negative_control_contrasts.csv
    obs082_failure_mode_inventory.csv
    obs082_blockers.csv
    obs082_candidate_intervention_hypotheses.csv
    obs082_report.md

Outputs
-------
    obs083_input_manifest.csv
    obs083_matched_negative_control_design.csv
    obs083_relation_control_contrast.csv
    obs083_carrier_control_contrast.csv
    obs083_contract_control_contrast.csv
    obs083_geometry_needed_control_contrast.csv
    obs083_failure_localization_matrix.csv
    obs083_repair_specificity_sharpening.csv
    obs083_diagnostic_subclass_assignments.csv
    obs083_readiness_delta_from_obs082.csv
    obs083_blocker_refinement.csv
    obs083_report.md

"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

GUARDRAIL = (
    "OBS-083 is a diagnostic refinement audit only: no interventions performed; "
    "no causality/control/actionability claims; C4 remains diagnostic-only. v2 uses OBS-082 diffuse/generic priors and evidence-completeness gates."
)

OBS081_FILES = {
    "obs081_input_manifest": "rig_input_manifest.csv",
    "obs081_relation_registry": "rig_relation_registry.csv",
    "obs081_survival_matrix": "rig_survival_matrix.csv",
    "obs081_failure_localization": "rig_failure_localization.csv",
    "obs081_geometry_needed_ladder": "rig_geometry_needed_ladder.csv",
    "obs081_repair_recommendations": "rig_repair_recommendations.csv",
    "obs081_registry_report": "rig_registry_report.md",
}

OBS082_FILES = {
    "obs082_input_manifest": "obs082_intervention_readiness/obs082_input_manifest.csv",
    "obs082_readiness_scores": "obs082_intervention_readiness/obs082_relation_readiness_scores.csv",
    "obs082_negative_control_contrasts": "obs082_intervention_readiness/obs082_negative_control_contrasts.csv",
    "obs082_failure_mode_inventory": "obs082_intervention_readiness/obs082_failure_mode_inventory.csv",
    "obs082_blockers": "obs082_intervention_readiness/obs082_blockers.csv",
    "obs082_candidate_intervention_hypotheses": "obs082_intervention_readiness/obs082_candidate_intervention_hypotheses.csv",
    "obs082_report": "obs082_intervention_readiness/obs082_report.md",
}

RELATION_COLS = ["relation", "rig_relation", "comparison", "task", "relation_id"]
CARRIER_COLS = ["carrier", "carrier_name", "feature_family", "feature_set", "rig_carrier"]
RECORD_COLS = ["record_id", "rig_record_id", "relation_carrier_id", "record", "id"]
CLASS_COLS = ["readiness_class", "obs082_class", "class", "readiness_tier"]
BLOCKER_COLS = ["readiness_limiter", "primary_limiter", "blocker", "blockers", "fatal_blockers"]
REPAIR_TEXT_COLS = [
    "repair_recommendation",
    "repair_annotation",
    "recommendation",
    "repair",
    "repair_text",
    "obs081_repair_recommendation",
]
SCORE_COLS = [
    "score_basis",
    "obs080d_carrier_mean_ba",
    "obs080d_mean_ba",
    "carrier_mean_ba",
    "mean_ba",
    "balanced_accuracy",
    "ba",
    "readiness_score",
    "invariance_strength_score",
    "survival_score",
    "rig_survival_score",
]
NEGATIVE_SCORE_COLS = ["negative_control_contrast_score", "negative_control_score", "contrast_score"]
LOCALIZATION_SCORE_COLS = ["failure_localization_score", "localization_score"]
REPAIR_SCORE_COLS = ["repair_specificity_score", "repair_score"]

CONTRACT_KEYWORDS = {
    "numeric_transform": ["numeric", "rank", "quantile", "zscore", "scaled", "transform"],
    "scale_band": ["scale", "band", "low", "mid", "high"],
    "feature_family_projection": ["feature", "family", "projection", "project"],
    "structural_resampling": ["resample", "bootstrap", "leave", "structure", "obs080d"],
}

CARRIER_ROLE_ORDER = [
    "stability_core_3",
    "stability_plus_geometry",
    "geometry_scores_only",
    "path_shares_only",
    "no_window",
    "strict_numeric_all",
]


@dataclass
class Thresholds:
    strong_delta: float
    moderate_delta: float
    contrast_threshold: float
    localization_threshold: float
    repair_threshold: float
    c4_contrast: float
    c4_localization: float
    c4_repair: float
    survival_floor: float


def clean_col(name: Any) -> str:
    return str(name).strip()


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm_key(value: Any) -> str:
    txt = norm_text(value)
    txt = re.sub(r"\s+", "_", txt)
    txt = re.sub(r"[^A-Za-z0-9_./:-]+", "", txt)
    return txt


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for low, original in lower.items():
            if cand.lower() in low:
                return original
    return None


def numeric_value(row: pd.Series, candidates: Iterable[str]) -> tuple[float | None, str]:
    for col in candidates:
        if col in row.index:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val), col
    return None, ""


def clamp01(x: float | None) -> float | None:
    if x is None or pd.isna(x):
        return None
    return max(0.0, min(1.0, float(x)))


def safe_mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def score_from_delta(delta: float | None, strong_delta: float) -> float | None:
    if delta is None or pd.isna(delta):
        return None
    if strong_delta <= 0:
        return None
    return clamp01(abs(delta) / strong_delta)


def classify_delta(delta: float | None, thresholds: Thresholds) -> str:
    if delta is None or pd.isna(delta):
        return "missing"
    ad = abs(float(delta))
    if ad >= thresholds.strong_delta:
        return "strong"
    if ad >= thresholds.moderate_delta:
        return "moderate"
    return "weak"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.suffix.lower() != ".csv":
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [clean_col(c) for c in df.columns]
        return df
    except Exception as exc:  # noqa: BLE001 - reportable provenance issue
        print(f"WARN: failed to read {path}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def file_manifest(registry_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, rel in {**OBS081_FILES, **OBS082_FILES}.items():
        path = registry_dir / rel
        row: dict[str, Any] = {
            "artifact_label": label,
            "relative_path": rel,
            "path": str(path),
            "exists": path.exists(),
            "file_type": path.suffix.lower().lstrip("."),
            "rows": None,
            "columns": None,
            "read_status": "not_read",
        }
        if path.exists() and path.suffix.lower() == ".csv":
            df = read_csv_if_exists(path)
            row["rows"] = int(len(df))
            row["columns"] = int(len(df.columns))
            row["read_status"] = "ok" if not df.empty or len(df.columns) > 0 else "empty_or_failed"
        elif path.exists():
            row["read_status"] = "exists_non_csv"
        else:
            row["read_status"] = "missing"
        rows.append(row)
    return pd.DataFrame(rows)


def load_inputs(registry_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for label, rel in {**OBS081_FILES, **OBS082_FILES}.items():
        path = registry_dir / rel
        if path.suffix.lower() == ".csv":
            data[label] = read_csv_if_exists(path)
        else:
            data[label] = pd.DataFrame()
    return data


def ensure_record_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    record_col = find_col(out, RECORD_COLS)
    relation_col = find_col(out, RELATION_COLS)
    carrier_col = find_col(out, CARRIER_COLS)

    if record_col is None:
        if relation_col and carrier_col:
            out["record_id"] = out[relation_col].map(norm_key) + "__" + out[carrier_col].map(norm_key)
        else:
            out["record_id"] = [f"record_{i:03d}" for i in range(len(out))]
    else:
        out["record_id"] = out[record_col].map(norm_key)

    if relation_col is None:
        parsed = out["record_id"].str.split("__", n=1, expand=True)
        out["relation"] = parsed[0] if parsed.shape[1] >= 1 else ""
    else:
        out["relation"] = out[relation_col].map(norm_key)

    if carrier_col is None:
        parsed = out["record_id"].str.split("__", n=1, expand=True)
        out["carrier"] = parsed[1] if parsed.shape[1] > 1 else "unknown_carrier"
    else:
        out["carrier"] = out[carrier_col].map(norm_key)

    out["record_id"] = out["relation"].map(norm_key) + "__" + out["carrier"].map(norm_key)
    return out


def prefix_nonkey(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = ensure_record_fields(df)
    key_cols = {"record_id", "relation", "carrier"}
    renamed = {c: f"{prefix}{c}" for c in out.columns if c not in key_cols and not c.startswith(prefix)}
    return out.rename(columns=renamed)


def build_base_records(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = [
        data.get("obs081_relation_registry", pd.DataFrame()),
        data.get("obs082_readiness_scores", pd.DataFrame()),
        data.get("obs081_survival_matrix", pd.DataFrame()),
    ]
    base = pd.DataFrame()
    for cand in candidates:
        if cand is not None and not cand.empty:
            base = ensure_record_fields(cand)
            break
    if base.empty:
        raise SystemExit(
            "No usable OBS-081/082 registry records found. Expected rig_relation_registry.csv "
            "or obs082_relation_readiness_scores.csv."
        )

    base = base.drop_duplicates(subset=["record_id"], keep="first").copy()

    # Merge in key OBS-082/OBS-081 data by record_id, preserving original fields with prefixes.
    merge_specs = [
        ("obs082_readiness_scores", "obs082_"),
        ("obs082_negative_control_contrasts", "obs082_neg_"),
        ("obs082_failure_mode_inventory", "obs082_fail_"),
        ("obs082_blockers", "obs082_block_"),
        ("obs081_survival_matrix", "obs081_surv_"),
        ("obs081_failure_localization", "obs081_fail_"),
        ("obs081_geometry_needed_ladder", "obs081_geom_"),
        ("obs081_repair_recommendations", "obs081_repair_"),
    ]
    for label, prefix in merge_specs:
        df = data.get(label, pd.DataFrame())
        if df.empty:
            continue
        pdf = prefix_nonkey(df, prefix)
        # Avoid exploding if source has repeated rows per record: keep first for base-level merge.
        pdf = pdf.drop_duplicates(subset=["record_id"], keep="first")
        add_cols = [c for c in pdf.columns if c not in {"relation", "carrier"}]
        base = base.merge(pdf[add_cols], on="record_id", how="left")

    # Score basis and provenance.
    score_candidates = []
    for c in SCORE_COLS:
        score_candidates.extend([c, f"obs082_{c}", f"obs081_surv_{c}"])
    score_vals = []
    score_sources = []
    for _, row in base.iterrows():
        val, src = numeric_value(row, score_candidates)
        score_vals.append(val)
        score_sources.append(src)
    base["survival_score"] = score_vals
    base["survival_score_source"] = score_sources
    base["survival_evidence_available"] = base["survival_score"].notna()

    class_candidates = []
    for c in CLASS_COLS:
        class_candidates.extend([c, f"obs082_{c}"])
    obs082_classes = []
    for _, row in base.iterrows():
        val = ""
        for c in class_candidates:
            if c in row.index and norm_text(row[c]):
                val = norm_text(row[c])
                break
        obs082_classes.append(val or "unknown")
    base["obs082_class"] = obs082_classes

    return base


def matched_negative_control_design(records: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    record_ids = set(records["record_id"])

    for _, target in records.iterrows():
        rid = target["record_id"]
        rel = target["relation"]
        car = target["carrier"]
        status = first_text(target, ["rig_status", "obs081_rig_status", "obs082_rig_status"])
        obs082_class = norm_text(target.get("obs082_class", "unknown")) or "unknown"

        # Relation controls: same carrier, different relation.
        for _, ctrl in records[(records["carrier"] == car) & (records["relation"] != rel)].iterrows():
            rows.append(control_row(rid, rel, car, status, obs082_class, "relation", ctrl["record_id"], ctrl["relation"], car,
                                    "same carrier, different relation should not reproduce target-specific contrast",
                                    "Tests whether carrier preserves target relation specifically", True))

        # Carrier controls: same relation, different carrier.
        for _, ctrl in records[(records["relation"] == rel) & (records["carrier"] != car)].iterrows():
            rows.append(control_row(rid, rel, car, status, obs082_class, "carrier", ctrl["record_id"], rel, ctrl["carrier"],
                                    "same relation, different carrier should show differentiated carrier role",
                                    "Tests whether carrier role is specific or overbroad", True))

        # Contract controls: design rows, evidence may be missing.
        for fam in CONTRACT_KEYWORDS:
            rows.append(control_row(rid, rel, car, status, obs082_class, "contract", f"{rid}__contract__{fam}", rel, car,
                                    f"contract family {fam} should reveal structured survival or weakness when evidence exists",
                                    "Tests whether survival/failure localizes to contract family", contract_evidence_available(data, rid, fam)))

        # Geometry-needed controls.
        for gc in ["stability_core_3", "stability_plus_geometry", "geometry_scores_only", "path_shares_only"]:
            if gc == car:
                continue
            ctrl_id = f"{rel}__{gc}"
            rows.append(control_row(rid, rel, car, status, obs082_class, "geometry_needed", ctrl_id, rel, gc,
                                    "geometry/compact/path carriers should show differentiated necessity or sharpening",
                                    "Tests geometry-needed label versus carrier role", ctrl_id in record_ids))

        # Failure-mode controls.
        rows.append(control_row(rid, rel, car, status, obs082_class, "failure_mode", f"{rid}__failure_mode_control", rel, car,
                                "matched controls should not show the same failure locus if localization is specific",
                                "Tests whether failure is localized rather than diffuse", True))

        # Permutation/shuffle controls are optional and usually missing at OBS-083 registry layer.
        rows.append(control_row(rid, rel, car, status, obs082_class, "permutation", f"{rid}__permutation_control", rel, car,
                                "shuffle/permutation should weaken relation if available; insufficient alone for RIG specificity",
                                "Baseline non-randomness control, not full negative-control contrast", permutation_evidence_available(data, rid)))

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "record_id", "relation", "carrier", "target_rig_status", "target_obs082_class",
            "control_family", "control_id", "control_relation", "control_carrier",
            "control_contract", "control_expected_behavior", "match_rationale",
            "contrast_hypothesis", "available_artifact_source", "evidence_available",
            "control_strength_grade", "control_limitations",
        ])
    return out


def control_row(
    rid: str,
    rel: str,
    car: str,
    status: str,
    obs082_class: str,
    family: str,
    ctrl_id: str,
    ctrl_rel: str,
    ctrl_car: str,
    expected: str,
    rationale: str,
    evidence_available: bool,
) -> dict[str, Any]:
    if family == "contract":
        control_contract = ctrl_id.rsplit("__contract__", 1)[-1]
    else:
        control_contract = ""
    if family in {"relation", "carrier"} and evidence_available:
        grade = "A"
        limitation = "matched registry record available"
    elif evidence_available:
        grade = "B"
        limitation = "derived or partial artifact evidence available"
    else:
        grade = "D"
        limitation = "optional evidence missing; not scored as positive evidence"
    return {
        "record_id": rid,
        "relation": rel,
        "carrier": car,
        "target_rig_status": status,
        "target_obs082_class": obs082_class,
        "control_family": family,
        "control_id": ctrl_id,
        "control_relation": ctrl_rel,
        "control_carrier": ctrl_car,
        "control_contract": control_contract,
        "control_expected_behavior": expected,
        "match_rationale": rationale,
        "contrast_hypothesis": "target should show stronger or structurally different evidence than matched control",
        "available_artifact_source": artifact_source_for_family(family, evidence_available),
        "evidence_available": bool(evidence_available),
        "control_strength_grade": grade,
        "control_limitations": limitation,
    }


def artifact_source_for_family(family: str, available: bool) -> str:
    if not available:
        return "missing_optional_evidence"
    return {
        "relation": "OBS-081/082 relation x carrier records",
        "carrier": "OBS-081/082 relation x carrier records",
        "contract": "OBS-080/081 survival/contract evidence when present",
        "geometry_needed": "OBS-081 geometry-needed ladder and carrier records",
        "failure_mode": "OBS-081/082 failure localization/blocker evidence",
        "permutation": "permutation/shuffle evidence if present in available artifacts",
    }.get(family, "available_artifacts")


def contract_evidence_available(data: dict[str, pd.DataFrame], record_id: str, family: str) -> bool:
    for label in ["obs081_survival_matrix", "obs082_readiness_scores"]:
        df = data.get(label, pd.DataFrame())
        if df.empty:
            continue
        txt_cols = [c for c in df.columns if df[c].dtype == object]
        hay = " ".join(" ".join(df[c].astype(str).head(200).tolist()).lower() for c in txt_cols)
        if any(k in hay for k in CONTRACT_KEYWORDS[family]):
            return True
        if any(any(k in str(c).lower() for k in CONTRACT_KEYWORDS[family]) for c in df.columns):
            return True
    return False


def permutation_evidence_available(data: dict[str, pd.DataFrame], record_id: str) -> bool:
    for df in data.values():
        if df.empty:
            continue
        cols = " ".join(map(str, df.columns)).lower()
        if "permutation" in cols or "shuffle" in cols or "null" in cols:
            return True
    return False


def first_text(row: pd.Series, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in row.index and norm_text(row[c]):
            return norm_text(row[c])
    return ""


def relation_control_contrast(records: pd.DataFrame, thresholds: Thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    aggs: list[dict[str, Any]] = []
    for _, target in records.iterrows():
        controls = records[(records["carrier"] == target["carrier"]) & (records["relation"] != target["relation"])]
        deltas: list[float] = []
        for _, ctrl in controls.iterrows():
            delta = compute_delta(target["survival_score"], ctrl["survival_score"])
            if delta is not None:
                deltas.append(abs(delta))
            rows.append({
                "record_id": target["record_id"],
                "relation": target["relation"],
                "carrier": target["carrier"],
                "target_score_basis": target["survival_score"],
                "target_score_source": target["survival_score_source"],
                "control_relation": ctrl["relation"],
                "control_record_id": ctrl["record_id"],
                "control_score_basis": ctrl["survival_score"],
                "control_score_source": ctrl["survival_score_source"],
                "relation_contrast_delta": delta,
                "relation_contrast_abs_delta": abs(delta) if delta is not None else None,
                "specificity_direction": direction(delta),
                "expected_direction": expected_relation_direction(target["relation"], ctrl["relation"]),
                "contrast_support": classify_delta(delta, thresholds),
                "evidence_available": delta is not None,
                "interpretation": relation_interpretation(delta, thresholds),
            })
        max_delta = max(deltas) if deltas else None
        mean_delta = safe_mean(deltas)
        aggs.append({
            "record_id": target["record_id"],
            "relation_contrast_score": score_from_delta(max_delta, thresholds.strong_delta) if max_delta is not None else None,
            "relation_contrast_max_abs_delta": max_delta,
            "relation_contrast_mean_abs_delta": mean_delta,
            "relation_control_count": int(len(controls)),
            "relation_control_evidence_available": bool(deltas),
        })
    return pd.DataFrame(rows), pd.DataFrame(aggs)


def carrier_control_contrast(records: pd.DataFrame, thresholds: Thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    aggs: list[dict[str, Any]] = []
    for _, target in records.iterrows():
        controls = records[(records["relation"] == target["relation"]) & (records["carrier"] != target["carrier"])]
        deltas: list[float] = []
        for _, ctrl in controls.iterrows():
            delta = compute_delta(target["survival_score"], ctrl["survival_score"])
            if delta is not None:
                deltas.append(abs(delta))
            observed_role = observed_carrier_role(target["carrier"], ctrl["carrier"], delta, thresholds)
            rows.append({
                "record_id": target["record_id"],
                "relation": target["relation"],
                "target_carrier": target["carrier"],
                "target_score_basis": target["survival_score"],
                "target_score_source": target["survival_score_source"],
                "control_carrier": ctrl["carrier"],
                "control_record_id": ctrl["record_id"],
                "control_score_basis": ctrl["survival_score"],
                "control_score_source": ctrl["survival_score_source"],
                "carrier_contrast_delta": delta,
                "carrier_contrast_abs_delta": abs(delta) if delta is not None else None,
                "carrier_role_expected": expected_carrier_role(target["carrier"]),
                "carrier_role_observed": observed_role,
                "carrier_specificity_support": classify_delta(delta, thresholds),
                "overbroad_carrier_flag": bool(delta is not None and abs(delta) < thresholds.moderate_delta),
                "geometry_sharpening_flag": bool(is_geometry_carrier(target["carrier"]) or is_geometry_carrier(ctrl["carrier"])),
                "evidence_available": delta is not None,
                "interpretation": carrier_interpretation(delta, thresholds, target["carrier"], ctrl["carrier"]),
            })
        max_delta = max(deltas) if deltas else None
        aggs.append({
            "record_id": target["record_id"],
            "carrier_contrast_score": score_from_delta(max_delta, thresholds.strong_delta) if max_delta is not None else None,
            "carrier_contrast_max_abs_delta": max_delta,
            "carrier_control_count": int(len(controls)),
            "carrier_control_evidence_available": bool(deltas),
        })
    return pd.DataFrame(rows), pd.DataFrame(aggs)


def compute_delta(a: Any, b: Any) -> float | None:
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


def direction(delta: float | None) -> str:
    if delta is None or pd.isna(delta):
        return "missing"
    if delta > 0:
        return "target_stronger"
    if delta < 0:
        return "target_weaker"
    return "similar"


def expected_relation_direction(target_relation: str, control_relation: str) -> str:
    # Conservative heuristic from OBS-080/081 interpretation: Cp2_vs_Cp3 and three_way are subtler.
    subtle = {"Cp2_vs_Cp3", "three_way"}
    if target_relation in subtle and control_relation not in subtle:
        return "target_may_be_weaker_or_geometry_sensitive"
    if target_relation not in subtle and control_relation in subtle:
        return "target_may_be_stronger"
    return "no_predeclared_direction"


def relation_interpretation(delta: float | None, thresholds: Thresholds) -> str:
    cls = classify_delta(delta, thresholds)
    if cls == "missing":
        return "Missing score evidence; contrast not treated as positive evidence."
    if cls == "strong":
        return "Matched relation contrast is strong enough to support specificity evidence, subject to other limiters."
    if cls == "moderate":
        return "Matched relation contrast is moderate; useful but not sufficient alone."
    return "Matched relation contrast is weak; relation specificity remains limited."


def is_geometry_carrier(carrier: str) -> bool:
    c = carrier.lower()
    return "geometry" in c or "geom" in c


def expected_carrier_role(carrier: str) -> str:
    c = carrier.lower()
    if c == "stability_core_3":
        return "compact_primary_candidate"
    if c == "stability_plus_geometry":
        return "geometry_sharpening_candidate"
    if c == "geometry_scores_only":
        return "geometry_only_control"
    if c == "path_shares_only":
        return "path_support_control"
    if c == "no_window":
        return "non_window_redundancy_control"
    if c == "strict_numeric_all":
        return "strict_numeric_reference_control"
    return "unspecified_carrier_role"


def observed_carrier_role(target_carrier: str, control_carrier: str, delta: float | None, thresholds: Thresholds) -> str:
    if delta is None:
        return "unscored_missing_evidence"
    if abs(delta) < thresholds.moderate_delta:
        return "similar_or_redundant"
    if target_carrier == "stability_core_3" and delta > 0:
        return "compact_sufficient_relative_to_control"
    if is_geometry_carrier(target_carrier) and delta > 0:
        return "geometry_sharpening_or_required_candidate"
    if delta < 0:
        return "target_weaker_than_control"
    return "target_specificity_candidate"


def carrier_interpretation(delta: float | None, thresholds: Thresholds, target_carrier: str, control_carrier: str) -> str:
    cls = classify_delta(delta, thresholds)
    if cls == "missing":
        return "Missing score evidence; carrier contrast not treated as positive evidence."
    if cls == "weak":
        return "Carrier contrast is weak; carrier may be redundant or overbroad for this relation."
    if is_geometry_carrier(target_carrier) or is_geometry_carrier(control_carrier):
        return "Carrier contrast involves geometry; interpret as sharpening/necessity audit, not proof of intrinsic geometry."
    return f"Carrier contrast is {cls}; carrier role is more differentiated for this relation."


def contract_control_contrast(records: pd.DataFrame, data: dict[str, pd.DataFrame], thresholds: Thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    survival = data.get("obs081_survival_matrix", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    aggs: list[dict[str, Any]] = []

    long_available = False
    surv = pd.DataFrame()
    if not survival.empty:
        surv = ensure_record_fields(survival)
        long_available = any("contract" in c.lower() for c in surv.columns) or any("transform" in c.lower() for c in surv.columns)

    for _, rec in records.iterrows():
        rid = rec["record_id"]
        fam_scores: dict[str, list[float]] = {fam: [] for fam in CONTRACT_KEYWORDS}
        if not surv.empty:
            subset = surv[surv["record_id"] == rid]
            if not subset.empty:
                for fam, keys in CONTRACT_KEYWORDS.items():
                    # First: score columns whose names indicate the contract family.
                    family_cols = [c for c in subset.columns if any(k in c.lower() for k in keys)]
                    for col in family_cols:
                        vals = pd.to_numeric(subset[col], errors="coerce").dropna().tolist()
                        fam_scores[fam].extend([float(v) for v in vals])
                    # Second: long table with contract family values.
                    contract_col = find_col(subset, ["contract_family", "contract", "transform_family", "variant"])
                    score_col = find_col(subset, SCORE_COLS)
                    if contract_col and score_col:
                        for _, srow in subset.iterrows():
                            if any(k in norm_text(srow[contract_col]).lower() for k in keys):
                                val = pd.to_numeric(pd.Series([srow[score_col]]), errors="coerce").iloc[0]
                                if pd.notna(val):
                                    fam_scores[fam].append(float(val))

        record_scores = [v for vals in fam_scores.values() for v in vals]
        for fam, scores in fam_scores.items():
            if scores:
                family_mean = sum(scores) / len(scores)
                family_min = min(scores)
                family_max = max(scores)
                sensitivity = family_max - family_min
                evidence = True
            else:
                family_mean = family_min = family_max = sensitivity = None
                evidence = False
            rows.append({
                "record_id": rid,
                "relation": rec["relation"],
                "carrier": rec["carrier"],
                "contract_family": fam,
                "contract_variant": "aggregate_or_missing",
                "score": family_mean,
                "family_mean_score": family_mean,
                "family_min_score": family_min,
                "family_max_score": family_max,
                "contract_sensitivity": sensitivity,
                "weakest_contract_variant": "not_available" if not evidence else "family_min",
                "strongest_contract_variant": "not_available" if not evidence else "family_max",
                "contract_localization_flag": bool(evidence and sensitivity is not None and sensitivity >= thresholds.moderate_delta),
                "evidence_available": evidence,
                "evidence_completeness": "contract_family_score_available" if evidence else "missing_optional_contract_columns",
                "interpretation": "Contract-family evidence available." if evidence else "No contract-family columns found; missing evidence is not treated as positive evidence.",
            })
        max_sens = None
        if record_scores:
            # Across all available contract evidence, dispersion can suggest contract-local sensitivity.
            max_sens = max(record_scores) - min(record_scores) if len(record_scores) > 1 else 0.0
        aggs.append({
            "record_id": rid,
            "contract_contrast_score": score_from_delta(max_sens, thresholds.strong_delta) if max_sens is not None else None,
            "contract_sensitivity_max": max_sens,
            "contract_control_evidence_available": bool(record_scores),
            "contract_evidence_mode": "available" if record_scores else "missing_optional_columns",
        })
    return pd.DataFrame(rows), pd.DataFrame(aggs)


def geometry_needed_control_contrast(records: pd.DataFrame, thresholds: Thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    aggs: list[dict[str, Any]] = []
    rels = sorted(records["relation"].dropna().unique().tolist())
    for rel in rels:
        sub = records[records["relation"] == rel]
        scores = {row["carrier"]: row["survival_score"] for _, row in sub.iterrows()}
        compact = scores.get("stability_core_3")
        enriched = scores.get("stability_plus_geometry")
        geom_only = scores.get("geometry_scores_only")
        path_share = scores.get("path_shares_only")
        strict_numeric = scores.get("strict_numeric_all")

        gains = []
        for val in [enriched, geom_only]:
            d = compute_delta(val, compact)
            if d is not None:
                gains.append(d)
        geometry_gain = max(gains) if gains else None
        geometry_score = score_from_delta(geometry_gain, thresholds.strong_delta) if geometry_gain is not None else None
        role = refined_geometry_role(compact, enriched, geom_only, geometry_gain, thresholds)
        rows.append({
            "relation": rel,
            "compact_record_id": f"{rel}__stability_core_3",
            "compact_score": compact,
            "geometry_record_id": f"{rel}__stability_plus_geometry|{rel}__geometry_scores_only",
            "geometry_score": max([v for v in [enriched, geom_only] if v is not None and pd.notna(v)], default=None),
            "enriched_score": enriched,
            "geometry_only_score": geom_only,
            "geometry_gain": geometry_gain,
            "path_share_score": path_share,
            "strict_numeric_score": strict_numeric,
            "geometry_needed_level_obs081": obs081_geometry_label(sub),
            "geometry_needed_level_obs083": role_to_gneed(role),
            "geometry_role": role,
            "geometry_overclaim_risk": geometry_overclaim_risk(role, compact, enriched, geom_only),
            "evidence_available": compact is not None and (enriched is not None or geom_only is not None),
            "interpretation": geometry_interpretation(role),
        })
        for _, row in sub.iterrows():
            aggs.append({
                "record_id": row["record_id"],
                "geometry_contrast_score": geometry_score,
                "geometry_gain": geometry_gain,
                "geometry_control_evidence_available": compact is not None and (enriched is not None or geom_only is not None),
                "geometry_role_for_relation": role,
            })
    return pd.DataFrame(rows), pd.DataFrame(aggs)


def refined_geometry_role(compact: Any, enriched: Any, geom_only: Any, gain: float | None, thresholds: Thresholds) -> str:
    compact_ok = compact is not None and pd.notna(compact) and float(compact) >= thresholds.survival_floor
    if compact_ok and (gain is None or gain < thresholds.moderate_delta):
        return "not_needed_or_redundant"
    if compact_ok and gain is not None and gain >= thresholds.moderate_delta:
        return "geometry_sharpens"
    if not compact_ok and any(v is not None and pd.notna(v) and float(v) >= thresholds.survival_floor for v in [enriched, geom_only]):
        return "geometry_required_candidate_needs_controls"
    return "unresolved"


def role_to_gneed(role: str) -> str:
    return {
        "not_needed_or_redundant": "G1_compact_sufficient_or_redundant",
        "geometry_sharpens": "G2_geometry_sharpens",
        "geometry_required_candidate_needs_controls": "G3_geometry_required_candidate_needs_controls",
        "unresolved": "Gx_unresolved",
    }.get(role, "Gx_unresolved")


def geometry_overclaim_risk(role: str, compact: Any, enriched: Any, geom_only: Any) -> str:
    if role == "geometry_required_candidate_needs_controls":
        return "medium"
    if role == "geometry_sharpens":
        return "medium"
    if role == "not_needed_or_redundant":
        return "high_if_claiming_geometry_required"
    return "high_due_to_missing_evidence"


def geometry_interpretation(role: str) -> str:
    if role == "geometry_sharpens":
        return "Geometry appears to sharpen this relation; this is not proof that geometry is required."
    if role == "geometry_required_candidate_needs_controls":
        return "Geometry may be required, but matched controls are needed before elevating the claim."
    if role == "not_needed_or_redundant":
        return "Compact or non-geometry evidence appears sufficient/redundant; avoid geometry inflation."
    return "Geometry-needed status unresolved due to missing or incomplete evidence."


def obs081_geometry_label(sub: pd.DataFrame) -> str:
    for _, row in sub.iterrows():
        for col in row.index:
            if "geometry_needed" in col.lower() or "gneed" in col.lower():
                txt = norm_text(row[col])
                if txt:
                    return txt
    return "unknown"


def build_failure_localization_matrix(
    records: pd.DataFrame,
    rel_agg: pd.DataFrame,
    car_agg: pd.DataFrame,
    contract_agg: pd.DataFrame,
    geom_agg: pd.DataFrame,
    thresholds: Thresholds,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a conservative localization matrix.

    v2 rule: OBS-082's global result (diffuse failure localization in 23/24
    records) is treated as a prior. Relation/carrier/geometry contrast may
    suggest *where to look next*, but contrast proxies alone are capped and do
    not become high localization evidence. High/moderate localization requires
    an explicit artifact locus, not merely an available contrast table.
    """
    agg = records.copy()
    for frame in [rel_agg, car_agg, contract_agg, geom_agg]:
        if frame is not None and not frame.empty:
            agg = agg.merge(frame, on="record_id", how="left")

    rows: list[dict[str, Any]] = []
    loc_aggs: list[dict[str, Any]] = []
    for _, row in agg.iterrows():
        obs082_fail = collect_failure_text(row)
        diffuse_prior = obs082_diffuse_prior(obs082_fail)
        explicit_locus = explicit_failure_locus_from_text(obs082_fail)

        # Raw contrast proxies. These are not direct localization evidence.
        relation_proxy = row.get("relation_contrast_score")
        carrier_proxy = row.get("carrier_contrast_score")
        contract_proxy = row.get("contract_contrast_score")
        geometry_proxy = row.get("geometry_contrast_score")
        transform_proxy = contract_proxy
        scale_band_proxy = contract_proxy if contract_proxy is not None and pd.notna(contract_proxy) else None
        feature_family_proxy = contract_proxy if contract_proxy is not None and pd.notna(contract_proxy) else None
        structural_resampling_proxy = contract_proxy if contract_proxy is not None and pd.notna(contract_proxy) else None
        cohort_proxy = None
        transition_proxy = relation_proxy

        axis_scores_raw = {
            "relation_local": relation_proxy,
            "carrier_local": carrier_proxy,
            "contract_local": contract_proxy,
            "transform_local": transform_proxy,
            "scale_band_local": scale_band_proxy,
            "feature_family_local": feature_family_proxy,
            "structural_resampling_local": structural_resampling_proxy,
            "geometry_needed_local": geometry_proxy,
            "cohort_local": cohort_proxy,
            "transition_local": transition_proxy,
        }
        clean_axes = {k: clamp01(v) for k, v in axis_scores_raw.items() if v is not None and pd.notna(v)}

        primary_locus = "unresolved_missing_evidence"
        secondary_locus = ""
        primary_proxy = None
        if explicit_locus:
            primary_locus = explicit_locus
            primary_proxy = max(clean_axes.values(), default=0.0)
        elif clean_axes:
            sorted_axes = sorted(clean_axes.items(), key=lambda kv: kv[1], reverse=True)
            primary_locus, primary_proxy = sorted_axes[0]
            secondary_locus = sorted_axes[1][0] if len(sorted_axes) > 1 else ""

        # Evidence basis determines caps. Contrast-only evidence can suggest a
        # locus but cannot overturn the OBS-082 diffuse prior.
        contract_direct = bool(row.get("contract_control_evidence_available")) and primary_locus in {
            "contract_local", "transform_local", "scale_band_local", "feature_family_local", "structural_resampling_local"
        }
        if explicit_locus:
            evidence_basis = "direct_artifact_locus"
            cap = 1.0
        elif contract_direct:
            evidence_basis = "contract_family_proxy"
            cap = 0.50
        elif clean_axes:
            evidence_basis = "contrast_proxy_only"
            cap = 0.35 if diffuse_prior else 0.50
        else:
            evidence_basis = "obs082_diffuse_prior_or_missing"
            cap = 0.25 if diffuse_prior else 0.35

        if primary_proxy is None:
            raw_l = 0.0
        else:
            raw_l = float(primary_proxy)
        localization_score = clamp01(min(raw_l, cap))

        # Explicit artifacts may exceed the prior. Otherwise diffuse prior stays dominant.
        if diffuse_prior and evidence_basis != "direct_artifact_locus":
            localization_score = clamp01(min(localization_score or 0.0, 0.35))

        diffuse_score = clamp01(1.0 - float(localization_score or 0.0))
        confidence = localization_confidence_v2(localization_score, evidence_basis, diffuse_prior, thresholds)
        if confidence == "diffuse":
            limiter = "obs082_diffuse_prior_or_missing_direct_locus"
        elif confidence == "weak":
            limiter = "contrast_proxy_only_not_direct_localization"
        else:
            limiter = "direct_locus_not_primary_limiter"

        outrow = {
            "record_id": row["record_id"],
            "relation": row["relation"],
            "carrier": row["carrier"],
            "obs082_failure_localization": obs082_fail or "OBS-082 global prior: diffuse failure localization in 23/24 records",
            "obs082_diffuse_prior_flag": bool(diffuse_prior),
            "relation_local_score": min_score_for_display(relation_proxy, cap if primary_locus == "relation_local" else 0.35),
            "carrier_local_score": min_score_for_display(carrier_proxy, cap if primary_locus == "carrier_local" else 0.35),
            "contract_local_score": min_score_for_display(contract_proxy, cap if primary_locus == "contract_local" else 0.50),
            "transform_local_score": min_score_for_display(transform_proxy, cap if primary_locus == "transform_local" else 0.50),
            "scale_band_local_score": min_score_for_display(scale_band_proxy, cap if primary_locus == "scale_band_local" else 0.50),
            "feature_family_local_score": min_score_for_display(feature_family_proxy, cap if primary_locus == "feature_family_local" else 0.50),
            "structural_resampling_local_score": min_score_for_display(structural_resampling_proxy, cap if primary_locus == "structural_resampling_local" else 0.50),
            "geometry_needed_local_score": min_score_for_display(geometry_proxy, cap if primary_locus == "geometry_needed_local" else 0.35),
            "cohort_local_score": cohort_proxy,
            "transition_local_score": min_score_for_display(transition_proxy, cap if primary_locus == "transition_local" else 0.35),
            "diffuse_score": diffuse_score,
            "primary_failure_locus": primary_locus,
            "secondary_failure_locus": secondary_locus,
            "localization_confidence": confidence,
            "failure_localization_score": localization_score,
            "localization_limiter": limiter,
            "localization_evidence_basis": evidence_basis,
            "explicit_failure_locus_source": explicit_locus or "none",
            "interpretation": localization_interpretation_v2(confidence, primary_locus, evidence_basis),
        }
        rows.append(outrow)
        loc_aggs.append({
            "record_id": row["record_id"],
            "failure_localization_score": localization_score,
            "primary_failure_locus": primary_locus,
            "localization_confidence": confidence,
            "diffuse_score": diffuse_score,
            "localization_evidence_basis": evidence_basis,
            "obs082_diffuse_prior_flag": bool(diffuse_prior),
        })
    return pd.DataFrame(rows), pd.DataFrame(loc_aggs)



def collect_failure_text(row: pd.Series) -> str:
    parts = []
    for c in row.index:
        lc = c.lower()
        if any(k in lc for k in ["failure", "block", "limiter", "localization", "locus", "diffuse"]):
            txt = norm_text(row[c])
            if txt:
                parts.append(txt)
    return "; ".join(dict.fromkeys(parts))


def obs082_diffuse_prior(text: str) -> bool:
    """Return True unless explicit text clearly contradicts the OBS-082 diffuse prior."""
    t = norm_text(text).lower()
    if not t:
        return True
    diffuse_markers = ["diffuse", "generic", "broad", "weak", "missing", "unclear", "insufficient", "localization_limited"]
    if any(m in t for m in diffuse_markers):
        return True
    direct_markers = ["relation_local", "carrier_local", "contract_local", "scale_band_local", "feature_family_local", "geometry_needed_local", "cohort_local", "transition_local"]
    return not any(m in t for m in direct_markers)


def explicit_failure_locus_from_text(text: str) -> str:
    t = norm_text(text).lower()
    # Require explicit machine-like locus markers, not general English words such
    # as "localization" or "geometry" appearing in generic repair text.
    ordered = [
        "relation_local", "carrier_local", "contract_local", "transform_local",
        "scale_band_local", "feature_family_local", "structural_resampling_local",
        "geometry_needed_local", "cohort_local", "transition_local",
        "boundary_local", "seam_local",
    ]
    for marker in ordered:
        if marker in t:
            return marker
    return ""


def min_score_for_display(value: Any, cap: float) -> float | None:
    if value is None or pd.isna(value):
        return None
    return clamp01(min(float(value), cap))


def localization_confidence_v2(score: float | None, evidence_basis: str, diffuse_prior: bool, thresholds: Thresholds) -> str:
    if score is None or pd.isna(score):
        return "diffuse"
    score = float(score)
    if evidence_basis != "direct_artifact_locus":
        if score >= 0.30:
            return "weak"
        return "diffuse"
    if score >= 0.60 and not diffuse_prior:
        return "high"
    if score >= 0.45:
        return "moderate"
    if score >= 0.30:
        return "weak"
    return "diffuse"


def localization_confidence(primary_score: float | None, diffuse_score: float | None, thresholds: Thresholds) -> str:
    # Retained for backward compatibility with earlier helper calls; v2 uses
    # localization_confidence_v2 in the main matrix builder.
    if primary_score is None or pd.isna(primary_score):
        return "diffuse"
    if diffuse_score is None:
        diffuse_score = 1.0 - primary_score
    if primary_score >= 0.60 and diffuse_score <= 0.30:
        return "high"
    if primary_score >= 0.45 and diffuse_score <= 0.45:
        return "moderate"
    if primary_score >= 0.30:
        return "weak"
    return "diffuse"


def localization_interpretation_v2(confidence: str, locus: str, evidence_basis: str) -> str:
    if evidence_basis != "direct_artifact_locus":
        return (
            f"Localization remains {confidence}: {locus} is derived from {evidence_basis}, "
            "so it is a next-test pointer, not a localized failure claim."
        )
    if confidence in {"high", "moderate"}:
        return f"Failure localization is {confidence} with explicit primary locus {locus}; still diagnostic-only."
    return "Failure remains weak/diffuse despite explicit locus text; no repair target should be claimed."


def localization_interpretation(confidence: str, locus: str) -> str:
    if confidence in {"high", "moderate"}:
        return f"Failure localization is {confidence} with primary locus {locus}; still diagnostic-only until repair specificity and controls are sufficient."
    if confidence == "weak":
        return f"Weak localization signal around {locus}; failure remains a limiter."
    return "Failure remains diffuse or unresolved; no repair target should be claimed."



def repair_specificity_sharpening(records: pd.DataFrame, loc_agg: pd.DataFrame, thresholds: Thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = records.merge(loc_agg, on="record_id", how="left") if not loc_agg.empty else records.copy()
    rows: list[dict[str, Any]] = []
    aggs: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        original = collect_repair_text(row)
        relation = row["relation"]
        carrier = row["carrier"]
        locus = norm_text(row.get("primary_failure_locus", "unresolved_missing_evidence")) or "unresolved_missing_evidence"
        loc_score = row.get("failure_localization_score")
        loc_basis = norm_text(row.get("localization_evidence_basis", "obs082_diffuse_prior_or_missing"))
        loc_ok_direct = (
            loc_score is not None and pd.notna(loc_score)
            and float(loc_score) >= thresholds.localization_threshold
            and loc_basis == "direct_artifact_locus"
        )
        metric_text = observed_metric_text(row)
        matched_control_text = observed_matched_control_text(row)
        level, reason = repair_level_v2(original, relation, carrier, locus, loc_ok_direct, metric_text, matched_control_text)
        score = repair_score(level)
        sharpened = sharpened_repair_annotation_v2(level, relation, carrier, locus, loc_basis)
        hypothesis_ready = False  # Deliberately conservative: OBS-083 does not promote to hypothesis-ready.
        why_not = "OBS-083 does not perform interventions and does not promote records to actionability; "
        why_not += f"repair specificity only {level}: {reason}"
        rows.append({
            "record_id": row["record_id"],
            "relation": relation,
            "carrier": carrier,
            "obs081_repair_recommendation": original or "missing",
            "obs082_repair_specificity": first_text(row, ["obs082_repair_specificity_score", "obs082_repair_specificity", "repair_specificity_score"]),
            "primary_failure_locus": locus,
            "failure_localization_score": loc_score,
            "localization_evidence_basis": loc_basis,
            "repair_specificity_level": level,
            "repair_specificity_score": score,
            "sharpened_repair_annotation": sharpened,
            "required_missing_evidence": missing_repair_evidence(level),
            "candidate_metric": metric_text or "missing_specific_metric",
            "candidate_metric_available": bool(metric_text),
            "matched_negative_control_required": matched_control_text or "missing_matched_control",
            "matched_negative_control_available_for_repair": bool(matched_control_text),
            "hypothesis_ready_flag": hypothesis_ready,
            "why_not_hypothesis_ready": why_not,
            "evidence_basis": repair_evidence_basis(original, loc_basis, metric_text, matched_control_text),
            "interpretation": "Repair annotation sharpened diagnostically only; no repair or intervention claim is made.",
        })
        aggs.append({
            "record_id": row["record_id"],
            "repair_specificity_level": level,
            "repair_specificity_score": score,
            "candidate_metric_available": bool(metric_text),
            "matched_negative_control_available_for_repair": bool(matched_control_text),
        })
    return pd.DataFrame(rows), pd.DataFrame(aggs)


def collect_repair_text(row: pd.Series) -> str:
    parts = []
    for c in row.index:
        lc = c.lower()
        if "repair" in lc or "recommend" in lc:
            txt = norm_text(row[c])
            if txt and txt.lower() not in {"nan", "none"}:
                parts.append(txt)
    return "; ".join(dict.fromkeys(parts))


def observed_metric_text(row: pd.Series) -> str:
    for c in row.index:
        lc = c.lower()
        if "candidate_metric" in lc or lc.endswith("metric") or "expected_metric" in lc:
            txt = norm_text(row[c])
            if txt and "missing" not in txt.lower() and txt.lower() not in {"nan", "none"}:
                return txt
    return ""


def observed_matched_control_text(row: pd.Series) -> str:
    for c in row.index:
        lc = c.lower()
        if "matched_negative_control" in lc or "repair_control" in lc:
            txt = norm_text(row[c])
            if txt and "missing" not in txt.lower() and txt.lower() not in {"nan", "none"}:
                return txt
    return ""


def repair_level_v2(
    original: str,
    relation: str,
    carrier: str,
    locus: str,
    loc_ok_direct: bool,
    metric_text: str,
    matched_control_text: str,
) -> tuple[str, str]:
    if not original:
        return "R0", "no repair annotation available"
    # OBS-082 generic repair specificity is the prior. Known registry relation
    # and carrier can sharpen an annotation to R3, but not to R4 unless failure
    # localization and a metric are explicitly artifact-supported.
    if loc_ok_direct and metric_text and matched_control_text:
        return "R5", "relation, carrier, direct locus, metric, and matched control are present; still requires separate audit"
    if loc_ok_direct and metric_text:
        return "R4", "direct locus and metric present; matched negative control for repair missing"
    if relation and carrier:
        return "R3", "relation and carrier are known from registry; direct locus/metric/control insufficient"
    if relation:
        return "R2", "relation known, carrier/locus/metric/control insufficient"
    return "R1", "generic repair annotation"


def repair_level(original: str, relation: str, carrier: str, locus: str, loc_ok: bool) -> tuple[str, str]:
    # Backward-compatible wrapper. v2 repair_specificity_sharpening uses repair_level_v2.
    return repair_level_v2(original, relation, carrier, locus, loc_ok, "", "")


def repair_score(level: str) -> float:
    return {
        "R0": 0.00,
        "R1": 0.20,
        "R2": 0.40,
        "R3": 0.55,
        "R4": 0.70,
        "R5": 0.85,
    }.get(level, 0.0)


def sharpened_repair_annotation_v2(level: str, relation: str, carrier: str, locus: str, loc_basis: str) -> str:
    if level == "R0":
        return "No repair annotation available; retain descriptive-only status unless future evidence supplies a target."
    if level in {"R4", "R5"}:
        return (
            f"For relation {relation} through carrier {carrier}, an explicit artifact locus {locus} is available. "
            "This remains a diagnostic repair-hypothesis candidate only and requires a separate readiness audit."
        )
    if level == "R3":
        return (
            f"For relation {relation} through carrier {carrier}, repair language can be made relation+carrier-specific, "
            f"but localization evidence is {loc_basis}; no repair target is claimed."
        )
    if level == "R2":
        return f"For relation {relation}, repair language remains too broad without carrier and direct failure-locus specificity."
    return "Generic repair annotation; no relation/carrier/failure-locus-specific hypothesis should be claimed."


def sharpened_repair_annotation(level: str, relation: str, carrier: str, locus: str, loc_ok: bool) -> str:
    return sharpened_repair_annotation_v2(level, relation, carrier, locus, "direct_artifact_locus" if loc_ok else "not_direct")


def missing_repair_evidence(level: str) -> str:
    return {
        "R0": "repair annotation; relation/carrier/locus; metric; matched control",
        "R1": "relation; carrier; localized failure site; metric; matched control",
        "R2": "carrier; localized failure site; metric; matched control",
        "R3": "direct localized failure site; predeclared metric; matched control",
        "R4": "matched negative control; separate hypothesis-readiness audit",
        "R5": "separate hypothesis-readiness audit before any actionability claim",
    }.get(level, "unknown")


def repair_evidence_basis(original: str, loc_basis: str, metric_text: str, matched_control_text: str) -> str:
    if not original:
        return "missing_original_repair_annotation"
    parts = ["obs081_annotation", f"localization={loc_basis}"]
    parts.append("metric=available" if metric_text else "metric=missing")
    parts.append("repair_control=available" if matched_control_text else "repair_control=missing")
    return ";".join(parts)



def diagnostic_subclasses(
    records: pd.DataFrame,
    rel_agg: pd.DataFrame,
    car_agg: pd.DataFrame,
    contract_agg: pd.DataFrame,
    geom_agg: pd.DataFrame,
    loc_agg: pd.DataFrame,
    repair_agg: pd.DataFrame,
    thresholds: Thresholds,
) -> pd.DataFrame:
    out = records[["record_id", "relation", "carrier", "obs082_class", "survival_score", "survival_score_source"]].copy()
    for frame in [rel_agg, car_agg, contract_agg, geom_agg, loc_agg, repair_agg]:
        if frame is not None and not frame.empty:
            out = out.merge(frame, on="record_id", how="left")

    rows = []
    for _, row in out.iterrows():
        nr = val_or_zero(row.get("relation_contrast_score"))
        nc = val_or_zero(row.get("carrier_contrast_score"))
        nk = val_or_zero(row.get("contract_contrast_score"))
        ng = val_or_zero(row.get("geometry_contrast_score"))
        np = 0.0  # Optional permutation evidence is not assumed present.
        n_score = clamp01(0.30 * nr + 0.30 * nc + 0.15 * nk + 0.15 * ng + 0.10 * np)
        l_score = row.get("failure_localization_score")
        l_score = clamp01(l_score) if l_score is not None and pd.notna(l_score) else 0.0
        q_score = row.get("repair_specificity_score")
        q_score = clamp01(q_score) if q_score is not None and pd.notna(q_score) else 0.0
        survival = row.get("survival_score")
        survival_ok = survival is not None and pd.notna(survival) and float(survival) >= thresholds.survival_floor
        repair_level_value = norm_text(row.get("repair_specificity_level", "R0")) or "R0"
        loc_basis = norm_text(row.get("localization_evidence_basis", "obs082_diffuse_prior_or_missing"))
        candidate_metric_available = bool(row.get("candidate_metric_available"))
        matched_control_available = bool(row.get("matched_negative_control_available_for_repair"))
        completeness = evidence_completeness(row, [
            "relation_control_evidence_available",
            "carrier_control_evidence_available",
            "contract_control_evidence_available",
            "geometry_control_evidence_available",
        ])
        c4_evidence_ok = (
            loc_basis == "direct_artifact_locus"
            and repair_level_value in {"R4", "R5"}
            and candidate_metric_available
            and matched_control_available
            and not completeness.startswith("0/")
        )

        subclass, primary, secondary, next_test = assign_subclass(
            n_score, l_score, q_score, survival_ok, thresholds, c4_evidence_ok, repair_level_value, loc_basis
        )
        rationale = subclass_rationale(subclass, n_score, l_score, q_score, survival_ok, c4_evidence_ok, loc_basis, repair_level_value)
        rows.append({
            "record_id": row["record_id"],
            "relation": row["relation"],
            "carrier": row["carrier"],
            "obs081_rig_status": first_text(row, ["rig_status", "obs081_rig_status"]),
            "obs082_class": row.get("obs082_class", "unknown"),
            "survival_score": survival,
            "survival_score_source": row.get("survival_score_source", ""),
            "survival_evidence_available": survival is not None and pd.notna(survival),
            "relation_contrast_score": row.get("relation_contrast_score"),
            "carrier_contrast_score": row.get("carrier_contrast_score"),
            "contract_contrast_score": row.get("contract_contrast_score"),
            "geometry_contrast_score": row.get("geometry_contrast_score"),
            "permutation_contrast_score": None,
            "negative_control_strength_score": n_score,
            "negative_control_evidence_completeness": completeness,
            "failure_localization_score": l_score,
            "primary_failure_locus": row.get("primary_failure_locus", "unknown"),
            "localization_confidence": row.get("localization_confidence", "unknown"),
            "localization_evidence_basis": loc_basis,
            "obs082_diffuse_prior_flag": row.get("obs082_diffuse_prior_flag", True),
            "repair_specificity_level": repair_level_value,
            "repair_specificity_score": q_score,
            "candidate_metric_available": candidate_metric_available,
            "matched_negative_control_available_for_repair": matched_control_available,
            "c4_evidence_gate_passed": bool(c4_evidence_ok),
            "subclass": subclass,
            "readiness_statement": "diagnostic-only" if subclass != "C4_promising_next_test_candidate" else "diagnostic-only_promising_next-test_candidate",
            "primary_limiter": primary,
            "secondary_limiter": secondary,
            "candidate_next_test_type": next_test,
            "rationale": rationale,
            "guardrail_note": GUARDRAIL,
        })
    return pd.DataFrame(rows)


def val_or_zero(v: Any) -> float:
    if v is None or pd.isna(v):
        return 0.0
    return float(clamp01(v) or 0.0)


def evidence_completeness(row: pd.Series, bool_cols: list[str]) -> str:
    available = 0
    total = 0
    for c in bool_cols:
        if c in row.index:
            total += 1
            if bool(row[c]):
                available += 1
    if total == 0:
        return "no_component_evidence_flags_available"
    return f"{available}/{total}_contrast_components_available"


def assign_subclass(
    n: float,
    l: float,
    q: float,
    survival_ok: bool,
    thresholds: Thresholds,
    c4_evidence_ok: bool = False,
    repair_level_value: str = "R0",
    loc_basis: str = "obs082_diffuse_prior_or_missing",
) -> tuple[str, str, str, str]:
    if not survival_ok:
        return "C0_descriptive-only", "survival_or_score_basis_limited", "contrast_localization_repair_unresolved", "none"
    if n >= thresholds.c4_contrast and l >= thresholds.c4_localization and q >= thresholds.c4_repair and c4_evidence_ok:
        return "C4_promising_next_test_candidate", "none_within_class_C", "requires_separate_hypothesis_readiness_audit", "predeclared_non_interventional_next_test"
    # Strong contrast but no direct localization remains localization-limited,
    # even if contrast proxies produce a weak localization pointer.
    if n >= thresholds.contrast_threshold and (l < thresholds.localization_threshold or loc_basis != "direct_artifact_locus"):
        return "C2_localization-limited", "failure_localization", "repair_specificity", "localization"
    if n >= thresholds.contrast_threshold and l >= thresholds.localization_threshold and q < thresholds.repair_threshold:
        return "C3_repair-specificity-limited", "repair_specificity", "none", "repair_specificity"
    if n >= thresholds.contrast_threshold and l >= thresholds.localization_threshold and q >= thresholds.repair_threshold:
        return "C3_repair-specificity-limited", "repair_hypothesis_evidence_gate", "candidate_metric_or_matched_control_missing", "repair_specificity"
    if n < thresholds.contrast_threshold and (l >= 0.25 or q >= 0.35):
        return "C1_contrast-limited", "negative_control_contrast", "localization_or_repair_secondary", "matched_controls"
    return "C0_descriptive-only", "broad_diagnostic_limits", "negative_control_localization_repair", "none"


def subclass_rationale(subclass: str, n: float, l: float, q: float, survival_ok: bool, c4_evidence_ok: bool, loc_basis: str, repair_level_value: str) -> str:
    if not survival_ok:
        return "Survival/score evidence is missing or below floor; record remains descriptive-only."
    gate = "passed" if c4_evidence_ok else "failed"
    return (
        f"Assigned {subclass}: N={n:.3f}, L={l:.3f}, Q={q:.3f}; "
        f"localization_basis={loc_basis}; repair_level={repair_level_value}; C4_evidence_gate={gate}. "
        "C4 remains diagnostic-only if assigned."
    )



def readiness_delta(subclasses: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in subclasses.iterrows():
        rec = records[records["record_id"] == row["record_id"]]
        recrow = rec.iloc[0] if not rec.empty else pd.Series(dtype=object)
        old_n, _ = numeric_value(recrow, [*NEGATIVE_SCORE_COLS, *[f"obs082_{c}" for c in NEGATIVE_SCORE_COLS]])
        old_l, _ = numeric_value(recrow, [*LOCALIZATION_SCORE_COLS, *[f"obs082_{c}" for c in LOCALIZATION_SCORE_COLS]])
        old_q, _ = numeric_value(recrow, [*REPAIR_SCORE_COLS, *[f"obs082_{c}" for c in REPAIR_SCORE_COLS]])
        rows.append({
            "record_id": row["record_id"],
            "obs082_class": row.get("obs082_class", "unknown"),
            "obs083_subclass": row["subclass"],
            "obs082_negative_control_score": old_n,
            "obs083_negative_control_score": row["negative_control_strength_score"],
            "delta_negative_control": compute_delta(row["negative_control_strength_score"], old_n) if old_n is not None else None,
            "obs082_failure_localization_score": old_l,
            "obs083_failure_localization_score": row["failure_localization_score"],
            "delta_failure_localization": compute_delta(row["failure_localization_score"], old_l) if old_l is not None else None,
            "obs082_repair_specificity_score": old_q,
            "obs083_repair_specificity_score": row["repair_specificity_score"],
            "delta_repair_specificity": compute_delta(row["repair_specificity_score"], old_q) if old_q is not None else None,
            "status_change_summary": "OBS-083 refines Class C only; no actionability or candidate-ready promotion claimed.",
        })
    return pd.DataFrame(rows)


def blocker_refinement(subclasses: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in subclasses.iterrows():
        rec = records[records["record_id"] == row["record_id"]]
        recrow = rec.iloc[0] if not rec.empty else pd.Series(dtype=object)
        old_blockers = collect_blocker_text(recrow)
        primary = row["primary_limiter"]
        secondary = row["secondary_limiter"]
        rows.append({
            "record_id": row["record_id"],
            "obs082_blockers": old_blockers or "unknown",
            "obs083_primary_blocker": primary,
            "obs083_secondary_blocker": secondary,
            "blocker_removed_flag": False,
            "new_blocker_flag": False,
            "remaining_missing_evidence": remaining_missing_evidence(row),
            "recommended_next_evidence_type": row["candidate_next_test_type"],
            "guardrail_note": GUARDRAIL,
        })
    return pd.DataFrame(rows)


def collect_blocker_text(row: pd.Series) -> str:
    parts = []
    for c in row.index:
        if any(k in c.lower() for k in ["block", "limiter", "fatal"]):
            txt = norm_text(row[c])
            if txt:
                parts.append(txt)
    return "; ".join(dict.fromkeys(parts))


def remaining_missing_evidence(row: pd.Series) -> str:
    missing = []
    if row["negative_control_strength_score"] < 0.50:
        missing.append("stronger_matched_negative_controls")
    if row["failure_localization_score"] < 0.50:
        missing.append("sharper_failure_localization")
    if row["repair_specificity_score"] < 0.50:
        missing.append("more_specific_repair_annotation")
    if row.get("negative_control_evidence_completeness", "").startswith("0/"):
        missing.append("evidence_completeness")
    return ";".join(missing) if missing else "separate_hypothesis_readiness_audit_required"


def write_report(
    out_path: Path,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    subclasses: pd.DataFrame,
    relation_controls: pd.DataFrame,
    carrier_controls: pd.DataFrame,
    failure_matrix: pd.DataFrame,
    repair_table: pd.DataFrame,
) -> None:
    counts = subclasses["subclass"].value_counts().sort_index() if not subclasses.empty else pd.Series(dtype=int)
    c4_count = int((subclasses["subclass"] == "C4_promising_next_test_candidate").sum()) if not subclasses.empty else 0
    input_rows = manifest[["artifact_label", "exists", "rows", "read_status"]].to_markdown(index=False)
    subclass_rows = counts.rename_axis("subclass").reset_index(name="count").to_markdown(index=False)
    top_sub = subclasses[[
        "record_id", "subclass", "negative_control_strength_score", "failure_localization_score",
        "repair_specificity_score", "primary_limiter",
    ]].to_markdown(index=False) if not subclasses.empty else "No records scored."

    text = f"""# OBS-083 — RIG Negative-Control and Failure-Localization Strengthening

## State

Diagnostic subclassing audit completed with v2 conservative prior-aware gates.

{GUARDRAIL}

## Scope

- Input registry directory: `{args.registry_dir}`
- Output directory: `{args.output_dir}`
- Uses OBS-081/OBS-082 artifacts when available.
- Uses conservative fallbacks when optional OBS-080/contract columns are missing.
- Missing evidence is recorded as missing and is never counted as positive evidence.

## Inputs

{input_rows}

## Thresholds

| threshold | value |
|---|---:|
| strong_delta | {args.strong_delta:.3f} |
| moderate_delta | {args.moderate_delta:.3f} |
| contrast_threshold | {args.contrast_threshold:.3f} |
| localization_threshold | {args.localization_threshold:.3f} |
| repair_threshold | {args.repair_threshold:.3f} |
| c4_contrast | {args.c4_contrast:.3f} |
| c4_localization | {args.c4_localization:.3f} |
| c4_repair | {args.c4_repair:.3f} |
| survival_floor | {args.survival_floor:.3f} |

## Method summary

OBS-083 refines the OBS-082 Class C result by constructing conservative prior-aware diagnostics:

1. matched negative-control design rows;
2. relation-control contrasts;
3. carrier-control contrasts;
4. contract/transformation control contrasts where artifact columns exist;
5. geometry-needed control contrasts;
6. failure-localization matrix that treats OBS-082 diffuse localization as the prior;
7. repair-specificity sharpening table gated by direct locus, metric, and matched-control evidence;
8. C0–C4 diagnostic subclass assignments.

C4 is explicitly retained as **diagnostic-only / promising next-test candidate**. It is not candidate-ready, actionable, causal, or intervention-ready. In v2, C4 additionally requires direct localization evidence, R4/R5 repair specificity, a candidate metric, and a matched negative control for the repair claim.

## Subclass counts

{subclass_rows}

## Diagnostic subclass assignments

{top_sub}

## Relation-control evidence

- Rows written: {len(relation_controls)}
- Evidence-available rows: {int(relation_controls.get('evidence_available', pd.Series(dtype=bool)).fillna(False).sum()) if not relation_controls.empty else 0}

Relation-control contrast tests whether a target relation differs from matched relations under the same carrier. Weak relation contrast remains a negative-control limiter.

## Carrier-control evidence

- Rows written: {len(carrier_controls)}
- Evidence-available rows: {int(carrier_controls.get('evidence_available', pd.Series(dtype=bool)).fillna(False).sum()) if not carrier_controls.empty else 0}

Carrier-control contrast tests whether a carrier has a differentiated role rather than acting as an overbroad separability substrate.

## Failure-localization evidence

- Rows written: {len(failure_matrix)}
- High/moderate localization rows: {int(failure_matrix.get('localization_confidence', pd.Series(dtype=str)).isin(['high', 'moderate']).sum()) if not failure_matrix.empty else 0}
- Direct artifact locus rows: {int((failure_matrix.get('localization_evidence_basis', pd.Series(dtype=str)) == 'direct_artifact_locus').sum()) if not failure_matrix.empty else 0}

Failure localization is interpreted as diagnostic addressability only. It is not a causal mechanism or repair target unless future criteria are met.

## Repair-specificity evidence

- Rows written: {len(repair_table)}
- R4 diagnostic repair-candidate annotations: {int((repair_table.get('repair_specificity_level', pd.Series(dtype=str)) == 'R4').sum()) if not repair_table.empty else 0}
- R3 relation+carrier-specific annotations: {int((repair_table.get('repair_specificity_level', pd.Series(dtype=str)) == 'R3').sum()) if not repair_table.empty else 0}
- Hypothesis-ready rows: 0

OBS-083 deliberately does not promote repair annotations to actionability. R3 means relation+carrier-specific annotation; R4 requires direct locus plus metric evidence and still is not a validated repair hypothesis.

## Canonical result statement

OBS-083 refines the OBS-082 diagnostic-only registry by constructing matched relation, carrier, contract, geometry-needed, and failure-localization contrasts over the OBS-081 relation × carrier records. The audit assigns each record to a diagnostic subclass: C0 descriptive-only, C1 contrast-limited, C2 localization-limited, C3 repair-specificity-limited, or C4 promising next-test candidate. OBS-083 performs no interventions and establishes no causality, control, actionability, external generalization, or formal topology.

## Outputs

- `obs083_input_manifest.csv`
- `obs083_matched_negative_control_design.csv`
- `obs083_relation_control_contrast.csv`
- `obs083_carrier_control_contrast.csv`
- `obs083_contract_control_contrast.csv`
- `obs083_geometry_needed_control_contrast.csv`
- `obs083_failure_localization_matrix.csv`
- `obs083_repair_specificity_sharpening.csv`
- `obs083_diagnostic_subclass_assignments.csv`
- `obs083_readiness_delta_from_obs082.csv`
- `obs083_blocker_refinement.csv`
- `obs083_report.md`

## Limitations

- Optional contract-family evidence is only scored when available in the loaded artifacts.
- Missing evidence is not imputed.
- Contrast-derived localization is capped as a conservative proxy and remains diagnostic; it cannot by itself overcome the OBS-082 diffuse-localization prior.
- C4 remains within Class C unless a separate readiness audit proves otherwise.
"""
    out_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OBS-083 diagnostic audit for RIG negative-control and failure-localization strengthening."
    )
    parser.add_argument("--registry-dir", default="outputs/rig_registry", help="Directory containing OBS-081/082 artifacts.")
    parser.add_argument(
        "--output-dir",
        default="outputs/rig_registry/obs083_negative_control_localization",
        help="Directory for OBS-083 outputs.",
    )
    parser.add_argument("--strong-delta", type=float, default=0.10, help="Delta treated as strong contrast.")
    parser.add_argument("--moderate-delta", type=float, default=0.05, help="Delta treated as moderate contrast.")
    parser.add_argument("--contrast-threshold", type=float, default=0.50, help="Minimum aggregate N for contrast-sufficient diagnostic refinement.")
    parser.add_argument("--localization-threshold", type=float, default=0.50, help="Minimum L for localization-sufficient diagnostic refinement.")
    parser.add_argument("--repair-threshold", type=float, default=0.50, help="Minimum Q for repair-specificity-sufficient diagnostic refinement.")
    parser.add_argument("--c4-contrast", type=float, default=0.60, help="C4 diagnostic-only promising next-test N threshold.")
    parser.add_argument("--c4-localization", type=float, default=0.60, help="C4 diagnostic-only promising next-test L threshold.")
    parser.add_argument("--c4-repair", type=float, default=0.50, help="C4 diagnostic-only promising next-test Q threshold.")
    parser.add_argument("--survival-floor", type=float, default=0.45, help="Minimum survival/score basis to avoid C0 due to missing/weak survival evidence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_dir = Path(args.registry_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = Thresholds(
        strong_delta=args.strong_delta,
        moderate_delta=args.moderate_delta,
        contrast_threshold=args.contrast_threshold,
        localization_threshold=args.localization_threshold,
        repair_threshold=args.repair_threshold,
        c4_contrast=args.c4_contrast,
        c4_localization=args.c4_localization,
        c4_repair=args.c4_repair,
        survival_floor=args.survival_floor,
    )

    manifest = file_manifest(registry_dir)
    data = load_inputs(registry_dir)
    records = build_base_records(data)

    design = matched_negative_control_design(records, data)
    relation_table, relation_agg = relation_control_contrast(records, thresholds)
    carrier_table, carrier_agg = carrier_control_contrast(records, thresholds)
    contract_table, contract_agg = contract_control_contrast(records, data, thresholds)
    geometry_table, geometry_agg = geometry_needed_control_contrast(records, thresholds)
    failure_matrix, localization_agg = build_failure_localization_matrix(
        records, relation_agg, carrier_agg, contract_agg, geometry_agg, thresholds
    )
    repair_table, repair_agg = repair_specificity_sharpening(records, localization_agg, thresholds)
    subclasses = diagnostic_subclasses(
        records, relation_agg, carrier_agg, contract_agg, geometry_agg, localization_agg, repair_agg, thresholds
    )
    delta = readiness_delta(subclasses, records)
    blockers = blocker_refinement(subclasses, records)

    outputs = {
        "obs083_input_manifest.csv": manifest,
        "obs083_matched_negative_control_design.csv": design,
        "obs083_relation_control_contrast.csv": relation_table,
        "obs083_carrier_control_contrast.csv": carrier_table,
        "obs083_contract_control_contrast.csv": contract_table,
        "obs083_geometry_needed_control_contrast.csv": geometry_table,
        "obs083_failure_localization_matrix.csv": failure_matrix,
        "obs083_repair_specificity_sharpening.csv": repair_table,
        "obs083_diagnostic_subclass_assignments.csv": subclasses,
        "obs083_readiness_delta_from_obs082.csv": delta,
        "obs083_blocker_refinement.csv": blockers,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False)

    write_report(
        output_dir / "obs083_report.md",
        args,
        manifest,
        subclasses,
        relation_table,
        carrier_table,
        failure_matrix,
        repair_table,
    )

    counts = subclasses["subclass"].value_counts().sort_index().to_dict() if not subclasses.empty else {}
    print("OBS-083 diagnostic refinement complete")
    print(f"records_scored={len(subclasses)}")
    print(f"output_dir={output_dir}")
    print(f"subclass_counts={counts}")
    print(GUARDRAIL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
