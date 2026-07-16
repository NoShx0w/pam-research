#!/usr/bin/env python3
"""
obs084a_bridge_resolution_and_partition_freeze.py

OBS-084a — Bridge Resolution and Two-Way Partition Freeze
==========================================================

v4 patch
--------
Canonical carrier membership now overrides generic name-based leakage heuristics.
In particular, ``jaccard_loss`` remains a geometry predictor because it is
explicitly admitted by the agreeing OBS-080c/080d carrier manifests. Carrier
freeze gates require exact ordered equality for canonical contracts, complete
feature presence, predictor allowance, count agreement, and non-unresolved row
statuses.

Purpose
-------
Resolve the final pre-discovery design decisions for OBS-084 after canonical
lineage bridging. The script converts bridge proposals into an inspectable,
versioned freeze package covering:

* one validated scientific composite observation-key specification;
* exact carrier feature manifests;
* final field-role assignments;
* one selected structural cluster unit;
* a deterministic two-way discovery/confirmation partition;
* per-record partition-balance audits for the 12 C2 records;
* a predeclared seam-distance discretization protocol;
* a frozen candidate-support vocabulary;
* source hashes and repository commit identity.

The script is conservative. When no human-reviewed override is supplied, it
marks rule-based resolutions as ``proposed_auto`` rather than pretending they
were manually reviewed. A formal freeze is emitted only if all hard gates pass.

This script does NOT:

* generate candidate failure supports;
* inspect confirmation outcomes;
* compute localization contrasts;
* assign FL levels or direct witnesses;
* propose repairs or interventions;
* establish causality, control, actionability, external generalization, or
  formal topology.

Default inputs
--------------
outputs/rig_registry/obs084_direct_failure_witness/lineage_bridge/

Optional human-review overrides
-------------------------------
Pass ``--review-dir PATH`` containing any of:

* reviewed_observation_key_spec.csv
* reviewed_carrier_feature_manifest.csv
* reviewed_field_roles.csv
* reviewed_cluster_unit_selection.csv
* reviewed_seam_discretization_protocol.csv
* reviewed_support_vocabulary.csv

Overrides are validated and copied into the frozen outputs. Missing override
files fall back to explicit rule-based proposals.

Default outputs
---------------
outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution/
    obs084a_reviewed_observation_key_spec.csv
    obs084a_reviewed_carrier_feature_manifest.csv
    obs084a_reviewed_field_roles.csv
    obs084a_cluster_unit_selection.csv
    obs084a_two_way_partition_manifest.csv
    obs084a_partition_balance_final.csv
    obs084a_seam_discretization_protocol.csv
    obs084a_support_vocabulary_freeze.csv
    obs084a_source_hash_manifest.csv
    obs084a_bridge_resolution_summary.csv
    obs084a_bridge_resolution_report.md
    obs084a_freeze_manifest.json

Run
---
PYTHONPATH=src .venv/bin/python \
  experiments/studies/obs084a_bridge_resolution_and_partition_freeze.py

Guardrail
---------
A successful freeze means only that the discovery/confirmation design is
inspectable and immutable enough to begin OBS-084 candidate discovery. It is not
evidence that any direct failure support exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


SCRIPT_VERSION = "1.2.0"
DEFAULT_LINEAGE_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/lineage_bridge"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution"
)

REQUIRED_LINEAGE_FILES = {
    "canonical_sources": "obs084a_canonical_source_manifest.csv",
    "record_sources": "obs084a_registry_record_to_prediction_source.csv",
    "key_crosswalk": "obs084a_observation_key_crosswalk.csv",
    "carrier_features": "obs084a_carrier_feature_resolution.csv",
    "cluster_hierarchy": "obs084a_cluster_hierarchy.csv",
    "partition_balance": "obs084a_partition_balance_by_record.csv",
    "support_families": "obs084a_support_family_resolution.csv",
    "field_roles": "obs084a_field_role_classification.csv",
}

REVIEW_FILES = {
    "observation_key": "reviewed_observation_key_spec.csv",
    "carrier_features": "reviewed_carrier_feature_manifest.csv",
    "field_roles": "reviewed_field_roles.csv",
    "cluster_unit": "reviewed_cluster_unit_selection.csv",
    "seam_protocol": "reviewed_seam_discretization_protocol.csv",
    "support_vocabulary": "reviewed_support_vocabulary.csv",
}

CANONICAL_CARRIERS = (
    "stability_core_3",
    "geometry_scores_only",
    "path_shares_only",
    "stability_plus_geometry",
    "no_window",
    "strict_numeric_all",
)

CORE_FEATURES = (
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
)

SCIENTIFIC_OBSERVATION_KEY = (
    "case", "object", "cohort", "scale_index_from", "scale_index_to",
)
ALIGNMENT_KEY = ("case", "object", "cohort", "candidate_rank")
GEOMETRY_CONTRACT_PATHS = (
    Path("outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_manifest.csv"),
    Path("outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_feature_contract_manifest.csv"),
)
EXPECTED_GEOMETRY_FEATURES = (
    "centroid_drift", "id_score", "jaccard_loss", "overlap_score",
    "pinch_score_total", "shape_score", "support_score",
)
EXPECTED_STABILITY_PLUS_GEOMETRY_FEATURES = (*CORE_FEATURES, *EXPECTED_GEOMETRY_FEATURES)
EXACT_CANONICAL_CARRIER_FEATURES = {
    "stability_core_3": CORE_FEATURES,
    "geometry_scores_only": EXPECTED_GEOMETRY_FEATURES,
    "stability_plus_geometry": EXPECTED_STABILITY_PLUS_GEOMETRY_FEATURES,
}

FORBIDDEN_PREDICTOR_PATTERNS = (
    r"(^|_)(label|class|target|true|predicted|prediction|probability|proba)($|_)",
    r"record_id|relation|carrier|subclass|readiness|blocker",
    r"fold|split|holdout|validation|confusion|correct|error|loss",
    r"artifact|manifest|commit|version|source_path",
)

GROUPING_PATTERNS = (
    r"object|cohort|transition|route|path|trajectory|window|fold|split|scheme|cluster",
)

PROVENANCE_PATTERNS = (
    r"corpus|campaign|model|prompt|source|artifact|manifest|provenance|commit|version|run_id",
)

OUTCOME_PATTERNS = (
    r"(^|_)(label|class|target|true_label|true_class|y_true|regime)($|_)",
    r"predicted|probability|proba|margin|correct|error|loss",
)

SUPPORT_ORDER = (
    "object",
    "cohort",
    "transition",
    "scale_band",
    "contract_or_transform",
    "feature_family",
    "seam_relative",
    "window",
    "route_or_path",
    "provenance_slice",
)


@dataclass(frozen=True)
class FreezeDecision:
    gate: str
    passed: bool
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--lineage-dir", type=Path, default=DEFAULT_LINEAGE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--review-dir", type=Path, default=None)
    p.add_argument("--discovery-fraction", type=float, default=0.67)
    p.add_argument("--partition-seed", default="obs084a-two-way-freeze-v1")
    p.add_argument("--min-clusters-total", type=int, default=12)
    p.add_argument("--min-clusters-per-partition", type=int, default=4)
    p.add_argument("--min-clusters-per-class-partition", type=int, default=2)
    p.add_argument("--seam-bins", type=int, default=3)
    p.add_argument("--allow-auto-freeze", action="store_true",
                   help="Allow a formal freeze from validated auto proposals. Without this flag, unresolved human review yields proposal_only status.")
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required lineage artifact missing: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_existing_column(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def string_series(df: pd.DataFrame, column: str) -> pd.Series:
    selected = df.loc[:, df.columns == column]
    if isinstance(selected, pd.Series):
        return selected.astype("string")
    if selected.shape[1] == 0:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return selected.iloc[:, 0].astype("string")


def stable_partition(value: Any, seed: str, discovery_fraction: float) -> str:
    token = f"{seed}|{value}".encode("utf-8")
    n = int(hashlib.sha256(token).hexdigest()[:16], 16)
    u = n / float(0xFFFFFFFFFFFFFFFF)
    return "discovery" if u < discovery_fraction else "confirmation"


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def load_lineage(lineage_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        key: read_csv_required(lineage_dir / filename)
        for key, filename in REQUIRED_LINEAGE_FILES.items()
    }


def load_review(review_dir: Path | None, key: str) -> pd.DataFrame | None:
    if review_dir is None:
        return None
    path = review_dir / REVIEW_FILES[key]
    if not path.exists():
        return None
    return read_csv_required(path)


def select_canonical_feature_table(sources: pd.DataFrame, repo_root: Path) -> tuple[Path, pd.DataFrame]:
    role_col = first_existing_column(sources, ("source_role", "artifact_role", "role"))
    path_col = first_existing_column(sources, ("selected_path", "artifact_path", "path"))
    if not role_col or not path_col:
        raise ValueError("Canonical source manifest lacks role/path columns")
    rows = sources[string_series(sources, role_col) == "canonical_feature_table"]
    if rows.empty:
        raise ValueError("No canonical_feature_table selected in lineage manifest")
    raw = str(rows.iloc[0][path_col])
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path, read_csv_required(path)


def propose_observation_key(
    crosswalk: pd.DataFrame,
    feature_df: pd.DataFrame,
    review: pd.DataFrame | None,
) -> pd.DataFrame:
    if review is not None and not review.empty:
        out = review.copy()
        if "resolution_status" not in out.columns:
            out["resolution_status"] = "reviewed_override"
        return out

    missing = [c for c in SCIENTIFIC_OBSERVATION_KEY if c not in feature_df.columns]
    if missing:
        return pd.DataFrame([{
            "canonical_key_name": "scientific_observation_key",
            "feature_table_column": "|".join(SCIENTIFIC_OBSERVATION_KEY),
            "key_columns_json": json.dumps(SCIENTIFIC_OBSERVATION_KEY),
            "non_null_rows": 0,
            "unique_values": 0,
            "uniqueness_ratio": 0.0,
            "null_key_rows": len(feature_df),
            "duplicate_key_rows": len(feature_df),
            "key_role": "scientific_composite",
            "resolution_status": "unresolved_missing_columns",
            "resolution_basis": f"missing canonical key columns: {missing}",
            "selected": False,
        }])

    key_frame = feature_df[list(SCIENTIFIC_OBSERVATION_KEY)]
    null_mask = key_frame.isna().any(axis=1)
    serialized = key_frame.astype(str).agg("|".join, axis=1)
    duplicate_mask = serialized.duplicated(keep=False)
    unique_values = int(serialized.nunique(dropna=False))
    valid = int(null_mask.sum()) == 0 and int(duplicate_mask.sum()) == 0 and unique_values == len(feature_df)

    alignment_missing = [c for c in ALIGNMENT_KEY if c not in feature_df.columns]
    if alignment_missing:
        alignment_unique = False
        alignment_duplicates = len(feature_df)
    else:
        align_serialized = feature_df[list(ALIGNMENT_KEY)].astype(str).agg("|".join, axis=1)
        alignment_duplicates = int(align_serialized.duplicated(keep=False).sum())
        alignment_unique = alignment_duplicates == 0 and align_serialized.nunique() == len(feature_df)

    return pd.DataFrame([{
        "canonical_key_name": "scientific_observation_key",
        "feature_table_column": "|".join(SCIENTIFIC_OBSERVATION_KEY),
        "key_columns_json": json.dumps(SCIENTIFIC_OBSERVATION_KEY),
        "non_null_rows": int((~null_mask).sum()),
        "unique_values": unique_values,
        "uniqueness_ratio": unique_values / max(len(feature_df), 1),
        "null_key_rows": int(null_mask.sum()),
        "duplicate_key_rows": int(duplicate_mask.sum()),
        "key_role": "scientific_composite",
        "alignment_key_columns_json": json.dumps(ALIGNMENT_KEY),
        "alignment_key_unique": alignment_unique,
        "alignment_duplicate_rows": alignment_duplicates,
        "resolution_status": "resolved_validated_scientific_composite" if valid else "unresolved_key_validation_failed",
        "resolution_basis": "predeclared OBS-078 scientific identity; independently validated for completeness and uniqueness",
        "review_note": "alignment key is diagnostic only and is not the scientific observation identity",
        "selected": valid,
    }])


def classify_field_role(name: str) -> tuple[str, str]:
    lname = name.lower()
    if any(re.search(p, lname) for p in FORBIDDEN_PREDICTOR_PATTERNS):
        return "forbidden_predictive_leakage", "matched forbidden/leakage pattern"
    if any(re.search(p, lname) for p in OUTCOME_PATTERNS):
        return "outcome", "matched outcome/evaluation pattern"
    if any(re.search(p, lname) for p in PROVENANCE_PATTERNS):
        return "provenance", "matched provenance pattern"
    if any(re.search(p, lname) for p in GROUPING_PATTERNS):
        return "grouping_partition", "matched grouping/partition pattern"
    return "candidate_predictor", "numeric/feature candidate subject to carrier manifest"


def resolve_field_roles(
    lineage_roles: pd.DataFrame,
    feature_df: pd.DataFrame,
    review: pd.DataFrame | None,
    canonical_predictor_overrides: Iterable[str] = (),
) -> pd.DataFrame:
    if review is not None and not review.empty:
        out = review.copy()
        if "resolution_status" not in out.columns:
            out["resolution_status"] = "reviewed_override"
        return out

    override_set = {str(x) for x in canonical_predictor_overrides}
    numeric_cols = set(feature_df.select_dtypes(include="number").columns.astype(str))
    rows: list[dict[str, Any]] = []
    for col in map(str, feature_df.columns):
        if col in override_set:
            role = "candidate_predictor"
            basis = "canonical OBS-078/080 carrier contract overrides generic name-based leakage heuristics"
            status = "canonical_contract_override"
            review_note = (
                'geometry observable named "loss"; not an evaluation outcome'
                if col == "jaccard_loss"
                else "explicitly admitted by a frozen canonical carrier definition"
            )
        else:
            role, basis = classify_field_role(col)
            status = "proposed_auto"
            review_note = ""

        if role == "candidate_predictor" and col not in numeric_cols:
            role = "unused_non_numeric"
            basis = "non-numeric and not required for grouping/provenance"
            status = "unresolved_canonical_feature_non_numeric" if col in override_set else "proposed_auto"

        rows.append({
            "field_name": col,
            "field_role": role,
            "predictor_allowed": role == "candidate_predictor",
            "grouping_allowed": role in {"grouping_partition", "provenance"},
            "matching_allowed": role in {"grouping_partition", "provenance", "candidate_predictor"},
            "canonical_contract_override": col in override_set,
            "resolution_basis": basis,
            "review_note": review_note,
            "resolution_status": status,
        })
    return pd.DataFrame(rows)


def load_geometry_contract(repo_root: Path) -> tuple[list[str], dict[str, str]]:
    resolved: list[list[str]] = []
    hashes: dict[str, str] = {}
    for rel in GEOMETRY_CONTRACT_PATHS:
        path = repo_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical geometry contract manifest: {path}")
        df = read_csv_required(path)
        required = {"feature_contract", "feature", "feature_index"}
        if not required.issubset(df.columns):
            raise ValueError(f"Geometry manifest lacks required columns {sorted(required)}: {path}")
        subset = df[df["feature_contract"].astype(str) == "geometry_scores_only"].copy()
        subset = subset.sort_values("feature_index")
        features = subset["feature"].astype(str).tolist()
        resolved.append(features)
        hashes[str(rel)] = sha256_file(path)
    if resolved[0] != resolved[1]:
        raise ValueError(f"OBS-080c/080d geometry contracts disagree: {resolved[0]} != {resolved[1]}")
    if tuple(resolved[0]) != EXPECTED_GEOMETRY_FEATURES:
        raise ValueError(f"Canonical geometry contract differs from expected ordered features: {resolved[0]}")
    return resolved[0], hashes


def resolve_carrier_features(
    lineage_carriers: pd.DataFrame,
    field_roles: pd.DataFrame,
    feature_df: pd.DataFrame,
    review: pd.DataFrame | None,
    repo_root: Path,
) -> pd.DataFrame:
    if review is not None and not review.empty:
        out = review.copy()
        if "resolution_status" not in out.columns:
            out["resolution_status"] = "reviewed_override"
        return out

    allowed = set(
        field_roles.loc[field_roles["predictor_allowed"].map(normalize_bool), "field_name"].astype(str)
    )
    numeric = set(feature_df.select_dtypes(include="number").columns.astype(str))
    allowed &= numeric
    geometry_features, geometry_hashes = load_geometry_contract(repo_root)

    # Canonical carrier membership is authoritative. Generic field-name heuristics
    # may classify a feature for audit purposes, but cannot silently delete a
    # feature explicitly admitted by the OBS-078/080 manifests.
    canonical_override = set(CORE_FEATURES) | set(geometry_features)
    canonical_available = canonical_override & numeric & set(map(str, feature_df.columns))
    allowed |= canonical_available

    path_tokens = ("share", "path", "route", "cohort")
    window_tokens = ("window", "local", "bounded", "delta_d", "lambda")
    rows: list[dict[str, Any]] = []

    for carrier in CANONICAL_CARRIERS:
        exact_contract = carrier in EXACT_CANONICAL_CARRIER_FEATURES
        if carrier == "stability_core_3":
            expected = list(CORE_FEATURES)
            rule = "exact canonical three-feature core"
            resolved_status = "resolved_exact_canonical_core"
        elif carrier == "geometry_scores_only":
            expected = list(geometry_features)
            rule = "exact OBS-080c/080d geometry_scores_only contract; manifests must agree in order and content"
            resolved_status = "resolved_from_agreeing_canonical_manifests"
        elif carrier == "path_shares_only":
            expected = sorted(f for f in allowed if any(t in f.lower() for t in path_tokens))
            rule = "numeric non-leakage fields matching path/share vocabulary"
            resolved_status = "proposed_auto"
        elif carrier == "stability_plus_geometry":
            expected = [*CORE_FEATURES, *geometry_features]
            rule = "exact ordered union of canonical stability core and canonical OBS-080 geometry contract"
            resolved_status = "resolved_from_canonical_manifests"
        elif carrier == "no_window":
            expected = sorted(f for f in allowed if not any(t in f.lower() for t in window_tokens))
            rule = "all allowed numeric fields excluding window/local vocabulary"
            resolved_status = "proposed_auto"
        else:
            expected = sorted(allowed)
            rule = "all numeric fields classified predictor_allowed, including canonical-contract overrides"
            resolved_status = "proposed_auto"

        emitted = [
            f for f in expected
            if f in feature_df.columns and f in numeric and f in allowed
        ]
        ordered_exact_match = emitted == expected
        expected_count = len(expected)
        emitted_count = len(emitted)

        for idx, feature in enumerate(expected):
            present = feature in feature_df.columns and feature in numeric
            predictor_allowed = feature in allowed
            if exact_contract:
                if not present:
                    row_status = "unresolved_canonical_feature_missing"
                elif not predictor_allowed:
                    row_status = "unresolved_canonical_feature_not_allowed"
                elif not ordered_exact_match:
                    row_status = "unresolved_exact_contract_mismatch"
                else:
                    row_status = resolved_status
            else:
                row_status = resolved_status if present and predictor_allowed else "unresolved_feature_availability"

            rows.append({
                "carrier": carrier,
                "feature_name": feature,
                "feature_index": idx,
                "feature_present": present,
                "predictor_allowed": predictor_allowed,
                "canonical_contract_override": feature in canonical_override,
                "exact_canonical_contract": exact_contract,
                "expected_feature_count": expected_count,
                "emitted_feature_count": emitted_count,
                "ordered_exact_match": ordered_exact_match,
                "resolution_rule": rule,
                "resolution_status": row_status,
                "obs080c_manifest_sha256": geometry_hashes.get(str(GEOMETRY_CONTRACT_PATHS[0]), "") if carrier in {"geometry_scores_only", "stability_plus_geometry"} else "",
                "obs080d_manifest_sha256": geometry_hashes.get(str(GEOMETRY_CONTRACT_PATHS[1]), "") if carrier in {"geometry_scores_only", "stability_plus_geometry"} else "",
            })

        if not expected:
            rows.append({
                "carrier": carrier,
                "feature_name": "",
                "feature_index": -1,
                "feature_present": False,
                "predictor_allowed": False,
                "canonical_contract_override": False,
                "exact_canonical_contract": exact_contract,
                "expected_feature_count": 0,
                "emitted_feature_count": 0,
                "ordered_exact_match": False,
                "resolution_rule": rule,
                "resolution_status": "unresolved_no_features",
                "obs080c_manifest_sha256": "",
                "obs080d_manifest_sha256": "",
            })

    return pd.DataFrame(rows)


def choose_cluster_unit(
    cluster_hierarchy: pd.DataFrame,
    feature_df: pd.DataFrame,
    review: pd.DataFrame | None,
) -> pd.DataFrame:
    if review is not None and not review.empty:
        out = review.copy()
        if "resolution_status" not in out.columns:
            out["resolution_status"] = "reviewed_override"
        return out

    preference = ("object", "object_id", "route_id", "path_id", "transition", "transition_id", "cohort")
    rows: list[dict[str, Any]] = []
    selected: str | None = None
    for candidate in preference:
        if candidate in feature_df.columns:
            s = string_series(feature_df, candidate)
            n = int(s.nunique(dropna=True))
            if n >= 2:
                selected = candidate
                break

    if selected is None:
        # Search lineage proposals for a column present in canonical feature table.
        unit_col = first_existing_column(cluster_hierarchy, ("unit_column", "cluster_column", "field"))
        if unit_col:
            for value in string_series(cluster_hierarchy, unit_col).dropna().unique().tolist():
                if str(value) in feature_df.columns:
                    selected = str(value)
                    break

    if selected is None:
        return pd.DataFrame([{
            "cluster_unit": "",
            "unique_clusters": 0,
            "selected": False,
            "selection_basis": "no defensible cluster field found",
            "resolution_status": "unresolved",
        }])

    n = int(string_series(feature_df, selected).nunique(dropna=True))
    rows.append({
        "cluster_unit": selected,
        "unique_clusters": n,
        "selected": True,
        "selection_basis": "preference order prioritizing object-level independence before route/path",
        "resolution_status": "proposed_auto",
    })
    return pd.DataFrame(rows)


def build_partition_manifest(
    feature_df: pd.DataFrame,
    key_spec: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    seed: str,
    discovery_fraction: float,
) -> pd.DataFrame:
    selected_cluster = cluster_selection.loc[cluster_selection.get("selected", False).map(normalize_bool)]
    if selected_cluster.empty:
        return pd.DataFrame()
    cluster_col = str(selected_cluster.iloc[0]["cluster_unit"])
    if cluster_col not in feature_df.columns:
        return pd.DataFrame()

    selected_key = key_spec.loc[key_spec.get("selected", False).map(normalize_bool)]
    key_col = str(selected_key.iloc[0]["feature_table_column"]) if not selected_key.empty else ""

    work = pd.DataFrame(index=feature_df.index)
    work["row_index"] = feature_df.index.astype(int)
    work["cluster_unit"] = cluster_col
    work["cluster_id"] = string_series(feature_df, cluster_col).fillna("__missing_cluster__")
    work["partition"] = work["cluster_id"].map(lambda x: stable_partition(x, seed, discovery_fraction))

    if key_col and "|" not in key_col and key_col in feature_df.columns:
        work["observation_key"] = string_series(feature_df, key_col)
    elif key_col and "|" in key_col:
        cols = [c for c in key_col.split("|") if c in feature_df.columns]
        work["observation_key"] = feature_df[cols].astype(str).agg("|".join, axis=1) if cols else work["row_index"].astype(str)
    else:
        work["observation_key"] = work["row_index"].astype(str)

    class_col = first_existing_column(feature_df, ("case", "regime", "label", "true_label", "condition", "corpus"))
    work["class_value"] = string_series(feature_df, class_col) if class_col else "__unknown__"
    work["partition_seed"] = seed
    work["discovery_fraction"] = discovery_fraction
    return work[[
        "observation_key", "row_index", "cluster_unit", "cluster_id", "class_value",
        "partition", "partition_seed", "discovery_fraction",
    ]]


def parse_relation_classes(relation: str) -> list[str]:
    if relation == "three_way":
        return ["C", "Cp2", "Cp3"]
    if "_vs_" in relation:
        a, b = relation.split("_vs_", 1)
        return [a, b]
    return []


def build_partition_balance(
    partition_manifest: pd.DataFrame,
    record_sources: pd.DataFrame,
    min_clusters_per_partition: int,
    min_clusters_per_class_partition: int,
) -> pd.DataFrame:
    if partition_manifest.empty:
        return pd.DataFrame()
    rid_col = first_existing_column(record_sources, ("record_id",))
    rel_col = first_existing_column(record_sources, ("relation",))
    subclass_col = first_existing_column(record_sources, ("subclass", "diagnostic_subclass"))
    elig_col = first_existing_column(record_sources, ("confirmation_eligibility", "fl3_eligibility"))
    if not rid_col:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, rec in record_sources.iterrows():
        rid = str(rec[rid_col])
        relation = str(rec[rel_col]) if rel_col else rid.split("__", 1)[0]
        subclass = str(rec[subclass_col]) if subclass_col else ""
        eligible = (
            "eligible" in str(rec[elig_col]).lower() if elig_col else "C2" in subclass
        )
        classes = parse_relation_classes(relation)
        subset = partition_manifest.copy()
        if classes and "class_value" in subset.columns:
            observed = subset["class_value"].astype(str)
            # Accept exact classes; if aliases don't match, retain all rows but mark mapping unresolved.
            exact = subset[observed.isin(classes)]
            class_mapping_resolved = not exact.empty
            if class_mapping_resolved:
                subset = exact
        else:
            class_mapping_resolved = False

        for partition in ("discovery", "confirmation"):
            part = subset[subset["partition"] == partition]
            cluster_count = int(part["cluster_id"].nunique())
            row_count = int(len(part))
            class_counts = part.groupby("class_value")["cluster_id"].nunique().to_dict()
            min_class_clusters = min(class_counts.values()) if class_counts else 0
            passed = (
                cluster_count >= min_clusters_per_partition
                and min_class_clusters >= min_clusters_per_class_partition
            )
            rows.append({
                "record_id": rid,
                "relation": relation,
                "subclass": subclass,
                "confirmation_eligible": eligible,
                "partition": partition,
                "row_count": row_count,
                "unique_clusters": cluster_count,
                "class_cluster_counts_json": json.dumps(class_counts, sort_keys=True),
                "min_class_clusters": int(min_class_clusters),
                "class_mapping_resolved": class_mapping_resolved,
                "balance_pass": passed,
                "limitation": "" if passed else "insufficient cluster/class support or unresolved class mapping",
            })
    return pd.DataFrame(rows)


def resolve_seam_protocol(
    feature_df: pd.DataFrame,
    review: pd.DataFrame | None,
    bins: int,
) -> pd.DataFrame:
    if review is not None and not review.empty:
        out = review.copy()
        if "resolution_status" not in out.columns:
            out["resolution_status"] = "reviewed_override"
        return out

    seam_candidates = [
        str(c) for c in feature_df.columns
        if "seam" in str(c).lower() and pd.api.types.is_numeric_dtype(feature_df[c])
    ]
    if not seam_candidates:
        return pd.DataFrame([{
            "support_family": "seam_relative",
            "source_field": "",
            "discretization_method": "unavailable",
            "bin_count": 0,
            "threshold_basis": "none",
            "threshold_values_json": "[]",
            "fit_partition": "discovery_only",
            "resolution_status": "unavailable",
        }])

    field = seam_candidates[0]
    # Freeze method, not empirical cut values. Values must be fitted on discovery only.
    labels = ["near", "intermediate", "far"] if bins == 3 else [f"bin_{i+1}" for i in range(bins)]
    return pd.DataFrame([{
        "support_family": "seam_relative",
        "source_field": field,
        "discretization_method": "equal_frequency_quantiles",
        "bin_count": bins,
        "threshold_basis": "quantiles fitted on discovery partition only; then applied unchanged to confirmation",
        "threshold_values_json": "[]",
        "bin_labels_json": json.dumps(labels),
        "fit_partition": "discovery_only",
        "outcome_blind": True,
        "resolution_status": "proposed_auto",
    }])


def freeze_support_vocabulary(
    support_families: pd.DataFrame,
    seam_protocol: pd.DataFrame,
    review: pd.DataFrame | None,
) -> pd.DataFrame:
    if review is not None and not review.empty:
        out = review.copy()
        if "resolution_status" not in out.columns:
            out["resolution_status"] = "reviewed_override"
        return out

    fam_col = first_existing_column(support_families, ("support_family",))
    avail_col = first_existing_column(support_families, ("available",))
    disc_col = first_existing_column(support_families, ("requires_predeclared_discretization",))
    rows: list[dict[str, Any]] = []
    for family in SUPPORT_ORDER:
        source = support_families[string_series(support_families, fam_col) == family] if fam_col else pd.DataFrame()
        available = normalize_bool(source.iloc[0][avail_col]) if not source.empty and avail_col else False
        requires_disc = normalize_bool(source.iloc[0][disc_col]) if not source.empty and disc_col else False
        included = available
        status = "included"
        if not available:
            included = False
            status = "excluded_unavailable"
        elif requires_disc and family == "seam_relative":
            seam_ok = not seam_protocol.empty and str(seam_protocol.iloc[0].get("resolution_status", "")) not in {"unavailable", "unresolved"}
            included = seam_ok
            status = "included_with_frozen_discretization" if seam_ok else "excluded_discretization_unresolved"
        rows.append({
            "support_family": family,
            "included_in_discovery_vocabulary": included,
            "requires_predeclared_discretization": requires_disc,
            "max_conjunction_depth": 2,
            "freeze_status": status,
            "resolution_status": "proposed_auto",
        })
    return pd.DataFrame(rows)


def build_source_hash_manifest(
    canonical_sources: pd.DataFrame,
    repo_root: Path,
    code_path: Path,
    commit: str,
) -> pd.DataFrame:
    path_col = first_existing_column(canonical_sources, ("selected_path", "artifact_path", "path"))
    role_col = first_existing_column(canonical_sources, ("source_role", "role"))
    rows: list[dict[str, Any]] = []
    if path_col:
        for _, row in canonical_sources.iterrows():
            raw = str(row[path_col])
            p = Path(raw)
            if not p.is_absolute():
                p = repo_root / p
            rows.append({
                "source_role": str(row[role_col]) if role_col else "",
                "artifact_path": raw,
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() and p.is_file() else "",
                "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0,
                "code_commit": commit,
            })
    for rel in GEOMETRY_CONTRACT_PATHS:
        p = repo_root / rel
        rows.append({
            "source_role": "canonical_geometry_contract_manifest",
            "artifact_path": str(rel),
            "exists": p.exists(),
            "sha256": sha256_file(p) if p.exists() else "",
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "code_commit": commit,
        })
    rows.append({
        "source_role": "freeze_script",
        "artifact_path": str(code_path),
        "exists": code_path.exists(),
        "sha256": sha256_file(code_path) if code_path.exists() else "",
        "size_bytes": code_path.stat().st_size if code_path.exists() else 0,
        "code_commit": commit,
    })
    return pd.DataFrame(rows)


def evaluate_gates(
    key_spec: pd.DataFrame,
    carrier_manifest: pd.DataFrame,
    field_roles: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    partition_manifest: pd.DataFrame,
    partition_balance: pd.DataFrame,
    support_vocab: pd.DataFrame,
    source_hashes: pd.DataFrame,
    allow_auto_freeze: bool,
) -> list[FreezeDecision]:
    decisions: list[FreezeDecision] = []

    selected_keys = key_spec[key_spec.get("selected", False).map(normalize_bool)] if not key_spec.empty else pd.DataFrame()
    key_ok = len(selected_keys) == 1
    if key_ok:
        row = selected_keys.iloc[0]
        key_ok = (
            str(row.get("resolution_status", "")).startswith("resolved_")
            and int(row.get("null_key_rows", 1)) == 0
            and int(row.get("duplicate_key_rows", 1)) == 0
            and normalize_bool(row.get("alignment_key_unique", False))
        )
    decisions.append(FreezeDecision(
        "observation_key_resolved", key_ok,
        "pass" if key_ok else "fail",
        f"selected key rows={len(selected_keys)}; scientific key complete/unique and alignment key unique={key_ok}",
    ))

    carrier_ok = True
    carrier_detail: list[str] = []
    carrier_col = carrier_manifest.get("carrier", pd.Series(dtype=str)).astype(str)
    for carrier in CANONICAL_CARRIERS:
        subset = carrier_manifest[carrier_col == carrier].copy()
        if subset.empty:
            carrier_ok = False
            carrier_detail.append(f"{carrier}:missing_manifest_rows")
            continue

        subset = subset.sort_values("feature_index")
        named = subset[subset.get("feature_name", "").astype(str) != ""].copy()
        emitted_names = named.get("feature_name", pd.Series(dtype=str)).astype(str).tolist()
        all_present = bool(named.get("feature_present", False).map(normalize_bool).all()) if not named.empty else False
        all_allowed = bool(named.get("predictor_allowed", False).map(normalize_bool).all()) if not named.empty else False
        statuses = named.get("resolution_status", pd.Series(dtype=str)).astype(str)
        all_statuses_resolved = bool((~statuses.str.startswith("unresolved")).all()) if not named.empty else False
        expected_count_values = set(pd.to_numeric(named.get("expected_feature_count", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
        emitted_count_values = set(pd.to_numeric(named.get("emitted_feature_count", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
        count_metadata_ok = (
            len(expected_count_values) == 1
            and len(emitted_count_values) == 1
            and next(iter(expected_count_values)) == len(emitted_names)
            and next(iter(emitted_count_values)) == len(emitted_names)
        )

        exact_expected = list(EXACT_CANONICAL_CARRIER_FEATURES.get(carrier, ()))
        if exact_expected:
            exact_match = emitted_names == exact_expected
            exact_flags_ok = bool(named.get("ordered_exact_match", False).map(normalize_bool).all())
        else:
            exact_match = True
            exact_flags_ok = True

        ok = (
            bool(emitted_names)
            and all_present
            and all_allowed
            and all_statuses_resolved
            and count_metadata_ok
            and exact_match
            and exact_flags_ok
        )
        carrier_ok &= ok
        if not ok:
            carrier_detail.append(
                f"{carrier}:names={emitted_names};expected={exact_expected or 'proposal'};"
                f"present={all_present};allowed={all_allowed};statuses={all_statuses_resolved};"
                f"counts={count_metadata_ok};exact={exact_match and exact_flags_ok}"
            )

    decisions.append(FreezeDecision(
        "carrier_features_resolved", carrier_ok,
        "pass" if carrier_ok else "fail",
        (
            "all six carriers have complete allowed feature manifests; exact canonical carriers match ordered definitions"
            if carrier_ok
            else "carrier manifest failures: " + " | ".join(carrier_detail)
        ),
    ))

    manual_roles = int((field_roles.get("field_role", "") == "manual_review").sum()) if not field_roles.empty else 0
    decisions.append(FreezeDecision(
        "field_roles_resolved", manual_roles == 0,
        "pass" if manual_roles == 0 else "fail",
        f"manual-review fields remaining={manual_roles}",
    ))

    selected_clusters = cluster_selection[cluster_selection.get("selected", False).map(normalize_bool)] if not cluster_selection.empty else pd.DataFrame()
    decisions.append(FreezeDecision(
        "cluster_unit_selected", len(selected_clusters) == 1,
        "pass" if len(selected_clusters) == 1 else "fail",
        f"selected cluster rows={len(selected_clusters)}",
    ))

    both_partitions = set(partition_manifest.get("partition", pd.Series(dtype=str)).astype(str).unique()) >= {"discovery", "confirmation"}
    decisions.append(FreezeDecision(
        "two_way_partition_created", not partition_manifest.empty and both_partitions,
        "pass" if not partition_manifest.empty and both_partitions else "fail",
        f"partitions={sorted(set(partition_manifest.get('partition', pd.Series(dtype=str)).astype(str)))}",
    ))

    eligible = partition_balance[partition_balance.get("confirmation_eligible", False).map(normalize_bool)] if not partition_balance.empty else pd.DataFrame()
    eligible_pass = not eligible.empty and bool(eligible.get("balance_pass", False).map(normalize_bool).all())
    decisions.append(FreezeDecision(
        "eligible_record_partition_balance", eligible_pass,
        "pass" if eligible_pass else "fail",
        f"eligible partition rows={len(eligible)}, passing={int(eligible.get('balance_pass', False).map(normalize_bool).sum()) if not eligible.empty else 0}",
    ))

    vocab_ok = not support_vocab.empty and bool(support_vocab.get("included_in_discovery_vocabulary", False).map(normalize_bool).any())
    decisions.append(FreezeDecision(
        "support_vocabulary_frozen", vocab_ok,
        "pass" if vocab_ok else "fail",
        f"included support families={int(support_vocab.get('included_in_discovery_vocabulary', False).map(normalize_bool).sum()) if not support_vocab.empty else 0}",
    ))

    hashes_ok = not source_hashes.empty and bool(source_hashes.get("exists", False).map(normalize_bool).all()) and bool((source_hashes.get("sha256", "").astype(str).str.len() == 64).all())
    decisions.append(FreezeDecision(
        "source_hashes_complete", hashes_ok,
        "pass" if hashes_ok else "fail",
        f"hashed sources={int((source_hashes.get('sha256', '').astype(str).str.len() == 64).sum()) if not source_hashes.empty else 0}/{len(source_hashes)}",
    ))

    auto_rows = 0
    for df in (key_spec, carrier_manifest, field_roles, cluster_selection, support_vocab):
        if not df.empty and "resolution_status" in df.columns:
            auto_rows += int(df["resolution_status"].astype(str).str.startswith("proposed_auto").sum())
    review_ok = allow_auto_freeze or auto_rows == 0
    decisions.append(FreezeDecision(
        "human_review_or_explicit_auto_freeze", review_ok,
        "pass" if review_ok else "proposal_only",
        f"auto-proposed rows={auto_rows}; allow_auto_freeze={allow_auto_freeze}",
    ))
    return decisions


def write_report(
    output_path: Path,
    status: str,
    decisions: list[FreezeDecision],
    key_spec: pd.DataFrame,
    carrier_manifest: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    partition_manifest: pd.DataFrame,
    partition_balance: pd.DataFrame,
    seam_protocol: pd.DataFrame,
    support_vocab: pd.DataFrame,
    source_hashes: pd.DataFrame,
    freeze_id: str,
) -> None:
    gate_df = pd.DataFrame([d.__dict__ for d in decisions])
    carrier_counts = (
        carrier_manifest[carrier_manifest.get("feature_name", "").astype(str) != ""]
        .groupby("carrier").size().rename("feature_count").reset_index()
        if not carrier_manifest.empty else pd.DataFrame()
    )
    part_counts = (
        partition_manifest.groupby("partition").agg(rows=("row_index", "size"), clusters=("cluster_id", "nunique")).reset_index()
        if not partition_manifest.empty else pd.DataFrame()
    )
    eligible_summary = (
        partition_balance[partition_balance.get("confirmation_eligible", False).map(normalize_bool)]
        if not partition_balance.empty else pd.DataFrame()
    )

    text = f"""# OBS-084a — Bridge Resolution and Two-Way Partition Freeze

## State

Resolution/freeze pass completed.

Overall status: `{status}`

Freeze manifest ID: `{freeze_id}`

This is a pre-discovery design-freeze artifact only. It performs no candidate
generation, confirmation-outcome inspection, localization contrast, witness
assignment, FL promotion, repair design, intervention, or causal analysis.

## Canonical interpretation

This pass resolves the final bridge decisions required before OBS-084 discovery:
observation identity, carrier predictors, field roles, cluster unit, deterministic
discovery/confirmation partition, seam discretization protocol, support vocabulary,
and source identity.

A successful freeze means only that discovery may begin under an inspectable and
immutable protocol. It is not evidence that a failure support exists.

## Freeze gates

{df_to_markdown(gate_df)}

## Observation-key specification

{df_to_markdown(key_spec)}

## Carrier feature counts

{df_to_markdown(carrier_counts) if not carrier_counts.empty else 'No carrier features resolved.'}

## Cluster-unit selection

{df_to_markdown(cluster_selection)}

## Two-way partition

{df_to_markdown(part_counts) if not part_counts.empty else 'No partition was created.'}

Partition assignment is deterministic at the selected cluster level. Rows from
the same cluster cannot cross discovery and confirmation.

## Per-record partition balance

- Balance rows: {len(partition_balance)}
- FL3-eligible rows: {len(eligible_summary)}
- Passing FL3-eligible rows: {int(eligible_summary.get('balance_pass', False).map(normalize_bool).sum()) if not eligible_summary.empty else 0}

A globally balanced split is insufficient. Every confirmation-eligible C2 record
must retain adequate cluster and class support in both partitions.

## Seam discretization protocol

{df_to_markdown(seam_protocol)}

Seam thresholds are not fitted by this script. The method is frozen here; cut
values must be estimated on discovery only and applied unchanged to confirmation.

## Support vocabulary

{df_to_markdown(support_vocab)}

Unavailable support families remain excluded. Candidate conjunction depth is
capped to prevent uncontrolled combinatorial discovery.

## Source identity

- Hashed source rows: {int((source_hashes.get('sha256', '').astype(str).str.len() == 64).sum()) if not source_hashes.empty else 0}
- Repository/code commit recorded: {source_hashes['code_commit'].iloc[0] if not source_hashes.empty and 'code_commit' in source_hashes.columns else 'unknown'}

## Discovery gate

OBS-084 candidate discovery may begin only when all hard gates pass and the
freeze status is `frozen_ready_for_discovery`.

If the status is `proposal_ready_for_human_review`, the generated artifacts are
inspectable proposals only. Review or explicitly rerun with `--allow-auto-freeze`
after accepting the rule-based decisions.

If any technical gate fails, discovery remains blocked.

## Outputs

- `obs084a_reviewed_observation_key_spec.csv`
- `obs084a_reviewed_carrier_feature_manifest.csv`
- `obs084a_reviewed_field_roles.csv`
- `obs084a_cluster_unit_selection.csv`
- `obs084a_two_way_partition_manifest.csv`
- `obs084a_partition_balance_final.csv`
- `obs084a_seam_discretization_protocol.csv`
- `obs084a_support_vocabulary_freeze.csv`
- `obs084a_source_hash_manifest.csv`
- `obs084a_bridge_resolution_summary.csv`
- `obs084a_freeze_manifest.json`
- `obs084a_bridge_resolution_report.md`

## Limitations

- Rule-based proposals are not silently described as human-reviewed.
- Observation-key overlap does not prove semantic equivalence across every
  prediction artifact; future discovery must use the frozen key specification.
- Carrier manifests derived from vocabulary rules should be reviewed against
  upstream feature manifests before irreversible candidate freeze.
- Cluster assignment supports structural separation, not proof of statistical
  independence.
- No confirmation outcomes are inspected or unlocked here.

## Canonical result statement

OBS-084a bridge resolution converts the canonical OBS-078–083 evidence spine into
an inspectable pre-discovery protocol covering observation identity, carrier
features, field roles, structural clustering, deterministic two-way partitioning,
support vocabulary, seam discretization, and source hashes. It establishes study-
design readiness only and no direct failure support, causal origin, repair target,
actionability, external generalization, or formal topology.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    lineage_dir = args.lineage_dir if args.lineage_dir.is_absolute() else repo_root / args.lineage_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    review_dir = None if args.review_dir is None else (args.review_dir if args.review_dir.is_absolute() else repo_root / args.review_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (0.5 <= args.discovery_fraction <= 0.85):
        raise ValueError("--discovery-fraction must be between 0.50 and 0.85")
    if args.seam_bins < 2:
        raise ValueError("--seam-bins must be >= 2")

    lineage = load_lineage(lineage_dir)
    feature_path, feature_df = select_canonical_feature_table(lineage["canonical_sources"], repo_root)

    key_spec = propose_observation_key(
        lineage["key_crosswalk"], feature_df, load_review(review_dir, "observation_key")
    )
    geometry_features, _geometry_hashes = load_geometry_contract(repo_root)
    canonical_predictor_overrides = (*CORE_FEATURES, *geometry_features)
    field_roles = resolve_field_roles(
        lineage["field_roles"], feature_df, load_review(review_dir, "field_roles"),
        canonical_predictor_overrides=canonical_predictor_overrides,
    )
    carrier_manifest = resolve_carrier_features(
        lineage["carrier_features"], field_roles, feature_df,
        load_review(review_dir, "carrier_features"), repo_root,
    )
    cluster_selection = choose_cluster_unit(
        lineage["cluster_hierarchy"], feature_df, load_review(review_dir, "cluster_unit")
    )
    partition_manifest = build_partition_manifest(
        feature_df, key_spec, cluster_selection, args.partition_seed, args.discovery_fraction
    )
    partition_balance = build_partition_balance(
        partition_manifest, lineage["record_sources"],
        args.min_clusters_per_partition, args.min_clusters_per_class_partition,
    )
    seam_protocol = resolve_seam_protocol(
        feature_df, load_review(review_dir, "seam_protocol"), args.seam_bins
    )
    support_vocab = freeze_support_vocabulary(
        lineage["support_families"], seam_protocol,
        load_review(review_dir, "support_vocabulary"),
    )

    script_path = Path(__file__).resolve()
    commit = git_commit(repo_root)
    source_hashes = build_source_hash_manifest(
        lineage["canonical_sources"], repo_root, script_path, commit
    )

    decisions = evaluate_gates(
        key_spec, carrier_manifest, field_roles, cluster_selection,
        partition_manifest, partition_balance, support_vocab, source_hashes,
        args.allow_auto_freeze,
    )
    technical_pass = all(d.passed for d in decisions if d.gate != "human_review_or_explicit_auto_freeze")
    review_pass = next(d.passed for d in decisions if d.gate == "human_review_or_explicit_auto_freeze")
    if technical_pass and review_pass:
        status = "frozen_ready_for_discovery"
    elif technical_pass:
        status = "proposal_ready_for_human_review"
    else:
        status = "freeze_blocked_by_unresolved_gates"

    # Write tabular outputs before building the manifest hash.
    outputs: dict[str, pd.DataFrame] = {
        "obs084a_reviewed_observation_key_spec.csv": key_spec,
        "obs084a_reviewed_carrier_feature_manifest.csv": carrier_manifest,
        "obs084a_reviewed_field_roles.csv": field_roles,
        "obs084a_cluster_unit_selection.csv": cluster_selection,
        "obs084a_two_way_partition_manifest.csv": partition_manifest,
        "obs084a_partition_balance_final.csv": partition_balance,
        "obs084a_seam_discretization_protocol.csv": seam_protocol,
        "obs084a_support_vocabulary_freeze.csv": support_vocab,
        "obs084a_source_hash_manifest.csv": source_hashes,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False)

    freeze_payload = {
        "schema": "obs084a_freeze_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "status": status,
        "repo_commit": commit,
        "canonical_feature_table": str(feature_path.relative_to(repo_root) if feature_path.is_relative_to(repo_root) else feature_path),
        "partition_seed": args.partition_seed,
        "discovery_fraction": args.discovery_fraction,
        "technical_gates": [d.__dict__ for d in decisions],
        "artifact_hashes": {
            name: sha256_file(output_dir / name) for name in outputs
        },
    }
    freeze_id = stable_json_hash(freeze_payload)
    freeze_payload["freeze_manifest_id"] = freeze_id
    (output_dir / "obs084a_freeze_manifest.json").write_text(
        json.dumps(freeze_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = pd.DataFrame([{
        "script_version": SCRIPT_VERSION,
        "overall_status": status,
        "freeze_manifest_id": freeze_id,
        "canonical_feature_rows": len(feature_df),
        "selected_key_count": int(key_spec.get("selected", False).map(normalize_bool).sum()) if not key_spec.empty else 0,
        "resolved_carrier_count": int(carrier_manifest[carrier_manifest.get("feature_name", "").astype(str) != ""]["carrier"].nunique()) if not carrier_manifest.empty else 0,
        "selected_cluster_unit": str(cluster_selection.loc[cluster_selection.get("selected", False).map(normalize_bool), "cluster_unit"].iloc[0]) if not cluster_selection.empty and cluster_selection.get("selected", False).map(normalize_bool).any() else "",
        "discovery_rows": int((partition_manifest.get("partition", "") == "discovery").sum()) if not partition_manifest.empty else 0,
        "confirmation_rows": int((partition_manifest.get("partition", "") == "confirmation").sum()) if not partition_manifest.empty else 0,
        "eligible_balance_rows": int(partition_balance.get("confirmation_eligible", False).map(normalize_bool).sum()) if not partition_balance.empty else 0,
        "eligible_balance_pass_rows": int((partition_balance.get("confirmation_eligible", False).map(normalize_bool) & partition_balance.get("balance_pass", False).map(normalize_bool)).sum()) if not partition_balance.empty else 0,
        "included_support_families": int(support_vocab.get("included_in_discovery_vocabulary", False).map(normalize_bool).sum()) if not support_vocab.empty else 0,
        "repo_commit": commit,
    }])
    summary.to_csv(output_dir / "obs084a_bridge_resolution_summary.csv", index=False)

    write_report(
        output_dir / "obs084a_bridge_resolution_report.md",
        status, decisions, key_spec, carrier_manifest, cluster_selection,
        partition_manifest, partition_balance, seam_protocol, support_vocab,
        source_hashes, freeze_id,
    )

    print(f"OBS-084a bridge resolution complete: {status}")
    print(f"Canonical feature rows: {len(feature_df)}")
    print(f"Freeze manifest ID: {freeze_id}")
    print(f"Outputs: {output_dir}")
    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
