#!/usr/bin/env python3
"""
obs084c_direct_failure_support_confirmation.py

OBS-084c — Direct Failure-Support Reserved Confirmation
=======================================================

Purpose
-------
Open the frozen OBS-084a confirmation partition exactly once and evaluate only
those FL2 candidates sealed by OBS-084b. This script performs no new support
search, no new predicate search, no threshold fitting on confirmation, and no
candidate ranking.

The confirmation instrument:

* requires an OBS-084a ``frozen_ready_for_discovery`` manifest;
* validates the exact OBS-084b candidate-manifest identity and contents;
* validates the OBS-084b script and discovery bundle used to seal candidates;
* independently recomputes discovery-only support thresholds from frozen
  feature values and requires exact agreement with the recorded OBS-084b
  threshold artifact;
* creates a one-time confirmation-opening lock before reading the canonical
  feature table for outcome evaluation;
* evaluates all sealed candidates, including unfavorable and untestable ones;
* applies the recorded scale and seam thresholds unchanged to confirmation;
* reconstructs the six frozen scientific carriers exactly;
* fits balanced logistic-regression diagnostics only inside confirmation,
  using leave-one-confirmation-object-out folds;
* retains object as the dependence and resampling unit;
* evaluates the exact sealed support, complement, predicate, metric, controls,
  direction, minimum effect, and exclusions;
* applies Benjamini-Hochberg correction over exactly the sealed candidate
  family;
* assigns at most FL3, and only to records already marked confirmation-eligible.

Scoped meaning of FL3
---------------------
``fl3_confirmed_direct_witness`` means artifact-direct evidence for the exact
record, predicate, support, contract, partition, and provenance identified in
the versioned witness. It does not mean causal, metaphysical, actionable,
externally generalized, or formally topological directness.

Default inputs
--------------
outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution/
    obs084a_freeze_manifest.json
    frozen OBS-084a CSV artifacts

outputs/rig_registry/obs084_direct_failure_witness/discovery/
    obs084b_candidate_freeze_manifest.json
    obs084b_candidate_freeze_manifest.csv
    obs084b_support_thresholds.csv
    obs084b_input_manifest.csv

outputs/rig_registry/obs083_negative_control_localization/
    obs083_diagnostic_subclass_assignments.csv
    obs083_relation_control_contrast.csv
    obs083_carrier_control_contrast.csv

outputs/rig_registry/rig_relation_registry.csv

Default outputs
---------------
outputs/rig_registry/obs084_direct_failure_witness/confirmation/
    obs084c_input_manifest.csv
    obs084c_candidate_manifest_validation.csv
    obs084c_confirmation_observation_losses.csv
    obs084c_support_complement_validation.csv
    obs084c_confirmation_site_contrasts.csv
    obs084c_confirmation_control_adjustment.csv
    obs084c_cluster_uncertainty.csv
    obs084c_multiplicity_audit.csv
    obs084c_candidate_outcomes.csv
    obs084c_direct_witness_registry.csv
    obs084c_confirmation_failures.csv
    obs084c_confirmation_summary.csv
    obs084c_confirmation_manifest.json
    obs084c_confirmation_report.md

A one-time opening lock is written beside the OBS-084b discovery artifacts:

    obs084c_confirmation_opening_lock.json

Run
---
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs084c_direct_failure_support_confirmation.py

Canonical guardrail
-------------------
Reserved confirmation can earn a scoped FL3 witness only when a sealed FL2
candidate reproduces under every predeclared confirmation gate. Null, reversed,
control-explained, multiplicity-failed, or untestable outcomes are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
MODEL_RANDOM_STATE = 84003

DEFAULT_FREEZE_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution"
)
DEFAULT_DISCOVERY_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/discovery"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/confirmation"
)
DEFAULT_OBS083_DIR = Path(
    "outputs/rig_registry/obs083_negative_control_localization"
)
DEFAULT_REGISTRY_PATH = Path("outputs/rig_registry/rig_relation_registry.csv")
DEFAULT_DISCOVERY_SCRIPT = Path(
    "experiments/studies/obs084b_direct_failure_support_discovery.py"
)

# This value identifies the sealed family shown in the canonical OBS-084b run.
# It remains overrideable for an explicitly different, internally valid sealed
# candidate bundle.
DEFAULT_EXPECTED_CANDIDATE_MANIFEST_ID = (
    "0d58d3abd25677683bb29b25c5b4e1fc2fdd1fab83866893c2151a80b97fd4f5"
)

OBS083_FILES = {
    "subclasses": "obs083_diagnostic_subclass_assignments.csv",
    "relation_controls": "obs083_relation_control_contrast.csv",
    "carrier_controls": "obs083_carrier_control_contrast.csv",
}

DISCOVERY_FILES = {
    "candidate_json": "obs084b_candidate_freeze_manifest.json",
    "candidate_csv": "obs084b_candidate_freeze_manifest.csv",
    "thresholds": "obs084b_support_thresholds.csv",
    "input_manifest": "obs084b_input_manifest.csv",
    "summary": "obs084b_discovery_summary.csv",
    "report": "obs084b_discovery_report.md",
}

OUTCOME_STATUS_ORDER = {
    "fl3_confirmed_direct_witness": 0,
    "confirmation_reproduced_but_claim_capped_at_fl2": 1,
    "confirmation_multiplicity_not_survived": 2,
    "confirmation_control_explained": 3,
    "confirmation_uncertain_cluster_support": 4,
    "confirmation_signal_absent": 5,
    "confirmation_direction_reversed": 6,
    "confirmation_complement_inadmissible": 7,
    "confirmation_support_unavailable": 8,
    "confirmation_not_testable": 9,
    "confirmation_protocol_mismatch": 10,
}


@dataclass(frozen=True)
class ConfirmationFailure:
    stage: str
    candidate_id: str
    record_id: str
    reason: str
    detail: str = ""


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    p.add_argument("--discovery-dir", type=Path, default=DEFAULT_DISCOVERY_DIR)
    p.add_argument("--obs083-dir", type=Path, default=DEFAULT_OBS083_DIR)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--discovery-script", type=Path, default=DEFAULT_DISCOVERY_SCRIPT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--expected-candidate-manifest-id",
        default=DEFAULT_EXPECTED_CANDIDATE_MANIFEST_ID,
        help="Exact sealed OBS-084b candidate-manifest ID. Use 'auto' only for a separately documented run.",
    )
    p.add_argument("--n-cluster-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--confirmation-fdr", type=float, default=0.10)
    p.add_argument("--min-direction-consistency", type=float, default=0.75)
    p.add_argument("--min-control-adjusted-effect", type=float, default=0.05)
    p.add_argument("--min-positive-control-share", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    p.add_argument(
        "--require-repo-commit",
        action="store_true",
        help="Require the current repository commit to equal the OBS-084a frozen commit.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen and sealed artifacts without opening or evaluating confirmation evidence.",
    )
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


def stable_hash(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "ok"}


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_No rows._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_module_from_path(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("obs084b_frozen_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import OBS-084b helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def values_equal(a: Any, b: Any, atol: float = 1e-12) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        af = float(a)
        bf = float(b)
        if np.isfinite(af) and np.isfinite(bf):
            return bool(np.isclose(af, bf, rtol=0.0, atol=atol))
    except Exception:
        pass
    return str(a) == str(b)


# -----------------------------------------------------------------------------
# Frozen and sealed bundle validation
# -----------------------------------------------------------------------------


def validate_discovery_script_hash(
    discovery_input_manifest: pd.DataFrame,
    discovery_script: Path,
) -> tuple[bool, str, str]:
    require_columns(
        discovery_input_manifest,
        ["artifact_role", "artifact_path", "sha256"],
        "OBS-084b input manifest",
    )
    rows = discovery_input_manifest[
        discovery_input_manifest["artifact_role"].astype(str) == "obs084b_script"
    ]
    expected = str(rows.iloc[0]["sha256"]) if len(rows) == 1 else ""
    actual = sha256_file(discovery_script) if discovery_script.exists() else ""
    return bool(expected and expected == actual), expected, actual


def parse_candidate_support(
    candidate: Mapping[str, Any],
    b: ModuleType,
) -> Any:
    raw = candidate.get("support_query_json", "[]")
    query = json.loads(str(raw)) if not isinstance(raw, list) else raw
    if not isinstance(query, list) or not query:
        raise ValueError("support_query_json must contain one or more conditions")
    families: list[str] = []
    columns: list[str] = []
    values: list[str] = []
    for item in query:
        if not isinstance(item, dict):
            raise ValueError("support condition is not an object")
        if str(item.get("operator", "")) != "eq":
            raise ValueError("OBS-084c supports only the sealed equality operator")
        families.append(str(item.get("support_family", "")))
        columns.append(str(item.get("column", "")))
        values.append(str(item.get("value", "")))
    if any(not x for x in families + columns):
        raise ValueError("support condition has an empty family or column")
    if len(query) > 2:
        raise ValueError("sealed support exceeds maximum conjunction depth two")
    return b.make_support_definition(families, columns, values)


def validate_candidate_bundle(
    discovery_dir: Path,
    discovery_script: Path,
    expected_candidate_manifest_id: str,
    freeze_payload: Mapping[str, Any],
    freeze_tables: Mapping[str, pd.DataFrame],
    b: ModuleType,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {role: discovery_dir / name for role, name in DISCOVERY_FILES.items()}
    for role in ("candidate_json", "candidate_csv", "thresholds", "input_manifest"):
        if not paths[role].exists():
            raise FileNotFoundError(paths[role])

    payload = json.loads(paths["candidate_json"].read_text(encoding="utf-8"))
    manifest_id = str(payload.get("candidate_manifest_id", ""))
    payload_without_id = dict(payload)
    payload_without_id.pop("candidate_manifest_id", None)
    computed_id = stable_hash(payload_without_id)

    validations: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str = "") -> None:
        validations.append(
            {
                "validation_check": check,
                "passed": bool(passed),
                "status": "pass" if passed else "fail",
                "detail": detail,
            }
        )

    add(
        "candidate_manifest_internal_hash",
        bool(manifest_id and manifest_id == computed_id),
        f"declared={manifest_id}; computed={computed_id}",
    )
    if expected_candidate_manifest_id and expected_candidate_manifest_id.lower() != "auto":
        add(
            "expected_candidate_manifest_identity",
            manifest_id == expected_candidate_manifest_id,
            f"expected={expected_candidate_manifest_id}; observed={manifest_id}",
        )
    else:
        add(
            "expected_candidate_manifest_identity",
            True,
            "explicit auto mode; internal identity still validated",
        )

    add(
        "candidate_manifest_schema",
        payload.get("schema") == "obs084b_candidate_manifest_v1",
        str(payload.get("schema")),
    )
    add(
        "candidate_manifest_status",
        payload.get("status") == "sealed_FL2_candidates",
        str(payload.get("status")),
    )
    add(
        "obs084a_freeze_identity",
        str(payload.get("obs084a_freeze_manifest_id", ""))
        == str(freeze_payload.get("freeze_manifest_id", "")),
        (
            f"candidate={payload.get('obs084a_freeze_manifest_id')}; "
            f"current={freeze_payload.get('freeze_manifest_id')}"
        ),
    )

    expected_confirmation_id = b.confirmation_partition_id(freeze_tables["partition"])
    add(
        "confirmation_partition_identity",
        str(payload.get("confirmation_partition_id", "")) == expected_confirmation_id,
        (
            f"candidate={payload.get('confirmation_partition_id')}; "
            f"current={expected_confirmation_id}"
        ),
    )

    candidate_csv = read_csv_required(paths["candidate_csv"])
    json_candidates = pd.DataFrame(payload.get("candidates", []))
    declared_count = int(payload.get("candidate_count", -1))
    add(
        "candidate_count_consistency",
        declared_count == len(json_candidates) == len(candidate_csv),
        f"declared={declared_count}; json={len(json_candidates)}; csv={len(candidate_csv)}",
    )

    required = [
        "candidate_id",
        "record_id",
        "relation",
        "carrier",
        "subclass",
        "confirmation_eligible",
        "failure_predicate",
        "failure_mode",
        "support_definition",
        "support_query_json",
        "metric",
        "expected_direction",
        "threshold_basis",
        "minimum_effect",
        "confirmation_partition_id",
        "eligible_control_records_json",
        "candidate_status",
        "fl_maturity",
        "candidate_manifest_status",
    ]
    missing_json = [c for c in required if c not in json_candidates.columns]
    missing_csv = [c for c in required if c not in candidate_csv.columns]
    add(
        "candidate_required_fields",
        not missing_json and not missing_csv,
        f"missing_json={missing_json}; missing_csv={missing_csv}",
    )

    duplicate_ids = (
        json_candidates["candidate_id"].astype(str).duplicated().any()
        if "candidate_id" in json_candidates.columns
        else True
    )
    add("candidate_ids_unique", not duplicate_ids, "")

    if not json_candidates.empty and not candidate_csv.empty and "candidate_id" in candidate_csv:
        json_ids = set(json_candidates["candidate_id"].astype(str))
        csv_ids = set(candidate_csv["candidate_id"].astype(str))
        add("candidate_csv_json_identity", json_ids == csv_ids, "candidate ID sets")
        critical = [
            "record_id",
            "relation",
            "carrier",
            "failure_predicate",
            "failure_mode",
            "support_query_json",
            "metric",
            "expected_direction",
            "threshold_basis",
            "minimum_effect",
            "confirmation_eligible",
            "candidate_status",
            "fl_maturity",
        ]
        mismatch_count = 0
        if json_ids == csv_ids:
            j = json_candidates.set_index("candidate_id")
            c = candidate_csv.set_index("candidate_id")
            for cid in sorted(json_ids):
                for col in critical:
                    if col not in j.columns or col not in c.columns:
                        continue
                    if not values_equal(j.at[cid, col], c.at[cid, col]):
                        mismatch_count += 1
        add(
            "candidate_csv_json_critical_fields",
            mismatch_count == 0,
            f"mismatches={mismatch_count}",
        )

    support_errors: list[str] = []
    for _, row in json_candidates.iterrows():
        try:
            support = parse_candidate_support(row, b)
            if support.support_definition != str(row["support_definition"]):
                support_errors.append(
                    f"{row['candidate_id']}: support definition mismatch"
                )
        except Exception as exc:
            support_errors.append(f"{row.get('candidate_id', '')}: {exc}")
    add(
        "sealed_support_queries_parse",
        not support_errors,
        json.dumps(support_errors[:10]),
    )

    status_ok = True
    if not json_candidates.empty:
        status_ok = bool(
            json_candidates["candidate_status"]
            .astype(str)
            .str.startswith("fl2_candidate_nominated")
            .all()
            and json_candidates["fl_maturity"].astype(str).eq("FL2").all()
            and json_candidates["candidate_manifest_status"]
            .astype(str)
            .eq("sealed_FL2_discovery_candidates")
            .all()
        )
    add("sealed_fl2_statuses", status_ok, "")

    input_manifest = read_csv_required(paths["input_manifest"])
    script_ok, expected_script_sha, actual_script_sha = validate_discovery_script_hash(
        input_manifest, discovery_script
    )
    add(
        "obs084b_script_hash",
        script_ok,
        f"expected={expected_script_sha}; actual={actual_script_sha}",
    )

    validation_df = pd.DataFrame(validations)
    if validation_df.empty or not validation_df["passed"].map(normalize_bool).all():
        failed = validation_df.loc[~validation_df["passed"].map(normalize_bool)]
        raise RuntimeError(
            "OBS-084b candidate bundle validation failed: "
            + "; ".join(failed["validation_check"].astype(str).tolist())
        )
    return payload, json_candidates, validation_df, input_manifest


def build_input_manifest(
    repo_root: Path,
    freeze_dir: Path,
    discovery_dir: Path,
    output_dir: Path,
    freeze_payload: Mapping[str, Any],
    source_validation: pd.DataFrame,
    discovery_script: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(role: str, path: Path) -> None:
        exists = path.exists() and path.is_file()
        try:
            relative = path.relative_to(repo_root)
            label = str(relative)
        except Exception:
            label = str(path)
        rows.append(
            {
                "artifact_role": role,
                "artifact_path": label,
                "exists": exists,
                "sha256": sha256_file(path) if exists else "",
                "obs084a_freeze_manifest_id": freeze_payload.get("freeze_manifest_id", ""),
            }
        )

    add("obs084a_freeze_manifest", freeze_dir / "obs084a_freeze_manifest.json")
    for _, row in source_validation.iterrows():
        rows.append(
            {
                "artifact_role": str(row.get("source_role", "")),
                "artifact_path": str(row.get("artifact_path", "")),
                "exists": normalize_bool(row.get("exists", False)),
                "sha256": str(row.get("actual_sha256", "")),
                "obs084a_freeze_manifest_id": freeze_payload.get("freeze_manifest_id", ""),
            }
        )
    for role, name in DISCOVERY_FILES.items():
        path = discovery_dir / name
        if path.exists():
            add(f"obs084b_{role}", path)
    add("obs084b_script", discovery_script)
    add("obs084c_script", Path(__file__).resolve())
    return pd.DataFrame(rows).drop_duplicates(["artifact_role", "artifact_path"])


# -----------------------------------------------------------------------------
# One-time confirmation opening and threshold application
# -----------------------------------------------------------------------------


def create_opening_lock(
    lock_path: Path,
    freeze_manifest_id: str,
    candidate_manifest_id: str,
    confirmation_partition_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            "Reserved confirmation has already been opened or completed. "
            f"Lock: {lock_path}; status={existing.get('status')!r}"
        )
    payload = {
        "schema": "obs084c_confirmation_opening_lock_v1",
        "status": "confirmation_partition_opened_incomplete",
        "opened_at": utc_now(),
        "obs084a_freeze_manifest_id": freeze_manifest_id,
        "obs084b_candidate_manifest_id": candidate_manifest_id,
        "confirmation_partition_id": confirmation_partition_id,
        "obs084c_script_sha256": sha256_file(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "note": "One-time reserved confirmation opening. Do not delete or bypass this lock without a documented protocol deviation.",
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def complete_opening_lock(
    lock_path: Path,
    lock_payload: Mapping[str, Any],
    confirmation_manifest_id: str,
    overall_status: str,
) -> None:
    payload = dict(lock_payload)
    payload.update(
        {
            "status": "confirmation_completed",
            "completed_at": utc_now(),
            "confirmation_manifest_id": confirmation_manifest_id,
            "overall_status": overall_status,
        }
    )
    lock_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def compare_threshold_artifacts(
    recorded: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> pd.DataFrame:
    key = ["support_family", "source_field", "threshold_name"]
    require_columns(recorded, [*key, "threshold_value"], "OBS-084b threshold artifact")
    require_columns(recomputed, [*key, "threshold_value"], "recomputed thresholds")
    merged = recorded.merge(
        recomputed,
        on=key,
        how="outer",
        suffixes=("_recorded", "_recomputed"),
        indicator=True,
    )
    merged["threshold_value_match"] = merged.apply(
        lambda r: (
            r["_merge"] == "both"
            and values_equal(
                r.get("threshold_value_recorded"),
                r.get("threshold_value_recomputed"),
                atol=1e-12,
            )
        ),
        axis=1,
    )
    merged["validation_pass"] = (
        merged["_merge"].eq("both") & merged["threshold_value_match"]
    )
    if not merged["validation_pass"].all():
        raise RuntimeError(
            "Recorded OBS-084b support thresholds do not match independent "
            "discovery-only recomputation from frozen feature values"
        )
    return merged


def threshold_pair(
    thresholds: pd.DataFrame,
    support_family: str,
) -> tuple[float, float, pd.DataFrame]:
    sub = thresholds[thresholds["support_family"].astype(str) == support_family].copy()
    q33 = sub[sub["threshold_name"].astype(str) == "q33"]
    q67 = sub[sub["threshold_name"].astype(str) == "q67"]
    if len(q33) != 1 or len(q67) != 1:
        raise RuntimeError(f"Expected one q33 and q67 threshold for {support_family}")
    return float(q33.iloc[0]["threshold_value"]), float(q67.iloc[0]["threshold_value"]), sub


def apply_recorded_support_thresholds(
    confirmation: pd.DataFrame,
    recorded_thresholds: pd.DataFrame,
    seam_protocol: pd.DataFrame,
) -> pd.DataFrame:
    out = confirmation.copy()
    q33, q67, _ = threshold_pair(recorded_thresholds, "scale_band")
    out["scale_band"] = pd.cut(
        pd.to_numeric(out["transition_midpoint"], errors="coerce"),
        bins=[-np.inf, q33, q67, np.inf],
        labels=["early", "middle", "late"],
        include_lowest=True,
    ).astype("object")

    sq33, sq67, seam_rows = threshold_pair(recorded_thresholds, "seam_relative")
    if seam_protocol.empty:
        raise RuntimeError("Frozen seam discretization protocol is missing")
    source_field = str(seam_protocol.iloc[0].get("source_field", ""))
    if not source_field or source_field not in out.columns:
        raise RuntimeError(f"Frozen seam source field unavailable: {source_field!r}")
    direction = " ".join(seam_rows.get("direction_rule", pd.Series(dtype=str)).astype(str))
    high_means_near = "higher" in direction.lower() and "more seam-adjacent" in direction.lower()
    labels = ["far", "intermediate", "near"] if high_means_near else ["near", "intermediate", "far"]
    out["seam_relative_region"] = pd.cut(
        pd.to_numeric(out[source_field], errors="coerce"),
        bins=[-np.inf, sq33, sq67, np.inf],
        labels=labels,
        include_lowest=True,
    ).astype("object")
    return out


# -----------------------------------------------------------------------------
# Confirmation-only diagnostic model
# -----------------------------------------------------------------------------


def confirmation_oof_predictions(
    record: Mapping[str, Any],
    confirmation_df: pd.DataFrame,
    features: Sequence[str],
    seed: int,
    b: ModuleType,
) -> tuple[pd.DataFrame, list[ConfirmationFailure]]:
    record_id = str(record["record_id"])
    relation = str(record["relation"])
    carrier = str(record["carrier"])
    classes = b.parse_relation_classes(relation)
    failures: list[ConfirmationFailure] = []

    missing_features = [f for f in features if f not in confirmation_df.columns]
    if missing_features:
        failures.append(
            ConfirmationFailure(
                "confirmation_model",
                "",
                record_id,
                "missing_frozen_carrier_features",
                json.dumps(missing_features),
            )
        )
        return pd.DataFrame(), failures

    sub = confirmation_df[confirmation_df["case"].astype(str).isin(classes)].copy()
    if sub.empty or set(sub["case"].astype(str).unique()) != set(classes):
        failures.append(
            ConfirmationFailure(
                "confirmation_model",
                "",
                record_id,
                "missing_relation_class_in_confirmation",
                json.dumps(sorted(sub.get("case", pd.Series(dtype=str)).astype(str).unique())),
            )
        )
        return pd.DataFrame(), failures

    feature_frame = sub[list(features)].apply(pd.to_numeric, errors="coerce")
    sub["predictor_missing_fraction"] = feature_frame.isna().mean(axis=1)
    sub["predictor_missing_any"] = feature_frame.isna().any(axis=1).astype(float)

    pred_parts: list[pd.DataFrame] = []
    clusters = sorted(sub["cluster_id"].dropna().astype(str).unique())
    for fold_index, heldout in enumerate(clusters):
        test_mask = sub["cluster_id"].astype(str) == heldout
        train = sub.loc[~test_mask].copy()
        test = sub.loc[test_mask].copy()
        train_classes = set(train["case"].astype(str).unique())
        if train.empty or test.empty or train_classes != set(classes):
            failures.append(
                ConfirmationFailure(
                    "confirmation_model",
                    "",
                    record_id,
                    "invalid_confirmation_leave_cluster_fold",
                    json.dumps(
                        {
                            "heldout_cluster": heldout,
                            "n_train": len(train),
                            "n_test": len(test),
                            "train_classes": sorted(train_classes),
                            "expected_classes": list(classes),
                        },
                        sort_keys=True,
                    ),
                )
            )
            continue

        model = b.make_model(seed + fold_index)
        x_train = train[list(features)].apply(pd.to_numeric, errors="coerce")
        x_test = test[list(features)].apply(pd.to_numeric, errors="coerce")
        model.fit(x_train, train["case"].astype(str))
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        model_classes = list(model.named_steps["model"].classes_)
        class_index = {str(c): i for i, c in enumerate(model_classes)}

        true_prob: list[float] = []
        max_other_prob: list[float] = []
        margins: list[float] = []
        for true_label, prob_row in zip(test["case"].astype(str), probabilities):
            if true_label not in class_index:
                true_prob.append(np.nan)
                max_other_prob.append(np.nan)
                margins.append(np.nan)
                continue
            tp = float(prob_row[class_index[true_label]])
            others = [float(prob_row[i]) for c, i in class_index.items() if c != true_label]
            mo = max(others) if others else 0.0
            true_prob.append(tp)
            max_other_prob.append(mo)
            margins.append(tp - mo)

        part = test.copy()
        part["record_id"] = record_id
        part["relation"] = relation
        part["carrier"] = carrier
        part["true_regime"] = part["case"].astype(str)
        part["predicted_regime"] = pd.Series(predicted, index=part.index).astype(str)
        part["predicted_probability"] = true_prob
        part["max_other_probability"] = max_other_prob
        part["true_class_margin"] = margins
        part["signed_margin"] = margins
        part["correct"] = part["true_regime"].eq(part["predicted_regime"])
        part["misclassification_loss"] = 1.0 - part["correct"].astype(float)
        part["margin_loss"] = -pd.to_numeric(part["true_class_margin"], errors="coerce")
        part["log_loss"] = -np.log(
            np.clip(pd.to_numeric(part["predicted_probability"], errors="coerce"), 1e-12, 1.0)
        )
        part["fold_id"] = f"confirmation_leave_object_out::{heldout}"
        part["heldout_cluster"] = heldout
        part["partition_role"] = "confirmation"
        part["diagnostic_model"] = "logreg_balanced_scaled_median_imputed"
        part["carrier_features_json"] = json.dumps(list(features))
        pred_parts.append(part)

    if not pred_parts:
        return pd.DataFrame(), failures
    out = pd.concat(pred_parts, ignore_index=True)
    if (out["partition_role"] != "confirmation").any():
        raise RuntimeError("Non-confirmation rows entered the confirmation model output")
    if out["observation_key"].duplicated().any():
        raise RuntimeError(f"Duplicate confirmation predictions for record {record_id}")
    return out, failures


# -----------------------------------------------------------------------------
# Candidate evaluation and FL3 entitlement
# -----------------------------------------------------------------------------


def predicate_from_candidate(candidate: Mapping[str, Any], b: ModuleType) -> dict[str, Any]:
    name = str(candidate["failure_predicate"])
    canonical = next((p for p in b.PREDICATES if p["failure_predicate"] == name), None)
    if canonical is None:
        raise ValueError(f"Unknown sealed failure predicate: {name}")
    observed = {
        "failure_predicate": name,
        "failure_mode": str(candidate["failure_mode"]),
        "metric": str(candidate["metric"]),
        "expected_direction": str(candidate["expected_direction"]),
        "minimum_effect": float(candidate["minimum_effect"]),
        "threshold_basis": str(candidate["threshold_basis"]),
    }
    for field in ("failure_mode", "metric", "expected_direction"):
        if str(observed[field]) != str(canonical[field]):
            raise ValueError(
                f"Sealed predicate field {field} differs from OBS-084b canonical definition"
            )
    return observed


def make_confirmation_args(
    candidate_payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> SimpleNamespace:
    config = candidate_payload.get("search_configuration", {})
    rows = config.get("minimum_rows", {})
    clusters = config.get("minimum_clusters", {})
    return SimpleNamespace(
        min_site_rows=int(rows.get("site", 8)),
        min_complement_rows=int(rows.get("complement", 12)),
        min_class_rows=int(rows.get("per_class", 2)),
        min_site_clusters=int(clusters.get("site", 2)),
        min_complement_clusters=int(clusters.get("complement", 2)),
        min_shared_clusters=int(clusters.get("shared", 2)),
        n_cluster_bootstrap=int(args.n_cluster_bootstrap),
        n_permutations=int(args.n_permutations),
        alpha=float(args.alpha),
        seed=int(args.seed),
        min_direction_consistency=float(args.min_direction_consistency),
        min_control_adjusted_effect=float(args.min_control_adjusted_effect),
        min_positive_control_share=float(args.min_positive_control_share),
    )


def candidate_control_set_matches(
    candidate: Mapping[str, Any],
    controls: pd.DataFrame,
) -> tuple[bool, list[str], list[str]]:
    declared_raw = candidate.get("eligible_control_records_json", "[]")
    declared = sorted(map(str, json.loads(str(declared_raw))))
    observed = sorted(
        controls.loc[
            (controls["record_id"].astype(str) == str(candidate["record_id"]))
            & controls["evidence_available"].map(normalize_bool),
            "control_record_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    return declared == observed, declared, observed


def outcome_status(row: Mapping[str, Any]) -> str:
    if not normalize_bool(row.get("protocol_match", False)):
        return "confirmation_protocol_mismatch"
    if not normalize_bool(row.get("record_testable", False)):
        return "confirmation_not_testable"
    if not normalize_bool(row.get("support_columns_available", False)):
        return "confirmation_support_unavailable"
    if not normalize_bool(row.get("complement_admissible", False)):
        return "confirmation_complement_inadmissible"
    if not normalize_bool(row.get("direction_match", False)):
        return "confirmation_direction_reversed"
    if not normalize_bool(row.get("predicate_semantics_pass", False)) or not normalize_bool(
        row.get("minimum_effect_pass", False)
    ):
        return "confirmation_signal_absent"
    if not normalize_bool(row.get("cluster_sensitivity_pass", False)):
        return "confirmation_uncertain_cluster_support"
    if not normalize_bool(row.get("control_robustness_pass", False)):
        return "confirmation_control_explained"
    if not normalize_bool(row.get("confirmation_multiplicity_pass", False)):
        return "confirmation_multiplicity_not_survived"
    if normalize_bool(row.get("confirmation_eligible", False)):
        return "fl3_confirmed_direct_witness"
    return "confirmation_reproduced_but_claim_capped_at_fl2"


def build_direct_witness_registry(
    outcomes: pd.DataFrame,
    freeze_payload: Mapping[str, Any],
    candidate_manifest_id: str,
    confirmation_partition_id: str,
    input_manifest: pd.DataFrame,
) -> pd.DataFrame:
    confirmed = outcomes[
        outcomes["confirmation_status"].astype(str) == "fl3_confirmed_direct_witness"
    ].copy()
    if confirmed.empty:
        return pd.DataFrame(
            columns=[
                "witness_id",
                "candidate_id",
                "record_id",
                "relation",
                "carrier",
                "failure_predicate",
                "failure_mode",
                "support_definition",
                "confirmation_site_relative_contrast",
                "confirmation_q_sealed_family",
                "confirmation_control_adjusted_contrast",
                "fl_maturity",
                "directness_scope",
                "claim_entitlement",
                "obs084a_freeze_manifest_id",
                "obs084b_candidate_manifest_id",
                "confirmation_partition_id",
                "source_hashes_json",
                "witness_version_hash",
            ]
        )

    source_hashes = json.dumps(
        input_manifest[["artifact_role", "artifact_path", "sha256"]]
        .fillna("")
        .to_dict("records"),
        sort_keys=True,
    )
    rows: list[dict[str, Any]] = []
    for _, row in confirmed.iterrows():
        core = {
            "candidate_id": str(row["candidate_id"]),
            "record_id": str(row["record_id"]),
            "failure_predicate": str(row["failure_predicate"]),
            "support_query_json": str(row["support_query_json"]),
            "freeze_manifest_id": str(freeze_payload.get("freeze_manifest_id", "")),
            "candidate_manifest_id": candidate_manifest_id,
            "confirmation_partition_id": confirmation_partition_id,
        }
        witness_hash = stable_hash(core)
        rows.append(
            {
                "witness_id": "OBS084C-W-" + witness_hash[:20],
                "candidate_id": row["candidate_id"],
                "record_id": row["record_id"],
                "relation": row["relation"],
                "carrier": row["carrier"],
                "failure_predicate": row["failure_predicate"],
                "failure_mode": row["failure_mode"],
                "support_definition": row["support_definition"],
                "support_query_json": row["support_query_json"],
                "confirmation_site_relative_contrast": row["confirmation_site_relative_contrast"],
                "confirmation_bootstrap_ci_low": row["confirmation_bootstrap_ci_low"],
                "confirmation_bootstrap_ci_high": row["confirmation_bootstrap_ci_high"],
                "confirmation_q_sealed_family": row["confirmation_q_sealed_family"],
                "confirmation_control_adjusted_contrast": row["median_control_adjusted_contrast"],
                "positive_control_adjusted_share": row["positive_control_adjusted_share"],
                "fl_maturity": "FL3",
                "directness_scope": "artifact-direct for the declared record, predicate, support, carrier contract, confirmation partition, and provenance only",
                "claim_entitlement": "confirmed localized degradation witness; no causal origin, repair target, actionability, external generalization, or formal topology",
                "obs084a_freeze_manifest_id": freeze_payload.get("freeze_manifest_id", ""),
                "obs084b_candidate_manifest_id": candidate_manifest_id,
                "confirmation_partition_id": confirmation_partition_id,
                "source_hashes_json": source_hashes,
                "witness_version_hash": witness_hash,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_report(
    path: Path,
    status: str,
    freeze_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    confirmation_manifest_id: str,
    input_manifest: pd.DataFrame,
    validation: pd.DataFrame,
    threshold_validation: pd.DataFrame,
    observation_losses: pd.DataFrame,
    outcomes: pd.DataFrame,
    witnesses: pd.DataFrame,
    failures: pd.DataFrame,
    lock_path: Path,
) -> None:
    outcome_counts = (
        outcomes["confirmation_status"].value_counts().rename_axis("confirmation_status").reset_index(name="count")
        if not outcomes.empty
        else pd.DataFrame(columns=["confirmation_status", "count"])
    )
    display_cols = [
        "candidate_id",
        "record_id",
        "failure_predicate",
        "support_definition",
        "discovery_site_relative_contrast",
        "confirmation_site_relative_contrast",
        "confirmation_bootstrap_ci_low",
        "confirmation_permutation_p",
        "confirmation_q_sealed_family",
        "median_control_adjusted_contrast",
        "confirmation_status",
    ]
    lines = [
        "# OBS-084c — Direct Failure-Support Reserved Confirmation",
        "",
        "## State",
        "",
        f"Confirmation completed with status: `{status}`",
        "",
        f"OBS-084a freeze manifest: `{freeze_payload.get('freeze_manifest_id', '')}`",
        f"OBS-084b candidate manifest: `{candidate_payload.get('candidate_manifest_id', '')}`",
        f"OBS-084c confirmation manifest: `{confirmation_manifest_id}`",
        "",
        "This stage opened the frozen confirmation partition once and evaluated only the sealed OBS-084b FL2 candidate family. It performed no new support search, predicate search, threshold fitting, or candidate ranking.",
        "",
        "## Canonical guardrails",
        "",
        "> Directness is artifact-direct, not metaphysically direct and not causally direct.",
        "",
        "> Discovery nominates a support; reserved evidence earns the localization claim.",
        "",
        "Any FL3 result is scoped to the declared record, predicate, support, carrier contract, partition, and provenance. No result is actionable, causal, externally generalized, repaired, or formally topological.",
        "",
        "## One-time opening",
        "",
        f"Confirmation opening lock: `{lock_path}`",
        "",
        "The lock was created before confirmation outcome evaluation and completed only after all confirmation artifacts were written.",
        "",
        "## Frozen and sealed input validation",
        "",
        markdown_table(validation, 100),
        "",
        "## Input artifact identity",
        "",
        markdown_table(input_manifest[[c for c in ["artifact_role", "artifact_path", "exists", "sha256"] if c in input_manifest.columns]], 100),
        "",
        "## Discovery-threshold verification",
        "",
        markdown_table(
            threshold_validation[[c for c in [
                "support_family", "source_field", "threshold_name",
                "threshold_value_recorded", "threshold_value_recomputed",
                "validation_pass",
            ] if c in threshold_validation.columns]],
            20,
        ),
        "",
        "Recorded OBS-084b scale and seam cut values were independently recomputed from frozen discovery feature values. The verified recorded values were then applied unchanged to confirmation.",
        "",
        "## Confirmation-only diagnostic instrument",
        "",
        f"- Sealed candidates evaluated: {len(outcomes)}",
        f"- Confirmation observation-loss rows: {len(observation_losses)}",
        f"- Confirmation observations represented: {observation_losses['observation_id'].nunique() if not observation_losses.empty else 0}",
        "- Structural dependence unit: object (`cluster_id`)",
        "- Diagnostic model: confirmation-only leave-one-object-out balanced logistic regression",
        "- Multiplicity family: exactly the sealed OBS-084b candidate family",
        "",
        "## Outcome counts",
        "",
        markdown_table(outcome_counts, 30),
        "",
        "## Candidate outcomes",
        "",
        markdown_table(outcomes[[c for c in display_cols if c in outcomes.columns]], 100),
        "",
        "## FL3 direct-witness registry",
        "",
        markdown_table(
            witnesses[[c for c in [
                "witness_id", "candidate_id", "record_id", "failure_predicate",
                "support_definition", "confirmation_site_relative_contrast",
                "confirmation_bootstrap_ci_low", "confirmation_q_sealed_family",
                "confirmation_control_adjusted_contrast", "fl_maturity",
                "directness_scope", "witness_version_hash",
            ] if c in witnesses.columns]],
            50,
        ),
        "",
        "## Failures and exclusions",
        "",
        markdown_table(failures, 100) if not failures.empty else "_No execution failures._",
        "",
        "## Interpretation",
        "",
        "Candidates that reproduced directionally but failed uncertainty, controls, multiplicity, or claim-entitlement gates remain unconfirmed FL2 evidence. C1 contrast-limited candidates cannot be promoted beyond FL2 in this stage even if their confirmation signal reproduces.",
        "",
        "## Canonical result statement",
        "",
        "OBS-084c evaluates the complete sealed OBS-084b candidate family once on the frozen confirmation partition. It may establish scoped FL3 artifact-direct witnesses, but establishes no causal origin, repair target, intervention readiness, actionability, external generalization, or formal topology.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if not (0 < args.alpha < 1):
        raise ValueError("--alpha must be in (0, 1)")
    if not (0 < args.confirmation_fdr <= 1):
        raise ValueError("--confirmation-fdr must be in (0, 1]")
    if args.n_cluster_bootstrap < 1 or args.n_permutations < 1:
        raise ValueError("bootstrap and permutation counts must be positive")

    repo_root = args.repo_root.resolve()
    freeze_dir = resolve_path(repo_root, args.freeze_dir)
    discovery_dir = resolve_path(repo_root, args.discovery_dir)
    obs083_dir = resolve_path(repo_root, args.obs083_dir)
    registry_path = resolve_path(repo_root, args.registry)
    discovery_script = resolve_path(repo_root, args.discovery_script)
    output_dir = resolve_path(repo_root, args.output_dir)
    lock_path = discovery_dir / "obs084c_confirmation_opening_lock.json"

    # Import only the exact OBS-084b implementation whose hash is later checked.
    b = load_module_from_path(discovery_script)

    freeze_payload, freeze_tables, source_validation = b.load_and_validate_freeze(
        repo_root, freeze_dir, args.require_repo_commit
    )
    candidate_payload, candidates, candidate_validation, discovery_input_manifest = (
        validate_candidate_bundle(
            discovery_dir,
            discovery_script,
            args.expected_candidate_manifest_id,
            freeze_payload,
            freeze_tables,
            b,
        )
    )
    candidate_manifest_id = str(candidate_payload["candidate_manifest_id"])
    confirmation_partition_id = b.confirmation_partition_id(freeze_tables["partition"])

    # Validate registry/control artifacts before opening confirmation.
    subclasses_path = obs083_dir / OBS083_FILES["subclasses"]
    relation_controls_path = obs083_dir / OBS083_FILES["relation_controls"]
    carrier_controls_path = obs083_dir / OBS083_FILES["carrier_controls"]
    for path in (
        subclasses_path,
        relation_controls_path,
        carrier_controls_path,
        registry_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    registry = read_csv_required(registry_path)
    subclasses = read_csv_required(subclasses_path)
    record_catalog = b.load_record_catalog(
        registry, subclasses, freeze_tables["partition_balance"]
    )
    carrier_features = b.load_carrier_features(freeze_tables["carrier_features"])
    controls = b.load_controls(
        read_csv_required(relation_controls_path),
        read_csv_required(carrier_controls_path),
    )

    candidate_records = set(candidates["record_id"].astype(str))
    missing_records = sorted(candidate_records - set(record_catalog["record_id"].astype(str)))
    if missing_records:
        raise RuntimeError(f"Sealed candidates reference unknown registry records: {missing_records}")

    input_manifest = build_input_manifest(
        repo_root,
        freeze_dir,
        discovery_dir,
        output_dir,
        freeze_payload,
        source_validation,
        discovery_script,
    )

    if args.validate_only:
        print("OBS-084c validation complete: sealed bundle valid; confirmation not opened")
        print(f"Candidates sealed: {len(candidates)}")
        print(f"Candidate manifest ID: {candidate_manifest_id}")
        return 0

    if (output_dir / "obs084c_confirmation_manifest.json").exists():
        raise RuntimeError("OBS-084c confirmation manifest already exists; reserved confirmation cannot be rerun")

    # Open exactly once before reading canonical feature values for confirmation.
    lock_payload = create_opening_lock(
        lock_path,
        str(freeze_payload["freeze_manifest_id"]),
        candidate_manifest_id,
        confirmation_partition_id,
        output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_feature_path = resolve_path(
        repo_root, str(freeze_payload["canonical_feature_table"])
    )
    feature_df = read_csv_required(canonical_feature_path)
    b.validate_key_and_partition(
        feature_df,
        freeze_tables["observation_key"],
        freeze_tables["partition"],
    )
    prepared = b.prepare_feature_table(feature_df, freeze_tables["partition"])
    discovery_features = prepared[
        prepared["partition"].astype(str) == "discovery"
    ].copy()
    confirmation = prepared[
        prepared["partition"].astype(str) == "confirmation"
    ].copy()
    if confirmation.empty:
        raise RuntimeError("Frozen confirmation partition is empty")
    if set(discovery_features["observation_key"].astype(str)) & set(
        confirmation["observation_key"].astype(str)
    ):
        raise RuntimeError("Discovery and confirmation keys overlap")

    # Recompute discovery-only thresholds solely to validate the recorded frozen
    # values. Confirmation values are never used to fit a threshold.
    _, recomputed_thresholds = b.derive_support_fields(
        discovery_features, freeze_tables["seam_protocol"]
    )
    recorded_thresholds = read_csv_required(
        discovery_dir / DISCOVERY_FILES["thresholds"]
    )
    threshold_validation = compare_threshold_artifacts(
        recorded_thresholds, recomputed_thresholds
    )
    confirmation = apply_recorded_support_thresholds(
        confirmation, recorded_thresholds, freeze_tables["seam_protocol"]
    )

    # Build confirmation-only observation losses for all registry records so the
    # exact sealed relation and carrier controls can be re-evaluated.
    observation_parts: list[pd.DataFrame] = []
    observation_by_record: dict[str, pd.DataFrame] = {}
    failure_rows: list[dict[str, Any]] = []
    for _, record in record_catalog.iterrows():
        rid = str(record["record_id"])
        carrier = str(record["carrier"])
        if carrier not in carrier_features:
            failure_rows.append(
                ConfirmationFailure(
                    "confirmation_model", "", rid,
                    "carrier_missing_from_frozen_manifest", carrier,
                ).__dict__
            )
            continue
        obs, failures = confirmation_oof_predictions(
            record,
            confirmation,
            carrier_features[carrier],
            args.seed,
            b,
        )
        failure_rows.extend(f.__dict__ for f in failures)
        if not obs.empty:
            observation_by_record[rid] = obs
            observation_parts.append(obs)

    observation_losses = (
        pd.concat(observation_parts, ignore_index=True)
        if observation_parts
        else pd.DataFrame()
    )
    if not observation_losses.empty and (
        observation_losses["partition_role"].astype(str) != "confirmation"
    ).any():
        raise RuntimeError("Non-confirmation rows entered confirmation observation losses")

    cargs = make_confirmation_args(candidate_payload, args)
    record_index = record_catalog.set_index(record_catalog["record_id"].astype(str))

    support_validation_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    control_detail_rows: list[dict[str, Any]] = []
    control_aggregate_rows: list[dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        cid = str(candidate["candidate_id"])
        rid = str(candidate["record_id"])
        protocol_match = True
        protocol_notes: list[str] = []

        try:
            support = parse_candidate_support(candidate, b)
            predicate = predicate_from_candidate(candidate, b)
        except Exception as exc:
            failure_rows.append(
                ConfirmationFailure(
                    "candidate_protocol", cid, rid, "candidate_protocol_mismatch", str(exc)
                ).__dict__
            )
            support_validation_rows.append(
                {
                    "candidate_id": cid,
                    "record_id": rid,
                    "protocol_match": False,
                    "support_columns_available": False,
                    "record_testable": False,
                    "validation_status": "confirmation_protocol_mismatch",
                    "validation_detail": str(exc),
                }
            )
            continue

        if rid not in observation_by_record:
            support_validation_rows.append(
                {
                    "candidate_id": cid,
                    "record_id": rid,
                    "support_id": support.support_id,
                    "support_definition": support.support_definition,
                    "protocol_match": True,
                    "support_columns_available": False,
                    "record_testable": False,
                    "validation_status": "confirmation_not_testable",
                    "validation_detail": "record-level confirmation observation losses unavailable",
                }
            )
            continue

        rdf = observation_by_record[rid]
        support_columns_available = all(col in rdf.columns for col in support.columns)
        if not support_columns_available:
            support_validation_rows.append(
                {
                    "candidate_id": cid,
                    "record_id": rid,
                    "support_id": support.support_id,
                    "support_definition": support.support_definition,
                    "protocol_match": True,
                    "support_columns_available": False,
                    "record_testable": True,
                    "validation_status": "confirmation_support_unavailable",
                    "validation_detail": json.dumps(
                        [col for col in support.columns if col not in rdf.columns]
                    ),
                }
            )
            continue

        current = record_index.loc[rid]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[0]
        for field in ("relation", "carrier"):
            if str(candidate[field]) != str(current[field]):
                protocol_match = False
                protocol_notes.append(
                    f"{field}: sealed={candidate[field]!r}, current={current[field]!r}"
                )
        current_eligible = normalize_bool(current.get("confirmation_eligible", False))
        if normalize_bool(candidate["confirmation_eligible"]) != current_eligible:
            protocol_match = False
            protocol_notes.append("confirmation eligibility changed")
        controls_match, declared_controls, observed_controls = candidate_control_set_matches(
            candidate, controls
        )
        if not controls_match:
            protocol_match = False
            protocol_notes.append(
                f"control set mismatch: declared={declared_controls}; observed={observed_controls}"
            )

        threshold = float(current["threshold"]) if pd.notna(current.get("threshold")) else np.nan
        result = b.compute_site_contrast(rdf, support, predicate, threshold, cargs)
        result.update(
            {
                "candidate_id": cid,
                "record_id": rid,
                "relation": candidate["relation"],
                "carrier": candidate["carrier"],
                "subclass": candidate["subclass"],
                "confirmation_eligible": normalize_bool(candidate["confirmation_eligible"]),
                "protocol_match": protocol_match,
                "protocol_notes": "; ".join(protocol_notes),
                "record_testable": True,
                "support_columns_available": True,
                "discovery_site_relative_contrast": float(candidate["site_relative_contrast"]),
                "discovery_bootstrap_ci_low": float(candidate["bootstrap_ci_low"]),
                "discovery_bootstrap_ci_high": float(candidate["bootstrap_ci_high"]),
                "discovery_permutation_p": float(candidate["permutation_p"]),
                "discovery_q_record_predicate": float(candidate["permutation_q_record_predicate"]),
                "discovery_control_adjusted_contrast": float(candidate["median_control_adjusted_contrast"]),
                "confirmation_site_relative_contrast": result["site_relative_contrast"],
                "complement_definition": str(candidate.get("complement_definition", "")),
                "sealed_matching_variables": str(candidate.get("matching_variables", "")),
                "sealed_exposure_normalization": str(candidate.get("exposure_normalization", "")),
            }
        )
        direction_match = (
            np.isfinite(result["site_relative_contrast"])
            and np.isfinite(float(candidate["site_relative_contrast"]))
            and np.sign(result["site_relative_contrast"])
            == np.sign(float(candidate["site_relative_contrast"]))
            and result["site_relative_contrast"] > 0
        )
        result["direction_match"] = direction_match
        result["minimum_effect_pass"] = bool(
            np.isfinite(result["site_relative_contrast"])
            and result["site_relative_contrast"] >= float(candidate["minimum_effect"])
        )

        uncertainty = b.cluster_uncertainty(
            rdf,
            support,
            str(candidate["metric"]),
            cargs,
            f"{cid}::confirmation",
        )
        p_value, permutation_count, permutation_method = b.permutation_p_value(
            rdf,
            support,
            str(candidate["metric"]),
            cargs,
            f"{cid}::confirmation",
        )
        result.update(
            {
                "confirmation_bootstrap_ci_low": uncertainty["bootstrap_ci_low"],
                "confirmation_bootstrap_ci_high": uncertainty["bootstrap_ci_high"],
                "confirmation_direction_consistency": uncertainty["direction_consistency"],
                "confirmation_bootstrap_positive_share": uncertainty["bootstrap_positive_share"],
                "confirmation_loo_successful_count": uncertainty["loo_successful_count"],
                "confirmation_independent_cluster_count": uncertainty["independent_cluster_count"],
                "confirmation_permutation_p": p_value if np.isfinite(p_value) else 1.0,
                "confirmation_permutation_count": permutation_count,
                "confirmation_permutation_method": permutation_method,
            }
        )
        result["cluster_sensitivity_pass"] = bool(
            np.isfinite(uncertainty["bootstrap_ci_low"])
            and uncertainty["bootstrap_ci_low"] > 0
            and np.isfinite(uncertainty["direction_consistency"])
            and uncertainty["direction_consistency"] >= args.min_direction_consistency
            and uncertainty["loo_successful_count"] >= 2
        )

        support_validation_rows.append(
            {
                "candidate_id": cid,
                "record_id": rid,
                "support_id": support.support_id,
                "support_definition": support.support_definition,
                "support_query_json": support.support_query_json,
                "support_columns": "|".join(support.columns),
                "support_values": "|".join(support.values),
                "protocol_match": protocol_match,
                "support_columns_available": True,
                "record_testable": True,
                "n_site_rows": result["n_site_rows"],
                "n_complement_rows": result["n_complement_rows"],
                "n_site_clusters": result["n_site_clusters"],
                "n_complement_clusters": result["n_complement_clusters"],
                "n_shared_clusters": result["n_shared_clusters"],
                "complement_admissible": result["complement_admissible"],
                "matching_check_json": result["matching_check_json"],
                "validation_status": result["initial_status"] if protocol_match else "confirmation_protocol_mismatch",
                "validation_detail": "; ".join(protocol_notes),
            }
        )
        contrast_rows.append(result)
        uncertainty_rows.append(
            {
                "candidate_id": cid,
                "record_id": rid,
                "support_id": support.support_id,
                "failure_predicate": candidate["failure_predicate"],
                **uncertainty,
                "permutation_p": p_value if np.isfinite(p_value) else 1.0,
                "permutation_count": permutation_count,
                "permutation_method": permutation_method,
            }
        )

    support_validation = pd.DataFrame(support_validation_rows)
    contrasts = pd.DataFrame(contrast_rows)
    uncertainty_df = pd.DataFrame(uncertainty_rows)

    # Control adjustment is computed before multiplicity but uses only the sealed
    # candidate support and the current frozen OBS-083 control set.
    if not contrasts.empty:
        for _, target in contrasts.iterrows():
            detail, aggregate = b.compute_control_adjustment(
                target,
                observation_by_record,
                controls,
                record_catalog,
                cargs,
            )
            for row in detail:
                row["candidate_id"] = target["candidate_id"]
                row["confirmation_partition_id"] = confirmation_partition_id
            aggregate["candidate_id"] = target["candidate_id"]
            control_detail_rows.extend(detail)
            control_aggregate_rows.append(aggregate)

    control_details = pd.DataFrame(control_detail_rows)
    control_aggregates = pd.DataFrame(control_aggregate_rows)
    if not control_details.empty:
        control_details["row_type"] = "control_record"
    if not control_aggregates.empty:
        control_aggregates["row_type"] = "candidate_aggregate"
    control_output = pd.concat(
        [x for x in (control_details, control_aggregates) if not x.empty],
        ignore_index=True,
        sort=False,
    ) if (not control_details.empty or not control_aggregates.empty) else pd.DataFrame()
    if not contrasts.empty and not control_aggregates.empty:
        contrasts = contrasts.merge(
            control_aggregates,
            on=["candidate_id", "record_id", "support_id", "failure_predicate"],
            how="left",
            validate="one_to_one",
        )

    # Every sealed candidate remains in the family. Untestable candidates receive
    # p=1 and remain in the denominator.
    base = candidates.copy()
    if not contrasts.empty:
        outcome_work = base.merge(
            contrasts,
            on=[
                "candidate_id", "record_id", "relation", "carrier", "subclass",
                "confirmation_eligible", "failure_predicate", "failure_mode",
            ],
            how="left",
            suffixes=("_sealed", ""),
            validate="one_to_one",
        )
    else:
        outcome_work = base.copy()

    if not support_validation.empty:
        val_keep = [
            "candidate_id", "protocol_match", "support_columns_available",
            "record_testable", "validation_status", "validation_detail",
        ]
        existing = [c for c in val_keep if c in support_validation.columns]
        outcome_work = outcome_work.merge(
            support_validation[existing].drop_duplicates("candidate_id"),
            on="candidate_id",
            how="left",
            suffixes=("", "_validation"),
            validate="one_to_one",
        )
        for col in ("protocol_match", "support_columns_available", "record_testable"):
            vcol = f"{col}_validation"
            if vcol in outcome_work.columns:
                outcome_work[col] = outcome_work[col].fillna(outcome_work[vcol]) if col in outcome_work else outcome_work[vcol]

    for col, default in (
        ("protocol_match", False),
        ("support_columns_available", False),
        ("record_testable", False),
        ("complement_admissible", False),
        ("direction_match", False),
        ("predicate_semantics_pass", False),
        ("minimum_effect_pass", False),
        ("cluster_sensitivity_pass", False),
        ("control_robustness_pass", False),
        ("control_robustness_status", "not_evaluated"),
        ("median_control_adjusted_contrast", np.nan),
        ("positive_control_adjusted_share", np.nan),
        ("confirmation_permutation_p", 1.0),
    ):
        if col not in outcome_work.columns:
            outcome_work[col] = default
        else:
            outcome_work[col] = outcome_work[col].fillna(default)

    outcome_work["confirmation_q_sealed_family"] = b.bh_adjust(
        pd.to_numeric(outcome_work["confirmation_permutation_p"], errors="coerce").fillna(1.0)
    )
    outcome_work["confirmation_multiplicity_pass"] = (
        outcome_work["confirmation_q_sealed_family"] <= args.confirmation_fdr
    )
    outcome_work["confirmation_status"] = outcome_work.apply(outcome_status, axis=1)
    outcome_work["confirmation_fl_maturity"] = np.where(
        outcome_work["confirmation_status"].eq("fl3_confirmed_direct_witness"),
        "FL3",
        "FL2_unconfirmed",
    )
    outcome_work["confirmation_claim_scope"] = np.where(
        outcome_work["confirmation_status"].eq("fl3_confirmed_direct_witness"),
        "artifact-direct localized degradation witness for the exact sealed record/predicate/support/contract/provenance",
        "no FL3 claim entitlement",
    )

    multiplicity = outcome_work[
        [
            "candidate_id", "record_id", "failure_predicate",
            "confirmation_permutation_p", "confirmation_q_sealed_family",
            "confirmation_multiplicity_pass", "confirmation_status",
        ]
    ].copy()
    multiplicity["multiplicity_family"] = "sealed_obs084b_candidate_family"
    multiplicity["candidate_denominator"] = len(outcome_work)
    multiplicity["correction_rule"] = "Benjamini-Hochberg over exactly all sealed OBS-084b candidates; untestable candidates retained with p=1"
    multiplicity["confirmation_fdr"] = args.confirmation_fdr

    witnesses = build_direct_witness_registry(
        outcome_work,
        freeze_payload,
        candidate_manifest_id,
        confirmation_partition_id,
        input_manifest,
    )
    failures_df = pd.DataFrame(failure_rows)

    n_fl3 = int(outcome_work["confirmation_status"].eq("fl3_confirmed_direct_witness").sum())
    n_capped = int(
        outcome_work["confirmation_status"].eq(
            "confirmation_reproduced_but_claim_capped_at_fl2"
        ).sum()
    )
    if n_fl3 > 0:
        overall_status = "fl3_direct_witnesses_confirmed"
    elif n_capped > 0:
        overall_status = "confirmation_completed_no_fl3_with_capped_reproductions"
    else:
        overall_status = "valid_null_no_sealed_candidate_reached_fl3"

    # Stable, explicit output column selection.
    outcome_columns = [
        "candidate_id", "record_id", "relation", "carrier", "subclass",
        "confirmation_eligible", "failure_predicate", "failure_mode",
        "support_definition", "support_query_json", "metric",
        "expected_direction", "threshold_basis", "minimum_effect",
        "discovery_site_relative_contrast", "confirmation_site_relative_contrast",
        "direction_match", "n_site_rows", "n_complement_rows",
        "n_site_clusters", "n_complement_clusters", "n_shared_clusters",
        "complement_admissible", "predicate_semantics_pass", "minimum_effect_pass",
        "confirmation_bootstrap_ci_low", "confirmation_bootstrap_ci_high",
        "confirmation_direction_consistency", "confirmation_loo_successful_count",
        "cluster_sensitivity_pass", "confirmation_permutation_p",
        "confirmation_q_sealed_family", "confirmation_multiplicity_pass",
        "admissible_control_count", "median_control_site_relative_contrast",
        "median_control_adjusted_contrast", "positive_control_adjusted_share",
        "control_robustness_pass", "control_robustness_status",
        "protocol_match", "protocol_notes", "confirmation_status",
        "confirmation_fl_maturity", "confirmation_claim_scope",
    ]
    outcomes = outcome_work[[c for c in outcome_columns if c in outcome_work.columns]].copy()
    outcomes["status_order"] = outcomes["confirmation_status"].map(OUTCOME_STATUS_ORDER).fillna(99)
    outcomes = outcomes.sort_values(["status_order", "candidate_id"]).drop(columns="status_order")

    # Write all artifacts before completing the one-time opening lock.
    output_tables = {
        "obs084c_input_manifest.csv": input_manifest,
        "obs084c_candidate_manifest_validation.csv": candidate_validation,
        "obs084c_confirmation_observation_losses.csv": observation_losses,
        "obs084c_support_complement_validation.csv": support_validation,
        "obs084c_confirmation_site_contrasts.csv": contrasts,
        "obs084c_confirmation_control_adjustment.csv": control_output,
        "obs084c_cluster_uncertainty.csv": uncertainty_df,
        "obs084c_multiplicity_audit.csv": multiplicity,
        "obs084c_candidate_outcomes.csv": outcomes,
        "obs084c_direct_witness_registry.csv": witnesses,
        "obs084c_confirmation_failures.csv": failures_df,
    }
    for name, frame in output_tables.items():
        frame.to_csv(output_dir / name, index=False)

    confirmation_payload = {
        "schema": "obs084c_confirmation_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "overall_status": overall_status,
        "obs084a_freeze_manifest_id": freeze_payload["freeze_manifest_id"],
        "obs084b_candidate_manifest_id": candidate_manifest_id,
        "confirmation_partition_id": confirmation_partition_id,
        "one_time_opening_lock_sha256_at_open": sha256_file(lock_path),
        "sealed_candidate_count": int(len(candidates)),
        "evaluated_candidate_count": int(len(outcomes)),
        "fl3_direct_witness_count": int(len(witnesses)),
        "capped_reproduction_count": n_capped,
        "confirmation_configuration": {
            "model": "confirmation-only leave-one-object-out balanced logistic regression",
            "cluster_unit": "object",
            "cluster_bootstrap": args.n_cluster_bootstrap,
            "permutations": args.n_permutations,
            "alpha": args.alpha,
            "confirmation_fdr": args.confirmation_fdr,
            "min_direction_consistency": args.min_direction_consistency,
            "min_control_adjusted_effect": args.min_control_adjusted_effect,
            "min_positive_control_share": args.min_positive_control_share,
            "multiplicity_family": "exact sealed OBS-084b candidate family",
            "threshold_rule": "OBS-084b discovery-fitted thresholds independently verified and applied unchanged",
        },
        "input_hashes": input_manifest[
            ["artifact_role", "artifact_path", "sha256"]
        ].fillna("").to_dict("records"),
        "candidate_outcomes": outcomes.to_dict("records"),
        "direct_witnesses": witnesses.to_dict("records"),
    }
    confirmation_manifest_id = stable_hash(confirmation_payload)
    confirmation_payload["confirmation_manifest_id"] = confirmation_manifest_id
    (output_dir / "obs084c_confirmation_manifest.json").write_text(
        json.dumps(confirmation_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "overall_status": overall_status,
                "obs084a_freeze_manifest_id": freeze_payload["freeze_manifest_id"],
                "obs084b_candidate_manifest_id": candidate_manifest_id,
                "confirmation_manifest_id": confirmation_manifest_id,
                "sealed_candidates": len(candidates),
                "evaluated_candidates": len(outcomes),
                "fl3_direct_witnesses": len(witnesses),
                "capped_fl2_reproductions": n_capped,
                "confirmation_observation_loss_rows": len(observation_losses),
                "confirmation_unique_observations": (
                    observation_losses["observation_id"].nunique()
                    if not observation_losses.empty
                    else 0
                ),
                "confirmation_partition_opened_once": True,
                "current_repo_commit": git_commit(repo_root),
            }
        ]
    )
    summary.to_csv(output_dir / "obs084c_confirmation_summary.csv", index=False)

    write_report(
        output_dir / "obs084c_confirmation_report.md",
        overall_status,
        freeze_payload,
        candidate_payload,
        confirmation_manifest_id,
        input_manifest,
        candidate_validation,
        threshold_validation,
        observation_losses,
        outcomes,
        witnesses,
        failures_df,
        lock_path,
    )
    complete_opening_lock(
        lock_path,
        lock_payload,
        confirmation_manifest_id,
        overall_status,
    )

    print(f"OBS-084c confirmation complete: {overall_status}")
    print(f"Sealed candidates evaluated: {len(outcomes)}/{len(candidates)}")
    print(f"FL3 direct witnesses: {len(witnesses)}")
    print(f"Confirmation manifest ID: {confirmation_manifest_id}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
