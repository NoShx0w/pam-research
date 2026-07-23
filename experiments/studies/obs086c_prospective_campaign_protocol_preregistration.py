#!/usr/bin/env python3
"""
obs086c_prospective_campaign_protocol_preregistration.py

OBS-086c — Prospective Campaign Protocol Preregistration
========================================================

Purpose
-------
Convert the five frozen OBS-086b protocol-selected operational families into
complete, auditable, evidence-independent prospective campaign protocol
profiles.  OBS-086c freezes execution semantics before any new scientific
observation is acquired, assigned, inspected, or evaluated.

The study is artifact-only.  It performs no new simulation, no campaign
activation, no roster assignment, no random-seed reveal, no candidate search,
no threshold fitting, no gate modification, and no observed-evidence
inspection.

Core result
-----------
OBS-086c creates:

* exactly five executable-but-not-activated protocol profiles;
* exactly one nonactivatable low-coverage held reference;
* zero globally selected campaigns;
* a frozen activation contract;
* a commit–reveal partition-assignment contract;
* outcome-blind admissibility, exclusion, replacement, monitoring, and
  structural futility rules;
* a one-time confirmation-opening contract;
* a gate and claim-entitlement ceiling; and
* a manifest binding every generated protocol artifact to the frozen
  OBS-086b lineage.

Scientific boundary
-------------------
A preregistered protocol profile is not an activated campaign, not observed
evidence, not a passage guarantee, and not an increase in claim entitlement.
The two ``three_way__no_window`` profiles remain FL3-entitlement-capped.  The
three ``C_vs_Cp3__path_shares_only`` profiles retain FL3 eligibility only if all
frozen gates later pass on properly separated discovery and confirmation
partitions.

Canonical run
-------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086c_prospective_campaign_protocol_preregistration.py \\
  --overwrite

Validation only
---------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086c_prospective_campaign_protocol_preregistration.py \\
  --validate-only

Self-test
---------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086c_prospective_campaign_protocol_preregistration.py \\
  --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "obs086c_prospective_campaign_protocol_preregistration_v1"
STATE_COMPLETED = "prospective_campaign_protocol_preregistration_completed"

DEFAULT_EXPECTED_OBS086B_MANIFEST_ID = (
    "d32f61955a41abf1ecd036863a766fbf0617fb2a0bf0a34b691487f9bed94119"
)
DEFAULT_EXPECTED_OBS086B_SCRIPT_SHA256 = (
    "0f9b4704f0ad3499f713bc0b0ce05c13ac4d37aea21776f45ad1ac26b85b6147"
)

DEFAULT_OBS086B_DIR = Path(
    "outputs/rig_registry/obs086_campaign_design/"
    "obs086b_robust_campaign_family_selection"
)
DEFAULT_OBS086B_SCRIPT = Path(
    "experiments/studies/obs086b_robust_campaign_family_selection.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs086_campaign_design/"
    "obs086c_prospective_campaign_protocol_preregistration"
)

DEFAULT_PROTOCOL_ANCHORS = (
    Path("experiments/studies/obs084b_direct_failure_support_discovery.py"),
    Path("experiments/studies/obs084c_direct_failure_support_confirmation.py"),
    Path("docs/05_project/084_rig_direct_failure_support_witness_protocol.md"),
)

CANONICAL_CLUSTER_GRID = (3, 4, 5, 6, 8, 10, 12)
CANONICAL_RELIABILITY_TARGETS = (0.50, 0.80, 0.90)
EXPECTED_OBS086B_OUTPUT_ARTIFACTS = 13
EXPECTED_OPERATIONAL_FAMILIES = 24
EXPECTED_PARETO_FAMILIES = 24
EXPECTED_SELECTED_PROFILES = 5
EXPECTED_HELD_REFERENCES = 1
EXPECTED_REJECTED_FAMILIES = 0
EXPECTED_OBS086C_OUTPUT_ARTIFACTS = 19  # excludes obs086c_manifest.json
MINIMUM_EFFECTIVE_CLUSTERS_FOR_EXACT_GATE_AT_ALPHA_0_10 = 4

SCIENTIFIC_OBSERVATION_KEY = (
    "case",
    "object",
    "cohort",
    "scale_index_from",
    "scale_index_to",
)

CANONICAL_SELECTED_SIGNATURES = {
    ("three_way__no_window", 0.50, 8, 10, "fl3_entitlement_capped"),
    ("three_way__no_window", 0.80, 8, 12, "fl3_entitlement_capped"),
    ("C_vs_Cp3__path_shares_only", 0.50, 12, 12, "fl3_entitled"),
    ("C_vs_Cp3__path_shares_only", 0.80, 10, 10, "fl3_entitled"),
    ("C_vs_Cp3__path_shares_only", 0.90, 12, 12, "fl3_entitled"),
}
CANONICAL_HELD_SIGNATURES = {
    ("three_way__no_window", 0.90, 8, 12, "fl3_entitlement_capped"),
}

SELECTED_REQUIRED_COLUMNS = {
    "operational_family_id",
    "address_id",
    "record_id",
    "support_id",
    "relation",
    "carrier",
    "entitlement_status",
    "reliability_target",
    "discovery_nominal_k",
    "confirmation_nominal_k",
    "total_nominal_objects",
    "partition_allocation_imbalance",
    "defensible_target_reaching_scenario_cells",
    "defensible_stress_test_coverage",
    "origin_minimum_support_efficiency",
    "origin_minimum_mean_effective_clusters",
    "protocol_selection_status",
    "protocol_selection_reason",
}

HELD_REQUIRED_COLUMNS = SELECTED_REQUIRED_COLUMNS.copy()

FAMILY_SUMMARY_REQUIRED_COLUMNS = {
    *SELECTED_REQUIRED_COLUMNS,
    "pareto_nondominated",
    "pareto_status",
}

PROTOCOL_FREEZE_REQUIRED_COLUMNS = {
    "operational_family_id",
    "record_id",
    "carrier",
    "entitlement_status",
    "reliability_target",
    "discovery_nominal_independent_objects",
    "confirmation_nominal_independent_objects",
    "total_nominal_independent_objects",
    "protocol_freeze_status",
}

ENTITLEMENT_REQUIRED_COLUMNS = {
    "entitlement_status",
    "reliability_target",
    "pareto_status",
    "protocol_selection_status",
    "entitlement_preserved",
}

ENTITLEMENT_RANK = {
    "fl3_entitlement_capped": 0,
    "fl3_entitled": 1,
}


@dataclass(frozen=True)
class StudyFailure:
    stage: str
    scope_id: str
    reason: str
    detail: str = ""
    severity: str = "error"


# -----------------------------------------------------------------------------
# CLI and generic helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OBS-086c: deterministic prospective campaign protocol "
            "preregistration from frozen OBS-086b artifacts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--obs086b-dir",
        type=Path,
        default=DEFAULT_OBS086B_DIR,
        help="Frozen OBS-086b output directory.",
    )
    parser.add_argument(
        "--obs086b-script",
        type=Path,
        default=DEFAULT_OBS086B_SCRIPT,
        help="Frozen OBS-086b study script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="OBS-086c output directory.",
    )
    parser.add_argument(
        "--expected-obs086b-manifest-id",
        default=DEFAULT_EXPECTED_OBS086B_MANIFEST_ID,
        help="Required frozen OBS-086b manifest identity.",
    )
    parser.add_argument(
        "--expected-obs086b-script-sha256",
        default=DEFAULT_EXPECTED_OBS086B_SCRIPT_SHA256,
        help="Required frozen OBS-086b script SHA256.",
    )
    parser.add_argument(
        "--expected-obs086b-commit",
        default="",
        help=(
            "Optional explicit OBS-086b freeze commit. When omitted, resolve "
            "the newest ancestor commit containing the exact frozen script and "
            "manifest bytes."
        ),
    )
    parser.add_argument(
        "--protocol-anchor",
        action="append",
        type=Path,
        default=None,
        help=(
            "Semantic protocol anchor file. May be repeated. Defaults to the "
            "canonical OBS-084 discovery, confirmation, and protocol files."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen lineage and inputs, then exit without writing outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic regression tests and exit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [normalize(val) for val in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            value_float = float(item)
            if not math.isfinite(value_float):
                return None
            return value_float
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, Path):
            return item.as_posix()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_under_root(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot normalize boolean value: {value!r}")


def stable_row_id(prefix: str, payload: Mapping[str, Any], length: int = 24) -> str:
    digest = sha256_bytes(canonical_json(dict(payload)).encode("utf-8"))
    return f"{prefix}-{digest[:length]}"


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def safe_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite, found {value!r}.")
    return result


def safe_int(value: Any, *, name: str) -> int:
    number = safe_float(value, name=name)
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9:
        raise RuntimeError(f"{name} must be integer-valued, found {value!r}.")
    return rounded


def unique_text(values: Iterable[Any]) -> str:
    cleaned = sorted({str(value) for value in values if not pd.isna(value)})
    if not cleaned:
        return ""
    return cleaned[0] if len(cleaned) == 1 else "|".join(cleaned)


def markdown_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in view.iterrows():
        rendered: list[str] = []
        for value in row.tolist():
            if pd.isna(value):
                text = ""
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.6g}"
            else:
                text = str(value)
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    if len(frame) > max_rows:
        lines.extend(["", f"_Showing {max_rows:,} of {len(frame):,} rows._"])
    return "\n".join(lines)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def failures_frame(failures: Sequence[StudyFailure]) -> pd.DataFrame:
    columns = ["stage", "scope_id", "reason", "detail", "severity"]
    return pd.DataFrame([failure.__dict__ for failure in failures], columns=columns)


# -----------------------------------------------------------------------------
# Git and frozen-lineage validation
# -----------------------------------------------------------------------------


def run_git(
    repo_root: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def git_head(repo_root: Path) -> str:
    return run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()


def require_commit_ancestor(repo_root: Path, commit: str) -> None:
    result = run_git(
        repo_root,
        ["merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Required frozen commit is not an ancestor of HEAD: {commit}"
        )


def git_blob_bytes(repo_root: Path, commit: str, relative_path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def resolve_obs086b_freeze_commit(
    repo_root: Path,
    obs086b_script: Path,
    obs086b_manifest: Path,
    expected_script_sha256: str,
    expected_manifest_sha256: str,
    explicit_commit: str,
) -> str:
    script_rel = repo_relative(obs086b_script, repo_root)
    manifest_rel = repo_relative(obs086b_manifest, repo_root)

    if explicit_commit:
        require_commit_ancestor(repo_root, explicit_commit)
        candidates = [explicit_commit]
    else:
        result = run_git(
            repo_root,
            ["log", "--format=%H", "--all", "--", script_rel, manifest_rel],
        )
        candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    for commit in candidates:
        try:
            require_commit_ancestor(repo_root, commit)
        except RuntimeError:
            continue
        script_bytes = git_blob_bytes(repo_root, commit, script_rel)
        manifest_bytes = git_blob_bytes(repo_root, commit, manifest_rel)
        if script_bytes is None or manifest_bytes is None:
            continue
        if sha256_bytes(script_bytes) != expected_script_sha256:
            continue
        if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
            continue
        return commit

    if explicit_commit:
        raise RuntimeError(
            "Explicit OBS-086b commit does not contain the exact frozen script "
            "and manifest bytes."
        )
    raise RuntimeError(
        "Could not resolve an ancestor commit containing the exact frozen "
        "OBS-086b script and manifest. Commit OBS-086b before running OBS-086c, "
        "or provide --expected-obs086b-commit."
    )


# -----------------------------------------------------------------------------
# Frozen OBS-086b inputs
# -----------------------------------------------------------------------------


def obs086b_paths(obs086b_dir: Path) -> dict[str, Path]:
    return {
        "input_manifest": obs086b_dir / "obs086b_input_manifest.csv",
        "scenario_evaluation": (
            obs086b_dir / "obs086b_scenario_allocation_evaluation.csv"
        ),
        "family_summary": obs086b_dir / "obs086b_operational_family_summary.csv",
        "dominance": obs086b_dir / "obs086b_family_dominance_table.csv",
        "pareto_frontier": obs086b_dir / "obs086b_pareto_frontier.csv",
        "selected_families": (
            obs086b_dir / "obs086b_selected_campaign_families.csv"
        ),
        "held_families": obs086b_dir / "obs086b_held_campaign_families.csv",
        "rejected_families": (
            obs086b_dir / "obs086b_rejected_campaign_families.csv"
        ),
        "address_recommendations": (
            obs086b_dir / "obs086b_address_recommendations.csv"
        ),
        "protocol_freeze": obs086b_dir / "obs086b_protocol_freeze_table.csv",
        "entitlement_overlay": obs086b_dir / "obs086b_entitlement_overlay.csv",
        "failures": obs086b_dir / "obs086b_failures.csv",
        "report": obs086b_dir / "obs086b_report.md",
        "manifest": obs086b_dir / "obs086b_manifest.json",
    }


def validate_obs086b_manifest_core(
    manifest: Mapping[str, Any],
    expected_manifest_id: str,
) -> None:
    if manifest.get("obs086b_manifest_id") != expected_manifest_id:
        raise RuntimeError(
            "OBS-086b manifest identity mismatch: expected "
            f"{expected_manifest_id}, found {manifest.get('obs086b_manifest_id')}."
        )
    if manifest.get("state") != "robust_campaign_family_selection_completed":
        raise RuntimeError(
            f"OBS-086b state is not completed: {manifest.get('state')!r}"
        )
    if manifest.get("schema_version") != (
        "obs086b_robust_campaign_family_selection_v1_0_1"
    ):
        raise RuntimeError(
            f"Unexpected OBS-086b schema: {manifest.get('schema_version')!r}"
        )
    execution = manifest.get("execution", {})
    expected_counts = {
        "operational_family_rows": EXPECTED_OPERATIONAL_FAMILIES,
        "pareto_family_rows": EXPECTED_PARETO_FAMILIES,
        "protocol_selected_family_rows": EXPECTED_SELECTED_PROFILES,
        "protocol_held_family_rows": EXPECTED_HELD_REFERENCES,
        "rejected_family_rows": EXPECTED_REJECTED_FAMILIES,
        "failures": 0,
    }
    for field, expected in expected_counts.items():
        actual = int(execution.get(field, -1))
        if actual != expected:
            raise RuntimeError(
                f"OBS-086b execution count mismatch for {field}: "
                f"expected {expected}, found {actual}."
            )
    selection_contract = manifest.get("selection_contract", {})
    grid = tuple(int(value) for value in selection_contract.get("cluster_grid", []))
    if grid != CANONICAL_CLUSTER_GRID:
        raise RuntimeError("OBS-086b cluster grid differs from the canonical grid.")
    targets = tuple(
        float(value) for value in selection_contract.get("reliability_targets", [])
    )
    if targets != CANONICAL_RELIABILITY_TARGETS:
        raise RuntimeError("OBS-086b reliability targets are noncanonical.")


def validate_declared_obs086b_artifacts(
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> pd.DataFrame:
    declared = manifest.get("output_artifacts")
    if not isinstance(declared, list):
        raise RuntimeError("OBS-086b manifest has no output_artifacts list.")
    if len(declared) != EXPECTED_OBS086B_OUTPUT_ARTIFACTS:
        raise RuntimeError(
            "Unexpected OBS-086b artifact count: expected "
            f"{EXPECTED_OBS086B_OUTPUT_ARTIFACTS}, found {len(declared)}."
        )

    rows: list[dict[str, Any]] = []
    for item in declared:
        relative_path = str(item["artifact_path"])
        path = resolve_under_root(Path(relative_path), repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen OBS-086b artifact missing: {path}")
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        expected_size = int(item["size_bytes"])
        expected_sha = str(item["sha256"])
        if actual_size != expected_size:
            raise RuntimeError(
                f"Frozen artifact size mismatch for {relative_path}: expected "
                f"{expected_size}, found {actual_size}."
            )
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Frozen artifact hash mismatch for {relative_path}: expected "
                f"{expected_sha}, found {actual_sha}."
            )
        rows.append(
            {
                "input_role": "frozen_obs086b_output",
                "artifact_path": relative_path,
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "validation_status": "validated",
                "outcome_bearing_data_inspected": False,
            }
        )
    return pd.DataFrame(rows)


def selected_signature(frame: pd.DataFrame) -> set[tuple[str, float, int, int, str]]:
    return {
        (
            str(row["record_id"]),
            round(float(row["reliability_target"]), 2),
            safe_int(row["discovery_nominal_k"], name="discovery_nominal_k"),
            safe_int(
                row["confirmation_nominal_k"], name="confirmation_nominal_k"
            ),
            str(row["entitlement_status"]),
        )
        for _, row in frame.iterrows()
    }


def validate_selected_and_held_frames(
    selected: pd.DataFrame,
    held: pd.DataFrame,
    family_summary: pd.DataFrame,
    protocol_freeze: pd.DataFrame,
    entitlement_overlay: pd.DataFrame,
) -> None:
    require_columns(selected, SELECTED_REQUIRED_COLUMNS, "selected families")
    require_columns(held, HELD_REQUIRED_COLUMNS, "held families")
    require_columns(
        family_summary, FAMILY_SUMMARY_REQUIRED_COLUMNS, "operational family summary"
    )
    require_columns(protocol_freeze, PROTOCOL_FREEZE_REQUIRED_COLUMNS, "protocol freeze")
    require_columns(
        entitlement_overlay, ENTITLEMENT_REQUIRED_COLUMNS, "entitlement overlay"
    )

    if len(selected) != EXPECTED_SELECTED_PROFILES:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTED_PROFILES} selected families, found {len(selected)}."
        )
    if len(held) != EXPECTED_HELD_REFERENCES:
        raise RuntimeError(
            f"Expected {EXPECTED_HELD_REFERENCES} held family, found {len(held)}."
        )
    if len(family_summary) != EXPECTED_OPERATIONAL_FAMILIES:
        raise RuntimeError(
            f"Expected {EXPECTED_OPERATIONAL_FAMILIES} operational families, "
            f"found {len(family_summary)}."
        )
    for name, frame in [
        ("selected", selected),
        ("held", held),
        ("family summary", family_summary),
    ]:
        if frame["operational_family_id"].astype(str).duplicated().any():
            raise RuntimeError(f"{name} has duplicate operational_family_id values.")

    selected_ids = set(selected["operational_family_id"].astype(str))
    held_ids = set(held["operational_family_id"].astype(str))
    family_ids = set(family_summary["operational_family_id"].astype(str))
    if selected_ids & held_ids:
        raise RuntimeError("Selected and held family IDs overlap.")
    if not selected_ids.issubset(family_ids):
        raise RuntimeError("Selected family set is not a subset of the family summary.")
    if not held_ids.issubset(family_ids):
        raise RuntimeError("Held family set is not a subset of the family summary.")

    if selected_signature(selected) != CANONICAL_SELECTED_SIGNATURES:
        raise RuntimeError(
            "Selected family signatures differ from the frozen canonical OBS-086b set."
        )
    if selected_signature(held) != CANONICAL_HELD_SIGNATURES:
        raise RuntimeError(
            "Held family signature differs from the frozen canonical OBS-086b hold."
        )

    if not selected["protocol_selection_status"].astype(str).eq(
        "protocol_selected_for_preregistration_review"
    ).all():
        raise RuntimeError("Selected families contain an unexpected protocol status.")
    if not held["protocol_selection_status"].astype(str).eq(
        "protocol_hold_low_coverage"
    ).all():
        raise RuntimeError("Held family contains an unexpected protocol status.")

    for frame_name, frame in [("selected", selected), ("held", held)]:
        for _, row in frame.iterrows():
            d_k = safe_int(row["discovery_nominal_k"], name="discovery_nominal_k")
            c_k = safe_int(
                row["confirmation_nominal_k"], name="confirmation_nominal_k"
            )
            if d_k not in CANONICAL_CLUSTER_GRID or c_k not in CANONICAL_CLUSTER_GRID:
                raise RuntimeError(
                    f"{frame_name} family uses allocation outside the tested k grid."
                )
            total = safe_int(row["total_nominal_objects"], name="total_nominal_objects")
            if total != d_k + c_k:
                raise RuntimeError(
                    f"{frame_name} family total nominal objects does not equal d_k+c_k."
                )
            coverage = safe_float(
                row["defensible_stress_test_coverage"],
                name="defensible_stress_test_coverage",
            )
            if coverage < 0 or coverage > 1:
                raise RuntimeError("Stress-test coverage must lie in [0, 1].")
            efficiency = safe_float(
                row["origin_minimum_support_efficiency"],
                name="origin_minimum_support_efficiency",
            )
            if not (0 < efficiency <= 1):
                raise RuntimeError(
                    "Origin minimum support efficiency must lie in (0, 1]."
                )

    selected_merge = selected.merge(
        family_summary,
        on="operational_family_id",
        how="left",
        suffixes=("_selected", "_family"),
        validate="one_to_one",
    )
    if selected_merge.filter(regex="_family$").isna().all(axis=1).any():
        raise RuntimeError("Selected family failed to resolve in family summary.")

    freeze_ids = set(protocol_freeze["operational_family_id"].astype(str))
    if freeze_ids != selected_ids:
        raise RuntimeError(
            "OBS-086b protocol-freeze table does not exactly match selected families."
        )
    selected_lookup = selected.set_index("operational_family_id")
    freeze_lookup = protocol_freeze.set_index("operational_family_id")
    if not selected_lookup.index.is_unique or not freeze_lookup.index.is_unique:
        raise RuntimeError("Selected or protocol-freeze family IDs are not unique.")
    for family_id in sorted(selected_ids):
        source = selected_lookup.loc[family_id]
        frozen = freeze_lookup.loc[family_id]
        comparisons = [
            (
                safe_int(source["discovery_nominal_k"], name="discovery_nominal_k"),
                safe_int(
                    frozen["discovery_nominal_independent_objects"],
                    name="discovery_nominal_independent_objects",
                ),
                "discovery allocation",
            ),
            (
                safe_int(source["confirmation_nominal_k"], name="confirmation_nominal_k"),
                safe_int(
                    frozen["confirmation_nominal_independent_objects"],
                    name="confirmation_nominal_independent_objects",
                ),
                "confirmation allocation",
            ),
            (
                safe_int(source["total_nominal_objects"], name="total_nominal_objects"),
                safe_int(
                    frozen["total_nominal_independent_objects"],
                    name="total_nominal_independent_objects",
                ),
                "total allocation",
            ),
        ]
        for expected, observed, label in comparisons:
            if expected != observed:
                raise RuntimeError(
                    f"OBS-086b protocol-freeze {label} mismatch for {family_id}."
                )
        if str(source["entitlement_status"]) != str(frozen["entitlement_status"]):
            raise RuntimeError(
                f"OBS-086b protocol-freeze entitlement mismatch for {family_id}."
            )

    if not entitlement_overlay["entitlement_preserved"].map(normalize_bool).all():
        raise RuntimeError("OBS-086b entitlement overlay does not preserve entitlement.")
    overlay_statuses = set(entitlement_overlay["entitlement_status"].astype(str))
    required_statuses = {"fl3_entitled", "fl3_entitlement_capped"}
    if not required_statuses.issubset(overlay_statuses):
        raise RuntimeError(
            "OBS-086b entitlement overlay does not contain both canonical entitlement statuses."
        )


def validate_protocol_anchors(
    repo_root: Path,
    anchors: Sequence[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        path = resolve_under_root(anchor, repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Required protocol semantic anchor missing: {path}")
        rows.append(
            {
                "input_role": "protocol_semantic_anchor_no_outcomes_read",
                "artifact_path": repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "validation_status": "hashed_and_frozen_as_semantic_anchor",
                "outcome_bearing_data_inspected": False,
            }
        )
    return pd.DataFrame(rows)


def validate_frozen_inputs(
    *,
    repo_root: Path,
    obs086b_dir: Path,
    obs086b_script: Path,
    expected_manifest_id: str,
    expected_script_sha256: str,
    explicit_commit: str,
    protocol_anchors: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    paths = obs086b_paths(obs086b_dir)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required OBS-086b {name} file is missing: {path}")
    if not obs086b_script.is_file():
        raise FileNotFoundError(f"Frozen OBS-086b script is missing: {obs086b_script}")

    actual_script_sha = sha256_file(obs086b_script)
    if actual_script_sha != expected_script_sha256:
        raise RuntimeError(
            "OBS-086b script hash mismatch: expected "
            f"{expected_script_sha256}, found {actual_script_sha}."
        )

    manifest_bytes = paths["manifest"].read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_obs086b_manifest_core(manifest, expected_manifest_id)
    declared_inventory = validate_declared_obs086b_artifacts(manifest, repo_root)

    freeze_commit = resolve_obs086b_freeze_commit(
        repo_root=repo_root,
        obs086b_script=obs086b_script,
        obs086b_manifest=paths["manifest"],
        expected_script_sha256=expected_script_sha256,
        expected_manifest_sha256=manifest_sha,
        explicit_commit=explicit_commit,
    )

    frames = {
        "family_summary": pd.read_csv(paths["family_summary"]),
        "selected_families": pd.read_csv(paths["selected_families"]),
        "held_families": pd.read_csv(paths["held_families"]),
        "protocol_freeze": pd.read_csv(paths["protocol_freeze"]),
        "entitlement_overlay": pd.read_csv(paths["entitlement_overlay"]),
        "failures": pd.read_csv(paths["failures"]),
    }
    if not frames["failures"].empty:
        raise RuntimeError("Frozen OBS-086b failures table is not empty.")
    validate_selected_and_held_frames(
        frames["selected_families"],
        frames["held_families"],
        frames["family_summary"],
        frames["protocol_freeze"],
        frames["entitlement_overlay"],
    )

    anchor_inventory = validate_protocol_anchors(repo_root, protocol_anchors)
    direct_rows = pd.DataFrame(
        [
            {
                "input_role": "frozen_obs086b_script",
                "artifact_path": repo_relative(obs086b_script, repo_root),
                "size_bytes": obs086b_script.stat().st_size,
                "sha256": actual_script_sha,
                "validation_status": "validated",
                "outcome_bearing_data_inspected": False,
            },
            {
                "input_role": "frozen_obs086b_manifest",
                "artifact_path": repo_relative(paths["manifest"], repo_root),
                "size_bytes": paths["manifest"].stat().st_size,
                "sha256": manifest_sha,
                "validation_status": "validated",
                "outcome_bearing_data_inspected": False,
            },
        ]
    )
    input_inventory = pd.concat(
        [direct_rows, declared_inventory, anchor_inventory],
        ignore_index=True,
    ).sort_values(["input_role", "artifact_path"]).reset_index(drop=True)

    lineage = {
        "obs086b_commit": freeze_commit,
        "obs086b_manifest_id": expected_manifest_id,
        "obs086b_manifest_sha256": manifest_sha,
        "obs086b_script_sha256": actual_script_sha,
        "obs086b_output_artifacts_validated": len(declared_inventory),
        "obs086b_state": manifest["state"],
        "obs086a_manifest_id": manifest.get("frozen_lineage", {}).get(
            "obs086a_manifest_id", ""
        ),
        "obs085d_manifest_id": manifest.get("frozen_lineage", {}).get(
            "obs085d_manifest_id", ""
        ),
        "protocol_semantic_anchor_count": len(anchor_inventory),
        "protocol_semantic_anchor_set_sha256": sha256_bytes(
            canonical_json(
                anchor_inventory[["artifact_path", "sha256"]].to_dict("records")
            ).encode("utf-8")
        ),
        "current_repo_head": git_head(repo_root),
    }
    return manifest, frames, input_inventory, lineage


# -----------------------------------------------------------------------------
# Protocol synthesis
# -----------------------------------------------------------------------------


def profile_id_for_row(row: Mapping[str, Any], obs086b_manifest_id: str) -> str:
    payload = {
        "operational_family_id": str(row["operational_family_id"]),
        "record_id": str(row["record_id"]),
        "carrier": str(row["carrier"]),
        "reliability_target": round(float(row["reliability_target"]), 12),
        "discovery_nominal_k": safe_int(
            row["discovery_nominal_k"], name="discovery_nominal_k"
        ),
        "confirmation_nominal_k": safe_int(
            row["confirmation_nominal_k"], name="confirmation_nominal_k"
        ),
        "entitlement_status": str(row["entitlement_status"]),
        "obs086b_manifest_id": obs086b_manifest_id,
    }
    return stable_row_id("PR", payload)


def build_protocol_profile_registry(
    selected: pd.DataFrame,
    obs086b_manifest_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in selected.sort_values(
        ["record_id", "reliability_target", "operational_family_id"]
    ).iterrows():
        d_k = safe_int(source["discovery_nominal_k"], name="discovery_nominal_k")
        c_k = safe_int(
            source["confirmation_nominal_k"], name="confirmation_nominal_k"
        )
        entitlement = str(source["entitlement_status"])
        profile_id = profile_id_for_row(source, obs086b_manifest_id)
        rows.append(
            {
                "protocol_profile_id": profile_id,
                "source_operational_family_id": source["operational_family_id"],
                "address_id": source["address_id"],
                "record_id": source["record_id"],
                "support_id": source["support_id"],
                "relation": source["relation"],
                "carrier": source["carrier"],
                "entitlement_status": entitlement,
                "maximum_claim_entitlement": (
                    "fl3_artifact_direct_witness_if_all_frozen_gates_pass"
                    if entitlement == "fl3_entitled"
                    else "fl2_localized_support_only_fl3_prohibited"
                ),
                "fl3_activation_compatible": entitlement == "fl3_entitled",
                "reliability_target": float(source["reliability_target"]),
                "discovery_nominal_k": d_k,
                "confirmation_nominal_k": c_k,
                "total_nominal_objects": d_k + c_k,
                "partition_allocation_imbalance": safe_int(
                    source["partition_allocation_imbalance"],
                    name="partition_allocation_imbalance",
                ),
                "defensible_target_reaching_scenario_cells": safe_int(
                    source["defensible_target_reaching_scenario_cells"],
                    name="defensible_target_reaching_scenario_cells",
                ),
                "defensible_stress_test_coverage": float(
                    source["defensible_stress_test_coverage"]
                ),
                "origin_minimum_support_efficiency": float(
                    source["origin_minimum_support_efficiency"]
                ),
                "origin_minimum_mean_effective_clusters": float(
                    source["origin_minimum_mean_effective_clusters"]
                ),
                "protocol_profile_status": "executable_pending_evidence_independent_activation",
                "activation_status": "not_activated",
                "global_campaign_selection_status": "no_global_winner_selected_in_obs086c",
                "new_simulation_performed": False,
                "observed_evidence_inspected": False,
                "assignment_performed": False,
                "confirmation_opened": False,
                "claim_entitlement_preserved": True,
                "source_obs086b_manifest_id": obs086b_manifest_id,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_SELECTED_PROFILES:
        raise RuntimeError("Protocol profile registry does not contain exactly five rows.")
    if frame["protocol_profile_id"].duplicated().any():
        raise RuntimeError("Protocol profile IDs are not unique.")
    return frame.reset_index(drop=True)


def build_activation_contract(profiles: pd.DataFrame) -> pd.DataFrame:
    allowed_reasons = [
        "intended_claim_entitlement_is_compatible",
        "nominal_object_budget_is_sufficient",
        "required_carrier_is_available_under_frozen_schema",
        "prespecified_measurement_pipeline_is_available",
        "independent_cluster_roster_is_structurally_feasible",
        "operational_schedule_is_feasible",
        "reliability_target_was_selected_before_outcome_access",
    ]
    forbidden_reasons = [
        "observed_effect_direction",
        "observed_effect_magnitude",
        "preliminary_gate_passage",
        "preliminary_p_value_or_q_value",
        "discovery_candidate_rank_or_status",
        "confirmation_outcome",
        "favorable_delta_or_control_response_lambda_cell",
        "simulator_scenario_matching_after_roster_review",
        "post_hoc_budget_reallocation_based_on_results",
        "carrier_switching_after_outcome_access",
    ]
    rows: list[dict[str, Any]] = []
    for _, profile in profiles.iterrows():
        rows.append(
            {
                "protocol_profile_id": profile["protocol_profile_id"],
                "record_id": profile["record_id"],
                "carrier": profile["carrier"],
                "reliability_target": profile["reliability_target"],
                "entitlement_status": profile["entitlement_status"],
                "activation_eligibility": "eligible_for_future_activation_record",
                "activation_decision_owner": "future_preregistered_activation_record",
                "activation_must_precede": (
                    "eligible_roster_outcome_access; seed_reveal; scientific_evaluation"
                ),
                "allowed_activation_reasons_json": canonical_json(allowed_reasons),
                "forbidden_activation_reasons_json": canonical_json(forbidden_reasons),
                "required_nominal_object_budget": profile["total_nominal_objects"],
                "required_discovery_objects": profile["discovery_nominal_k"],
                "required_confirmation_objects": profile["confirmation_nominal_k"],
                "required_carrier": profile["carrier"],
                "maximum_claim_entitlement": profile["maximum_claim_entitlement"],
                "activation_priority_rule": (
                    "filter by entitlement compatibility, resource feasibility, carrier "
                    "availability, measurement availability, and structural independence; "
                    "apply any target preference fixed in the activation record; stable "
                    "protocol_profile_id tie-break only"
                ),
                "activation_status_at_obs086c": "not_activated",
                "global_winner_selected": False,
            }
        )
    return pd.DataFrame(rows)


def build_observation_identity_contract() -> pd.DataFrame:
    rows = [
        {
            "identity_level": "source_object",
            "canonical_identifier": "canonical_object_id",
            "definition": (
                "the provenance-stable source object entering eligibility screening"
            ),
            "cardinality_guardrail": "one canonical_object_id per source object",
            "partition_role": "assignable unit subject to cluster blocking",
            "prohibited_conflation": "not an analysis row and not an effective cluster",
        },
        {
            "identity_level": "scientific_observation",
            "canonical_identifier": "observation_key",
            "definition": (
                "the frozen OBS-084 scientific observation key over "
                + ", ".join(SCIENTIFIC_OBSERVATION_KEY)
            ),
            "cardinality_guardrail": "unique over the complete eligible observation table",
            "partition_role": "all rows for one scientific observation remain together",
            "prohibited_conflation": "not interchangeable with source object or carrier record",
        },
        {
            "identity_level": "carrier_record",
            "canonical_identifier": "record_id|carrier|support_id",
            "definition": (
                "the frozen relation/carrier/support contract under which evidence is evaluated"
            ),
            "cardinality_guardrail": "profile-fixed and immutable after activation",
            "partition_role": "same contract applied unchanged in both partitions",
            "prohibited_conflation": "not a source object and not a cluster",
        },
        {
            "identity_level": "effective_cluster",
            "canonical_identifier": "effective_cluster_id",
            "definition": (
                "the structurally independent cluster used for exact cluster-aware inference"
            ),
            "cardinality_guardrail": (
                "one effective_cluster_id may not appear in both discovery and confirmation"
            ),
            "partition_role": "independence block and effective-support accounting unit",
            "prohibited_conflation": "nominal object count may exceed effective-cluster count",
        },
        {
            "identity_level": "analysis_row",
            "canonical_identifier": "analysis_row_id",
            "definition": "one computed row entering a frozen statistic or model",
            "cardinality_guardrail": (
                "all analysis rows inherit source-object, observation, cluster, and partition IDs"
            ),
            "partition_role": "never independently randomized or migrated",
            "prohibited_conflation": "row count does not establish independent support",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["scientific_observation_key_json"] = canonical_json(
        list(SCIENTIFIC_OBSERVATION_KEY)
    )
    frame["contract_status"] = "frozen_before_activation"
    return frame


def build_partition_assignment_contract(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    algorithm = (
        "sha256_commit_reveal_cluster_blocked_object_assignment_v1: verify "
        "SHA256(seed_reveal)==seed_commitment; canonicalize and hash eligible roster; "
        "derive stable cluster and object ranks from seed_reveal, protocol_profile_id, "
        "roster_manifest_id, effective_cluster_id, and canonical_object_id; allocate "
        "exact discovery and confirmation nominal object quotas without placing one "
        "effective cluster in both partitions; assign remaining eligible objects only "
        "to role-specific reserve pools; fail closed when exact quotas and cluster "
        "separation cannot both be satisfied"
    )
    for _, profile in profiles.iterrows():
        rows.append(
            {
                "protocol_profile_id": profile["protocol_profile_id"],
                "assignment_algorithm_id": (
                    "sha256_commit_reveal_cluster_blocked_object_assignment_v1"
                ),
                "assignment_algorithm_specification": algorithm,
                "seed_commitment_field": "sha256_seed_commitment",
                "seed_commitment_timing": (
                    "recorded before the eligible roster identity is known"
                ),
                "seed_reveal_field": "seed_reveal_utf8",
                "seed_reveal_timing": (
                    "after eligible roster freeze and roster_manifest_id creation; "
                    "before assignment table generation"
                ),
                "seed_verification_rule": (
                    "lowercase_hex(SHA256(UTF8(seed_reveal_utf8))) equals "
                    "sha256_seed_commitment"
                ),
                "roster_manifest_rule": (
                    "canonical CSV bytes sorted by canonical_object_id and bound to "
                    "schema, provenance, eligibility, and effective_cluster_id"
                ),
                "roster_manifest_id_rule": "SHA256(canonical eligible roster bytes)",
                "assignment_hash_material": (
                    "seed_reveal_utf8|protocol_profile_id|roster_manifest_id|"
                    "effective_cluster_id|canonical_object_id|assignment_role"
                ),
                "discovery_nominal_object_quota": profile["discovery_nominal_k"],
                "confirmation_nominal_object_quota": profile["confirmation_nominal_k"],
                "partition_independence_rule": (
                    "no effective_cluster_id and no canonical_object_id may occur in both partitions"
                ),
                "nested_row_rule": (
                    "all rows and observations belonging to an assigned source object remain in its partition"
                ),
                "post_freeze_migration": "prohibited",
                "assignment_output_required": "complete_hashed_assignment_table_before_outcome_access",
                "assignment_status_at_obs086c": "algorithm_frozen_no_roster_assigned",
            }
        )
    return pd.DataFrame(rows)


def build_admissibility_rules() -> pd.DataFrame:
    definitions = [
        (
            "AD-001",
            "source_object_identity_complete",
            "canonical_object_id is present, parseable, and provenance-stable",
            "EX-001_missing_or_invalid_object_identity",
        ),
        (
            "AD-002",
            "scientific_observation_key_complete",
            "all frozen scientific-observation key fields are present",
            "EX-002_incomplete_scientific_observation_key",
        ),
        (
            "AD-003",
            "scientific_observation_key_unique",
            "observation_key is unique at the frozen scientific-observation grain",
            "EX-003_duplicate_scientific_observation_identity",
        ),
        (
            "AD-004",
            "effective_cluster_identity_resolved",
            "effective_cluster_id is present and structurally interpretable",
            "EX-004_unresolvable_effective_cluster_identity",
        ),
        (
            "AD-005",
            "required_carrier_complete",
            "all profile-required carrier fields are present and finite where required",
            "EX-005_missing_required_carrier",
        ),
        (
            "AD-006",
            "prespecified_measurement_available",
            "the frozen measurement can be computed without threshold or definition changes",
            "EX-006_failed_prespecified_measurement",
        ),
        (
            "AD-007",
            "required_control_available",
            "the frozen control record and fields are available under the profile contract",
            "EX-007_missing_required_control",
        ),
        (
            "AD-008",
            "provenance_integrity_verified",
            "source paths, hashes, lineage identifiers, and transformation provenance verify",
            "EX-008_invalid_provenance",
        ),
        (
            "AD-009",
            "schema_valid",
            "all required fields, types, and finite-value constraints pass the frozen schema",
            "EX-009_schema_invalid",
        ),
        (
            "AD-010",
            "partition_assignment_valid",
            "assignment hash, quotas, role labels, and commit–reveal verification pass",
            "EX-010_partition_assignment_invalid",
        ),
        (
            "AD-011",
            "cross_partition_cluster_separation",
            "no effective_cluster_id or canonical_object_id crosses partitions",
            "EX-011_cross_partition_cluster_or_object_overlap",
        ),
        (
            "AD-012",
            "outcome_blind_admissibility",
            "admissibility was decided without effect, direction, p-value, gate, or candidate status",
            "EX-012_outcome_access_before_admissibility_freeze",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "rule_order": index + 1,
                "admissibility_rule_id": rule_id,
                "condition_id": condition,
                "frozen_predicate": predicate,
                "failure_exclusion_code": exclusion,
                "evaluation_stage": "pre_analysis_before_discovery_outcomes",
                "outcome_dependency_allowed": False,
                "rule_status": "frozen",
            }
            for index, (rule_id, condition, predicate, exclusion) in enumerate(definitions)
        ]
    )


def build_exclusion_reason_codes() -> pd.DataFrame:
    allowed = [
        ("EX-001_missing_or_invalid_object_identity", "identity", True),
        ("EX-002_incomplete_scientific_observation_key", "identity", True),
        ("EX-003_duplicate_scientific_observation_identity", "identity", True),
        ("EX-004_unresolvable_effective_cluster_identity", "independence", True),
        ("EX-005_missing_required_carrier", "carrier", True),
        ("EX-006_failed_prespecified_measurement", "measurement", True),
        ("EX-007_missing_required_control", "control", True),
        ("EX-008_invalid_provenance", "provenance", True),
        ("EX-009_schema_invalid", "schema", True),
        ("EX-010_partition_assignment_invalid", "assignment", False),
        ("EX-011_cross_partition_cluster_or_object_overlap", "independence", False),
        ("EX-012_outcome_access_before_admissibility_freeze", "protocol_deviation", False),
    ]
    forbidden = [
        "unfavorable_effect_direction",
        "small_effect_magnitude",
        "failed_gate",
        "large_p_value_or_q_value",
        "weak_candidate_contribution",
        "undesirable_simulator_correspondence",
        "discovery_rank_not_preferred",
        "confirmation_result_unfavorable",
    ]
    rows: list[dict[str, Any]] = []
    for code, category, replacement_allowed in allowed:
        rows.append(
            {
                "exclusion_reason_code": code,
                "reason_category": category,
                "code_status": "allowed_prespecified_structural_exclusion",
                "replacement_eligible": replacement_allowed,
                "outcome_dependent": False,
                "usage_rule": "record before any outcome-bearing evaluation",
            }
        )
    for index, reason in enumerate(forbidden, start=1):
        rows.append(
            {
                "exclusion_reason_code": f"FORBID-{index:03d}_{reason}",
                "reason_category": "outcome_dependent",
                "code_status": "prohibited_exclusion_basis",
                "replacement_eligible": False,
                "outcome_dependent": True,
                "usage_rule": "never valid as an exclusion or replacement trigger",
            }
        )
    return pd.DataFrame(rows)


def build_replacement_rules() -> pd.DataFrame:
    rules = [
        (
            1,
            "replacement_source",
            "replacement may use only the preassigned reserve pool for the same protocol profile and partition role",
            "allow_with_constraints",
        ),
        (
            2,
            "replacement_trigger",
            "replacement may occur only after an allowed prespecified structural exclusion code",
            "allow_with_constraints",
        ),
        (
            3,
            "partition_preservation",
            "a discovery object may be replaced only by a discovery-reserve object and confirmation only by confirmation reserve",
            "required",
        ),
        (
            4,
            "cluster_separation",
            "replacement may not introduce an effective cluster used by the opposite partition",
            "required",
        ),
        (
            5,
            "allocation_preservation",
            "replacement restores but never enlarges the frozen nominal analysis quota",
            "required",
        ),
        (
            6,
            "selection_order",
            "choose the next reserve object by the frozen assignment rank without discretion",
            "required",
        ),
        (
            7,
            "outcome_blindness",
            "effect direction, magnitude, gate result, p-value, candidate passage, and simulator resemblance are prohibited replacement inputs",
            "required",
        ),
        (
            8,
            "reserve_exhaustion",
            "when no valid same-role reserve remains, apply the structural continue-or-futility rule; do not migrate objects",
            "required",
        ),
        (
            9,
            "audit_record",
            "write original object, exclusion code, reserve object, partition, ranks, timestamps, and hashes before analysis resumes",
            "required",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "rule_order": order,
                "replacement_rule_id": f"RR-{order:03d}",
                "condition_id": condition,
                "frozen_rule": rule,
                "decision": decision,
                "outcome_dependency_allowed": False,
                "rule_status": "frozen",
            }
            for order, condition, rule, decision in rules
        ]
    )


def reserve_screening_envelope(nominal_k: int, efficiency: float) -> int:
    if nominal_k <= 0:
        raise ValueError("nominal_k must be positive")
    if not (0 < efficiency <= 1):
        raise ValueError("efficiency must lie in (0, 1]")
    return max(nominal_k, int(math.ceil(nominal_k / efficiency - 1e-12)))


def build_reserve_pool_recommendations(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, profile in profiles.iterrows():
        efficiency = float(profile["origin_minimum_support_efficiency"])
        for partition, nominal_field in [
            ("discovery", "discovery_nominal_k"),
            ("confirmation", "confirmation_nominal_k"),
        ]:
            nominal_k = safe_int(profile[nominal_field], name=nominal_field)
            screening = reserve_screening_envelope(nominal_k, efficiency)
            rows.append(
                {
                    "protocol_profile_id": profile["protocol_profile_id"],
                    "record_id": profile["record_id"],
                    "carrier": profile["carrier"],
                    "reliability_target": profile["reliability_target"],
                    "partition": partition,
                    "planned_analysis_objects": nominal_k,
                    "origin_minimum_support_efficiency": efficiency,
                    "artifact_derived_maximum_screened_or_reserved_objects": screening,
                    "artifact_derived_reserve_objects": screening - nominal_k,
                    "derivation_rule": (
                        "ceil(planned_analysis_objects / frozen OBS-086b origin minimum support efficiency)"
                    ),
                    "derivation_lineage": (
                        "OBS-086b origin support-efficiency field inherited from OBS-086a/OBS-085d"
                    ),
                    "analysis_allocation_may_expand": False,
                    "reserve_use": (
                        "same-partition structural replacement only; no outcome-based replacement"
                    ),
                    "interpretation": (
                        "artifact-derived screening-envelope recommendation, not a guaranteed sufficiency bound and not support-grid extrapolation"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["record_id", "reliability_target", "partition"]
    ).reset_index(drop=True)


def build_effective_support_monitoring_rules() -> pd.DataFrame:
    permitted = [
        "eligible_object_count",
        "assigned_analysis_object_count",
        "available_same_partition_reserve_count",
        "effective_cluster_count",
        "cluster_duplication_count",
        "carrier_completeness_rate",
        "measurement_missingness_count",
        "control_availability_count",
        "partition_allocation_integrity",
        "provenance_validation_status",
    ]
    forbidden = [
        "effect_estimate",
        "effect_direction",
        "gate_result",
        "p_value_or_q_value",
        "candidate_passage",
        "candidate_ranking",
        "discovery_model_score",
        "confirmation_outcome",
        "delta_or_control_response_lambda_selection",
    ]
    rows: list[dict[str, Any]] = []
    for order, metric in enumerate(permitted, start=1):
        rows.append(
            {
                "monitoring_rule_id": f"MON-P-{order:03d}",
                "metric_or_signal": metric,
                "monitoring_status": "permitted_blinded_structural_monitoring",
                "decision_use": "continue_freeze_or_structural_futility_only",
                "outcome_bearing": False,
                "minimum_check_frequency": "at roster freeze and after each replacement",
            }
        )
    for order, metric in enumerate(forbidden, start=1):
        rows.append(
            {
                "monitoring_rule_id": f"MON-F-{order:03d}",
                "metric_or_signal": metric,
                "monitoring_status": "prohibited_before_frozen_evaluation_stage",
                "decision_use": "none",
                "outcome_bearing": True,
                "minimum_check_frequency": "not_applicable",
            }
        )
    return pd.DataFrame(rows)


def build_continue_futility_rules(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, profile in profiles.iterrows():
        for partition, nominal_field in [
            ("discovery", "discovery_nominal_k"),
            ("confirmation", "confirmation_nominal_k"),
        ]:
            nominal_k = safe_int(profile[nominal_field], name=nominal_field)
            common = {
                "protocol_profile_id": profile["protocol_profile_id"],
                "partition": partition,
                "planned_analysis_objects": nominal_k,
                "minimum_effective_clusters": (
                    MINIMUM_EFFECTIVE_CLUSTERS_FOR_EXACT_GATE_AT_ALPHA_0_10
                ),
                "outcome_dependency_allowed": False,
            }
            rows.extend(
                [
                    {
                        **common,
                        "decision_order": 1,
                        "decision": "invalidate_protocol_execution",
                        "condition": (
                            "lineage, seed commitment, roster manifest, assignment, provenance, or partition independence fails"
                        ),
                        "action": "stop; record protocol deviation; do not evaluate outcomes",
                    },
                    {
                        **common,
                        "decision_order": 2,
                        "decision": "continue_acquisition_or_structural_replacement",
                        "condition": (
                            "analysis-object quota or minimum effective-cluster requirement is unmet and a valid same-partition reserve remains"
                        ),
                        "action": "take next reserve object by frozen rank and rerun structural validation",
                    },
                    {
                        **common,
                        "decision_order": 3,
                        "decision": "freeze_roster_and_evaluate",
                        "condition": (
                            "exact nominal analysis quota is filled; at least four effective clusters are present; all admissibility, completeness, control, provenance, and partition checks pass"
                        ),
                        "action": "hash final roster; forbid migration; proceed to frozen evaluation sequence",
                    },
                    {
                        **common,
                        "decision_order": 4,
                        "decision": "terminate_for_structural_futility",
                        "condition": (
                            "required nominal or effective support remains unattainable after valid same-partition reserves are exhausted"
                        ),
                        "action": "terminate without outcome inspection or gate reinterpretation",
                    },
                ]
            )
    return pd.DataFrame(rows).sort_values(
        ["protocol_profile_id", "partition", "decision_order"]
    ).reset_index(drop=True)


def build_evaluation_sequence() -> pd.DataFrame:
    steps = [
        (1, "validate_frozen_lineage", "both", "validate all upstream hashes and manifests"),
        (2, "validate_activation_record", "both", "confirm evidence-independent profile activation"),
        (3, "verify_seed_commitment_and_roster_manifest", "both", "verify commit–reveal prerequisites"),
        (4, "generate_and_hash_partition_assignment", "both", "assign exact quotas without cross-partition clusters"),
        (5, "apply_frozen_admissibility_rules", "both", "finalize structural exclusions and replacements"),
        (6, "freeze_final_partition_rosters", "both", "write immutable discovery and confirmation roster hashes"),
        (7, "construct_frozen_carrier", "discovery", "construct only the profile-declared carrier"),
        (8, "compute_frozen_support_statistics", "discovery", "apply frozen support definitions and cluster accounting"),
        (9, "run_discovery_only_candidate_procedure", "discovery", "no confirmation access; no threshold fitting"),
        (10, "seal_discovery_candidate_identity", "discovery", "freeze predicate, support, complement, controls, metrics, and candidate family"),
        (11, "write_and_verify_discovery_manifest", "discovery", "all discovery artifacts must hash-verify"),
        (12, "evaluate_confirmation_opening_prerequisites", "confirmation", "all opening conditions must pass"),
        (13, "create_one_time_confirmation_opening_lock", "confirmation", "open reserved confirmation exactly once"),
        (14, "construct_confirmation_carrier_unchanged", "confirmation", "reuse frozen discovery definitions without modification"),
        (15, "evaluate_only_discovery_sealed_candidate", "confirmation", "no support, predicate, threshold, or candidate search"),
        (16, "apply_frozen_confirmation_gate_conjunction", "confirmation", "retain all failures and multiplicity denominator"),
        (17, "write_claim_entitlement_decision", "confirmation", "cap claim at profile entitlement ceiling"),
        (18, "complete_confirmation_lock_and_manifest", "confirmation", "write final hashes and completion state"),
    ]
    return pd.DataFrame(
        [
            {
                "step_order": order,
                "evaluation_step_id": step_id,
                "partition_scope": scope,
                "frozen_action": action,
                "search_or_threshold_modification_allowed": False,
                "step_status": "preregistered_not_executed",
            }
            for order, step_id, scope, action in steps
        ]
    )


def build_confirmation_opening_contract() -> pd.DataFrame:
    prerequisites = [
        "discovery_roster_frozen_and_hashed",
        "discovery_exclusions_and_replacements_finalized",
        "discovery_evaluation_complete",
        "candidate_identity_and_candidate_family_sealed",
        "discovery_artifact_manifest_written",
        "all_discovery_artifact_hashes_verified",
        "confirmation_roster_hash_matches_preopening_commitment",
        "no_prior_confirmation_opening_lock_or_completion_manifest_exists",
    ]
    rows: list[dict[str, Any]] = []
    for order, prerequisite in enumerate(prerequisites, start=1):
        rows.append(
            {
                "prerequisite_order": order,
                "opening_condition_id": f"CO-{order:03d}",
                "opening_prerequisite": prerequisite,
                "required_value": True,
                "failure_action": "do_not_open_confirmation",
                "override_allowed": False,
            }
        )
    rows.append(
        {
            "prerequisite_order": len(prerequisites) + 1,
            "opening_condition_id": "CO-LOCK",
            "opening_prerequisite": (
                "exclusive confirmation opening lock is created before any confirmation outcome-bearing value is read"
            ),
            "required_value": True,
            "failure_action": "invalidate_confirmation_execution",
            "override_allowed": False,
        }
    )
    rows.append(
        {
            "prerequisite_order": len(prerequisites) + 2,
            "opening_condition_id": "CO-ONCE",
            "opening_prerequisite": "confirmation_partition_opened_exactly_once",
            "required_value": True,
            "failure_action": "invalidate_repeat_opening_or_rerun",
            "override_allowed": False,
        }
    )
    return pd.DataFrame(rows)


def build_gate_contract() -> pd.DataFrame:
    gates = [
        (1, "discovery_partition_only", "discovery", "all discovery fitting and search rows carry partition_role=discovery", "invalidate"),
        (2, "candidate_family_sealed_before_confirmation", "discovery", "candidate identity, predicate, support, complement, controls, metrics, thresholds, and multiplicity family are frozen", "do_not_open_confirmation"),
        (3, "protocol_match", "confirmation", "confirmation implementation exactly matches the sealed discovery protocol", "confirmation_protocol_mismatch"),
        (4, "record_testable", "confirmation", "profile record has complete confirmation observation losses", "confirmation_not_testable"),
        (5, "support_columns_available", "confirmation", "sealed support columns exist unchanged", "confirmation_support_unavailable"),
        (6, "complement_admissible", "confirmation", "sealed complement is present and satisfies frozen minimum support", "confirmation_complement_inadmissible"),
        (7, "direction_match", "confirmation", "confirmation contrast follows the sealed expected direction", "confirmation_direction_reversed"),
        (8, "predicate_semantics_pass", "confirmation", "frozen failure predicate semantics are satisfied", "confirmation_signal_absent"),
        (9, "minimum_effect_pass", "confirmation", "frozen minimum-effect threshold is satisfied", "confirmation_signal_absent"),
        (10, "cluster_sensitivity_pass", "confirmation", "frozen cluster-aware uncertainty gate passes", "confirmation_uncertain_cluster_support"),
        (11, "control_robustness_pass", "confirmation", "frozen negative-control adjustment gate passes", "confirmation_control_explained"),
        (12, "confirmation_multiplicity_pass", "confirmation", "frozen multiplicity correction over the exact sealed candidate family passes", "confirmation_multiplicity_not_survived"),
        (13, "confirmation_eligible", "confirmation", "all frozen conjunction gates pass and profile entitlement permits FL3", "confirmation_reproduced_but_claim_capped_at_fl2"),
    ]
    return pd.DataFrame(
        [
            {
                "gate_order": order,
                "gate_id": gate_id,
                "partition_scope": scope,
                "frozen_gate_definition": definition,
                "failure_status": failure,
                "threshold_modification_allowed": False,
                "test_substitution_allowed": False,
                "one_sided_conversion_allowed": False,
                "partition_pooling_allowed": False,
                "post_hoc_cluster_redefinition_allowed": False,
                "semantic_source_anchors_json": canonical_json(
                    [path.as_posix() for path in DEFAULT_PROTOCOL_ANCHORS]
                ),
            }
            for order, gate_id, scope, definition, failure in gates
        ]
    )


def build_claim_entitlement_table(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, profile in profiles.iterrows():
        entitled = str(profile["entitlement_status"]) == "fl3_entitled"
        rows.append(
            {
                "protocol_profile_id": profile["protocol_profile_id"],
                "source_operational_family_id": profile[
                    "source_operational_family_id"
                ],
                "record_id": profile["record_id"],
                "carrier": profile["carrier"],
                "reliability_target": profile["reliability_target"],
                "inherited_entitlement_status": profile["entitlement_status"],
                "fl3_claim_authorized_if_all_gates_pass": entitled,
                "maximum_claim_maturity": (
                    "fl3_artifact_direct_witness"
                    if entitled
                    else "fl2_localized_support_only"
                ),
                "entitlement_cap_reason": (
                    "none_beyond_standard_scope_ceiling"
                    if entitled
                    else "source_carrier_contract_is_fl3_entitlement_capped"
                ),
                "passing_all_gates_can_raise_entitlement": False,
                "causal_origin_claim_allowed": False,
                "repair_target_claim_allowed": False,
                "intervention_readiness_claim_allowed": False,
                "actionability_claim_allowed": False,
                "external_generalization_claim_allowed": False,
                "formal_topology_claim_allowed": False,
                "claim_decision_status": "not_evaluated_no_campaign_activated",
            }
        )
    return pd.DataFrame(rows)


def build_protocol_freeze_table(profiles: pd.DataFrame) -> pd.DataFrame:
    profile_ids = sorted(profiles["protocol_profile_id"].astype(str))
    frozen_items = [
        ("lineage", "obs086b_manifest_id", "exact frozen manifest identity", "global"),
        ("profile_set", "protocol_profile_ids", canonical_json(profile_ids), "global"),
        ("activation", "global_campaign_selection", "none in OBS-086c", "global"),
        ("activation", "activation_inputs", "entitlement, budget, carrier, measurement, structural feasibility, predeclared target preference", "global"),
        ("assignment", "assignment_algorithm", "sha256_commit_reveal_cluster_blocked_object_assignment_v1", "global"),
        ("assignment", "seed_commitment_timing", "before eligible roster identity is known", "global"),
        ("assignment", "seed_reveal_timing", "after roster hash; before assignment", "global"),
        ("assignment", "partition_pooling", "prohibited", "global"),
        ("assignment", "post_freeze_object_migration", "prohibited", "global"),
        ("admissibility", "outcome_dependent_exclusion", "prohibited", "global"),
        ("replacement", "same_partition_reserve_only", "required", "global"),
        ("replacement", "analysis_quota_expansion", "prohibited", "global"),
        ("support", "minimum_effective_clusters_for_exact_gate", str(MINIMUM_EFFECTIVE_CLUSTERS_FOR_EXACT_GATE_AT_ALPHA_0_10), "each_partition"),
        ("monitoring", "permitted_inputs", "structural and blinded only", "global"),
        ("futility", "outcome_based_futility", "prohibited", "global"),
        ("discovery", "candidate_search_scope", "discovery partition only", "global"),
        ("confirmation", "opening_count", "exactly once", "global"),
        ("confirmation", "candidate_search", "prohibited", "global"),
        ("gates", "gate_modification", "prohibited", "global"),
        ("entitlement", "entitlement_increase", "prohibited", "global"),
        ("scope", "new_simulation", "none", "global"),
        ("scope", "observed_evidence_inspection", "none", "global"),
    ]
    rows: list[dict[str, Any]] = []
    for order, (category, field, value, scope) in enumerate(frozen_items, start=1):
        rows.append(
            {
                "freeze_order": order,
                "freeze_category": category,
                "frozen_field": field,
                "frozen_value": value,
                "scope": scope,
                "mutability_after_obs086c": "immutable_without_documented_protocol_deviation",
                "freeze_status": "frozen_before_activation",
            }
        )
    return pd.DataFrame(rows)


def build_held_family_register(
    held: pd.DataFrame,
    obs086b_manifest_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in held.iterrows():
        rows.append(
            {
                "held_reference_id": stable_row_id(
                    "HR",
                    {
                        "operational_family_id": source["operational_family_id"],
                        "obs086b_manifest_id": obs086b_manifest_id,
                    },
                ),
                "source_operational_family_id": source["operational_family_id"],
                "address_id": source["address_id"],
                "record_id": source["record_id"],
                "carrier": source["carrier"],
                "entitlement_status": source["entitlement_status"],
                "reliability_target": source["reliability_target"],
                "discovery_nominal_k": source["discovery_nominal_k"],
                "confirmation_nominal_k": source["confirmation_nominal_k"],
                "total_nominal_objects": source["total_nominal_objects"],
                "defensible_stress_test_coverage": source[
                    "defensible_stress_test_coverage"
                ],
                "held_reference_status": "nonactivatable_low_coverage_reference",
                "activation_allowed": False,
                "hold_reason": source["protocol_selection_reason"],
                "promotion_rule": (
                    "cannot be promoted inside OBS-086c; requires a new prospective design study without reopening frozen evidence"
                ),
                "observed_evidence_inspected": False,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_HELD_REFERENCES:
        raise RuntimeError("Held family register must contain exactly one reference.")
    return frame


# -----------------------------------------------------------------------------
# Output writing, report, and manifest
# -----------------------------------------------------------------------------


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "input_manifest": output_dir / "obs086c_input_manifest.csv",
        "profile_registry": output_dir / "obs086c_protocol_profile_registry.csv",
        "activation_contract": output_dir / "obs086c_activation_contract.csv",
        "observation_identity": (
            output_dir / "obs086c_observation_identity_contract.csv"
        ),
        "partition_assignment": (
            output_dir / "obs086c_partition_assignment_contract.csv"
        ),
        "admissibility_rules": output_dir / "obs086c_admissibility_rules.csv",
        "exclusion_codes": output_dir / "obs086c_exclusion_reason_codes.csv",
        "replacement_rules": output_dir / "obs086c_replacement_rules.csv",
        "reserve_recommendations": (
            output_dir / "obs086c_reserve_pool_recommendations.csv"
        ),
        "monitoring_rules": (
            output_dir / "obs086c_effective_support_monitoring_rules.csv"
        ),
        "continue_futility": output_dir / "obs086c_continue_futility_rules.csv",
        "evaluation_sequence": output_dir / "obs086c_evaluation_sequence.csv",
        "confirmation_opening": (
            output_dir / "obs086c_confirmation_opening_contract.csv"
        ),
        "gate_contract": output_dir / "obs086c_gate_contract.csv",
        "claim_entitlement": output_dir / "obs086c_claim_entitlement_table.csv",
        "protocol_freeze": output_dir / "obs086c_protocol_freeze_table.csv",
        "held_register": output_dir / "obs086c_held_family_register.csv",
        "failures": output_dir / "obs086c_failures.csv",
        "report": output_dir / "obs086c_report.md",
        "manifest": output_dir / "obs086c_manifest.json",
    }


def artifact_inventory(outputs: Mapping[str, Path], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in sorted(outputs.items()):
        if name == "manifest" or not path.is_file():
            continue
        rows.append(
            {
                "artifact_name": name,
                "artifact_path": repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def validate_synthesized_contracts(
    profiles: pd.DataFrame,
    activation: pd.DataFrame,
    assignment: pd.DataFrame,
    admissibility: pd.DataFrame,
    exclusions: pd.DataFrame,
    replacements: pd.DataFrame,
    reserves: pd.DataFrame,
    monitoring: pd.DataFrame,
    continue_futility: pd.DataFrame,
    evaluation: pd.DataFrame,
    confirmation: pd.DataFrame,
    gates: pd.DataFrame,
    entitlement: pd.DataFrame,
    held: pd.DataFrame,
) -> None:
    if len(profiles) != EXPECTED_SELECTED_PROFILES:
        raise RuntimeError("Exactly five executable profiles are required.")
    if len(held) != EXPECTED_HELD_REFERENCES:
        raise RuntimeError("Exactly one held reference is required.")
    if profiles["activation_status"].astype(str).ne("not_activated").any():
        raise RuntimeError("OBS-086c may not activate a campaign.")
    if profiles["observed_evidence_inspected"].map(normalize_bool).any():
        raise RuntimeError("OBS-086c may not inspect observed evidence.")
    if profiles["assignment_performed"].map(normalize_bool).any():
        raise RuntimeError("OBS-086c may not assign a nonexistent roster.")
    profile_ids = set(profiles["protocol_profile_id"].astype(str))
    for name, frame in [
        ("activation", activation),
        ("assignment", assignment),
        ("reserves", reserves),
        ("continue/futility", continue_futility),
        ("entitlement", entitlement),
    ]:
        if set(frame["protocol_profile_id"].astype(str)) != profile_ids:
            raise RuntimeError(f"{name} contract does not cover exactly all profiles.")
    if exclusions.loc[
        exclusions["code_status"].eq("allowed_prespecified_structural_exclusion"),
        "outcome_dependent",
    ].map(normalize_bool).any():
        raise RuntimeError("An allowed exclusion code is outcome-dependent.")
    if replacements["outcome_dependency_allowed"].map(normalize_bool).any():
        raise RuntimeError("A replacement rule permits outcome dependence.")
    permitted_monitoring = monitoring["monitoring_status"].eq(
        "permitted_blinded_structural_monitoring"
    )
    if monitoring.loc[permitted_monitoring, "outcome_bearing"].map(normalize_bool).any():
        raise RuntimeError("Permitted monitoring contains an outcome-bearing signal.")
    if continue_futility["outcome_dependency_allowed"].map(normalize_bool).any():
        raise RuntimeError("Continue/futility rules permit outcome dependence.")
    if evaluation["search_or_threshold_modification_allowed"].map(normalize_bool).any():
        raise RuntimeError("Evaluation sequence permits search or threshold modification.")
    if confirmation["override_allowed"].map(normalize_bool).any():
        raise RuntimeError("Confirmation opening contract contains an override.")
    forbidden_gate_columns = [
        "threshold_modification_allowed",
        "test_substitution_allowed",
        "one_sided_conversion_allowed",
        "partition_pooling_allowed",
        "post_hoc_cluster_redefinition_allowed",
    ]
    if any(gates[column].map(normalize_bool).any() for column in forbidden_gate_columns):
        raise RuntimeError("Gate contract permits a prohibited modification.")
    capped = entitlement["inherited_entitlement_status"].eq("fl3_entitlement_capped")
    if entitlement.loc[capped, "fl3_claim_authorized_if_all_gates_pass"].map(
        normalize_bool
    ).any():
        raise RuntimeError("An entitlement-capped profile authorizes FL3.")
    if len(reserves) != EXPECTED_SELECTED_PROFILES * 2:
        raise RuntimeError("Reserve recommendations must contain two rows per profile.")
    if (
        pd.to_numeric(reserves["artifact_derived_maximum_screened_or_reserved_objects"])
        < pd.to_numeric(reserves["planned_analysis_objects"])
    ).any():
        raise RuntimeError("A reserve screening envelope is below the analysis quota.")


def write_report(
    path: Path,
    state: str,
    lineage: Mapping[str, Any],
    profiles: pd.DataFrame,
    activation: pd.DataFrame,
    reserves: pd.DataFrame,
    held: pd.DataFrame,
    gates: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    profile_view = profiles[
        [
            "protocol_profile_id",
            "record_id",
            "carrier",
            "entitlement_status",
            "reliability_target",
            "discovery_nominal_k",
            "confirmation_nominal_k",
            "total_nominal_objects",
            "defensible_stress_test_coverage",
            "protocol_profile_status",
        ]
    ]
    reserve_view = reserves[
        [
            "protocol_profile_id",
            "partition",
            "planned_analysis_objects",
            "origin_minimum_support_efficiency",
            "artifact_derived_maximum_screened_or_reserved_objects",
            "artifact_derived_reserve_objects",
        ]
    ]
    held_view = held[
        [
            "held_reference_id",
            "record_id",
            "carrier",
            "reliability_target",
            "discovery_nominal_k",
            "confirmation_nominal_k",
            "defensible_stress_test_coverage",
            "held_reference_status",
        ]
    ]
    gate_view = gates[
        [
            "gate_order",
            "gate_id",
            "partition_scope",
            "failure_status",
        ]
    ]
    lines = [
        "# OBS-086c — Prospective Campaign Protocol Preregistration",
        "",
        "## State",
        "",
        f"`{state}`",
        "",
        (
            "OBS-086c converts the five frozen OBS-086b protocol-selected families "
            "into complete prospective protocol profiles. It does not activate a "
            "campaign, assign a roster, reveal a randomization seed, run a simulation, "
            "or inspect observed scientific evidence."
        ),
        "",
        "## Frozen lineage",
        "",
        f"- OBS-086b commit: `{lineage['obs086b_commit']}`",
        f"- OBS-086b manifest ID: `{lineage['obs086b_manifest_id']}`",
        f"- OBS-086b manifest SHA256: `{lineage['obs086b_manifest_sha256']}`",
        f"- OBS-086b script SHA256: `{lineage['obs086b_script_sha256']}`",
        f"- OBS-086b output artifacts validated: **{lineage['obs086b_output_artifacts_validated']}**",
        f"- Protocol semantic anchors frozen: **{lineage['protocol_semantic_anchor_count']}**",
        f"- Current repository HEAD: `{lineage['current_repo_head']}`",
        "",
        "## Completion result",
        "",
        f"- Executable protocol profiles: **{len(profiles)}**",
        f"- Nonactivatable held references: **{len(held)}**",
        "- Globally selected campaigns: **0**",
        "- New simulations: **0**",
        "- Observed evidence inspected: **0**",
        f"- Validation failures: **{len(failures)}**",
        "",
        "## Executable protocol profiles",
        "",
        markdown_table(profile_view),
        "",
        "No profile is activated by OBS-086c. A future activation record must use only evidence-independent entitlement, resource, carrier, measurement, structural-feasibility, and predeclared target criteria.",
        "",
        "## Activation boundary",
        "",
        "- Activation must precede outcome access.",
        "- Delta and control-response lambda remain uncertainty axes and may not be selected as campaign properties.",
        "- The two entitlement-capped profiles cannot be activated for an FL3 claim.",
        "- OBS-086c selects no global winner among the five profiles.",
        "",
        "## Commit–reveal partition contract",
        "",
        "The assignment algorithm is frozen but no objects are assigned in OBS-086c. A seed commitment must be recorded before the eligible roster identity is known. After the roster is frozen and hashed, the seed is revealed and exact discovery/confirmation quotas are assigned deterministically while preventing object or effective-cluster overlap across partitions.",
        "",
        "Discovery and confirmation may never be pooled, and no object may migrate after roster freeze.",
        "",
        "## Artifact-derived reserve recommendations",
        "",
        markdown_table(reserve_view),
        "",
        "The screening envelopes are calculated from the frozen origin minimum support-efficiency field. They are replacement-planning recommendations only: they do not enlarge the analysis allocation, guarantee effective support, or extrapolate the tested k grid.",
        "",
        "## Held family reference",
        "",
        markdown_table(held_view),
        "",
        "The held family remains nonactivatable because its maximum frozen stress-grid coverage is below majority coverage. OBS-086c cannot promote it.",
        "",
        "## Frozen gate identities",
        "",
        markdown_table(gate_view),
        "",
        "All thresholds, test definitions, controls, multiplicity logic, cluster semantics, and conjunction rules remain frozen. Threshold weakening, gate deletion, test substitution, one-sided conversion, partition pooling, and post hoc cluster redefinition are prohibited.",
        "",
        "## Confirmation opening",
        "",
        "Confirmation may be opened exactly once, only after the discovery roster, exclusions, evaluation, candidate identity, candidate family, manifest, and artifact hashes are frozen and verified. A failed discovery result does not authorize confirmation search.",
        "",
        "## Claim entitlement",
        "",
        "- `three_way__no_window`: maximum claim remains FL2 localized support; FL3 is prohibited.",
        "- `C_vs_Cp3__path_shares_only`: FL3 artifact-direct witness remains conditionally available only if every frozen discovery and confirmation gate later passes.",
        "- No profile authorizes causal origin, repair target, intervention readiness, actionability, external generalization, or formal topology claims.",
        "",
        "## Interpretation boundary",
        "",
        "> OBS-086c is a prospective protocol freeze only.",
        "",
        "> It creates no witness, performs no campaign, and evaluates no scientific outcome.",
        "",
        "> A protocol profile is not a guarantee of passage and does not increase claim entitlement.",
        "",
        "> Discovery and confirmation remain separate; no frozen evidence gate may be weakened.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    *,
    repo_root: Path,
    outputs: Mapping[str, Path],
    state: str,
    lineage: Mapping[str, Any],
    profiles: pd.DataFrame,
    held: pd.DataFrame,
    activation: pd.DataFrame,
    reserves: pd.DataFrame,
    admissibility: pd.DataFrame,
    exclusions: pd.DataFrame,
    replacements: pd.DataFrame,
    monitoring: pd.DataFrame,
    continue_futility: pd.DataFrame,
    evaluation: pd.DataFrame,
    confirmation: pd.DataFrame,
    gates: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, Any]:
    profile_ids = sorted(profiles["protocol_profile_id"].astype(str))
    held_ids = sorted(held["held_reference_id"].astype(str))
    core = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "state": state,
        "scope": (
            "deterministic artifact-only prospective campaign protocol preregistration"
        ),
        "claim_ceiling": (
            "prospective protocol freeze only; no activation, assignment, seed reveal, "
            "simulation, observed evidence, passage guarantee, causal attribution, "
            "actionability, external generalization, formal topology, or entitlement increase"
        ),
        "frozen_lineage": dict(lineage),
        "protocol_contract": {
            "executable_profile_count": len(profile_ids),
            "held_reference_count": len(held_ids),
            "globally_selected_campaign_count": 0,
            "cluster_grid": list(CANONICAL_CLUSTER_GRID),
            "reliability_targets": list(CANONICAL_RELIABILITY_TARGETS),
            "minimum_effective_clusters_for_exact_gate_at_alpha_0_10": (
                MINIMUM_EFFECTIVE_CLUSTERS_FOR_EXACT_GATE_AT_ALPHA_0_10
            ),
            "scientific_observation_key": list(SCIENTIFIC_OBSERVATION_KEY),
            "assignment_algorithm_id": (
                "sha256_commit_reveal_cluster_blocked_object_assignment_v1"
            ),
            "partition_pooling": "prohibited",
            "post_freeze_object_migration": "prohibited",
            "outcome_dependent_exclusion": "prohibited",
            "outcome_dependent_replacement": "prohibited",
            "outcome_dependent_continue_or_futility": "prohibited",
            "confirmation_opening_count": "exactly_once",
            "gate_modification": "prohibited",
            "entitlement_increase": "prohibited",
            "reserve_recommendation_rule": (
                "ceil(partition nominal analysis objects / frozen origin minimum support efficiency)"
            ),
            "reserve_semantics": (
                "screening and same-partition replacement recommendation only; analysis quota unchanged"
            ),
        },
        "execution": {
            "executable_protocol_profiles": len(profiles),
            "held_references": len(held),
            "globally_selected_campaigns": 0,
            "activation_contract_rows": len(activation),
            "reserve_recommendation_rows": len(reserves),
            "admissibility_rule_rows": len(admissibility),
            "exclusion_reason_rows": len(exclusions),
            "replacement_rule_rows": len(replacements),
            "monitoring_rule_rows": len(monitoring),
            "continue_futility_rule_rows": len(continue_futility),
            "evaluation_sequence_rows": len(evaluation),
            "confirmation_opening_rows": len(confirmation),
            "gate_contract_rows": len(gates),
            "new_simulations": 0,
            "observed_evidence_inspected": 0,
            "assignments_performed": 0,
            "confirmations_opened": 0,
            "failures": len(failures),
        },
        "protocol_profile_set": {
            "profile_count": len(profile_ids),
            "profile_ids_sha256": sha256_bytes(
                canonical_json(profile_ids).encode("utf-8")
            ),
        },
        "held_reference_set": {
            "held_count": len(held_ids),
            "held_ids_sha256": sha256_bytes(
                canonical_json(held_ids).encode("utf-8")
            ),
        },
        "output_artifacts": artifact_inventory(outputs, repo_root),
        "mandatory_statements": [
            "OBS-086b remains frozen and unchanged.",
            "Exactly five protocol profiles are registered and none is activated.",
            "The OBS-086b low-coverage hold remains nonactivatable.",
            "No new simulation or observed-evidence evaluation was performed.",
            "Delta and control_response_lambda remain uncertainty axes.",
            "Discovery and confirmation remain separate and may not be pooled.",
            "Assignment uses a frozen commit–reveal algorithm after roster freeze.",
            "Admissibility, replacement, monitoring, and futility decisions are outcome-blind.",
            "Confirmation may be opened exactly once after discovery sealing and hash verification.",
            "No frozen evidence gate may be weakened or substituted.",
            "Passing all gates cannot elevate an entitlement-capped profile.",
            "OBS-086c cannot create a witness or increase claim entitlement.",
        ],
    }
    artifacts = core["output_artifacts"]
    if len(artifacts) != EXPECTED_OBS086C_OUTPUT_ARTIFACTS:
        raise RuntimeError(
            "Unexpected OBS-086c manifest-recorded artifact count: expected "
            f"{EXPECTED_OBS086C_OUTPUT_ARTIFACTS}, found {len(artifacts)}."
        )
    return {
        "obs086c_manifest_id": sha256_bytes(canonical_json(core).encode("utf-8")),
        **core,
    }


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def synthetic_selected_and_held() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, Any]] = []
    source_rows = [
        ("three_way__no_window", "no_window", "fl3_entitlement_capped", 0.50, 8, 10, 0.68),
        ("three_way__no_window", "no_window", "fl3_entitlement_capped", 0.80, 8, 12, 0.56),
        ("C_vs_Cp3__path_shares_only", "path_shares_only", "fl3_entitled", 0.50, 12, 12, 0.68),
        ("C_vs_Cp3__path_shares_only", "path_shares_only", "fl3_entitled", 0.80, 10, 10, 0.56),
        ("C_vs_Cp3__path_shares_only", "path_shares_only", "fl3_entitled", 0.90, 12, 12, 0.52),
    ]
    for index, (record, carrier, entitlement, target, d_k, c_k, coverage) in enumerate(source_rows):
        selected_rows.append(
            {
                "operational_family_id": f"OF-SYN-{index:03d}",
                "address_id": f"ADDR-{record}",
                "record_id": record,
                "support_id": f"SUP-{carrier}",
                "relation": "synthetic_relation",
                "carrier": carrier,
                "entitlement_status": entitlement,
                "reliability_target": target,
                "discovery_nominal_k": d_k,
                "confirmation_nominal_k": c_k,
                "total_nominal_objects": d_k + c_k,
                "partition_allocation_imbalance": abs(d_k - c_k),
                "defensible_target_reaching_scenario_cells": round(coverage * 25),
                "defensible_stress_test_coverage": coverage,
                "origin_minimum_support_efficiency": 0.30,
                "origin_minimum_mean_effective_clusters": 2.0,
                "protocol_selection_status": "protocol_selected_for_preregistration_review",
                "protocol_selection_reason": "synthetic",
            }
        )
    held = pd.DataFrame(
        [
            {
                "operational_family_id": "OF-SYN-HOLD",
                "address_id": "ADDR-three_way__no_window",
                "record_id": "three_way__no_window",
                "support_id": "SUP-no_window",
                "relation": "synthetic_relation",
                "carrier": "no_window",
                "entitlement_status": "fl3_entitlement_capped",
                "reliability_target": 0.90,
                "discovery_nominal_k": 8,
                "confirmation_nominal_k": 12,
                "total_nominal_objects": 20,
                "partition_allocation_imbalance": 4,
                "defensible_target_reaching_scenario_cells": 10,
                "defensible_stress_test_coverage": 0.40,
                "origin_minimum_support_efficiency": 0.30,
                "origin_minimum_mean_effective_clusters": 2.0,
                "protocol_selection_status": "protocol_hold_low_coverage",
                "protocol_selection_reason": "synthetic_low_coverage",
            }
        ]
    )
    return pd.DataFrame(selected_rows), held


def run_self_test() -> None:
    selected, held_source = synthetic_selected_and_held()
    manifest_id = DEFAULT_EXPECTED_OBS086B_MANIFEST_ID
    profiles = build_protocol_profile_registry(selected, manifest_id)
    profiles_2 = build_protocol_profile_registry(selected, manifest_id)
    assert canonical_json(profiles.to_dict("records")) == canonical_json(
        profiles_2.to_dict("records")
    )
    assert len(profiles) == 5
    assert profiles["protocol_profile_id"].nunique() == 5
    assert not profiles["activation_status"].ne("not_activated").any()

    activation = build_activation_contract(profiles)
    assignment = build_partition_assignment_contract(profiles)
    admissibility = build_admissibility_rules()
    exclusions = build_exclusion_reason_codes()
    replacements = build_replacement_rules()
    reserves = build_reserve_pool_recommendations(profiles)
    monitoring = build_effective_support_monitoring_rules()
    continue_futility = build_continue_futility_rules(profiles)
    evaluation = build_evaluation_sequence()
    confirmation = build_confirmation_opening_contract()
    gates = build_gate_contract()
    entitlement = build_claim_entitlement_table(profiles)
    held = build_held_family_register(held_source, manifest_id)

    validate_synthesized_contracts(
        profiles,
        activation,
        assignment,
        admissibility,
        exclusions,
        replacements,
        reserves,
        monitoring,
        continue_futility,
        evaluation,
        confirmation,
        gates,
        entitlement,
        held,
    )

    assert reserve_screening_envelope(12, 0.30) == 40
    assert reserve_screening_envelope(8, 1.00) == 8
    capped = entitlement["inherited_entitlement_status"].eq(
        "fl3_entitlement_capped"
    )
    assert not entitlement.loc[
        capped, "fl3_claim_authorized_if_all_gates_pass"
    ].map(normalize_bool).any()
    assert not held["activation_allowed"].map(normalize_bool).any()
    assert len(continue_futility) == 5 * 2 * 4
    assert len(reserves) == 10

    print("OBS-086c self-test passed")
    print(f"Synthetic executable profiles: {len(profiles)}")
    print(f"Synthetic held references: {len(held)}")
    print(f"Synthetic reserve rows: {len(reserves)}")
    print(f"Synthetic continue/futility rows: {len(continue_futility)}")
    print(f"Synthetic gate rows: {len(gates)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    repo_root = args.repo_root.resolve()
    obs086b_dir = resolve_under_root(args.obs086b_dir, repo_root)
    obs086b_script = resolve_under_root(args.obs086b_script, repo_root)
    output_dir = resolve_under_root(args.output_dir, repo_root)
    protocol_anchors = tuple(args.protocol_anchor or DEFAULT_PROTOCOL_ANCHORS)

    _manifest, frames, input_manifest, lineage = validate_frozen_inputs(
        repo_root=repo_root,
        obs086b_dir=obs086b_dir,
        obs086b_script=obs086b_script,
        expected_manifest_id=args.expected_obs086b_manifest_id,
        expected_script_sha256=args.expected_obs086b_script_sha256,
        explicit_commit=args.expected_obs086b_commit,
        protocol_anchors=protocol_anchors,
    )

    print("OBS-086c validation complete")
    print(f"Frozen OBS-086b commit: {lineage['obs086b_commit']}")
    print(f"Frozen OBS-086b manifest: {lineage['obs086b_manifest_id']}")
    print(
        "Frozen OBS-086b artifacts validated: "
        f"{lineage['obs086b_output_artifacts_validated']}"
    )
    print(
        "Protocol semantic anchors frozen: "
        f"{lineage['protocol_semantic_anchor_count']}"
    )
    print(f"Frozen selected families: {len(frames['selected_families'])}")
    print(f"Frozen held families: {len(frames['held_families'])}")

    if args.validate_only:
        print("Validation-only mode complete; no OBS-086c outputs written.")
        return 0

    profiles = build_protocol_profile_registry(
        frames["selected_families"], lineage["obs086b_manifest_id"]
    )
    activation = build_activation_contract(profiles)
    observation_identity = build_observation_identity_contract()
    assignment = build_partition_assignment_contract(profiles)
    admissibility = build_admissibility_rules()
    exclusions = build_exclusion_reason_codes()
    replacements = build_replacement_rules()
    reserves = build_reserve_pool_recommendations(profiles)
    monitoring = build_effective_support_monitoring_rules()
    continue_futility = build_continue_futility_rules(profiles)
    evaluation = build_evaluation_sequence()
    confirmation = build_confirmation_opening_contract()
    gates = build_gate_contract()
    entitlement = build_claim_entitlement_table(profiles)
    protocol_freeze = build_protocol_freeze_table(profiles)
    held = build_held_family_register(
        frames["held_families"], lineage["obs086b_manifest_id"]
    )
    failure_table = failures_frame([])

    validate_synthesized_contracts(
        profiles,
        activation,
        assignment,
        admissibility,
        exclusions,
        replacements,
        reserves,
        monitoring,
        continue_futility,
        evaluation,
        confirmation,
        gates,
        entitlement,
        held,
    )

    outputs = output_paths(output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    write_csv(input_manifest, outputs["input_manifest"])
    write_csv(profiles, outputs["profile_registry"])
    write_csv(activation, outputs["activation_contract"])
    write_csv(observation_identity, outputs["observation_identity"])
    write_csv(assignment, outputs["partition_assignment"])
    write_csv(admissibility, outputs["admissibility_rules"])
    write_csv(exclusions, outputs["exclusion_codes"])
    write_csv(replacements, outputs["replacement_rules"])
    write_csv(reserves, outputs["reserve_recommendations"])
    write_csv(monitoring, outputs["monitoring_rules"])
    write_csv(continue_futility, outputs["continue_futility"])
    write_csv(evaluation, outputs["evaluation_sequence"])
    write_csv(confirmation, outputs["confirmation_opening"])
    write_csv(gates, outputs["gate_contract"])
    write_csv(entitlement, outputs["claim_entitlement"])
    write_csv(protocol_freeze, outputs["protocol_freeze"])
    write_csv(held, outputs["held_register"])
    write_csv(failure_table, outputs["failures"])
    write_report(
        outputs["report"],
        STATE_COMPLETED,
        lineage,
        profiles,
        activation,
        reserves,
        held,
        gates,
        failure_table,
    )

    output_manifest = build_manifest(
        repo_root=repo_root,
        outputs=outputs,
        state=STATE_COMPLETED,
        lineage=lineage,
        profiles=profiles,
        held=held,
        activation=activation,
        reserves=reserves,
        admissibility=admissibility,
        exclusions=exclusions,
        replacements=replacements,
        monitoring=monitoring,
        continue_futility=continue_futility,
        evaluation=evaluation,
        confirmation=confirmation,
        gates=gates,
        failures=failure_table,
    )
    outputs["manifest"].write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("OBS-086c execution complete")
    print(f"State: {STATE_COMPLETED}")
    print(f"Manifest: {output_manifest['obs086c_manifest_id']}")
    print(f"Executable protocol profiles: {len(profiles)}")
    print(f"Held references: {len(held)}")
    print("Globally selected campaigns: 0")
    print("New simulations: 0")
    print("Observed evidence inspected: 0")
    print(f"Failures: {len(failure_table)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"OBS-086c failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
