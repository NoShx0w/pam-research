#!/usr/bin/env python3
"""
obs086b_robust_campaign_family_selection.py

OBS-086b — Robust Campaign Family Selection
============================================

Purpose
-------
Collapse the frozen OBS-086a scenario-conditioned candidate set into auditable
operational discovery/confirmation allocation families, preserve their complete
nondominated Pareto frontier, and select a small protocol-review set under a
frozen deterministic rule.

OBS-086b treats ``delta`` and ``control_response_lambda`` as uncertainty axes,
not selectable campaign settings.  It therefore evaluates every candidate
allocation against the complete frozen stress-test grid for its address and
reliability target, computes conservative family metrics, preserves every
nonredundant cost/coverage trade-off, and separately identifies the operational
families eligible to advance toward preregistration review.

The study performs no new simulation, no interpolation or extrapolation, no
threshold fitting, no gate modification, no evidence evaluation, and no
selection using observed campaign outcomes.

Core design logic
-----------------
1. Read and validate the frozen OBS-086a artifact family.
2. Form an operational family for every unique sealed allocation:

       (address, reliability target, discovery nominal k,
        confirmation nominal k)

3. Re-evaluate that fixed allocation across all 25 frozen simulator stress-test
   cells using the OBS-086a partition-level robust probability vectors.
4. Keep discovery and confirmation separate; paired probability is the minimum
   of their simulator-robust probabilities at their respective fixed k values.
5. Exclude materially nonmonotone scenario evaluations from defensible target
   coverage and report them explicitly as holds.
6. Compute and preserve the Pareto frontier within each address and reliability target.
7. Enter frontier families within one stress cell of the group maximum into a
   frozen protocol decision band.
8. Advance one lexicographically preferred family when the band reaches at
   least 50% of the frozen stress grid; otherwise retain one low-coverage hold.

Dominance contract
------------------
Family A dominates family B only when A has:

* equal or lower total nominal objects;
* equal or higher defensible stress-test coverage;
* equal or higher worst-case paired probability over all tested scenarios;
* equal or lower discovery/confirmation allocation imbalance; and
* at least one strict improvement beyond the declared numeric epsilon.

Entitlement is preserved and compared conservatively.  In canonical OBS-086a
inputs entitlement is address-fixed, so no cross-entitlement dominance is
permitted or needed.

Protocol-selection contract
---------------------------
Within each address and reliability target, the protocol layer:

* preserves the full Pareto frontier;
* identifies the maximum defensible stress-cell count;
* admits families within one cell of that maximum;
* requires at least 50% stress-grid coverage for advancement; and
* selects lexicographically by minimum total objects, minimum partition
  imbalance, highest minimum probability over defensible cells, highest median
  all-scenario probability, lowest partition gap, lowest simulator spread, and
  stable family ID.

When no near-maximum family reaches majority coverage, one deterministic family
is retained as a low-coverage hold rather than advanced.

Interpretation ceiling
----------------------
A Pareto family is a nonredundant prospective trade-off.  A protocol-selected
family is a bounded preregistration-review candidate.  Neither is observed
evidence, a guarantee of future passage, permission to choose simulator
stress-test conditions, or an increase in claim entitlement.

Canonical run
-------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086b_robust_campaign_family_selection.py \\
  --overwrite

Validation only
---------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086b_robust_campaign_family_selection.py \\
  --validate-only

Engineering smoke run
---------------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086b_robust_campaign_family_selection.py \\
  --smoke --overwrite

Self-test
---------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086b_robust_campaign_family_selection.py \\
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


SCRIPT_VERSION = "1.0.1"
SCHEMA_VERSION = "obs086b_robust_campaign_family_selection_v1_0_1"

DEFAULT_EXPECTED_OBS086A_MANIFEST_ID = (
    "df6714be8b7e0ccaffd7b9df56e067db14426ee98f60c0aefb98bab3da2e72d7"
)
DEFAULT_EXPECTED_OBS086A_SCRIPT_SHA256 = (
    "d2a4724d17774faebf0f93306d6fa34489143fc1545fbe462d301b76eb97d4d6"
)

DEFAULT_OBS086A_DIR = Path(
    "outputs/rig_registry/obs086_campaign_design/"
    "obs086a_campaign_design_synthesis"
)
DEFAULT_OBS086A_SCRIPT = Path(
    "experiments/studies/obs086a_campaign_design_synthesis.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs086_campaign_design/"
    "obs086b_robust_campaign_family_selection"
)

CANONICAL_CLUSTER_GRID = (3, 4, 5, 6, 8, 10, 12)
CANONICAL_RELIABILITY_TARGETS = (0.50, 0.80, 0.90)
EXPECTED_ADDRESS_COUNT = 6
EXPECTED_PARTITIONS = ("confirmation", "discovery")
EXPECTED_SCENARIOS_PER_ADDRESS_TARGET = 25
EXPECTED_PARTITION_ROWS = 900
EXPECTED_PAIRED_ROWS = 450
EXPECTED_SEALED_ROWS = 87
EXPECTED_ADDRESS_SUMMARY_ROWS = 18
EXPECTED_OBS086A_OUTPUT_ARTIFACTS = 12

PROTOCOL_NEAR_MAXIMUM_CELL_SHORTFALL = 1
PROTOCOL_MINIMUM_DEFENSIBLE_COVERAGE = 0.50

BASE_METADATA_COLUMNS = [
    "address_id",
    "record_id",
    "support_id",
    "relation",
    "carrier",
    "entitlement_status",
]
SCENARIO_COLUMNS = ["delta", "control_response_lambda"]

PARTITION_REQUIRED_COLUMNS = {
    *BASE_METADATA_COLUMNS,
    "partition_design_id",
    "partition",
    *SCENARIO_COLUMNS,
    "reliability_target",
    "tested_cluster_grid_json",
    "robust_probability_vector_json",
    "simulator_spread_vector_json",
    "material_nonmonotone_any",
    "candidate_eligible_partition",
    "within_tested_support_envelope",
    "extrapolation_beyond_tested_k_prohibited",
}

PAIRED_REQUIRED_COLUMNS = {
    *BASE_METADATA_COLUMNS,
    "paired_design_id",
    *SCENARIO_COLUMNS,
    "reliability_target",
    "discovery_minimum_nominal_k",
    "confirmation_minimum_nominal_k",
    "minimum_total_nominal_objects",
    "partition_allocation_imbalance",
    "paired_robust_probability_at_selected_allocations",
    "paired_robust_final_tested_probability",
    "partition_final_probability_gap",
    "maximum_simulator_probability_spread_across_partitions",
    "paired_selected_support_efficiency_min",
    "paired_selected_mean_effective_clusters_min",
    "material_nonmonotone_any_partition",
    "paired_design_action",
    "sealed_candidate_eligible",
}

SEALED_REQUIRED_COLUMNS = {
    *PAIRED_REQUIRED_COLUMNS,
    "target_global_rank",
    "address_target_rank",
    "sealed_candidate_status",
}

ADDRESS_SUMMARY_REQUIRED_COLUMNS = {
    *BASE_METADATA_COLUMNS,
    "reliability_target",
    "tested_scenario_cells",
    "sealed_candidate_cells",
    "candidate_family_status",
}

FAMILY_KEY = [
    *BASE_METADATA_COLUMNS,
    "reliability_target",
    "discovery_nominal_k",
    "confirmation_nominal_k",
]

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
    severity: str = "warning"


# -----------------------------------------------------------------------------
# CLI and generic helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OBS-086b: deterministic robust campaign-family selection from "
            "frozen OBS-086a artifacts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--obs086a-dir",
        type=Path,
        default=DEFAULT_OBS086A_DIR,
        help="Frozen OBS-086a output directory.",
    )
    parser.add_argument(
        "--obs086a-script",
        type=Path,
        default=DEFAULT_OBS086A_SCRIPT,
        help="Frozen OBS-086a study script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="OBS-086b output directory.",
    )
    parser.add_argument(
        "--expected-obs086a-manifest-id",
        default=DEFAULT_EXPECTED_OBS086A_MANIFEST_ID,
        help="Required frozen OBS-086a manifest identity.",
    )
    parser.add_argument(
        "--expected-obs086a-script-sha256",
        default=DEFAULT_EXPECTED_OBS086A_SCRIPT_SHA256,
        help="Required frozen OBS-086a script SHA256.",
    )
    parser.add_argument(
        "--expected-obs086a-commit",
        default="",
        help=(
            "Optional explicit OBS-086a freeze commit. When omitted, the script "
            "resolves the newest ancestor commit containing the exact frozen "
            "OBS-086a script and manifest bytes."
        ),
    )
    parser.add_argument(
        "--dominance-epsilon",
        type=float,
        default=1e-12,
        help="Numeric epsilon for Pareto weak/strict comparisons.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process one selected address after full frozen-input validation.",
    )
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help="Engineering-only address limit after frozen-input validation.",
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
    return json.dumps(
        value,
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


def unique_text(values: Iterable[Any]) -> str:
    normalized = sorted({str(value) for value in values if not pd.isna(value)})
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0]
    return "|".join(normalized)


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    columns = list(view.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for _, row in view.iterrows():
        values: list[str] = []
        for value in row.tolist():
            if pd.isna(value):
                text = ""
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        rows.append("")
        rows.append(f"_Showing {max_rows:,} of {len(frame):,} rows._")
    return "\n".join(rows)


def quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(q)) if not numeric.empty else np.nan


def parse_probability_vector(value: Any, cluster_grid: Sequence[int]) -> dict[int, float]:
    try:
        raw = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid probability-vector JSON: {value!r}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Probability vector must decode to an object.")
    expected = {str(int(k)) for k in cluster_grid}
    if set(raw) != expected:
        raise ValueError(
            f"Probability vector keys differ from frozen grid: {sorted(raw)}"
        )
    result: dict[int, float] = {}
    for k in cluster_grid:
        probability = float(raw[str(int(k))])
        if not math.isfinite(probability) or probability < 0 or probability > 1:
            raise ValueError(f"Invalid probability {probability} at k={k}.")
        result[int(k)] = probability
    return result


# -----------------------------------------------------------------------------
# Git and frozen-lineage validation
# -----------------------------------------------------------------------------


def run_git(repo_root: Path, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
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


def resolve_obs086a_freeze_commit(
    repo_root: Path,
    obs086a_script: Path,
    obs086a_manifest: Path,
    expected_script_sha256: str,
    expected_manifest_sha256: str,
    explicit_commit: str,
) -> str:
    script_rel = repo_relative(obs086a_script, repo_root)
    manifest_rel = repo_relative(obs086a_manifest, repo_root)

    if explicit_commit:
        require_commit_ancestor(repo_root, explicit_commit)
        candidates = [explicit_commit]
    else:
        result = run_git(
            repo_root,
            [
                "log",
                "--format=%H",
                "--all",
                "--",
                script_rel,
                manifest_rel,
            ],
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
            "Explicit OBS-086a commit does not contain the exact frozen script "
            "and manifest bytes."
        )
    raise RuntimeError(
        "Could not resolve an ancestor commit containing the exact frozen "
        "OBS-086a script and manifest. Commit OBS-086a before running OBS-086b, "
        "or provide --expected-obs086a-commit."
    )


# -----------------------------------------------------------------------------
# Frozen OBS-086a inputs
# -----------------------------------------------------------------------------


def obs086a_paths(obs086a_dir: Path) -> dict[str, Path]:
    return {
        "input_manifest": obs086a_dir / "obs086a_input_manifest.csv",
        "partition_envelope": obs086a_dir / "obs086a_partition_design_envelope.csv",
        "paired_designs": obs086a_dir / "obs086a_paired_partition_designs.csv",
        "sealed_candidates": obs086a_dir / "obs086a_sealed_candidate_set.csv",
        "address_summary": obs086a_dir / "obs086a_address_decision_summary.csv",
        "support_attrition": obs086a_dir / "obs086a_support_attrition_summary.csv",
        "partition_plan": obs086a_dir / "obs086a_partition_allocation_plan.csv",
        "outside_envelope": obs086a_dir / "obs086a_outside_tested_envelope.csv",
        "protocol_rules": obs086a_dir / "obs086a_protocol_decision_rules.csv",
        "entitlement_overlay": obs086a_dir / "obs086a_entitlement_overlay.csv",
        "failures": obs086a_dir / "obs086a_failures.csv",
        "report": obs086a_dir / "obs086a_report.md",
        "manifest": obs086a_dir / "obs086a_manifest.json",
    }


def validate_manifest_core(
    manifest: Mapping[str, Any],
    expected_manifest_id: str,
) -> None:
    if manifest.get("obs086a_manifest_id") != expected_manifest_id:
        raise RuntimeError(
            "OBS-086a manifest identity mismatch: expected "
            f"{expected_manifest_id}, found {manifest.get('obs086a_manifest_id')}."
        )
    if manifest.get("state") != "campaign_design_synthesis_completed":
        raise RuntimeError(
            f"OBS-086a state is not completed: {manifest.get('state')!r}"
        )
    if manifest.get("schema_version") != "obs086a_campaign_design_synthesis_v1":
        raise RuntimeError(
            f"Unexpected OBS-086a schema: {manifest.get('schema_version')!r}"
        )
    execution = manifest.get("execution", {})
    if int(execution.get("failures", -1)) != 0:
        raise RuntimeError("Frozen OBS-086a manifest records execution failures.")
    design_contract = manifest.get("design_contract", {})
    if tuple(int(k) for k in design_contract.get("cluster_grid", [])) != CANONICAL_CLUSTER_GRID:
        raise RuntimeError("Frozen OBS-086a cluster grid differs from the canonical grid.")
    targets = tuple(float(v) for v in design_contract.get("reliability_targets", []))
    if targets != CANONICAL_RELIABILITY_TARGETS:
        raise RuntimeError("Frozen OBS-086a reliability targets differ from canonical targets.")


def validate_declared_artifacts(
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> pd.DataFrame:
    declared = manifest.get("output_artifacts")
    if not isinstance(declared, list):
        raise RuntimeError("OBS-086a manifest has no output_artifacts list.")
    if len(declared) != EXPECTED_OBS086A_OUTPUT_ARTIFACTS:
        raise RuntimeError(
            "Unexpected OBS-086a artifact count: expected "
            f"{EXPECTED_OBS086A_OUTPUT_ARTIFACTS}, found {len(declared)}."
        )

    rows: list[dict[str, Any]] = []
    for item in declared:
        rel = str(item["artifact_path"])
        path = resolve_under_root(Path(rel), repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen OBS-086a artifact missing: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        expected_size = int(item["size_bytes"])
        expected_hash = str(item["sha256"])
        if actual_size != expected_size:
            raise RuntimeError(
                f"Frozen artifact size mismatch for {rel}: "
                f"expected {expected_size}, found {actual_size}."
            )
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen artifact hash mismatch for {rel}: "
                f"expected {expected_hash}, found {actual_hash}."
            )
        rows.append(
            {
                "input_role": "frozen_obs086a_output",
                "artifact_path": rel,
                "size_bytes": actual_size,
                "sha256": actual_hash,
                "validation_status": "validated",
            }
        )
    return pd.DataFrame(rows)


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def validate_partition_frame(frame: pd.DataFrame) -> None:
    require_columns(frame, PARTITION_REQUIRED_COLUMNS, "partition envelope")
    if len(frame) != EXPECTED_PARTITION_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PARTITION_ROWS} partition rows, found {len(frame)}."
        )
    if frame["address_id"].nunique() != EXPECTED_ADDRESS_COUNT:
        raise RuntimeError("Unexpected address count in partition envelope.")
    if set(frame["partition"].astype(str)) != set(EXPECTED_PARTITIONS):
        raise RuntimeError("Partition envelope does not contain the frozen partitions.")
    targets = tuple(sorted(frame["reliability_target"].astype(float).unique()))
    if targets != CANONICAL_RELIABILITY_TARGETS:
        raise RuntimeError("Partition envelope reliability targets are noncanonical.")
    key = ["address_id", "partition", *SCENARIO_COLUMNS, "reliability_target"]
    if frame.duplicated(key).any():
        raise RuntimeError("Partition envelope has duplicate scenario-partition keys.")
    counts = frame.groupby(["address_id", "reliability_target", "partition"]).size()
    if not counts.eq(EXPECTED_SCENARIOS_PER_ADDRESS_TARGET).all():
        raise RuntimeError("Partition envelope scenario counts differ from 25.")
    for _, row in frame.iterrows():
        grid = tuple(int(k) for k in json.loads(str(row["tested_cluster_grid_json"])))
        if grid != CANONICAL_CLUSTER_GRID:
            raise RuntimeError("Partition row has a noncanonical tested cluster grid.")
        parse_probability_vector(row["robust_probability_vector_json"], grid)
        parse_probability_vector(row["simulator_spread_vector_json"], grid)


def validate_paired_frame(frame: pd.DataFrame) -> None:
    require_columns(frame, PAIRED_REQUIRED_COLUMNS, "paired designs")
    if len(frame) != EXPECTED_PAIRED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_ROWS} paired rows, found {len(frame)}."
        )
    if frame["paired_design_id"].duplicated().any():
        raise RuntimeError("Paired design IDs are not unique.")
    key = ["address_id", *SCENARIO_COLUMNS, "reliability_target"]
    if frame.duplicated(key).any():
        raise RuntimeError("Paired designs have duplicate scenario keys.")


def validate_sealed_frame(sealed: pd.DataFrame, paired: pd.DataFrame) -> None:
    require_columns(sealed, SEALED_REQUIRED_COLUMNS, "sealed candidates")
    if len(sealed) != EXPECTED_SEALED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_SEALED_ROWS} sealed rows, found {len(sealed)}."
        )
    if sealed["paired_design_id"].duplicated().any():
        raise RuntimeError("Sealed candidate IDs are not unique.")
    if not sealed["sealed_candidate_eligible"].map(normalize_bool).all():
        raise RuntimeError("Sealed candidate file contains an ineligible row.")
    paired_ids = set(paired["paired_design_id"].astype(str))
    sealed_ids = set(sealed["paired_design_id"].astype(str))
    if not sealed_ids.issubset(paired_ids):
        raise RuntimeError("Sealed candidates are not a subset of paired designs.")
    paired_eligible = set(
        paired.loc[
            paired["sealed_candidate_eligible"].map(normalize_bool),
            "paired_design_id",
        ].astype(str)
    )
    if sealed_ids != paired_eligible:
        raise RuntimeError("Sealed candidate set differs from eligible paired rows.")
    for column in ["discovery_minimum_nominal_k", "confirmation_minimum_nominal_k"]:
        values = {int(v) for v in pd.to_numeric(sealed[column], errors="raise")}
        if not values.issubset(set(CANONICAL_CLUSTER_GRID)):
            raise RuntimeError(f"Sealed {column} contains support outside frozen grid.")
    if sealed["material_nonmonotone_any_partition"].map(normalize_bool).any():
        raise RuntimeError("Sealed candidate set contains material nonmonotonicity.")


def validate_address_summary(frame: pd.DataFrame) -> None:
    require_columns(frame, ADDRESS_SUMMARY_REQUIRED_COLUMNS, "address summary")
    if len(frame) != EXPECTED_ADDRESS_SUMMARY_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ADDRESS_SUMMARY_ROWS} address rows, found {len(frame)}."
        )


def validate_frozen_inputs(
    repo_root: Path,
    obs086a_dir: Path,
    obs086a_script: Path,
    expected_manifest_id: str,
    expected_script_sha256: str,
    explicit_commit: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    paths = obs086a_paths(obs086a_dir)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required OBS-086a {name} missing: {path}")
    if not obs086a_script.is_file():
        raise FileNotFoundError(f"Frozen OBS-086a script missing: {obs086a_script}")

    script_hash = sha256_file(obs086a_script)
    if script_hash != expected_script_sha256:
        raise RuntimeError(
            "OBS-086a script hash mismatch: expected "
            f"{expected_script_sha256}, found {script_hash}."
        )

    manifest_bytes = paths["manifest"].read_bytes()
    manifest_sha256 = sha256_bytes(manifest_bytes)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_manifest_core(manifest, expected_manifest_id)
    artifact_rows = validate_declared_artifacts(manifest, repo_root)

    freeze_commit = resolve_obs086a_freeze_commit(
        repo_root=repo_root,
        obs086a_script=obs086a_script,
        obs086a_manifest=paths["manifest"],
        expected_script_sha256=expected_script_sha256,
        expected_manifest_sha256=manifest_sha256,
        explicit_commit=explicit_commit,
    )

    frames = {
        "partition_envelope": pd.read_csv(paths["partition_envelope"]),
        "paired_designs": pd.read_csv(paths["paired_designs"]),
        "sealed_candidates": pd.read_csv(paths["sealed_candidates"]),
        "address_summary": pd.read_csv(paths["address_summary"]),
        "failures": pd.read_csv(paths["failures"]),
    }
    validate_partition_frame(frames["partition_envelope"])
    validate_paired_frame(frames["paired_designs"])
    validate_sealed_frame(frames["sealed_candidates"], frames["paired_designs"])
    validate_address_summary(frames["address_summary"])
    if len(frames["failures"]) != 0:
        raise RuntimeError("Frozen OBS-086a failures table is not empty.")

    execution = manifest.get("execution", {})
    if int(execution.get("partition_design_rows", -1)) != len(frames["partition_envelope"]):
        raise RuntimeError("Manifest partition-row count does not match artifact.")
    if int(execution.get("paired_design_rows", -1)) != len(frames["paired_designs"]):
        raise RuntimeError("Manifest paired-row count does not match artifact.")
    if int(execution.get("sealed_candidate_rows", -1)) != len(frames["sealed_candidates"]):
        raise RuntimeError("Manifest sealed-row count does not match artifact.")

    lineage = {
        "obs086a_commit": freeze_commit,
        "obs086a_manifest_id": expected_manifest_id,
        "obs086a_manifest_sha256": manifest_sha256,
        "obs086a_script_sha256": script_hash,
        "obs086a_output_artifacts_validated": len(artifact_rows),
        "obs086a_partition_rows": len(frames["partition_envelope"]),
        "obs086a_paired_rows": len(frames["paired_designs"]),
        "obs086a_sealed_rows": len(frames["sealed_candidates"]),
        "current_repo_head": git_head(repo_root),
    }

    artifact_rows = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "input_role": "frozen_obs086a_manifest",
                        "artifact_path": repo_relative(paths["manifest"], repo_root),
                        "size_bytes": paths["manifest"].stat().st_size,
                        "sha256": manifest_sha256,
                        "validation_status": "validated",
                    },
                    {
                        "input_role": "frozen_obs086a_script",
                        "artifact_path": repo_relative(obs086a_script, repo_root),
                        "size_bytes": obs086a_script.stat().st_size,
                        "sha256": script_hash,
                        "validation_status": "validated",
                    },
                ]
            ),
            artifact_rows,
        ],
        ignore_index=True,
    )
    return manifest, frames, artifact_rows, lineage


# -----------------------------------------------------------------------------
# Operational family construction and fixed-allocation evaluation
# -----------------------------------------------------------------------------


def build_candidate_allocations(sealed: pd.DataFrame) -> pd.DataFrame:
    working = sealed.copy()
    working["discovery_nominal_k"] = pd.to_numeric(
        working["discovery_minimum_nominal_k"], errors="raise"
    ).astype(int)
    working["confirmation_nominal_k"] = pd.to_numeric(
        working["confirmation_minimum_nominal_k"], errors="raise"
    ).astype(int)

    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(FAMILY_KEY, sort=True, dropna=False):
        metadata = dict(zip(FAMILY_KEY, keys))
        payload = {
            "address_id": metadata["address_id"],
            "reliability_target": float(metadata["reliability_target"]),
            "discovery_nominal_k": int(metadata["discovery_nominal_k"]),
            "confirmation_nominal_k": int(metadata["confirmation_nominal_k"]),
        }
        d_k = int(metadata["discovery_nominal_k"])
        c_k = int(metadata["confirmation_nominal_k"])
        rows.append(
            {
                "operational_family_id": stable_row_id("OF", payload),
                **metadata,
                "total_nominal_objects": d_k + c_k,
                "maximum_partition_nominal_k": max(d_k, c_k),
                "partition_allocation_imbalance": abs(d_k - c_k),
                "origin_sealed_candidate_cells": len(group),
                "origin_sealed_candidate_ids_json": canonical_json(
                    sorted(group["paired_design_id"].astype(str).tolist())
                ),
                "origin_delta_values_json": canonical_json(
                    sorted({float(v) for v in group["delta"]})
                ),
                "origin_control_response_lambda_values_json": canonical_json(
                    sorted({float(v) for v in group["control_response_lambda"]})
                ),
                "origin_minimum_paired_probability": float(
                    pd.to_numeric(
                        group["paired_robust_probability_at_selected_allocations"],
                        errors="coerce",
                    ).min()
                ),
                "origin_median_paired_probability": float(
                    pd.to_numeric(
                        group["paired_robust_probability_at_selected_allocations"],
                        errors="coerce",
                    ).median()
                ),
                "origin_minimum_support_efficiency": float(
                    pd.to_numeric(
                        group["paired_selected_support_efficiency_min"],
                        errors="coerce",
                    ).min()
                ),
                "origin_minimum_mean_effective_clusters": float(
                    pd.to_numeric(
                        group["paired_selected_mean_effective_clusters_min"],
                        errors="coerce",
                    ).min()
                ),
                "origin_material_nonmonotone": bool(
                    group["material_nonmonotone_any_partition"]
                    .map(normalize_bool)
                    .any()
                ),
                "scenario_axis_semantics": (
                    "delta and control_response_lambda are frozen simulator "
                    "stress-test axes, not selectable campaign settings"
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No operational allocation families could be formed.")
    return frame.sort_values(
        [
            "address_id",
            "reliability_target",
            "total_nominal_objects",
            "partition_allocation_imbalance",
            "discovery_nominal_k",
            "confirmation_nominal_k",
        ]
    ).reset_index(drop=True)


def build_origin_scenario_set(sealed: pd.DataFrame) -> set[tuple[str, float, int, int, float, float]]:
    result: set[tuple[str, float, int, int, float, float]] = set()
    for _, row in sealed.iterrows():
        result.add(
            (
                str(row["address_id"]),
                float(row["reliability_target"]),
                int(float(row["discovery_minimum_nominal_k"])),
                int(float(row["confirmation_minimum_nominal_k"])),
                float(row["delta"]),
                float(row["control_response_lambda"]),
            )
        )
    return result


def evaluate_operational_families(
    families: pd.DataFrame,
    partition_envelope: pd.DataFrame,
    sealed: pd.DataFrame,
    cluster_grid: Sequence[int],
) -> pd.DataFrame:
    origin_scenarios = build_origin_scenario_set(sealed)
    rows: list[dict[str, Any]] = []

    index_columns = [
        "address_id",
        "reliability_target",
        *SCENARIO_COLUMNS,
        "partition",
    ]
    indexed = partition_envelope.set_index(index_columns)
    if not indexed.index.is_unique:
        duplicate_keys = indexed.index[indexed.index.duplicated()].unique()
        raise RuntimeError(
            "Partition-envelope lookup index is not unique; duplicate keys include: "
            f"{list(duplicate_keys[:10])}"
        )

    for _, family in families.iterrows():
        address = str(family["address_id"])
        target = float(family["reliability_target"])
        d_k = int(family["discovery_nominal_k"])
        c_k = int(family["confirmation_nominal_k"])

        scenarios = (
            partition_envelope.loc[
                partition_envelope["address_id"].astype(str).eq(address)
                & partition_envelope["reliability_target"].astype(float).eq(target),
                SCENARIO_COLUMNS,
            ]
            .drop_duplicates()
            .sort_values(SCENARIO_COLUMNS)
        )
        if len(scenarios) != EXPECTED_SCENARIOS_PER_ADDRESS_TARGET:
            raise RuntimeError(
                f"Family {family['operational_family_id']} does not have 25 stress scenarios."
            )

        for _, scenario in scenarios.iterrows():
            delta = float(scenario["delta"])
            lambda_value = float(scenario["control_response_lambda"])
            discovery = indexed.loc[(address, target, delta, lambda_value, "discovery")]
            confirmation = indexed.loc[(address, target, delta, lambda_value, "confirmation")]

            d_vector = parse_probability_vector(
                discovery["robust_probability_vector_json"], cluster_grid
            )
            c_vector = parse_probability_vector(
                confirmation["robust_probability_vector_json"], cluster_grid
            )
            d_spread = parse_probability_vector(
                discovery["simulator_spread_vector_json"], cluster_grid
            )
            c_spread = parse_probability_vector(
                confirmation["simulator_spread_vector_json"], cluster_grid
            )

            d_probability = float(d_vector[d_k])
            c_probability = float(c_vector[c_k])
            paired_probability = min(d_probability, c_probability)
            d_reached = d_probability >= target
            c_reached = c_probability >= target
            nonmonotone = bool(
                normalize_bool(discovery["material_nonmonotone_any"])
                or normalize_bool(confirmation["material_nonmonotone_any"])
            )

            if nonmonotone and d_reached and c_reached:
                action = "hold_target_reached_material_nonmonotonicity"
                defensible_reach = False
            elif nonmonotone:
                action = "hold_material_nonmonotonicity"
                defensible_reach = False
            elif d_reached and c_reached:
                action = "fixed_allocation_target_reached"
                defensible_reach = True
            elif d_reached != c_reached:
                action = "fixed_allocation_partition_discordance"
                defensible_reach = False
            else:
                action = "fixed_allocation_below_target"
                defensible_reach = False

            scenario_key = (
                address,
                target,
                d_k,
                c_k,
                delta,
                lambda_value,
            )
            rows.append(
                {
                    "operational_family_id": family["operational_family_id"],
                    **{column: family[column] for column in BASE_METADATA_COLUMNS},
                    "reliability_target": target,
                    "discovery_nominal_k": d_k,
                    "confirmation_nominal_k": c_k,
                    "total_nominal_objects": d_k + c_k,
                    "partition_allocation_imbalance": abs(d_k - c_k),
                    "delta": delta,
                    "control_response_lambda": lambda_value,
                    "origin_sealed_member_scenario": scenario_key in origin_scenarios,
                    "discovery_robust_probability_at_fixed_k": d_probability,
                    "confirmation_robust_probability_at_fixed_k": c_probability,
                    "paired_robust_probability_at_fixed_allocation": paired_probability,
                    "discovery_target_reached": d_reached,
                    "confirmation_target_reached": c_reached,
                    "paired_target_reached_before_stability_hold": d_reached and c_reached,
                    "material_nonmonotone_any_partition": nonmonotone,
                    "defensible_paired_target_reached": defensible_reach,
                    "partition_probability_gap_at_fixed_allocation": abs(
                        d_probability - c_probability
                    ),
                    "maximum_simulator_spread_at_fixed_allocation": max(
                        float(d_spread[d_k]), float(c_spread[c_k])
                    ),
                    "paired_probability_shortfall": max(0.0, target - paired_probability),
                    "scenario_evaluation_action": action,
                    "scenario_axis_semantics": (
                        "stress-test uncertainty cell; not an observed or selectable setting"
                    ),
                    "partition_separation_required": True,
                    "pooled_evaluation_prohibited": True,
                    "extrapolation_beyond_tested_k_prohibited": True,
                }
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "address_id",
            "reliability_target",
            "total_nominal_objects",
            "operational_family_id",
            "delta",
            "control_response_lambda",
        ]
    ).reset_index(drop=True)


def classify_coverage(coverage: float) -> str:
    if coverage >= 1.0 - 1e-12:
        return "universal_over_tested_stress_grid"
    if coverage >= 0.80:
        return "broad_tested_stress_coverage"
    if coverage >= 0.50:
        return "majority_tested_stress_coverage"
    if coverage > 0:
        return "restricted_tested_stress_coverage"
    return "no_defensible_tested_stress_coverage"


def summarize_operational_families(
    families: pd.DataFrame,
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    family_lookup = families.set_index("operational_family_id")
    if not family_lookup.index.is_unique:
        duplicate_ids = family_lookup.index[family_lookup.index.duplicated()].unique()
        raise RuntimeError(
            "Operational family IDs are not unique; duplicates include: "
            f"{list(duplicate_ids[:10])}"
        )
    rows: list[dict[str, Any]] = []

    for family_id, group in evaluations.groupby("operational_family_id", sort=True):
        source = family_lookup.loc[family_id]
        scenario_count = len(group)
        if scenario_count != EXPECTED_SCENARIOS_PER_ADDRESS_TARGET:
            raise RuntimeError(f"Unexpected scenario count for family {family_id}.")
        defensible = group["defensible_paired_target_reached"].map(normalize_bool)
        raw_reached = group["paired_target_reached_before_stability_hold"].map(normalize_bool)
        nonmonotone = group["material_nonmonotone_any_partition"].map(normalize_bool)
        paired = pd.to_numeric(
            group["paired_robust_probability_at_fixed_allocation"], errors="raise"
        )
        passing = paired.loc[defensible]
        coverage = float(defensible.mean())
        action_counts = group["scenario_evaluation_action"].value_counts().to_dict()

        rows.append(
            {
                "operational_family_id": family_id,
                **source.to_dict(),
                "tested_stress_scenario_cells": scenario_count,
                "defensible_target_reaching_scenario_cells": int(defensible.sum()),
                "raw_target_reaching_scenario_cells": int(raw_reached.sum()),
                "material_nonmonotone_hold_cells": int(nonmonotone.sum()),
                "partition_discordant_cells_at_fixed_allocation": int(
                    action_counts.get("fixed_allocation_partition_discordance", 0)
                ),
                "below_target_cells_at_fixed_allocation": int(
                    action_counts.get("fixed_allocation_below_target", 0)
                ),
                "defensible_stress_test_coverage": coverage,
                "origin_sealed_member_share": float(
                    group["origin_sealed_member_scenario"].map(normalize_bool).mean()
                ),
                "coverage_class": classify_coverage(coverage),
                "worst_case_paired_probability_all_scenarios": float(paired.min()),
                "q10_paired_probability_all_scenarios": quantile(paired, 0.10),
                "q25_paired_probability_all_scenarios": quantile(paired, 0.25),
                "median_paired_probability_all_scenarios": float(paired.median()),
                "mean_paired_probability_all_scenarios": float(paired.mean()),
                "maximum_paired_probability_all_scenarios": float(paired.max()),
                "minimum_paired_probability_defensible_scenarios": (
                    float(passing.min()) if not passing.empty else np.nan
                ),
                "median_paired_probability_defensible_scenarios": (
                    float(passing.median()) if not passing.empty else np.nan
                ),
                "maximum_partition_probability_gap": float(
                    pd.to_numeric(
                        group["partition_probability_gap_at_fixed_allocation"],
                        errors="raise",
                    ).max()
                ),
                "mean_partition_probability_gap": float(
                    pd.to_numeric(
                        group["partition_probability_gap_at_fixed_allocation"],
                        errors="raise",
                    ).mean()
                ),
                "maximum_simulator_spread_at_fixed_allocation": float(
                    pd.to_numeric(
                        group["maximum_simulator_spread_at_fixed_allocation"],
                        errors="raise",
                    ).max()
                ),
                "mean_probability_shortfall": float(
                    pd.to_numeric(group["paired_probability_shortfall"], errors="raise").mean()
                ),
                "maximum_probability_shortfall": float(
                    pd.to_numeric(group["paired_probability_shortfall"], errors="raise").max()
                ),
                "all_tested_scenarios_defensibly_reach_target": bool(defensible.all()),
                "operational_family_eligible": bool(
                    coverage > 0 and not normalize_bool(source["origin_material_nonmonotone"])
                ),
                "family_interpretation": (
                    "fixed discovery/confirmation allocation evaluated over the complete "
                    "frozen stress-test grid; coverage is combinatorial, not a real-world probability"
                ),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "address_id",
            "reliability_target",
            "total_nominal_objects",
            "defensible_stress_test_coverage",
            "operational_family_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Pareto dominance and family selection
# -----------------------------------------------------------------------------


def entitlement_no_worse(a: str, b: str) -> bool:
    return ENTITLEMENT_RANK.get(str(a), -1) >= ENTITLEMENT_RANK.get(str(b), -1)


def family_dominates(a: Mapping[str, Any], b: Mapping[str, Any], epsilon: float) -> bool:
    if str(a["address_id"]) != str(b["address_id"]):
        return False
    if abs(float(a["reliability_target"]) - float(b["reliability_target"])) > epsilon:
        return False
    if not normalize_bool(a["operational_family_eligible"]):
        return False
    if not normalize_bool(b["operational_family_eligible"]):
        return False
    if not entitlement_no_worse(str(a["entitlement_status"]), str(b["entitlement_status"])):
        return False

    weak = [
        float(a["total_nominal_objects"]) <= float(b["total_nominal_objects"]) + epsilon,
        float(a["defensible_stress_test_coverage"]) + epsilon
        >= float(b["defensible_stress_test_coverage"]),
        float(a["worst_case_paired_probability_all_scenarios"]) + epsilon
        >= float(b["worst_case_paired_probability_all_scenarios"]),
        float(a["partition_allocation_imbalance"])
        <= float(b["partition_allocation_imbalance"]) + epsilon,
    ]
    if not all(weak):
        return False

    strict = [
        float(a["total_nominal_objects"]) < float(b["total_nominal_objects"]) - epsilon,
        float(a["defensible_stress_test_coverage"])
        > float(b["defensible_stress_test_coverage"]) + epsilon,
        float(a["worst_case_paired_probability_all_scenarios"])
        > float(b["worst_case_paired_probability_all_scenarios"]) + epsilon,
        float(a["partition_allocation_imbalance"])
        < float(b["partition_allocation_imbalance"]) - epsilon,
        ENTITLEMENT_RANK.get(str(a["entitlement_status"]), -1)
        > ENTITLEMENT_RANK.get(str(b["entitlement_status"]), -1),
    ]
    return any(strict)


def build_family_dominance_table(
    family_summary: pd.DataFrame,
    epsilon: float,
) -> pd.DataFrame:
    columns = [
        "address_id",
        "reliability_target",
        "dominating_family_id",
        "dominated_family_id",
        "dominating_total_nominal_objects",
        "dominated_total_nominal_objects",
        "dominating_stress_test_coverage",
        "dominated_stress_test_coverage",
        "dominating_worst_case_probability",
        "dominated_worst_case_probability",
        "dominating_partition_imbalance",
        "dominated_partition_imbalance",
        "dominance_rule",
    ]
    rows: list[dict[str, Any]] = []
    for _, group in family_summary.groupby(
        ["address_id", "reliability_target"], sort=True
    ):
        records = group.to_dict("records")
        for a in records:
            for b in records:
                if a["operational_family_id"] == b["operational_family_id"]:
                    continue
                if family_dominates(a, b, epsilon):
                    rows.append(
                        {
                            "address_id": a["address_id"],
                            "reliability_target": a["reliability_target"],
                            "dominating_family_id": a["operational_family_id"],
                            "dominated_family_id": b["operational_family_id"],
                            "dominating_total_nominal_objects": a["total_nominal_objects"],
                            "dominated_total_nominal_objects": b["total_nominal_objects"],
                            "dominating_stress_test_coverage": a[
                                "defensible_stress_test_coverage"
                            ],
                            "dominated_stress_test_coverage": b[
                                "defensible_stress_test_coverage"
                            ],
                            "dominating_worst_case_probability": a[
                                "worst_case_paired_probability_all_scenarios"
                            ],
                            "dominated_worst_case_probability": b[
                                "worst_case_paired_probability_all_scenarios"
                            ],
                            "dominating_partition_imbalance": a[
                                "partition_allocation_imbalance"
                            ],
                            "dominated_partition_imbalance": b[
                                "partition_allocation_imbalance"
                            ],
                            "dominance_rule": (
                                "no_more_cost; no_less coverage; no_lower worst-case "
                                "probability; no_more imbalance; at least one strict improvement"
                            ),
                        }
                    )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["address_id", "reliability_target", "dominated_family_id", "dominating_family_id"]
    ).reset_index(drop=True)


def annotate_frontier(
    family_summary: pd.DataFrame,
    dominance: pd.DataFrame,
) -> pd.DataFrame:
    result = family_summary.copy()
    dominated_by: dict[str, list[str]] = {}
    dominates: dict[str, list[str]] = {}
    for _, row in dominance.iterrows():
        dominated_by.setdefault(str(row["dominated_family_id"]), []).append(
            str(row["dominating_family_id"])
        )
        dominates.setdefault(str(row["dominating_family_id"]), []).append(
            str(row["dominated_family_id"])
        )

    result["dominated_by_count"] = result["operational_family_id"].map(
        lambda value: len(dominated_by.get(str(value), []))
    )
    result["dominates_count"] = result["operational_family_id"].map(
        lambda value: len(dominates.get(str(value), []))
    )
    result["dominated_by_family_ids_json"] = result["operational_family_id"].map(
        lambda value: canonical_json(sorted(dominated_by.get(str(value), [])))
    )
    result["dominates_family_ids_json"] = result["operational_family_id"].map(
        lambda value: canonical_json(sorted(dominates.get(str(value), [])))
    )
    result["pareto_nondominated"] = (
        result["operational_family_eligible"].map(normalize_bool)
        & result["dominated_by_count"].eq(0)
    )
    result["pareto_status"] = np.select(
        [
            ~result["operational_family_eligible"].map(normalize_bool),
            result["pareto_nondominated"],
        ],
        [
            "ineligible_operational_family",
            "nondominated_operational_family",
        ],
        default="dominated_operational_family",
    )
    return result


def selection_roles(group: pd.DataFrame, epsilon: float) -> dict[str, list[str]]:
    """Return descriptive Pareto-anchor roles without inventing flat anchors."""
    roles: dict[str, list[str]] = {str(fid): [] for fid in group["operational_family_id"]}
    if group.empty:
        return roles

    max_coverage = float(group["defensible_stress_test_coverage"].max())
    min_cost = float(group["total_nominal_objects"].min())
    worst = pd.to_numeric(
        group["worst_case_paired_probability_all_scenarios"], errors="raise"
    )
    worst_is_informative = float(worst.max() - worst.min()) > epsilon
    max_worst = float(worst.max())
    min_imbalance = float(group["partition_allocation_imbalance"].min())

    for _, row in group.iterrows():
        family_id = str(row["operational_family_id"])
        if abs(float(row["defensible_stress_test_coverage"]) - max_coverage) <= epsilon:
            roles[family_id].append("coverage_anchor")
        if abs(float(row["total_nominal_objects"]) - min_cost) <= epsilon:
            roles[family_id].append("cost_anchor")
        if (
            worst_is_informative
            and abs(
                float(row["worst_case_paired_probability_all_scenarios"]) - max_worst
            )
            <= epsilon
        ):
            roles[family_id].append("worst_case_probability_anchor")
        if abs(float(row["partition_allocation_imbalance"]) - min_imbalance) <= epsilon:
            roles[family_id].append("partition_balance_anchor")
    return roles


def protocol_lexicographic_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen protocol tie-break hierarchy deterministically."""
    return frame.sort_values(
        [
            "total_nominal_objects",
            "partition_allocation_imbalance",
            "minimum_paired_probability_defensible_scenarios",
            "median_paired_probability_all_scenarios",
            "maximum_partition_probability_gap",
            "maximum_simulator_spread_at_fixed_allocation",
            "operational_family_id",
        ],
        ascending=[True, True, False, False, True, True, True],
        na_position="last",
    )


def build_selection_layers(
    annotated: pd.DataFrame,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preserve the Pareto frontier and add a separate protocol decision layer.

    Within each address × reliability target group:

    1. identify the maximum defensible stress-cell count;
    2. retain frontier families within one cell of that maximum;
    3. require at least 50% defensible coverage for advancement;
    4. select one family by the frozen lexicographic tie-break hierarchy;
    5. when no near-maximum family reaches majority coverage, retain one
       deterministic low-coverage protocol hold instead of selecting it.
    """
    result = annotated.copy()
    result["selection_roles_json"] = canonical_json([])
    result["maximum_group_defensible_cells"] = np.nan
    result["defensible_cell_shortfall_from_group_maximum"] = np.nan
    result["near_maximum_coverage_eligible"] = False
    result["majority_coverage_required"] = True
    result["protocol_selection_rank"] = np.nan
    result["protocol_selection_status"] = np.where(
        result["pareto_nondominated"].map(normalize_bool),
        "pareto_frontier_not_protocol_selected",
        "not_on_pareto_frontier",
    )
    result["protocol_selection_reason"] = np.where(
        result["pareto_nondominated"].map(normalize_bool),
        "awaiting_frozen_protocol_selection_rule",
        "ineligible_or_pareto_dominated",
    )
    result["protocol_freeze_review_eligible"] = False

    selected_ids: list[str] = []
    held_ids: list[str] = []

    for _, group in result.groupby(["address_id", "reliability_target"], sort=True):
        frontier_mask = group["pareto_nondominated"].map(normalize_bool)
        frontier = group.loc[frontier_mask].copy()
        if frontier.empty:
            continue

        role_map = selection_roles(frontier, epsilon)
        result.loc[frontier.index, "selection_roles_json"] = frontier[
            "operational_family_id"
        ].map(lambda value: canonical_json(role_map.get(str(value), [])))

        max_cells = int(frontier["defensible_target_reaching_scenario_cells"].max())
        shortfall = max_cells - frontier["defensible_target_reaching_scenario_cells"].astype(int)
        near_maximum = shortfall.le(PROTOCOL_NEAR_MAXIMUM_CELL_SHORTFALL)
        majority = frontier["defensible_stress_test_coverage"].astype(float).ge(
            PROTOCOL_MINIMUM_DEFENSIBLE_COVERAGE - epsilon
        )

        result.loc[frontier.index, "maximum_group_defensible_cells"] = max_cells
        result.loc[
            frontier.index, "defensible_cell_shortfall_from_group_maximum"
        ] = shortfall.astype(int)
        result.loc[frontier.index, "near_maximum_coverage_eligible"] = near_maximum

        advancement_pool = frontier.loc[near_maximum & majority].copy()
        if not advancement_pool.empty:
            ordered = protocol_lexicographic_order(advancement_pool)
            for rank, index in enumerate(ordered.index, start=1):
                result.at[index, "protocol_selection_rank"] = rank
            chosen_index = ordered.index[0]
            chosen_id = str(result.at[chosen_index, "operational_family_id"])
            selected_ids.append(chosen_id)
            result.at[chosen_index, "protocol_selection_status"] = (
                "protocol_selected_for_preregistration_review"
            )
            result.at[chosen_index, "protocol_selection_reason"] = (
                "within_one_stress_cell_of_group_maximum; majority_coverage; "
                "lexicographically_best_under_frozen_rule"
            )
            result.at[chosen_index, "protocol_freeze_review_eligible"] = True

            for index in frontier.index:
                if index == chosen_index:
                    continue
                if bool(near_maximum.loc[index]) and bool(majority.loc[index]):
                    result.at[index, "protocol_selection_status"] = (
                        "pareto_near_maximum_not_protocol_selected"
                    )
                    result.at[index, "protocol_selection_reason"] = (
                        "eligible_near_maximum_family_ranked_below_selected_family"
                    )
                elif not bool(near_maximum.loc[index]):
                    result.at[index, "protocol_selection_status"] = (
                        "pareto_frontier_outside_near_maximum_band"
                    )
                    result.at[index, "protocol_selection_reason"] = (
                        "more_than_one_stress_cell_below_group_maximum"
                    )
                else:
                    result.at[index, "protocol_selection_status"] = (
                        "pareto_frontier_below_majority_coverage"
                    )
                    result.at[index, "protocol_selection_reason"] = (
                        "near_maximum_but_below_frozen_majority_coverage_requirement"
                    )
        else:
            hold_pool = frontier.loc[near_maximum].copy()
            if hold_pool.empty:
                hold_pool = frontier.copy()
            ordered = protocol_lexicographic_order(hold_pool)
            for rank, index in enumerate(ordered.index, start=1):
                result.at[index, "protocol_selection_rank"] = rank
            chosen_index = ordered.index[0]
            chosen_id = str(result.at[chosen_index, "operational_family_id"])
            held_ids.append(chosen_id)
            result.at[chosen_index, "protocol_selection_status"] = (
                "protocol_hold_low_coverage"
            )
            result.at[chosen_index, "protocol_selection_reason"] = (
                "near_maximum_family_retained_for_review_but_group_maximum_is_below_"
                "majority_coverage"
            )
            result.at[chosen_index, "protocol_freeze_review_eligible"] = False

            for index in frontier.index:
                if index == chosen_index:
                    continue
                if bool(near_maximum.loc[index]):
                    result.at[index, "protocol_selection_status"] = (
                        "pareto_low_coverage_not_primary_hold"
                    )
                    result.at[index, "protocol_selection_reason"] = (
                        "low_coverage_near_maximum_family_ranked_below_primary_hold"
                    )
                else:
                    result.at[index, "protocol_selection_status"] = (
                        "pareto_frontier_outside_near_maximum_band"
                    )
                    result.at[index, "protocol_selection_reason"] = (
                        "more_than_one_stress_cell_below_group_maximum"
                    )

    pareto = result.loc[result["pareto_nondominated"].map(normalize_bool)].copy()
    selected = result.loc[
        result["operational_family_id"].astype(str).isin(selected_ids)
    ].copy()
    held = result.loc[result["operational_family_id"].astype(str).isin(held_ids)].copy()
    rejected = result.loc[~result["pareto_nondominated"].map(normalize_bool)].copy()
    rejected["rejection_reason"] = np.select(
        [
            ~rejected["operational_family_eligible"].map(normalize_bool),
            rejected["dominated_by_count"].gt(0),
        ],
        [
            "no_defensible_target_coverage_or_origin_stability_failure",
            "pareto_dominated_within_address_and_reliability_target",
        ],
        default="not_retained_under_frozen_family_contract",
    )

    pareto = pareto.sort_values(
        [
            "address_id",
            "reliability_target",
            "total_nominal_objects",
            "defensible_stress_test_coverage",
            "operational_family_id",
        ],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    selected = selected.sort_values(
        ["address_id", "reliability_target", "protocol_selection_rank", "operational_family_id"]
    ).reset_index(drop=True)
    held = held.sort_values(
        ["address_id", "reliability_target", "protocol_selection_rank", "operational_family_id"]
    ).reset_index(drop=True)
    rejected = rejected.sort_values(
        ["address_id", "reliability_target", "total_nominal_objects", "operational_family_id"]
    ).reset_index(drop=True)
    return result, pareto, selected, held, rejected


# -----------------------------------------------------------------------------
# Decision-facing outputs
# -----------------------------------------------------------------------------


def build_address_recommendations(
    obs086a_address_summary: pd.DataFrame,
    annotated: pd.DataFrame,
    pareto: pd.DataFrame,
    selected: pd.DataFrame,
    held: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, address_row in obs086a_address_summary.sort_values(
        ["address_id", "reliability_target"]
    ).iterrows():
        address_id = str(address_row["address_id"])
        target = float(address_row["reliability_target"])
        mask = (
            annotated["address_id"].astype(str).eq(address_id)
            & annotated["reliability_target"].astype(float).eq(target)
        )
        families = annotated.loc[mask]
        pareto_group = pareto.loc[
            pareto["address_id"].astype(str).eq(address_id)
            & pareto["reliability_target"].astype(float).eq(target)
        ]
        selected_group = selected.loc[
            selected["address_id"].astype(str).eq(address_id)
            & selected["reliability_target"].astype(float).eq(target)
        ]
        held_group = held.loc[
            held["address_id"].astype(str).eq(address_id)
            & held["reliability_target"].astype(float).eq(target)
        ]

        if families.empty:
            recommendation = "no_operational_family_from_obs086a_sealed_set"
            rationale = (
                "OBS-086a produced no sealed scenario-conditioned allocation for "
                "this address-target."
            )
        elif not selected_group.empty:
            recommendation = "advance_protocol_selected_family_to_preregistration_review"
            rationale = (
                "A Pareto family lies within one frozen stress cell of maximum group "
                "coverage, reaches majority coverage, and is lexicographically preferred."
            )
        elif not held_group.empty:
            recommendation = "hold_low_coverage_family_for_design_review"
            rationale = (
                "The deterministic near-maximum family is retained as a hold because "
                "the address-target maximum remains below majority stress-grid coverage."
            )
        elif pareto_group.empty:
            recommendation = "no_family_retained_under_frozen_contract"
            rationale = "No eligible nondominated family retained defensible target coverage."
        else:
            recommendation = "pareto_frontier_without_protocol_selection"
            rationale = "The frontier exists, but no family met the frozen protocol rule."

        rows.append(
            {
                **{column: address_row[column] for column in BASE_METADATA_COLUMNS},
                "reliability_target": target,
                "obs086a_sealed_scenario_cells": int(address_row["sealed_candidate_cells"]),
                "operational_family_count": len(families),
                "nondominated_family_count": len(pareto_group),
                "protocol_selected_family_count": len(selected_group),
                "protocol_held_family_count": len(held_group),
                "maximum_defensible_stress_test_coverage": (
                    float(families["defensible_stress_test_coverage"].max())
                    if not families.empty
                    else 0.0
                ),
                "minimum_total_nominal_objects_on_frontier": (
                    float(pareto_group["total_nominal_objects"].min())
                    if not pareto_group.empty
                    else np.nan
                ),
                "protocol_selected_operational_family_ids_json": canonical_json(
                    sorted(selected_group["operational_family_id"].astype(str).tolist())
                ),
                "protocol_held_operational_family_ids_json": canonical_json(
                    sorted(held_group["operational_family_id"].astype(str).tolist())
                ),
                "recommendation": recommendation,
                "recommendation_rationale": rationale,
                "scenario_coverage_semantics": (
                    "fraction of 25 frozen stress-test cells; not a probability over real campaigns"
                ),
                "entitlement_preserved": True,
            }
        )
    return pd.DataFrame(rows)


def build_protocol_freeze_table(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "operational_family_id",
                *BASE_METADATA_COLUMNS,
                "reliability_target",
                "discovery_nominal_independent_objects",
                "confirmation_nominal_independent_objects",
                "total_nominal_independent_objects",
                "protocol_freeze_status",
            ]
        )
    result = selected[
        [
            "operational_family_id",
            *BASE_METADATA_COLUMNS,
            "reliability_target",
            "discovery_nominal_k",
            "confirmation_nominal_k",
            "total_nominal_objects",
            "partition_allocation_imbalance",
            "defensible_stress_test_coverage",
            "coverage_class",
            "worst_case_paired_probability_all_scenarios",
            "median_paired_probability_all_scenarios",
            "maximum_partition_probability_gap",
            "maximum_simulator_spread_at_fixed_allocation",
            "origin_minimum_support_efficiency",
            "origin_minimum_mean_effective_clusters",
            "selection_roles_json",
            "maximum_group_defensible_cells",
            "defensible_cell_shortfall_from_group_maximum",
            "near_maximum_coverage_eligible",
            "majority_coverage_required",
            "protocol_selection_rank",
            "protocol_selection_status",
            "protocol_selection_reason",
        ]
    ].copy()
    result = result.rename(
        columns={
            "discovery_nominal_k": "discovery_nominal_independent_objects",
            "confirmation_nominal_k": "confirmation_nominal_independent_objects",
            "total_nominal_objects": "total_nominal_independent_objects",
        }
    )
    result["protocol_freeze_status"] = "protocol_selected_candidate_for_preregistration_review"
    result["acquisition_rule"] = (
        "acquire genuinely independent objects separately for discovery and confirmation"
    )
    result["confirmation_sealing_rule"] = (
        "seal confirmation protocol and object-assignment rule before confirmation-data access"
    )
    result["pooling_rule"] = "discovery and confirmation pooling is prohibited"
    result["replacement_rule"] = (
        "replacement may occur only under pre-registered non-outcome-based admissibility rules"
    )
    result["evaluation_rule"] = (
        "apply the frozen evidence contract independently in both partitions"
    )
    result["stress_axis_rule"] = (
        "delta and control_response_lambda cannot be selected, fitted, or inferred from campaign outcomes"
    )
    result["stop_rule"] = (
        "stop or continue only under pre-registered support-completeness and futility rules; "
        "never condition continuation on observed effect direction or gate passage"
    )
    result["claim_rule"] = (
        "carry entitlement unchanged; protocol success does not itself create a witness"
    )
    return result.sort_values(
        ["address_id", "reliability_target", "total_nominal_independent_objects", "operational_family_id"]
    ).reset_index(drop=True)


def build_entitlement_overlay(annotated: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        annotated.groupby(
            [
                "entitlement_status",
                "reliability_target",
                "pareto_status",
                "protocol_selection_status",
            ],
            sort=True,
        )
        .agg(
            operational_families=("operational_family_id", "size"),
            maximum_stress_test_coverage=("defensible_stress_test_coverage", "max"),
            minimum_total_nominal_objects=("total_nominal_objects", "min"),
        )
        .reset_index()
    )
    grouped["entitlement_preserved"] = True
    grouped["entitlement_interpretation"] = (
        "Pareto retention and protocol selection cannot create a witness or increase entitlement"
    )
    return grouped


def build_protocol_decision_rules() -> pd.DataFrame:
    rows = [
        (1, "lineage_or_schema_invalid", "invalidate", "Family selection cannot proceed from unverified OBS-086a artifacts."),
        (2, "partition_boundary_breached", "invalidate", "Discovery and confirmation independence is frozen."),
        (3, "allocation_outside_tested_k_grid", "invalidate", "OBS-086b does not interpolate or extrapolate support."),
        (4, "origin_candidate_material_nonmonotonicity", "reject_family", "OBS-086a holds materially nonmonotone candidates rather than sealing them."),
        (5, "zero_defensible_stress_coverage", "reject_family", "The fixed allocation does not defensibly reach the target in any tested stress cell."),
        (6, "pareto_dominated", "reject_family", "Another family is no worse on all frozen dominance objectives and strictly better on at least one."),
        (7, "pareto_nondominated", "retain_on_pareto_frontier", "The allocation is a non-redundant cost/coverage/reliability/balance trade-off."),
        (8, "within_one_cell_of_group_maximum_and_majority_coverage", "enter_protocol_selection_pool", "The family is near maximum tested coverage and reaches at least half of the frozen 25-cell grid."),
        (9, "protocol_selection_pool", "select_lexicographically", "Choose minimum total objects, minimum imbalance, highest minimum defensible probability, highest median probability, lowest partition gap, lowest simulator spread, then stable family ID."),
        (10, "group_maximum_below_majority_coverage", "hold_one_low_coverage_family", "Retain the lexicographically preferred near-maximum family as a design hold rather than advancing it."),
        (11, "partial_stress_coverage", "retain_uncertainty_warning", "Stress-test coverage is combinatorial and does not authorize selecting delta or lambda."),
        (12, "future_effective_support_shortfall", "apply_preregistered_continue_or_futility_rule", "Continuation and replacement must not depend on observed effect direction, magnitude, or passage."),
    ]
    return pd.DataFrame(
        rows,
        columns=["rule_order", "condition_id", "decision", "rationale"],
    )


# -----------------------------------------------------------------------------
# Output writing, report, and manifest
# -----------------------------------------------------------------------------


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "input_manifest": output_dir / "obs086b_input_manifest.csv",
        "scenario_evaluation": output_dir / "obs086b_scenario_allocation_evaluation.csv",
        "family_summary": output_dir / "obs086b_operational_family_summary.csv",
        "dominance": output_dir / "obs086b_family_dominance_table.csv",
        "pareto_frontier": output_dir / "obs086b_pareto_frontier.csv",
        "selected_families": output_dir / "obs086b_selected_campaign_families.csv",
        "held_families": output_dir / "obs086b_held_campaign_families.csv",
        "rejected_families": output_dir / "obs086b_rejected_campaign_families.csv",
        "address_recommendations": output_dir / "obs086b_address_recommendations.csv",
        "protocol_freeze": output_dir / "obs086b_protocol_freeze_table.csv",
        "entitlement_overlay": output_dir / "obs086b_entitlement_overlay.csv",
        "failures": output_dir / "obs086b_failures.csv",
        "report": output_dir / "obs086b_report.md",
        "manifest": output_dir / "obs086b_manifest.json",
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def failures_frame(failures: Sequence[StudyFailure]) -> pd.DataFrame:
    columns = ["stage", "scope_id", "reason", "detail", "severity"]
    return pd.DataFrame(
        [
            {
                "stage": failure.stage,
                "scope_id": failure.scope_id,
                "reason": failure.reason,
                "detail": failure.detail,
                "severity": failure.severity,
            }
            for failure in failures
        ],
        columns=columns,
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def artifact_inventory(outputs: Mapping[str, Path], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in sorted(outputs.items()):
        if name == "manifest" or not path.is_file():
            continue
        rows.append(
            {
                "artifact_path": repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_report(
    path: Path,
    state: str,
    lineage: Mapping[str, Any],
    family_summary: pd.DataFrame,
    pareto: pd.DataFrame,
    selected: pd.DataFrame,
    held: pd.DataFrame,
    rejected: pd.DataFrame,
    address_recommendations: pd.DataFrame,
    dominance: pd.DataFrame,
    protocol_rules: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    target_rows: list[dict[str, Any]] = []
    for target, group in family_summary.groupby("reliability_target", sort=True):
        target_rows.append(
            {
                "reliability_target": target,
                "operational_families": len(group),
                "nondominated_families": int(
                    group["pareto_nondominated"].map(normalize_bool).sum()
                ),
                "protocol_selected_families": int(
                    group["protocol_selection_status"]
                    .astype(str)
                    .eq("protocol_selected_for_preregistration_review")
                    .sum()
                ),
                "protocol_held_families": int(
                    group["protocol_selection_status"]
                    .astype(str)
                    .eq("protocol_hold_low_coverage")
                    .sum()
                ),
                "addresses_with_families": group["address_id"].nunique(),
                "maximum_stress_test_coverage": float(
                    group["defensible_stress_test_coverage"].max()
                ),
            }
        )
    target_summary = pd.DataFrame(target_rows)

    frontier_view_columns = [
        "operational_family_id",
        "record_id",
        "carrier",
        "entitlement_status",
        "reliability_target",
        "discovery_nominal_k",
        "confirmation_nominal_k",
        "total_nominal_objects",
        "defensible_target_reaching_scenario_cells",
        "defensible_stress_test_coverage",
        "coverage_class",
        "worst_case_paired_probability_all_scenarios",
        "minimum_paired_probability_defensible_scenarios",
        "median_paired_probability_all_scenarios",
        "partition_allocation_imbalance",
        "selection_roles_json",
        "protocol_selection_status",
    ]
    protocol_view_columns = [
        "operational_family_id",
        "record_id",
        "carrier",
        "entitlement_status",
        "reliability_target",
        "discovery_nominal_k",
        "confirmation_nominal_k",
        "total_nominal_objects",
        "defensible_target_reaching_scenario_cells",
        "maximum_group_defensible_cells",
        "defensible_cell_shortfall_from_group_maximum",
        "defensible_stress_test_coverage",
        "minimum_paired_probability_defensible_scenarios",
        "median_paired_probability_all_scenarios",
        "partition_allocation_imbalance",
        "protocol_selection_rank",
        "protocol_selection_status",
        "protocol_selection_reason",
    ]
    recommendation_view = address_recommendations[
        [
            "record_id",
            "carrier",
            "entitlement_status",
            "reliability_target",
            "operational_family_count",
            "nondominated_family_count",
            "protocol_selected_family_count",
            "protocol_held_family_count",
            "maximum_defensible_stress_test_coverage",
            "minimum_total_nominal_objects_on_frontier",
            "recommendation",
        ]
    ]

    lines = [
        "# OBS-086b — Robust Campaign Family Selection",
        "",
        "## State",
        "",
        f"`{state}`",
        "",
        (
            "OBS-086b deterministically collapses the frozen OBS-086a "
            "scenario-conditioned candidates into fixed operational allocation "
            "families, preserves their nondominated Pareto frontier, and applies a "
            "separate frozen protocol-selection rule. No new simulation, threshold "
            "modification, gate modification, interpolation, extrapolation, or "
            "observed-evidence evaluation was performed."
        ),
        "",
        "## Frozen lineage",
        "",
        f"- OBS-086a commit: `{lineage['obs086a_commit']}`",
        f"- OBS-086a manifest ID: `{lineage['obs086a_manifest_id']}`",
        f"- OBS-086a manifest SHA256: `{lineage['obs086a_manifest_sha256']}`",
        f"- OBS-086a script SHA256: `{lineage['obs086a_script_sha256']}`",
        f"- OBS-086a output artifacts validated: **{lineage['obs086a_output_artifacts_validated']}**",
        f"- Frozen partition rows: **{lineage['obs086a_partition_rows']:,}**",
        f"- Frozen paired rows: **{lineage['obs086a_paired_rows']:,}**",
        f"- Frozen sealed scenario-conditioned rows: **{lineage['obs086a_sealed_rows']:,}**",
        f"- Current repository HEAD: `{lineage['current_repo_head']}`",
        "",
        "## Operational-family contract",
        "",
        "- Family identity: address, reliability target, discovery nominal k, confirmation nominal k.",
        "- Each family is re-evaluated over all 25 frozen stress-test cells for its address and target.",
        "- Discovery and confirmation remain separate; paired probability is their minimum.",
        "- Both partition probabilities are simulator-robust minima from OBS-086a.",
        "- Materially nonmonotone scenario evaluations are holds, not defensible target coverage.",
        "- Stress-test coverage is a fraction of the frozen grid, not a probability over real campaigns.",
        "- No support value outside k = [3, 4, 5, 6, 8, 10, 12] is evaluated.",
        "",
        "## Protocol-selection contract",
        "",
        f"- Preserve every nondominated Pareto family.",
        f"- Within each address-target group, identify the maximum defensible stress-cell count.",
        f"- Enter families within **{PROTOCOL_NEAR_MAXIMUM_CELL_SHORTFALL}** cell of that maximum into the near-maximum band.",
        f"- Require defensible stress-test coverage of at least **{PROTOCOL_MINIMUM_DEFENSIBLE_COVERAGE:.2f}** for advancement.",
        "- Select one advancement family lexicographically by minimum total objects, minimum partition imbalance, highest minimum probability over defensible cells, highest median all-scenario probability, lowest partition gap, lowest simulator spread, and stable family ID.",
        "- When the group maximum remains below majority coverage, retain one deterministic low-coverage hold rather than advancing it.",
        "",
        "## Family synthesis by reliability target",
        "",
        markdown_table(target_summary),
        "",
        "## Nondominated Pareto frontier",
        "",
        markdown_table(pareto[frontier_view_columns], max_rows=100) if not pareto.empty else "_No rows._",
        "",
        "## Protocol-selected campaign families",
        "",
        markdown_table(selected[protocol_view_columns], max_rows=40) if not selected.empty else "_No rows._",
        "",
        "## Protocol holds",
        "",
        markdown_table(held[protocol_view_columns], max_rows=40) if not held.empty else "_No rows._",
        "",
        "## Address recommendations",
        "",
        markdown_table(recommendation_view, max_rows=60),
        "",
        "## Dominance and protocol result",
        "",
        f"- Pairwise dominance relations: **{len(dominance):,}**",
        f"- Nondominated Pareto families: **{len(pareto):,}**",
        f"- Protocol-selected families: **{len(selected):,}**",
        f"- Low-coverage protocol holds: **{len(held):,}**",
        f"- Rejected or dominated families: **{len(rejected):,}**",
        "",
        (
            "The all-scenario worst-case probability is retained as a global fragility "
            "diagnostic. When it is constant within a comparison group, it is not labeled "
            "as a selection anchor. The minimum probability over defensible scenarios is "
            "used only after the frozen near-maximum coverage and majority-coverage rules."
        ),
        "",
        "## Protocol decision rules",
        "",
        markdown_table(protocol_rules, max_rows=30),
        "",
        "## Output counts",
        "",
        f"- Operational families: **{len(family_summary):,}**",
        f"- Pareto families retained: **{len(pareto):,}**",
        f"- Protocol-selected families: **{len(selected):,}**",
        f"- Protocol holds: **{len(held):,}**",
        f"- Rejected families: **{len(rejected):,}**",
        f"- Address-target recommendations: **{len(address_recommendations):,}**",
        f"- Failures: **{len(failures):,}**",
        "",
        "## Interpretation boundary",
        "",
        "> OBS-086b is prospective family selection only.",
        "",
        "> Delta and control-response lambda remain uncertainty axes; they cannot be selected as campaign properties.",
        "",
        "> Pareto retention and protocol selection are distinct. Neither is observed evidence or a guarantee of passage.",
        "",
        "> A low-coverage hold is not authorized for preregistration advancement.",
        "",
        "> Discovery and confirmation may not be pooled, no frozen gate may be weakened, and claim entitlement remains unchanged.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    repo_root: Path,
    outputs: Mapping[str, Path],
    state: str,
    lineage: Mapping[str, Any],
    args: argparse.Namespace,
    scenario_evaluation: pd.DataFrame,
    family_summary: pd.DataFrame,
    pareto: pd.DataFrame,
    selected: pd.DataFrame,
    held: pd.DataFrame,
    rejected: pd.DataFrame,
    dominance: pd.DataFrame,
    address_recommendations: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, Any]:
    pareto_ids = sorted(
        pareto.get("operational_family_id", pd.Series(dtype=str)).astype(str)
    )
    selected_ids = sorted(
        selected.get("operational_family_id", pd.Series(dtype=str)).astype(str)
    )
    held_ids = sorted(
        held.get("operational_family_id", pd.Series(dtype=str)).astype(str)
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "state": state,
        "scope": (
            "deterministic artifact-only robust campaign-family Pareto retention "
            "and protocol selection"
        ),
        "claim_ceiling": (
            "prospective family selection only; no observed witness, causal attribution, "
            "simulator truth, gate modification, partition pooling, guaranteed passage, "
            "stress-axis selection, or entitlement increase"
        ),
        "frozen_lineage": dict(lineage),
        "selection_contract": {
            "cluster_grid": list(CANONICAL_CLUSTER_GRID),
            "reliability_targets": list(CANONICAL_RELIABILITY_TARGETS),
            "family_key": [
                "address_id",
                "reliability_target",
                "discovery_nominal_k",
                "confirmation_nominal_k",
            ],
            "stress_scenarios_per_address_target": EXPECTED_SCENARIOS_PER_ADDRESS_TARGET,
            "fixed_allocation_rule": (
                "evaluate each unique OBS-086a sealed allocation across all frozen stress-test cells"
            ),
            "paired_probability_rule": (
                "minimum of discovery and confirmation simulator-robust probabilities at fixed partition k"
            ),
            "nonmonotonicity_rule": (
                "materially nonmonotone scenario evaluations are held and excluded from defensible coverage"
            ),
            "dominance_rule": (
                "no_more total nominal objects; no_less defensible stress coverage; "
                "no_lower worst-case paired probability; no_more partition imbalance; "
                "at least one strict improvement"
            ),
            "dominance_epsilon": args.dominance_epsilon,
            "protocol_near_maximum_cell_shortfall": (
                PROTOCOL_NEAR_MAXIMUM_CELL_SHORTFALL
            ),
            "protocol_minimum_defensible_coverage": (
                PROTOCOL_MINIMUM_DEFENSIBLE_COVERAGE
            ),
            "protocol_selection_rule": (
                "within each address-target group preserve the Pareto frontier; enter "
                "families within one stress cell of maximum coverage; require majority "
                "coverage; choose minimum total objects, minimum imbalance, highest "
                "minimum defensible probability, highest median all-scenario probability, "
                "lowest partition gap, lowest simulator spread, then stable family ID"
            ),
            "low_coverage_hold_rule": (
                "when no near-maximum family reaches majority coverage, retain one "
                "lexicographically preferred near-maximum family as a non-advancing hold"
            ),
            "scenario_coverage_semantics": (
                "combinatorial coverage of frozen simulator stress-test cells, not a probability over real campaigns"
            ),
            "extrapolation": "prohibited",
        },
        "execution": {
            "smoke": bool(args.smoke),
            "address_limit": args.address_limit,
            "selected_addresses": int(family_summary["address_id"].nunique()),
            "scenario_evaluation_rows": len(scenario_evaluation),
            "operational_family_rows": len(family_summary),
            "dominance_relations": len(dominance),
            "pareto_family_rows": len(pareto),
            "protocol_selected_family_rows": len(selected),
            "protocol_held_family_rows": len(held),
            "rejected_family_rows": len(rejected),
            "address_recommendation_rows": len(address_recommendations),
            "failures": len(failures),
        },
        "pareto_family_set": {
            "family_count": len(pareto_ids),
            "family_ids_sha256": sha256_bytes(canonical_json(pareto_ids).encode("utf-8")),
        },
        "protocol_selected_family_set": {
            "family_count": len(selected_ids),
            "family_ids_sha256": sha256_bytes(canonical_json(selected_ids).encode("utf-8")),
        },
        "protocol_held_family_set": {
            "family_count": len(held_ids),
            "family_ids_sha256": sha256_bytes(canonical_json(held_ids).encode("utf-8")),
        },
        "output_artifacts": artifact_inventory(outputs, repo_root),
        "mandatory_statements": [
            "OBS-086a remains frozen and unchanged.",
            "Delta and control_response_lambda are uncertainty axes, not selectable campaign settings.",
            "Discovery and confirmation remain separate and may not be pooled.",
            "No interpolation or extrapolation beyond the frozen support grid was performed.",
            "Stress-test coverage is combinatorial and is not a real-world probability.",
            "The Pareto frontier is preserved separately from protocol selection.",
            "A protocol-selected family is prospective only and does not guarantee passage.",
            "A low-coverage protocol hold is not authorized for preregistration advancement.",
            "No frozen evidence gate may be weakened on the basis of family selection.",
            "OBS-086b cannot create a witness or increase claim entitlement.",
        ],
    }
    return {
        "obs086b_manifest_id": sha256_bytes(canonical_json(core).encode("utf-8")),
        **core,
    }


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    address = "synthetic_address"
    metadata = {
        "address_id": address,
        "record_id": "synthetic_record",
        "support_id": "synthetic_support",
        "relation": "synthetic_relation",
        "carrier": "synthetic_carrier",
        "entitlement_status": "fl3_entitled",
    }
    partition_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    target = 0.80

    for scenario_index in range(25):
        delta = float(scenario_index // 5)
        lambda_value = float(scenario_index % 5) / 10.0
        if scenario_index < 10:
            threshold_k = 4
        elif scenario_index < 20:
            threshold_k = 6
        else:
            threshold_k = 99

        for partition in EXPECTED_PARTITIONS:
            vector: dict[str, float] = {}
            for k in CANONICAL_CLUSTER_GRID:
                if threshold_k != 99 and k >= threshold_k:
                    probability = 0.85
                else:
                    probability = 0.40
                if partition == "confirmation":
                    probability = max(0.0, probability - 0.01)
                vector[str(k)] = probability
            partition_rows.append(
                {
                    "partition_design_id": stable_row_id(
                        "PD",
                        {
                            "address": address,
                            "target": target,
                            "delta": delta,
                            "lambda": lambda_value,
                            "partition": partition,
                        },
                    ),
                    **metadata,
                    "partition": partition,
                    "delta": delta,
                    "control_response_lambda": lambda_value,
                    "reliability_target": target,
                    "tested_cluster_grid_json": canonical_json(list(CANONICAL_CLUSTER_GRID)),
                    "robust_probability_vector_json": canonical_json(vector),
                    "simulator_spread_vector_json": canonical_json(
                        {str(k): 0.01 for k in CANONICAL_CLUSTER_GRID}
                    ),
                    "material_nonmonotone_any": False,
                    "candidate_eligible_partition": True,
                    "within_tested_support_envelope": threshold_k != 99,
                    "extrapolation_beyond_tested_k_prohibited": True,
                }
            )

    allocations = [(4, 4), (6, 6), (8, 8)]
    for allocation_index, (d_k, c_k) in enumerate(allocations):
        # One seed member is sufficient to form the operational family.  The
        # fixed-allocation evaluator then expands it over all 25 scenarios.
        scenario_index = min(allocation_index * 10, 19)
        delta = float(scenario_index // 5)
        lambda_value = float(scenario_index % 5) / 10.0
        sealed_rows.append(
            {
                "paired_design_id": stable_row_id(
                    "PP", {"allocation": allocation_index, "address": address}
                ),
                **metadata,
                "delta": delta,
                "control_response_lambda": lambda_value,
                "reliability_target": target,
                "discovery_minimum_nominal_k": d_k,
                "confirmation_minimum_nominal_k": c_k,
                "minimum_total_nominal_objects": d_k + c_k,
                "partition_allocation_imbalance": 0,
                "paired_robust_probability_at_selected_allocations": 0.84,
                "paired_robust_final_tested_probability": 0.84,
                "partition_final_probability_gap": 0.01,
                "maximum_simulator_probability_spread_across_partitions": 0.01,
                "paired_selected_support_efficiency_min": 0.30,
                "paired_selected_mean_effective_clusters_min": 2.0,
                "material_nonmonotone_any_partition": False,
                "paired_design_action": "seal_for_targeted_design_evaluation",
                "sealed_candidate_eligible": True,
                "target_global_rank": allocation_index + 1,
                "address_target_rank": allocation_index + 1,
                "sealed_candidate_status": "sealed_scenario_conditioned_candidate_for_targeted_evaluation",
            }
        )
    return pd.DataFrame(partition_rows), pd.DataFrame(sealed_rows)


def run_self_test() -> None:
    partition, sealed = synthetic_frames()
    families = build_candidate_allocations(sealed)
    evaluations = evaluate_operational_families(
        families, partition, sealed, CANONICAL_CLUSTER_GRID
    )
    summary = summarize_operational_families(families, evaluations)
    dominance = build_family_dominance_table(summary, 1e-12)
    annotated = annotate_frontier(summary, dominance)
    annotated, pareto, selected, held, rejected = build_selection_layers(
        annotated, 1e-12
    )

    assert len(families) == 3
    assert len(evaluations) == 75
    coverage = {
        (int(row["discovery_nominal_k"]), int(row["confirmation_nominal_k"])): float(
            row["defensible_stress_test_coverage"]
        )
        for _, row in summary.iterrows()
    }
    assert abs(coverage[(4, 4)] - 0.40) < 1e-12
    assert abs(coverage[(6, 6)] - 0.80) < 1e-12
    assert abs(coverage[(8, 8)] - 0.80) < 1e-12
    assert len(dominance) >= 1
    dominated_8 = rejected.loc[
        rejected["discovery_nominal_k"].astype(int).eq(8)
        & rejected["confirmation_nominal_k"].astype(int).eq(8)
    ]
    assert len(dominated_8) == 1

    pareto_allocations = {
        (int(row["discovery_nominal_k"]), int(row["confirmation_nominal_k"]))
        for _, row in pareto.iterrows()
    }
    assert pareto_allocations == {(4, 4), (6, 6)}
    selected_allocations = {
        (int(row["discovery_nominal_k"]), int(row["confirmation_nominal_k"]))
        for _, row in selected.iterrows()
    }
    assert selected_allocations == {(6, 6)}
    assert held.empty

    # Flat all-scenario worst-case values must not create a meaningless anchor.
    for value in pareto["selection_roles_json"]:
        assert "worst_case_probability_anchor" not in json.loads(value)

    # Low-coverage groups retain one deterministic hold rather than advancing.
    low_coverage = annotated.loc[
        annotated["discovery_nominal_k"].astype(int).eq(4)
        & annotated["confirmation_nominal_k"].astype(int).eq(4)
    ].copy()
    low_coverage["dominated_by_count"] = 0
    low_coverage["pareto_nondominated"] = True
    low_coverage["pareto_status"] = "nondominated_operational_family"
    _, low_pareto, low_selected, low_held, low_rejected = build_selection_layers(
        low_coverage, 1e-12
    )
    assert len(low_pareto) == 1
    assert low_selected.empty
    assert len(low_held) == 1
    assert low_rejected.empty
    assert low_held.iloc[0]["protocol_selection_status"] == "protocol_hold_low_coverage"

    # Determinism regression.
    evaluations_2 = evaluate_operational_families(
        families, partition, sealed, CANONICAL_CLUSTER_GRID
    )
    summary_2 = summarize_operational_families(families, evaluations_2)
    left = canonical_json(summary.fillna("NA").to_dict("records"))
    right = canonical_json(summary_2.fillna("NA").to_dict("records"))
    assert left == right

    # Partition discordance regression.
    modified = partition.copy()
    mask = (
        modified["partition"].eq("confirmation")
        & modified["delta"].eq(0.0)
        & modified["control_response_lambda"].eq(0.0)
    )
    for idx in modified.index[mask]:
        vector = json.loads(modified.at[idx, "robust_probability_vector_json"])
        vector["6"] = 0.10
        modified.at[idx, "robust_probability_vector_json"] = canonical_json(vector)
    discordant_eval = evaluate_operational_families(
        families, modified, sealed, CANONICAL_CLUSTER_GRID
    )
    assert (
        discordant_eval["scenario_evaluation_action"]
        == "fixed_allocation_partition_discordance"
    ).any()

    print("OBS-086b self-test passed")
    print(f"Synthetic operational families: {len(families)}")
    print(f"Synthetic scenario evaluations: {len(evaluations)}")
    print(f"Synthetic Pareto families: {len(pareto)}")
    print(f"Synthetic protocol-selected families: {len(selected)}")
    print(f"Synthetic protocol holds: {len(held)}")
    print(f"Synthetic rejected families: {len(rejected)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.dominance_epsilon < 0:
        raise ValueError("--dominance-epsilon must be nonnegative.")
    if args.address_limit is not None and args.address_limit <= 0:
        raise ValueError("--address-limit must be positive.")

    repo_root = args.repo_root.resolve()
    obs086a_dir = resolve_under_root(args.obs086a_dir, repo_root)
    obs086a_script = resolve_under_root(args.obs086a_script, repo_root)
    output_dir = resolve_under_root(args.output_dir, repo_root)

    _manifest, frames, input_manifest, lineage = validate_frozen_inputs(
        repo_root=repo_root,
        obs086a_dir=obs086a_dir,
        obs086a_script=obs086a_script,
        expected_manifest_id=args.expected_obs086a_manifest_id,
        expected_script_sha256=args.expected_obs086a_script_sha256,
        explicit_commit=args.expected_obs086a_commit,
    )

    print("OBS-086b validation complete")
    print(f"Frozen OBS-086a commit: {lineage['obs086a_commit']}")
    print(f"Frozen OBS-086a manifest: {lineage['obs086a_manifest_id']}")
    print(
        "Frozen OBS-086a artifacts validated: "
        f"{lineage['obs086a_output_artifacts_validated']}"
    )
    print(f"Frozen partition rows: {len(frames['partition_envelope']):,}")
    print(f"Frozen paired rows: {len(frames['paired_designs']):,}")
    print(f"Frozen sealed rows: {len(frames['sealed_candidates']):,}")

    if args.validate_only:
        print("Validation-only mode complete; no OBS-086b outputs written.")
        return 0

    selected_addresses = sorted(
        frames["sealed_candidates"]["address_id"].astype(str).unique()
    )
    limit = args.address_limit
    if args.smoke and limit is None:
        limit = 1
    if limit is not None:
        selected_addresses = selected_addresses[:limit]

    partition = frames["partition_envelope"].loc[
        frames["partition_envelope"]["address_id"].astype(str).isin(selected_addresses)
    ].copy()
    sealed = frames["sealed_candidates"].loc[
        frames["sealed_candidates"]["address_id"].astype(str).isin(selected_addresses)
    ].copy()
    if limit is None:
        # Preserve all six frozen addresses in the decision-facing summary,
        # including addresses for which OBS-086a produced no sealed family.
        address_summary = frames["address_summary"].copy()
    else:
        address_summary = frames["address_summary"].loc[
            frames["address_summary"]["address_id"].astype(str).isin(selected_addresses)
        ].copy()

    failures: list[StudyFailure] = []
    families = build_candidate_allocations(sealed)
    scenario_evaluation = evaluate_operational_families(
        families, partition, sealed, CANONICAL_CLUSTER_GRID
    )
    family_summary = summarize_operational_families(families, scenario_evaluation)
    dominance = build_family_dominance_table(family_summary, args.dominance_epsilon)
    annotated = annotate_frontier(family_summary, dominance)
    annotated, pareto_frontier, selected, held, rejected = build_selection_layers(
        annotated, args.dominance_epsilon
    )
    address_recommendations = build_address_recommendations(
        address_summary, annotated, pareto_frontier, selected, held
    )
    protocol_freeze = build_protocol_freeze_table(selected)
    entitlement_overlay = build_entitlement_overlay(annotated)
    protocol_rules = build_protocol_decision_rules()
    failure_table = failures_frame(failures)

    expected_scenario_rows = len(families) * EXPECTED_SCENARIOS_PER_ADDRESS_TARGET
    if len(scenario_evaluation) != expected_scenario_rows:
        raise RuntimeError(
            f"Scenario evaluation row mismatch: expected {expected_scenario_rows}, "
            f"found {len(scenario_evaluation)}."
        )
    if len(pareto_frontier) + len(rejected) != len(annotated):
        raise RuntimeError("Pareto/rejected family partition is incomplete.")
    if set(selected["operational_family_id"]).intersection(
        set(held["operational_family_id"])
    ):
        raise RuntimeError("Protocol-selected and held family sets overlap.")
    if not set(selected["operational_family_id"]).issubset(
        set(pareto_frontier["operational_family_id"])
    ):
        raise RuntimeError("Protocol-selected families must be on the Pareto frontier.")
    if not set(held["operational_family_id"]).issubset(
        set(pareto_frontier["operational_family_id"])
    ):
        raise RuntimeError("Protocol-held families must be on the Pareto frontier.")

    outputs = output_paths(output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    write_csv(input_manifest, outputs["input_manifest"])
    write_csv(scenario_evaluation, outputs["scenario_evaluation"])
    write_csv(annotated, outputs["family_summary"])
    write_csv(dominance, outputs["dominance"])
    write_csv(pareto_frontier, outputs["pareto_frontier"])
    write_csv(selected, outputs["selected_families"])
    write_csv(held, outputs["held_families"])
    write_csv(rejected, outputs["rejected_families"])
    write_csv(address_recommendations, outputs["address_recommendations"])
    write_csv(protocol_freeze, outputs["protocol_freeze"])
    write_csv(entitlement_overlay, outputs["entitlement_overlay"])
    write_csv(failure_table, outputs["failures"])

    state = "robust_campaign_family_selection_completed"
    write_report(
        outputs["report"],
        state,
        lineage,
        annotated,
        pareto_frontier,
        selected,
        held,
        rejected,
        address_recommendations,
        dominance,
        protocol_rules,
        failure_table,
    )
    output_manifest = build_manifest(
        repo_root=repo_root,
        outputs=outputs,
        state=state,
        lineage=lineage,
        args=args,
        scenario_evaluation=scenario_evaluation,
        family_summary=annotated,
        pareto=pareto_frontier,
        selected=selected,
        held=held,
        rejected=rejected,
        dominance=dominance,
        address_recommendations=address_recommendations,
        failures=failure_table,
    )
    outputs["manifest"].write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("OBS-086b execution complete")
    print(f"State: {state}")
    print(f"Manifest: {output_manifest['obs086b_manifest_id']}")
    print(f"Selected addresses: {annotated['address_id'].nunique()}")
    print(f"Operational families: {len(annotated)}")
    print(f"Scenario allocation evaluations: {len(scenario_evaluation)}")
    print(f"Pareto frontier families: {len(pareto_frontier)}")
    print(f"Protocol-selected families: {len(selected)}")
    print(f"Protocol holds: {len(held)}")
    print(f"Rejected families: {len(rejected)}")
    print(f"Failures: {len(failure_table)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"OBS-086b failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
