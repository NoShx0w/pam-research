#!/usr/bin/env python3
"""
obs086a_campaign_design_synthesis.py

OBS-086a — Campaign Design Synthesis
====================================

Purpose
-------
Convert the frozen OBS-085c/OBS-085d campaign attainability and bottleneck
artifacts into a deterministic, audit-ready prospective campaign-design
instrument.

OBS-086a performs no new simulation, no threshold fitting, no candidate search
over observed outcomes, no gate modification, and no evidence evaluation.  It
uses only the frozen OBS-085d trajectory and effective-support summaries to
construct:

1. simulator-robust, partition-specific design envelopes;
2. discovery/confirmation paired allocations;
3. a sealed scenario-conditioned candidate set inside the tested k grid;
4. address-level design summaries;
5. nominal-versus-effective support attrition summaries;
6. explicit outside-envelope and no-go tables;
7. pre-declared protocol decision rules; and
8. an entitlement-preserving overlay.

"Future design" semantics
-------------------------
A future design is a prospective allocation of genuinely independent objects
to discovery and confirmation partitions, evaluated under the frozen evidence
contract.  The OBS-085 simulator axes ``delta`` and
``control_response_lambda`` are stress-test conditions.  They are not observed
facts and are not operationally selectable properties of a real campaign.

Frozen lineage
--------------
Canonical execution requires:

* OBS-085d completion commit:
  e78160bc6f88c7edce45dee83755c8b7caea7d3f
* OBS-085d manifest ID:
  32884243ca122cf8b88a39d9511b157da02e8456dfa11eec47f7c647bd018023
* OBS-085d script SHA256:
  4ac0b63d6784388f75546893c2e74eaf09589188ad8dac82a193402cf4ddfe6d

The script validates that commit as an ancestor of HEAD, validates the frozen
OBS-085d script hash, validates the OBS-085d manifest identity, and verifies
every output artifact declared by that manifest before synthesis.

Design rules
------------
* Simulator robustness is conservative: the design probability at each tested
  k is the minimum across both qualified simulators.
* Discovery and confirmation remain separate.  A paired design is eligible
  only when both partitions independently reach the reliability target.
* Candidate nominal support is selected only from the frozen tested grid.
* No extrapolation beyond the largest tested k is performed.
* Materially nonmonotone trajectories are held for review rather than sealed.
* Entitlement status is carried through unchanged.

Interpretation ceiling
----------------------
OBS-086a is prospective design synthesis only.  It cannot create an observed
witness, establish causal attribution, validate simulator truth, guarantee
future passage, weaken a frozen gate, pool discovery with confirmation, or
increase claim entitlement.

Canonical run
-------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086a_campaign_design_synthesis.py \\
  --overwrite

Validation only
---------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086a_campaign_design_synthesis.py \\
  --validate-only

Engineering smoke run
---------------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086a_campaign_design_synthesis.py \\
  --smoke --overwrite

Self-test
---------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs086a_campaign_design_synthesis.py \\
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
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "obs086a_campaign_design_synthesis_v1"

DEFAULT_EXPECTED_OBS085D_MANIFEST_ID = (
    "32884243ca122cf8b88a39d9511b157da02e8456dfa11eec47f7c647bd018023"
)
DEFAULT_EXPECTED_OBS085D_SCRIPT_SHA256 = (
    "4ac0b63d6784388f75546893c2e74eaf09589188ad8dac82a193402cf4ddfe6d"
)
DEFAULT_OBS085D_COMMIT = "e78160bc6f88c7edce45dee83755c8b7caea7d3f"

DEFAULT_OBS085D_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085d_campaign_bottleneck_localization"
)
DEFAULT_OBS085D_SCRIPT = Path(
    "experiments/studies/obs085d_campaign_bottleneck_localization.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs086_campaign_design/"
    "obs086a_campaign_design_synthesis"
)

CANONICAL_RELIABILITY_TARGETS = (0.50, 0.80, 0.90)
CANONICAL_CLUSTER_GRID = (3, 4, 5, 6, 8, 10, 12)
EXPECTED_ADDRESS_COUNT = 6
EXPECTED_PARTITIONS = ("confirmation", "discovery")
EXPECTED_SIMULATORS = (
    "joint_gaussian_regularized_cluster",
    "joint_wild_cluster_rademacher",
)
EXPECTED_TRAJECTORY_ROWS = 600
EXPECTED_EFFECTIVE_SUPPORT_CELL_ROWS = 4_200
EXPECTED_SCENARIOS_PER_ADDRESS_PARTITION_SIMULATOR = 25
EXPECTED_OBS085D_OUTPUT_ARTIFACTS = 15

DEFAULT_HIGH_ATTRITION_EFFICIENCY_THRESHOLD = 1.0 / 3.0
DEFAULT_MODERATE_ATTRITION_EFFICIENCY_THRESHOLD = 0.50

BASE_METADATA_COLUMNS = [
    "address_id",
    "record_id",
    "support_id",
    "relation",
    "carrier",
    "entitlement_status",
]

SCENARIO_COLUMNS = ["delta", "control_response_lambda"]

TRAJECTORY_REQUIRED_COLUMNS = {
    *BASE_METADATA_COLUMNS,
    "partition",
    "simulator_id",
    "failure_predicate",
    "trajectory_class",
    "probability_shape",
    "empirically_passable",
    "maximum_gate_passage_probability",
    "final_tested_gate_passage_probability",
    "material_nonmonotone",
    "final_mean_effective_cluster_count",
    *SCENARIO_COLUMNS,
}

EFFECTIVE_SUPPORT_REQUIRED_COLUMNS = {
    *BASE_METADATA_COLUMNS,
    "aggregation_level",
    "partition",
    "simulator_id",
    "failure_predicate",
    "prospective_cluster_count",
    "nominal_cluster_count",
    "mean_effective_cluster_count",
    "nominal_support_efficiency",
    "probability_effective_k_at_least_4",
    "probability_effective_k_at_least_6",
    "probability_effective_k_at_least_8",
    *SCENARIO_COLUMNS,
}

PARTITION_DESIGN_KEY = [
    *BASE_METADATA_COLUMNS,
    "partition",
    *SCENARIO_COLUMNS,
    "reliability_target",
]

PAIRED_DESIGN_KEY = [
    *BASE_METADATA_COLUMNS,
    *SCENARIO_COLUMNS,
    "reliability_target",
]


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
            "OBS-086a: deterministic prospective campaign-design synthesis "
            "from frozen OBS-085d artifacts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--obs085d-dir",
        type=Path,
        default=DEFAULT_OBS085D_DIR,
        help="Frozen OBS-085d output directory.",
    )
    parser.add_argument(
        "--obs085d-script",
        type=Path,
        default=DEFAULT_OBS085D_SCRIPT,
        help="Frozen OBS-085d study script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="OBS-086a output directory.",
    )
    parser.add_argument(
        "--expected-obs085d-manifest-id",
        default=DEFAULT_EXPECTED_OBS085D_MANIFEST_ID,
        help="Required frozen OBS-085d manifest identity.",
    )
    parser.add_argument(
        "--expected-obs085d-script-sha256",
        default=DEFAULT_EXPECTED_OBS085D_SCRIPT_SHA256,
        help="Required frozen OBS-085d script SHA256.",
    )
    parser.add_argument(
        "--obs085d-commit",
        default=DEFAULT_OBS085D_COMMIT,
        help="Required OBS-085d completion commit ancestor.",
    )
    parser.add_argument(
        "--reliability-targets",
        default="0.50,0.80,0.90",
        help="Comma-separated reliability targets in (0,1].",
    )
    parser.add_argument(
        "--high-attrition-efficiency-threshold",
        type=float,
        default=DEFAULT_HIGH_ATTRITION_EFFICIENCY_THRESHOLD,
        help=(
            "Support efficiency below this value is labeled high attrition. "
            "Default: 1/3."
        ),
    )
    parser.add_argument(
        "--moderate-attrition-efficiency-threshold",
        type=float,
        default=DEFAULT_MODERATE_ATTRITION_EFFICIENCY_THRESHOLD,
        help=(
            "Support efficiency below this value, but not below the high "
            "threshold, is labeled moderate attrition. Default: 0.50."
        ),
    )
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help="Engineering-only limit on sorted address IDs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Engineering smoke mode; defaults to one address.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen lineage and schemas, then exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic synthetic regressions and exit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_under_root(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_reliability_targets(value: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        number = float(token)
        if not (0.0 < number <= 1.0):
            raise ValueError(
                f"Reliability target must be in (0, 1], received {number!r}."
            )
        values.append(number)
    if not values:
        raise ValueError("At least one reliability target is required.")
    return tuple(sorted(set(values)))


def probability_column(k: int) -> str:
    return f"probability_k{k:02d}"


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot normalize boolean value {value!r}.")


def optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def first_k_reaching(
    cluster_grid: Sequence[int],
    probabilities: Sequence[float],
    target: float,
) -> int | None:
    for k, probability in zip(cluster_grid, probabilities):
        if float(probability) >= float(target):
            return int(k)
    return None


def first_k_positive(
    cluster_grid: Sequence[int],
    probabilities: Sequence[float],
) -> int | None:
    for k, probability in zip(cluster_grid, probabilities):
        if float(probability) > 0.0:
            return int(k)
    return None


def unique_text(values: Iterable[Any]) -> str:
    texts = sorted({str(value) for value in values if not pd.isna(value)})
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    return canonical_json(texts)


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()

    def render(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in headers) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append(f"_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def stable_row_id(prefix: str, payload: Mapping[str, Any], length: int = 24) -> str:
    digest = sha256_bytes(canonical_json(dict(payload)).encode("utf-8"))
    return f"{prefix}-{digest[:length]}"


def git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_commit_ancestor(repo_root: Path, commit: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Required commit is not an ancestor of HEAD: {commit}. {detail}"
        )


# -----------------------------------------------------------------------------
# Frozen-input validation
# -----------------------------------------------------------------------------


def obs085d_paths(obs085d_dir: Path) -> dict[str, Path]:
    return {
        "manifest": obs085d_dir / "obs085d_manifest.json",
        "address_profiles": obs085d_dir / "obs085d_address_design_profiles.csv",
        "trajectories": obs085d_dir / "obs085d_cell_trajectory_classification.csv",
        "stopping": obs085d_dir / "obs085d_design_stopping_table.csv",
        "effective_support": obs085d_dir / "obs085d_effective_support_summary.csv",
        "entitlement": obs085d_dir / "obs085d_entitlement_overlay.csv",
        "failures": obs085d_dir / "obs085d_failures.csv",
        "marginal": obs085d_dir / "obs085d_marginal_support_value.csv",
        "plateau": obs085d_dir / "obs085d_plateau_decomposition.csv",
    }


def validate_manifest_core(
    manifest: Mapping[str, Any],
    expected_manifest_id: str,
) -> tuple[int, ...]:
    actual_id = str(manifest.get("obs085d_manifest_id", ""))
    if actual_id != expected_manifest_id:
        raise RuntimeError(
            "OBS-085d manifest ID mismatch: "
            f"expected {expected_manifest_id}, found {actual_id or '<missing>'}."
        )
    if manifest.get("schema_version") != "obs085d_campaign_bottleneck_localization_v1":
        raise RuntimeError(
            "Unexpected OBS-085d schema version: "
            f"{manifest.get('schema_version')!r}."
        )
    if manifest.get("state") != "campaign_bottleneck_localization_completed":
        raise RuntimeError(
            "OBS-085d is not in the required completed state: "
            f"{manifest.get('state')!r}."
        )
    artifacts = manifest.get("output_artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("OBS-085d manifest output_artifacts is missing or invalid.")
    if len(artifacts) != EXPECTED_OBS085D_OUTPUT_ARTIFACTS:
        raise RuntimeError(
            "Unexpected OBS-085d artifact count: "
            f"expected {EXPECTED_OBS085D_OUTPUT_ARTIFACTS}, found {len(artifacts)}."
        )
    contract = manifest.get("analysis_contract", {})
    cluster_grid = tuple(int(value) for value in contract.get("cluster_grid", []))
    if cluster_grid != CANONICAL_CLUSTER_GRID:
        raise RuntimeError(
            f"Unexpected OBS-085d cluster grid: {cluster_grid!r}; "
            f"expected {CANONICAL_CLUSTER_GRID!r}."
        )
    return cluster_grid


def validate_declared_artifacts(
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in manifest["output_artifacts"]:
        declared_path = str(item["artifact_path"])
        path = resolve_under_root(Path(declared_path), repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen OBS-085d artifact is missing: {path}")
        actual_size = path.stat().st_size
        expected_size = int(item["size_bytes"])
        if actual_size != expected_size:
            raise RuntimeError(
                f"Frozen OBS-085d artifact size mismatch for {path}: "
                f"expected {expected_size}, found {actual_size}."
            )
        actual_sha = sha256_file(path)
        expected_sha = str(item["sha256"])
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Frozen OBS-085d artifact hash mismatch for {path}: "
                f"expected {expected_sha}, found {actual_sha}."
            )
        rows.append(
            {
                "source_study": "OBS-085d",
                "artifact_path": repo_relative(path, repo_root),
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "validation_status": "validated",
            }
        )
    return pd.DataFrame(rows)


def validate_trajectory_frame(
    frame: pd.DataFrame,
    cluster_grid: Sequence[int],
) -> None:
    required = set(TRAJECTORY_REQUIRED_COLUMNS) | {
        probability_column(k) for k in cluster_grid
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"OBS-085d trajectory table is missing columns: {missing}")

    key = ["address_id", "partition", "simulator_id", *SCENARIO_COLUMNS]
    if frame.duplicated(key).any():
        examples = frame.loc[frame.duplicated(key, keep=False), key].head(10)
        raise RuntimeError(
            "Duplicate OBS-085d trajectory keys detected:\n"
            + examples.to_string(index=False)
        )

    for k in cluster_grid:
        values = pd.to_numeric(frame[probability_column(k)], errors="coerce")
        if values.isna().any() or ((values < 0.0) | (values > 1.0)).any():
            raise RuntimeError(
                f"Invalid probability values in {probability_column(k)}."
            )

    partitions = tuple(sorted(frame["partition"].astype(str).unique()))
    simulators = tuple(sorted(frame["simulator_id"].astype(str).unique()))
    if partitions != tuple(sorted(EXPECTED_PARTITIONS)):
        raise RuntimeError(f"Unexpected partition set: {partitions!r}.")
    if simulators != tuple(sorted(EXPECTED_SIMULATORS)):
        raise RuntimeError(f"Unexpected simulator set: {simulators!r}.")

    counts = frame.groupby(["address_id", "partition", "simulator_id"]).size()
    if not (
        counts == EXPECTED_SCENARIOS_PER_ADDRESS_PARTITION_SIMULATOR
    ).all():
        raise RuntimeError(
            "Each address/partition/simulator context must contain exactly "
            f"{EXPECTED_SCENARIOS_PER_ADDRESS_PARTITION_SIMULATOR} scenario cells."
        )

    metadata_counts = frame.groupby("address_id")[BASE_METADATA_COLUMNS[1:]].nunique()
    if (metadata_counts > 1).any().any():
        raise RuntimeError("Address metadata is not invariant across trajectories.")


def validate_effective_support_frame(
    frame: pd.DataFrame,
    trajectory: pd.DataFrame,
    cluster_grid: Sequence[int],
) -> pd.DataFrame:
    missing = sorted(EFFECTIVE_SUPPORT_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"OBS-085d effective-support table is missing columns: {missing}"
        )
    cell = frame.loc[
        frame["aggregation_level"].astype(str).eq("cell")
    ].copy()
    if cell.empty:
        raise RuntimeError("OBS-085d effective-support cell rows are absent.")

    cell["prospective_cluster_count"] = pd.to_numeric(
        cell["prospective_cluster_count"], errors="raise"
    ).astype(int)
    observed_grid = tuple(
        sorted(cell["prospective_cluster_count"].dropna().unique().tolist())
    )
    if observed_grid != tuple(cluster_grid):
        raise RuntimeError(
            f"Effective-support grid mismatch: {observed_grid!r}."
        )

    key = [
        "address_id",
        "partition",
        "simulator_id",
        *SCENARIO_COLUMNS,
        "prospective_cluster_count",
    ]
    if cell.duplicated(key).any():
        raise RuntimeError("Duplicate effective-support cell keys detected.")

    expected_keys = trajectory[
        ["address_id", "partition", "simulator_id", *SCENARIO_COLUMNS]
    ].assign(_join=1).merge(
        pd.DataFrame({"prospective_cluster_count": list(cluster_grid), "_join": 1}),
        on="_join",
    ).drop(columns="_join")
    merged = expected_keys.merge(cell[key], on=key, how="left", indicator=True)
    if not merged["_merge"].eq("both").all():
        missing_rows = merged.loc[merged["_merge"].ne("both"), key].head(10)
        raise RuntimeError(
            "Effective-support rows do not cover every trajectory/k cell:\n"
            + missing_rows.to_string(index=False)
        )

    probability_columns = [
        "probability_effective_k_at_least_4",
        "probability_effective_k_at_least_6",
        "probability_effective_k_at_least_8",
    ]
    for column in probability_columns:
        values = pd.to_numeric(cell[column], errors="coerce")
        if values.isna().any() or ((values < 0.0) | (values > 1.0)).any():
            raise RuntimeError(f"Invalid effective-support probability in {column}.")
    return cell


def validate_auxiliary_frames(
    address_profiles: pd.DataFrame,
    stopping: pd.DataFrame,
    failures: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    if address_profiles.empty:
        raise RuntimeError("OBS-085d address design profiles are empty.")
    if stopping.empty:
        raise RuntimeError("OBS-085d design stopping table is empty.")
    if not failures.empty:
        raise RuntimeError(
            "OBS-085d failure ledger is non-empty; design synthesis is blocked."
        )

    execution = manifest.get("execution", {})
    expected_trajectories = int(execution.get("cell_trajectory_rows", -1))
    expected_support = int(execution.get("frozen_summary_rows_selected", -1))
    if expected_trajectories != EXPECTED_TRAJECTORY_ROWS:
        raise RuntimeError(
            f"Unexpected manifest trajectory count: {expected_trajectories}."
        )
    if expected_support != EXPECTED_EFFECTIVE_SUPPORT_CELL_ROWS:
        raise RuntimeError(
            f"Unexpected manifest summary count: {expected_support}."
        )


def validate_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    repo_root = args.repo_root.resolve()
    obs085d_dir = resolve_under_root(args.obs085d_dir, repo_root)
    obs085d_script = resolve_under_root(args.obs085d_script, repo_root)
    paths = obs085d_paths(obs085d_dir)

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required OBS-085d {name} file is missing: {path}")
    if not obs085d_script.is_file():
        raise FileNotFoundError(f"Frozen OBS-085d script is missing: {obs085d_script}")

    require_commit_ancestor(repo_root, args.obs085d_commit)
    current_head = git_head(repo_root)

    actual_script_sha = sha256_file(obs085d_script)
    if actual_script_sha != args.expected_obs085d_script_sha256:
        raise RuntimeError(
            "OBS-085d script hash mismatch: "
            f"expected {args.expected_obs085d_script_sha256}, "
            f"found {actual_script_sha}."
        )

    manifest_sha = sha256_file(paths["manifest"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    cluster_grid = validate_manifest_core(
        manifest,
        args.expected_obs085d_manifest_id,
    )
    input_inventory = validate_declared_artifacts(manifest, repo_root)

    trajectory = pd.read_csv(paths["trajectories"])
    effective_support = pd.read_csv(paths["effective_support"])
    address_profiles = pd.read_csv(paths["address_profiles"])
    stopping = pd.read_csv(paths["stopping"])
    failures = pd.read_csv(paths["failures"])

    validate_trajectory_frame(trajectory, cluster_grid)
    cell_support = validate_effective_support_frame(
        effective_support,
        trajectory,
        cluster_grid,
    )
    validate_auxiliary_frames(address_profiles, stopping, failures, manifest)

    if len(trajectory) != EXPECTED_TRAJECTORY_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAJECTORY_ROWS} trajectories, found {len(trajectory)}."
        )
    if len(cell_support) != EXPECTED_EFFECTIVE_SUPPORT_CELL_ROWS:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_EFFECTIVE_SUPPORT_CELL_ROWS} effective-support cell rows, "
            f"found {len(cell_support)}."
        )
    if trajectory["address_id"].nunique() != EXPECTED_ADDRESS_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ADDRESS_COUNT} addresses, found "
            f"{trajectory['address_id'].nunique()}."
        )

    lineage = {
        "obs085d_commit": args.obs085d_commit,
        "obs085d_manifest_id": args.expected_obs085d_manifest_id,
        "obs085d_manifest_sha256": manifest_sha,
        "obs085d_script_sha256": actual_script_sha,
        "obs085d_output_artifacts_validated": len(input_inventory),
        "obs085d_state": manifest["state"],
        "obs085c_manifest_id": manifest.get("frozen_lineage", {}).get(
            "obs085c_manifest_id", ""
        ),
        "current_repo_head": current_head,
    }

    print("OBS-086a validation complete")
    print(f"Frozen OBS-085d manifest: {args.expected_obs085d_manifest_id}")
    print(f"Frozen OBS-085d artifacts validated: {len(input_inventory)}")
    print(f"Frozen trajectory rows: {len(trajectory):,}")
    print(f"Frozen effective-support cell rows: {len(cell_support):,}")
    print(f"Cluster grid: {list(cluster_grid)}")

    return (
        manifest,
        input_inventory,
        trajectory,
        cell_support,
        lineage,
    )


# -----------------------------------------------------------------------------
# Design synthesis
# -----------------------------------------------------------------------------


def support_metrics_at_k(
    support_group: pd.DataFrame,
    k: int | None,
) -> dict[str, Any]:
    empty = {
        "selected_mean_effective_cluster_count_min": np.nan,
        "selected_mean_effective_cluster_count_mean": np.nan,
        "selected_nominal_support_efficiency_min": np.nan,
        "selected_nominal_support_efficiency_mean": np.nan,
        "selected_probability_effective_k_at_least_4_min": np.nan,
        "selected_probability_effective_k_at_least_6_min": np.nan,
        "selected_probability_effective_k_at_least_8_min": np.nan,
    }
    if k is None:
        return empty
    selected = support_group.loc[
        support_group["prospective_cluster_count"].astype(int).eq(int(k))
    ]
    if selected.empty:
        return empty
    return {
        "selected_mean_effective_cluster_count_min": float(
            selected["mean_effective_cluster_count"].min()
        ),
        "selected_mean_effective_cluster_count_mean": float(
            selected["mean_effective_cluster_count"].mean()
        ),
        "selected_nominal_support_efficiency_min": float(
            selected["nominal_support_efficiency"].min()
        ),
        "selected_nominal_support_efficiency_mean": float(
            selected["nominal_support_efficiency"].mean()
        ),
        "selected_probability_effective_k_at_least_4_min": float(
            selected["probability_effective_k_at_least_4"].min()
        ),
        "selected_probability_effective_k_at_least_6_min": float(
            selected["probability_effective_k_at_least_6"].min()
        ),
        "selected_probability_effective_k_at_least_8_min": float(
            selected["probability_effective_k_at_least_8"].min()
        ),
    }


def classify_partition_design(
    simulator_first_target_k: Sequence[int | None],
    simulator_first_positive_k: Sequence[int | None],
    material_nonmonotone_any: bool,
    robust_target_reached: bool,
) -> tuple[str, str]:
    target_count = sum(value is not None for value in simulator_first_target_k)
    positive_count = sum(value is not None for value in simulator_first_positive_k)
    simulator_count = len(simulator_first_target_k)

    if robust_target_reached:
        if material_nonmonotone_any:
            return (
                "robust_target_reached_material_nonmonotonicity",
                "hold_for_nonmonotonicity_review",
            )
        return (
            "robust_target_reached",
            "candidate_within_tested_envelope",
        )
    if target_count == simulator_count:
        return (
            "simulator_incompatible_target_timing",
            (
                "hold_for_nonmonotonicity_review"
                if material_nonmonotone_any
                else "no_go_simulator_discordance"
            ),
        )
    if 0 < target_count < simulator_count:
        return (
            "simulator_discordant_target_reach",
            "no_go_simulator_discordance",
        )
    if positive_count == simulator_count:
        return (
            "robust_passage_below_target",
            "outside_tested_reliability_envelope_no_extrapolation",
        )
    if 0 < positive_count < simulator_count:
        return (
            "simulator_discordant_passage_only",
            "no_go_simulator_discordance",
        )
    return (
        "empirically_never_passable",
        "no_go_under_frozen_contract",
    )


def build_partition_design_envelope(
    trajectory: pd.DataFrame,
    effective_support: pd.DataFrame,
    cluster_grid: Sequence[int],
    reliability_targets: Sequence[float],
) -> pd.DataFrame:
    support_index_columns = [
        "address_id",
        "partition",
        *SCENARIO_COLUMNS,
    ]
    rows: list[dict[str, Any]] = []

    group_columns = [
        *BASE_METADATA_COLUMNS,
        "partition",
        *SCENARIO_COLUMNS,
    ]
    for keys, group in trajectory.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, keys))
        group = group.sort_values("simulator_id").reset_index(drop=True)
        if len(group) != len(EXPECTED_SIMULATORS):
            raise RuntimeError(
                "Partition design group does not contain both simulators: "
                f"{metadata!r}"
            )

        probability_matrix = np.asarray(
            [
                [
                    float(row[probability_column(k)])
                    for k in cluster_grid
                ]
                for _, row in group.iterrows()
            ],
            dtype=float,
        )
        robust_vector = probability_matrix.min(axis=0)
        mean_vector = probability_matrix.mean(axis=0)
        spread_vector = probability_matrix.max(axis=0) - probability_matrix.min(axis=0)

        support_mask = np.ones(len(effective_support), dtype=bool)
        for column in support_index_columns:
            support_mask &= (
                effective_support[column].astype(str).to_numpy()
                == str(metadata[column])
            )
        support_group = effective_support.loc[support_mask].copy()
        expected_support_rows = len(EXPECTED_SIMULATORS) * len(cluster_grid)
        if len(support_group) != expected_support_rows:
            raise RuntimeError(
                "Unexpected effective-support row count for partition design group "
                f"{metadata!r}: expected {expected_support_rows}, "
                f"found {len(support_group)}."
            )

        simulator_names = group["simulator_id"].astype(str).tolist()
        material_nonmonotone_any = bool(
            group["material_nonmonotone"].map(normalize_bool).any()
        )
        probability_shapes = sorted(group["probability_shape"].astype(str).unique())
        trajectory_classes = sorted(group["trajectory_class"].astype(str).unique())

        for target in reliability_targets:
            simulator_first_target_k = [
                first_k_reaching(cluster_grid, row, target)
                for row in probability_matrix
            ]
            simulator_first_positive_k = [
                first_k_positive(cluster_grid, row)
                for row in probability_matrix
            ]
            selected_k = first_k_reaching(cluster_grid, robust_vector, target)
            first_robust_positive_k = first_k_positive(cluster_grid, robust_vector)
            design_class, design_action = classify_partition_design(
                simulator_first_target_k,
                simulator_first_positive_k,
                material_nonmonotone_any,
                selected_k is not None,
            )

            selected_index = (
                list(cluster_grid).index(selected_k)
                if selected_k is not None
                else None
            )
            selected_probability = (
                float(robust_vector[selected_index])
                if selected_index is not None
                else np.nan
            )
            selected_mean_probability = (
                float(mean_vector[selected_index])
                if selected_index is not None
                else np.nan
            )
            selected_spread = (
                float(spread_vector[selected_index])
                if selected_index is not None
                else np.nan
            )

            support_selected = support_metrics_at_k(support_group, selected_k)
            support_max = support_metrics_at_k(support_group, int(cluster_grid[-1]))
            support_max = {
                key.replace("selected_", "max_tested_"): value
                for key, value in support_max.items()
            }

            id_payload = {
                "address_id": metadata["address_id"],
                "partition": metadata["partition"],
                "delta": float(metadata["delta"]),
                "control_response_lambda": float(
                    metadata["control_response_lambda"]
                ),
                "reliability_target": float(target),
            }

            rows.append(
                {
                    "partition_design_id": stable_row_id("PD", id_payload),
                    **metadata,
                    "failure_predicate": unique_text(group["failure_predicate"]),
                    "reliability_target": float(target),
                    "scenario_axis_semantics": (
                        "simulated stress-test axes; not observed facts or "
                        "operationally selectable campaign properties"
                    ),
                    "probability_semantics": (
                        "frozen conditional gate-passage probability"
                    ),
                    "tested_cluster_grid_json": canonical_json(
                        [int(k) for k in cluster_grid]
                    ),
                    "simulators_required": len(EXPECTED_SIMULATORS),
                    "simulators_observed": len(group),
                    "simulator_ids_json": canonical_json(simulator_names),
                    "simulator_first_target_k_json": canonical_json(
                        {
                            simulator: k
                            for simulator, k in zip(
                                simulator_names,
                                simulator_first_target_k,
                            )
                        }
                    ),
                    "simulator_first_positive_k_json": canonical_json(
                        {
                            simulator: k
                            for simulator, k in zip(
                                simulator_names,
                                simulator_first_positive_k,
                            )
                        }
                    ),
                    "robust_probability_vector_json": canonical_json(
                        {
                            str(int(k)): float(value)
                            for k, value in zip(cluster_grid, robust_vector)
                        }
                    ),
                    "mean_probability_vector_json": canonical_json(
                        {
                            str(int(k)): float(value)
                            for k, value in zip(cluster_grid, mean_vector)
                        }
                    ),
                    "simulator_spread_vector_json": canonical_json(
                        {
                            str(int(k)): float(value)
                            for k, value in zip(cluster_grid, spread_vector)
                        }
                    ),
                    "minimum_nominal_k_reaching_target": selected_k,
                    "first_robust_positive_passage_k": first_robust_positive_k,
                    "target_reached_by_both_simulators": selected_k is not None,
                    "simulator_target_reach_count_any_tested_k": sum(
                        value is not None for value in simulator_first_target_k
                    ),
                    "robust_gate_passage_probability_at_selected_k": selected_probability,
                    "mean_gate_passage_probability_at_selected_k": selected_mean_probability,
                    "simulator_probability_spread_at_selected_k": selected_spread,
                    "robust_final_tested_probability": float(robust_vector[-1]),
                    "mean_final_tested_probability": float(mean_vector[-1]),
                    "simulator_probability_spread_at_max_tested_k": float(
                        spread_vector[-1]
                    ),
                    "maximum_simulator_probability_spread": float(
                        spread_vector.max()
                    ),
                    "material_nonmonotone_any": material_nonmonotone_any,
                    "probability_shapes_json": canonical_json(probability_shapes),
                    "trajectory_classes_json": canonical_json(trajectory_classes),
                    **support_selected,
                    **support_max,
                    "design_class": design_class,
                    "design_action": design_action,
                    "candidate_eligible_partition": (
                        design_action == "candidate_within_tested_envelope"
                    ),
                    "within_tested_support_envelope": selected_k is not None,
                    "extrapolation_beyond_tested_k_prohibited": True,
                    "partition_pooling_prohibited": True,
                }
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "address_id",
            "partition",
            "delta",
            "control_response_lambda",
            "reliability_target",
        ]
    ).reset_index(drop=True)


def classify_paired_design(
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> tuple[str, str, bool]:
    discovery_reached = bool(discovery["target_reached_by_both_simulators"])
    confirmation_reached = bool(confirmation["target_reached_by_both_simulators"])
    discovery_eligible = bool(discovery["candidate_eligible_partition"])
    confirmation_eligible = bool(confirmation["candidate_eligible_partition"])

    if discovery_eligible and confirmation_eligible:
        return (
            "paired_robust_candidate_within_tested_envelope",
            "seal_for_targeted_design_evaluation",
            True,
        )
    if discovery_reached and confirmation_reached:
        return (
            "paired_target_reached_with_stability_hold",
            "hold_for_nonmonotonicity_review",
            False,
        )
    if discovery_reached != confirmation_reached:
        return (
            "partition_discordant_target_reach",
            "no_go_partition_discordance",
            False,
        )

    actions = {str(discovery["design_action"]), str(confirmation["design_action"])}
    if "no_go_simulator_discordance" in actions:
        return (
            "simulator_discordance_in_at_least_one_partition",
            "no_go_simulator_discordance",
            False,
        )
    if "no_go_under_frozen_contract" in actions:
        return (
            "empirically_never_passable_in_at_least_one_partition",
            "no_go_under_frozen_contract",
            False,
        )
    return (
        "paired_passage_below_target",
        "outside_tested_reliability_envelope_no_extrapolation",
        False,
    )


def build_paired_partition_designs(
    partition_envelope: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = PAIRED_DESIGN_KEY

    for keys, group in partition_envelope.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        metadata = dict(zip(group_columns, keys))
        indexed = {
            str(row["partition"]): row
            for _, row in group.iterrows()
        }
        if set(indexed) != set(EXPECTED_PARTITIONS):
            raise RuntimeError(
                "Paired design group does not contain discovery and confirmation: "
                f"{metadata!r}"
            )

        discovery = indexed["discovery"]
        confirmation = indexed["confirmation"]
        paired_class, paired_action, eligible = classify_paired_design(
            discovery,
            confirmation,
        )

        discovery_k = optional_int(
            discovery["minimum_nominal_k_reaching_target"]
        )
        confirmation_k = optional_int(
            confirmation["minimum_nominal_k_reaching_target"]
        )
        total_k = (
            discovery_k + confirmation_k
            if discovery_k is not None and confirmation_k is not None
            else None
        )
        max_partition_k = (
            max(discovery_k, confirmation_k)
            if discovery_k is not None and confirmation_k is not None
            else None
        )
        allocation_imbalance = (
            abs(discovery_k - confirmation_k)
            if discovery_k is not None and confirmation_k is not None
            else None
        )

        selected_probabilities = [
            optional_float(
                discovery["robust_gate_passage_probability_at_selected_k"]
            ),
            optional_float(
                confirmation["robust_gate_passage_probability_at_selected_k"]
            ),
        ]
        selected_probabilities = [
            value for value in selected_probabilities if value is not None
        ]
        selected_efficiencies = [
            optional_float(discovery["selected_nominal_support_efficiency_min"]),
            optional_float(confirmation["selected_nominal_support_efficiency_min"]),
        ]
        selected_efficiencies = [
            value for value in selected_efficiencies if value is not None
        ]
        selected_effective_counts = [
            optional_float(
                discovery["selected_mean_effective_cluster_count_min"]
            ),
            optional_float(
                confirmation["selected_mean_effective_cluster_count_min"]
            ),
        ]
        selected_effective_counts = [
            value for value in selected_effective_counts if value is not None
        ]

        id_payload = {
            "address_id": metadata["address_id"],
            "delta": float(metadata["delta"]),
            "control_response_lambda": float(
                metadata["control_response_lambda"]
            ),
            "reliability_target": float(metadata["reliability_target"]),
        }

        rows.append(
            {
                "paired_design_id": stable_row_id("PP", id_payload),
                **metadata,
                "scenario_axis_semantics": (
                    "simulated stress-test axes; not observed facts or "
                    "operationally selectable campaign properties"
                ),
                "probability_semantics": (
                    "frozen conditional gate-passage probability"
                ),
                "discovery_partition_design_id": discovery["partition_design_id"],
                "confirmation_partition_design_id": confirmation[
                    "partition_design_id"
                ],
                "discovery_minimum_nominal_k": discovery_k,
                "confirmation_minimum_nominal_k": confirmation_k,
                "minimum_total_nominal_objects": total_k,
                "maximum_partition_nominal_k": max_partition_k,
                "partition_allocation_imbalance": allocation_imbalance,
                "discovery_robust_probability_at_selected_k": discovery[
                    "robust_gate_passage_probability_at_selected_k"
                ],
                "confirmation_robust_probability_at_selected_k": confirmation[
                    "robust_gate_passage_probability_at_selected_k"
                ],
                "paired_robust_probability_at_selected_allocations": (
                    min(selected_probabilities)
                    if len(selected_probabilities) == 2
                    else np.nan
                ),
                "discovery_robust_final_tested_probability": discovery[
                    "robust_final_tested_probability"
                ],
                "confirmation_robust_final_tested_probability": confirmation[
                    "robust_final_tested_probability"
                ],
                "paired_robust_final_tested_probability": min(
                    float(discovery["robust_final_tested_probability"]),
                    float(confirmation["robust_final_tested_probability"]),
                ),
                "partition_final_probability_gap": abs(
                    float(discovery["robust_final_tested_probability"])
                    - float(confirmation["robust_final_tested_probability"])
                ),
                "maximum_simulator_probability_spread_across_partitions": max(
                    float(discovery["maximum_simulator_probability_spread"]),
                    float(confirmation["maximum_simulator_probability_spread"]),
                ),
                "discovery_selected_support_efficiency_min": discovery[
                    "selected_nominal_support_efficiency_min"
                ],
                "confirmation_selected_support_efficiency_min": confirmation[
                    "selected_nominal_support_efficiency_min"
                ],
                "paired_selected_support_efficiency_min": (
                    min(selected_efficiencies)
                    if len(selected_efficiencies) == 2
                    else np.nan
                ),
                "discovery_selected_mean_effective_clusters_min": discovery[
                    "selected_mean_effective_cluster_count_min"
                ],
                "confirmation_selected_mean_effective_clusters_min": confirmation[
                    "selected_mean_effective_cluster_count_min"
                ],
                "paired_selected_mean_effective_clusters_min": (
                    min(selected_effective_counts)
                    if len(selected_effective_counts) == 2
                    else np.nan
                ),
                "discovery_design_class": discovery["design_class"],
                "confirmation_design_class": confirmation["design_class"],
                "material_nonmonotone_any_partition": bool(
                    discovery["material_nonmonotone_any"]
                    or confirmation["material_nonmonotone_any"]
                ),
                "paired_design_class": paired_class,
                "paired_design_action": paired_action,
                "sealed_candidate_eligible": eligible,
                "partition_separation_required": True,
                "pooled_evaluation_prohibited": True,
                "posthoc_object_migration_prohibited": True,
                "confirmation_protocol_sealed_before_confirmation_access": True,
                "simulator_stress_test_not_observed_evidence": True,
                "extrapolation_beyond_tested_k_prohibited": True,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "address_id",
            "delta",
            "control_response_lambda",
            "reliability_target",
        ]
    ).reset_index(drop=True)


def build_sealed_candidate_set(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    sealed = paired.loc[
        paired["sealed_candidate_eligible"].map(normalize_bool)
    ].copy()
    if sealed.empty:
        return sealed

    sealed = sealed.sort_values(
        [
            "reliability_target",
            "minimum_total_nominal_objects",
            "paired_robust_probability_at_selected_allocations",
            "paired_selected_support_efficiency_min",
            "partition_final_probability_gap",
            "address_id",
            "delta",
            "control_response_lambda",
        ],
        ascending=[True, True, False, False, True, True, True, True],
    ).reset_index(drop=True)

    sealed["target_global_rank"] = (
        sealed.groupby("reliability_target", sort=True).cumcount() + 1
    )
    sealed["address_target_rank"] = (
        sealed.groupby(
            ["address_id", "reliability_target"],
            sort=True,
        ).cumcount()
        + 1
    )
    sealed["sealed_candidate_status"] = (
        "sealed_scenario_conditioned_candidate_for_targeted_evaluation"
    )
    sealed["candidate_interpretation"] = (
        "prospective design candidate only; not observed evidence and not a "
        "guarantee of passage"
    )
    return sealed


def build_address_decision_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [*BASE_METADATA_COLUMNS, "reliability_target"]

    for keys, group in paired.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, keys))
        eligible = group.loc[
            group["sealed_candidate_eligible"].map(normalize_bool)
        ]
        scenario_cells = len(group)
        sealed_count = len(eligible)
        action_counts = group["paired_design_action"].value_counts().to_dict()

        if sealed_count == 0:
            family_status = "no_sealed_candidate_under_tested_envelope"
        elif sealed_count == scenario_cells:
            family_status = "sealed_candidate_across_all_tested_scenarios"
        else:
            family_status = "sealed_candidate_for_restricted_scenario_subset"

        candidate_ids = sorted(eligible["paired_design_id"].astype(str).tolist())
        total_k = pd.to_numeric(
            eligible["minimum_total_nominal_objects"],
            errors="coerce",
        )

        rows.append(
            {
                **metadata,
                "tested_scenario_cells": scenario_cells,
                "sealed_candidate_cells": sealed_count,
                "sealed_candidate_share": (
                    sealed_count / scenario_cells if scenario_cells else np.nan
                ),
                "partition_discordant_cells": int(
                    action_counts.get("no_go_partition_discordance", 0)
                ),
                "simulator_discordant_cells": int(
                    action_counts.get("no_go_simulator_discordance", 0)
                ),
                "outside_tested_reliability_envelope_cells": int(
                    action_counts.get(
                        "outside_tested_reliability_envelope_no_extrapolation",
                        0,
                    )
                ),
                "no_go_under_frozen_contract_cells": int(
                    action_counts.get("no_go_under_frozen_contract", 0)
                ),
                "nonmonotonicity_hold_cells": int(
                    action_counts.get("hold_for_nonmonotonicity_review", 0)
                ),
                "minimum_total_nominal_objects_among_candidates": (
                    float(total_k.min()) if not total_k.empty else np.nan
                ),
                "median_total_nominal_objects_among_candidates": (
                    float(total_k.median()) if not total_k.empty else np.nan
                ),
                "maximum_total_nominal_objects_among_candidates": (
                    float(total_k.max()) if not total_k.empty else np.nan
                ),
                "maximum_paired_robust_final_probability": float(
                    group["paired_robust_final_tested_probability"].max()
                ),
                "median_paired_robust_final_probability": float(
                    group["paired_robust_final_tested_probability"].median()
                ),
                "candidate_family_status": family_status,
                "sealed_candidate_ids_json": canonical_json(candidate_ids),
                "scenario_conditioning_warning": (
                    "candidate share is over simulator stress-test cells; "
                    "delta and lambda are not selectable campaign properties"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["address_id", "reliability_target"]
    ).reset_index(drop=True)


def attrition_class(
    efficiency: float | None,
    high_threshold: float,
    moderate_threshold: float,
) -> str:
    if efficiency is None or not math.isfinite(efficiency):
        return "not_applicable_no_candidate"
    if efficiency < high_threshold:
        return "high_attrition"
    if efficiency < moderate_threshold:
        return "moderate_attrition"
    return "lower_attrition_within_observed_envelope"


def build_support_attrition_summary(
    partition_envelope: pd.DataFrame,
    high_threshold: float,
    moderate_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        *BASE_METADATA_COLUMNS,
        "partition",
        "reliability_target",
    ]

    for keys, group in partition_envelope.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        metadata = dict(zip(group_columns, keys))
        candidates = group.loc[
            group["candidate_eligible_partition"].map(normalize_bool)
        ].copy()
        efficiencies = pd.to_numeric(
            candidates["selected_nominal_support_efficiency_min"],
            errors="coerce",
        ).dropna()
        effective_counts = pd.to_numeric(
            candidates["selected_mean_effective_cluster_count_min"],
            errors="coerce",
        ).dropna()
        selected_k = pd.to_numeric(
            candidates["minimum_nominal_k_reaching_target"],
            errors="coerce",
        ).dropna()

        minimum_efficiency = (
            float(efficiencies.min()) if not efficiencies.empty else None
        )
        classification = attrition_class(
            minimum_efficiency,
            high_threshold,
            moderate_threshold,
        )

        if classification == "high_attrition":
            protection = (
                "pre-register completeness, admissibility, and non-outcome-based "
                "replacement rules; do not assume nominal k equals effective k"
            )
        elif classification == "moderate_attrition":
            protection = (
                "pre-register support-loss monitoring and non-outcome-based "
                "replacement rules"
            )
        elif classification == "lower_attrition_within_observed_envelope":
            protection = (
                "preserve independence and frozen admissibility checks; retain "
                "support-loss monitoring"
            )
        else:
            protection = "no candidate allocation under tested envelope"

        rows.append(
            {
                **metadata,
                "candidate_scenario_cells": len(candidates),
                "selected_nominal_k_values_json": canonical_json(
                    sorted({int(value) for value in selected_k.tolist()})
                ),
                "minimum_selected_nominal_k": (
                    float(selected_k.min()) if not selected_k.empty else np.nan
                ),
                "median_selected_nominal_k": (
                    float(selected_k.median()) if not selected_k.empty else np.nan
                ),
                "maximum_selected_nominal_k": (
                    float(selected_k.max()) if not selected_k.empty else np.nan
                ),
                "minimum_selected_support_efficiency": (
                    minimum_efficiency
                    if minimum_efficiency is not None
                    else np.nan
                ),
                "median_selected_support_efficiency": (
                    float(efficiencies.median())
                    if not efficiencies.empty
                    else np.nan
                ),
                "minimum_selected_mean_effective_clusters": (
                    float(effective_counts.min())
                    if not effective_counts.empty
                    else np.nan
                ),
                "median_selected_mean_effective_clusters": (
                    float(effective_counts.median())
                    if not effective_counts.empty
                    else np.nan
                ),
                "attrition_class": classification,
                "support_protection_requirement": protection,
                "high_attrition_efficiency_threshold": high_threshold,
                "moderate_attrition_efficiency_threshold": moderate_threshold,
                "extrapolated_nominal_k": np.nan,
                "extrapolation_status": "not_performed",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["address_id", "partition", "reliability_target"]
    ).reset_index(drop=True)


def build_partition_allocation_plan(
    sealed: pd.DataFrame,
) -> pd.DataFrame:
    if sealed.empty:
        return pd.DataFrame(
            columns=[
                "paired_design_id",
                *BASE_METADATA_COLUMNS,
                *SCENARIO_COLUMNS,
                "reliability_target",
                "discovery_nominal_independent_objects",
                "confirmation_nominal_independent_objects",
                "total_nominal_independent_objects",
                "partition_protocol",
                "pooling_rule",
                "replacement_rule",
                "evaluation_rule",
                "claim_rule",
            ]
        )

    plan = sealed[
        [
            "paired_design_id",
            *BASE_METADATA_COLUMNS,
            *SCENARIO_COLUMNS,
            "reliability_target",
            "discovery_minimum_nominal_k",
            "confirmation_minimum_nominal_k",
            "minimum_total_nominal_objects",
            "discovery_selected_mean_effective_clusters_min",
            "confirmation_selected_mean_effective_clusters_min",
            "discovery_selected_support_efficiency_min",
            "confirmation_selected_support_efficiency_min",
            "paired_robust_probability_at_selected_allocations",
            "paired_selected_support_efficiency_min",
            "target_global_rank",
            "address_target_rank",
        ]
    ].copy()
    plan = plan.rename(
        columns={
            "discovery_minimum_nominal_k": (
                "discovery_nominal_independent_objects"
            ),
            "confirmation_minimum_nominal_k": (
                "confirmation_nominal_independent_objects"
            ),
            "minimum_total_nominal_objects": (
                "total_nominal_independent_objects"
            ),
        }
    )
    plan["partition_protocol"] = (
        "acquire and evaluate discovery and confirmation independently; seal "
        "the confirmation protocol before confirmation-data access"
    )
    plan["pooling_rule"] = (
        "pooling discovery and confirmation for evidence evaluation is prohibited"
    )
    plan["replacement_rule"] = (
        "any replacement rule must be pre-registered and independent of observed "
        "effect direction, magnitude, or gate passage"
    )
    plan["evaluation_rule"] = (
        "apply the frozen evidence contract separately in each partition"
    )
    plan["claim_rule"] = (
        "carry entitlement status unchanged; candidate success cannot increase "
        "claim entitlement"
    )
    plan["scenario_axis_semantics"] = (
        "delta and control_response_lambda are simulator stress-test axes, "
        "not operational campaign settings"
    )
    return plan.sort_values(
        [
            "reliability_target",
            "target_global_rank",
            "address_id",
        ]
    ).reset_index(drop=True)


def build_outside_tested_envelope(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    outside = paired.loc[
        ~paired["sealed_candidate_eligible"].map(normalize_bool)
    ].copy()
    if outside.empty:
        return outside

    next_action_map = {
        "no_go_partition_discordance": (
            "do not pool partitions; investigate acquisition or partition "
            "sensitivity before proposing a new design"
        ),
        "no_go_simulator_discordance": (
            "do not seal; simulator-robust target reach is absent"
        ),
        "outside_tested_reliability_envelope_no_extrapolation": (
            "do not infer a larger-k requirement; any grid extension requires "
            "a new pre-registered simulation study"
        ),
        "no_go_under_frozen_contract": (
            "do not pursue under the frozen tested design envelope"
        ),
        "hold_for_nonmonotonicity_review": (
            "hold; resolve materially nonmonotone behavior before sealing"
        ),
    }
    outside["required_next_action"] = outside["paired_design_action"].map(
        next_action_map
    ).fillna("manual audit required")
    outside["numeric_extrapolation_performed"] = False
    outside["claim_entitlement_change_permitted"] = False
    return outside.sort_values(
        [
            "paired_design_action",
            "reliability_target",
            "address_id",
            "delta",
            "control_response_lambda",
        ]
    ).reset_index(drop=True)


def build_protocol_decision_rules() -> pd.DataFrame:
    rows = [
        {
            "rule_order": 1,
            "condition_id": "lineage_or_schema_invalid",
            "condition": (
                "Any frozen OBS-085d manifest, hash, commit, schema, or row-count "
                "validation fails."
            ),
            "decision": "invalidate",
            "rationale": "Design synthesis cannot proceed from unverified evidence artifacts.",
        },
        {
            "rule_order": 2,
            "condition_id": "partition_boundary_breached",
            "condition": (
                "Discovery and confirmation objects are pooled, migrated post hoc, "
                "or evaluated under a confirmation protocol opened after data access."
            ),
            "decision": "invalidate",
            "rationale": "Partition independence is a frozen requirement.",
        },
        {
            "rule_order": 3,
            "condition_id": "paired_robust_target_reached",
            "condition": (
                "Both qualified simulators reach the target independently in both "
                "partitions at tested k, with no material nonmonotonicity."
            ),
            "decision": "seal_for_targeted_design_evaluation",
            "rationale": (
                "The design is inside the tested envelope under conservative "
                "simulator and partition rules."
            ),
        },
        {
            "rule_order": 4,
            "condition_id": "material_nonmonotonicity",
            "condition": (
                "The reliability target is reached but at least one frozen trajectory "
                "is materially nonmonotone."
            ),
            "decision": "hold_for_nonmonotonicity_review",
            "rationale": "A nominal-support recommendation is unstable across tested k.",
        },
        {
            "rule_order": 5,
            "condition_id": "partition_discordant_target_reach",
            "condition": "Only one partition reaches the target robustly.",
            "decision": "no_go_partition_discordance",
            "rationale": "Discovery performance cannot substitute for confirmation performance.",
        },
        {
            "rule_order": 6,
            "condition_id": "simulator_discordant_target_reach",
            "condition": "Only one qualified simulator reaches the target in a partition.",
            "decision": "no_go_simulator_discordance",
            "rationale": "The design is not robust across the qualified simulator family.",
        },
        {
            "rule_order": 7,
            "condition_id": "target_not_reached_by_max_tested_k",
            "condition": (
                "Robust passage is nonzero but the reliability target is not reached "
                "by the maximum tested k."
            ),
            "decision": "outside_tested_reliability_envelope_no_extrapolation",
            "rationale": (
                "OBS-086a does not numerically extrapolate nominal support beyond "
                "the frozen tested grid."
            ),
        },
        {
            "rule_order": 8,
            "condition_id": "empirically_never_passable",
            "condition": "At least one required partition remains non-passing.",
            "decision": "no_go_under_frozen_contract",
            "rationale": "Support expansion alone did not establish passage in the tested envelope.",
        },
        {
            "rule_order": 9,
            "condition_id": "future_effective_support_shortfall",
            "condition": (
                "A later observed campaign fails a pre-registered effective-support "
                "minimum before evidence evaluation."
            ),
            "decision": "apply_pre_registered_continue_or_futility_rule",
            "rationale": (
                "Replacement or continuation must not depend on observed effect "
                "direction, magnitude, or gate passage."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["gate_modification_permitted"] = False
    frame["partition_pooling_permitted"] = False
    frame["claim_entitlement_increase_permitted"] = False
    return frame


def build_entitlement_overlay(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        paired.groupby(
            [
                "entitlement_status",
                "reliability_target",
                "paired_design_action",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            scenario_conditioned_designs=("paired_design_id", "size"),
            addresses=("address_id", "nunique"),
            sealed_candidate_designs=(
                "sealed_candidate_eligible",
                lambda values: int(sum(normalize_bool(value) for value in values)),
            ),
            maximum_paired_robust_final_probability=(
                "paired_robust_final_tested_probability",
                "max",
            ),
            median_paired_robust_final_probability=(
                "paired_robust_final_tested_probability",
                "median",
            ),
        )
        .reset_index()
    )
    grouped["entitlement_preserved"] = True
    grouped["entitlement_interpretation"] = (
        "design eligibility does not create a witness or increase entitlement"
    )
    return grouped


# -----------------------------------------------------------------------------
# Outputs, reporting, and manifest
# -----------------------------------------------------------------------------


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "input_manifest": output_dir / "obs086a_input_manifest.csv",
        "partition_envelope": (
            output_dir / "obs086a_partition_design_envelope.csv"
        ),
        "paired_designs": (
            output_dir / "obs086a_paired_partition_designs.csv"
        ),
        "sealed_candidates": (
            output_dir / "obs086a_sealed_candidate_set.csv"
        ),
        "address_summary": (
            output_dir / "obs086a_address_decision_summary.csv"
        ),
        "support_attrition": (
            output_dir / "obs086a_support_attrition_summary.csv"
        ),
        "partition_plan": (
            output_dir / "obs086a_partition_allocation_plan.csv"
        ),
        "outside_envelope": (
            output_dir / "obs086a_outside_tested_envelope.csv"
        ),
        "protocol_rules": (
            output_dir / "obs086a_protocol_decision_rules.csv"
        ),
        "entitlement_overlay": (
            output_dir / "obs086a_entitlement_overlay.csv"
        ),
        "failures": output_dir / "obs086a_failures.csv",
        "report": output_dir / "obs086a_report.md",
        "manifest": output_dir / "obs086a_manifest.json",
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite."
            )
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


def artifact_inventory(
    outputs: Mapping[str, Path],
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in sorted(outputs.items()):
        if name == "manifest":
            continue
        if not path.is_file():
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
    cluster_grid: Sequence[int],
    reliability_targets: Sequence[float],
    partition_envelope: pd.DataFrame,
    paired: pd.DataFrame,
    sealed: pd.DataFrame,
    address_summary: pd.DataFrame,
    support_attrition: pd.DataFrame,
    entitlement: pd.DataFrame,
    protocol_rules: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    target_summary = (
        paired.groupby("reliability_target", sort=True)
        .agg(
            paired_scenario_designs=("paired_design_id", "size"),
            sealed_candidates=(
                "sealed_candidate_eligible",
                lambda values: int(sum(normalize_bool(value) for value in values)),
            ),
        )
        .reset_index()
    )
    candidate_addresses = (
        paired.loc[paired["sealed_candidate_eligible"].map(normalize_bool)]
        .groupby("reliability_target", sort=True)["address_id"]
        .nunique()
        .rename("addresses_with_candidates")
        .reset_index()
    )
    target_summary = target_summary.merge(
        candidate_addresses,
        on="reliability_target",
        how="left",
    )
    target_summary["addresses_with_candidates"] = (
        target_summary["addresses_with_candidates"].fillna(0).astype(int)
    )

    decision_summary = (
        paired.groupby(
            ["reliability_target", "paired_design_action"],
            sort=True,
        )
        .size()
        .rename("scenario_conditioned_designs")
        .reset_index()
    )

    address_view = address_summary[
        [
            "address_id",
            "record_id",
            "carrier",
            "entitlement_status",
            "reliability_target",
            "tested_scenario_cells",
            "sealed_candidate_cells",
            "sealed_candidate_share",
            "minimum_total_nominal_objects_among_candidates",
            "maximum_total_nominal_objects_among_candidates",
            "candidate_family_status",
        ]
    ]

    attrition_view = (
        support_attrition.groupby(
            ["partition", "reliability_target", "attrition_class"],
            sort=True,
        )
        .size()
        .rename("address_profiles")
        .reset_index()
    )

    entitlement_view = entitlement[
        [
            "entitlement_status",
            "reliability_target",
            "paired_design_action",
            "scenario_conditioned_designs",
            "sealed_candidate_designs",
        ]
    ]

    lines = [
        "# OBS-086a — Campaign Design Synthesis",
        "",
        "## State",
        "",
        f"`{state}`",
        "",
        (
            "OBS-086a deterministically synthesizes prospective campaign-design "
            "candidates from the frozen OBS-085d artifacts. No new simulation, "
            "threshold modification, gate modification, or observed-evidence "
            "evaluation was performed."
        ),
        "",
        "## Frozen lineage",
        "",
        f"- OBS-085d commit: `{lineage['obs085d_commit']}`",
        f"- OBS-085d manifest ID: `{lineage['obs085d_manifest_id']}`",
        f"- OBS-085d manifest SHA256: `{lineage['obs085d_manifest_sha256']}`",
        f"- OBS-085d script SHA256: `{lineage['obs085d_script_sha256']}`",
        (
            "- OBS-085d output artifacts validated: "
            f"**{lineage['obs085d_output_artifacts_validated']}**"
        ),
        f"- Current repository HEAD: `{lineage['current_repo_head']}`",
        "",
        "## Design contract",
        "",
        f"- Tested nominal-support grid: **{list(cluster_grid)}**",
        f"- Conditional gate-passage reliability targets: **{list(reliability_targets)}**",
        (
            "- Simulator robustness: minimum conditional gate-passage probability across both "
            "qualified simulators at each tested k."
        ),
        (
            "- Partition requirement: discovery and confirmation must each "
            "independently reach the target."
        ),
        "- Extrapolation beyond the tested support grid: **not performed**.",
        "- Materially nonmonotone candidates: **held, not sealed**.",
        "",
        "> `delta` and `control_response_lambda` are simulator stress-test axes. "
        "They are not observed facts and are not operationally selectable "
        "properties of a real campaign.",
        "",
        "## Candidate synthesis by reliability target",
        "",
        markdown_table(target_summary),
        "",
        "## Design decisions",
        "",
        markdown_table(decision_summary),
        "",
        "## Address-level candidate families",
        "",
        markdown_table(address_view, max_rows=60),
        "",
        "## Support attrition",
        "",
        markdown_table(attrition_view),
        "",
        (
            "Nominal support is not treated as effective support. OBS-086a carries "
            "the frozen OBS-085d support-efficiency estimates into every candidate "
            "allocation and performs no numeric extrapolation beyond k=12."
        ),
        "",
        "## Entitlement overlay",
        "",
        markdown_table(entitlement_view, max_rows=60),
        "",
        "## Protocol decision rules",
        "",
        markdown_table(
            protocol_rules[
                [
                    "rule_order",
                    "condition_id",
                    "decision",
                    "rationale",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## Output counts",
        "",
        f"- Partition-specific design rows: **{len(partition_envelope):,}**",
        f"- Paired partition design rows: **{len(paired):,}**",
        f"- Sealed scenario-conditioned candidates: **{len(sealed):,}**",
        f"- Address decision profiles: **{len(address_summary):,}**",
        f"- Failures: **{len(failures):,}**",
        "",
        "## Interpretation boundary",
        "",
        "> OBS-086a is prospective design synthesis only.",
        "",
        (
            "> A sealed candidate is not observed evidence, is not a guarantee of "
            "future passage, and does not authorize post hoc object replacement."
        ),
        "",
        (
            "> Discovery and confirmation may not be pooled, and candidate status "
            "does not justify weakening any frozen evidence gate."
        ),
        "",
        (
            "> The study cannot create an FL3 witness, establish causal attribution, "
            "validate simulator truth, or increase claim entitlement."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    repo_root: Path,
    outputs: Mapping[str, Path],
    state: str,
    lineage: Mapping[str, Any],
    cluster_grid: Sequence[int],
    reliability_targets: Sequence[float],
    args: argparse.Namespace,
    partition_envelope: pd.DataFrame,
    paired: pd.DataFrame,
    sealed: pd.DataFrame,
    address_summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, Any]:
    candidate_ids = sorted(sealed.get("paired_design_id", pd.Series(dtype=str)).astype(str))
    core = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "state": state,
        "scope": (
            "deterministic artifact-only prospective campaign-design synthesis"
        ),
        "claim_ceiling": (
            "prospective design synthesis only; no observed witness, causal "
            "attribution, simulator truth, gate modification, partition pooling, "
            "guaranteed passage, or entitlement increase"
        ),
        "frozen_lineage": dict(lineage),
        "design_contract": {
            "cluster_grid": [int(k) for k in cluster_grid],
            "reliability_targets": [float(value) for value in reliability_targets],
            "simulator_robustness_rule": (
                "minimum conditional gate-passage probability across both qualified simulators "
                "at each tested k"
            ),
            "partition_pairing_rule": (
                "discovery and confirmation must independently reach the target"
            ),
            "selected_support_rule": (
                "minimum tested k reaching the target; no interpolation or "
                "extrapolation"
            ),
            "nonmonotonicity_rule": (
                "materially nonmonotone trajectories are held rather than sealed"
            ),
            "high_attrition_efficiency_threshold": (
                args.high_attrition_efficiency_threshold
            ),
            "moderate_attrition_efficiency_threshold": (
                args.moderate_attrition_efficiency_threshold
            ),
            "probability_semantics": (
                "frozen conditional gate-passage probability"
            ),
            "scenario_axis_semantics": (
                "delta and control_response_lambda are simulator stress-test axes, "
                "not observed facts or operationally selectable campaign settings"
            ),
        },
        "execution": {
            "smoke": bool(args.smoke),
            "address_limit": args.address_limit,
            "selected_addresses": int(partition_envelope["address_id"].nunique()),
            "partition_design_rows": len(partition_envelope),
            "paired_design_rows": len(paired),
            "sealed_candidate_rows": len(sealed),
            "address_summary_rows": len(address_summary),
            "failures": len(failures),
        },
        "sealed_candidate_set": {
            "candidate_count": len(candidate_ids),
            "candidate_ids_sha256": sha256_bytes(
                canonical_json(candidate_ids).encode("utf-8")
            ),
        },
        "output_artifacts": artifact_inventory(outputs, repo_root),
        "mandatory_statements": [
            "OBS-085c and OBS-085d remain frozen and unchanged.",
            "Simulator stress-test axes are not observed or operationally selectable campaign properties.",
            "Discovery and confirmation remain separate and may not be pooled.",
            "No numeric extrapolation beyond the frozen tested support grid was performed.",
            "A sealed candidate is prospective only and does not guarantee passage.",
            "Nominal support is not equivalent to effective support.",
            "No frozen evidence gate may be weakened on the basis of this synthesis.",
            "OBS-086a cannot create a witness or increase claim entitlement.",
        ],
    }
    return {
        "obs086a_manifest_id": sha256_bytes(canonical_json(core).encode("utf-8")),
        **core,
    }


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_grid = CANONICAL_CLUSTER_GRID
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []

    scenarios = [
        # paired stable candidate at target 0.5
        (0.5, 0.0, {
            "discovery": [0.0, 0.1, 0.3, 0.55, 0.70, 0.80, 0.90],
            "confirmation": [0.0, 0.05, 0.2, 0.4, 0.60, 0.72, 0.82],
        }),
        # discovery reaches, confirmation does not
        (0.5, 0.5, {
            "discovery": [0.0, 0.1, 0.3, 0.55, 0.70, 0.80, 0.90],
            "confirmation": [0.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
        }),
        # never passable
        (0.0, 0.0, {
            "discovery": [0.0] * 7,
            "confirmation": [0.0] * 7,
        }),
    ]

    simulator_offsets = {
        EXPECTED_SIMULATORS[0]: 0.0,
        EXPECTED_SIMULATORS[1]: -0.01,
    }

    for delta, lam, partition_vectors in scenarios:
        for partition, vector in partition_vectors.items():
            for simulator, offset in simulator_offsets.items():
                adjusted = [max(0.0, min(1.0, value + offset)) for value in vector]
                row = {
                    "address_id": "synthetic_address",
                    "record_id": "synthetic_record",
                    "support_id": "synthetic_support",
                    "relation": "synthetic_relation",
                    "carrier": "synthetic_carrier",
                    "entitlement_status": "fl3_entitlement_capped",
                    "partition": partition,
                    "simulator_id": simulator,
                    "failure_predicate": "measurement_missingness_concentration",
                    "delta": delta,
                    "control_response_lambda": lam,
                    "trajectory_class": (
                        "empirically_never_passable"
                        if max(adjusted) == 0
                        else "early_passable"
                    ),
                    "probability_shape": (
                        "all_zero"
                        if max(adjusted) == 0
                        else "observed_non_decreasing"
                    ),
                    "empirically_passable": max(adjusted) > 0,
                    "maximum_gate_passage_probability": max(adjusted),
                    "final_tested_gate_passage_probability": adjusted[-1],
                    "material_nonmonotone": False,
                    "final_mean_effective_cluster_count": 4.0,
                }
                for k, probability in zip(cluster_grid, adjusted):
                    row[probability_column(k)] = probability
                rows.append(row)

                for k in cluster_grid:
                    efficiency = 0.4
                    support_rows.append(
                        {
                            "address_id": "synthetic_address",
                            "record_id": "synthetic_record",
                            "support_id": "synthetic_support",
                            "relation": "synthetic_relation",
                            "carrier": "synthetic_carrier",
                            "entitlement_status": "fl3_entitlement_capped",
                            "aggregation_level": "cell",
                            "partition": partition,
                            "simulator_id": simulator,
                            "failure_predicate": (
                                "measurement_missingness_concentration"
                            ),
                            "delta": delta,
                            "control_response_lambda": lam,
                            "prospective_cluster_count": k,
                            "nominal_cluster_count": k,
                            "mean_effective_cluster_count": k * efficiency,
                            "nominal_support_efficiency": efficiency,
                            "probability_effective_k_at_least_4": (
                                0.5 if k >= 10 else 0.0
                            ),
                            "probability_effective_k_at_least_6": (
                                0.2 if k >= 12 else 0.0
                            ),
                            "probability_effective_k_at_least_8": 0.0,
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(support_rows)


def run_self_test() -> None:
    trajectory, support = synthetic_inputs()
    targets = (0.50,)
    partition = build_partition_design_envelope(
        trajectory,
        support,
        CANONICAL_CLUSTER_GRID,
        targets,
    )
    assert len(partition) == 6, len(partition)

    paired = build_paired_partition_designs(partition)
    assert len(paired) == 3, len(paired)

    sealed = build_sealed_candidate_set(paired)
    assert len(sealed) == 1, len(sealed)
    candidate = sealed.iloc[0]
    assert int(candidate["discovery_minimum_nominal_k"]) == 6
    assert int(candidate["confirmation_minimum_nominal_k"]) == 8
    assert int(candidate["minimum_total_nominal_objects"]) == 14

    actions = set(paired["paired_design_action"].astype(str))
    assert "seal_for_targeted_design_evaluation" in actions
    assert "no_go_partition_discordance" in actions
    assert "no_go_under_frozen_contract" in actions

    address = build_address_decision_summary(paired)
    assert len(address) == 1
    assert int(address.iloc[0]["sealed_candidate_cells"]) == 1

    attrition = build_support_attrition_summary(
        partition,
        DEFAULT_HIGH_ATTRITION_EFFICIENCY_THRESHOLD,
        DEFAULT_MODERATE_ATTRITION_EFFICIENCY_THRESHOLD,
    )
    assert set(
        attrition.loc[
            attrition["candidate_scenario_cells"] > 0,
            "attrition_class",
        ]
    ) == {"moderate_attrition"}

    plan = build_partition_allocation_plan(sealed)
    assert len(plan) == 1
    outside = build_outside_tested_envelope(paired)
    assert len(outside) == 2
    rules = build_protocol_decision_rules()
    assert len(rules) >= 8

    first_hash = sha256_bytes(
        canonical_json(sorted(sealed["paired_design_id"].tolist())).encode("utf-8")
    )
    second_hash = sha256_bytes(
        canonical_json(sorted(sealed["paired_design_id"].tolist())).encode("utf-8")
    )
    assert first_hash == second_hash

    try:
        parse_reliability_targets("0,0.5")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid reliability target was not rejected.")

    print("OBS-086a self-test passed")
    print(f"Synthetic partition designs: {len(partition)}")
    print(f"Synthetic paired designs: {len(paired)}")
    print(f"Synthetic sealed candidates: {len(sealed)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    args.repo_root = args.repo_root.resolve()
    reliability_targets = parse_reliability_targets(args.reliability_targets)

    if args.smoke and args.address_limit is None:
        args.address_limit = 1
    if args.address_limit is not None and args.address_limit <= 0:
        raise ValueError("--address-limit must be positive.")
    if args.high_attrition_efficiency_threshold < 0.0:
        raise ValueError("High attrition threshold must be nonnegative.")
    if (
        args.moderate_attrition_efficiency_threshold
        < args.high_attrition_efficiency_threshold
    ):
        raise ValueError(
            "Moderate attrition threshold must be greater than or equal to "
            "the high attrition threshold."
        )
    if args.moderate_attrition_efficiency_threshold > 1.0:
        raise ValueError("Attrition efficiency thresholds may not exceed 1.")

    (
        obs085d_manifest,
        input_inventory,
        trajectory,
        effective_support,
        lineage,
    ) = validate_frozen_inputs(args)
    cluster_grid = tuple(
        int(value)
        for value in obs085d_manifest["analysis_contract"]["cluster_grid"]
    )

    if args.validate_only:
        print("OBS-086a frozen-input validation succeeded")
        return 0

    selected_addresses = sorted(trajectory["address_id"].astype(str).unique())
    if args.address_limit is not None:
        selected_addresses = selected_addresses[: args.address_limit]
        trajectory = trajectory.loc[
            trajectory["address_id"].astype(str).isin(selected_addresses)
        ].copy()
        effective_support = effective_support.loc[
            effective_support["address_id"].astype(str).isin(selected_addresses)
        ].copy()

    partition_envelope = build_partition_design_envelope(
        trajectory,
        effective_support,
        cluster_grid,
        reliability_targets,
    )
    paired = build_paired_partition_designs(partition_envelope)
    sealed = build_sealed_candidate_set(paired)
    address_summary = build_address_decision_summary(paired)
    support_attrition = build_support_attrition_summary(
        partition_envelope,
        args.high_attrition_efficiency_threshold,
        args.moderate_attrition_efficiency_threshold,
    )
    partition_plan = build_partition_allocation_plan(sealed)
    outside_envelope = build_outside_tested_envelope(paired)
    protocol_rules = build_protocol_decision_rules()
    entitlement_overlay = build_entitlement_overlay(paired)
    failures = failures_frame([])

    output_dir = resolve_under_root(args.output_dir, args.repo_root)
    prepare_output_dir(output_dir, args.overwrite)
    outputs = output_paths(output_dir)

    write_csv(input_inventory, outputs["input_manifest"])
    write_csv(partition_envelope, outputs["partition_envelope"])
    write_csv(paired, outputs["paired_designs"])
    write_csv(sealed, outputs["sealed_candidates"])
    write_csv(address_summary, outputs["address_summary"])
    write_csv(support_attrition, outputs["support_attrition"])
    write_csv(partition_plan, outputs["partition_plan"])
    write_csv(outside_envelope, outputs["outside_envelope"])
    write_csv(protocol_rules, outputs["protocol_rules"])
    write_csv(entitlement_overlay, outputs["entitlement_overlay"])
    write_csv(failures, outputs["failures"])

    canonical_contract = (
        tuple(reliability_targets) == CANONICAL_RELIABILITY_TARGETS
        and math.isclose(
            args.high_attrition_efficiency_threshold,
            DEFAULT_HIGH_ATTRITION_EFFICIENCY_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            args.moderate_attrition_efficiency_threshold,
            DEFAULT_MODERATE_ATTRITION_EFFICIENCY_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )
    if args.smoke or args.address_limit is not None:
        state = "campaign_design_synthesis_engineering_smoke_completed"
    elif not canonical_contract:
        state = "campaign_design_synthesis_noncanonical_contract_completed"
    else:
        state = "campaign_design_synthesis_completed"
    write_report(
        outputs["report"],
        state,
        lineage,
        cluster_grid,
        reliability_targets,
        partition_envelope,
        paired,
        sealed,
        address_summary,
        support_attrition,
        entitlement_overlay,
        protocol_rules,
        failures,
    )
    manifest = build_manifest(
        args.repo_root,
        outputs,
        state,
        lineage,
        cluster_grid,
        reliability_targets,
        args,
        partition_envelope,
        paired,
        sealed,
        address_summary,
        failures,
    )
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("OBS-086a execution complete")
    print(f"State: {state}")
    print(f"Manifest: {manifest['obs086a_manifest_id']}")
    print(f"Selected addresses: {trajectory['address_id'].nunique()}")
    print(f"Partition design rows: {len(partition_envelope):,}")
    print(f"Paired design rows: {len(paired):,}")
    print(f"Sealed scenario-conditioned candidates: {len(sealed):,}")
    print(f"Outside-envelope/no-go rows: {len(outside_envelope):,}")
    print(f"Failures: {len(failures):,}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"OBS-086a failed: {exc}", file=sys.stderr)
        raise
