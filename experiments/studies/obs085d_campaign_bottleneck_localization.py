#!/usr/bin/env python3
"""
obs085d_campaign_bottleneck_localization.py

OBS-085d — Campaign Bottleneck Localization
=============================================

Purpose
-------
Localize why complete evidence-gate passage in the frozen OBS-085c
prospective campaign plateaus despite increasing nominal independent-object
support.

OBS-085d is an artifact-only diagnostic study.  It performs no new simulation,
no new candidate search, no threshold fitting, no simulator qualification, and
no retrospective modification of OBS-085b or OBS-085c.  It reads the frozen
OBS-085c replicate shards and summary artifacts, validates their lineage and
hashes, and produces deterministic bottleneck-localization tables.

Frozen lineage
--------------
Canonical execution requires:

* OBS-085c study-script commit:
  f40b442dc06e9d9ae19466e01b73a2314485d1dc
* OBS-085c output-freeze commit:
  83cbdb6bfb6cb185646c178578890eb4c02a5f21
* OBS-085c manifest ID:
  3341203b8c1e0024847fa054548a3c2ad6c263f271f22c947069281f4cde00ac
* OBS-085c script SHA256:
  b3d850a2da3ce43d8d44661729b793ae4e438acb41039468d8cb70dd9221926b

The script validates both commits as ancestors of HEAD, the frozen script hash,
the manifest identity, every OBS-085c output hash declared by the manifest, the
replicate-shard schema, and replicate-versus-summary counts.

Primary diagnostics
-------------------
1. Cell trajectories across k = 3, 4, 5, 6, 8, 10, 12.
2. Leave-one-gate-out diagnostic passage and single-gate rescue counts.
3. First-failed, last-failed, and blocker-multiplicity localization.
4. Persistent non-passage decomposition after the empirically observed coverage
   plateau begins.
5. Nominal-versus-effective cluster support and support efficiency.
6. Address-level prospective design profiles.
7. Discovery/confirmation and Gaussian/wild concordance.
8. Marginal value of each prospective-support increment.
9. A diagnostic stopping table separating coverage plateau from continuing
   probability gains.

Interpretation ceiling
----------------------
A leave-one-gate-out result is a localization diagnostic, not an alternative
evidence result and not a recommendation to remove a gate.  OBS-085d cannot
create an observed witness, establish causal attribution, validate simulator
truth, reinterpret OBS-085b/085c, or increase claim entitlement.

Canonical run
-------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085d_campaign_bottleneck_localization.py \\
  --overwrite

Validation only
---------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085d_campaign_bottleneck_localization.py \\
  --validate-only

Engineering smoke run
---------------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085d_campaign_bottleneck_localization.py \\
  --smoke --overwrite

Self-test
---------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085d_campaign_bottleneck_localization.py \\
  --self-test
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "obs085d_campaign_bottleneck_localization_v1"
DEFAULT_EXPECTED_OBS085C_MANIFEST_ID = (
    "3341203b8c1e0024847fa054548a3c2ad6c263f271f22c947069281f4cde00ac"
)
DEFAULT_EXPECTED_OBS085C_SCRIPT_SHA256 = (
    "b3d850a2da3ce43d8d44661729b793ae4e438acb41039468d8cb70dd9221926b"
)
DEFAULT_OBS085C_SCRIPT_COMMIT = "f40b442dc06e9d9ae19466e01b73a2314485d1dc"
DEFAULT_OBS085C_OUTPUT_COMMIT = "83cbdb6bfb6cb185646c178578890eb4c02a5f21"
DEFAULT_OBS085C_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085c_campaign_attainability_simulation"
)
DEFAULT_OBS085C_SCRIPT = Path(
    "experiments/studies/obs085c_campaign_attainability_simulation.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085d_campaign_bottleneck_localization"
)
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_NONMONOTONE_TOLERANCE = 0.02
DEFAULT_PLATEAU_EPSILON = 0.01
DEFAULT_DOMINANT_MARGIN = 0.10
CANONICAL_CLUSTER_GRID = (3, 4, 5, 6, 8, 10, 12)
RELIABILITY_TARGETS = (0.50, 0.80, 0.90)
EXPECTED_FULL_SUMMARY_ROWS = 4_200
EXPECTED_FULL_REPLICATE_ROWS = 4_200_000
EXPECTED_ADDRESS_COUNT = 6

GATE_ORDER = (
    "support_available_pass",
    "complement_admissible_pass",
    "effect_direction_reproduced_pass",
    "target_contrast_positive_pass",
    "minimum_effect_pass",
    "cluster_uncertainty_pass",
    "raw_statistical_threshold_pass",
    "multiplicity_adjusted_threshold_pass",
    "control_adjusted_contrast_pass",
    "control_specificity_pass",
)

GATE_CATEGORY = {
    "support_available_pass": "structurally_unavailable",
    "complement_admissible_pass": "structurally_unavailable",
    "effect_direction_reproduced_pass": "directionality_limited",
    "target_contrast_positive_pass": "directionality_limited",
    "minimum_effect_pass": "effect_limited",
    "cluster_uncertainty_pass": "effective_support_limited",
    "raw_statistical_threshold_pass": "statistical_resolution_limited",
    "multiplicity_adjusted_threshold_pass": "statistical_resolution_limited",
    "control_adjusted_contrast_pass": "control_contamination_limited",
    "control_specificity_pass": "specificity_limited",
}

CELL_KEY_COLUMNS = [
    "address_id",
    "partition",
    "simulator_id",
    "prospective_cluster_count",
    "delta",
    "control_response_lambda",
]

BASE_CELL_COLUMNS = [
    "address_id",
    "record_id",
    "support_id",
    "relation",
    "carrier",
    "entitlement_status",
    "partition",
    "simulator_id",
    "failure_predicate",
    "delta",
    "control_response_lambda",
]

REPLICATE_USE_COLUMNS = [
    "scenario_id",
    "base_scenario_id",
    "address_id",
    "partition",
    "simulator_id",
    "prospective_cluster_count",
    "delta",
    "control_response_lambda",
    "replicate",
    "effective_resolution_attainable",
    "independent_cluster_count",
    "raw_permutation_p",
    "cluster_uncertainty_decision_status",
    *GATE_ORDER,
    "overall_gate_pass",
]

REQUIRED_SUMMARY_COLUMNS = {
    "scenario_id",
    "base_scenario_id",
    "address_id",
    "record_id",
    "support_id",
    "relation",
    "carrier",
    "entitlement_status",
    "partition",
    "simulator_id",
    "failure_predicate",
    "prospective_cluster_count",
    "delta",
    "control_response_lambda",
    "replicates",
    "overall_gate_pass_count",
    "conditional_gate_passage_probability",
    "empirically_passable",
    "effective_resolution_attainable_probability",
    "mean_independent_cluster_count",
    "min_independent_cluster_count",
    "max_independent_cluster_count",
    "min_raw_permutation_p",
    *{f"{gate}_count" for gate in GATE_ORDER},
    *{f"{gate}_probability" for gate in GATE_ORDER},
}


@dataclass(frozen=True)
class StudyFailure:
    stage: str
    scope_id: str
    reason: str
    detail: str = ""
    severity: str = "warning"


@dataclass
class CellAccumulator:
    n: int = 0
    overall_pass: int = 0
    gate_pass: np.ndarray = field(
        default_factory=lambda: np.zeros(len(GATE_ORDER), dtype=np.int64)
    )
    single_gate_blocker: np.ndarray = field(
        default_factory=lambda: np.zeros(len(GATE_ORDER), dtype=np.int64)
    )
    first_failed: np.ndarray = field(
        default_factory=lambda: np.zeros(len(GATE_ORDER), dtype=np.int64)
    )
    last_failed: np.ndarray = field(
        default_factory=lambda: np.zeros(len(GATE_ORDER), dtype=np.int64)
    )
    blocker_multiplicity: np.ndarray = field(
        default_factory=lambda: np.zeros(len(GATE_ORDER) + 1, dtype=np.int64)
    )
    effective_histogram: Counter[int] = field(default_factory=Counter)
    effective_resolution_attainable_count: int = 0
    raw_p_min: float = math.inf
    uncertainty_status: Counter[str] = field(default_factory=Counter)


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--obs085c-dir", type=Path, default=DEFAULT_OBS085C_DIR)
    parser.add_argument("--obs085c-script", type=Path, default=DEFAULT_OBS085C_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-obs085c-manifest-id",
        default=DEFAULT_EXPECTED_OBS085C_MANIFEST_ID,
    )
    parser.add_argument(
        "--expected-obs085c-script-sha256",
        default=DEFAULT_EXPECTED_OBS085C_SCRIPT_SHA256,
    )
    parser.add_argument(
        "--obs085c-script-commit",
        default=DEFAULT_OBS085C_SCRIPT_COMMIT,
    )
    parser.add_argument(
        "--obs085c-output-commit",
        default=DEFAULT_OBS085C_OUTPUT_COMMIT,
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--nonmonotone-tolerance",
        type=float,
        default=DEFAULT_NONMONOTONE_TOLERANCE,
    )
    parser.add_argument(
        "--plateau-epsilon",
        type=float,
        default=DEFAULT_PLATEAU_EPSILON,
    )
    parser.add_argument(
        "--dominant-margin",
        type=float,
        default=DEFAULT_DOMINANT_MARGIN,
    )
    parser.add_argument(
        "--cluster-grid",
        default=",".join(str(k) for k in CANONICAL_CLUSTER_GRID),
        help="Diagnostic subset only; canonical run uses the complete frozen grid.",
    )
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help="Deterministic address prefix; makes the run noncanonical.",
    )
    parser.add_argument("--max-report-rows", type=int, default=40)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
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
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_under_root(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_cluster_grid(value: str) -> tuple[int, ...]:
    try:
        grid = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    except ValueError as exc:
        raise ValueError(f"Invalid cluster grid: {value}") from exc
    if not grid or any(k <= 0 for k in grid):
        raise ValueError("Cluster grid must contain positive integers")
    return grid


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "", "nan", "none"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def normalize_bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool, copy=False)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.fillna(0).to_numpy(dtype=float) != 0.0
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "t": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "f": False,
        "": False,
        "nan": False,
        "none": False,
    }
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    unknown = sorted(set(lowered.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown boolean tokens in {series.name}: {unknown[:10]}")
    return lowered.map(mapping).to_numpy(dtype=bool)


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def optional_int(value: Any) -> int | None:
    number = finite_or_nan(value)
    return int(number) if math.isfinite(number) else None


def first_finite_min(values: Iterable[Any]) -> int | None:
    finite = [int(float(value)) for value in values if math.isfinite(finite_or_nan(value))]
    return min(finite) if finite else None


def safe_probability(count: int, total: int) -> float:
    return count / total if total else float("nan")


def exact_histogram_median(histogram: Mapping[int, int]) -> float:
    total = sum(histogram.values())
    if total == 0:
        return float("nan")
    targets = ((total - 1) // 2, total // 2)
    results: list[int] = []
    cumulative = 0
    for value, count in sorted(histogram.items()):
        next_cumulative = cumulative + count
        for target in targets:
            if cumulative <= target < next_cumulative:
                results.append(value)
        cumulative = next_cumulative
    return float(sum(results) / len(results)) if results else float("nan")


def pearson_or_nan(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    finite = np.isfinite(xa) & np.isfinite(ya)
    if finite.sum() < 2:
        return float("nan")
    xa = xa[finite]
    ya = ya[finite]
    if np.allclose(xa, xa[0]) or np.allclose(ya, ya[0]):
        return 1.0 if np.allclose(xa, ya) else float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).copy()
    columns = [str(column) for column in clipped.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in clipped.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    if len(frame) > max_rows:
        lines.extend(["", f"_Additional {len(frame) - max_rows:,} rows omitted._"])
    return "\n".join(lines)


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_commit_ancestor(repo_root: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Required commit is not an ancestor of HEAD: {commit}")


# -----------------------------------------------------------------------------
# Frozen-input validation
# -----------------------------------------------------------------------------


def obs085c_paths(obs085c_dir: Path) -> dict[str, Path]:
    return {
        "manifest": obs085c_dir / "obs085c_manifest.json",
        "summary": obs085c_dir / "obs085c_gate_passage_summary.csv",
        "address_manifest": obs085c_dir / "obs085c_address_manifest.csv",
        "gate_contract": obs085c_dir / "obs085c_gate_contract.csv",
        "scenario_manifest": obs085c_dir / "obs085c_scenario_manifest.csv",
        "attainability_map": obs085c_dir / "obs085c_attainability_map.csv",
        "minimum_required_clusters": obs085c_dir / "obs085c_minimum_required_clusters.csv",
        "simulator_envelope": obs085c_dir / "obs085c_simulator_envelope.csv",
        "failures": obs085c_dir / "obs085c_failures.csv",
        "replicate_dir": obs085c_dir / "replicates",
    }


def validate_manifest_core(
    manifest: Mapping[str, Any],
    expected_manifest_id: str,
) -> None:
    actual = str(manifest.get("obs085c_manifest_id", ""))
    if actual != expected_manifest_id:
        raise RuntimeError(
            f"OBS-085c manifest identity mismatch: expected {expected_manifest_id}, got {actual}"
        )
    if manifest.get("state") != "campaign_attainability_simulation_completed":
        raise RuntimeError(f"Unexpected OBS-085c state: {manifest.get('state')}")
    execution = manifest.get("execution", {})
    if int(execution.get("base_simulation_count", -1)) != 600_000:
        raise RuntimeError("OBS-085c base simulation count is not canonical")
    if int(execution.get("written_nested_replicate_rows", -1)) != EXPECTED_FULL_REPLICATE_ROWS:
        raise RuntimeError("OBS-085c replicate row count is not canonical")
    if not normalize_bool(execution.get("complete_replicate_vectors_written", False)):
        raise RuntimeError("OBS-085c complete replicate vectors were not retained")
    grid = tuple(int(value) for value in manifest.get("prospective_cluster_grid", []))
    if grid != CANONICAL_CLUSTER_GRID:
        raise RuntimeError(f"Unexpected OBS-085c prospective cluster grid: {grid}")


def validate_declared_artifacts(
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifacts = manifest.get("output_artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("OBS-085c manifest declares no output artifacts")
    for item in artifacts:
        relative = Path(str(item["artifact_path"]))
        path = resolve_under_root(relative, repo_root)
        expected_size = int(item["size_bytes"])
        expected_hash = str(item["sha256"])
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen OBS-085c artifact: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"Size mismatch for {relative}: expected {expected_size}, got {actual_size}"
            )
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Hash mismatch for {relative}: expected {expected_hash}, got {actual_hash}"
            )
        rows.append(
            {
                "artifact_role": "frozen_obs085c_output",
                "artifact_path": str(relative),
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "validation_status": "validated",
            }
        )
    return pd.DataFrame(rows).sort_values("artifact_path")


def validate_summary(summary: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SUMMARY_COLUMNS - set(summary.columns))
    if missing:
        raise RuntimeError(f"OBS-085c summary missing columns: {missing}")
    if len(summary) != EXPECTED_FULL_SUMMARY_ROWS:
        raise RuntimeError(
            f"OBS-085c summary row count mismatch: {len(summary)} != {EXPECTED_FULL_SUMMARY_ROWS}"
        )
    if summary["address_id"].nunique() != EXPECTED_ADDRESS_COUNT:
        raise RuntimeError("OBS-085c summary does not contain six frozen addresses")
    grid = tuple(sorted(summary["prospective_cluster_count"].astype(int).unique()))
    if grid != CANONICAL_CLUSTER_GRID:
        raise RuntimeError(f"OBS-085c summary cluster grid mismatch: {grid}")
    if summary.duplicated(CELL_KEY_COLUMNS).any():
        duplicate = summary.loc[summary.duplicated(CELL_KEY_COLUMNS, keep=False), CELL_KEY_COLUMNS]
        raise RuntimeError(
            "OBS-085c summary has duplicate cell rows: "
            + duplicate.head(5).to_dict("records").__repr__()
        )


def validate_replicate_headers(replicate_dir: Path, cluster_grid: Sequence[int]) -> None:
    for k in cluster_grid:
        path = replicate_dir / f"obs085c_replicates_k{k:02d}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing OBS-085c replicate shard: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n\r").split(",")
        missing = [column for column in REPLICATE_USE_COLUMNS if column not in header]
        if missing:
            raise RuntimeError(f"Replicate shard {path.name} missing columns: {missing}")


def validate_frozen_inputs(args: argparse.Namespace) -> tuple[
    dict[str, Path],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    repo_root = args.repo_root.resolve()
    obs085c_dir = resolve_under_root(args.obs085c_dir, repo_root)
    obs085c_script = resolve_under_root(args.obs085c_script, repo_root)
    paths = obs085c_paths(obs085c_dir)

    if not paths["manifest"].exists():
        raise FileNotFoundError(paths["manifest"])
    if not obs085c_script.exists():
        raise FileNotFoundError(obs085c_script)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    validate_manifest_core(manifest, args.expected_obs085c_manifest_id)

    actual_script_hash = sha256_file(obs085c_script)
    if actual_script_hash != args.expected_obs085c_script_sha256:
        raise RuntimeError(
            "OBS-085c script hash mismatch: "
            f"expected {args.expected_obs085c_script_sha256}, got {actual_script_hash}"
        )

    require_commit_ancestor(repo_root, args.obs085c_script_commit)
    require_commit_ancestor(repo_root, args.obs085c_output_commit)
    head = git_head(repo_root)

    input_manifest = validate_declared_artifacts(manifest, repo_root)
    extra = pd.DataFrame(
        [
            {
                "artifact_role": "frozen_obs085c_manifest",
                "artifact_path": repo_relative(paths["manifest"], repo_root),
                "expected_size_bytes": paths["manifest"].stat().st_size,
                "actual_size_bytes": paths["manifest"].stat().st_size,
                "expected_sha256": sha256_file(paths["manifest"]),
                "actual_sha256": sha256_file(paths["manifest"]),
                "validation_status": "validated",
            },
            {
                "artifact_role": "frozen_obs085c_script",
                "artifact_path": repo_relative(obs085c_script, repo_root),
                "expected_size_bytes": obs085c_script.stat().st_size,
                "actual_size_bytes": obs085c_script.stat().st_size,
                "expected_sha256": args.expected_obs085c_script_sha256,
                "actual_sha256": actual_script_hash,
                "validation_status": "validated",
            },
        ]
    )
    input_manifest = pd.concat([input_manifest, extra], ignore_index=True).sort_values(
        ["artifact_role", "artifact_path"]
    )

    summary = pd.read_csv(paths["summary"])
    validate_summary(summary)
    validate_replicate_headers(paths["replicate_dir"], CANONICAL_CLUSTER_GRID)

    lineage = {
        "obs085c_manifest_id": manifest["obs085c_manifest_id"],
        "obs085c_manifest_sha256": sha256_file(paths["manifest"]),
        "obs085c_script_sha256": actual_script_hash,
        "obs085c_script_commit": args.obs085c_script_commit,
        "obs085c_output_commit": args.obs085c_output_commit,
        "obs085c_output_hashes_checked": len(manifest["output_artifacts"]),
        "current_repo_head": head,
    }
    return paths, manifest, summary, input_manifest, lineage


# -----------------------------------------------------------------------------
# Streaming replicate analysis
# -----------------------------------------------------------------------------


def cell_key_from_values(values: Sequence[Any]) -> tuple[Any, ...]:
    return (
        str(values[0]),
        str(values[1]),
        str(values[2]),
        int(values[3]),
        float(values[4]),
        float(values[5]),
    )


def process_replicate_chunk(
    chunk: pd.DataFrame,
    accumulators: dict[tuple[Any, ...], CellAccumulator],
) -> int:
    for column in ("prospective_cluster_count", "replicate", "independent_cluster_count"):
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    for column in ("delta", "control_response_lambda", "raw_permutation_p"):
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    gate_matrix = np.column_stack(
        [normalize_bool_array(chunk[gate]) for gate in GATE_ORDER]
    )
    reported_overall = normalize_bool_array(chunk["overall_gate_pass"])
    computed_overall = gate_matrix.all(axis=1)
    if not np.array_equal(reported_overall, computed_overall):
        mismatches = int(np.count_nonzero(reported_overall != computed_overall))
        raise RuntimeError(f"Replicate overall-gate mismatch in {mismatches} rows")

    failures = ~gate_matrix
    failure_count = failures.sum(axis=1).astype(np.int64)
    first_failed = np.argmax(failures, axis=1)
    last_failed = len(GATE_ORDER) - 1 - np.argmax(failures[:, ::-1], axis=1)
    effective = chunk["independent_cluster_count"].fillna(0).astype(int).to_numpy()
    effective_attainable = normalize_bool_array(chunk["effective_resolution_attainable"])
    raw_p = chunk["raw_permutation_p"].to_numpy(dtype=float)
    statuses = chunk["cluster_uncertainty_decision_status"].fillna("missing").astype(str).to_numpy()

    grouped = chunk.groupby(CELL_KEY_COLUMNS, sort=False, dropna=False).indices
    for raw_key, indices in grouped.items():
        key_values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = cell_key_from_values(key_values)
        idx = np.asarray(indices, dtype=np.int64)
        acc = accumulators.setdefault(key, CellAccumulator())
        group_overall = computed_overall[idx]
        group_failures = failures[idx]
        group_failure_count = failure_count[idx]

        acc.n += len(idx)
        acc.overall_pass += int(group_overall.sum())
        acc.gate_pass += gate_matrix[idx].sum(axis=0, dtype=np.int64)
        single_mask = group_failure_count == 1
        if np.any(single_mask):
            acc.single_gate_blocker += group_failures[single_mask].sum(
                axis=0, dtype=np.int64
            )
        failed_mask = group_failure_count > 0
        if np.any(failed_mask):
            acc.first_failed += np.bincount(
                first_failed[idx][failed_mask], minlength=len(GATE_ORDER)
            )[: len(GATE_ORDER)]
            acc.last_failed += np.bincount(
                last_failed[idx][failed_mask], minlength=len(GATE_ORDER)
            )[: len(GATE_ORDER)]
        acc.blocker_multiplicity += np.bincount(
            group_failure_count, minlength=len(GATE_ORDER) + 1
        )[: len(GATE_ORDER) + 1]
        acc.effective_histogram.update(
            {
                int(value): int(count)
                for value, count in zip(
                    *np.unique(effective[idx], return_counts=True)
                )
            }
        )
        acc.effective_resolution_attainable_count += int(
            effective_attainable[idx].sum()
        )
        finite_raw = raw_p[idx][np.isfinite(raw_p[idx])]
        if finite_raw.size:
            acc.raw_p_min = min(acc.raw_p_min, float(finite_raw.min()))
        acc.uncertainty_status.update(Counter(statuses[idx].tolist()))
    return len(chunk)


def stream_replicate_shards(
    replicate_dir: Path,
    cluster_grid: Sequence[int],
    selected_addresses: set[str] | None,
    chunk_size: int,
) -> tuple[dict[tuple[Any, ...], CellAccumulator], int]:
    accumulators: dict[tuple[Any, ...], CellAccumulator] = {}
    processed = 0
    for shard_index, k in enumerate(cluster_grid, start=1):
        path = replicate_dir / f"obs085c_replicates_k{k:02d}.csv.gz"
        shard_rows = 0
        for chunk in pd.read_csv(path, usecols=REPLICATE_USE_COLUMNS, chunksize=chunk_size):
            if selected_addresses is not None:
                chunk = chunk[chunk["address_id"].astype(str).isin(selected_addresses)]
            if chunk.empty:
                continue
            cluster_values = set(
                pd.to_numeric(chunk["prospective_cluster_count"], errors="raise")
                .astype(int)
                .unique()
            )
            if cluster_values != {int(k)}:
                raise RuntimeError(
                    f"Shard {path.name} contains unexpected cluster counts: {cluster_values}"
                )
            count = process_replicate_chunk(chunk, accumulators)
            processed += count
            shard_rows += count
        print(
            f"[OBS-085d] analyzed shard {shard_index}/{len(cluster_grid)} "
            f"k={k}: rows={shard_rows:,}; cumulative={processed:,}",
            flush=True,
        )
    return accumulators, processed


def validate_accumulators_against_summary(
    accumulators: Mapping[tuple[Any, ...], CellAccumulator],
    summary: pd.DataFrame,
) -> list[StudyFailure]:
    failures: list[StudyFailure] = []
    lookup = {
        cell_key_from_values(row): values
        for row, values in zip(
            summary[CELL_KEY_COLUMNS].itertuples(index=False, name=None),
            summary.to_dict("records"),
        )
    }
    if set(accumulators) != set(lookup):
        missing = set(lookup) - set(accumulators)
        extra = set(accumulators) - set(lookup)
        failures.append(
            StudyFailure(
                "replicate_summary_validation",
                "cell_key_set",
                "cell_key_mismatch",
                f"missing={len(missing)} extra={len(extra)}",
                "fatal",
            )
        )
        return failures

    for key, acc in accumulators.items():
        expected = lookup[key]
        scope = "::".join(map(str, key))
        if acc.n != int(expected["replicates"]):
            failures.append(
                StudyFailure(
                    "replicate_summary_validation",
                    scope,
                    "replicate_count_mismatch",
                    f"{acc.n} != {expected['replicates']}",
                    "fatal",
                )
            )
        if acc.overall_pass != int(expected["overall_gate_pass_count"]):
            failures.append(
                StudyFailure(
                    "replicate_summary_validation",
                    scope,
                    "overall_pass_count_mismatch",
                    f"{acc.overall_pass} != {expected['overall_gate_pass_count']}",
                    "fatal",
                )
            )
        for index, gate in enumerate(GATE_ORDER):
            expected_count = int(expected[f"{gate}_count"])
            actual_count = int(acc.gate_pass[index])
            if actual_count != expected_count:
                failures.append(
                    StudyFailure(
                        "replicate_summary_validation",
                        scope,
                        "gate_count_mismatch",
                        f"{gate}: {actual_count} != {expected_count}",
                        "fatal",
                    )
                )
    return failures


# -----------------------------------------------------------------------------
# Cell-level deterministic diagnostics
# -----------------------------------------------------------------------------


def summary_metadata_lookup(summary: pd.DataFrame) -> dict[tuple[Any, ...], dict[str, Any]]:
    columns = [
        "scenario_id",
        "base_scenario_id",
        "address_id",
        "record_id",
        "support_id",
        "relation",
        "carrier",
        "entitlement_status",
        "partition",
        "simulator_id",
        "failure_predicate",
        "prospective_cluster_count",
        "delta",
        "control_response_lambda",
    ]
    return {
        cell_key_from_values(row[CELL_KEY_COLUMNS].tolist()): {
            column: row[column] for column in columns
        }
        for _, row in summary.iterrows()
    }


def gate_removal_diagnostics(
    accumulators: Mapping[tuple[Any, ...], CellAccumulator],
    metadata: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, acc in accumulators.items():
        base = dict(metadata[key])
        for index, gate in enumerate(GATE_ORDER):
            single = int(acc.single_gate_blocker[index])
            leave_one_out = int(acc.overall_pass + single)
            rows.append(
                {
                    **base,
                    "gate_name": gate,
                    "gate_order": index + 1,
                    "replicates": acc.n,
                    "frozen_complete_pass_count": acc.overall_pass,
                    "frozen_complete_pass_probability": safe_probability(
                        acc.overall_pass, acc.n
                    ),
                    "gate_failure_count": int(acc.n - acc.gate_pass[index]),
                    "gate_failure_probability": safe_probability(
                        int(acc.n - acc.gate_pass[index]), acc.n
                    ),
                    "single_gate_blocker_count": single,
                    "single_gate_blocker_probability": safe_probability(single, acc.n),
                    "leave_one_gate_out_pass_count": leave_one_out,
                    "leave_one_gate_out_passage_probability": safe_probability(
                        leave_one_out, acc.n
                    ),
                    "diagnostic_probability_uplift": safe_probability(single, acc.n),
                    "interpretation": (
                        "diagnostic localization only; gate remains frozen and required"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        CELL_KEY_COLUMNS + ["gate_order"]
    )


def first_failure_summary(
    accumulators: Mapping[tuple[Any, ...], CellAccumulator],
    metadata: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, acc in accumulators.items():
        base = dict(metadata[key])
        rows.append(
            {
                **base,
                "first_failure_status": "overall_pass",
                "gate_name": "none",
                "gate_order": 0,
                "replicate_count": acc.overall_pass,
                "replicate_probability": safe_probability(acc.overall_pass, acc.n),
                "replicates": acc.n,
            }
        )
        for index, gate in enumerate(GATE_ORDER):
            count = int(acc.first_failed[index])
            rows.append(
                {
                    **base,
                    "first_failure_status": "first_failed_gate",
                    "gate_name": gate,
                    "gate_order": index + 1,
                    "replicate_count": count,
                    "replicate_probability": safe_probability(count, acc.n),
                    "replicates": acc.n,
                }
            )
    return pd.DataFrame(rows).sort_values(CELL_KEY_COLUMNS + ["gate_order"])


def terminal_blocker_summary(
    accumulators: Mapping[tuple[Any, ...], CellAccumulator],
    metadata: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, acc in accumulators.items():
        base = dict(metadata[key])
        rows.append(
            {
                **base,
                "diagnostic_type": "overall_pass",
                "diagnostic_value": "none",
                "gate_order": 0,
                "replicate_count": acc.overall_pass,
                "replicate_probability": safe_probability(acc.overall_pass, acc.n),
                "replicates": acc.n,
            }
        )
        for index, gate in enumerate(GATE_ORDER):
            for diagnostic_type, vector in (
                ("single_gate_blocker", acc.single_gate_blocker),
                ("last_failed_gate", acc.last_failed),
            ):
                count = int(vector[index])
                rows.append(
                    {
                        **base,
                        "diagnostic_type": diagnostic_type,
                        "diagnostic_value": gate,
                        "gate_order": index + 1,
                        "replicate_count": count,
                        "replicate_probability": safe_probability(count, acc.n),
                        "replicates": acc.n,
                    }
                )
        for multiplicity, count in enumerate(acc.blocker_multiplicity):
            rows.append(
                {
                    **base,
                    "diagnostic_type": "failed_gate_multiplicity",
                    "diagnostic_value": str(multiplicity),
                    "gate_order": -1,
                    "replicate_count": int(count),
                    "replicate_probability": safe_probability(int(count), acc.n),
                    "replicates": acc.n,
                }
            )
    return pd.DataFrame(rows).sort_values(
        CELL_KEY_COLUMNS + ["diagnostic_type", "gate_order", "diagnostic_value"]
    )


def effective_support_cell_frame(
    accumulators: Mapping[tuple[Any, ...], CellAccumulator],
    metadata: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, acc in accumulators.items():
        base = dict(metadata[key])
        nominal_k = int(base["prospective_cluster_count"])
        total_effective = sum(value * count for value, count in acc.effective_histogram.items())
        mean_effective = total_effective / acc.n if acc.n else float("nan")
        rows.append(
            {
                **base,
                "aggregation_level": "cell",
                "replicates": acc.n,
                "nominal_cluster_count": nominal_k,
                "mean_effective_cluster_count": mean_effective,
                "median_effective_cluster_count": exact_histogram_median(
                    acc.effective_histogram
                ),
                "minimum_effective_cluster_count": (
                    min(acc.effective_histogram) if acc.effective_histogram else float("nan")
                ),
                "maximum_effective_cluster_count": (
                    max(acc.effective_histogram) if acc.effective_histogram else float("nan")
                ),
                "nominal_support_efficiency": (
                    mean_effective / nominal_k if nominal_k else float("nan")
                ),
                "probability_effective_k_at_least_4": safe_probability(
                    sum(count for value, count in acc.effective_histogram.items() if value >= 4),
                    acc.n,
                ),
                "probability_effective_k_at_least_6": safe_probability(
                    sum(count for value, count in acc.effective_histogram.items() if value >= 6),
                    acc.n,
                ),
                "probability_effective_k_at_least_8": safe_probability(
                    sum(count for value, count in acc.effective_histogram.items() if value >= 8),
                    acc.n,
                ),
                "effective_resolution_attainable_probability": safe_probability(
                    acc.effective_resolution_attainable_count, acc.n
                ),
                "minimum_observed_raw_p": (
                    acc.raw_p_min if math.isfinite(acc.raw_p_min) else float("nan")
                ),
                "effective_cluster_histogram_json": canonical_json(
                    dict(sorted(acc.effective_histogram.items()))
                ),
                "uncertainty_status_counts_json": canonical_json(
                    dict(sorted(acc.uncertainty_status.items()))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(CELL_KEY_COLUMNS)


def aggregate_effective_support(
    accumulators: Mapping[tuple[Any, ...], CellAccumulator],
    metadata: Mapping[tuple[Any, ...], Mapping[str, Any]],
    group_columns: Sequence[str],
    aggregation_level: str,
) -> pd.DataFrame:
    grouped_hist: defaultdict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    grouped_n: Counter[tuple[Any, ...]] = Counter()
    grouped_attain: Counter[tuple[Any, ...]] = Counter()
    grouped_raw_min: dict[tuple[Any, ...], float] = defaultdict(lambda: math.inf)
    for key, acc in accumulators.items():
        meta = metadata[key]
        group_key = tuple(meta[column] for column in group_columns)
        grouped_hist[group_key].update(acc.effective_histogram)
        grouped_n[group_key] += acc.n
        grouped_attain[group_key] += acc.effective_resolution_attainable_count
        grouped_raw_min[group_key] = min(grouped_raw_min[group_key], acc.raw_p_min)

    rows: list[dict[str, Any]] = []
    for group_key, histogram in grouped_hist.items():
        base = dict(zip(group_columns, group_key))
        nominal_k = int(base["prospective_cluster_count"])
        n = grouped_n[group_key]
        mean_effective = sum(value * count for value, count in histogram.items()) / n
        rows.append(
            {
                **base,
                "aggregation_level": aggregation_level,
                "replicates": n,
                "nominal_cluster_count": nominal_k,
                "mean_effective_cluster_count": mean_effective,
                "median_effective_cluster_count": exact_histogram_median(histogram),
                "minimum_effective_cluster_count": min(histogram),
                "maximum_effective_cluster_count": max(histogram),
                "nominal_support_efficiency": mean_effective / nominal_k,
                "probability_effective_k_at_least_4": safe_probability(
                    sum(count for value, count in histogram.items() if value >= 4), n
                ),
                "probability_effective_k_at_least_6": safe_probability(
                    sum(count for value, count in histogram.items() if value >= 6), n
                ),
                "probability_effective_k_at_least_8": safe_probability(
                    sum(count for value, count in histogram.items() if value >= 8), n
                ),
                "effective_resolution_attainable_probability": safe_probability(
                    grouped_attain[group_key], n
                ),
                "minimum_observed_raw_p": (
                    grouped_raw_min[group_key]
                    if math.isfinite(grouped_raw_min[group_key])
                    else float("nan")
                ),
                "effective_cluster_histogram_json": canonical_json(dict(sorted(histogram.items()))),
                "uncertainty_status_counts_json": "",
            }
        )
    return pd.DataFrame(rows)


def effective_support_summary(
    accumulators: Mapping[tuple[Any, ...], CellAccumulator],
    metadata: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> pd.DataFrame:
    cell = effective_support_cell_frame(accumulators, metadata)
    global_frame = aggregate_effective_support(
        accumulators,
        metadata,
        ["prospective_cluster_count"],
        "global",
    )
    context_frame = aggregate_effective_support(
        accumulators,
        metadata,
        ["partition", "simulator_id", "prospective_cluster_count"],
        "partition_simulator",
    )
    all_columns = sorted(set(cell.columns) | set(global_frame.columns) | set(context_frame.columns))
    frames = [frame.reindex(columns=all_columns) for frame in (cell, context_frame, global_frame)]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["aggregation_level", "partition", "simulator_id", "address_id", "prospective_cluster_count"],
        na_position="first",
    )


# -----------------------------------------------------------------------------
# Cross-k trajectories and plateau localization
# -----------------------------------------------------------------------------


def minimum_k_reaching(ordered: pd.DataFrame, target: float) -> int | None:
    reached = ordered[ordered["conditional_gate_passage_probability"].ge(target)]
    return int(reached.iloc[0]["prospective_cluster_count"]) if not reached.empty else None


def build_cell_trajectories(
    summary: pd.DataFrame,
    cluster_grid: Sequence[int],
    nonmonotone_tolerance: float,
    plateau_epsilon: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in summary.groupby(BASE_CELL_COLUMNS, dropna=False, sort=False):
        ordered = group.sort_values("prospective_cluster_count")
        grid = ordered["prospective_cluster_count"].astype(int).tolist()
        if tuple(grid) != tuple(cluster_grid):
            raise RuntimeError(f"Incomplete cluster trajectory for {key}: {grid}")
        probabilities = ordered["conditional_gate_passage_probability"].astype(float).to_numpy()
        effective_means = ordered["mean_independent_cluster_count"].astype(float).to_numpy()
        first_positive_indices = np.flatnonzero(probabilities > 0)
        first_pass_k = int(grid[int(first_positive_indices[0])]) if first_positive_indices.size else None
        if first_pass_k is None:
            trajectory_class = "empirically_never_passable"
        elif first_pass_k <= 5:
            trajectory_class = "early_passable"
        else:
            trajectory_class = "late_passable"

        diffs = np.diff(probabilities)
        material_nonmonotone = bool(np.any(diffs < -nonmonotone_tolerance))
        if np.allclose(probabilities, 0.0):
            probability_shape = "all_zero"
        elif material_nonmonotone:
            probability_shape = "materially_nonmonotone"
        elif np.all(diffs >= -1e-15):
            probability_shape = "observed_non_decreasing"
        else:
            probability_shape = "minor_nonmonotonicity_within_tolerance"

        max_probability = float(np.max(probabilities))
        final_probability = float(probabilities[-1])
        first_probability = (
            float(probabilities[int(first_positive_indices[0])])
            if first_positive_indices.size
            else 0.0
        )
        improving_without_reliability = bool(
            first_pass_k is not None
            and max_probability < RELIABILITY_TARGETS[0]
            and final_probability - first_probability >= plateau_epsilon
        )
        late_increment = float(probabilities[-1] - probabilities[-2])
        late_plateau = bool(abs(late_increment) <= plateau_epsilon)

        base = dict(zip(BASE_CELL_COLUMNS, key if isinstance(key, tuple) else (key,)))
        row: dict[str, Any] = {
            **base,
            "trajectory_class": trajectory_class,
            "probability_shape": probability_shape,
            "first_empirically_passable_k": first_pass_k,
            "empirically_passable": first_pass_k is not None,
            "maximum_gate_passage_probability": max_probability,
            "final_tested_gate_passage_probability": final_probability,
            "k_at_maximum_probability": int(grid[int(np.argmax(probabilities))]),
            "minimum_k_for_probability_0_50": minimum_k_reaching(ordered, 0.50),
            "minimum_k_for_probability_0_80": minimum_k_reaching(ordered, 0.80),
            "minimum_k_for_probability_0_90": minimum_k_reaching(ordered, 0.90),
            "material_nonmonotone": material_nonmonotone,
            "probability_improving_without_0_50_reliability": improving_without_reliability,
            "late_probability_increment": late_increment,
            "late_probability_plateau_within_epsilon": late_plateau,
            "late_gain_from_k6_to_max": (
                final_probability
                - float(probabilities[grid.index(6)])
                if 6 in grid
                else float("nan")
            ),
            "final_mean_effective_cluster_count": float(effective_means[-1]),
            "probability_vector_json": canonical_json(
                {str(k): float(p) for k, p in zip(grid, probabilities)}
            ),
            "mean_effective_cluster_vector_json": canonical_json(
                {str(k): float(v) for k, v in zip(grid, effective_means)}
            ),
        }
        for k, probability in zip(grid, probabilities):
            row[f"probability_k{k:02d}"] = float(probability)
        row["trajectory_hash"] = sha256_bytes(canonical_json(row).encode("utf-8"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["partition", "simulator_id", "address_id", "control_response_lambda", "delta"]
    )


def passable_set_by_k(
    summary: pd.DataFrame,
    cluster_grid: Sequence[int],
    scope_filter: Mapping[str, Any] | None = None,
) -> dict[int, frozenset[str]]:
    frame = summary
    if scope_filter:
        for column, value in scope_filter.items():
            frame = frame[frame[column].eq(value)]
    sets: dict[int, frozenset[str]] = {}
    for k in cluster_grid:
        current = frame[
            frame["prospective_cluster_count"].eq(k)
            & frame["conditional_gate_passage_probability"].gt(0)
        ]
        identifiers = current.apply(
            lambda row: canonical_json(
                {
                    column: row[column]
                    for column in BASE_CELL_COLUMNS
                    if column not in {"record_id", "support_id", "relation", "carrier", "entitlement_status", "failure_predicate"}
                }
            ),
            axis=1,
        )
        sets[k] = frozenset(identifiers.tolist())
    return sets


def coverage_plateau_start(
    summary: pd.DataFrame,
    cluster_grid: Sequence[int],
    scope_filter: Mapping[str, Any] | None = None,
) -> int:
    sets = passable_set_by_k(summary, cluster_grid, scope_filter)
    for index, k in enumerate(cluster_grid):
        if all(sets[later] == sets[k] for later in cluster_grid[index:]):
            return int(k)
    return int(cluster_grid[-1])


def plateau_decomposition(
    trajectories: pd.DataFrame,
    gate_removal: pd.DataFrame,
    summary: pd.DataFrame,
    plateau_start_k: int,
    dominant_margin: float,
) -> pd.DataFrame:
    plateau_gates = gate_removal[
        gate_removal["prospective_cluster_count"].ge(plateau_start_k)
    ].copy()
    plateau_summary = summary[
        summary["prospective_cluster_count"].ge(plateau_start_k)
    ].copy()
    rows: list[dict[str, Any]] = []
    trajectory_key = [
        "address_id",
        "partition",
        "simulator_id",
        "delta",
        "control_response_lambda",
    ]

    for _, trajectory in trajectories.iterrows():
        mask = np.ones(len(plateau_gates), dtype=bool)
        summary_mask = np.ones(len(plateau_summary), dtype=bool)
        for column in trajectory_key:
            mask &= plateau_gates[column].eq(trajectory[column]).to_numpy()
            summary_mask &= plateau_summary[column].eq(trajectory[column]).to_numpy()
        gate_group = plateau_gates.loc[mask]
        summary_group = plateau_summary.loc[summary_mask]
        if gate_group.empty or summary_group.empty:
            raise RuntimeError("Missing plateau diagnostic rows for a cell trajectory")

        rescue_by_gate = (
            gate_group.groupby("gate_name")["single_gate_blocker_count"].sum().to_dict()
        )
        trials = int(gate_group.drop_duplicates("prospective_cluster_count")["replicates"].sum())
        rescue_by_category: Counter[str] = Counter()
        for gate, count in rescue_by_gate.items():
            rescue_by_category[GATE_CATEGORY[gate]] += int(count)
        failure_by_category: defaultdict[str, list[float]] = defaultdict(list)
        for row in gate_group.itertuples(index=False):
            failure_by_category[GATE_CATEGORY[row.gate_name]].append(
                float(row.gate_failure_probability)
            )
        category_failure_burden = {
            category: float(np.mean(values)) for category, values in failure_by_category.items()
        }
        support_probability = float(
            summary_group["support_available_pass_probability"].mean()
        )
        complement_probability = float(
            summary_group["complement_admissible_pass_probability"].mean()
        )

        if normalize_bool(trajectory["empirically_passable"]):
            limiting_class = "empirically_passable"
            basis = "complete_pass_observed"
            dominant_gate = "none"
        elif support_probability == 0.0 or complement_probability == 0.0:
            limiting_class = "structurally_unavailable"
            basis = "zero_structural_gate_pass_probability"
            dominant_gate = (
                "support_available_pass"
                if support_probability == 0.0
                else "complement_admissible_pass"
            )
        else:
            max_rescue = max(rescue_by_category.values(), default=0)
            rescue_winners = sorted(
                category
                for category, count in rescue_by_category.items()
                if count == max_rescue and count > 0
            )
            if len(rescue_winners) == 1:
                limiting_class = rescue_winners[0]
                basis = "single_gate_rescue"
                candidate_gates = [
                    gate
                    for gate, count in rescue_by_gate.items()
                    if GATE_CATEGORY[gate] == limiting_class and count > 0
                ]
                dominant_gate = max(candidate_gates, key=lambda gate: rescue_by_gate[gate])
            elif len(rescue_winners) > 1:
                limiting_class = "mixed_gate_limited"
                basis = "tied_single_gate_rescue_categories"
                dominant_gate = "multiple"
            else:
                burdens = sorted(
                    category_failure_burden.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if burdens:
                    top_category, top_burden = burdens[0]
                    second_burden = burdens[1][1] if len(burdens) > 1 else 0.0
                    if top_burden >= 0.5 and top_burden - second_burden >= dominant_margin:
                        limiting_class = top_category
                        basis = "dominant_failure_burden"
                        candidate_gates = gate_group[
                            gate_group["gate_name"].map(GATE_CATEGORY).eq(top_category)
                        ]
                        dominant_gate = str(
                            candidate_gates.groupby("gate_name")["gate_failure_probability"]
                            .mean()
                            .idxmax()
                        )
                    else:
                        limiting_class = "mixed_gate_limited"
                        basis = "no_single_gate_rescue_or_dominant_failure_category"
                        dominant_gate = "multiple"
                else:
                    limiting_class = "unscorable"
                    basis = "missing_gate_burden"
                    dominant_gate = "none"

        rows.append(
            {
                **{column: trajectory[column] for column in BASE_CELL_COLUMNS},
                "coverage_plateau_start_k": plateau_start_k,
                "persistent_nonpassage": not normalize_bool(trajectory["empirically_passable"]),
                "plateau_limiting_class": limiting_class,
                "localization_basis": basis,
                "dominant_gate": dominant_gate,
                "plateau_replicates": trials,
                "mean_support_available_pass_probability": support_probability,
                "mean_complement_admissible_pass_probability": complement_probability,
                "category_single_gate_rescue_counts_json": canonical_json(
                    dict(sorted(rescue_by_category.items()))
                ),
                "category_failure_burden_json": canonical_json(
                    dict(sorted(category_failure_burden.items()))
                ),
                "gate_single_gate_rescue_counts_json": canonical_json(
                    dict(sorted(rescue_by_gate.items()))
                ),
                "maximum_gate_passage_probability": trajectory[
                    "maximum_gate_passage_probability"
                ],
                "final_tested_gate_passage_probability": trajectory[
                    "final_tested_gate_passage_probability"
                ],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["persistent_nonpassage", "plateau_limiting_class", "partition", "simulator_id", "address_id"],
        ascending=[False, True, True, True, True],
    )


# -----------------------------------------------------------------------------
# Address, concordance, marginal-value, and stopping diagnostics
# -----------------------------------------------------------------------------


def dominant_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "unscorable"
    counts = clean.value_counts()
    winners = sorted(counts[counts.eq(counts.max())].index.tolist())
    return winners[0] if len(winners) == 1 else "mixed:" + ",".join(winners)


def address_design_profiles(
    trajectories: pd.DataFrame,
    plateau: pd.DataFrame,
    effective_support: pd.DataFrame,
    max_k: int,
) -> pd.DataFrame:
    trajectory = trajectories.merge(
        plateau[
            [
                "address_id",
                "partition",
                "simulator_id",
                "delta",
                "control_response_lambda",
                "plateau_limiting_class",
            ]
        ],
        on=["address_id", "partition", "simulator_id", "delta", "control_response_lambda"],
        how="left",
        validate="one_to_one",
    )
    effective_max = effective_support[
        effective_support["aggregation_level"].eq("cell")
        & effective_support["prospective_cluster_count"].eq(max_k)
    ][
        [
            "address_id",
            "partition",
            "simulator_id",
            "delta",
            "control_response_lambda",
            "nominal_support_efficiency",
            "mean_effective_cluster_count",
        ]
    ]
    trajectory = trajectory.merge(
        effective_max,
        on=["address_id", "partition", "simulator_id", "delta", "control_response_lambda"],
        how="left",
        validate="one_to_one",
    )

    def aggregate(group: pd.DataFrame, scope: str, base: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **base,
            "profile_scope": scope,
            "address_effect_control_cells": len(group),
            "empirically_passable_cells": int(group["empirically_passable"].sum()),
            "empirically_passable_share": float(group["empirically_passable"].mean()),
            "minimum_k_for_any_passage": first_finite_min(
                group["first_empirically_passable_k"]
            ),
            "minimum_k_for_any_0_50_reliability": first_finite_min(
                group["minimum_k_for_probability_0_50"]
            ),
            "minimum_k_for_any_0_80_reliability": first_finite_min(
                group["minimum_k_for_probability_0_80"]
            ),
            "minimum_k_for_any_0_90_reliability": first_finite_min(
                group["minimum_k_for_probability_0_90"]
            ),
            "cells_reaching_0_50": int(
                group["minimum_k_for_probability_0_50"].notna().sum()
            ),
            "cells_reaching_0_80": int(
                group["minimum_k_for_probability_0_80"].notna().sum()
            ),
            "cells_reaching_0_90": int(
                group["minimum_k_for_probability_0_90"].notna().sum()
            ),
            "maximum_tested_gate_passage_probability": float(
                group["maximum_gate_passage_probability"].max()
            ),
            "median_final_gate_passage_probability": float(
                group["final_tested_gate_passage_probability"].median()
            ),
            "dominant_persistent_nonpassage_class": dominant_text(
                group.loc[
                    ~group["empirically_passable"].astype(bool),
                    "plateau_limiting_class",
                ]
            ),
            "mean_nominal_support_efficiency_at_max_k": float(
                group["nominal_support_efficiency"].mean()
            ),
            "mean_effective_cluster_count_at_max_k": float(
                group["mean_effective_cluster_count"].mean()
            ),
        }

    rows: list[dict[str, Any]] = []
    for key, group in trajectory.groupby(
        ["address_id", "partition", "simulator_id"], dropna=False
    ):
        rows.append(
            aggregate(
                group,
                "address_partition_simulator",
                {
                    "address_id": key[0],
                    "partition": key[1],
                    "simulator_id": key[2],
                },
            )
        )
    for address_id, group in trajectory.groupby("address_id", dropna=False):
        rows.append(
            aggregate(
                group,
                "address_all_contexts",
                {"address_id": address_id, "partition": "all", "simulator_id": "all"},
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["profile_scope", "address_id", "partition", "simulator_id"]
    )


def compare_trajectory_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    merge_columns: Sequence[str],
    left_label: str,
    right_label: str,
    cluster_grid: Sequence[int],
    comparison_type: str,
) -> pd.DataFrame:
    probability_columns = [f"probability_k{k:02d}" for k in cluster_grid]
    selected = [
        *merge_columns,
        "first_empirically_passable_k",
        "minimum_k_for_probability_0_50",
        "minimum_k_for_probability_0_80",
        "minimum_k_for_probability_0_90",
        "empirically_passable",
        "maximum_gate_passage_probability",
        "final_tested_gate_passage_probability",
        *probability_columns,
    ]
    merged = left[selected].merge(
        right[selected],
        on=list(merge_columns),
        how="inner",
        suffixes=(f"_{left_label}", f"_{right_label}"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        x = [float(row[f"{column}_{left_label}"]) for column in probability_columns]
        y = [float(row[f"{column}_{right_label}"]) for column in probability_columns]
        differences = np.abs(np.asarray(x) - np.asarray(y))
        result = {column: row[column] for column in merge_columns}
        result.update(
            {
                "comparison_type": comparison_type,
                "left_label": left_label,
                "right_label": right_label,
                "mean_absolute_probability_difference": float(differences.mean()),
                "maximum_absolute_probability_difference": float(differences.max()),
                "trajectory_pearson_correlation": pearson_or_nan(x, y),
                "first_pass_k_left": row[f"first_empirically_passable_k_{left_label}"],
                "first_pass_k_right": row[f"first_empirically_passable_k_{right_label}"],
                "first_pass_k_agreement": (
                    optional_int(row[f"first_empirically_passable_k_{left_label}"])
                    == optional_int(row[f"first_empirically_passable_k_{right_label}"])
                ),
                "empirical_passability_agreement": (
                    normalize_bool(row[f"empirically_passable_{left_label}"])
                    == normalize_bool(row[f"empirically_passable_{right_label}"])
                ),
                "minimum_k_0_50_agreement": (
                    optional_int(row[f"minimum_k_for_probability_0_50_{left_label}"])
                    == optional_int(row[f"minimum_k_for_probability_0_50_{right_label}"])
                ),
                "minimum_k_0_80_agreement": (
                    optional_int(row[f"minimum_k_for_probability_0_80_{left_label}"])
                    == optional_int(row[f"minimum_k_for_probability_0_80_{right_label}"])
                ),
                "minimum_k_0_90_agreement": (
                    optional_int(row[f"minimum_k_for_probability_0_90_{left_label}"])
                    == optional_int(row[f"minimum_k_for_probability_0_90_{right_label}"])
                ),
                "final_probability_left": row[
                    f"final_tested_gate_passage_probability_{left_label}"
                ],
                "final_probability_right": row[
                    f"final_tested_gate_passage_probability_{right_label}"
                ],
                "probability_vector_left_json": canonical_json(
                    {str(k): value for k, value in zip(cluster_grid, x)}
                ),
                "probability_vector_right_json": canonical_json(
                    {str(k): value for k, value in zip(cluster_grid, y)}
                ),
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def partition_concordance(
    trajectories: pd.DataFrame,
    cluster_grid: Sequence[int],
) -> pd.DataFrame:
    discovery = trajectories[trajectories["partition"].eq("discovery")]
    confirmation = trajectories[trajectories["partition"].eq("confirmation")]
    return compare_trajectory_pairs(
        discovery,
        confirmation,
        ["address_id", "simulator_id", "delta", "control_response_lambda"],
        "discovery",
        "confirmation",
        cluster_grid,
        "partition",
    ).sort_values(["simulator_id", "address_id", "control_response_lambda", "delta"])


def simulator_concordance(
    trajectories: pd.DataFrame,
    cluster_grid: Sequence[int],
) -> pd.DataFrame:
    gaussian_id = "joint_gaussian_regularized_cluster"
    wild_id = "joint_wild_cluster_rademacher"
    gaussian = trajectories[trajectories["simulator_id"].eq(gaussian_id)]
    wild = trajectories[trajectories["simulator_id"].eq(wild_id)]
    return compare_trajectory_pairs(
        gaussian,
        wild,
        ["address_id", "partition", "delta", "control_response_lambda"],
        "gaussian",
        "wild",
        cluster_grid,
        "simulator",
    ).sort_values(["partition", "address_id", "control_response_lambda", "delta"])


def marginal_support_value(
    summary: pd.DataFrame,
    cluster_grid: Sequence[int],
) -> pd.DataFrame:
    identity = [
        "address_id",
        "partition",
        "simulator_id",
        "delta",
        "control_response_lambda",
    ]
    scopes: list[tuple[str, dict[str, Any]]] = [("global", {})]
    for partition in sorted(summary["partition"].unique()):
        for simulator in sorted(summary["simulator_id"].unique()):
            scopes.append(
                (
                    "partition_simulator",
                    {"partition": partition, "simulator_id": simulator},
                )
            )

    rows: list[dict[str, Any]] = []
    for scope_name, scope_filter in scopes:
        frame = summary
        for column, value in scope_filter.items():
            frame = frame[frame[column].eq(value)]
        for previous_k, next_k in zip(cluster_grid[:-1], cluster_grid[1:]):
            previous = frame[frame["prospective_cluster_count"].eq(previous_k)][
                identity
                + [
                    "conditional_gate_passage_probability",
                    "mean_independent_cluster_count",
                ]
            ].rename(
                columns={
                    "conditional_gate_passage_probability": "probability_previous",
                    "mean_independent_cluster_count": "effective_previous",
                }
            )
            following = frame[frame["prospective_cluster_count"].eq(next_k)][
                identity
                + [
                    "conditional_gate_passage_probability",
                    "mean_independent_cluster_count",
                ]
            ].rename(
                columns={
                    "conditional_gate_passage_probability": "probability_next",
                    "mean_independent_cluster_count": "effective_next",
                }
            )
            paired = previous.merge(following, on=identity, validate="one_to_one")
            gain = paired["probability_next"] - paired["probability_previous"]
            row: dict[str, Any] = {
                "aggregation_scope": scope_name,
                "partition": scope_filter.get("partition", "all"),
                "simulator_id": scope_filter.get("simulator_id", "all"),
                "previous_cluster_count": previous_k,
                "next_cluster_count": next_k,
                "evaluated_cells": len(paired),
                "newly_empirically_passable_cells": int(
                    (paired["probability_previous"].eq(0) & paired["probability_next"].gt(0)).sum()
                ),
                "lost_empirically_passable_cells": int(
                    (paired["probability_previous"].gt(0) & paired["probability_next"].eq(0)).sum()
                ),
                "mean_gate_passage_probability_gain": float(gain.mean()),
                "median_gate_passage_probability_gain": float(gain.median()),
                "minimum_gate_passage_probability_gain": float(gain.min()),
                "maximum_gate_passage_probability_gain": float(gain.max()),
                "positive_probability_gain_share": float(gain.gt(0).mean()),
                "mean_effective_cluster_gain": float(
                    (paired["effective_next"] - paired["effective_previous"]).mean()
                ),
            }
            for target in RELIABILITY_TARGETS:
                label = str(target).replace(".", "_")
                row[f"newly_reached_probability_{label}"] = int(
                    (
                        paired["probability_previous"].lt(target)
                        & paired["probability_next"].ge(target)
                    ).sum()
                )
                row[f"lost_probability_{label}"] = int(
                    (
                        paired["probability_previous"].ge(target)
                        & paired["probability_next"].lt(target)
                    ).sum()
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["aggregation_scope", "partition", "simulator_id", "previous_cluster_count"]
    )


def design_stopping_table(
    summary: pd.DataFrame,
    cluster_grid: Sequence[int],
    plateau_epsilon: float,
) -> pd.DataFrame:
    scopes: list[tuple[str, dict[str, Any]]] = [("global", {})]
    for partition in sorted(summary["partition"].unique()):
        for simulator in sorted(summary["simulator_id"].unique()):
            scopes.append(
                (
                    "partition_simulator",
                    {"partition": partition, "simulator_id": simulator},
                )
            )
    rows: list[dict[str, Any]] = []
    for scope_name, scope_filter in scopes:
        frame = summary
        for column, value in scope_filter.items():
            frame = frame[frame[column].eq(value)]
        plateau_start = coverage_plateau_start(frame, cluster_grid)
        previous_passable: set[str] = set()
        previous_mean: float | None = None
        max_mean = float(
            frame[frame["prospective_cluster_count"].eq(max(cluster_grid))][
                "conditional_gate_passage_probability"
            ].mean()
        )
        for k in cluster_grid:
            current = frame[frame["prospective_cluster_count"].eq(k)]
            cell_ids = set(
                current.loc[
                    current["conditional_gate_passage_probability"].gt(0)
                ].apply(
                    lambda row: canonical_json(
                        {
                            "address_id": row["address_id"],
                            "partition": row["partition"],
                            "simulator_id": row["simulator_id"],
                            "delta": float(row["delta"]),
                            "control_response_lambda": float(row["control_response_lambda"]),
                        }
                    ),
                    axis=1,
                )
            )
            mean_probability = float(current["conditional_gate_passage_probability"].mean())
            if k < 4:
                status = "structurally_unattainable"
            elif k < plateau_start:
                status = "coverage_expanding"
            elif max_mean - mean_probability > plateau_epsilon:
                status = "coverage_plateau_probability_gain_continues"
            else:
                status = "coverage_plateau_probability_gain_within_epsilon"
            rows.append(
                {
                    "aggregation_scope": scope_name,
                    "partition": scope_filter.get("partition", "all"),
                    "simulator_id": scope_filter.get("simulator_id", "all"),
                    "prospective_cluster_count": k,
                    "evaluated_cells": len(current),
                    "empirically_passable_cells": len(cell_ids),
                    "empirically_passable_share": len(cell_ids) / len(current),
                    "newly_passable_cells_from_previous": len(cell_ids - previous_passable),
                    "lost_passable_cells_from_previous": len(previous_passable - cell_ids),
                    "mean_gate_passage_probability": mean_probability,
                    "mean_probability_gain_from_previous": (
                        float("nan") if previous_mean is None else mean_probability - previous_mean
                    ),
                    "cells_reaching_0_50": int(
                        current["conditional_gate_passage_probability"].ge(0.50).sum()
                    ),
                    "cells_reaching_0_80": int(
                        current["conditional_gate_passage_probability"].ge(0.80).sum()
                    ),
                    "cells_reaching_0_90": int(
                        current["conditional_gate_passage_probability"].ge(0.90).sum()
                    ),
                    "mean_effective_cluster_count": float(
                        current["mean_independent_cluster_count"].mean()
                    ),
                    "coverage_plateau_start_k": plateau_start,
                    "diagnostic_design_status": status,
                }
            )
            previous_passable = cell_ids
            previous_mean = mean_probability
    return pd.DataFrame(rows).sort_values(
        ["aggregation_scope", "partition", "simulator_id", "prospective_cluster_count"]
    )


def entitlement_overlay(
    trajectories: pd.DataFrame,
    plateau: pd.DataFrame,
) -> pd.DataFrame:
    merged = trajectories.merge(
        plateau[
            [
                "address_id",
                "partition",
                "simulator_id",
                "delta",
                "control_response_lambda",
                "plateau_limiting_class",
                "persistent_nonpassage",
            ]
        ],
        on=["address_id", "partition", "simulator_id", "delta", "control_response_lambda"],
        how="left",
        validate="one_to_one",
    )
    return (
        merged.groupby(
            [
                "entitlement_status",
                "partition",
                "simulator_id",
                "plateau_limiting_class",
            ],
            dropna=False,
        )
        .agg(
            address_effect_control_cells=("address_id", "size"),
            addresses=("address_id", "nunique"),
            empirically_passable_cells=("empirically_passable", "sum"),
            persistent_nonpassage_cells=("persistent_nonpassage", "sum"),
            maximum_gate_passage_probability=(
                "maximum_gate_passage_probability",
                "max",
            ),
            mean_final_gate_passage_probability=(
                "final_tested_gate_passage_probability",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            ["entitlement_status", "partition", "simulator_id", "plateau_limiting_class"]
        )
    )


# -----------------------------------------------------------------------------
# Outputs, report, and manifest
# -----------------------------------------------------------------------------


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "obs085d_manifest.json": output_dir / "obs085d_manifest.json",
        "obs085d_input_manifest.csv": output_dir / "obs085d_input_manifest.csv",
        "obs085d_cell_trajectory_classification.csv": output_dir
        / "obs085d_cell_trajectory_classification.csv",
        "obs085d_gate_removal_diagnostics.csv": output_dir
        / "obs085d_gate_removal_diagnostics.csv",
        "obs085d_first_failure_summary.csv": output_dir
        / "obs085d_first_failure_summary.csv",
        "obs085d_terminal_blocker_summary.csv": output_dir
        / "obs085d_terminal_blocker_summary.csv",
        "obs085d_plateau_decomposition.csv": output_dir
        / "obs085d_plateau_decomposition.csv",
        "obs085d_effective_support_summary.csv": output_dir
        / "obs085d_effective_support_summary.csv",
        "obs085d_address_design_profiles.csv": output_dir
        / "obs085d_address_design_profiles.csv",
        "obs085d_partition_concordance.csv": output_dir
        / "obs085d_partition_concordance.csv",
        "obs085d_simulator_concordance.csv": output_dir
        / "obs085d_simulator_concordance.csv",
        "obs085d_marginal_support_value.csv": output_dir
        / "obs085d_marginal_support_value.csv",
        "obs085d_design_stopping_table.csv": output_dir
        / "obs085d_design_stopping_table.csv",
        "obs085d_entitlement_overlay.csv": output_dir
        / "obs085d_entitlement_overlay.csv",
        "obs085d_failures.csv": output_dir / "obs085d_failures.csv",
        "obs085d_report.md": output_dir / "obs085d_report.md",
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def failures_frame(failures: Sequence[StudyFailure]) -> pd.DataFrame:
    columns = ["stage", "scope_id", "reason", "detail", "severity"]
    if not failures:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([failure.__dict__ for failure in failures], columns=columns)


def artifact_inventory(outputs: Mapping[str, Path], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in sorted(outputs.items()):
        if name == "obs085d_manifest.json" or not path.is_file():
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
    trajectories: pd.DataFrame,
    gate_removal: pd.DataFrame,
    plateau: pd.DataFrame,
    effective_support: pd.DataFrame,
    marginal: pd.DataFrame,
    stopping: pd.DataFrame,
    partition_cmp: pd.DataFrame,
    simulator_cmp: pd.DataFrame,
    failures: pd.DataFrame,
    args: argparse.Namespace,
    processed_rows: int,
    plateau_start_k: int,
) -> None:
    trajectory_counts = (
        trajectories.groupby(["trajectory_class", "probability_shape"], dropna=False)
        .size()
        .rename("cells")
        .reset_index()
    )
    plateau_counts = (
        plateau.groupby(["persistent_nonpassage", "plateau_limiting_class"], dropna=False)
        .size()
        .rename("cells")
        .reset_index()
    )
    rescue_global = (
        gate_removal.groupby("gate_name", dropna=False)
        .agg(
            single_gate_rescues=("single_gate_blocker_count", "sum"),
            mean_single_gate_blocker_probability=(
                "single_gate_blocker_probability",
                "mean",
            ),
            mean_gate_failure_probability=("gate_failure_probability", "mean"),
        )
        .reset_index()
        .sort_values("single_gate_rescues", ascending=False)
    )
    effective_global = effective_support[
        effective_support["aggregation_level"].eq("global")
    ].sort_values("prospective_cluster_count")
    marginal_global = marginal[marginal["aggregation_scope"].eq("global")]
    stopping_global = stopping[stopping["aggregation_scope"].eq("global")]
    partition_summary = pd.DataFrame(
        [
            {
                "matched_cells": len(partition_cmp),
                "mean_absolute_probability_difference": partition_cmp[
                    "mean_absolute_probability_difference"
                ].mean(),
                "first_pass_k_agreement_share": partition_cmp[
                    "first_pass_k_agreement"
                ].mean(),
                "empirical_passability_agreement_share": partition_cmp[
                    "empirical_passability_agreement"
                ].mean(),
            }
        ]
    )
    simulator_summary = pd.DataFrame(
        [
            {
                "matched_cells": len(simulator_cmp),
                "mean_absolute_probability_difference": simulator_cmp[
                    "mean_absolute_probability_difference"
                ].mean(),
                "first_pass_k_agreement_share": simulator_cmp[
                    "first_pass_k_agreement"
                ].mean(),
                "empirical_passability_agreement_share": simulator_cmp[
                    "empirical_passability_agreement"
                ].mean(),
            }
        ]
    )

    lines = [
        "# OBS-085d — Campaign Bottleneck Localization",
        "",
        "## State",
        "",
        f"`{state}`",
        "",
        "OBS-085d deterministically localizes the frozen OBS-085c prospective-campaign bottlenecks. No new simulation or threshold modification was performed.",
        "",
        "## Frozen lineage",
        "",
        f"- OBS-085c manifest ID: `{lineage['obs085c_manifest_id']}`",
        f"- OBS-085c manifest SHA256: `{lineage['obs085c_manifest_sha256']}`",
        f"- OBS-085c script SHA256: `{lineage['obs085c_script_sha256']}`",
        f"- OBS-085c script commit: `{lineage['obs085c_script_commit']}`",
        f"- OBS-085c output commit: `{lineage['obs085c_output_commit']}`",
        f"- OBS-085c output hashes checked: **{lineage['obs085c_output_hashes_checked']}**",
        f"- Current repository HEAD: `{lineage['current_repo_head']}`",
        "",
        "## Execution integrity",
        "",
        f"- Frozen replicate rows analyzed: **{processed_rows:,}**",
        f"- Cell trajectories: **{len(trajectories):,}**",
        f"- Empirical passable-set coverage plateau begins at tested k=**{plateau_start_k}**.",
        f"- Nonmonotonicity tolerance: **{args.nonmonotone_tolerance:g}** absolute probability.",
        f"- Plateau epsilon: **{args.plateau_epsilon:g}** absolute probability.",
        "",
        "## Cell trajectory classification",
        "",
        markdown_table(trajectory_counts, args.max_report_rows),
        "",
        "## Persistent non-passage decomposition",
        "",
        markdown_table(plateau_counts, args.max_report_rows),
        "",
        "> A limiting class localizes frozen-gate behavior. It is not a recommendation to weaken or remove that gate.",
        "",
        "## Leave-one-gate-out diagnostic rescue",
        "",
        markdown_table(rescue_global, args.max_report_rows),
        "",
        "## Nominal versus effective support",
        "",
        markdown_table(effective_global, args.max_report_rows),
        "",
        "## Marginal value of added support",
        "",
        markdown_table(marginal_global, args.max_report_rows),
        "",
        "## Diagnostic design stopping table",
        "",
        markdown_table(stopping_global, args.max_report_rows),
        "",
        "Coverage plateau and probability saturation are distinct. A stable passable-cell set can coexist with continuing probability gains inside that set.",
        "",
        "## Partition concordance",
        "",
        markdown_table(partition_summary, args.max_report_rows),
        "",
        "## Simulator concordance",
        "",
        markdown_table(simulator_summary, args.max_report_rows),
        "",
        "## Failures",
        "",
        markdown_table(failures, args.max_report_rows),
        "",
        "## Interpretation boundary",
        "",
        "> OBS-085d is diagnostic localization only.",
        "",
        "> Leave-one-gate-out passage is not an alternative evidence result and does not justify removing a frozen gate.",
        "",
        "> Prospective template replication is not additional observed evidence.",
        "",
        "> The study cannot create an FL3 witness, establish causal attribution, validate simulator truth, or increase claim entitlement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    repo_root: Path,
    outputs: Mapping[str, Path],
    state: str,
    lineage: Mapping[str, Any],
    args: argparse.Namespace,
    cluster_grid: Sequence[int],
    processed_rows: int,
    summary_rows: int,
    trajectory_rows: int,
    plateau_start_k: int,
    persistent_nonpassage_cells: int,
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "state": state,
        "scope": "deterministic artifact-only prospective campaign bottleneck localization",
        "claim_ceiling": (
            "diagnostic localization only; no observed witness, causal attribution, "
            "simulator truth, retrospective gate modification, or entitlement increase"
        ),
        "frozen_lineage": dict(lineage),
        "analysis_contract": {
            "cluster_grid": [int(k) for k in cluster_grid],
            "gate_order": list(GATE_ORDER),
            "reliability_targets": list(RELIABILITY_TARGETS),
            "nonmonotone_tolerance": args.nonmonotone_tolerance,
            "plateau_epsilon": args.plateau_epsilon,
            "dominant_failure_margin": args.dominant_margin,
            "leave_one_gate_out_semantics": (
                "diagnostic only; all other frozen gates remain required"
            ),
            "coverage_plateau_definition": (
                "earliest tested k whose empirically passable cell set is unchanged "
                "at every larger tested k"
            ),
        },
        "execution": {
            "smoke": args.smoke,
            "address_limit": args.address_limit,
            "chunk_size": args.chunk_size,
            "frozen_summary_rows_selected": summary_rows,
            "frozen_replicate_rows_analyzed": processed_rows,
            "cell_trajectory_rows": trajectory_rows,
            "coverage_plateau_start_k": plateau_start_k,
            "persistent_nonpassage_cells": persistent_nonpassage_cells,
        },
        "output_artifacts": artifact_inventory(outputs, repo_root),
        "mandatory_statements": [
            "OBS-085b and OBS-085c remain frozen and unchanged.",
            "Leave-one-gate-out passage is a localization diagnostic, not an alternative evidence result.",
            "A localized bottleneck does not justify gate removal.",
            "Prospective template replication is not additional observed evidence.",
            "Coverage plateau is distinct from passage-probability saturation.",
            "OBS-085d cannot create an observed witness or increase claim entitlement.",
        ],
    }
    return {
        "obs085d_manifest_id": sha256_bytes(canonical_json(core).encode("utf-8")),
        **core,
    }


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def synthetic_replicate_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = {
        "scenario_id": "SCENARIO-K04",
        "base_scenario_id": "BASE-1",
        "address_id": "A1",
        "partition": "discovery",
        "simulator_id": "joint_gaussian_regularized_cluster",
        "prospective_cluster_count": 4,
        "delta": 1.0,
        "control_response_lambda": 0.0,
        "effective_resolution_attainable": True,
        "independent_cluster_count": 4,
        "raw_permutation_p": 0.0625,
        "cluster_uncertainty_decision_status": "exact_positive_certificate",
    }
    patterns = [
        [True] * 10,
        [True, True, True, True, True, True, False, True, True, True],
        [True, True, True, True, True, False, False, True, True, True],
        [False, True, True, True, True, True, True, True, True, True],
    ]
    for replicate, pattern in enumerate(patterns):
        row = {**base, "replicate": replicate}
        row.update(dict(zip(GATE_ORDER, pattern)))
        row["overall_gate_pass"] = all(pattern)
        rows.append(row)
    return pd.DataFrame(rows)


def run_self_test() -> None:
    frame = synthetic_replicate_rows()
    with tempfile.TemporaryDirectory() as temporary:
        replicate_dir = Path(temporary)
        shard = replicate_dir / "obs085c_replicates_k04.csv.gz"
        frame.to_csv(shard, index=False, compression="gzip")
        accumulators, processed = stream_replicate_shards(
            replicate_dir,
            [4],
            None,
            chunk_size=2,
        )
    if processed != 4 or len(accumulators) != 1:
        raise AssertionError("Streaming accumulator self-test failed")
    acc = next(iter(accumulators.values()))
    if acc.overall_pass != 1:
        raise AssertionError("Overall-pass self-test failed")
    raw_gate_index = GATE_ORDER.index("raw_statistical_threshold_pass")
    support_gate_index = GATE_ORDER.index("support_available_pass")
    if acc.single_gate_blocker[raw_gate_index] != 1:
        raise AssertionError("Single-gate rescue self-test failed")
    if acc.single_gate_blocker[support_gate_index] != 1:
        raise AssertionError("Structural single-gate blocker self-test failed")
    if acc.blocker_multiplicity[2] != 1:
        raise AssertionError("Multi-gate blocker self-test failed")
    if acc.first_failed[support_gate_index] != 1:
        raise AssertionError("First-failure self-test failed")
    if acc.last_failed[raw_gate_index] != 2:
        raise AssertionError("Last-failure self-test failed")

    summary_rows: list[dict[str, Any]] = []
    for address, probabilities in (
        ("A1", [0.0, 0.1, 0.2, 0.3]),
        ("A2", [0.0, 0.0, 0.0, 0.0]),
    ):
        for k, probability in zip((3, 4, 6, 12), probabilities):
            row = {
                "address_id": address,
                "record_id": f"R-{address}",
                "support_id": f"S-{address}",
                "relation": "rel",
                "carrier": "car",
                "entitlement_status": "capped",
                "partition": "discovery",
                "simulator_id": "sim",
                "failure_predicate": "missingness",
                "delta": 1.0,
                "control_response_lambda": 0.0,
                "prospective_cluster_count": k,
                "conditional_gate_passage_probability": probability,
                "mean_independent_cluster_count": float(k),
            }
            summary_rows.append(row)
    trajectories = build_cell_trajectories(
        pd.DataFrame(summary_rows),
        (3, 4, 6, 12),
        nonmonotone_tolerance=0.02,
        plateau_epsilon=0.01,
    )
    classes = dict(zip(trajectories["address_id"], trajectories["trajectory_class"]))
    if classes != {"A1": "early_passable", "A2": "empirically_never_passable"}:
        raise AssertionError(f"Trajectory classification self-test failed: {classes}")
    plateau_start = coverage_plateau_start(
        pd.DataFrame(summary_rows),
        (3, 4, 6, 12),
    )
    if plateau_start != 4:
        raise AssertionError(f"Coverage plateau self-test failed: {plateau_start}")

    payload = {"b": 2, "a": [1, 3]}
    if sha256_bytes(canonical_json(payload).encode()) != sha256_bytes(
        canonical_json({"a": [1, 3], "b": 2}).encode()
    ):
        raise AssertionError("Canonical hashing self-test failed")
    print(
        "OBS-085d self-test passed: chunked replicate reading, frozen-gate "
        "reconstruction, leave-one-gate-out rescue, first/last failure, blocker "
        "multiplicity, trajectory classification, coverage plateau, and "
        "deterministic hashing"
    )


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.nonmonotone_tolerance < 0 or args.plateau_epsilon < 0:
        raise ValueError("Tolerances must be nonnegative")
    if args.dominant_margin < 0:
        raise ValueError("--dominant-margin must be nonnegative")

    repo_root = args.repo_root.resolve()
    cluster_grid = parse_cluster_grid(args.cluster_grid)
    paths, source_manifest, full_summary, input_manifest, lineage = validate_frozen_inputs(args)

    selected_addresses: set[str] | None = None
    if args.smoke:
        cluster_grid = tuple(k for k in (3, 4, 6, 12) if k in cluster_grid)
        args.address_limit = 2 if args.address_limit is None else args.address_limit
    if args.address_limit is not None:
        ordered_addresses = sorted(full_summary["address_id"].astype(str).unique())
        selected_addresses = set(ordered_addresses[: args.address_limit])

    summary = full_summary[
        full_summary["prospective_cluster_count"].astype(int).isin(cluster_grid)
    ].copy()
    if selected_addresses is not None:
        summary = summary[summary["address_id"].astype(str).isin(selected_addresses)].copy()
    if summary.empty:
        raise RuntimeError("No OBS-085c summary cells selected")

    canonical_run = (
        not args.smoke
        and args.address_limit is None
        and tuple(cluster_grid) == CANONICAL_CLUSTER_GRID
    )
    state = (
        "campaign_bottleneck_localization_completed"
        if canonical_run
        else "diagnostic_engineering_subset_completed"
    )

    expected_rows = int(summary["replicates"].sum())
    print("OBS-085d validation complete")
    print(f"Frozen OBS-085c manifest: {lineage['obs085c_manifest_id']}")
    print(f"Frozen OBS-085c artifacts validated: {lineage['obs085c_output_hashes_checked']}")
    print(f"Selected addresses: {summary['address_id'].nunique()}")
    print(f"Selected cluster counts: {list(cluster_grid)}")
    print(f"Selected summary cells: {len(summary):,}")
    print(f"Expected replicate rows: {expected_rows:,}")

    if args.validate_only:
        return 0

    output_dir = resolve_under_root(args.output_dir, repo_root)
    prepare_output_dir(output_dir, args.overwrite)
    outputs = output_paths(output_dir)

    accumulators, processed_rows = stream_replicate_shards(
        paths["replicate_dir"],
        cluster_grid,
        selected_addresses,
        args.chunk_size,
    )
    if processed_rows != expected_rows:
        raise RuntimeError(
            f"Replicate row count mismatch: analyzed {processed_rows:,}, expected {expected_rows:,}"
        )

    failures = validate_accumulators_against_summary(accumulators, summary)
    fatal = [failure for failure in failures if failure.severity == "fatal"]
    if fatal:
        raise RuntimeError(
            f"Replicate-summary validation produced {len(fatal)} fatal failures"
        )

    metadata = summary_metadata_lookup(summary)
    gate_removal = gate_removal_diagnostics(accumulators, metadata)
    first_failure = first_failure_summary(accumulators, metadata)
    terminal = terminal_blocker_summary(accumulators, metadata)
    effective_support = effective_support_summary(accumulators, metadata)
    trajectories = build_cell_trajectories(
        summary,
        cluster_grid,
        args.nonmonotone_tolerance,
        args.plateau_epsilon,
    )
    plateau_start_k = coverage_plateau_start(summary, cluster_grid)
    plateau = plateau_decomposition(
        trajectories,
        gate_removal,
        summary,
        plateau_start_k,
        args.dominant_margin,
    )
    address_profiles = address_design_profiles(
        trajectories,
        plateau,
        effective_support,
        max(cluster_grid),
    )
    partition_cmp = partition_concordance(trajectories, cluster_grid)
    simulator_cmp = simulator_concordance(trajectories, cluster_grid)
    marginal = marginal_support_value(summary, cluster_grid)
    stopping = design_stopping_table(summary, cluster_grid, args.plateau_epsilon)
    entitlement = entitlement_overlay(trajectories, plateau)
    failure_frame = failures_frame(failures)

    input_manifest.to_csv(outputs["obs085d_input_manifest.csv"], index=False)
    trajectories.to_csv(outputs["obs085d_cell_trajectory_classification.csv"], index=False)
    gate_removal.to_csv(outputs["obs085d_gate_removal_diagnostics.csv"], index=False)
    first_failure.to_csv(outputs["obs085d_first_failure_summary.csv"], index=False)
    terminal.to_csv(outputs["obs085d_terminal_blocker_summary.csv"], index=False)
    plateau.to_csv(outputs["obs085d_plateau_decomposition.csv"], index=False)
    effective_support.to_csv(outputs["obs085d_effective_support_summary.csv"], index=False)
    address_profiles.to_csv(outputs["obs085d_address_design_profiles.csv"], index=False)
    partition_cmp.to_csv(outputs["obs085d_partition_concordance.csv"], index=False)
    simulator_cmp.to_csv(outputs["obs085d_simulator_concordance.csv"], index=False)
    marginal.to_csv(outputs["obs085d_marginal_support_value.csv"], index=False)
    stopping.to_csv(outputs["obs085d_design_stopping_table.csv"], index=False)
    entitlement.to_csv(outputs["obs085d_entitlement_overlay.csv"], index=False)
    failure_frame.to_csv(outputs["obs085d_failures.csv"], index=False)

    write_report(
        outputs["obs085d_report.md"],
        state,
        lineage,
        trajectories,
        gate_removal,
        plateau,
        effective_support,
        marginal,
        stopping,
        partition_cmp,
        simulator_cmp,
        failure_frame,
        args,
        processed_rows,
        plateau_start_k,
    )

    manifest = build_manifest(
        repo_root,
        outputs,
        state,
        lineage,
        args,
        cluster_grid,
        processed_rows,
        len(summary),
        len(trajectories),
        plateau_start_k,
        int(plateau["persistent_nonpassage"].sum()),
    )
    outputs["obs085d_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("OBS-085d execution complete")
    print(f"State: {state}")
    print(f"Manifest: {manifest['obs085d_manifest_id']}")
    print(f"Replicate rows analyzed: {processed_rows:,} / {expected_rows:,}")
    print(f"Cell trajectories: {len(trajectories):,}")
    print(f"Coverage plateau begins at k={plateau_start_k}")
    print(f"Persistent non-passage cells: {int(plateau['persistent_nonpassage'].sum()):,}")
    print(f"Failures: {len(failure_frame):,}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # explicit CLI boundary
        print(f"OBS-085d failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
