#!/usr/bin/env python3
"""
obs085c_campaign_attainability_simulation.py

OBS-085c — Prospective Campaign Attainability Simulation
=========================================================

Purpose
-------
Estimate when the frozen OBS-085b evidence contract becomes mathematically
attainable and empirically passable under prospectively expanded independent
object-cluster support.

OBS-085c is a design-metrology study.  It does not repair, reinterpret, or
rerun OBS-085b.  The completed OBS-085b result remains frozen: the effective
cluster count never exceeded three, the smallest observed exact sign-flip
p-value was 0.125, and complete gate passage was structurally impossible at the
frozen alpha of 0.10.

Frozen lineage
--------------
Canonical execution requires:

* OBS-085b0 simulator qualification commit: 639de23
* OBS-085b conditional gate-passage commit: c1e9649
* OBS-085b manifest ID:
  90eacf9a6b96e73875dcef146d8641efe58d511d62d82ee0115f5da03952f84a
* OBS-085b script SHA256:
  cb1ac506dcf3d93c67c6743da4128dc0dd17a52106316dcb1c01e603b821fed6

The script validates the frozen OBS-085b manifest identity, all declared
OBS-085b output hashes, the frozen OBS-085b script hash, and the repository
commit ancestry before simulation.

Prospective design intervention
-------------------------------
The only scientific design intervention is the requested prospective object
count.  For each frozen address, partition, simulator, effect condition, and
outer replicate:

1. Source object templates are sampled jointly across target and frozen
   controls from the original jointly estimable object set.
2. New synthetic object identities are assigned.
3. The frozen qualified Gaussian or wild simulator is extended to the new
   object count while retaining its original address-level fit.
4. The frozen missingness injection and all ten OBS-085b gates are evaluated.
5. Requested object counts are nested prefixes of one maximum-count campaign,
   ensuring paired comparisons across prospective designs.

Repeated source templates are not treated as additional observed evidence.
They are prospective exchangeable object-cluster draws under a declared joint
cluster-template replication assumption.

Attainability vocabulary
------------------------
* structurally_attainable:
    1 / 2**k <= frozen alpha.
* empirically_passable:
    at least one simulated replicate passes all ten frozen gates.
* reliably_passable:
    the estimated conditional gate-passage probability reaches a declared
    reporting target on the tested design grid.

Uncertainty computation
-----------------------
The exact one-sided sign-flip p-value is computed for every finite prospective
cluster vector with an exact meet-in-the-middle subset-count algorithm.  This
is exact through the default maximum k=12 without enumerating all sign vectors
for every replicate.

The OBS-085b cluster-uncertainty gate is decided as follows:

* all finite object contributions strictly positive:
    exact positive-support certificate; the percentile-bootstrap lower bound
    must be positive;
* direction consistency below the frozen threshold, fewer than two leave-one-
  object-out estimates, or non-positive mean:
    exact gate failure without unnecessary bootstrap work;
* otherwise:
    deterministic percentile cluster bootstrap using the predeclared number of
    draws.  These ambiguous decisions are explicitly labeled as a numerical
    approximation extension; thresholds and gate semantics remain unchanged.

Default canonical design
------------------------
* frozen addresses: 6
* partitions: discovery, confirmation
* qualified simulators: 2
* frozen OBS-085b effect/control scenarios: 25
* prospective object counts: 3, 4, 5, 6, 8, 10, 12
* outer replicates per address × partition × simulator × scenario: 1000
* nested evaluated rows: 4,200,000
* replicate vectors: gzip CSV sharded by prospective object count

Run
---
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085c_campaign_attainability_simulation.py \\
  --overwrite

Validation only
---------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085c_campaign_attainability_simulation.py \\
  --validate-only

Engineering smoke run
---------------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085c_campaign_attainability_simulation.py \\
  --smoke --overwrite

Self-test
---------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085c_campaign_attainability_simulation.py \\
  --self-test

Interpretation ceiling
----------------------
OBS-085c estimates conditional prospective design behavior under frozen
qualified simulator assumptions.  It does not establish causal truth,
classical power, an observed witness, or increased claim entitlement.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence, TextIO

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "obs085c_campaign_attainability_simulation_v1"
DEFAULT_MASTER_SEED = 85200
DEFAULT_REPLICATES = 1000
DEFAULT_CLUSTER_GRID = "3,4,5,6,8,10,12"
DEFAULT_BOOTSTRAP_DRAWS = 2048
DEFAULT_EXPECTED_OBS085B_MANIFEST_ID = (
    "90eacf9a6b96e73875dcef146d8641efe58d511d62d82ee0115f5da03952f84a"
)
DEFAULT_EXPECTED_OBS085B_SCRIPT_SHA256 = (
    "cb1ac506dcf3d93c67c6743da4128dc0dd17a52106316dcb1c01e603b821fed6"
)
DEFAULT_OBS085B0_COMMIT = "639de23"
DEFAULT_OBS085B_COMMIT = "c1e9649"
EXPECTED_ADDRESS_COUNT = 6
AUTHORIZED_PREDICATE = "measurement_missingness_concentration"
AUTHORIZED_SIMULATORS = (
    "joint_gaussian_regularized_cluster",
    "joint_wild_cluster_rademacher",
)
AUTHORIZED_PARTITIONS = ("discovery", "confirmation")
CANONICAL_CLUSTER_GRID = (3, 4, 5, 6, 8, 10, 12)
RELIABILITY_TARGETS = (0.50, 0.80, 0.90)

DEFAULT_OBS085B_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085b_conditional_gate_passage_sensitivity"
)
DEFAULT_OBS085B_SCRIPT = Path(
    "experiments/studies/obs085b_conditional_gate_passage_sensitivity.py"
)
DEFAULT_OBS085B0_SCRIPT = Path(
    "experiments/studies/obs085b0_simulator_qualification.py"
)
DEFAULT_OBS085A_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/obs085a_structural_feasibility"
)
DEFAULT_OBS084_DISCOVERY_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/discovery"
)
DEFAULT_OBS084_CONFIRMATION_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/confirmation"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085c_campaign_attainability_simulation"
)

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

REPLICATE_COLUMNS = [
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
    "theoretical_sign_configurations",
    "minimum_attainable_exact_p",
    "structurally_attainable",
    "effective_minimum_attainable_exact_p",
    "effective_resolution_attainable",
    "delta",
    "control_response_lambda",
    "replicate",
    "seed",
    "source_template_count",
    "source_template_unique_count",
    "source_template_reuse_fraction",
    "source_template_ids_json",
    "target_contrast",
    "target_contrast_before_injection",
    "target_response_from_simulated_null",
    "median_control_contrast",
    "control_adjusted_contrast",
    "positive_control_adjusted_share",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_positive_share",
    "bootstrap_method",
    "bootstrap_draw_count",
    "cluster_uncertainty_decision_status",
    "direction_consistency",
    "loo_successful_count",
    "independent_cluster_count",
    "raw_permutation_p",
    "m1_adjusted_p",
    "permutation_count",
    "permutation_method",
    "target_site_rows",
    "target_complement_rows",
    "target_site_clusters",
    "target_complement_clusters",
    "shared_cluster_count",
    "range_violation_count",
    "identity_violation_count",
    *GATE_ORDER,
    "overall_gate_pass",
    "failed_gates_json",
    "replicate_hash",
]

NUMERIC_SUMMARY_FIELDS = (
    "target_contrast",
    "target_response_from_simulated_null",
    "control_adjusted_contrast",
    "positive_control_adjusted_share",
    "bootstrap_ci_low",
    "bootstrap_positive_share",
    "direction_consistency",
    "independent_cluster_count",
    "raw_permutation_p",
    "source_template_unique_count",
    "source_template_reuse_fraction",
)


@dataclass(frozen=True)
class StudyFailure:
    stage: str
    scope_id: str
    reason: str
    detail: str = ""
    severity: str = "warning"


@dataclass(frozen=True)
class FrozenGateParameters:
    alpha: float
    minimum_effect: float
    minimum_site_rows: int
    minimum_complement_rows: int
    minimum_shared_clusters: int
    minimum_direction_consistency: float
    minimum_control_adjusted_effect: float
    minimum_positive_control_share: float
    effect_direction_tolerance: float


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--obs085b-dir", type=Path, default=DEFAULT_OBS085B_DIR)
    parser.add_argument("--obs085b-script", type=Path, default=DEFAULT_OBS085B_SCRIPT)
    parser.add_argument("--obs085b0-script", type=Path, default=DEFAULT_OBS085B0_SCRIPT)
    parser.add_argument("--obs085a-dir", type=Path, default=DEFAULT_OBS085A_DIR)
    parser.add_argument(
        "--obs084-discovery-dir",
        type=Path,
        default=DEFAULT_OBS084_DISCOVERY_DIR,
    )
    parser.add_argument(
        "--obs084-confirmation-dir",
        type=Path,
        default=DEFAULT_OBS084_CONFIRMATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-obs085b-manifest-id",
        default=DEFAULT_EXPECTED_OBS085B_MANIFEST_ID,
    )
    parser.add_argument(
        "--expected-obs085b-script-sha256",
        default=DEFAULT_EXPECTED_OBS085B_SCRIPT_SHA256,
    )
    parser.add_argument("--obs085b0-commit", default=DEFAULT_OBS085B0_COMMIT)
    parser.add_argument("--obs085b-commit", default=DEFAULT_OBS085B_COMMIT)
    parser.add_argument("--cluster-grid", default=DEFAULT_CLUSTER_GRID)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--max-controls", type=int, default=4)
    parser.add_argument("--replicate-chunk-size", type=int, default=5000)
    parser.add_argument(
        "--write-replicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write complete replicate vectors, sharded by prospective cluster count.",
    )
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help="Deterministic prefix; non-smoke use marks the run diagnostic-only.",
    )
    parser.add_argument("--max-report-rows", type=int, default=50)
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


def stable_id(*parts: Any, prefix: str = "") -> str:
    digest = sha256_bytes(canonical_json(parts).encode("utf-8"))[:24]
    return f"{prefix}{digest}"


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "ok"}


def parse_integer_grid(text: str, *, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {text!r}") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(value < 2 for value in values):
        raise ValueError(f"{name} values must be at least 2")
    if any(value > 20 for value in values):
        raise ValueError(f"{name} values above 20 are outside this instrument")
    return sorted(set(values))


def repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    require_file(path, path.name)
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> Any:
    require_file(path, path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def load_module_from_path(path: Path, name: str) -> ModuleType:
    require_file(path, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_git(repo_root: Path, args: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def markdown_table(frame: pd.DataFrame, max_rows: int = 50) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()
    columns = list(shown.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in shown.itertuples(index=False, name=None):
        rendered = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(rendered) + " |")
    suffix = "\n\n_Additional rows omitted._" if len(frame) > len(shown) else ""
    return "\n".join([header, separator, *rows]) + suffix


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def mean_or_nan(values: Iterable[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    return float(finite.mean()) if finite.size else float("nan")


# -----------------------------------------------------------------------------
# Frozen lineage and input validation
# -----------------------------------------------------------------------------


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.repo_root
    bdir = repo_path(root, args.obs085b_dir)
    return {
        "obs085b_manifest": bdir / "obs085b_manifest.json",
        "obs085b_input_manifest": bdir / "obs085b_input_manifest.csv",
        "obs085b_authorized_cells": bdir / "obs085b_authorized_cells.csv",
        "obs085b_address_manifest": bdir / "obs085b_address_manifest.csv",
        "obs085b_gate_contract": bdir / "obs085b_gate_contract.csv",
        "obs085b_scenario_manifest": bdir / "obs085b_scenario_manifest.csv",
        "obs085b_script": repo_path(root, args.obs085b_script),
        "obs085b0_script": repo_path(root, args.obs085b0_script),
        "obs085a_control_availability": repo_path(root, args.obs085a_dir)
        / "obs085a_control_availability.csv",
        "obs084b_discovery_observation_losses": repo_path(
            root, args.obs084_discovery_dir
        )
        / "obs084b_discovery_observation_losses.csv",
        "obs084c_confirmation_observation_losses": repo_path(
            root, args.obs084_confirmation_dir
        )
        / "obs084c_confirmation_observation_losses.csv",
    }


def validate_required_inputs(paths: Mapping[str, Path]) -> None:
    for role, path in paths.items():
        require_file(path, role)


def validate_manifest_identity(manifest: Mapping[str, Any], expected_id: str) -> str:
    observed = str(manifest.get("obs085b_manifest_id", ""))
    core = {key: value for key, value in manifest.items() if key != "obs085b_manifest_id"}
    calculated = sha256_bytes(canonical_json(core).encode("utf-8"))
    if observed != calculated:
        raise RuntimeError(
            "OBS-085b manifest self-hash mismatch: "
            f"declared={observed}; calculated={calculated}"
        )
    if str(expected_id).lower() != "auto" and observed != str(expected_id):
        raise RuntimeError(
            "Unexpected OBS-085b manifest identity: "
            f"expected={expected_id}; observed={observed}"
        )
    if manifest.get("state") != "conditional_gate_passage_sensitivity_completed":
        raise RuntimeError("OBS-085c requires completed canonical OBS-085b outputs")
    if manifest.get("execution", {}).get("smoke"):
        raise RuntimeError("OBS-085c cannot use an OBS-085b smoke bundle")
    return observed


def validate_output_hashes(manifest: Mapping[str, Any], repo_root: Path) -> int:
    checked = 0
    for item in manifest.get("output_artifacts", []):
        path = repo_path(repo_root, Path(str(item["artifact_path"])))
        require_file(path, f"OBS-085b output {item['artifact_path']}")
        actual_size = path.stat().st_size
        expected_size = int(item["size_bytes"])
        if actual_size != expected_size:
            raise RuntimeError(
                f"OBS-085b output size mismatch for {path}: "
                f"expected={expected_size}; observed={actual_size}"
            )
        actual_hash = sha256_file(path)
        expected_hash = str(item["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"OBS-085b output hash mismatch for {path}: "
                f"expected={expected_hash}; observed={actual_hash}"
            )
        checked += 1
    if checked < 1:
        raise RuntimeError("OBS-085b manifest declared no output artifacts")
    return checked


def validate_script_hash(
    input_manifest: pd.DataFrame,
    script_path: Path,
    expected_hash: str,
) -> str:
    require_columns(
        input_manifest,
        ["artifact_role", "artifact_path", "sha256"],
        "OBS-085b input manifest",
    )
    rows = input_manifest[input_manifest["artifact_role"].eq("obs085b_script")]
    if len(rows) != 1:
        raise RuntimeError("OBS-085b input manifest must contain one obs085b_script row")
    declared = str(rows.iloc[0]["sha256"])
    actual = sha256_file(script_path)
    if declared != actual:
        raise RuntimeError(
            "OBS-085b script differs from its own input manifest: "
            f"declared={declared}; actual={actual}"
        )
    if str(expected_hash).lower() != "auto" and actual != str(expected_hash):
        raise RuntimeError(
            "Unexpected frozen OBS-085b script hash: "
            f"expected={expected_hash}; observed={actual}"
        )
    return actual


def validate_git_anchor(repo_root: Path, commit: str, label: str) -> str:
    try:
        resolved = run_git(repo_root, ["rev-parse", f"{commit}^{{commit}}"])
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", resolved, "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Required {label} commit {commit!r} is not an ancestor of HEAD"
        ) from exc
    return resolved


def validate_authorized_scope(
    manifest: Mapping[str, Any],
    authorized: pd.DataFrame,
    addresses: pd.DataFrame,
) -> None:
    require_columns(
        authorized,
        ["failure_predicate", "partition", "simulator_id", "qualification_status"],
        "OBS-085b authorized cells",
    )
    require_columns(
        addresses,
        [
            "address_id",
            "record_id",
            "support_id",
            "failure_predicate",
            "relation",
            "carrier",
            "support_query_json",
            "entitlement_status",
        ],
        "OBS-085b address manifest",
    )
    observed_cells = {
        (str(row.failure_predicate), str(row.partition), str(row.simulator_id))
        for row in authorized.itertuples(index=False)
    }
    expected_cells = {
        (AUTHORIZED_PREDICATE, partition, simulator)
        for partition in AUTHORIZED_PARTITIONS
        for simulator in AUTHORIZED_SIMULATORS
    }
    if observed_cells != expected_cells:
        raise RuntimeError(
            f"OBS-085b authorized cell set changed: {sorted(observed_cells)}"
        )
    observed_addresses = sorted(addresses["address_id"].astype(str).unique())
    manifest_addresses = sorted(str(value) for value in manifest.get("frozen_address_ids", []))
    if observed_addresses != manifest_addresses:
        raise RuntimeError("OBS-085b address manifest differs from frozen manifest IDs")
    if len(observed_addresses) != EXPECTED_ADDRESS_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ADDRESS_COUNT} frozen addresses; observed {len(observed_addresses)}"
        )
    if set(addresses["failure_predicate"].astype(str)) != {AUTHORIZED_PREDICATE}:
        raise RuntimeError("OBS-085c is restricted to the qualified missingness predicate")


def frozen_gate_parameters(
    manifest: Mapping[str, Any],
    gate_contract: pd.DataFrame,
) -> FrozenGateParameters:
    contract = dict(manifest.get("gate_contract", {}))
    require_columns(gate_contract, ["gate_name", "threshold"], "OBS-085b gate contract")
    expected_order = gate_contract.sort_values("gate_order")["gate_name"].astype(str).tolist()
    if expected_order != list(GATE_ORDER):
        raise RuntimeError("OBS-085b gate order changed")
    return FrozenGateParameters(
        alpha=float(contract["alpha"]),
        minimum_effect=float(contract["minimum_effect"]),
        minimum_site_rows=8,
        minimum_complement_rows=8,
        minimum_shared_clusters=2,
        minimum_direction_consistency=float(contract["minimum_direction_consistency"]),
        minimum_control_adjusted_effect=float(contract["minimum_control_adjusted_effect"]),
        minimum_positive_control_share=float(contract["minimum_positive_control_share"]),
        effect_direction_tolerance=1e-12,
    )


def validate_obs085b_lineage(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    FrozenGateParameters,
    dict[str, Any],
]:
    manifest = read_json(paths["obs085b_manifest"])
    manifest_id = validate_manifest_identity(manifest, args.expected_obs085b_manifest_id)
    hashes_checked = validate_output_hashes(manifest, args.repo_root)
    input_manifest = read_csv(paths["obs085b_input_manifest"])
    script_hash = validate_script_hash(
        input_manifest,
        paths["obs085b_script"],
        args.expected_obs085b_script_sha256,
    )
    authorized = read_csv(paths["obs085b_authorized_cells"])
    addresses = read_csv(paths["obs085b_address_manifest"], dtype={"address_id": "string"})
    gate_contract = read_csv(paths["obs085b_gate_contract"])
    scenarios = read_csv(paths["obs085b_scenario_manifest"])
    validate_authorized_scope(manifest, authorized, addresses)
    require_columns(
        scenarios,
        ["scenario_id", "delta", "control_response_lambda"],
        "OBS-085b scenario manifest",
    )
    frozen = frozen_gate_parameters(manifest, gate_contract)
    b0_commit = validate_git_anchor(args.repo_root, args.obs085b0_commit, "OBS-085b0")
    b_commit = validate_git_anchor(args.repo_root, args.obs085b_commit, "OBS-085b")
    lineage = {
        "obs085b_manifest_id": manifest_id,
        "obs085b_manifest_sha256": sha256_file(paths["obs085b_manifest"]),
        "obs085b_output_hashes_checked": hashes_checked,
        "obs085b_script_sha256": script_hash,
        "obs085b_script_version": str(manifest.get("script_version", "")),
        "obs085b_state": str(manifest.get("state", "")),
        "obs085b0_commit": b0_commit,
        "obs085b_commit": b_commit,
        "current_repo_head": run_git(args.repo_root, ["rev-parse", "HEAD"]),
    }
    return (
        manifest,
        authorized,
        addresses,
        gate_contract,
        scenarios,
        frozen,
        lineage,
    )


# -----------------------------------------------------------------------------
# Prospective scenario and design manifests
# -----------------------------------------------------------------------------


def theoretical_minimum_p(cluster_count: int) -> float:
    return 1.0 / float(2**cluster_count)


def design_manifest(cluster_grid: Sequence[int], frozen: FrozenGateParameters) -> pd.DataFrame:
    rows = []
    for cluster_count in cluster_grid:
        min_p = theoretical_minimum_p(cluster_count)
        rows.append(
            {
                "prospective_cluster_count": cluster_count,
                "theoretical_sign_configurations": 2**cluster_count,
                "minimum_attainable_exact_p": min_p,
                "frozen_alpha": frozen.alpha,
                "structurally_attainable": min_p <= frozen.alpha,
                "attainability_status": (
                    "structurally_attainable"
                    if min_p <= frozen.alpha
                    else "structurally_unattainable"
                ),
                "prospective_expansion_policy": "joint_cluster_template_bootstrap_nested_prefix",
                "sign_flip_engine": "exact_meet_in_the_middle_subset_count",
                "cluster_uncertainty_engine": (
                    "exact_certificates_plus_deterministic_bootstrap_for_ambiguous_vectors"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_scenario_manifest(
    base_scenarios: pd.DataFrame,
    cluster_grid: Sequence[int],
    replicates: int,
    master_seed: int,
    smoke: bool,
) -> pd.DataFrame:
    rows = []
    ordered = base_scenarios.sort_values(["delta", "control_response_lambda", "scenario_id"])
    for base in ordered.itertuples(index=False):
        for cluster_count in cluster_grid:
            rows.append(
                {
                    "scenario_id": stable_id(
                        "obs085c",
                        str(base.scenario_id),
                        cluster_count,
                        replicates,
                        master_seed,
                        prefix="CS-",
                    ),
                    "base_scenario_id": str(base.scenario_id),
                    "prospective_cluster_count": int(cluster_count),
                    "delta": float(base.delta),
                    "control_response_lambda": float(base.control_response_lambda),
                    "replicates": int(replicates),
                    "master_seed": int(master_seed),
                    "engineering_smoke": bool(smoke),
                    "nested_campaign_prefix": True,
                    "scenario_semantics": (
                        "prospective null gate-attainability calibration"
                        if float(base.delta) == 0.0
                        else "prospective conditional gate passage under expanded object support"
                    ),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Prospective object expansion and frozen simulator extension
# -----------------------------------------------------------------------------


def expand_address_joint_templates(
    b0: ModuleType,
    address: Any,
    prospective_cluster_count: int,
    rng: np.random.Generator,
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    original_ids = tuple(str(value) for value in address.cluster_ids)
    if len(original_ids) < 2:
        raise RuntimeError("Prospective expansion requires at least two source clusters")
    source_ids = tuple(
        str(value)
        for value in rng.choice(
            np.asarray(original_ids, dtype=object),
            size=prospective_cluster_count,
            replace=True,
        ).tolist()
    )
    target_ids = tuple(f"prospective_object_{index + 1:03d}" for index in range(prospective_cluster_count))
    components = [address.target, *address.controls]
    expanded_components = [
        b0.resample_component_by_cluster(component, source_ids, target_ids)
        for component in components
    ]
    expanded = b0.PreparedAddress(
        address_id=str(address.address_id),
        record_id=str(address.record_id),
        support_id=str(address.support_id),
        failure_predicate=str(address.failure_predicate),
        relation=str(address.relation),
        carrier=str(address.carrier),
        support_definition=str(address.support_definition),
        support_query_json=str(address.support_query_json),
        entitlement_status=str(address.entitlement_status),
        partition=str(address.partition),
        target=expanded_components[0],
        controls=list(expanded_components[1:]),
        cluster_ids=target_ids,
        selection_reason=(
            str(address.selection_reason)
            + "; prospective joint cluster-template replication"
        ),
    )
    return expanded, source_ids, target_ids


def simulate_prospective_components(
    b0: ModuleType,
    original_address: Any,
    expanded_address: Any,
    simulator: Any,
    rng: np.random.Generator,
) -> list[Any]:
    expanded_components = [expanded_address.target, *expanded_address.controls]
    expanded_ids = list(expanded_address.cluster_ids)
    expanded_residuals = b0.component_cluster_residuals(
        expanded_components,
        expanded_ids,
        expanded_address.failure_predicate,
    )

    if simulator.simulator_id == "joint_wild_cluster_rademacher":
        weights = rng.choice([-1.0, 1.0], size=len(expanded_ids))
        simulated = expanded_residuals * weights[:, None]
    elif simulator.simulator_id == "joint_gaussian_regularized_cluster":
        original_components = [original_address.target, *original_address.controls]
        original_ids = list(original_address.cluster_ids)
        original_residuals = b0.component_cluster_residuals(
            original_components,
            original_ids,
            original_address.failure_predicate,
        )
        covariance = b0.regularized_covariance(original_residuals)
        simulated = rng.multivariate_normal(
            mean=np.zeros(len(expanded_components)),
            cov=covariance,
            size=len(expanded_ids),
        )
    else:
        raise ValueError(f"Unauthorized simulator: {simulator.simulator_id}")

    transformed = []
    for component_index, component in enumerate(expanded_components):
        observed_map = {
            cluster_id: float(expanded_residuals[index, component_index])
            for index, cluster_id in enumerate(expanded_ids)
        }
        simulated_map = {
            cluster_id: float(simulated[index, component_index])
            for index, cluster_id in enumerate(expanded_ids)
        }
        transformed.append(
            b0.replace_component_cluster_effects(
                component,
                expanded_address.failure_predicate,
                observed_map,
                simulated_map,
            )
        )
    return transformed


def subset_component(b0: ModuleType, component: Any, cluster_ids: Sequence[str]) -> Any:
    allowed = {str(value) for value in cluster_ids}
    keep = component.frame["cluster_id"].astype(str).isin(allowed)
    frame = component.frame.loc[keep].copy().reset_index(drop=True)
    support_mask = component.support_mask.loc[keep].copy().reset_index(drop=True).astype(bool)
    return b0.PreparedComponent(
        record_id=component.record_id,
        control_family=component.control_family,
        frame=frame,
        support_mask=support_mask,
        center_offset=component.center_offset,
        baseline_contrast=component.baseline_contrast,
        native_scale=component.native_scale,
    )


def subset_components(
    b0: ModuleType,
    components: Sequence[Any],
    cluster_ids: Sequence[str],
) -> list[Any]:
    return [subset_component(b0, component, cluster_ids) for component in components]


# -----------------------------------------------------------------------------
# Exact sign-flip and efficient cluster uncertainty
# -----------------------------------------------------------------------------


def subset_sums(values: np.ndarray) -> np.ndarray:
    sums = np.asarray([0.0], dtype=float)
    for value in np.asarray(values, dtype=float):
        sums = np.concatenate([sums, sums + float(value)])
    return sums


def exact_one_sided_sign_flip_p(values: np.ndarray, tolerance: float = 1e-15) -> float:
    vector = np.asarray(values, dtype=float)
    vector = vector[np.isfinite(vector)]
    n = int(vector.size)
    if n < 2:
        return 1.0
    midpoint = n // 2
    left = subset_sums(vector[:midpoint])
    right = np.sort(subset_sums(vector[midpoint:]))
    count = 0
    for left_sum in left:
        count += int(np.searchsorted(right, tolerance - left_sum, side="right"))
    return float(count / (2**n))


def cluster_uncertainty(
    values: np.ndarray,
    frozen: FrozenGateParameters,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=float)
    vector = vector[np.isfinite(vector)]
    n = int(vector.size)
    if n < 2:
        return {
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "bootstrap_positive_share": float("nan"),
            "bootstrap_method": "unavailable_fewer_than_two_clusters",
            "bootstrap_draw_count": 0,
            "cluster_uncertainty_decision_status": "exact_insufficient_clusters",
            "direction_consistency": float("nan"),
            "loo_successful_count": 0,
            "independent_cluster_count": n,
            "raw_permutation_p": 1.0,
            "permutation_count": 0,
            "permutation_method": "unavailable_fewer_than_two_clusters",
            "cluster_uncertainty_pass": False,
        }

    total = float(vector.sum())
    loo = (total - vector) / (n - 1)
    finite_loo = loo[np.isfinite(loo)]
    direction_consistency = (
        float(np.mean(finite_loo > 0)) if finite_loo.size else float("nan")
    )
    raw_p = exact_one_sided_sign_flip_p(vector)

    if np.all(vector > 0):
        ci_low = float(np.min(vector))
        ci_high = float(np.max(vector))
        positive_share = 1.0
        method = "exact_positive_support_certificate"
        draw_count = 0
        decision_status = "exact_positive_certificate"
    elif (
        finite_loo.size < 2
        or not math.isfinite(direction_consistency)
        or direction_consistency < frozen.minimum_direction_consistency
        or float(vector.mean()) <= 0
    ):
        ci_low = float("nan")
        ci_high = float("nan")
        positive_share = float("nan")
        method = "short_circuit_exact_direction_failure"
        draw_count = 0
        decision_status = "exact_direction_failure"
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=(bootstrap_draws, n), endpoint=False)
        means = vector[indices].mean(axis=1)
        ci_low = float(np.quantile(means, frozen.alpha / 2.0))
        ci_high = float(np.quantile(means, 1.0 - frozen.alpha / 2.0))
        positive_share = float(np.mean(means > 0))
        method = "deterministic_percentile_cluster_bootstrap"
        draw_count = int(bootstrap_draws)
        decision_status = "mc_percentile_ambiguous_vector"

    passed = bool(
        math.isfinite(ci_low)
        and ci_low > 0
        and math.isfinite(direction_consistency)
        and direction_consistency >= frozen.minimum_direction_consistency
        and int(finite_loo.size) >= 2
    )
    return {
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_positive_share": positive_share,
        "bootstrap_method": method,
        "bootstrap_draw_count": draw_count,
        "cluster_uncertainty_decision_status": decision_status,
        "direction_consistency": direction_consistency,
        "loo_successful_count": int(finite_loo.size),
        "independent_cluster_count": n,
        "raw_permutation_p": raw_p,
        "permutation_count": int(2**n),
        "permutation_method": "exact_meet_in_the_middle_object_cluster_sign_flip",
        "cluster_uncertainty_pass": passed,
    }


# -----------------------------------------------------------------------------
# Frozen gate evaluation on a prospective campaign prefix
# -----------------------------------------------------------------------------


def support_cluster_counts(component: Any) -> tuple[int, int]:
    clusters = component.frame["cluster_id"].astype(str)
    support = component.support_mask.astype(bool)
    return int(clusters[support].nunique()), int(clusters[~support].nunique())


def evaluate_campaign_prefix(
    b0: ModuleType,
    b: ModuleType,
    original_address: Any,
    simulator: Any,
    contract: Any,
    scenario_id: str,
    base_scenario_id: str,
    delta: float,
    control_response: float,
    replicate: int,
    seed: int,
    source_ids: Sequence[str],
    prospective_ids: Sequence[str],
    simulated_components: Sequence[Any],
    injected_components: Sequence[Any],
    frozen: FrozenGateParameters,
    bootstrap_draws: int,
) -> dict[str, Any]:
    before = subset_components(b0, simulated_components, prospective_ids)
    after = subset_components(b0, injected_components, prospective_ids)
    target_before = before[0]
    target = after[0]
    controls = after[1:]
    metric = str(contract.metric)

    target_before_contrast = b0.site_relative_contrast(
        target_before.frame,
        target_before.support_mask,
        metric,
    )
    target_contrast = b0.site_relative_contrast(
        target.frame,
        target.support_mask,
        metric,
    )
    target_response = (
        float(target_contrast - target_before_contrast)
        if math.isfinite(target_contrast) and math.isfinite(target_before_contrast)
        else float("nan")
    )

    control_contrasts = [
        b0.site_relative_contrast(component.frame, component.support_mask, metric)
        for component in controls
    ]
    finite_controls = [value for value in control_contrasts if math.isfinite(value)]
    median_control = float(np.median(finite_controls)) if finite_controls else float("nan")
    control_adjusted = (
        float(target_contrast - median_control)
        if math.isfinite(target_contrast) and math.isfinite(median_control)
        else float("nan")
    )
    individual_adjusted = [
        float(target_contrast - value)
        for value in finite_controls
        if math.isfinite(target_contrast)
    ]
    positive_control_share = (
        float(np.mean(np.asarray(individual_adjusted) > 0))
        if individual_adjusted
        else float("nan")
    )

    site_rows = int(target.support_mask.sum())
    complement_rows = int((~target.support_mask).sum())
    site_clusters, complement_clusters = support_cluster_counts(target)
    shared_clusters = len(prospective_ids)

    vector = b.cluster_contrast_vector(
        b0,
        target,
        metric,
        prospective_ids,
    )
    uncertainty = cluster_uncertainty(
        vector,
        frozen,
        bootstrap_draws,
        stable_seed(seed, "cluster_uncertainty", len(prospective_ids)),
    )

    support_available = bool(metric in target.frame.columns and site_rows > 0)
    complement_admissible = bool(
        site_rows >= frozen.minimum_site_rows
        and complement_rows >= frozen.minimum_complement_rows
        and shared_clusters >= frozen.minimum_shared_clusters
        and site_clusters >= frozen.minimum_shared_clusters
        and complement_clusters >= frozen.minimum_shared_clusters
    )
    effect_direction = bool(
        delta > 0
        and math.isfinite(target_response)
        and target_response > frozen.effect_direction_tolerance
    )
    target_positive = bool(math.isfinite(target_contrast) and target_contrast > 0)
    minimum_effect = bool(
        math.isfinite(target_contrast)
        and target_contrast >= frozen.minimum_effect
    )
    raw_p = float(uncertainty["raw_permutation_p"])
    raw_threshold = bool(math.isfinite(raw_p) and raw_p <= frozen.alpha)
    m1_adjusted_p = raw_p
    multiplicity_threshold = bool(
        math.isfinite(m1_adjusted_p) and m1_adjusted_p <= frozen.alpha
    )
    control_adjusted_pass = bool(
        math.isfinite(control_adjusted)
        and control_adjusted >= frozen.minimum_control_adjusted_effect
    )
    control_specificity_pass = bool(
        math.isfinite(positive_control_share)
        and positive_control_share >= frozen.minimum_positive_control_share
    )

    gates = {
        "support_available_pass": support_available,
        "complement_admissible_pass": complement_admissible,
        "effect_direction_reproduced_pass": effect_direction,
        "target_contrast_positive_pass": target_positive,
        "minimum_effect_pass": minimum_effect,
        "cluster_uncertainty_pass": bool(uncertainty["cluster_uncertainty_pass"]),
        "raw_statistical_threshold_pass": raw_threshold,
        "multiplicity_adjusted_threshold_pass": multiplicity_threshold,
        "control_adjusted_contrast_pass": control_adjusted_pass,
        "control_specificity_pass": control_specificity_pass,
    }
    overall = all(gates.values())
    failed = [name for name in GATE_ORDER if not gates[name]]

    range_violations = sum(int(b0.range_violation_count(component.frame)) for component in after)
    identity_violations = sum(
        int(b0.identity_hash(before_component.frame) != b0.identity_hash(after_component.frame))
        for before_component, after_component in zip(before, after)
    )

    cluster_count = len(prospective_ids)
    unique_sources = len(set(str(value) for value in source_ids))
    min_p = theoretical_minimum_p(cluster_count)
    effective_count = int(uncertainty["independent_cluster_count"])
    effective_min_p = (
        theoretical_minimum_p(effective_count) if effective_count >= 1 else 1.0
    )
    payload = {
        "scenario_id": scenario_id,
        "base_scenario_id": base_scenario_id,
        "address_id": str(original_address.address_id),
        "record_id": str(original_address.record_id),
        "support_id": str(original_address.support_id),
        "relation": str(original_address.relation),
        "carrier": str(original_address.carrier),
        "entitlement_status": str(original_address.entitlement_status),
        "partition": str(original_address.partition),
        "simulator_id": str(simulator.simulator_id),
        "failure_predicate": str(original_address.failure_predicate),
        "prospective_cluster_count": cluster_count,
        "theoretical_sign_configurations": 2**cluster_count,
        "minimum_attainable_exact_p": min_p,
        "structurally_attainable": min_p <= frozen.alpha,
        "effective_minimum_attainable_exact_p": effective_min_p,
        "effective_resolution_attainable": effective_min_p <= frozen.alpha,
        "delta": float(delta),
        "control_response_lambda": float(control_response),
        "replicate": int(replicate),
        "seed": int(seed),
        "source_template_count": len(source_ids),
        "source_template_unique_count": unique_sources,
        "source_template_reuse_fraction": float(1.0 - unique_sources / len(source_ids)),
        "source_template_ids_json": canonical_json(list(source_ids)),
        "target_contrast": target_contrast,
        "target_contrast_before_injection": target_before_contrast,
        "target_response_from_simulated_null": target_response,
        "median_control_contrast": median_control,
        "control_adjusted_contrast": control_adjusted,
        "positive_control_adjusted_share": positive_control_share,
        "bootstrap_ci_low": uncertainty["bootstrap_ci_low"],
        "bootstrap_ci_high": uncertainty["bootstrap_ci_high"],
        "bootstrap_positive_share": uncertainty["bootstrap_positive_share"],
        "bootstrap_method": uncertainty["bootstrap_method"],
        "bootstrap_draw_count": uncertainty["bootstrap_draw_count"],
        "cluster_uncertainty_decision_status": uncertainty[
            "cluster_uncertainty_decision_status"
        ],
        "direction_consistency": uncertainty["direction_consistency"],
        "loo_successful_count": uncertainty["loo_successful_count"],
        "independent_cluster_count": uncertainty["independent_cluster_count"],
        "raw_permutation_p": raw_p,
        "m1_adjusted_p": m1_adjusted_p,
        "permutation_count": uncertainty["permutation_count"],
        "permutation_method": uncertainty["permutation_method"],
        "target_site_rows": site_rows,
        "target_complement_rows": complement_rows,
        "target_site_clusters": site_clusters,
        "target_complement_clusters": complement_clusters,
        "shared_cluster_count": shared_clusters,
        "range_violation_count": range_violations,
        "identity_violation_count": identity_violations,
        **gates,
        "overall_gate_pass": overall,
        "failed_gates_json": canonical_json(failed),
    }
    payload["replicate_hash"] = stable_id(
        *[payload[column] for column in REPLICATE_COLUMNS if column != "replicate_hash"],
        prefix="CR-",
    )
    return payload


# -----------------------------------------------------------------------------
# Sharded replicate retention and online summaries
# -----------------------------------------------------------------------------


class ShardedReplicateWriter:
    def __init__(
        self,
        directory: Path | None,
        cluster_grid: Sequence[int],
        chunk_size: int,
    ) -> None:
        self.directory = directory
        self.chunk_size = chunk_size
        self.buffers: dict[int, list[dict[str, Any]]] = {k: [] for k in cluster_grid}
        self.handles: dict[int, TextIO] = {}
        self.writers: dict[int, csv.DictWriter[str]] = {}
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            for cluster_count in cluster_grid:
                path = directory / f"obs085c_replicates_k{cluster_count:02d}.csv.gz"
                handle = gzip.open(path, "wt", encoding="utf-8", newline="")
                writer = csv.DictWriter(handle, fieldnames=REPLICATE_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                self.handles[cluster_count] = handle
                self.writers[cluster_count] = writer

    def add(self, row: dict[str, Any]) -> None:
        cluster_count = int(row["prospective_cluster_count"])
        if cluster_count not in self.writers:
            return
        self.buffers[cluster_count].append(row)
        if len(self.buffers[cluster_count]) >= self.chunk_size:
            self.flush(cluster_count)

    def flush(self, cluster_count: int) -> None:
        if cluster_count not in self.writers or not self.buffers[cluster_count]:
            return
        self.writers[cluster_count].writerows(self.buffers[cluster_count])
        self.buffers[cluster_count].clear()

    def close(self) -> None:
        for cluster_count in list(self.writers):
            self.flush(cluster_count)
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.writers.clear()

    def __enter__(self) -> "ShardedReplicateWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class OnlineSummary:
    def __init__(self) -> None:
        self.cells: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.failure_combinations: defaultdict[tuple[Any, ...], int] = defaultdict(int)
        self.ordered_pass: defaultdict[tuple[Any, ...], int] = defaultdict(int)
        self.uncertainty_status: defaultdict[tuple[Any, ...], int] = defaultdict(int)

    @staticmethod
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["scenario_id"],
            row["base_scenario_id"],
            row["address_id"],
            row["record_id"],
            row["support_id"],
            row["relation"],
            row["carrier"],
            row["entitlement_status"],
            row["partition"],
            row["simulator_id"],
            row["failure_predicate"],
            int(row["prospective_cluster_count"]),
            float(row["delta"]),
            float(row["control_response_lambda"]),
        )

    def add(self, row: Mapping[str, Any]) -> None:
        key = self.key(row)
        if key not in self.cells:
            self.cells[key] = {
                "metadata": {
                    "scenario_id": row["scenario_id"],
                    "base_scenario_id": row["base_scenario_id"],
                    "address_id": row["address_id"],
                    "record_id": row["record_id"],
                    "support_id": row["support_id"],
                    "relation": row["relation"],
                    "carrier": row["carrier"],
                    "entitlement_status": row["entitlement_status"],
                    "partition": row["partition"],
                    "simulator_id": row["simulator_id"],
                    "failure_predicate": row["failure_predicate"],
                    "prospective_cluster_count": int(row["prospective_cluster_count"]),
                    "minimum_attainable_exact_p": row["minimum_attainable_exact_p"],
                    "structurally_attainable": row["structurally_attainable"],
                    "delta": float(row["delta"]),
                    "control_response_lambda": float(row["control_response_lambda"]),
                },
                "n": 0,
                "overall": 0,
                "gate_counts": {gate: 0 for gate in GATE_ORDER},
                "numeric_sum": {field: 0.0 for field in NUMERIC_SUMMARY_FIELDS},
                "numeric_n": {field: 0 for field in NUMERIC_SUMMARY_FIELDS},
                "effective_attainable": 0,
                "numeric_min": {field: float("inf") for field in NUMERIC_SUMMARY_FIELDS},
                "numeric_max": {field: float("-inf") for field in NUMERIC_SUMMARY_FIELDS},
            }
        cell = self.cells[key]
        cell["n"] += 1
        cell["overall"] += int(normalize_bool(row["overall_gate_pass"]))
        cell["effective_attainable"] += int(
            normalize_bool(row["effective_resolution_attainable"])
        )
        for gate in GATE_ORDER:
            cell["gate_counts"][gate] += int(normalize_bool(row[gate]))
        for field in NUMERIC_SUMMARY_FIELDS:
            value = float(row[field])
            if math.isfinite(value):
                cell["numeric_sum"][field] += value
                cell["numeric_n"][field] += 1
                cell["numeric_min"][field] = min(cell["numeric_min"][field], value)
                cell["numeric_max"][field] = max(cell["numeric_max"][field], value)

        combo_key = (
            row["partition"],
            row["simulator_id"],
            int(row["prospective_cluster_count"]),
            float(row["delta"]),
            float(row["control_response_lambda"]),
            row["failed_gates_json"],
        )
        self.failure_combinations[combo_key] += 1

        cumulative = True
        for order, gate in enumerate(GATE_ORDER, start=1):
            cumulative = cumulative and normalize_bool(row[gate])
            ordered_key = (
                row["partition"],
                row["simulator_id"],
                int(row["prospective_cluster_count"]),
                float(row["delta"]),
                float(row["control_response_lambda"]),
                order,
                gate,
            )
            self.ordered_pass[ordered_key] += int(cumulative)

        status_key = (
            row["partition"],
            row["simulator_id"],
            int(row["prospective_cluster_count"]),
            row["cluster_uncertainty_decision_status"],
        )
        self.uncertainty_status[status_key] += 1

    def finalize(self) -> pd.DataFrame:
        rows = []
        for cell in self.cells.values():
            n = int(cell["n"])
            successes = int(cell["overall"])
            low, high = wilson_interval(successes, n)
            row = {
                **cell["metadata"],
                "replicates": n,
                "overall_gate_pass_count": successes,
                "conditional_gate_passage_probability": successes / n if n else float("nan"),
                "monte_carlo_wilson_low": low,
                "monte_carlo_wilson_high": high,
                "empirically_passable": successes > 0,
                "effective_resolution_attainable_count": int(cell["effective_attainable"]),
                "effective_resolution_attainable_probability": (
                    cell["effective_attainable"] / n if n else float("nan")
                ),
            }
            for gate in GATE_ORDER:
                count = int(cell["gate_counts"][gate])
                row[f"{gate}_count"] = count
                row[f"{gate}_probability"] = count / n if n else float("nan")
            for field in NUMERIC_SUMMARY_FIELDS:
                count = int(cell["numeric_n"][field])
                row[f"mean_{field}"] = (
                    cell["numeric_sum"][field] / count if count else float("nan")
                )
                row[f"min_{field}"] = (
                    cell["numeric_min"][field] if count else float("nan")
                )
                row[f"max_{field}"] = (
                    cell["numeric_max"][field] if count else float("nan")
                )
            rows.append(row)
        return pd.DataFrame(rows).sort_values(
            [
                "partition",
                "simulator_id",
                "address_id",
                "control_response_lambda",
                "delta",
                "prospective_cluster_count",
            ]
        )

    def failure_combination_frame(self) -> pd.DataFrame:
        rows = [
            {
                "partition": key[0],
                "simulator_id": key[1],
                "prospective_cluster_count": key[2],
                "delta": key[3],
                "control_response_lambda": key[4],
                "failed_gates_json": key[5],
                "replicate_count": count,
            }
            for key, count in self.failure_combinations.items()
        ]
        return pd.DataFrame(rows).sort_values(
            [
                "partition",
                "simulator_id",
                "prospective_cluster_count",
                "delta",
                "control_response_lambda",
                "replicate_count",
            ],
            ascending=[True, True, True, True, True, False],
        )

    def ordered_gate_frame(self) -> pd.DataFrame:
        denominators: defaultdict[tuple[Any, ...], int] = defaultdict(int)
        for cell in self.cells.values():
            metadata = cell["metadata"]
            denominator_key = (
                metadata["partition"],
                metadata["simulator_id"],
                metadata["prospective_cluster_count"],
                metadata["delta"],
                metadata["control_response_lambda"],
            )
            denominators[denominator_key] += int(cell["n"])
        rows = []
        for key, count in self.ordered_pass.items():
            denominator_key = key[:5]
            trials = denominators[denominator_key]
            rows.append(
                {
                    "partition": key[0],
                    "simulator_id": key[1],
                    "prospective_cluster_count": key[2],
                    "delta": key[3],
                    "control_response_lambda": key[4],
                    "gate_order": key[5],
                    "gate_name": key[6],
                    "cumulative_pass_count": count,
                    "replicates": trials,
                    "cumulative_pass_probability": count / trials if trials else float("nan"),
                }
            )
        return pd.DataFrame(rows).sort_values(
            [
                "partition",
                "simulator_id",
                "prospective_cluster_count",
                "delta",
                "control_response_lambda",
                "gate_order",
            ]
        )

    def uncertainty_status_frame(self) -> pd.DataFrame:
        rows = [
            {
                "partition": key[0],
                "simulator_id": key[1],
                "prospective_cluster_count": key[2],
                "cluster_uncertainty_decision_status": key[3],
                "replicate_count": count,
            }
            for key, count in self.uncertainty_status.items()
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        totals = frame.groupby(
            ["partition", "simulator_id", "prospective_cluster_count"]
        )["replicate_count"].transform("sum")
        frame["replicate_share"] = frame["replicate_count"] / totals
        return frame.sort_values(
            ["partition", "simulator_id", "prospective_cluster_count", "replicate_count"],
            ascending=[True, True, True, False],
        )


# -----------------------------------------------------------------------------
# Derived summaries
# -----------------------------------------------------------------------------


def gate_failure_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    identity = [
        "scenario_id",
        "address_id",
        "partition",
        "simulator_id",
        "prospective_cluster_count",
        "delta",
        "control_response_lambda",
        "replicates",
    ]
    for row in summary.itertuples(index=False):
        values = row._asdict()
        for gate in GATE_ORDER:
            probability = float(values[f"{gate}_probability"])
            rows.append(
                {
                    **{column: values[column] for column in identity},
                    "gate_name": gate,
                    "gate_order": GATE_ORDER.index(gate) + 1,
                    "pass_probability": probability,
                    "failure_probability": 1.0 - probability,
                }
            )
    return pd.DataFrame(rows)


def campaign_curve_summary(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby(
        [
            "partition",
            "simulator_id",
            "prospective_cluster_count",
            "control_response_lambda",
            "delta",
        ],
        dropna=False,
    )["conditional_gate_passage_probability"]
    return grouped.agg(
        addresses="size",
        macro_mean_gate_passage_probability="mean",
        minimum_address_probability="min",
        median_address_probability="median",
        maximum_address_probability="max",
    ).reset_index()


def minimum_required_clusters(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "address_id",
        "record_id",
        "support_id",
        "relation",
        "carrier",
        "entitlement_status",
        "partition",
        "simulator_id",
        "delta",
        "control_response_lambda",
    ]
    for key, group in summary.groupby(group_columns, dropna=False):
        ordered = group.sort_values("prospective_cluster_count")
        base = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        for target in RELIABILITY_TARGETS:
            reached = ordered[
                ordered["conditional_gate_passage_probability"].ge(target)
            ]
            if reached.empty:
                rows.append(
                    {
                        **base,
                        "target_gate_passage_probability": target,
                        "minimum_tested_cluster_count": float("nan"),
                        "threshold_status": "not_reached_on_tested_grid",
                    }
                )
            else:
                first = reached.iloc[0]
                rows.append(
                    {
                        **base,
                        "target_gate_passage_probability": target,
                        "minimum_tested_cluster_count": int(first["prospective_cluster_count"]),
                        "threshold_status": "reached_on_tested_grid",
                    }
                )
    return pd.DataFrame(rows)


def attainability_map(
    design: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    empirical = summary.groupby("prospective_cluster_count").agg(
        simulated_cells=("scenario_id", "size"),
        empirically_passable_cells=("empirically_passable", "sum"),
        maximum_gate_passage_probability=("conditional_gate_passage_probability", "max"),
        mean_gate_passage_probability=("conditional_gate_passage_probability", "mean"),
        minimum_observed_raw_p=("min_raw_permutation_p", "min"),
        maximum_effective_cluster_count=("max_independent_cluster_count", "max"),
        mean_effective_resolution_attainability=(
            "effective_resolution_attainable_probability", "mean"
        ),
    ).reset_index()
    result = design.merge(empirical, on="prospective_cluster_count", how="left", validate="one_to_one")
    result["empirically_passable"] = result["empirically_passable_cells"].fillna(0).gt(0)
    return result


def simulator_envelope(summary: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "address_id",
        "partition",
        "prospective_cluster_count",
        "delta",
        "control_response_lambda",
    ]
    grouped = summary.groupby(group_columns, dropna=False)[
        "conditional_gate_passage_probability"
    ]
    frame = grouped.agg(
        simulator_count="size",
        minimum_simulator_probability="min",
        maximum_simulator_probability="max",
        mean_simulator_probability="mean",
    ).reset_index()
    frame["between_simulator_spread"] = (
        frame["maximum_simulator_probability"] - frame["minimum_simulator_probability"]
    )
    return frame


def entitlement_overlay(summary: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "entitlement_status",
        "partition",
        "simulator_id",
        "prospective_cluster_count",
        "delta",
        "control_response_lambda",
    ]
    return summary.groupby(group_columns, dropna=False).agg(
        addresses=("address_id", "nunique"),
        mean_gate_passage_probability=("conditional_gate_passage_probability", "mean"),
        maximum_gate_passage_probability=("conditional_gate_passage_probability", "max"),
    ).reset_index()


def source_template_audit(summary: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "address_id",
        "partition",
        "simulator_id",
        "prospective_cluster_count",
    ]
    return summary.groupby(group_columns, dropna=False).agg(
        mean_unique_source_templates=("mean_source_template_unique_count", "mean"),
        minimum_unique_source_templates=("min_source_template_unique_count", "min"),
        maximum_unique_source_templates=("max_source_template_unique_count", "max"),
        mean_template_reuse_fraction=("mean_source_template_reuse_fraction", "mean"),
        mean_effective_clusters=("mean_independent_cluster_count", "mean"),
        minimum_effective_clusters=("min_independent_cluster_count", "min"),
        maximum_effective_clusters=("max_independent_cluster_count", "max"),
    ).reset_index()


# -----------------------------------------------------------------------------
# Outputs, report, and manifest
# -----------------------------------------------------------------------------


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "obs085c_manifest.json": output_dir / "obs085c_manifest.json",
        "obs085c_input_manifest.csv": output_dir / "obs085c_input_manifest.csv",
        "obs085c_authorized_cells.csv": output_dir / "obs085c_authorized_cells.csv",
        "obs085c_address_manifest.csv": output_dir / "obs085c_address_manifest.csv",
        "obs085c_gate_contract.csv": output_dir / "obs085c_gate_contract.csv",
        "obs085c_design_manifest.csv": output_dir / "obs085c_design_manifest.csv",
        "obs085c_scenario_manifest.csv": output_dir / "obs085c_scenario_manifest.csv",
        "obs085c_gate_passage_summary.csv": output_dir / "obs085c_gate_passage_summary.csv",
        "obs085c_gate_failure_summary.csv": output_dir / "obs085c_gate_failure_summary.csv",
        "obs085c_failure_combinations.csv": output_dir / "obs085c_failure_combinations.csv",
        "obs085c_ordered_gate_passage.csv": output_dir / "obs085c_ordered_gate_passage.csv",
        "obs085c_campaign_curve_summary.csv": output_dir / "obs085c_campaign_curve_summary.csv",
        "obs085c_attainability_map.csv": output_dir / "obs085c_attainability_map.csv",
        "obs085c_minimum_required_clusters.csv": output_dir / "obs085c_minimum_required_clusters.csv",
        "obs085c_simulator_envelope.csv": output_dir / "obs085c_simulator_envelope.csv",
        "obs085c_entitlement_overlay.csv": output_dir / "obs085c_entitlement_overlay.csv",
        "obs085c_source_template_audit.csv": output_dir / "obs085c_source_template_audit.csv",
        "obs085c_uncertainty_engine_audit.csv": output_dir / "obs085c_uncertainty_engine_audit.csv",
        "obs085c_failures.csv": output_dir / "obs085c_failures.csv",
        "obs085c_report.md": output_dir / "obs085c_report.md",
        "replicate_directory": output_dir / "replicates",
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def build_input_manifest(paths: Mapping[str, Path], repo_root: Path) -> pd.DataFrame:
    rows = []
    for role, path in paths.items():
        if path.exists() and path.is_file():
            rows.append(
                {
                    "artifact_role": role,
                    "artifact_path": repo_relative(path, repo_root),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows).sort_values("artifact_role")


def artifact_inventory(
    outputs: Mapping[str, Path],
    repo_root: Path,
) -> list[dict[str, Any]]:
    artifacts = []
    for name, path in sorted(outputs.items()):
        if name == "obs085c_manifest.json":
            continue
        if path.is_file():
            artifacts.append(
                {
                    "artifact_path": repo_relative(path, repo_root),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        elif path.is_dir():
            for child in sorted(path.glob("*.csv.gz")):
                artifacts.append(
                    {
                        "artifact_path": repo_relative(child, repo_root),
                        "size_bytes": child.stat().st_size,
                        "sha256": sha256_file(child),
                    }
                )
    return artifacts


def write_report(
    path: Path,
    state: str,
    lineage: Mapping[str, Any],
    design: pd.DataFrame,
    scenarios: pd.DataFrame,
    curves: pd.DataFrame,
    attainability: pd.DataFrame,
    thresholds: pd.DataFrame,
    uncertainty_audit: pd.DataFrame,
    failures: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    endpoint = curves[
        curves["delta"].eq(curves["delta"].max())
        & curves["control_response_lambda"].eq(0.0)
    ].copy()
    threshold_counts = (
        thresholds.groupby(
            [
                "partition",
                "simulator_id",
                "target_gate_passage_probability",
                "threshold_status",
            ],
            dropna=False,
        )
        .size()
        .rename("address_effect_control_cells")
        .reset_index()
    )
    lines = [
        "# OBS-085c — Prospective Campaign Attainability Simulation",
        "",
        "## State",
        "",
        f"`{state}`",
        "",
        "OBS-085c prospectively varies independent object-cluster support while preserving the frozen OBS-085b evidence thresholds and qualified simulator scope.",
        "",
        "## Frozen lineage",
        "",
        f"- OBS-085b manifest ID: `{lineage['obs085b_manifest_id']}`",
        f"- OBS-085b script SHA256: `{lineage['obs085b_script_sha256']}`",
        f"- OBS-085b output hashes checked: **{lineage['obs085b_output_hashes_checked']}**",
        f"- OBS-085b0 commit: `{lineage['obs085b0_commit']}`",
        f"- OBS-085b commit: `{lineage['obs085b_commit']}`",
        f"- Current repository HEAD: `{lineage['current_repo_head']}`",
        "",
        "## Prospective design intervention",
        "",
        "Future object clusters are generated by joint cluster-template replication across the frozen target and controls. Smaller object-count designs are nested prefixes of the same maximum-count campaign replicate.",
        "",
        "> Repeated templates are prospective exchangeable draws, not additional observed evidence.",
        "",
        "## Mathematical attainability",
        "",
        markdown_table(design, args.max_report_rows),
        "",
        "The first exact one-sided sign-flip design capable of reaching alpha=0.10 is k=4, where the minimum attainable p-value is 0.0625.",
        "",
        "## Execution design",
        "",
        f"- Prospective object counts: `{args.cluster_grid}`",
        f"- Frozen base scenarios: **{scenarios['base_scenario_id'].nunique()}**",
        f"- Nested scenario rows: **{len(scenarios)}**",
        f"- Replicates per base cell: **{args.replicates:,}**",
        f"- Ambiguous-vector bootstrap draws: **{args.bootstrap_draws:,}**",
        f"- Complete replicate vectors retained: **{args.write_replicates}**",
        "",
        "## Empirical attainability map",
        "",
        markdown_table(attainability, args.max_report_rows),
        "",
        "## Highest tested effect with no control response",
        "",
        markdown_table(endpoint, args.max_report_rows),
        "",
        "## Tested-grid reliability thresholds",
        "",
        markdown_table(threshold_counts, args.max_report_rows),
        "",
        "No interpolation or extrapolation is used. A minimum required cluster count is reported only when the requested passage target is reached on the tested grid.",
        "",
        "## Uncertainty-engine audit",
        "",
        markdown_table(uncertainty_audit, args.max_report_rows),
        "",
        "## Failures",
        "",
        markdown_table(failures, args.max_report_rows),
        "",
        "## Interpretation boundary",
        "",
        "> Structural attainability is not reliable passage.",
        "",
        "> Prospective gate passage is conditional on the frozen address, support, simulator, injection, control, and evidence-gate contracts.",
        "",
        "> This design simulation cannot create an observed FL3 witness or increase frozen claim entitlement.",
        "",
        "> Between-simulator spread is a model-sensitivity diagnostic, not a confidence interval.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    repo_root: Path,
    outputs: Mapping[str, Path],
    state: str,
    lineage: Mapping[str, Any],
    authorized: pd.DataFrame,
    addresses: pd.DataFrame,
    design: pd.DataFrame,
    scenarios: pd.DataFrame,
    summary: pd.DataFrame,
    frozen: FrozenGateParameters,
    args: argparse.Namespace,
    expected_rows: int,
    written_rows: int,
    base_simulation_count: int,
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "state": state,
        "scope": "prospective independent-object campaign attainability metrology",
        "claim_ceiling": (
            "prospective conditional design simulation only; no observed witness, "
            "classical power, causal attribution, simulator truth, or entitlement increase"
        ),
        "frozen_lineage": dict(lineage),
        "authorized_cells": authorized[
            ["failure_predicate", "partition", "simulator_id", "qualification_status"]
        ].to_dict("records"),
        "frozen_address_ids": sorted(addresses["address_id"].astype(str).unique()),
        "prospective_cluster_grid": [int(value) for value in design["prospective_cluster_count"]],
        "scenario_ids": sorted(scenarios["scenario_id"].astype(str).unique()),
        "execution": {
            "master_seed": args.master_seed,
            "replicates_per_base_cell": args.replicates,
            "base_simulation_count": base_simulation_count,
            "expected_nested_replicate_rows": expected_rows,
            "written_nested_replicate_rows": written_rows,
            "bootstrap_draws_for_ambiguous_vectors": args.bootstrap_draws,
            "smoke": args.smoke,
            "address_limit": args.address_limit,
            "complete_replicate_vectors_written": args.write_replicates,
            "replicate_sharding": "one gzip CSV per prospective cluster count",
        },
        "frozen_gate_contract": asdict(frozen),
        "prospective_expansion_contract": {
            "dependence_unit": "synthetic prospective object cluster",
            "source_policy": "joint cluster-template bootstrap with replacement",
            "nesting": "requested k values are prefixes of one maximum-k campaign replicate",
            "gaussian_extension": "frozen original-address covariance fit; new joint residual draws per prospective object",
            "wild_extension": "independent Rademacher multiplier per prospective object on jointly cloned source residual template",
            "sign_flip": "exact meet-in-the-middle subset count",
            "uncertainty": "exact certificates plus deterministic percentile bootstrap only for ambiguous vectors",
        },
        "summary_row_count": len(summary),
        "output_artifacts": artifact_inventory(outputs, repo_root),
        "mandatory_statements": [
            "OBS-085b remains frozen and structurally unattainable with at most three effective clusters.",
            "Prospective template replication is not additional observed evidence.",
            "Structural attainability is not reliable gate passage.",
            "Monte Carlo precision is not simulator-model certainty.",
            "Between-simulator spread is not a confidence interval.",
            "OBS-085c cannot increase frozen claim entitlement.",
        ],
    }
    manifest_id = sha256_bytes(canonical_json(core).encode("utf-8"))
    return {"obs085c_manifest_id": manifest_id, **core}


# -----------------------------------------------------------------------------
# Simulation orchestration
# -----------------------------------------------------------------------------


def run_simulation(
    b0: ModuleType,
    b: ModuleType,
    prepared: Sequence[Any],
    simulators: Sequence[Any],
    contract: Any,
    base_scenarios: pd.DataFrame,
    scenario_lookup: Mapping[tuple[str, int], str],
    cluster_grid: Sequence[int],
    frozen: FrozenGateParameters,
    args: argparse.Namespace,
    replicate_directory: Path | None,
) -> tuple[OnlineSummary, int, int, list[StudyFailure]]:
    accumulator = OnlineSummary()
    failures: list[StudyFailure] = []
    written = 0
    base_simulations = 0
    max_k = max(cluster_grid)
    completed_pairs = 0
    total_pairs = len(prepared) * len(simulators)

    with ShardedReplicateWriter(
        replicate_directory,
        cluster_grid,
        args.replicate_chunk_size,
    ) as writer:
        for address in prepared:
            for simulator in simulators:
                for scenario in base_scenarios.itertuples(index=False):
                    for replicate in range(args.replicates):
                        base_seed = stable_seed(
                            args.master_seed,
                            address.address_id,
                            address.partition,
                            simulator.simulator_id,
                            str(scenario.scenario_id),
                            replicate,
                        )
                        scope = (
                            f"{address.address_id}::{address.partition}::"
                            f"{simulator.simulator_id}::{scenario.scenario_id}::{replicate}"
                        )
                        try:
                            expanded, source_ids, prospective_ids = expand_address_joint_templates(
                                b0,
                                address,
                                max_k,
                                np.random.default_rng(stable_seed(base_seed, "templates")),
                            )
                            simulated = simulate_prospective_components(
                                b0,
                                address,
                                expanded,
                                simulator,
                                np.random.default_rng(stable_seed(base_seed, "simulator")),
                            )
                            injected = b0.apply_injection(
                                expanded,
                                simulated,
                                float(scenario.delta),
                                float(scenario.control_response_lambda),
                                np.random.default_rng(stable_seed(base_seed, "injection")),
                            )
                            base_simulations += 1
                            for cluster_count in cluster_grid:
                                prefix_ids = prospective_ids[:cluster_count]
                                prefix_sources = source_ids[:cluster_count]
                                row = evaluate_campaign_prefix(
                                    b0,
                                    b,
                                    address,
                                    simulator,
                                    contract,
                                    scenario_lookup[(str(scenario.scenario_id), cluster_count)],
                                    str(scenario.scenario_id),
                                    float(scenario.delta),
                                    float(scenario.control_response_lambda),
                                    replicate,
                                    base_seed,
                                    prefix_sources,
                                    prefix_ids,
                                    simulated,
                                    injected,
                                    frozen,
                                    args.bootstrap_draws,
                                )
                                accumulator.add(row)
                                writer.add(row)
                                written += 1
                        except Exception as exc:
                            failures.append(
                                StudyFailure(
                                    stage="prospective_campaign_simulation",
                                    scope_id=scope,
                                    reason="base_campaign_failed",
                                    detail=str(exc),
                                    severity="error",
                                )
                            )
                completed_pairs += 1
                print(
                    "[OBS-085c] completed address-simulator "
                    f"{completed_pairs}/{total_pairs}: "
                    f"{address.address_id}::{address.partition}::"
                    f"{simulator.simulator_id}; "
                    f"base simulations={base_simulations:,}; nested rows={written:,}",
                    flush=True,
                )
    return accumulator, written, base_simulations, failures


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def run_self_test(b0: ModuleType, b: ModuleType) -> int:
    p3 = exact_one_sided_sign_flip_p(np.asarray([1.0, 1.0, 1.0]))
    p4 = exact_one_sided_sign_flip_p(np.asarray([1.0, 1.0, 1.0, 1.0]))
    if not math.isclose(p3, 0.125):
        raise AssertionError(f"Expected k=3 minimum p=0.125, observed {p3}")
    if not math.isclose(p4, 0.0625):
        raise AssertionError(f"Expected k=4 minimum p=0.0625, observed {p4}")

    target = b0.synthetic_component("target", seed=11)
    controls = [
        b0.synthetic_component("control_1", seed=12),
        b0.synthetic_component("control_2", seed=13),
    ]
    address = b0.PreparedAddress(
        address_id="self-test-address",
        record_id="target",
        support_id="self-test-support",
        failure_predicate=AUTHORIZED_PREDICATE,
        relation="self_test_relation",
        carrier="self_test_carrier",
        support_definition="support_flag == true",
        support_query_json="[]",
        entitlement_status="fl3_entitled",
        partition="discovery",
        target=target,
        controls=controls,
        cluster_ids=("o1", "o2", "o3", "o4"),
        selection_reason="self_test",
    )
    simulator = {
        spec.simulator_id: spec for spec in b0.simulator_specs()
    }["joint_wild_cluster_rademacher"]
    contract = b0.predicate_contracts()[AUTHORIZED_PREDICATE]
    frozen = FrozenGateParameters(
        alpha=0.10,
        minimum_effect=0.10,
        minimum_site_rows=1,
        minimum_complement_rows=1,
        minimum_shared_clusters=2,
        minimum_direction_consistency=0.75,
        minimum_control_adjusted_effect=0.05,
        minimum_positive_control_share=0.50,
        effect_direction_tolerance=1e-12,
    )
    seed = stable_seed("obs085c-self-test")
    expanded, sources, ids = expand_address_joint_templates(
        b0,
        address,
        8,
        np.random.default_rng(stable_seed(seed, "templates")),
    )
    simulated = simulate_prospective_components(
        b0,
        address,
        expanded,
        simulator,
        np.random.default_rng(stable_seed(seed, "simulator")),
    )
    injected = b0.apply_injection(
        expanded,
        simulated,
        2.0,
        0.0,
        np.random.default_rng(stable_seed(seed, "injection")),
    )
    row = evaluate_campaign_prefix(
        b0,
        b,
        address,
        simulator,
        contract,
        "self-test-scenario",
        "self-test-base",
        2.0,
        0.0,
        0,
        seed,
        sources[:4],
        ids[:4],
        simulated,
        injected,
        frozen,
        256,
    )
    if row["prospective_cluster_count"] != 4:
        raise AssertionError("Prospective prefix size mismatch")
    if row["identity_violation_count"] != 0:
        raise AssertionError("Injection changed frozen identity columns")
    if row["range_violation_count"] != 0:
        raise AssertionError("Prospective campaign produced range violations")

    repeated = evaluate_campaign_prefix(
        b0,
        b,
        address,
        simulator,
        contract,
        "self-test-scenario",
        "self-test-base",
        2.0,
        0.0,
        0,
        seed,
        sources[:4],
        ids[:4],
        simulated,
        injected,
        frozen,
        256,
    )
    if row["replicate_hash"] != repeated["replicate_hash"]:
        raise AssertionError("Deterministic prospective replicate hash failed")

    print(
        "OBS-085c self-test passed: exact sign-flip resolution, prospective "
        "joint cluster expansion, frozen simulator extension, nested prefix "
        "evaluation, identity/range preservation, and deterministic hashing"
    )
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()

    b0_path = repo_path(args.repo_root, args.obs085b0_script)
    b_path = repo_path(args.repo_root, args.obs085b_script)
    b0 = load_module_from_path(b0_path, "obs085b0_frozen_instrument_for_obs085c")
    b = load_module_from_path(b_path, "obs085b_frozen_instrument_for_obs085c")

    if args.self_test:
        return run_self_test(b0, b)

    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2")
    if args.bootstrap_draws < 128:
        raise ValueError("--bootstrap-draws must be at least 128")
    if args.max_controls < 1:
        raise ValueError("--max-controls must be positive")
    if args.replicate_chunk_size < 1:
        raise ValueError("--replicate-chunk-size must be positive")

    if args.smoke:
        args.replicates = 4
        args.bootstrap_draws = 128
        args.cluster_grid = "3,4,6"
        args.address_limit = 1

    cluster_grid = parse_integer_grid(args.cluster_grid, name="cluster grid")
    if max(cluster_grid) > 12:
        raise ValueError(
            "Canonical exact sign-flip scope is capped at k=12 (2^12=4096); "
            "use a separately declared future extension for larger k"
        )

    paths = input_paths(args)
    validate_required_inputs(paths)
    (
        obs085b_manifest,
        authorized,
        addresses,
        gate_contract,
        base_scenarios,
        frozen,
        lineage,
    ) = validate_obs085b_lineage(args, paths)

    if args.address_limit is not None:
        if args.address_limit < 1:
            raise ValueError("--address-limit must be positive")
        addresses = addresses.head(args.address_limit).copy()

    control_availability = read_csv(paths["obs085a_control_availability"])
    discovery_observations = read_csv(
        paths["obs084b_discovery_observation_losses"],
        dtype={"record_id": "string"},
    )
    confirmation_observations = read_csv(
        paths["obs084c_confirmation_observation_losses"],
        dtype={"record_id": "string"},
    )
    contracts = b0.predicate_contracts()
    if AUTHORIZED_PREDICATE not in contracts:
        raise RuntimeError("Frozen OBS-085b0 script lacks missingness predicate contract")
    contract = contracts[AUTHORIZED_PREDICATE]

    preparation_failures: list[Any] = []
    prepared, _ = b0.prepare_selected_addresses(
        addresses,
        control_availability,
        discovery_observations,
        confirmation_observations,
        contracts,
        args.max_controls,
        preparation_failures,
    )
    failures: list[StudyFailure] = []
    for failure in preparation_failures:
        values = asdict(failure) if hasattr(failure, "__dataclass_fields__") else dict(failure)
        failures.append(
            StudyFailure(
                stage=str(values.get("stage", "address_preparation")),
                scope_id=str(values.get("scope_id", "")),
                reason=str(values.get("reason", "address_unscorable")),
                detail=str(values.get("detail", "")),
                severity=str(values.get("severity", "warning")),
            )
        )

    expected_prepared = len(addresses) * len(AUTHORIZED_PARTITIONS)
    if len(prepared) != expected_prepared:
        raise RuntimeError(
            f"Expected {expected_prepared} address-partition preparations; observed {len(prepared)}"
        )

    simulator_map = {spec.simulator_id: spec for spec in b0.simulator_specs()}
    missing = [name for name in AUTHORIZED_SIMULATORS if name not in simulator_map]
    if missing:
        raise RuntimeError(f"Frozen OBS-085b0 script lacks simulators: {missing}")
    simulators = [simulator_map[name] for name in AUTHORIZED_SIMULATORS]

    design = design_manifest(cluster_grid, frozen)
    scenarios = build_scenario_manifest(
        base_scenarios,
        cluster_grid,
        args.replicates,
        args.master_seed,
        args.smoke,
    )
    scenario_lookup = {
        (str(row.base_scenario_id), int(row.prospective_cluster_count)): str(row.scenario_id)
        for row in scenarios.itertuples(index=False)
    }

    expected_base_simulations = len(prepared) * len(simulators) * len(base_scenarios) * args.replicates
    expected_rows = expected_base_simulations * len(cluster_grid)

    print("OBS-085c validation complete")
    print(f"Frozen OBS-085b manifest: {lineage['obs085b_manifest_id']}")
    print(f"Prepared address-partition cells: {len(prepared)}")
    print(f"Qualified simulators: {len(simulators)}")
    print(f"Frozen base scenarios: {len(base_scenarios)}")
    print(f"Prospective cluster counts: {cluster_grid}")
    print(f"Expected base simulations: {expected_base_simulations:,}")
    print(f"Expected nested replicate rows: {expected_rows:,}")

    if args.validate_only:
        return 0

    output_dir = repo_path(args.repo_root, args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    outputs = output_paths(output_dir)

    input_manifest = build_input_manifest(paths, args.repo_root)
    input_manifest.to_csv(outputs["obs085c_input_manifest.csv"], index=False)
    authorized.to_csv(outputs["obs085c_authorized_cells.csv"], index=False)
    addresses.to_csv(outputs["obs085c_address_manifest.csv"], index=False)
    gate_contract.to_csv(outputs["obs085c_gate_contract.csv"], index=False)
    design.to_csv(outputs["obs085c_design_manifest.csv"], index=False)
    scenarios.to_csv(outputs["obs085c_scenario_manifest.csv"], index=False)

    accumulator, written_rows, base_simulations, simulation_failures = run_simulation(
        b0,
        b,
        prepared,
        simulators,
        contract,
        base_scenarios,
        scenario_lookup,
        cluster_grid,
        frozen,
        args,
        outputs["replicate_directory"] if args.write_replicates else None,
    )
    failures.extend(simulation_failures)

    summary = accumulator.finalize()
    gate_failures = gate_failure_summary(summary)
    failure_combinations = accumulator.failure_combination_frame()
    ordered_gate = accumulator.ordered_gate_frame()
    curves = campaign_curve_summary(summary)
    thresholds = minimum_required_clusters(summary)
    attainability = attainability_map(design, summary)
    envelope = simulator_envelope(summary)
    entitlement = entitlement_overlay(summary)
    template_audit = source_template_audit(summary)
    uncertainty_audit = accumulator.uncertainty_status_frame()
    failure_frame = pd.DataFrame([asdict(failure) for failure in failures])
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=["stage", "scope_id", "reason", "detail", "severity"])

    summary.to_csv(outputs["obs085c_gate_passage_summary.csv"], index=False)
    gate_failures.to_csv(outputs["obs085c_gate_failure_summary.csv"], index=False)
    failure_combinations.to_csv(outputs["obs085c_failure_combinations.csv"], index=False)
    ordered_gate.to_csv(outputs["obs085c_ordered_gate_passage.csv"], index=False)
    curves.to_csv(outputs["obs085c_campaign_curve_summary.csv"], index=False)
    attainability.to_csv(outputs["obs085c_attainability_map.csv"], index=False)
    thresholds.to_csv(outputs["obs085c_minimum_required_clusters.csv"], index=False)
    envelope.to_csv(outputs["obs085c_simulator_envelope.csv"], index=False)
    entitlement.to_csv(outputs["obs085c_entitlement_overlay.csv"], index=False)
    template_audit.to_csv(outputs["obs085c_source_template_audit.csv"], index=False)
    uncertainty_audit.to_csv(outputs["obs085c_uncertainty_engine_audit.csv"], index=False)
    failure_frame.to_csv(outputs["obs085c_failures.csv"], index=False)

    canonical_design = tuple(cluster_grid) == CANONICAL_CLUSTER_GRID
    diagnostic_limit = args.address_limit is not None and not args.smoke
    complete = (
        written_rows == expected_rows
        and base_simulations == expected_base_simulations
        and failure_frame.empty
        and not args.smoke
        and not diagnostic_limit
        and canonical_design
        and args.replicates == DEFAULT_REPLICATES
    )
    if args.smoke:
        state = "engineering_smoke_only"
    elif complete:
        state = "campaign_attainability_simulation_completed"
    else:
        state = "diagnostic_campaign_attainability_simulation_completed"

    write_report(
        outputs["obs085c_report.md"],
        state,
        lineage,
        design,
        scenarios,
        curves,
        attainability,
        thresholds,
        uncertainty_audit,
        failure_frame,
        args,
    )
    manifest = build_manifest(
        args.repo_root,
        outputs,
        state,
        lineage,
        authorized,
        addresses,
        design,
        scenarios,
        summary,
        frozen,
        args,
        expected_rows,
        written_rows,
        base_simulations,
    )
    outputs["obs085c_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("OBS-085c execution complete")
    print(f"State: {state}")
    print(f"Manifest: {manifest['obs085c_manifest_id']}")
    print(f"Base simulations: {base_simulations:,} / {expected_base_simulations:,}")
    print(f"Nested rows: {written_rows:,} / {expected_rows:,}")
    print(f"Summary rows: {len(summary):,}")
    print(f"Failures: {len(failure_frame):,}")
    return 0 if len(failure_frame) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
