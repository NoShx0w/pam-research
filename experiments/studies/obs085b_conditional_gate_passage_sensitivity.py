#!/usr/bin/env python3
"""
obs085b_conditional_gate_passage_sensitivity.py

OBS-085b — Conditional Gate-Passage Sensitivity
================================================

Purpose
-------
Estimate conditional frozen-gate passage probabilities only for simulator cells
qualified by OBS-085b0.  This study is instrument metrology.  It does not
estimate classical statistical power, establish a minimum detectable effect,
or assert that a qualified simulator is the true failure-generating mechanism.

Frozen authorization
--------------------
The canonical OBS-085b0 result authorizes exactly four cells:

* measurement_missingness_concentration × discovery ×
  joint_gaussian_regularized_cluster
* measurement_missingness_concentration × confirmation ×
  joint_gaussian_regularized_cluster
* measurement_missingness_concentration × discovery ×
  joint_wild_cluster_rademacher
* measurement_missingness_concentration × confirmation ×
  joint_wild_cluster_rademacher

OBS-085b reuses the frozen OBS-085b0 missingness qualification-address panel.
It performs no new address search, ranking, threshold fitting, simulator
qualification, support construction, or predicate construction.

Primary estimand
----------------
For each frozen address, partition, qualified simulator, injected effect level,
and control-response condition:

    P(all frozen conditional evidence gates pass |
      fixed address already selected,
      qualified simulator contract,
      declared injection contract)

Discovery and confirmation are reported separately.  Simulator families are
reported separately.  A derived simulator envelope is a model-sensitivity
range, not a confidence interval and not a pooled probability.

Multiplicity scope
------------------
The authorized missingness addresses were not members of the sealed OBS-084b
M13 confirmation family.  Therefore the primary estimand is fixed-address M1
conditional gate passage.  The M13 Benjamini-Hochberg burden is not identified
for this predicate without inventing a new joint candidate family.  OBS-085b
fails closed by recording this as a scope limit instead of manufacturing an
M13-adjusted probability.

Default canonical design
------------------------
* frozen missingness qualification addresses: 6
* partitions: discovery, confirmation
* qualified simulators: 2
* effect grid: 0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00
* control-response grid for positive effects: 0, 0.25, 0.50, 1.00
* outer replicates per cell: 1000
* expected replicate rows: 600,000
* object is the dependence unit
* exact object-cluster bootstrap when n_clusters ** n_clusters <= 4096
* exact cluster sign-flip test when 2 ** n_clusters <= 4096

Run
---
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085b_conditional_gate_passage_sensitivity.py \\
  --overwrite

Validation only
---------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085b_conditional_gate_passage_sensitivity.py \\
  --validate-only

Engineering smoke run
---------------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085b_conditional_gate_passage_sensitivity.py \\
  --smoke --overwrite

Interpretation ceiling
----------------------
Monte Carlo intervals quantify simulation-sampling error only.  They do not
quantify simulator-model uncertainty.  Between-simulator spread is a model
sensitivity diagnostic, not a statistical confidence interval.
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
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence, TextIO

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "obs085b_conditional_gate_passage_sensitivity_v1"
DEFAULT_MASTER_SEED = 85100
DEFAULT_EXPECTED_OBS085B0_MANIFEST_ID = (
    "3015094cef1ee6a3f2b098662b75668109491634be827cd62cffc7b598fc66e6"
)
EXPECTED_AUTHORIZED_ADDRESS_COUNT = 6
AUTHORIZED_PREDICATE = "measurement_missingness_concentration"
AUTHORIZED_SIMULATORS = (
    "joint_gaussian_regularized_cluster",
    "joint_wild_cluster_rademacher",
)
AUTHORIZED_PARTITIONS = ("discovery", "confirmation")
QUALIFIED_STATUSES = {"qualified", "qualified_with_scope_limit"}

DEFAULT_OBS085B0_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085b0_simulator_qualification"
)
DEFAULT_OBS085A_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085a_structural_feasibility"
)
DEFAULT_OBS084_DISCOVERY_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/discovery"
)
DEFAULT_OBS084_CONFIRMATION_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/confirmation"
)
DEFAULT_OBS085B0_SCRIPT = Path(
    "experiments/studies/obs085b0_simulator_qualification.py"
)
DEFAULT_PROTOCOL = Path(
    "docs/05_project/"
    "085_failure_support_detection_power_and_confirmation_feasibility_protocol.md"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085b_conditional_gate_passage_sensitivity"
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
    "replicate",
    "seed",
    "target_contrast",
    "target_contrast_before_injection",
    "target_response_from_simulated_null",
    "median_control_contrast",
    "control_adjusted_contrast",
    "positive_control_adjusted_share",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_positive_share",
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


@dataclass(frozen=True)
class StudyFailure:
    stage: str
    scope_id: str
    reason: str
    detail: str = ""
    severity: str = "warning"


@dataclass(frozen=True)
class GateContract:
    gate_name: str
    gate_order: int
    estimand_field: str
    pass_rule: str
    threshold: str
    provenance: str
    required_for_overall_pass: bool = True


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--obs085b0-dir", type=Path, default=DEFAULT_OBS085B0_DIR)
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
    parser.add_argument("--obs085b0-script", type=Path, default=DEFAULT_OBS085B0_SCRIPT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-obs085b0-manifest-id",
        default=DEFAULT_EXPECTED_OBS085B0_MANIFEST_ID,
        help=(
            "Exact frozen OBS-085b0 manifest ID. Use 'auto' only for a separately "
            "documented, internally valid qualification bundle."
        ),
    )
    parser.add_argument(
        "--expected-authorized-addresses",
        type=int,
        default=EXPECTED_AUTHORIZED_ADDRESS_COUNT,
    )
    parser.add_argument("--max-controls", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument(
        "--effect-grid",
        default="0.00,0.25,0.50,0.75,1.00,1.50,2.00",
    )
    parser.add_argument(
        "--control-response-grid",
        default="0.00,0.25,0.50,1.00",
    )
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--minimum-effect", type=float, default=0.10)
    parser.add_argument("--minimum-site-rows", type=int, default=8)
    parser.add_argument("--minimum-complement-rows", type=int, default=8)
    parser.add_argument("--minimum-shared-clusters", type=int, default=2)
    parser.add_argument("--minimum-direction-consistency", type=float, default=0.75)
    parser.add_argument("--minimum-control-adjusted-effect", type=float, default=0.05)
    parser.add_argument("--minimum-positive-control-share", type=float, default=0.50)
    parser.add_argument(
        "--effect-direction-tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--write-replicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write complete replicate-level gate vectors as gzip-compressed CSV.",
    )
    parser.add_argument("--replicate-chunk-size", type=int, default=5000)
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help=(
            "Deterministic prefix of the frozen address panel. Any non-smoke limit "
            "marks the run diagnostic-only and disables canonical completion state."
        ),
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


def parse_number_grid(text: str, *, name: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {text!r}") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} must contain finite nonnegative values")
    unique = sorted(set(values))
    return unique


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


def run_git(repo_root: Path, args: Sequence[str], *, check: bool = True) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        if check:
            raise
        return "unknown"


def load_module_from_path(path: Path, name: str) -> ModuleType:
    require_file(path, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
        / denom
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


# -----------------------------------------------------------------------------
# Frozen lineage and authorization validation
# -----------------------------------------------------------------------------


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.repo_root.resolve()
    b0_dir = repo_path(root, args.obs085b0_dir)
    a_dir = repo_path(root, args.obs085a_dir)
    discovery = repo_path(root, args.obs084_discovery_dir)
    confirmation = repo_path(root, args.obs084_confirmation_dir)
    return {
        "obs085b0_manifest": b0_dir / "obs085b0_manifest.json",
        "obs085b0_gate_matrix": b0_dir / "obs085b0_qualification_gate_matrix.csv",
        "obs085b0_address_manifest": b0_dir / "obs085b0_qualification_address_manifest.csv",
        "obs085b0_simulator_specs": b0_dir / "obs085b0_simulator_specs.csv",
        "obs085b0_injection_contracts": b0_dir / "obs085b0_injection_contracts.csv",
        "obs085b0_input_manifest": b0_dir / "obs085b0_input_manifest.csv",
        "obs085b0_script": repo_path(root, args.obs085b0_script),
        "obs085a_control_availability": a_dir / "obs085a_control_availability.csv",
        "obs084b_discovery_observation_losses": discovery
        / "obs084b_discovery_observation_losses.csv",
        "obs084c_confirmation_observation_losses": confirmation
        / "obs084c_confirmation_observation_losses.csv",
        "obs085_protocol": repo_path(root, args.protocol),
        "obs085b_script": Path(__file__).resolve(),
    }


def validate_required_inputs(paths: Mapping[str, Path]) -> None:
    for role, path in paths.items():
        require_file(path, role)


def validate_b0_internal_manifest(manifest: Mapping[str, Any]) -> None:
    declared = str(manifest.get("obs085b0_manifest_id", ""))
    core = dict(manifest)
    core.pop("obs085b0_manifest_id", None)
    computed = sha256_bytes(canonical_json(core).encode("utf-8"))
    if not declared or declared != computed:
        raise RuntimeError(
            "OBS-085b0 manifest internal identity mismatch: "
            f"declared={declared}; computed={computed}"
        )


def validate_b0_output_hashes(
    manifest: Mapping[str, Any],
    repo_root: Path,
) -> int:
    records = manifest.get("output_artifacts", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError("OBS-085b0 manifest contains no output artifact hashes")
    checked = 0
    for record in records:
        if not isinstance(record, Mapping):
            continue
        label = str(record.get("artifact_path", ""))
        expected = str(record.get("sha256", ""))
        if not label or not expected:
            raise RuntimeError("Malformed OBS-085b0 output hash record")
        path = repo_path(repo_root, Path(label))
        require_file(path, f"OBS-085b0 output {label}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"OBS-085b0 output hash mismatch for {label}: "
                f"expected={expected}; actual={actual}"
            )
        checked += 1
    return checked


def validate_b0_script_hash(
    input_manifest: pd.DataFrame,
    b0_script: Path,
) -> tuple[str, str]:
    require_columns(
        input_manifest,
        ["artifact_role", "sha256"],
        "OBS-085b0 input manifest",
    )
    rows = input_manifest[
        input_manifest["artifact_role"].astype(str).eq("obs085b0_script")
    ]
    if len(rows) != 1:
        raise RuntimeError(
            "OBS-085b0 input manifest must contain exactly one obs085b0_script row"
        )
    expected = str(rows.iloc[0]["sha256"])
    actual = sha256_file(b0_script)
    if expected != actual:
        raise RuntimeError(
            "Current OBS-085b0 script differs from the script used to create the "
            f"qualification bundle: expected={expected}; actual={actual}"
        )
    return expected, actual


def expected_authorized_cell_set() -> set[tuple[str, str, str]]:
    return {
        (AUTHORIZED_PREDICATE, partition, simulator)
        for partition in AUTHORIZED_PARTITIONS
        for simulator in AUTHORIZED_SIMULATORS
    }


def validate_authorization(
    gates: pd.DataFrame,
    address_manifest: pd.DataFrame,
    expected_address_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        gates,
        [
            "failure_predicate",
            "partition",
            "simulator_id",
            "qualification_status",
            "scope_limits_json",
        ],
        "OBS-085b0 qualification gate matrix",
    )
    qualified = gates[
        gates["qualification_status"].astype(str).isin(QUALIFIED_STATUSES)
    ].copy()
    observed = {
        (
            str(row.failure_predicate),
            str(row.partition),
            str(row.simulator_id),
        )
        for row in qualified.itertuples(index=False)
    }
    expected = expected_authorized_cell_set()
    if observed != expected:
        raise RuntimeError(
            "OBS-085b0 authorization matrix differs from the frozen OBS-085b "
            f"scope. expected={sorted(expected)}; observed={sorted(observed)}"
        )

    require_columns(
        address_manifest,
        [
            "address_id",
            "record_id",
            "support_id",
            "failure_predicate",
            "support_query_json",
            "support_definition",
            "relation",
            "carrier",
        ],
        "OBS-085b0 qualification address manifest",
    )
    panel = address_manifest[
        address_manifest["failure_predicate"].astype(str).eq(AUTHORIZED_PREDICATE)
    ].copy()
    panel = panel.sort_values(
        [column for column in ["qualification_selection_rank", "address_id"] if column in panel.columns]
    ).drop_duplicates("address_id")
    if len(panel) != expected_address_count:
        raise RuntimeError(
            "Frozen OBS-085b0 missingness address panel count mismatch: "
            f"expected={expected_address_count}; observed={len(panel)}"
        )
    if "selection_reason" not in panel.columns:
        panel["selection_reason"] = "frozen_obs085b0_qualification_panel"
    if "entitlement_status" not in panel.columns:
        panel["entitlement_status"] = "unknown"
    return qualified.sort_values(
        ["failure_predicate", "partition", "simulator_id"]
    ), panel.reset_index(drop=True)


def validate_obs085b0_lineage(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = read_json(paths["obs085b0_manifest"])
    validate_b0_internal_manifest(manifest)
    declared_id = str(manifest.get("obs085b0_manifest_id", ""))
    expected_id = str(args.expected_obs085b0_manifest_id)
    if expected_id.lower() != "auto" and declared_id != expected_id:
        raise RuntimeError(
            "Unexpected OBS-085b0 manifest identity: "
            f"expected={expected_id}; observed={declared_id}"
        )
    if str(manifest.get("state", "")) != "simulator_qualification_completed":
        raise RuntimeError(
            "OBS-085b requires a completed non-smoke OBS-085b0 qualification bundle"
        )
    if not normalize_bool(manifest.get("qualification_decision_enabled", False)):
        raise RuntimeError("OBS-085b0 qualification decisions were not enabled")

    output_hashes_checked = validate_b0_output_hashes(manifest, args.repo_root)
    input_manifest = read_csv(paths["obs085b0_input_manifest"])
    expected_script_sha, actual_script_sha = validate_b0_script_hash(
        input_manifest,
        paths["obs085b0_script"],
    )
    gates = read_csv(paths["obs085b0_gate_matrix"])
    address_manifest = read_csv(
        paths["obs085b0_address_manifest"],
        dtype={"address_id": "string"},
    )
    qualified, panel = validate_authorization(
        gates,
        address_manifest,
        args.expected_authorized_addresses,
    )
    lineage = {
        "obs085b0_manifest_id": declared_id,
        "obs085b0_manifest_sha256": sha256_file(paths["obs085b0_manifest"]),
        "obs085b0_output_hashes_checked": output_hashes_checked,
        "obs085b0_script_sha256_expected": expected_script_sha,
        "obs085b0_script_sha256_actual": actual_script_sha,
        "obs085b0_script_version": str(manifest.get("script_version", "")),
        "obs085b0_state": str(manifest.get("state", "")),
        "obs085b0_qualified_cell_count": len(qualified),
        "obs085b0_authorized_address_count": len(panel),
        "obs085b0_scope_limits": sorted(
            {
                item
                for raw in qualified["scope_limits_json"].astype(str)
                for item in parse_json_list(raw)
            }
        ),
        "current_repo_head": run_git(args.repo_root, ["rev-parse", "HEAD"], check=False),
    }
    return manifest, qualified, panel, lineage


def parse_json_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


# -----------------------------------------------------------------------------
# Frozen design and gate contract
# -----------------------------------------------------------------------------


def scenario_cells(
    effect_grid: Sequence[float],
    control_grid: Sequence[float],
) -> list[tuple[float, float]]:
    cells: list[tuple[float, float]] = []
    for delta in effect_grid:
        if math.isclose(delta, 0.0, abs_tol=1e-15):
            cells.append((0.0, 0.0))
        else:
            cells.extend((float(delta), float(value)) for value in control_grid)
    return cells


def build_scenario_manifest(
    effect_grid: Sequence[float],
    control_grid: Sequence[float],
    replicates: int,
    master_seed: int,
    smoke: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for delta, control_response in scenario_cells(effect_grid, control_grid):
        rows.append(
            {
                "scenario_id": stable_id(
                    "obs085b",
                    delta,
                    control_response,
                    replicates,
                    master_seed,
                    prefix="BS-",
                ),
                "delta": delta,
                "control_response_lambda": control_response,
                "replicates": replicates,
                "master_seed": master_seed,
                "engineering_smoke": smoke,
                "conditional_estimand": True,
                "scenario_semantics": (
                    "null gate-passage calibration"
                    if delta == 0
                    else "conditional frozen-gate passage under declared missingness injection"
                ),
            }
        )
    return pd.DataFrame(rows)


def gate_contracts(args: argparse.Namespace) -> list[GateContract]:
    return [
        GateContract(
            "support_available_pass",
            1,
            "target support and metric availability",
            "metric present and focal support has at least one row",
            ">= 1 support row",
            "frozen address/support identity",
        ),
        GateContract(
            "complement_admissible_pass",
            2,
            "focal/complement rows and shared object clusters",
            "site rows, complement rows, and shared clusters meet frozen minima",
            (
                f"site >= {args.minimum_site_rows}; complement >= "
                f"{args.minimum_complement_rows}; shared objects >= "
                f"{args.minimum_shared_clusters}"
            ),
            "OBS-085a structural evidence contract",
        ),
        GateContract(
            "effect_direction_reproduced_pass",
            3,
            "target_response_from_simulated_null",
            "positive injected target response for delta > 0",
            f"> {args.effect_direction_tolerance:g}",
            "OBS-085b0 direct injection-response qualification estimand",
        ),
        GateContract(
            "target_contrast_positive_pass",
            4,
            "target_contrast",
            "focal support missingness contrast is positive",
            "> 0",
            "frozen positive failure direction",
        ),
        GateContract(
            "minimum_effect_pass",
            5,
            "target_contrast",
            "focal support missingness contrast reaches declared minimum effect",
            f">= {args.minimum_effect:g}",
            "predeclared OBS-085b missingness contrast gate",
        ),
        GateContract(
            "cluster_uncertainty_pass",
            6,
            "object-cluster bootstrap and leave-one-object-out diagnostics",
            "bootstrap lower bound positive, direction consistency sufficient, and >=2 LOO estimates",
            (
                f"CI low > 0; direction consistency >= "
                f"{args.minimum_direction_consistency:g}; LOO successful >= 2"
            ),
            "OBS-084c cluster-sensitivity form; object retained as dependence unit",
        ),
        GateContract(
            "raw_statistical_threshold_pass",
            7,
            "one-sided object-cluster sign-flip p-value",
            "raw p-value reaches declared alpha",
            f"p <= {args.alpha:g}",
            "fixed-address conditional randomization diagnostic",
        ),
        GateContract(
            "multiplicity_adjusted_threshold_pass",
            8,
            "M1 adjusted p-value",
            "fixed-address M1 adjusted p-value reaches alpha",
            f"q_M1 <= {args.alpha:g}",
            "M1 only; M13 not identified for the missingness panel",
        ),
        GateContract(
            "control_adjusted_contrast_pass",
            9,
            "target contrast minus median frozen-control contrast",
            "median control-adjusted contrast reaches declared minimum",
            f">= {args.minimum_control_adjusted_effect:g}",
            "OBS-084c control-adjusted effect form",
        ),
        GateContract(
            "control_specificity_pass",
            10,
            "share of target-minus-control contrasts that are positive",
            "positive target-minus-control share reaches declared minimum",
            f">= {args.minimum_positive_control_share:g}",
            "OBS-084c positive control-adjusted share form",
        ),
    ]


def gate_contract_frame(contracts: Sequence[GateContract]) -> pd.DataFrame:
    return pd.DataFrame([asdict(contract) for contract in contracts])


# -----------------------------------------------------------------------------
# Exact object-cluster uncertainty helpers
# -----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def exact_bootstrap_indices(cluster_count: int) -> np.ndarray:
    if cluster_count < 1:
        return np.empty((0, 0), dtype=int)
    total = cluster_count ** cluster_count
    if total > 4096:
        raise ValueError(
            f"Exact cluster bootstrap too large for {cluster_count} clusters ({total})"
        )
    return np.asarray(
        list(itertools.product(range(cluster_count), repeat=cluster_count)),
        dtype=int,
    )


@lru_cache(maxsize=None)
def exact_sign_matrix(cluster_count: int) -> np.ndarray:
    if cluster_count < 1:
        return np.empty((0, 0), dtype=float)
    total = 2 ** cluster_count
    if total > 4096:
        raise ValueError(
            f"Exact cluster sign-flip family too large for {cluster_count} clusters"
        )
    return np.asarray(
        list(itertools.product([-1.0, 1.0], repeat=cluster_count)),
        dtype=float,
    )


def cluster_contrast_vector(
    b0: ModuleType,
    component: Any,
    metric: str,
    cluster_ids: Sequence[str],
) -> np.ndarray:
    contributions = b0.object_contributions(
        component.frame,
        component.support_mask,
        metric,
    )
    if contributions.empty:
        return np.asarray([], dtype=float)
    mapping = {
        str(row.cluster_id): float(row.object_contrast)
        for row in contributions.itertuples(index=False)
        if math.isfinite(float(row.object_contrast))
    }
    values = np.asarray(
        [mapping.get(str(cluster_id), float("nan")) for cluster_id in cluster_ids],
        dtype=float,
    )
    return values[np.isfinite(values)]


def exact_cluster_uncertainty(
    vector: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    values = np.asarray(vector, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        return {
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "bootstrap_positive_share": float("nan"),
            "direction_consistency": float("nan"),
            "loo_successful_count": 0,
            "independent_cluster_count": n,
            "raw_permutation_p": 1.0,
            "permutation_count": 0,
            "permutation_method": "unavailable_fewer_than_two_clusters",
        }

    bootstrap_index = exact_bootstrap_indices(n)
    bootstrap_values = values[bootstrap_index].mean(axis=1)
    ci_low = float(np.quantile(bootstrap_values, alpha / 2.0))
    ci_high = float(np.quantile(bootstrap_values, 1.0 - alpha / 2.0))
    bootstrap_positive_share = float(np.mean(bootstrap_values > 0))

    total = float(values.sum())
    loo = (total - values) / (n - 1)
    finite_loo = loo[np.isfinite(loo)]
    direction_consistency = (
        float(np.mean(finite_loo > 0)) if finite_loo.size else float("nan")
    )

    signs = exact_sign_matrix(n)
    permuted = (signs * values[None, :]).mean(axis=1)
    observed = float(values.mean())
    p_value = float(np.mean(permuted >= observed - 1e-15))
    return {
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_positive_share": bootstrap_positive_share,
        "direction_consistency": direction_consistency,
        "loo_successful_count": int(finite_loo.size),
        "independent_cluster_count": n,
        "raw_permutation_p": p_value,
        "permutation_count": int(len(permuted)),
        "permutation_method": "exact_object_cluster_sign_flip",
    }


# -----------------------------------------------------------------------------
# Replicate simulation and gate evaluation
# -----------------------------------------------------------------------------


def support_cluster_counts(component: Any) -> tuple[int, int]:
    clusters = component.frame["cluster_id"].astype(str)
    support = component.support_mask.astype(bool)
    return int(clusters[support].nunique()), int(clusters[~support].nunique())


def simulate_gate_replicate(
    b0: ModuleType,
    address: Any,
    simulator: Any,
    contract: Any,
    scenario_id: str,
    delta: float,
    control_response: float,
    replicate: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    simulated = b0.simulate_components(address, simulator, rng)
    target_before = simulated[0]
    injected = b0.apply_injection(
        address,
        simulated,
        delta,
        control_response,
        rng,
    )
    target = injected[0]
    controls = injected[1:]
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
        b0.site_relative_contrast(
            component.frame,
            component.support_mask,
            metric,
        )
        for component in controls
    ]
    finite_controls = [value for value in control_contrasts if math.isfinite(value)]
    median_control = (
        float(np.median(finite_controls)) if finite_controls else float("nan")
    )
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
    shared_clusters = len(address.cluster_ids)

    vector = cluster_contrast_vector(
        b0,
        target,
        metric,
        address.cluster_ids,
    )
    uncertainty = exact_cluster_uncertainty(vector, args.alpha)

    support_available = bool(metric in target.frame.columns and site_rows > 0)
    complement_admissible = bool(
        site_rows >= args.minimum_site_rows
        and complement_rows >= args.minimum_complement_rows
        and shared_clusters >= args.minimum_shared_clusters
        and site_clusters >= args.minimum_shared_clusters
        and complement_clusters >= args.minimum_shared_clusters
    )
    effect_direction = bool(
        delta > 0
        and math.isfinite(target_response)
        and target_response > args.effect_direction_tolerance
    )
    target_positive = bool(math.isfinite(target_contrast) and target_contrast > 0)
    minimum_effect = bool(
        math.isfinite(target_contrast)
        and target_contrast >= args.minimum_effect
    )
    cluster_uncertainty = bool(
        math.isfinite(float(uncertainty["bootstrap_ci_low"]))
        and float(uncertainty["bootstrap_ci_low"]) > 0
        and math.isfinite(float(uncertainty["direction_consistency"]))
        and float(uncertainty["direction_consistency"])
        >= args.minimum_direction_consistency
        and int(uncertainty["loo_successful_count"]) >= 2
    )
    raw_p = float(uncertainty["raw_permutation_p"])
    raw_threshold = bool(math.isfinite(raw_p) and raw_p <= args.alpha)

    # M1 is the only identified multiplicity family for this fixed-address
    # conditional estimand.  No pseudo-M13 family is manufactured.
    m1_adjusted_p = raw_p
    multiplicity_threshold = bool(
        math.isfinite(m1_adjusted_p) and m1_adjusted_p <= args.alpha
    )
    control_adjusted_pass = bool(
        math.isfinite(control_adjusted)
        and control_adjusted >= args.minimum_control_adjusted_effect
    )
    control_specificity_pass = bool(
        math.isfinite(positive_control_share)
        and positive_control_share >= args.minimum_positive_control_share
    )

    gates = {
        "support_available_pass": support_available,
        "complement_admissible_pass": complement_admissible,
        "effect_direction_reproduced_pass": effect_direction,
        "target_contrast_positive_pass": target_positive,
        "minimum_effect_pass": minimum_effect,
        "cluster_uncertainty_pass": cluster_uncertainty,
        "raw_statistical_threshold_pass": raw_threshold,
        "multiplicity_adjusted_threshold_pass": multiplicity_threshold,
        "control_adjusted_contrast_pass": control_adjusted_pass,
        "control_specificity_pass": control_specificity_pass,
    }
    overall = all(gates.values())
    failed = [name for name in GATE_ORDER if not gates[name]]

    range_violations = sum(
        int(b0.range_violation_count(component.frame)) for component in injected
    )
    identity_violations = sum(
        int(b0.identity_hash(before.frame) != b0.identity_hash(after.frame))
        for before, after in zip(simulated, injected)
    )

    payload = {
        "scenario_id": scenario_id,
        "address_id": str(address.address_id),
        "record_id": str(address.record_id),
        "support_id": str(address.support_id),
        "relation": str(address.relation),
        "carrier": str(address.carrier),
        "entitlement_status": str(address.entitlement_status),
        "partition": str(address.partition),
        "simulator_id": str(simulator.simulator_id),
        "failure_predicate": str(address.failure_predicate),
        "delta": float(delta),
        "control_response_lambda": float(control_response),
        "replicate": int(replicate),
        "seed": int(seed),
        "target_contrast": target_contrast,
        "target_contrast_before_injection": target_before_contrast,
        "target_response_from_simulated_null": target_response,
        "median_control_contrast": median_control,
        "control_adjusted_contrast": control_adjusted,
        "positive_control_adjusted_share": positive_control_share,
        "bootstrap_ci_low": uncertainty["bootstrap_ci_low"],
        "bootstrap_ci_high": uncertainty["bootstrap_ci_high"],
        "bootstrap_positive_share": uncertainty["bootstrap_positive_share"],
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
        prefix="BR-",
    )
    return payload


# -----------------------------------------------------------------------------
# Streaming replicate retention and online summaries
# -----------------------------------------------------------------------------


class ReplicateWriter:
    def __init__(self, path: Path | None, chunk_size: int) -> None:
        self.path = path
        self.chunk_size = chunk_size
        self.buffer: list[dict[str, Any]] = []
        self.handle: TextIO | None = None
        self.writer: csv.DictWriter[str] | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = gzip.open(path, "wt", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(
                self.handle,
                fieldnames=REPLICATE_COLUMNS,
                extrasaction="ignore",
            )
            self.writer.writeheader()

    def add(self, row: dict[str, Any]) -> None:
        if self.writer is None:
            return
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if self.writer is None or not self.buffer:
            return
        self.writer.writerows(self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "ReplicateWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class OnlineSummary:
    def __init__(self) -> None:
        self.cells: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.failure_combinations: defaultdict[tuple[Any, ...], int] = defaultdict(int)
        self.ordered_pass: defaultdict[tuple[Any, ...], int] = defaultdict(int)

    @staticmethod
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["scenario_id"],
            row["address_id"],
            row["record_id"],
            row["support_id"],
            row["relation"],
            row["carrier"],
            row["entitlement_status"],
            row["partition"],
            row["simulator_id"],
            row["failure_predicate"],
            float(row["delta"]),
            float(row["control_response_lambda"]),
        )

    def add(self, row: Mapping[str, Any]) -> None:
        key = self.key(row)
        if key not in self.cells:
            self.cells[key] = {
                "replicates": 0,
                "overall_passes": 0,
                "finite_target": 0,
                "target_contrast_sum": 0.0,
                "target_contrast_sq_sum": 0.0,
                "target_response_sum": 0.0,
                "control_adjusted_sum": 0.0,
                "raw_p_sum": 0.0,
                **{f"{gate}_count": 0 for gate in GATE_ORDER},
            }
        cell = self.cells[key]
        cell["replicates"] += 1
        overall = normalize_bool(row["overall_gate_pass"])
        cell["overall_passes"] += int(overall)
        for gate in GATE_ORDER:
            cell[f"{gate}_count"] += int(normalize_bool(row[gate]))

        target = float(row["target_contrast"])
        if math.isfinite(target):
            cell["finite_target"] += 1
            cell["target_contrast_sum"] += target
            cell["target_contrast_sq_sum"] += target * target
        response = float(row["target_response_from_simulated_null"])
        if math.isfinite(response):
            cell["target_response_sum"] += response
        adjusted = float(row["control_adjusted_contrast"])
        if math.isfinite(adjusted):
            cell["control_adjusted_sum"] += adjusted
        p_value = float(row["raw_permutation_p"])
        if math.isfinite(p_value):
            cell["raw_p_sum"] += p_value

        failed = tuple(parse_json_list(row["failed_gates_json"]))
        combo_key = key + (canonical_json(failed),)
        self.failure_combinations[combo_key] += 1

        cumulative = True
        for index, gate in enumerate(GATE_ORDER, start=1):
            cumulative = cumulative and normalize_bool(row[gate])
            ordered_key = key + (index, gate)
            self.ordered_pass[ordered_key] += int(cumulative)

    def summary_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        metadata_columns = [
            "scenario_id",
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
        for key, cell in sorted(self.cells.items()):
            metadata = dict(zip(metadata_columns, key))
            n = int(cell["replicates"])
            successes = int(cell["overall_passes"])
            low, high = wilson_interval(successes, n)
            probability = successes / n if n else float("nan")
            finite = int(cell["finite_target"])
            target_mean = (
                cell["target_contrast_sum"] / finite if finite else float("nan")
            )
            if finite > 1:
                variance = max(
                    0.0,
                    (
                        cell["target_contrast_sq_sum"]
                        - finite * target_mean * target_mean
                    )
                    / (finite - 1),
                )
                target_sd = math.sqrt(variance)
            else:
                target_sd = float("nan")
            row = {
                **metadata,
                "replicates": n,
                "finite_target_replicates": finite,
                "overall_gate_passes": successes,
                "conditional_gate_passage_probability": probability,
                "monte_carlo_se": (
                    math.sqrt(probability * (1.0 - probability) / n)
                    if n and math.isfinite(probability)
                    else float("nan")
                ),
                "wilson_95_low": low,
                "wilson_95_high": high,
                "mean_target_contrast": target_mean,
                "sd_target_contrast": target_sd,
                "mean_target_response_from_simulated_null": (
                    cell["target_response_sum"] / n if n else float("nan")
                ),
                "mean_control_adjusted_contrast": (
                    cell["control_adjusted_sum"] / n if n else float("nan")
                ),
                "mean_raw_permutation_p": (
                    cell["raw_p_sum"] / n if n else float("nan")
                ),
            }
            for gate in GATE_ORDER:
                count = int(cell[f"{gate}_count"])
                row[f"{gate}_count"] = count
                row[f"{gate}_probability"] = count / n if n else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)

    def failure_combination_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        metadata_columns = [
            "scenario_id",
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
            "failed_gates_json",
        ]
        totals = {
            key: int(cell["replicates"])
            for key, cell in self.cells.items()
        }
        for combo_key, count in sorted(self.failure_combinations.items()):
            base_key = combo_key[:-1]
            total = totals[base_key]
            rows.append(
                {
                    **dict(zip(metadata_columns, combo_key)),
                    "replicate_count": count,
                    "replicate_share": count / total if total else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def ordered_pass_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        metadata_columns = [
            "scenario_id",
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
            "gate_order",
            "gate_name",
        ]
        totals = {
            key: int(cell["replicates"])
            for key, cell in self.cells.items()
        }
        for ordered_key, count in sorted(self.ordered_pass.items()):
            base_key = ordered_key[:-2]
            total = totals[base_key]
            rows.append(
                {
                    **dict(zip(metadata_columns, ordered_key)),
                    "cumulative_pass_count": count,
                    "cumulative_pass_probability": (
                        count / total if total else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Derived summaries
# -----------------------------------------------------------------------------


def gate_failure_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "failure_predicate",
        "partition",
        "simulator_id",
        "delta",
        "control_response_lambda",
    ]
    for key, group in summary.groupby(group_columns, dropna=False):
        metadata = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        total = int(group["replicates"].sum())
        for gate in GATE_ORDER:
            passed = int(group[f"{gate}_count"].sum())
            rows.append(
                {
                    **metadata,
                    "gate_name": gate,
                    "replicates": total,
                    "pass_count": passed,
                    "pass_probability": passed / total if total else float("nan"),
                    "failure_probability": 1.0 - passed / total if total else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def aggregate_address_curves(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    group_columns = [
        "failure_predicate",
        "partition",
        "simulator_id",
        "delta",
        "control_response_lambda",
    ]
    return (
        summary.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            addresses=("address_id", "nunique"),
            macro_mean_gate_passage_probability=(
                "conditional_gate_passage_probability",
                "mean",
            ),
            minimum_address_probability=(
                "conditional_gate_passage_probability",
                "min",
            ),
            maximum_address_probability=(
                "conditional_gate_passage_probability",
                "max",
            ),
            median_address_probability=(
                "conditional_gate_passage_probability",
                "median",
            ),
            mean_target_contrast=("mean_target_contrast", "mean"),
            mean_control_adjusted_contrast=(
                "mean_control_adjusted_contrast",
                "mean",
            ),
        )
        .sort_values(group_columns)
    )


def minimum_tested_effects(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "address_id",
        "record_id",
        "support_id",
        "relation",
        "carrier",
        "entitlement_status",
        "partition",
        "simulator_id",
        "control_response_lambda",
    ]
    positive = summary[summary["delta"] > 0].copy()
    for key, group in positive.groupby(group_columns, dropna=False):
        metadata = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        ordered = group.sort_values("delta")
        for target in (0.50, 0.80, 0.90):
            reached = ordered[
                ordered["conditional_gate_passage_probability"] >= target
            ]
            if reached.empty:
                effect = float("nan")
                probability = float("nan")
                status = "not_reached_on_tested_grid"
            else:
                first = reached.iloc[0]
                effect = float(first["delta"])
                probability = float(first["conditional_gate_passage_probability"])
                status = "reached_at_tested_effect"
            rows.append(
                {
                    **metadata,
                    "target_gate_passage_probability": target,
                    "smallest_tested_delta": effect,
                    "observed_probability_at_smallest_tested_delta": probability,
                    "threshold_status": status,
                    "interpolation_used": False,
                    "extrapolation_used": False,
                }
            )
    return pd.DataFrame(rows)


def simulator_envelope(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    index = [
        "address_id",
        "record_id",
        "support_id",
        "relation",
        "carrier",
        "entitlement_status",
        "partition",
        "failure_predicate",
        "delta",
        "control_response_lambda",
    ]
    pivot = summary.pivot_table(
        index=index,
        columns="simulator_id",
        values="conditional_gate_passage_probability",
        aggfunc="first",
    ).reset_index()
    for simulator in AUTHORIZED_SIMULATORS:
        if simulator not in pivot.columns:
            pivot[simulator] = np.nan
    values = pivot[list(AUTHORIZED_SIMULATORS)].to_numpy(float)
    pivot["simulator_envelope_low"] = np.nanmin(values, axis=1)
    pivot["simulator_envelope_high"] = np.nanmax(values, axis=1)
    pivot["between_simulator_spread"] = (
        pivot["simulator_envelope_high"] - pivot["simulator_envelope_low"]
    )
    pivot["envelope_semantics"] = (
        "qualified-simulator model-sensitivity range; not a confidence interval"
    )
    return pivot


def entitlement_overlay(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    group_columns = [
        "entitlement_status",
        "partition",
        "simulator_id",
        "delta",
        "control_response_lambda",
    ]
    result = (
        summary.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            addresses=("address_id", "nunique"),
            macro_mean_gate_passage_probability=(
                "conditional_gate_passage_probability",
                "mean",
            ),
            minimum_address_probability=(
                "conditional_gate_passage_probability",
                "min",
            ),
            maximum_address_probability=(
                "conditional_gate_passage_probability",
                "max",
            ),
        )
        .sort_values(group_columns)
    )
    result["claim_rule"] = (
        "gate passage cannot increase claim entitlement beyond the frozen OBS-085a class"
    )
    return result


def null_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    null = summary[np.isclose(summary["delta"], 0.0)].copy()
    if null.empty:
        return pd.DataFrame()
    group_columns = ["partition", "simulator_id"]
    return (
        null.groupby(group_columns, as_index=False)
        .agg(
            addresses=("address_id", "nunique"),
            macro_mean_null_gate_passage=(
                "conditional_gate_passage_probability",
                "mean",
            ),
            maximum_address_null_gate_passage=(
                "conditional_gate_passage_probability",
                "max",
            ),
            macro_mean_null_target_contrast=("mean_target_contrast", "mean"),
        )
        .sort_values(group_columns)
    )


# -----------------------------------------------------------------------------
# Output and report
# -----------------------------------------------------------------------------


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "obs085b_manifest.json": output_dir / "obs085b_manifest.json",
        "obs085b_input_manifest.csv": output_dir / "obs085b_input_manifest.csv",
        "obs085b_authorized_cells.csv": output_dir / "obs085b_authorized_cells.csv",
        "obs085b_address_manifest.csv": output_dir / "obs085b_address_manifest.csv",
        "obs085b_gate_contract.csv": output_dir / "obs085b_gate_contract.csv",
        "obs085b_scenario_manifest.csv": output_dir / "obs085b_scenario_manifest.csv",
        "obs085b_replicate_outcomes.csv.gz": output_dir
        / "obs085b_replicate_outcomes.csv.gz",
        "obs085b_gate_passage_summary.csv": output_dir
        / "obs085b_gate_passage_summary.csv",
        "obs085b_gate_failure_summary.csv": output_dir
        / "obs085b_gate_failure_summary.csv",
        "obs085b_failure_combinations.csv": output_dir
        / "obs085b_failure_combinations.csv",
        "obs085b_ordered_gate_passage.csv": output_dir
        / "obs085b_ordered_gate_passage.csv",
        "obs085b_address_curve_summary.csv": output_dir
        / "obs085b_address_curve_summary.csv",
        "obs085b_minimum_tested_effect.csv": output_dir
        / "obs085b_minimum_tested_effect.csv",
        "obs085b_simulator_envelope.csv": output_dir
        / "obs085b_simulator_envelope.csv",
        "obs085b_entitlement_overlay.csv": output_dir
        / "obs085b_entitlement_overlay.csv",
        "obs085b_null_calibration.csv": output_dir / "obs085b_null_calibration.csv",
        "obs085b_failures.csv": output_dir / "obs085b_failures.csv",
        "obs085b_report.md": output_dir / "obs085b_report.md",
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}; use --overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def build_input_manifest(
    paths: Mapping[str, Path],
    repo_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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


def write_report(
    path: Path,
    state: str,
    lineage: Mapping[str, Any],
    authorized: pd.DataFrame,
    addresses: pd.DataFrame,
    scenarios: pd.DataFrame,
    gate_contract: pd.DataFrame,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    thresholds: pd.DataFrame,
    null_summary: pd.DataFrame,
    failures: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    positive = aggregate[aggregate["delta"] > 0].copy() if not aggregate.empty else pd.DataFrame()
    endpoint = (
        positive.sort_values("delta")
        .groupby(["partition", "simulator_id", "control_response_lambda"], as_index=False)
        .tail(1)
        if not positive.empty
        else pd.DataFrame()
    )
    endpoint_columns = [
        column
        for column in [
            "partition",
            "simulator_id",
            "control_response_lambda",
            "delta",
            "addresses",
            "macro_mean_gate_passage_probability",
            "minimum_address_probability",
            "maximum_address_probability",
        ]
        if column in endpoint.columns
    ]
    threshold_counts = (
        thresholds.groupby(
            [
                "partition",
                "simulator_id",
                "target_gate_passage_probability",
                "threshold_status",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "address_control_cells"})
        if not thresholds.empty
        else pd.DataFrame()
    )

    lines = [
        "# OBS-085b — Conditional Gate-Passage Sensitivity",
        "",
        "## State",
        "",
        f"`{state}`",
        "",
        (
            "OBS-085b estimates conditional frozen-gate passage only for the "
            "OBS-085b0-authorized missingness simulator cells. It does not "
            "estimate classical power or establish a minimum detectable effect."
        ),
        "",
        "## Frozen lineage",
        "",
        f"- OBS-085b0 manifest ID: `{lineage['obs085b0_manifest_id']}`",
        f"- OBS-085b0 script version: `{lineage['obs085b0_script_version']}`",
        f"- OBS-085b0 output hashes checked: **{lineage['obs085b0_output_hashes_checked']}**",
        f"- Current repository HEAD: `{lineage['current_repo_head']}`",
        "",
        "## Authorized scope",
        "",
        f"- Frozen qualification addresses reused: **{addresses['address_id'].nunique()}**",
        f"- Authorized predicate: `{AUTHORIZED_PREDICATE}`",
        f"- Authorized predicate × partition × simulator cells: **{len(authorized)}**",
        "",
        markdown_table(
            authorized[
                [
                    "failure_predicate",
                    "partition",
                    "simulator_id",
                    "qualification_status",
                    "scope_limits_json",
                ]
            ],
            args.max_report_rows,
        ),
        "",
        "No address was added, removed, or ranked after observing OBS-085b simulation results.",
        "",
        "## Conditional estimand",
        "",
        (
            "For each frozen address, partition, qualified simulator, effect "
            "level, and control-response condition, the primary quantity is "
            "`conditional_gate_passage_probability`."
        ),
        "",
        (
            "Discovery and confirmation remain separate. Gaussian and wild "
            "simulator results remain separate."
        ),
        "",
        "## Frozen design",
        "",
        f"- Scenarios: **{len(scenarios)}**",
        f"- Replicates per address × partition × simulator × scenario: **{args.replicates:,}**",
        f"- Effect grid: `{args.effect_grid}`",
        f"- Control-response grid: `{args.control_response_grid}`",
        f"- Complete replicate vectors retained: **{args.write_replicates}**",
        "",
        "## Gate contract",
        "",
        markdown_table(gate_contract, args.max_report_rows),
        "",
        "## Null calibration",
        "",
        markdown_table(null_summary, args.max_report_rows),
        "",
        "## Highest tested effect",
        "",
        markdown_table(endpoint[endpoint_columns], args.max_report_rows),
        "",
        "## Tested-grid thresholds",
        "",
        markdown_table(threshold_counts, args.max_report_rows),
        "",
        (
            "Threshold rows report the smallest tested delta reaching 50%, "
            "80%, or 90% passage. No interpolation or extrapolation is used."
        ),
        "",
        "## Failures",
        "",
        markdown_table(failures, args.max_report_rows),
        "",
        "## Multiplicity boundary",
        "",
        (
            "The primary multiplicity contract is M1 because each estimand is "
            "conditional on one fixed predeclared address. The sealed OBS-084b "
            "M13 family contained no authorized missingness candidate family; "
            "OBS-085b therefore does not invent or approximate an M13 "
            "Benjamini-Hochberg probability."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "> Monte Carlo intervals quantify simulation-sampling error only; "
            "they do not quantify simulator-model uncertainty."
        ),
        "",
        (
            "> A gate-passage probability is conditional on the frozen address, "
            "qualified simulator, declared missingness mechanism, and fixed gate "
            "contract. It is not evidence that missingness is the true cause of "
            "an observed failure."
        ),
        "",
        (
            "> Between-simulator spread is a model-sensitivity diagnostic, not "
            "a confidence interval."
        ),
        "",
        (
            "OBS-085b does not alter the null FL3 result of OBS-084 and cannot "
            "increase any address beyond its frozen OBS-085a claim entitlement."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    repo_root: Path,
    outputs: Mapping[str, Path],
    state: str,
    lineage: Mapping[str, Any],
    authorized: pd.DataFrame,
    addresses: pd.DataFrame,
    scenarios: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    expected_replicates: int,
    written_replicates: int,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for name, path in sorted(outputs.items()):
        if name == "obs085b_manifest.json" or not path.exists() or not path.is_file():
            continue
        artifacts.append(
            {
                "artifact_path": repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    core = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "state": state,
        "scope": "conditional simulator-qualified gate-passage metrology",
        "claim_ceiling": (
            "conditional gate-passage sensitivity only; no classical power, "
            "minimum detectable effect, causal attribution, or simulator truth"
        ),
        "frozen_lineage": dict(lineage),
        "authorized_cells": authorized[
            [
                "failure_predicate",
                "partition",
                "simulator_id",
                "qualification_status",
                "scope_limits_json",
            ]
        ].to_dict("records"),
        "frozen_address_ids": sorted(addresses["address_id"].astype(str).unique()),
        "scenario_ids": sorted(scenarios["scenario_id"].astype(str).unique()),
        "execution": {
            "master_seed": args.master_seed,
            "replicates_per_cell": args.replicates,
            "expected_replicate_rows": expected_replicates,
            "written_replicate_rows": written_replicates,
            "effect_grid": parse_number_grid(args.effect_grid, name="effect grid"),
            "control_response_grid": parse_number_grid(
                args.control_response_grid,
                name="control-response grid",
            ),
            "smoke": args.smoke,
            "address_limit": args.address_limit,
            "complete_replicate_vectors_written": args.write_replicates,
        },
        "gate_contract": {
            "multiplicity_family": "M1_fixed_predeclared_address",
            "m13_status": "not_identified_for_authorized_missingness_panel",
            "alpha": args.alpha,
            "minimum_effect": args.minimum_effect,
            "minimum_direction_consistency": args.minimum_direction_consistency,
            "minimum_control_adjusted_effect": args.minimum_control_adjusted_effect,
            "minimum_positive_control_share": args.minimum_positive_control_share,
            "cluster_unit": "object",
            "bootstrap": "exact object-cluster bootstrap when n^n <= 4096",
            "permutation": "exact one-sided object-cluster sign flip when 2^n <= 4096",
        },
        "summary_row_count": len(summary),
        "output_artifacts": artifacts,
        "mandatory_statements": [
            "Monte Carlo precision is not simulator-model certainty.",
            "Between-simulator spread is not a confidence interval.",
            "Conditional gate passage does not establish a minimum detectable effect.",
            "Conditional gate passage cannot increase frozen claim entitlement.",
        ],
    }
    manifest_id = sha256_bytes(canonical_json(core).encode("utf-8"))
    return {"obs085b_manifest_id": manifest_id, **core}


# -----------------------------------------------------------------------------
# Simulation orchestration
# -----------------------------------------------------------------------------


def run_simulation(
    b0: ModuleType,
    prepared: Sequence[Any],
    simulators: Sequence[Any],
    contract: Any,
    scenarios: pd.DataFrame,
    args: argparse.Namespace,
    replicate_path: Path | None,
    failures: list[StudyFailure],
) -> tuple[OnlineSummary, int]:
    accumulator = OnlineSummary()
    written = 0
    with ReplicateWriter(replicate_path, args.replicate_chunk_size) as writer:
        for address in prepared:
            for simulator in simulators:
                for scenario in scenarios.itertuples(index=False):
                    for replicate in range(args.replicates):
                        seed = stable_seed(
                            args.master_seed,
                            address.address_id,
                            address.partition,
                            simulator.simulator_id,
                            scenario.scenario_id,
                            replicate,
                        )
                        try:
                            row = simulate_gate_replicate(
                                b0,
                                address,
                                simulator,
                                contract,
                                str(scenario.scenario_id),
                                float(scenario.delta),
                                float(scenario.control_response_lambda),
                                replicate,
                                seed,
                                args,
                            )
                        except Exception as exc:
                            failures.append(
                                StudyFailure(
                                    stage="replicate_simulation",
                                    scope_id=(
                                        f"{address.address_id}::{address.partition}::"
                                        f"{simulator.simulator_id}::{scenario.scenario_id}::"
                                        f"{replicate}"
                                    ),
                                    reason="replicate_failed",
                                    detail=str(exc),
                                    severity="error",
                                )
                            )
                            continue
                        accumulator.add(row)
                        writer.add(row)
                        written += 1
    return accumulator, written


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def run_self_test(b0: ModuleType) -> int:
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
    contract = b0.predicate_contracts()[AUTHORIZED_PREDICATE]
    simulator_map = {spec.simulator_id: spec for spec in b0.simulator_specs()}

    args = argparse.Namespace(
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
    null_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    simulator = simulator_map["joint_wild_cluster_rademacher"]
    for replicate in range(64):
        null_rows.append(
            simulate_gate_replicate(
                b0,
                address,
                simulator,
                contract,
                "null",
                0.0,
                0.0,
                replicate,
                stable_seed("self-test", "null", replicate),
                args,
            )
        )
        effect_rows.append(
            simulate_gate_replicate(
                b0,
                address,
                simulator,
                contract,
                "effect",
                2.0,
                0.0,
                replicate,
                stable_seed("self-test", "effect", replicate),
                args,
            )
        )
    null_response = mean_or_nan(
        row["target_response_from_simulated_null"] for row in null_rows
    )
    effect_response = mean_or_nan(
        row["target_response_from_simulated_null"] for row in effect_rows
    )
    if abs(null_response) > 1e-12:
        raise AssertionError(f"Null injection changed target response: {null_response}")
    if not effect_response > 0:
        raise AssertionError(f"Positive missingness injection did not respond: {effect_response}")
    if any(row["range_violation_count"] for row in null_rows + effect_rows):
        raise AssertionError("Range violation in self-test")
    if any(row["identity_violation_count"] for row in null_rows + effect_rows):
        raise AssertionError("Identity violation in self-test")
    low, high = wilson_interval(50, 100)
    if not (0 < low < 0.5 < high < 1):
        raise AssertionError("Wilson interval self-test failed")
    vector = np.asarray([0.1, 0.2, 0.3, 0.4])
    uncertainty = exact_cluster_uncertainty(vector, 0.10)
    if uncertainty["permutation_count"] != 16:
        raise AssertionError("Exact sign-flip count self-test failed")
    if uncertainty["loo_successful_count"] != 4:
        raise AssertionError("LOO count self-test failed")
    print(
        "OBS-085b self-test passed: deterministic seeds, missingness direction, "
        "range/identity preservation, exact cluster inference, and Wilson intervals"
    )
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    b0_script = repo_path(args.repo_root, args.obs085b0_script)
    b0 = load_module_from_path(b0_script, "obs085b0_frozen_instrument")

    if args.self_test:
        return run_self_test(b0)

    if not (0 < args.alpha < 1):
        raise ValueError("--alpha must be in (0, 1)")
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2")
    if args.max_controls < 1:
        raise ValueError("--max-controls must be at least 1")
    if args.replicate_chunk_size < 1:
        raise ValueError("--replicate-chunk-size must be positive")
    if args.minimum_site_rows < 1 or args.minimum_complement_rows < 1:
        raise ValueError("row minima must be positive")
    if args.minimum_shared_clusters < 2:
        raise ValueError("--minimum-shared-clusters must be at least 2")

    if args.smoke:
        args.replicates = 8
        args.effect_grid = "0.00,0.50,1.00"
        args.control_response_grid = "0.00,1.00"
        args.address_limit = 1

    effect_grid = parse_number_grid(args.effect_grid, name="effect grid")
    control_grid = parse_number_grid(
        args.control_response_grid,
        name="control-response grid",
    )
    if 0.0 not in effect_grid:
        raise ValueError("Effect grid must include delta=0")

    paths = input_paths(args)
    validate_required_inputs(paths)
    _, authorized, panel, lineage = validate_obs085b0_lineage(args, paths)

    diagnostic_address_limit = args.address_limit is not None and not args.smoke
    if args.address_limit is not None:
        if args.address_limit < 1:
            raise ValueError("--address-limit must be positive")
        panel = panel.head(args.address_limit).copy()

    control_availability = read_csv(paths["obs085a_control_availability"])
    discovery_observations = read_csv(
        paths["obs084b_discovery_observation_losses"],
        dtype={"record_id": "string"},
    )
    confirmation_observations = read_csv(
        paths["obs084c_confirmation_observation_losses"],
        dtype={"record_id": "string"},
    )
    for label, frame in (
        ("discovery observation losses", discovery_observations),
        ("confirmation observation losses", confirmation_observations),
    ):
        require_columns(
            frame,
            [
                "record_id",
                "true_regime",
                "cluster_id",
                "predicted_probability",
                "max_other_probability",
            ],
            label,
        )

    contracts = b0.predicate_contracts()
    if AUTHORIZED_PREDICATE not in contracts:
        raise RuntimeError("Frozen OBS-085b0 script lacks the authorized predicate contract")
    contract = contracts[AUTHORIZED_PREDICATE]
    simulator_map = {spec.simulator_id: spec for spec in b0.simulator_specs()}
    missing_specs = [name for name in AUTHORIZED_SIMULATORS if name not in simulator_map]
    if missing_specs:
        raise RuntimeError(f"Frozen OBS-085b0 script lacks simulators: {missing_specs}")
    simulators = [simulator_map[name] for name in AUTHORIZED_SIMULATORS]

    preparation_failures: list[Any] = []
    prepared, baseline_vectors = b0.prepare_selected_addresses(
        panel,
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
                severity=str(values.get("severity", "error")),
            )
        )

    expected_prepared = len(panel) * len(AUTHORIZED_PARTITIONS)
    if len(prepared) != expected_prepared:
        raise RuntimeError(
            "Not every frozen address × partition could be prepared: "
            f"expected={expected_prepared}; observed={len(prepared)}"
        )
    observed_prepared_cells = {
        (str(item.address_id), str(item.partition)) for item in prepared
    }
    expected_prepared_cells = {
        (str(address_id), partition)
        for address_id in panel["address_id"].astype(str)
        for partition in AUTHORIZED_PARTITIONS
    }
    if observed_prepared_cells != expected_prepared_cells:
        raise RuntimeError("Prepared address × partition identity mismatch")

    scenarios = build_scenario_manifest(
        effect_grid,
        control_grid,
        args.replicates,
        args.master_seed,
        args.smoke,
    )
    expected_replicates = (
        len(prepared) * len(simulators) * len(scenarios) * args.replicates
    )

    print("OBS-085b validation complete: frozen OBS-085b0 authorization valid")
    print(
        "Authorized address panel: "
        f"{panel['address_id'].nunique():,} addresses × "
        f"{len(AUTHORIZED_PARTITIONS)} partitions × "
        f"{len(simulators)} qualified simulators"
    )
    print(
        "Conditional scenarios: "
        f"{len(scenarios):,}; expected replicate rows: {expected_replicates:,}"
    )
    if args.validate_only:
        return 0

    output_dir = repo_path(args.repo_root, args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    outputs = output_paths(output_dir)
    replicate_path = (
        outputs["obs085b_replicate_outcomes.csv.gz"]
        if args.write_replicates
        else None
    )

    accumulator, written_replicates = run_simulation(
        b0,
        prepared,
        simulators,
        contract,
        scenarios,
        args,
        replicate_path,
        failures,
    )
    if written_replicates != expected_replicates:
        raise RuntimeError(
            "Conditional replicate count mismatch: "
            f"expected={expected_replicates}; written={written_replicates}"
        )

    summary = accumulator.summary_frame()
    gate_failures = gate_failure_summary(summary)
    failure_combinations = accumulator.failure_combination_frame()
    ordered_passage = accumulator.ordered_pass_frame()
    aggregate = aggregate_address_curves(summary)
    thresholds = minimum_tested_effects(summary)
    envelope = simulator_envelope(summary)
    entitlement = entitlement_overlay(summary)
    null_summary = null_calibration(summary)
    contracts_frame = gate_contract_frame(gate_contracts(args))
    failures_frame = pd.DataFrame([asdict(failure) for failure in failures])
    if failures_frame.empty:
        failures_frame = pd.DataFrame(
            columns=["stage", "scope_id", "reason", "detail", "severity"]
        )

    selected_columns = [
        column
        for column in [
            "qualification_address_id",
            "qualification_selection_rank",
            "address_id",
            "record_id",
            "relation",
            "carrier",
            "support_id",
            "support_definition",
            "support_query_json",
            "failure_predicate",
            "entitlement_status",
            "selection_reason",
            "sealed_obs084b_candidate",
        ]
        if column in panel.columns
    ]
    input_manifest = build_input_manifest(paths, args.repo_root)

    frames = {
        "obs085b_input_manifest.csv": input_manifest,
        "obs085b_authorized_cells.csv": authorized,
        "obs085b_address_manifest.csv": panel[selected_columns],
        "obs085b_gate_contract.csv": contracts_frame,
        "obs085b_scenario_manifest.csv": scenarios,
        "obs085b_gate_passage_summary.csv": summary,
        "obs085b_gate_failure_summary.csv": gate_failures,
        "obs085b_failure_combinations.csv": failure_combinations,
        "obs085b_ordered_gate_passage.csv": ordered_passage,
        "obs085b_address_curve_summary.csv": aggregate,
        "obs085b_minimum_tested_effect.csv": thresholds,
        "obs085b_simulator_envelope.csv": envelope,
        "obs085b_entitlement_overlay.csv": entitlement,
        "obs085b_null_calibration.csv": null_summary,
        "obs085b_failures.csv": failures_frame,
    }
    for name, frame in frames.items():
        frame.to_csv(outputs[name], index=False)

    if args.smoke:
        state = "engineering_smoke_completed"
    elif diagnostic_address_limit:
        state = "diagnostic_subset_completed_no_canonical_sensitivity_decision"
    else:
        state = "conditional_gate_passage_sensitivity_completed"

    write_report(
        outputs["obs085b_report.md"],
        state,
        lineage,
        authorized,
        panel,
        scenarios,
        contracts_frame,
        summary,
        aggregate,
        thresholds,
        null_summary,
        failures_frame,
        args,
    )
    manifest = build_manifest(
        args.repo_root,
        outputs,
        state,
        lineage,
        authorized,
        panel,
        scenarios,
        summary,
        args,
        expected_replicates,
        written_replicates,
    )
    outputs["obs085b_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    threshold_reached = int(
        thresholds["threshold_status"].eq("reached_at_tested_effect").sum()
        if not thresholds.empty
        else 0
    )
    print(f"Conditional replicate rows written: {written_replicates:,}")
    print(
        "Tested-grid passage thresholds reached: "
        f"{threshold_reached:,}/{len(thresholds):,} address × partition × "
        "simulator × control-response × target-probability rows"
    )
    print(f"OBS-085b manifest ID: {manifest['obs085b_manifest_id']}")
    print(f"Outputs: {repo_relative(output_dir, args.repo_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"OBS-085b failed: {exc}", file=sys.stderr)
        raise
