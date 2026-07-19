#!/usr/bin/env python3
"""
obs085b0_simulator_qualification.py

OBS-085b0 — Injection and Simulator Qualification
==================================================

Purpose
-------
Qualify and freeze the predicate-specific injection operators, joint
object-cluster simulators, scenario definitions, null-calibration rules,
object-influence diagnostics, and deterministic random-seed controls used by
OBS-085b and OBS-085c.

This stage is instrument metrology only. It does not estimate canonical
simulated gate-passage probability, does not rerun candidate discovery, does
not reopen empirical confirmation search, and does not reinterpret OBS-084.

Scientific guardrails
---------------------
* OBS-085a and OBS-084 are read-only frozen inputs.
* Only addresses feasible in both discovery and confirmation are eligible for
  stochastic qualification.
* The realized OBS-084 target and control contrasts are centered away before
  synthetic injection.
* Injection is predicate specific. A universal additive shift to a final test
  statistic is prohibited.
* Targets and controls are simulated jointly with object as the independent
  cluster.
* Monte Carlo precision is not instrument-model certainty.
* Qualification diagnostics are not sensitivity estimates and must not be
  reported as power or detection probability.

Default inputs
--------------
outputs/rig_registry/obs085_detection_envelope/obs085a_structural_feasibility/
  obs085a_manifest.json
  obs085a_evidence_feasibility.csv
  obs085a_support_address_inventory.csv
  obs085a_control_availability.csv

outputs/rig_registry/obs084_direct_failure_witness/discovery/
  obs084b_discovery_observation_losses.csv

outputs/rig_registry/obs084_direct_failure_witness/confirmation/
  obs084c_confirmation_observation_losses.csv

Default outputs
---------------
outputs/rig_registry/obs085_detection_envelope/
  obs085b0_simulator_qualification/
    obs085b0_input_manifest.csv
    obs085b0_injection_contracts.csv
    obs085b0_simulator_specs.csv
    obs085b0_scenario_manifest.csv
    obs085b0_qualification_address_manifest.csv
    obs085b0_baseline_object_vectors.csv
    obs085b0_qualification_runs.csv
    obs085b0_null_preservation_audit.csv
    obs085b0_directionality_audit.csv
    obs085b0_specificity_audit.csv
    obs085b0_covariance_audit.csv
    obs085b0_object_influence_audit.csv
    obs085b0_reproducibility_audit.csv
    obs085b0_qualification_gate_matrix.csv
    obs085b0_simulator_family_summary.csv
    obs085b0_failures.csv
    obs085b0_manifest.json
    obs085b0_report.md

Canonical command
-----------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085b0_simulator_qualification.py

Validation-only command
-----------------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085b0_simulator_qualification.py --validate-only

Fast engineering smoke run
--------------------------
PYTHONPATH=src .venv/bin/python \\
  experiments/studies/obs085b0_simulator_qualification.py \\
  --smoke --overwrite

Interpretation ceiling
----------------------
A qualified simulator is a validated experimental instrument under its frozen
contract. Qualification is not evidence that the simulator is the true
failure-generating mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.2"
SCHEMA_VERSION = "obs085b0_simulator_qualification_v1"
DEFAULT_EXPECTED_OBS085A_COMMIT = "f98ba65"
DEFAULT_MASTER_SEED = 85000

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
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085b0_simulator_qualification"
)
DEFAULT_OBS085A_SCRIPT = Path(
    "experiments/studies/obs085a_structural_evidence_feasibility.py"
)
DEFAULT_PROTOCOL = Path(
    "docs/05_project/"
    "085_failure_support_detection_power_and_confirmation_feasibility_protocol.md"
)

EXPECTED_ADDRESS_UNIVERSE = 5736
EXPECTED_BOTH_FEASIBLE_ADDRESSES = 984
EXPECTED_BOTH_FEASIBLE_SUPPORTS = 246

IDENTITY_COLUMNS = [
    "case",
    "object",
    "cluster_id",
    "scale_index_from",
    "scale_index_to",
    "cohort",
    "observation_id",
    "observation_key",
    "transition",
    "transition_midpoint",
    "partition",
    "scale_band",
    "seam_relative_region",
    "record_id",
    "relation",
    "carrier",
    "true_regime",
]

PREDICATE_ORDER = [
    "relation_separation_attenuation",
    "log_loss_attenuation",
    "local_criterion_breach",
    "measurement_missingness_concentration",
]

QUALIFICATION_GATE_COLUMNS = [f"q{i}_pass" for i in range(1, 11)]


@dataclass(frozen=True)
class PredicateContract:
    failure_predicate: str
    failure_mode: str
    metric: str
    lowest_valid_artifact_level: str
    mathematical_scale: str
    range_preserving_transform: str
    target_fields_modified: tuple[str, ...]
    fields_must_remain_unchanged: tuple[str, ...]
    control_fields_modified: tuple[str, ...]
    target_control_response: str
    missingness_behavior: str
    transport_behavior: str
    estimator_rerun: str
    inverse_transform: str
    invalid_parameter_combinations: str
    interpretation_ceiling: str
    effect_scale_definition: str


@dataclass(frozen=True)
class SimulatorSpec:
    simulator_id: str
    simulator_family: str
    primary_simulator: bool
    residual_generation: str
    scale_estimation: str
    heterogeneity_model: str
    support_prevalence_model: str
    covariance_model: str
    missingness_model: str
    joint_target_control: bool
    cluster_unit: str
    artifact_level: str
    scope_limit: str


@dataclass(frozen=True)
class QualificationFailure:
    stage: str
    scope_id: str
    reason: str
    detail: str = ""
    severity: str = "warning"


@dataclass
class PreparedComponent:
    record_id: str
    control_family: str
    frame: pd.DataFrame
    support_mask: pd.Series
    center_offset: float
    baseline_contrast: float
    native_scale: float


@dataclass
class PreparedAddress:
    address_id: str
    record_id: str
    support_id: str
    failure_predicate: str
    relation: str
    carrier: str
    support_definition: str
    support_query_json: str
    entitlement_status: str
    partition: str
    target: PreparedComponent
    controls: list[PreparedComponent]
    cluster_ids: tuple[str, ...]
    selection_reason: str


# -----------------------------------------------------------------------------
# Argument parsing and generic utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
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
    parser.add_argument("--obs085a-script", type=Path, default=DEFAULT_OBS085A_SCRIPT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-obs085a-commit",
        default=DEFAULT_EXPECTED_OBS085A_COMMIT,
        help="Frozen OBS-085a repository commit that must be an ancestor of HEAD.",
    )
    parser.add_argument(
        "--require-exact-obs085a-commit",
        action="store_true",
        help="Require HEAD itself to equal --expected-obs085a-commit.",
    )
    parser.add_argument("--addresses-per-predicate", type=int, default=6)
    parser.add_argument("--max-controls", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=128)
    parser.add_argument(
        "--effect-grid",
        default="0.00,0.25,0.50,1.00",
        help="Qualification-only injection grid; not a sensitivity grid result.",
    )
    parser.add_argument(
        "--control-response-grid",
        default="0.00,0.50,1.00",
    )
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--finite-rate-min", type=float, default=0.95)
    parser.add_argument("--null-bias-tolerance", type=float, default=0.30)
    parser.add_argument("--directionality-min-rho", type=float, default=0.65)
    parser.add_argument("--monotonic-fraction-min", type=float, default=2.0 / 3.0)
    parser.add_argument("--specificity-ratio-max", type=float, default=3.0)
    parser.add_argument(
        "--specificity-target-response-floor",
        type=float,
        default=1e-6,
        help=(
            "Minimum mean absolute direct target response required before "
            "forming an off-target/target specificity ratio."
        ),
    )
    parser.add_argument("--object-influence-limit", type=float, default=2.5)
    parser.add_argument("--max-report-rows", type=int, default=200)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-upstream-validation-command",
        action="store_true",
        help="Do not invoke OBS-085a --validate-only; file/hash validation still runs.",
    )
    parser.add_argument(
        "--allow-unexpected-counts",
        action="store_true",
        help="Permit a valid alternative frozen universe with different counts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Engineering run: 1 address/predicate, 8 replicates, reduced grids.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal deterministic injection/simulator tests and exit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts: Any, prefix: str = "") -> str:
    digest = sha256_bytes(canonical_json(parts).encode("utf-8"))[:24]
    return f"{prefix}{digest}"


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "pass", "passed"}


def parse_number_grid(text: str, *, name: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value: {token!r}")
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if values[0] != 0.0:
        raise ValueError(f"{name} must include 0.0 for null qualification")
    if any(v < 0 for v in values):
        raise ValueError(f"{name} cannot contain negative values in the primary grid")
    return values


def repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    require_file(path, "CSV input")
    defaults: dict[str, Any] = {"low_memory": False}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def read_json(path: Path) -> Any:
    require_file(path, "JSON input")
    return json.loads(path.read_text(encoding="utf-8"))


def run_git(repo_root: Path, args: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def markdown_table(frame: pd.DataFrame, max_rows: int) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    try:
        return display.to_markdown(index=False)
    except Exception:
        columns = [str(c) for c in display.columns]
        rows = ["| " + " | ".join(columns) + " |"]
        rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in display.iterrows():
            rows.append(
                "| "
                + " | ".join(str(row[c]).replace("|", "\\|") for c in display.columns)
                + " |"
            )
        return "\n".join(rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def robust_scale(values: Iterable[float], floor: float = 1e-6) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float(floor)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < floor:
        scale = float(np.std(array, ddof=1)) if array.size > 1 else floor
    return float(max(scale, floor))


def pearson_safe(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return float("nan")
    xa = xa[mask]
    ya = ya[mask]
    if np.std(xa) <= 0 or np.std(ya) <= 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        return float("nan")
    return pearson_safe(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def canonical_frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for column in normalized.columns:
        if pd.api.types.is_float_dtype(normalized[column]):
            normalized[column] = normalized[column].round(12)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)


# -----------------------------------------------------------------------------
# Frozen lineage validation
# -----------------------------------------------------------------------------


def collect_manifest_hash_records(value: Any) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    if isinstance(value, dict):
        path_value = None
        hash_value = None
        for key in ("artifact_path", "output_path", "path", "file", "relative_path"):
            if key in value and isinstance(value[key], str):
                path_value = value[key]
                break
        for key in ("sha256", "artifact_sha256", "output_sha256", "file_sha256"):
            if key in value and isinstance(value[key], str):
                hash_value = value[key]
                break
        if path_value and hash_value and len(hash_value) == 64:
            records.append((path_value, hash_value))
        for child in value.values():
            records.extend(collect_manifest_hash_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(collect_manifest_hash_records(child))
    return records


def validate_obs085a_lineage(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    obs085a_dir = repo_path(repo_root, args.obs085a_dir)
    obs085a_script = repo_path(repo_root, args.obs085a_script)
    manifest_path = obs085a_dir / "obs085a_manifest.json"

    require_file(obs085a_script, "OBS-085a validation script")
    require_file(manifest_path, "OBS-085a manifest")

    expected_commit = str(args.expected_obs085a_commit).strip()
    head = run_git(repo_root, ["rev-parse", "HEAD"])
    if expected_commit:
        resolved_expected = run_git(
            repo_root,
            ["rev-parse", f"{expected_commit}^{{commit}}"],
        )
        if args.require_exact_obs085a_commit:
            if head != resolved_expected:
                raise RuntimeError(
                    "HEAD does not equal the frozen OBS-085a commit: "
                    f"HEAD={head}, expected={resolved_expected}"
                )
        else:
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", resolved_expected, head],
                cwd=repo_root,
                check=False,
            )
            if ancestor.returncode != 0:
                raise RuntimeError(
                    "Frozen OBS-085a commit is not an ancestor of HEAD: "
                    f"expected={resolved_expected}, HEAD={head}"
                )
    else:
        resolved_expected = ""

    validation_stdout = ""
    if not args.skip_upstream_validation_command:
        command = [
            sys.executable,
            str(obs085a_script),
            "--repo-root",
            str(repo_root),
            "--validate-only",
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OBS-085a --validate-only failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        validation_stdout = completed.stdout.strip()

    manifest = read_json(manifest_path)
    checked_hashes: list[dict[str, str]] = []
    for raw_path, expected_hash in collect_manifest_hash_records(manifest):
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.is_file():
            continue
        observed_hash = sha256_file(candidate)
        if observed_hash != expected_hash:
            raise RuntimeError(
                "OBS-085a manifest hash mismatch for "
                f"{candidate}: expected={expected_hash}, observed={observed_hash}"
            )
        checked_hashes.append(
            {
                "path": repo_relative(candidate, repo_root),
                "sha256": observed_hash,
            }
        )

    return {
        "head_commit": head,
        "expected_obs085a_commit": resolved_expected,
        "obs085a_manifest_path": repo_relative(manifest_path, repo_root),
        "obs085a_manifest_sha256": sha256_file(manifest_path),
        "obs085a_validation_stdout": validation_stdout,
        "manifest_hashes_checked": checked_hashes,
    }


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.repo_root.resolve()
    obs085a = repo_path(root, args.obs085a_dir)
    discovery = repo_path(root, args.obs084_discovery_dir)
    confirmation = repo_path(root, args.obs084_confirmation_dir)
    return {
        "obs085a_manifest": obs085a / "obs085a_manifest.json",
        "obs085a_evidence_feasibility": obs085a / "obs085a_evidence_feasibility.csv",
        "obs085a_support_address_inventory": obs085a / "obs085a_support_address_inventory.csv",
        "obs085a_control_availability": obs085a / "obs085a_control_availability.csv",
        "obs085a_effective_evidence": obs085a / "obs085a_effective_evidence.csv",
        "obs084b_discovery_observation_losses": discovery
        / "obs084b_discovery_observation_losses.csv",
        "obs084c_confirmation_observation_losses": confirmation
        / "obs084c_confirmation_observation_losses.csv",
        "obs085_protocol": repo_path(root, args.protocol),
        "obs085a_script": repo_path(root, args.obs085a_script),
        "obs085b0_script": Path(__file__).resolve(),
    }


def validate_required_inputs(paths: Mapping[str, Path]) -> None:
    for role, path in paths.items():
        require_file(path, role)


# -----------------------------------------------------------------------------
# Contracts and simulator family
# -----------------------------------------------------------------------------


def predicate_contracts() -> dict[str, PredicateContract]:
    unchanged = tuple(IDENTITY_COLUMNS)
    probability_fields = (
        "predicted_probability",
        "max_other_probability",
        "true_class_margin",
        "signed_margin",
        "correct",
        "misclassification_loss",
        "margin_loss",
        "log_loss",
    )
    interpretation = (
        "Instrument qualification conditional on the frozen diagnostic-loss "
        "artifact and the retained true-class/maximum-alternative probability "
        "pair. It does not reconstruct the complete class-probability vector "
        "or establish a true failure-generating mechanism."
    )
    return {
        "relation_separation_attenuation": PredicateContract(
            failure_predicate="relation_separation_attenuation",
            failure_mode="attenuation",
            metric="margin_loss",
            lowest_valid_artifact_level="diagnostic probability pair",
            mathematical_scale="true-class versus maximum-alternative log odds",
            range_preserving_transform="pair-mass-preserving logistic odds attenuation",
            target_fields_modified=probability_fields,
            fields_must_remain_unchanged=unchanged,
            control_fields_modified=probability_fields,
            target_control_response="control log-odds attenuation equals lambda times target attenuation",
            missingness_behavior="frozen observed missingness; no new missingness",
            transport_behavior="partition-specific centered baselines with common frozen operator",
            estimator_rerun="equal-weight class-balanced margin-loss site/complement contrast",
            inverse_transform="logistic inverse with original true-plus-max-other pair mass preserved",
            invalid_parameter_combinations="delta < 0; lambda outside frozen scenario grid; unavailable probability pair",
            interpretation_ceiling=interpretation,
            effect_scale_definition="delta times robust object-cluster scale of pair-logit site/complement contributions, floored at 0.25 log-odds",
        ),
        "log_loss_attenuation": PredicateContract(
            failure_predicate="log_loss_attenuation",
            failure_mode="attenuation",
            metric="log_loss",
            lowest_valid_artifact_level="diagnostic true-class probability",
            mathematical_scale="true-class versus maximum-alternative log odds",
            range_preserving_transform="pair-mass-preserving logistic odds attenuation",
            target_fields_modified=probability_fields,
            fields_must_remain_unchanged=unchanged,
            control_fields_modified=probability_fields,
            target_control_response="control log-odds attenuation equals lambda times target attenuation",
            missingness_behavior="frozen observed missingness; no new missingness",
            transport_behavior="partition-specific centered baselines with common frozen operator",
            estimator_rerun="equal-weight class-balanced log-loss site/complement contrast",
            inverse_transform="logistic inverse with original true-plus-max-other pair mass preserved",
            invalid_parameter_combinations="delta < 0; lambda outside frozen scenario grid; unavailable probability pair",
            interpretation_ceiling=interpretation,
            effect_scale_definition="delta times robust object-cluster scale of pair-logit site/complement contributions, floored at 0.25 log-odds",
        ),
        "local_criterion_breach": PredicateContract(
            failure_predicate="local_criterion_breach",
            failure_mode="threshold_breach",
            metric="misclassification_loss",
            lowest_valid_artifact_level="diagnostic probability ordering",
            mathematical_scale="true-class versus maximum-alternative log odds",
            range_preserving_transform="pair-mass-preserving logistic odds attenuation through the frozen argmax criterion",
            target_fields_modified=probability_fields,
            fields_must_remain_unchanged=unchanged,
            control_fields_modified=probability_fields,
            target_control_response="control log-odds attenuation equals lambda times target attenuation",
            missingness_behavior="frozen observed missingness; no new missingness",
            transport_behavior="partition-specific centered baselines with common frozen operator",
            estimator_rerun="equal-weight class-balanced misclassification-loss site/complement contrast",
            inverse_transform="logistic inverse followed by frozen correct/incorrect criterion",
            invalid_parameter_combinations="delta < 0; lambda outside frozen scenario grid; unavailable probability pair",
            interpretation_ceiling=(
                interpretation
                + " The binary breach response is discrete and may be stepwise rather than smooth."
            ),
            effect_scale_definition="delta times robust object-cluster pair-logit scale, floored at 0.50 log-odds",
        ),
        "measurement_missingness_concentration": PredicateContract(
            failure_predicate="measurement_missingness_concentration",
            failure_mode="missingness_concentration",
            metric="predictor_missing_fraction",
            lowest_valid_artifact_level="measurement-availability indicator",
            mathematical_scale="support-concentrated cumulative missingness hazard",
            range_preserving_transform="p_missing = 1 - exp(-hazard)",
            target_fields_modified=("predictor_missing_any", "predictor_missing_fraction"),
            fields_must_remain_unchanged=unchanged,
            control_fields_modified=("predictor_missing_any", "predictor_missing_fraction"),
            target_control_response="control missingness hazard equals lambda times target hazard",
            missingness_behavior="existing missingness retained; new missingness sampled only inside frozen support",
            transport_behavior="partition-specific empirical incidence with common frozen hazard operator",
            estimator_rerun="equal-weight class-balanced missingness-fraction site/complement contrast",
            inverse_transform="not applicable; Bernoulli realization retained as availability artifact",
            invalid_parameter_combinations="delta < 0; lambda outside frozen scenario grid; support unavailable",
            interpretation_ceiling=(
                "Instrument qualification conditional on a declared support-concentrated "
                "missingness mechanism. It does not identify the empirical cause or "
                "mechanism of missing measurements."
            ),
            effect_scale_definition="delta times frozen 0.75 cumulative-hazard unit with simulator-specific object heterogeneity",
        ),
    }


def simulator_specs() -> list[SimulatorSpec]:
    return [
        SimulatorSpec(
            simulator_id="joint_empirical_cluster_bootstrap",
            simulator_family="artifact_level_joint_cluster_resampling",
            primary_simulator=True,
            residual_generation="joint object-cluster bootstrap with a shared source-cluster draw across target and controls",
            scale_estimation="address-specific robust object-cluster scale with predicate floor",
            heterogeneity_model="empirical frozen object heterogeneity",
            support_prevalence_model="empirical support incidence carried by resampled objects",
            covariance_model="empirical joint target-control dependence through shared cluster resampling",
            missingness_model="frozen empirical missingness plus predicate-declared support hazard",
            joint_target_control=True,
            cluster_unit="object",
            artifact_level="diagnostic observation-loss artifact",
            scope_limit="four observed objects per partition; empirical cluster distribution is weakly identified",
        ),
        SimulatorSpec(
            simulator_id="joint_wild_cluster_rademacher",
            simulator_family="artifact_level_joint_wild_cluster",
            primary_simulator=True,
            residual_generation="shared Rademacher reflection of centered component-level cluster residuals",
            scale_estimation="address-specific robust object-cluster scale with predicate floor",
            heterogeneity_model="empirical cluster leverage with sign-reflected residual realization",
            support_prevalence_model="fixed empirical support incidence",
            covariance_model="cross-component residual signs shared within object cluster",
            missingness_model="fixed empirical baseline plus predicate-declared support hazard",
            joint_target_control=True,
            cluster_unit="object",
            artifact_level="diagnostic probability-pair or measurement-availability artifact",
            scope_limit="wild residual symmetry is a declared model assumption with four clusters",
        ),
        SimulatorSpec(
            simulator_id="joint_gaussian_regularized_cluster",
            simulator_family="artifact_level_regularized_hierarchical_cluster",
            primary_simulator=True,
            residual_generation="joint Gaussian cluster-effect draw on the predicate artifact scale",
            scale_estimation="regularized empirical covariance with 35% diagonal shrinkage and variance floor",
            heterogeneity_model="regularized empirical between-object heterogeneity",
            support_prevalence_model="fixed empirical support incidence",
            covariance_model="positive-semidefinite shrinkage covariance across target and controls",
            missingness_model="predicate-declared support hazard with joint Gaussian object effects",
            joint_target_control=True,
            cluster_unit="object",
            artifact_level="diagnostic probability-pair or measurement-availability artifact",
            scope_limit="Gaussian tails and covariance shrinkage are declared simulator assumptions",
        ),
    ]


def contract_frame(contracts: Mapping[str, PredicateContract]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for predicate in PREDICATE_ORDER:
        row = asdict(contracts[predicate])
        for key, value in list(row.items()):
            if isinstance(value, tuple):
                row[key] = json.dumps(list(value))
        row["contract_id"] = stable_id(row, prefix="IC-")
        rows.append(row)
    return pd.DataFrame(rows)


def simulator_frame(specs: Sequence[SimulatorSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = asdict(spec)
        row["simulator_spec_id"] = stable_id(row, prefix="SIM-")
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Address universe and deterministic qualification selection
# -----------------------------------------------------------------------------


def merge_address_inputs(
    evidence: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        evidence,
        ["record_id", "support_id", "failure_predicate"],
        "OBS-085a evidence feasibility",
    )
    require_columns(
        inventory,
        ["record_id", "support_id"],
        "OBS-085a support address inventory",
    )

    result = evidence.copy()
    feasible_column = None
    for candidate in (
        "end_to_end_evidence_feasible",
        "both_partitions_evidence_feasible",
        "evidence_feasible_both_partitions",
    ):
        if candidate in result.columns:
            feasible_column = candidate
            break
    if feasible_column is None:
        if {
            "discovery_evidence_feasible",
            "confirmation_evidence_feasible",
        }.issubset(result.columns):
            result["end_to_end_evidence_feasible"] = (
                result["discovery_evidence_feasible"].map(normalize_bool)
                & result["confirmation_evidence_feasible"].map(normalize_bool)
            )
            feasible_column = "end_to_end_evidence_feasible"
        else:
            raise ValueError(
                "OBS-085a evidence feasibility lacks an end-to-end feasibility field"
            )
    result["_both_feasible"] = result[feasible_column].map(normalize_bool)

    merge_keys = ["record_id", "support_id"]
    if "failure_predicate" in inventory.columns:
        merge_keys.append("failure_predicate")

    useful_inventory_columns = [
        column
        for column in inventory.columns
        if column in merge_keys
        or column
        in {
            "address_id",
            "relation",
            "carrier",
            "subclass",
            "support_depth",
            "support_families",
            "support_columns",
            "support_values",
            "support_definition",
            "support_query_json",
            "complement_definition",
            "candidate_test_id",
            "sealed_obs084b_candidate",
        }
    ]
    inventory_payload_columns = [
        column for column in useful_inventory_columns if column not in merge_keys
    ]

    # OBS-085a evidence artifacts may already contain fields carrying the
    # ``_inventory`` suffix from their own provenance-preserving joins.  Passing
    # pandas merge suffixes in that state can attempt to create a duplicate
    # column such as ``support_definition_inventory``.  Canonicalize those
    # pre-existing fields before adding the support-inventory payload.
    for base in inventory_payload_columns:
        alt = f"{base}_inventory"
        if alt not in result.columns:
            continue
        if base not in result.columns:
            result = result.rename(columns={alt: base})
            continue

        both_present = result[base].notna() & result[alt].notna()
        if both_present.any():
            left = result.loc[both_present, base].astype("string").str.strip()
            right = result.loc[both_present, alt].astype("string").str.strip()
            conflict = left.ne(right)
            if conflict.any():
                sample_index = conflict[conflict].index[:10]
                sample = result.loc[sample_index, merge_keys + [base, alt]]
                raise ValueError(
                    f"Conflicting pre-existing address fields for {base!r}:\n"
                    + sample.to_string(index=False)
                )
        result[base] = result[base].where(result[base].notna(), result[alt])
        result = result.drop(columns=[alt])

    inv = inventory[useful_inventory_columns].drop_duplicates()
    duplicate_inventory_keys = inv.duplicated(merge_keys, keep=False)
    if duplicate_inventory_keys.any():
        sample = inv.loc[duplicate_inventory_keys, useful_inventory_columns].head(10)
        raise ValueError(
            "OBS-085a support inventory is not unique on merge keys:\n"
            + sample.to_string(index=False)
        )
    inv = inv.drop_duplicates(merge_keys)

    # Rename overlaps explicitly rather than relying on pandas' suffix
    # machinery.  This prevents accidental duplicate-name creation and makes
    # the provenance of every coalesced field inspectable.
    inventory_rename: dict[str, str] = {}
    for base in inventory_payload_columns:
        if base in result.columns:
            alt = f"{base}_inventory"
            if alt in result.columns:
                raise ValueError(
                    f"Reserved merge column already exists after normalization: {alt}"
                )
            inventory_rename[base] = alt
    inv = inv.rename(columns=inventory_rename)
    result = result.merge(inv, on=merge_keys, how="left", validate="one_to_one")

    for base in inventory_payload_columns:
        alt = f"{base}_inventory"
        if alt not in result.columns:
            continue
        if base not in result.columns:
            result[base] = result[alt]
        else:
            result[base] = result[base].where(result[base].notna(), result[alt])
        result = result.drop(columns=[alt])

    if "address_id" not in result.columns:
        result["address_id"] = [
            stable_id(record, support, predicate, prefix="A-")
            for record, support, predicate in zip(
                result["record_id"],
                result["support_id"],
                result["failure_predicate"],
            )
        ]
    if "relation" not in result.columns:
        result["relation"] = result["record_id"].astype(str).str.split("__").str[0]
    if "carrier" not in result.columns:
        result["carrier"] = result["record_id"].astype(str).str.split("__").str[1]
    if "support_definition" not in result.columns:
        result["support_definition"] = result["support_id"].astype(str)
    if "support_query_json" not in result.columns:
        raise ValueError("Support query JSON is required for OBS-085b0 injection")

    if "entitlement_status" not in result.columns:
        if "e1_fl3_claim_entitlement" in result.columns:
            result["entitlement_status"] = np.where(
                result["e1_fl3_claim_entitlement"].map(normalize_bool),
                "fl3_entitled",
                "fl3_entitlement_capped",
            )
        else:
            result["entitlement_status"] = "unknown"

    duplicate = result.duplicated(["record_id", "support_id", "failure_predicate"])
    if duplicate.any():
        sample = result.loc[
            duplicate,
            ["record_id", "support_id", "failure_predicate"],
        ].head(10)
        raise ValueError(
            "Address identity is not unique after merge:\n" + sample.to_string(index=False)
        )
    return result


def prevalence_value(row: pd.Series) -> float:
    candidates: list[float] = []
    for prefix in ("discovery", "confirmation", ""):
        site_candidates = [
            f"{prefix + '_' if prefix else ''}n_site_rows",
            f"{prefix + '_' if prefix else ''}support_observations",
        ]
        comp_candidates = [
            f"{prefix + '_' if prefix else ''}n_complement_rows",
            f"{prefix + '_' if prefix else ''}complement_observations",
        ]
        site = next((row.get(c) for c in site_candidates if c in row.index), None)
        comp = next((row.get(c) for c in comp_candidates if c in row.index), None)
        try:
            site_f = float(site)
            comp_f = float(comp)
            if math.isfinite(site_f) and math.isfinite(comp_f) and site_f + comp_f > 0:
                candidates.append(site_f / (site_f + comp_f))
        except (TypeError, ValueError):
            pass
    return float(np.mean(candidates)) if candidates else float("nan")


def support_family_label(row: pd.Series) -> str:
    for column in ("support_families", "support_family", "support_columns"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return "unknown"


def deterministic_address_selection(
    universe: pd.DataFrame,
    addresses_per_predicate: int,
) -> pd.DataFrame:
    feasible = universe[universe["_both_feasible"]].copy()
    feasible["support_prevalence"] = feasible.apply(prevalence_value, axis=1)
    feasible["support_family_label"] = feasible.apply(support_family_label, axis=1)
    feasible["selection_hash"] = [
        stable_id(a, r, s, q)
        for a, r, s, q in zip(
            feasible["address_id"],
            feasible["record_id"],
            feasible["support_id"],
            feasible["failure_predicate"],
        )
    ]

    selected_rows: list[pd.Series] = []
    for predicate in PREDICATE_ORDER:
        group = feasible[feasible["failure_predicate"] == predicate].copy()
        if group.empty:
            continue
        valid_prev = group["support_prevalence"].dropna()
        if valid_prev.nunique() >= 3:
            try:
                group["prevalence_bin"] = pd.qcut(
                    group["support_prevalence"],
                    q=3,
                    labels=["rare", "middle", "common"],
                    duplicates="drop",
                ).astype(str)
            except ValueError:
                group["prevalence_bin"] = "undifferentiated"
        else:
            group["prevalence_bin"] = "undifferentiated"

        group["sealed_priority"] = (
            group.get("sealed_obs084b_candidate", False)
            if "sealed_obs084b_candidate" in group.columns
            else False
        )
        group["sealed_priority"] = group["sealed_priority"].map(normalize_bool)

        remaining = group.sort_values(
            ["sealed_priority", "selection_hash"],
            ascending=[False, True],
        ).copy()
        covered: dict[str, set[str]] = {
            "relation": set(),
            "carrier": set(),
            "entitlement_status": set(),
            "support_family_label": set(),
            "prevalence_bin": set(),
        }
        chosen_indices: list[int] = []
        while len(chosen_indices) < min(addresses_per_predicate, len(remaining)):
            best_index = None
            best_score: tuple[int, int, str] | None = None
            for index, row in remaining.iterrows():
                if index in chosen_indices:
                    continue
                novelty = sum(
                    str(row.get(dimension, "unknown")) not in covered[dimension]
                    for dimension in covered
                )
                sealed_bonus = int(normalize_bool(row.get("sealed_priority", False)))
                score = (novelty, sealed_bonus, str(row["selection_hash"]))
                if best_score is None or score[:2] > best_score[:2] or (
                    score[:2] == best_score[:2] and score[2] < best_score[2]
                ):
                    best_index = index
                    best_score = score
            if best_index is None:
                break
            chosen_indices.append(best_index)
            chosen = remaining.loc[best_index]
            for dimension in covered:
                covered[dimension].add(str(chosen.get(dimension, "unknown")))

        for rank, index in enumerate(chosen_indices, start=1):
            row = remaining.loc[index].copy()
            row["qualification_selection_rank"] = rank
            row["selection_reason"] = (
                "deterministic greedy coverage of relation, carrier, entitlement, "
                "support family, support-prevalence stratum, and sealed-candidate context"
            )
            selected_rows.append(row)

    if not selected_rows:
        raise RuntimeError("No both-partition-feasible qualification addresses selected")
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["qualification_address_id"] = [
        stable_id(row["address_id"], "OBS085b0", prefix="QA-")
        for _, row in selected.iterrows()
    ]
    return selected


def validate_universe_counts(
    universe: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, int]:
    counts = {
        "address_universe": int(len(universe)),
        "both_feasible_addresses": int(universe["_both_feasible"].sum()),
        "both_feasible_record_scoped_supports": int(
            universe.loc[universe["_both_feasible"], ["record_id", "support_id"]]
            .drop_duplicates()
            .shape[0]
        ),
    }
    expected = {
        "address_universe": EXPECTED_ADDRESS_UNIVERSE,
        "both_feasible_addresses": EXPECTED_BOTH_FEASIBLE_ADDRESSES,
        "both_feasible_record_scoped_supports": EXPECTED_BOTH_FEASIBLE_SUPPORTS,
    }
    mismatches = {
        key: {"observed": counts[key], "expected": expected[key]}
        for key in counts
        if counts[key] != expected[key]
    }
    if mismatches and not args.allow_unexpected_counts:
        raise RuntimeError(
            "Frozen OBS-085a universe counts differ from the canonical run: "
            + canonical_json(mismatches)
        )
    return counts


# -----------------------------------------------------------------------------
# Support predicates, controls, and frozen estimators
# -----------------------------------------------------------------------------


def parse_json_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(";") if item.strip()]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def apply_support_query(frame: pd.DataFrame, query_json: str) -> pd.Series:
    clauses = parse_json_list(query_json)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for clause in clauses:
        if not isinstance(clause, dict):
            raise ValueError(f"Invalid support clause: {clause!r}")
        column = str(clause.get("column", ""))
        operator = str(clause.get("operator", "eq")).lower()
        value = clause.get("value")
        if column not in frame.columns:
            raise ValueError(f"Support-query column unavailable: {column}")
        series = frame[column]
        if operator in {"eq", "=="}:
            mask &= series.astype(str) == str(value)
        elif operator in {"ne", "!="}:
            mask &= series.astype(str) != str(value)
        elif operator in {"gt", ">"}:
            mask &= pd.to_numeric(series, errors="coerce") > float(value)
        elif operator in {"ge", "gte", ">="}:
            mask &= pd.to_numeric(series, errors="coerce") >= float(value)
        elif operator in {"lt", "<"}:
            mask &= pd.to_numeric(series, errors="coerce") < float(value)
        elif operator in {"le", "lte", "<="}:
            mask &= pd.to_numeric(series, errors="coerce") <= float(value)
        elif operator == "in":
            values = {str(v) for v in (value if isinstance(value, list) else [value])}
            mask &= series.astype(str).isin(values)
        elif operator == "not_in":
            values = {str(v) for v in (value if isinstance(value, list) else [value])}
            mask &= ~series.astype(str).isin(values)
        elif operator == "between":
            low, high = value
            numeric = pd.to_numeric(series, errors="coerce")
            mask &= numeric.between(float(low), float(high), inclusive="both")
        else:
            raise ValueError(f"Unsupported support-query operator: {operator}")
    return mask


def class_balanced_mean(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty or metric not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[metric], errors="coerce")
    valid = values.notna() & frame["true_regime"].notna()
    if not valid.any():
        return float("nan")
    means = values[valid].groupby(frame.loc[valid, "true_regime"].astype(str)).mean()
    if len(means) < 2:
        return float("nan")
    return float(means.mean())


def site_relative_contrast(
    frame: pd.DataFrame,
    support_mask: pd.Series,
    metric: str,
) -> float:
    site = class_balanced_mean(frame.loc[support_mask], metric)
    complement = class_balanced_mean(frame.loc[~support_mask], metric)
    if not math.isfinite(site) or not math.isfinite(complement):
        return float("nan")
    return float(site - complement)


def object_contributions(
    frame: pd.DataFrame,
    support_mask: pd.Series,
    metric: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cluster_column = "cluster_id" if "cluster_id" in frame.columns else "object"
    for cluster, indices in frame.groupby(cluster_column).groups.items():
        local = frame.loc[indices]
        local_mask = support_mask.loc[indices]
        rows.append(
            {
                "cluster_id": str(cluster),
                "object_contrast": site_relative_contrast(local, local_mask, metric),
                "site_rows": int(local_mask.sum()),
                "complement_rows": int((~local_mask).sum()),
            }
        )
    return pd.DataFrame(rows)


def pair_logit(frame: pd.DataFrame) -> np.ndarray:
    p = np.clip(
        pd.to_numeric(frame["predicted_probability"], errors="coerce").to_numpy(float),
        1e-12,
        1.0,
    )
    q = np.clip(
        pd.to_numeric(frame["max_other_probability"], errors="coerce").to_numpy(float),
        1e-12,
        1.0,
    )
    share = np.clip(p / (p + q), 1e-12, 1.0 - 1e-12)
    return np.log(share / (1.0 - share))


def set_pair_logit(frame: pd.DataFrame, new_logit: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    p = np.clip(
        pd.to_numeric(result["predicted_probability"], errors="coerce").to_numpy(float),
        1e-12,
        1.0,
    )
    q = np.clip(
        pd.to_numeric(result["max_other_probability"], errors="coerce").to_numpy(float),
        1e-12,
        1.0,
    )
    pair_mass = np.clip(p + q, 1e-12, 1.0)
    share = 1.0 / (1.0 + np.exp(-np.clip(new_logit, -40.0, 40.0)))
    p_new = np.clip(pair_mass * share, 1e-12, 1.0 - 1e-12)
    q_new = np.clip(pair_mass * (1.0 - share), 1e-12, 1.0 - 1e-12)
    margin = p_new - q_new
    correct = margin > 0
    result["predicted_probability"] = p_new
    result["max_other_probability"] = q_new
    result["true_class_margin"] = margin
    result["signed_margin"] = margin
    result["correct"] = correct
    result["misclassification_loss"] = (~correct).astype(float)
    result["margin_loss"] = -margin
    result["log_loss"] = -np.log(p_new)
    return result


def apply_pair_logit_offset(
    frame: pd.DataFrame,
    support_mask: pd.Series,
    offset: float,
) -> pd.DataFrame:
    logits = pair_logit(frame)
    logits[support_mask.to_numpy(bool)] += float(offset)
    return set_pair_logit(frame, logits)


def range_violation_count(frame: pd.DataFrame) -> int:
    count = 0
    for column in ("predicted_probability", "max_other_probability"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            count += int(((values < 0) | (values > 1)).sum())
    if "predictor_missing_fraction" in frame.columns:
        values = pd.to_numeric(frame["predictor_missing_fraction"], errors="coerce")
        count += int(((values < 0) | (values > 1)).sum())
    return count


def find_center_offset(
    frame: pd.DataFrame,
    support_mask: pd.Series,
    metric: str,
) -> tuple[float, pd.DataFrame, float]:
    baseline = site_relative_contrast(frame, support_mask, metric)
    if metric == "predictor_missing_fraction":
        return 0.0, frame.copy(), baseline
    if not math.isfinite(baseline):
        return 0.0, frame.copy(), baseline

    coarse = np.linspace(-16.0, 16.0, 257)
    best_offset = 0.0
    best_abs = abs(baseline)
    for offset in coarse:
        shifted = apply_pair_logit_offset(frame, support_mask, float(offset))
        contrast = site_relative_contrast(shifted, support_mask, metric)
        if math.isfinite(contrast) and abs(contrast) < best_abs:
            best_abs = abs(contrast)
            best_offset = float(offset)
    step = float(coarse[1] - coarse[0])
    for _ in range(4):
        local = np.linspace(best_offset - step, best_offset + step, 33)
        for offset in local:
            shifted = apply_pair_logit_offset(frame, support_mask, float(offset))
            contrast = site_relative_contrast(shifted, support_mask, metric)
            if math.isfinite(contrast) and abs(contrast) < best_abs:
                best_abs = abs(contrast)
                best_offset = float(offset)
        step /= 8.0
    centered = apply_pair_logit_offset(frame, support_mask, best_offset)
    centered_contrast = site_relative_contrast(centered, support_mask, metric)
    return best_offset, centered, centered_contrast


def extract_control_records(
    control_availability: pd.DataFrame,
    address_row: pd.Series,
    partition: str,
    available_records: set[str],
    max_controls: int,
) -> list[tuple[str, str]]:
    frame = control_availability.copy()
    for column, value in (
        ("record_id", address_row["record_id"]),
        ("support_id", address_row["support_id"]),
        ("failure_predicate", address_row["failure_predicate"]),
    ):
        if column in frame.columns:
            frame = frame[frame[column].astype(str) == str(value)]
    if "partition" in frame.columns:
        frame = frame[frame["partition"].astype(str) == partition]

    candidates: list[tuple[str, str]] = []
    direct_columns = [
        column
        for column in frame.columns
        if column.endswith("control_record_id") or column == "control_record_id"
    ]
    for _, row in frame.iterrows():
        family = str(row.get("control_family", row.get("control_type", "control")))
        for column in direct_columns:
            value = row.get(column)
            if pd.notna(value):
                candidates.append((str(value), family))
        for column in frame.columns:
            lower = column.lower()
            if "control" not in lower or "json" not in lower:
                continue
            for item in parse_json_list(row.get(column)):
                if isinstance(item, dict):
                    record_id = item.get("record_id") or item.get("control_record_id")
                    item_family = item.get("control_family") or family
                    if record_id:
                        candidates.append((str(record_id), str(item_family)))
                elif item:
                    candidates.append((str(item), family))

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record_id, family in candidates:
        if record_id == str(address_row["record_id"]):
            continue
        if record_id not in available_records:
            continue
        if record_id in seen:
            continue
        seen.add(record_id)
        unique.append((record_id, family))

    family_order = {"relation_control": 0, "carrier_control": 1, "combined_control": 2}
    unique.sort(key=lambda item: (family_order.get(item[1], 9), item[0]))
    return unique[:max_controls]


def native_injection_scale(
    frame: pd.DataFrame,
    support_mask: pd.Series,
    predicate: str,
    metric: str,
) -> float:
    if predicate == "measurement_missingness_concentration":
        return 0.75
    latent = pair_logit(frame)
    temp = frame.copy()
    temp["_latent"] = latent
    contributions = object_contributions(temp, support_mask, "_latent")
    floor = 0.50 if predicate == "local_criterion_breach" else 0.25
    return robust_scale(contributions["object_contrast"], floor=floor)


def prepare_component(
    observations: pd.DataFrame,
    record_id: str,
    control_family: str,
    support_query_json: str,
    contract: PredicateContract,
) -> PreparedComponent:
    frame = observations[observations["record_id"].astype(str) == str(record_id)].copy()
    if frame.empty:
        raise ValueError(f"No observation-loss rows for record {record_id}")
    require_columns(frame, ["true_regime", "predicted_probability", "max_other_probability"], record_id)
    if "cluster_id" not in frame.columns:
        if "object" not in frame.columns:
            raise ValueError(f"No object cluster field for record {record_id}")
        frame["cluster_id"] = frame["object"].astype(str)
    support_mask = apply_support_query(frame, support_query_json)
    if not support_mask.any() or (~support_mask).sum() == 0:
        raise ValueError(f"Support/complement unavailable for record {record_id}")
    if contract.metric not in frame.columns:
        if contract.metric == "predictor_missing_fraction":
            frame["predictor_missing_fraction"] = 0.0
            frame["predictor_missing_any"] = False
        else:
            raise ValueError(f"Metric {contract.metric} unavailable for {record_id}")
    center_offset, centered, centered_contrast = find_center_offset(
        frame,
        support_mask,
        contract.metric,
    )
    scale = native_injection_scale(
        centered,
        support_mask,
        contract.failure_predicate,
        contract.metric,
    )
    return PreparedComponent(
        record_id=str(record_id),
        control_family=str(control_family),
        frame=centered.reset_index(drop=True),
        support_mask=support_mask.reset_index(drop=True),
        center_offset=float(center_offset),
        baseline_contrast=float(centered_contrast),
        native_scale=float(scale),
    )


def prepare_selected_addresses(
    selected: pd.DataFrame,
    control_availability: pd.DataFrame,
    discovery_observations: pd.DataFrame,
    confirmation_observations: pd.DataFrame,
    contracts: Mapping[str, PredicateContract],
    max_controls: int,
    failures: list[QualificationFailure],
) -> tuple[list[PreparedAddress], pd.DataFrame]:
    prepared: list[PreparedAddress] = []
    baseline_rows: list[dict[str, Any]] = []
    by_partition = {
        "discovery": discovery_observations,
        "confirmation": confirmation_observations,
    }

    for _, address in selected.iterrows():
        predicate = str(address["failure_predicate"])
        contract = contracts[predicate]
        for partition, observations in by_partition.items():
            available_records = set(observations["record_id"].astype(str).unique())
            try:
                controls = extract_control_records(
                    control_availability,
                    address,
                    partition,
                    available_records,
                    max_controls,
                )
                if not controls:
                    raise ValueError(
                        "No usable control records extracted from OBS-085a control availability"
                    )
                target = prepare_component(
                    observations,
                    str(address["record_id"]),
                    "target",
                    str(address["support_query_json"]),
                    contract,
                )
                control_components = [
                    prepare_component(
                        observations,
                        record_id,
                        family,
                        str(address["support_query_json"]),
                        contract,
                    )
                    for record_id, family in controls
                ]
                cluster_sets = [
                    set(target.frame["cluster_id"].astype(str).unique()),
                    *[
                        set(component.frame["cluster_id"].astype(str).unique())
                        for component in control_components
                    ],
                ]
                common_clusters = tuple(sorted(set.intersection(*cluster_sets)))
                if len(common_clusters) < 2:
                    raise ValueError(
                        f"Only {len(common_clusters)} jointly estimable object clusters"
                    )
                item = PreparedAddress(
                    address_id=str(address["address_id"]),
                    record_id=str(address["record_id"]),
                    support_id=str(address["support_id"]),
                    failure_predicate=predicate,
                    relation=str(address["relation"]),
                    carrier=str(address["carrier"]),
                    support_definition=str(address["support_definition"]),
                    support_query_json=str(address["support_query_json"]),
                    entitlement_status=str(address["entitlement_status"]),
                    partition=partition,
                    target=target,
                    controls=control_components,
                    cluster_ids=common_clusters,
                    selection_reason=str(address["selection_reason"]),
                )
                prepared.append(item)

                for role, component in [
                    ("target", target),
                    *[("control", c) for c in control_components],
                ]:
                    contributions = object_contributions(
                        component.frame,
                        component.support_mask,
                        contract.metric,
                    )
                    for _, contribution in contributions.iterrows():
                        baseline_rows.append(
                            {
                                "address_id": item.address_id,
                                "partition": partition,
                                "failure_predicate": predicate,
                                "component_role": role,
                                "record_id": component.record_id,
                                "control_family": component.control_family,
                                "cluster_id": contribution["cluster_id"],
                                "object_contrast": contribution["object_contrast"],
                                "site_rows": contribution["site_rows"],
                                "complement_rows": contribution["complement_rows"],
                                "center_offset": component.center_offset,
                                "centered_baseline_contrast": component.baseline_contrast,
                                "native_injection_scale": component.native_scale,
                            }
                        )
            except Exception as exc:
                failures.append(
                    QualificationFailure(
                        stage="address_preparation",
                        scope_id=f"{address['address_id']}::{partition}",
                        reason="qualification_address_unscorable",
                        detail=str(exc),
                        severity="error",
                    )
                )
    return prepared, pd.DataFrame(baseline_rows)


# -----------------------------------------------------------------------------
# Joint simulators and predicate-specific injection
# -----------------------------------------------------------------------------


def component_latent(component: PreparedComponent, predicate: str) -> pd.Series:
    if predicate == "measurement_missingness_concentration":
        return pd.to_numeric(
            component.frame.get("predictor_missing_fraction", 0.0),
            errors="coerce",
        ).fillna(0.0)
    return pd.Series(pair_logit(component.frame), index=component.frame.index)


def component_cluster_residuals(
    components: Sequence[PreparedComponent],
    cluster_ids: Sequence[str],
    predicate: str,
) -> np.ndarray:
    matrix = np.zeros((len(cluster_ids), len(components)), dtype=float)
    for component_index, component in enumerate(components):
        latent = component_latent(component, predicate)
        cluster = component.frame["cluster_id"].astype(str)
        means = latent.groupby(cluster).mean()
        grand = float(means.mean()) if len(means) else 0.0
        for cluster_index, cluster_id in enumerate(cluster_ids):
            matrix[cluster_index, component_index] = float(means.get(cluster_id, grand) - grand)
    return matrix


def regularized_covariance(matrix: np.ndarray) -> np.ndarray:
    n_components = matrix.shape[1]
    if matrix.shape[0] < 2:
        return np.eye(n_components) * 0.05**2
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance).astype(float)
    if covariance.shape != (n_components, n_components):
        covariance = np.eye(n_components) * float(np.nanvar(matrix))
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    diagonal = np.diag(np.maximum(np.diag(covariance), 0.05**2))
    covariance = 0.65 * covariance + 0.35 * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def resample_component_by_cluster(
    component: PreparedComponent,
    source_clusters: Sequence[str],
    target_cluster_ids: Sequence[str],
) -> PreparedComponent:
    pieces: list[pd.DataFrame] = []
    masks: list[pd.Series] = []
    cluster_series = component.frame["cluster_id"].astype(str)
    for source, target in zip(source_clusters, target_cluster_ids):
        indices = cluster_series[cluster_series == str(source)].index
        if len(indices) == 0:
            continue
        piece = component.frame.loc[indices].copy()
        piece["cluster_id"] = str(target)
        if "object" in piece.columns:
            piece["object"] = str(target)
        pieces.append(piece)
        mask = component.support_mask.loc[indices].copy()
        mask.index = piece.index
        masks.append(mask)
    if not pieces:
        raise RuntimeError("Joint cluster bootstrap produced no rows")
    frame = pd.concat(pieces, ignore_index=True)
    support_mask = pd.concat(masks, ignore_index=True).astype(bool)
    return PreparedComponent(
        record_id=component.record_id,
        control_family=component.control_family,
        frame=frame,
        support_mask=support_mask,
        center_offset=component.center_offset,
        baseline_contrast=component.baseline_contrast,
        native_scale=component.native_scale,
    )


def replace_component_cluster_effects(
    component: PreparedComponent,
    predicate: str,
    observed_residuals: Mapping[str, float],
    simulated_residuals: Mapping[str, float],
) -> PreparedComponent:
    frame = component.frame.copy()
    clusters = frame["cluster_id"].astype(str)
    delta = np.array(
        [
            float(simulated_residuals.get(cluster, 0.0))
            - float(observed_residuals.get(cluster, 0.0))
            for cluster in clusters
        ],
        dtype=float,
    )
    if predicate != "measurement_missingness_concentration":
        logits = pair_logit(frame) + delta
        frame = set_pair_logit(frame, logits)
    frame["_simulator_cluster_effect"] = delta
    return PreparedComponent(
        record_id=component.record_id,
        control_family=component.control_family,
        frame=frame,
        support_mask=component.support_mask.copy(),
        center_offset=component.center_offset,
        baseline_contrast=component.baseline_contrast,
        native_scale=component.native_scale,
    )


def simulate_components(
    address: PreparedAddress,
    simulator: SimulatorSpec,
    rng: np.random.Generator,
) -> list[PreparedComponent]:
    components = [address.target, *address.controls]
    cluster_ids = list(address.cluster_ids)
    if simulator.simulator_id == "joint_empirical_cluster_bootstrap":
        sources = rng.choice(cluster_ids, size=len(cluster_ids), replace=True).tolist()
        return [
            resample_component_by_cluster(component, sources, cluster_ids)
            for component in components
        ]

    residual_matrix = component_cluster_residuals(
        components,
        cluster_ids,
        address.failure_predicate,
    )
    if simulator.simulator_id == "joint_wild_cluster_rademacher":
        weights = rng.choice([-1.0, 1.0], size=len(cluster_ids))
        simulated = residual_matrix * weights[:, None]
    elif simulator.simulator_id == "joint_gaussian_regularized_cluster":
        covariance = regularized_covariance(residual_matrix)
        simulated = rng.multivariate_normal(
            mean=np.zeros(len(components)),
            cov=covariance,
            size=len(cluster_ids),
        )
    else:
        raise ValueError(f"Unknown simulator: {simulator.simulator_id}")

    transformed: list[PreparedComponent] = []
    for component_index, component in enumerate(components):
        observed_map = {
            cluster_id: float(residual_matrix[i, component_index])
            for i, cluster_id in enumerate(cluster_ids)
        }
        simulated_map = {
            cluster_id: float(simulated[i, component_index])
            for i, cluster_id in enumerate(cluster_ids)
        }
        transformed.append(
            replace_component_cluster_effects(
                component,
                address.failure_predicate,
                observed_map,
                simulated_map,
            )
        )
    return transformed


def inject_probability_failure(
    component: PreparedComponent,
    amount: float,
) -> PreparedComponent:
    frame = component.frame.copy()
    logits = pair_logit(frame)
    logits[component.support_mask.to_numpy(bool)] -= float(amount)
    frame = set_pair_logit(frame, logits)
    return PreparedComponent(
        record_id=component.record_id,
        control_family=component.control_family,
        frame=frame,
        support_mask=component.support_mask.copy(),
        center_offset=component.center_offset,
        baseline_contrast=component.baseline_contrast,
        native_scale=component.native_scale,
    )


def inject_missingness_failure(
    component: PreparedComponent,
    amount: float,
    rng: np.random.Generator,
) -> PreparedComponent:
    frame = component.frame.copy()
    if "predictor_missing_fraction" not in frame.columns:
        frame["predictor_missing_fraction"] = 0.0
    if "predictor_missing_any" not in frame.columns:
        frame["predictor_missing_any"] = False
    existing = pd.to_numeric(
        frame["predictor_missing_fraction"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    if amount <= 0:
        new_missing = np.zeros(len(frame), dtype=bool)
    else:
        cluster_effect = pd.to_numeric(
            frame.get("_simulator_cluster_effect", 0.0),
            errors="coerce",
        )
        if np.isscalar(cluster_effect):
            cluster_effect_array = np.zeros(len(frame), dtype=float)
        else:
            cluster_effect_array = np.asarray(cluster_effect, dtype=float)
        hazard = np.maximum(0.0, float(amount) * np.exp(np.clip(cluster_effect_array, -2, 2)))
        probability = 1.0 - np.exp(-hazard)
        probability[~component.support_mask.to_numpy(bool)] = 0.0
        new_missing = rng.random(len(frame)) < probability
    combined = np.maximum(existing, new_missing.astype(float))
    frame["predictor_missing_fraction"] = combined
    frame["predictor_missing_any"] = combined > 0
    return PreparedComponent(
        record_id=component.record_id,
        control_family=component.control_family,
        frame=frame,
        support_mask=component.support_mask.copy(),
        center_offset=component.center_offset,
        baseline_contrast=component.baseline_contrast,
        native_scale=component.native_scale,
    )


def apply_injection(
    address: PreparedAddress,
    components: Sequence[PreparedComponent],
    delta: float,
    control_response: float,
    rng: np.random.Generator,
) -> list[PreparedComponent]:
    injected: list[PreparedComponent] = []
    for index, component in enumerate(components):
        multiplier = 1.0 if index == 0 else float(control_response)
        amount = float(delta) * component.native_scale * multiplier
        if address.failure_predicate == "measurement_missingness_concentration":
            injected.append(inject_missingness_failure(component, amount, rng))
        else:
            injected.append(inject_probability_failure(component, amount))
    return injected


def identity_hash(frame: pd.DataFrame) -> str:
    columns = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    return canonical_frame_hash(frame[columns]) if columns else ""


def off_target_metric_changes(
    before: pd.DataFrame,
    after: pd.DataFrame,
    support_mask: pd.Series,
    target_metric: str,
) -> dict[str, float]:
    metrics = [
        "margin_loss",
        "log_loss",
        "misclassification_loss",
        "predictor_missing_fraction",
    ]
    result: dict[str, float] = {}
    for metric in metrics:
        if metric == target_metric or metric not in before.columns or metric not in after.columns:
            continue
        before_contrast = site_relative_contrast(before, support_mask, metric)
        after_contrast = site_relative_contrast(after, support_mask, metric)
        result[metric] = (
            float(after_contrast - before_contrast)
            if math.isfinite(before_contrast) and math.isfinite(after_contrast)
            else float("nan")
        )
    return result


def simulate_replicate(
    address: PreparedAddress,
    simulator: SimulatorSpec,
    contract: PredicateContract,
    delta: float,
    control_response: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    simulated = simulate_components(address, simulator, rng)
    before_target = simulated[0]
    injected = apply_injection(address, simulated, delta, control_response, rng)
    target = injected[0]
    controls = injected[1:]

    target_contrast = site_relative_contrast(
        target.frame,
        target.support_mask,
        contract.metric,
    )
    control_contrasts = [
        site_relative_contrast(component.frame, component.support_mask, contract.metric)
        for component in controls
    ]
    finite_controls = [value for value in control_contrasts if math.isfinite(value)]
    median_control = float(np.median(finite_controls)) if finite_controls else float("nan")
    adjusted = (
        float(target_contrast - median_control)
        if math.isfinite(target_contrast) and math.isfinite(median_control)
        else float("nan")
    )

    range_violations = sum(range_violation_count(component.frame) for component in injected)
    identity_violations = 0
    for before, after in zip(simulated, injected):
        if identity_hash(before.frame) != identity_hash(after.frame):
            identity_violations += 1
    cluster_count = len(
        set.intersection(
            *[
                set(component.frame["cluster_id"].astype(str).unique())
                for component in injected
            ]
        )
    )
    off_target = off_target_metric_changes(
        before_target.frame,
        target.frame,
        target.support_mask,
        contract.metric,
    )
    target_before = site_relative_contrast(
        before_target.frame,
        before_target.support_mask,
        contract.metric,
    )
    target_response = (
        float(target_contrast - target_before)
        if math.isfinite(target_contrast) and math.isfinite(target_before)
        else float("nan")
    )
    max_off_target = max(
        [abs(value) for value in off_target.values() if math.isfinite(value)],
        default=0.0,
    )

    return {
        "address_id": address.address_id,
        "record_id": address.record_id,
        "support_id": address.support_id,
        "failure_predicate": address.failure_predicate,
        "partition": address.partition,
        "simulator_id": simulator.simulator_id,
        "delta": float(delta),
        "control_response_lambda": float(control_response),
        "seed": int(seed),
        "target_contrast": target_contrast,
        "median_control_contrast": median_control,
        "control_adjusted_contrast": adjusted,
        "target_response_from_simulated_null": target_response,
        "control_contrasts_json": canonical_json(control_contrasts),
        "off_target_changes_json": canonical_json(off_target),
        "max_abs_off_target_change": max_off_target,
        "range_violation_count": int(range_violations),
        "identity_violation_count": int(identity_violations),
        "joint_cluster_count": int(cluster_count),
        "target_site_rows": int(target.support_mask.sum()),
        "target_complement_rows": int((~target.support_mask).sum()),
        "finite_target": bool(math.isfinite(target_contrast)),
        "finite_adjusted": bool(math.isfinite(adjusted)),
        "replicate_hash": stable_id(
            address.address_id,
            address.partition,
            simulator.simulator_id,
            delta,
            control_response,
            seed,
            target_contrast,
            control_contrasts,
            adjusted,
            range_violations,
            identity_violations,
            prefix="R-",
        ),
    }


def scenario_cells(
    effect_grid: Sequence[float],
    control_grid: Sequence[float],
) -> list[tuple[float, float]]:
    cells: list[tuple[float, float]] = []
    for delta in effect_grid:
        if delta == 0:
            cells.append((0.0, 0.0))
        else:
            cells.extend((float(delta), float(value)) for value in control_grid)
    return cells


def run_qualification_simulations(
    prepared: Sequence[PreparedAddress],
    specs: Sequence[SimulatorSpec],
    contracts: Mapping[str, PredicateContract],
    effect_grid: Sequence[float],
    control_grid: Sequence[float],
    replicates: int,
    master_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cells = scenario_cells(effect_grid, control_grid)
    for address_index, address in enumerate(prepared):
        contract = contracts[address.failure_predicate]
        for simulator_index, simulator in enumerate(specs):
            for cell_index, (delta, control_response) in enumerate(cells):
                for replicate in range(replicates):
                    seed = int(
                        np.random.SeedSequence(
                            [
                                master_seed,
                                address_index,
                                simulator_index,
                                cell_index,
                                replicate,
                            ]
                        ).generate_state(1)[0]
                    )
                    row = simulate_replicate(
                        address,
                        simulator,
                        contract,
                        delta,
                        control_response,
                        seed,
                    )
                    row["replicate_index"] = replicate
                    rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Qualification audits Q1-Q10
# -----------------------------------------------------------------------------


def null_preservation_audit(
    runs: pd.DataFrame,
    baseline_vectors: pd.DataFrame,
) -> pd.DataFrame:
    null = runs[runs["delta"] == 0].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["failure_predicate", "partition", "simulator_id", "address_id"]
    for keys, group in null.groupby(group_columns, sort=True):
        predicate, partition, simulator_id, address_id = keys
        baseline = baseline_vectors[
            (baseline_vectors["address_id"] == address_id)
            & (baseline_vectors["partition"] == partition)
            & (baseline_vectors["component_role"] == "target")
        ]
        scale = robust_scale(baseline["object_contrast"], floor=0.05)
        mean = float(pd.to_numeric(group["target_contrast"], errors="coerce").mean())
        standardized_bias = abs(mean) / scale if math.isfinite(mean) else float("inf")
        rows.append(
            {
                "failure_predicate": predicate,
                "partition": partition,
                "simulator_id": simulator_id,
                "address_id": address_id,
                "replicates": int(len(group)),
                "null_mean_target_contrast": mean,
                "null_sd_target_contrast": float(
                    pd.to_numeric(group["target_contrast"], errors="coerce").std(ddof=1)
                ),
                "baseline_object_scale": scale,
                "standardized_null_bias": standardized_bias,
                "finite_rate": float(group["finite_target"].mean()),
                "range_violations": int(group["range_violation_count"].sum()),
                "identity_violations": int(group["identity_violation_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def directionality_audit(runs: pd.DataFrame) -> pd.DataFrame:
    """Audit the injection operator on its direct target response.

    The noisy target/control contrast remains in the output as a downstream
    diagnostic, but Q2 is based on ``target_response_from_simulated_null``.
    This separates injection validity from sampling and matched-control noise.
    """
    primary = runs[runs["control_response_lambda"] == 0].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["failure_predicate", "partition", "simulator_id", "address_id"]
    for keys, group in primary.groupby(group_columns, sort=True):
        means = (
            group.groupby("delta", as_index=False)
            .agg(
                mean_target_contrast=("target_contrast", "mean"),
                mean_target_response=("target_response_from_simulated_null", "mean"),
                finite_rate=("finite_target", "mean"),
                finite_response_rate=(
                    "target_response_from_simulated_null",
                    lambda s: float(np.isfinite(pd.to_numeric(s, errors="coerce")).mean()),
                ),
            )
            .sort_values("delta")
        )
        deltas = means["delta"].to_numpy(float)
        contrast_values = means["mean_target_contrast"].to_numpy(float)
        response_values = means["mean_target_response"].to_numpy(float)

        contrast_adjacent = np.diff(contrast_values)
        response_adjacent = np.diff(response_values)
        contrast_monotonic = (
            float(np.mean(contrast_adjacent >= -1e-9))
            if len(contrast_adjacent)
            else float("nan")
        )
        response_monotonic = (
            float(np.mean(response_adjacent >= -1e-9))
            if len(response_adjacent)
            else float("nan")
        )
        contrast_endpoint = (
            float(contrast_values[-1] - contrast_values[0])
            if len(contrast_values)
            else float("nan")
        )
        response_endpoint = (
            float(response_values[-1] - response_values[0])
            if len(response_values)
            else float("nan")
        )
        response_rho = spearman_safe(deltas, response_values)

        rows.append(
            {
                "failure_predicate": keys[0],
                "partition": keys[1],
                "simulator_id": keys[2],
                "address_id": keys[3],
                "directionality_estimand": "target_response_from_simulated_null",
                "effect_levels": int(len(means)),
                "spearman_delta_target_response": response_rho,
                "monotonic_response_fraction": response_monotonic,
                "endpoint_target_response_change": response_endpoint,
                "minimum_finite_response_rate": float(
                    means["finite_response_rate"].min()
                ),
                "spearman_delta_target_contrast": spearman_safe(
                    deltas, contrast_values
                ),
                "monotonic_contrast_fraction": contrast_monotonic,
                "endpoint_target_contrast_change": contrast_endpoint,
                # Compatibility aliases now explicitly carry direct-response
                # semantics; new consumers should use the named fields above.
                "spearman_delta_target": response_rho,
                "monotonic_fraction": response_monotonic,
                "endpoint_target_change": response_endpoint,
                "minimum_finite_rate": float(means["finite_rate"].min()),
                "effect_curve_json": canonical_json(means.to_dict("records")),
            }
        )
    return pd.DataFrame(rows)


def specificity_audit(
    runs: pd.DataFrame,
    target_response_floor: float,
) -> pd.DataFrame:
    nonnull = runs[runs["delta"] > 0].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["failure_predicate", "partition", "simulator_id", "address_id"]
    for keys, group in nonnull.groupby(group_columns, sort=True):
        target = pd.to_numeric(
            group["target_response_from_simulated_null"], errors="coerce"
        )
        off_target = pd.to_numeric(group["max_abs_off_target_change"], errors="coerce")
        target_magnitude = float(np.nanmean(np.abs(target)))
        off_target_magnitude = float(np.nanmean(off_target))

        if not math.isfinite(target_magnitude) or not math.isfinite(off_target_magnitude):
            ratio = float("nan")
            ratio_status = "nonfinite_specificity_input"
        elif target_magnitude < target_response_floor:
            ratio = float("nan")
            ratio_status = "target_response_too_small_for_ratio"
        else:
            ratio = safe_ratio(off_target_magnitude, target_magnitude)
            ratio_status = "estimable"

        control_summary = (
            group.groupby("control_response_lambda", as_index=False)
            .agg(
                mean_control=("median_control_contrast", "mean"),
                mean_adjusted=("control_adjusted_contrast", "mean"),
            )
            .sort_values("control_response_lambda")
        )
        lambda_values = control_summary["control_response_lambda"].to_numpy(float)
        control_values = control_summary["mean_control"].to_numpy(float)
        adjusted_values = control_summary["mean_adjusted"].to_numpy(float)
        control_levels = int(len(np.unique(lambda_values[np.isfinite(lambda_values)])))

        if control_levels < 3:
            control_rho = float("nan")
            adjusted_rho = float("nan")
            control_status = "insufficient_control_response_levels"
        elif np.nanmax(control_values) - np.nanmin(control_values) <= 1e-12:
            control_rho = float("nan")
            adjusted_rho = spearman_safe(lambda_values, adjusted_values)
            control_status = "control_response_constant"
        else:
            control_rho = spearman_safe(lambda_values, control_values)
            adjusted_rho = spearman_safe(lambda_values, adjusted_values)
            control_status = (
                "estimable"
                if math.isfinite(control_rho)
                else "nonfinite_control_response"
            )

        rows.append(
            {
                "failure_predicate": keys[0],
                "partition": keys[1],
                "simulator_id": keys[2],
                "address_id": keys[3],
                "mean_abs_target_response": target_magnitude,
                "mean_max_abs_off_target_change": off_target_magnitude,
                "off_target_to_target_ratio": ratio,
                "specificity_ratio_status": ratio_status,
                "specificity_target_response_floor": float(target_response_floor),
                "control_response_levels": control_levels,
                "control_response_status": control_status,
                "control_response_rho": control_rho,
                "adjusted_response_rho": adjusted_rho,
                "identity_violations": int(group["identity_violation_count"].sum()),
                "range_violations": int(group["range_violation_count"].sum()),
                "control_curve_json": canonical_json(control_summary.to_dict("records")),
            }
        )
    return pd.DataFrame(rows)


def covariance_audit(
    runs: pd.DataFrame,
    baseline_vectors: pd.DataFrame,
    specs: Sequence[SimulatorSpec],
) -> pd.DataFrame:
    spec_map = {spec.simulator_id: spec for spec in specs}
    null = runs[runs["delta"] == 0].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["failure_predicate", "partition", "simulator_id", "address_id"]
    for keys, group in null.groupby(group_columns, sort=True):
        predicate, partition, simulator_id, address_id = keys
        baseline = baseline_vectors[
            (baseline_vectors["address_id"] == address_id)
            & (baseline_vectors["partition"] == partition)
        ]
        pivot = baseline.pivot_table(
            index="cluster_id",
            columns="component_role",
            values="object_contrast",
            aggfunc="mean",
        )
        baseline_corr = (
            pearson_safe(pivot.get("target", []), pivot.get("control", []))
            if {"target", "control"}.issubset(pivot.columns)
            else float("nan")
        )
        simulated_corr = pearson_safe(
            pd.to_numeric(group["target_contrast"], errors="coerce"),
            pd.to_numeric(group["median_control_contrast"], errors="coerce"),
        )
        baseline_estimable = bool(math.isfinite(baseline_corr))
        simulated_estimable = bool(math.isfinite(simulated_corr))
        covariance_estimable = baseline_estimable and simulated_estimable
        covariance_delta = (
            abs(simulated_corr - baseline_corr)
            if covariance_estimable
            else float("nan")
        )
        if covariance_estimable:
            covariance_status = "preservation_estimable"
        elif not baseline_estimable and not simulated_estimable:
            covariance_status = "baseline_and_simulated_correlation_unavailable"
        elif not baseline_estimable:
            covariance_status = "baseline_correlation_unavailable"
        else:
            covariance_status = "simulated_correlation_unavailable"

        rows.append(
            {
                "failure_predicate": predicate,
                "partition": partition,
                "simulator_id": simulator_id,
                "address_id": address_id,
                "baseline_target_control_correlation": baseline_corr,
                "simulated_target_control_correlation": simulated_corr,
                "absolute_correlation_change": covariance_delta,
                "covariance_estimable": covariance_estimable,
                "covariance_audit_status": covariance_status,
                "joint_target_control_declared": spec_map[
                    simulator_id
                ].joint_target_control,
                "covariance_model": spec_map[simulator_id].covariance_model,
                "joint_cluster_count_min": int(group["joint_cluster_count"].min()),
            }
        )
    return pd.DataFrame(rows)


def object_influence_audit(
    prepared: Sequence[PreparedAddress],
    specs: Sequence[SimulatorSpec],
    contracts: Mapping[str, PredicateContract],
    master_seed: int,
    replicates: int,
    max_delta: float,
) -> pd.DataFrame:
    """Evaluate finite LOO influence without conflating structural absence.

    A cluster omission can make the frozen estimator structurally unavailable
    before any simulator is invoked.  Such rows are retained as scope limits;
    only failures after a structurally estimable reduction are simulator
    execution failures.
    """
    rows: list[dict[str, Any]] = []
    loo_replicates = max(8, min(32, replicates // 4 if replicates >= 4 else 8))
    representatives: dict[tuple[str, str], PreparedAddress] = {}
    for address in prepared:
        representatives.setdefault((address.failure_predicate, address.partition), address)

    for (predicate, partition), address in sorted(representatives.items()):
        contract = contracts[predicate]
        for simulator_index, simulator in enumerate(specs):
            full_values: list[float] = []
            for replicate in range(loo_replicates):
                seed = int(
                    np.random.SeedSequence(
                        [master_seed, 9100, simulator_index, replicate, len(rows)]
                    ).generate_state(1)[0]
                )
                full_values.append(
                    simulate_replicate(
                        address,
                        simulator,
                        contract,
                        max_delta,
                        0.0,
                        seed,
                    )["target_contrast"]
                )
            full_mean = float(np.nanmean(full_values))
            full_scale = robust_scale(full_values, floor=0.05)
            for omitted in address.cluster_ids:
                def subset_component(component: PreparedComponent) -> PreparedComponent:
                    keep = component.frame["cluster_id"].astype(str) != str(omitted)
                    return PreparedComponent(
                        record_id=component.record_id,
                        control_family=component.control_family,
                        frame=component.frame.loc[keep].reset_index(drop=True),
                        support_mask=component.support_mask.loc[keep].reset_index(drop=True),
                        center_offset=component.center_offset,
                        baseline_contrast=component.baseline_contrast,
                        native_scale=component.native_scale,
                    )

                reduced = PreparedAddress(
                    address_id=address.address_id,
                    record_id=address.record_id,
                    support_id=address.support_id,
                    failure_predicate=address.failure_predicate,
                    relation=address.relation,
                    carrier=address.carrier,
                    support_definition=address.support_definition,
                    support_query_json=address.support_query_json,
                    entitlement_status=address.entitlement_status,
                    partition=address.partition,
                    target=subset_component(address.target),
                    controls=[subset_component(c) for c in address.controls],
                    cluster_ids=tuple(c for c in address.cluster_ids if c != omitted),
                    selection_reason=address.selection_reason,
                )

                structural_target = site_relative_contrast(
                    reduced.target.frame,
                    reduced.target.support_mask,
                    contract.metric,
                )
                structural_controls = [
                    site_relative_contrast(
                        component.frame,
                        component.support_mask,
                        contract.metric,
                    )
                    for component in reduced.controls
                ]
                finite_structural_controls = sum(
                    math.isfinite(value) for value in structural_controls
                )
                structural_estimable = bool(
                    math.isfinite(structural_target)
                    and finite_structural_controls >= 1
                    and len(reduced.cluster_ids) >= 2
                )

                values: list[float] = []
                simulation_error_count = 0
                if structural_estimable:
                    for replicate in range(loo_replicates):
                        seed = int(
                            np.random.SeedSequence(
                                [
                                    master_seed,
                                    9200,
                                    simulator_index,
                                    replicate,
                                    len(rows),
                                ]
                            ).generate_state(1)[0]
                        )
                        try:
                            values.append(
                                simulate_replicate(
                                    reduced,
                                    simulator,
                                    contract,
                                    max_delta,
                                    0.0,
                                    seed,
                                )["target_contrast"]
                            )
                        except Exception:
                            simulation_error_count += 1
                            values.append(float("nan"))

                finite_values = [value for value in values if math.isfinite(value)]
                if not structural_estimable:
                    loo_status = "structurally_unavailable"
                    loo_mean = float("nan")
                    standardized_change = float("nan")
                elif simulation_error_count > 0 or not finite_values:
                    loo_status = "simulation_failed"
                    loo_mean = (
                        float(np.mean(finite_values))
                        if finite_values
                        else float("nan")
                    )
                    standardized_change = float("nan")
                else:
                    loo_status = "estimable"
                    loo_mean = float(np.mean(finite_values))
                    standardized_change = abs(loo_mean - full_mean) / full_scale

                rows.append(
                    {
                        "failure_predicate": predicate,
                        "partition": partition,
                        "simulator_id": simulator.simulator_id,
                        "address_id": address.address_id,
                        "omitted_cluster": omitted,
                        "full_mean_target_contrast": full_mean,
                        "loo_mean_target_contrast": loo_mean,
                        "full_replicate_scale": full_scale,
                        "standardized_loo_change": standardized_change,
                        "loo_replicates": loo_replicates,
                        "loo_status": loo_status,
                        "structural_target_estimable": bool(
                            math.isfinite(structural_target)
                        ),
                        "structural_control_estimable_count": int(
                            finite_structural_controls
                        ),
                        "simulation_error_count": int(simulation_error_count),
                        "loo_successful": loo_status == "estimable",
                    }
                )
    return pd.DataFrame(rows)


def reproducibility_audit(
    prepared: Sequence[PreparedAddress],
    specs: Sequence[SimulatorSpec],
    contracts: Mapping[str, PredicateContract],
    master_seed: int,
    max_delta: float,
) -> pd.DataFrame:
    representatives: dict[tuple[str, str], PreparedAddress] = {}
    for address in prepared:
        representatives.setdefault((address.failure_predicate, address.partition), address)
    rows: list[dict[str, Any]] = []
    for address_index, ((predicate, partition), address) in enumerate(sorted(representatives.items())):
        for simulator_index, simulator in enumerate(specs):
            seed = int(
                np.random.SeedSequence(
                    [master_seed, 10000, address_index, simulator_index]
                ).generate_state(1)[0]
            )
            first = simulate_replicate(
                address,
                simulator,
                contracts[predicate],
                max_delta,
                0.5,
                seed,
            )
            second = simulate_replicate(
                address,
                simulator,
                contracts[predicate],
                max_delta,
                0.5,
                seed,
            )
            alternate = simulate_replicate(
                address,
                simulator,
                contracts[predicate],
                max_delta,
                0.5,
                seed + 1,
            )
            rows.append(
                {
                    "failure_predicate": predicate,
                    "partition": partition,
                    "simulator_id": simulator.simulator_id,
                    "address_id": address.address_id,
                    "seed": seed,
                    "first_hash": first["replicate_hash"],
                    "repeat_hash": second["replicate_hash"],
                    "alternate_seed_hash": alternate["replicate_hash"],
                    "same_seed_exact": first["replicate_hash"] == second["replicate_hash"],
                    "different_seed_varies": first["replicate_hash"] != alternate["replicate_hash"],
                }
            )
    return pd.DataFrame(rows)


def qualification_gate_matrix(
    null_audit: pd.DataFrame,
    directionality: pd.DataFrame,
    specificity: pd.DataFrame,
    covariance: pd.DataFrame,
    influence: pd.DataFrame,
    reproducibility: pd.DataFrame,
    specs: Sequence[SimulatorSpec],
    args: argparse.Namespace,
) -> pd.DataFrame:
    spec_map = {spec.simulator_id: spec for spec in specs}
    keys = ["failure_predicate", "partition", "simulator_id"]
    rows: list[dict[str, Any]] = []
    groups = null_audit[keys].drop_duplicates().sort_values(keys)
    for _, key_row in groups.iterrows():
        mask_null = np.logical_and.reduce(
            [null_audit[column] == key_row[column] for column in keys]
        )
        mask_direction = np.logical_and.reduce(
            [directionality[column] == key_row[column] for column in keys]
        )
        mask_specificity = np.logical_and.reduce(
            [specificity[column] == key_row[column] for column in keys]
        )
        mask_covariance = np.logical_and.reduce(
            [covariance[column] == key_row[column] for column in keys]
        )
        mask_influence = np.logical_and.reduce(
            [influence[column] == key_row[column] for column in keys]
        )
        mask_repro = np.logical_and.reduce(
            [reproducibility[column] == key_row[column] for column in keys]
        )
        n = null_audit[mask_null]
        d = directionality[mask_direction]
        s = specificity[mask_specificity]
        c = covariance[mask_covariance]
        i = influence[mask_influence]
        r = reproducibility[mask_repro]
        simulator = spec_map[str(key_row["simulator_id"])]

        q1 = bool(
            not n.empty
            and int(n["range_violations"].sum()) == 0
            and (s.empty or int(s["range_violations"].sum()) == 0)
        )
        q2 = bool(
            not d.empty
            and (
                d["spearman_delta_target_response"]
                >= args.directionality_min_rho
            ).all()
            and (d["endpoint_target_response_change"] > 0).all()
            and (
                d["monotonic_response_fraction"]
                >= args.monotonic_fraction_min
            ).all()
            and (
                d["minimum_finite_response_rate"] >= args.finite_rate_min
            ).all()
        )
        q3 = bool(
            not d.empty
            and (d["minimum_finite_rate"] >= args.finite_rate_min).all()
        )

        # Q4 qualifies joint target/control construction.  Covariance
        # preservation is reported separately and may be unscorable when the
        # four-cluster baseline correlation is undefined.
        q4 = bool(
            simulator.joint_target_control
            and not c.empty
            and c["joint_target_control_declared"].map(normalize_bool).all()
            and (c["joint_cluster_count_min"] >= 2).all()
        )
        covariance_estimable_count = (
            int(c["covariance_estimable"].map(normalize_bool).sum())
            if not c.empty
            else 0
        )
        if not q4:
            q4_status = "joint_construction_failed"
        elif covariance_estimable_count == len(c):
            q4_status = "joint_construction_pass_covariance_estimable"
        else:
            q4_status = "joint_construction_pass_covariance_unscorable"

        q5 = bool(not n.empty and (n["finite_rate"] >= args.finite_rate_min).all())
        q6 = bool(
            not n.empty
            and int(n["identity_violations"].sum()) == 0
            and (s.empty or int(s["identity_violations"].sum()) == 0)
        )
        q7 = q6 and q1
        q8 = bool(
            not n.empty
            and (n["standardized_null_bias"] <= args.null_bias_tolerance).all()
            and (n["finite_rate"] >= args.finite_rate_min).all()
        )

        loo_statuses = (
            i["loo_status"].astype(str)
            if not i.empty and "loo_status" in i.columns
            else pd.Series(dtype="string")
        )
        loo_structural_unavailable = int(
            (loo_statuses == "structurally_unavailable").sum()
        )
        loo_simulation_failed = int((loo_statuses == "simulation_failed").sum())
        loo_estimable = int((loo_statuses == "estimable").sum())
        if loo_simulation_failed > 0:
            q9 = False
            q9_status = "simulator_failure_after_estimable_reduction"
        elif loo_estimable == 0:
            q9 = False
            q9_status = "unscorable_no_estimable_loo_reduction"
        elif loo_structural_unavailable > 0:
            q9 = True
            q9_status = "pass_with_structural_loo_scope_limit"
        else:
            q9 = True
            q9_status = "pass"

        q10 = bool(
            not r.empty
            and r["same_seed_exact"].map(normalize_bool).all()
            and r["different_seed_varies"].map(normalize_bool).all()
        )

        finite_influence = (
            pd.to_numeric(i["standardized_loo_change"], errors="coerce")
            if not i.empty
            else pd.Series(dtype=float)
        )
        finite_influence = finite_influence[np.isfinite(finite_influence)]
        max_influence = (
            float(finite_influence.max())
            if not finite_influence.empty
            else float("nan")
        )
        estimable_specificity = (
            pd.to_numeric(
                s.loc[
                    s["specificity_ratio_status"] == "estimable",
                    "off_target_to_target_ratio",
                ],
                errors="coerce",
            )
            if not s.empty
            else pd.Series(dtype=float)
        )
        estimable_specificity = estimable_specificity[
            np.isfinite(estimable_specificity)
        ]
        specificity_ratio = (
            float(estimable_specificity.max())
            if not estimable_specificity.empty
            else float("nan")
        )

        scope_limits: list[str] = []
        if math.isfinite(max_influence) and max_influence > args.object_influence_limit:
            scope_limits.append("high_leave_one_object_out_influence")
        if loo_structural_unavailable > 0:
            scope_limits.append("structural_leave_one_object_out_unavailability")
        if math.isfinite(specificity_ratio) and specificity_ratio > args.specificity_ratio_max:
            scope_limits.append("substantial_coupled_off_target_metric_response")
        if not s.empty and (
            s["specificity_ratio_status"] != "estimable"
        ).any():
            scope_limits.append("specificity_ratio_unscorable_for_small_target_response")
        if not s.empty and (
            s["control_response_status"] != "estimable"
        ).any():
            scope_limits.append("control_response_curve_unscorable")
        if q4 and covariance_estimable_count < len(c):
            scope_limits.append("baseline_covariance_preservation_unscorable")
        if simulator.scope_limit:
            scope_limits.append(simulator.scope_limit)
        if str(key_row["failure_predicate"]) == "measurement_missingness_concentration":
            scope_limits.append(
                "missingness mechanism is simulator-defined because frozen baseline is sparse or zero"
            )

        gate_vector = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
        if args.smoke:
            status = "engineering_smoke_only"
        elif all(gate_vector):
            status = "qualified_with_scope_limit" if scope_limits else "qualified"
        else:
            status = "not_qualified"

        row = {
            **{column: key_row[column] for column in keys},
            **{
                f"q{index}_pass": value
                for index, value in enumerate(gate_vector, start=1)
            },
            "q2_estimand": "target_response_from_simulated_null",
            "q4_status": q4_status,
            "q9_status": q9_status,
            "qualification_decision_enabled": not args.smoke,
            "qualification_status": status,
            "scope_limits_json": canonical_json(sorted(set(scope_limits))),
            "max_standardized_object_influence": max_influence,
            "max_off_target_to_target_ratio": specificity_ratio,
            "loo_estimable_count": loo_estimable,
            "loo_structural_unavailable_count": loo_structural_unavailable,
            "loo_simulation_failed_count": loo_simulation_failed,
            "covariance_estimable_count": covariance_estimable_count,
            "failed_gates_json": canonical_json(
                [
                    f"Q{i}"
                    for i, passed in enumerate(gate_vector, start=1)
                    if not passed
                ]
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def simulator_family_summary(gates: pd.DataFrame) -> pd.DataFrame:
    if gates.empty:
        return pd.DataFrame()
    summary = (
        gates.groupby(["simulator_id", "failure_predicate"], as_index=False)
        .agg(
            partitions_evaluated=("partition", "nunique"),
            qualified_partitions=(
                "qualification_status",
                lambda s: int(
                    s.isin(["qualified", "qualified_with_scope_limit"]).sum()
                ),
            ),
            not_qualified_partitions=(
                "qualification_status",
                lambda s: int((s == "not_qualified").sum()),
            ),
            engineering_smoke_partitions=(
                "qualification_status",
                lambda s: int((s == "engineering_smoke_only").sum()),
            ),
            maximum_object_influence=("max_standardized_object_influence", "max"),
            maximum_specificity_ratio=("max_off_target_to_target_ratio", "max"),
        )
    )
    status_rows: list[str] = []
    for _, row in summary.iterrows():
        subset = gates[
            (gates["simulator_id"] == row["simulator_id"])
            & (gates["failure_predicate"] == row["failure_predicate"])
        ]
        statuses = set(subset["qualification_status"])
        if "not_qualified" in statuses:
            status_rows.append("not_qualified")
        elif "engineering_smoke_only" in statuses:
            status_rows.append("engineering_smoke_only")
        elif "qualified_with_scope_limit" in statuses:
            status_rows.append("qualified_with_scope_limit")
        elif "qualified" in statuses:
            status_rows.append("qualified")
        else:
            status_rows.append("unscorable")
    summary["qualification_status"] = status_rows
    return summary


# -----------------------------------------------------------------------------
# Output, manifest, and reporting
# -----------------------------------------------------------------------------


def build_input_manifest(paths: Mapping[str, Path], repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, path in paths.items():
        rows.append(
            {
                "artifact_role": role,
                "artifact_path": repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows).sort_values("artifact_role").reset_index(drop=True)


def scenario_manifest(
    effect_grid: Sequence[float],
    control_grid: Sequence[float],
    replicates: int,
    master_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for delta, control_response in scenario_cells(effect_grid, control_grid):
        scenario = {
            "delta": delta,
            "control_response_lambda": control_response,
            "replicates": replicates,
            "master_seed": master_seed,
            "qualification_only": True,
            "sensitivity_estimand_produced": False,
        }
        rows.append(
            {
                "scenario_id": stable_id(scenario, prefix="QS-"),
                **scenario,
                "scenario_semantics": (
                    "null estimator calibration"
                    if delta == 0
                    else "injection directionality and target-control response qualification"
                ),
            }
        )
    return pd.DataFrame(rows)


def output_paths(output_dir: Path) -> dict[str, Path]:
    names = [
        "obs085b0_input_manifest.csv",
        "obs085b0_injection_contracts.csv",
        "obs085b0_simulator_specs.csv",
        "obs085b0_scenario_manifest.csv",
        "obs085b0_qualification_address_manifest.csv",
        "obs085b0_baseline_object_vectors.csv",
        "obs085b0_qualification_runs.csv",
        "obs085b0_null_preservation_audit.csv",
        "obs085b0_directionality_audit.csv",
        "obs085b0_specificity_audit.csv",
        "obs085b0_covariance_audit.csv",
        "obs085b0_object_influence_audit.csv",
        "obs085b0_reproducibility_audit.csv",
        "obs085b0_qualification_gate_matrix.csv",
        "obs085b0_simulator_family_summary.csv",
        "obs085b0_failures.csv",
        "obs085b0_manifest.json",
        "obs085b0_report.md",
    ]
    return {name: output_dir / name for name in names}


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            existing = [p for p in path.iterdir()]
            if existing:
                raise FileExistsError(
                    f"Output directory is not empty: {path}. Use --overwrite."
                )
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_report(
    path: Path,
    lineage: Mapping[str, Any],
    universe_counts: Mapping[str, int],
    selected: pd.DataFrame,
    contract_table: pd.DataFrame,
    simulator_table: pd.DataFrame,
    scenarios: pd.DataFrame,
    null_audit: pd.DataFrame,
    directionality: pd.DataFrame,
    specificity: pd.DataFrame,
    covariance: pd.DataFrame,
    influence: pd.DataFrame,
    reproducibility: pd.DataFrame,
    gates: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    status_counts = (
        gates.groupby("qualification_status", as_index=False).size().rename(columns={"size": "count"})
        if not gates.empty
        else pd.DataFrame(columns=["qualification_status", "count"])
    )
    predicate_selection = (
        selected.groupby("failure_predicate", as_index=False)
        .agg(
            selected_addresses=("address_id", "nunique"),
            relations=("relation", "nunique"),
            carriers=("carrier", "nunique"),
            entitlement_classes=("entitlement_status", "nunique"),
            support_families=("support_family_label", "nunique"),
        )
    )
    gate_failures: list[dict[str, Any]] = []
    for gate in QUALIFICATION_GATE_COLUMNS:
        if gate in gates.columns:
            gate_failures.append(
                {
                    "qualification_gate": gate.replace("_pass", "").upper(),
                    "failed_predicate_partition_simulators": int((~gates[gate].map(normalize_bool)).sum()),
                    "passed_predicate_partition_simulators": int(gates[gate].map(normalize_bool).sum()),
                }
            )
    gate_failure_table = pd.DataFrame(gate_failures)

    lines: list[str] = []
    lines.append("# OBS-085b0 — Injection and Simulator Qualification")
    lines.append("")
    lines.append("## State")
    lines.append("")
    completed = bool(
        not args.smoke
        and not gates.empty
        and gates["qualification_status"].isin(
            ["qualified", "qualified_with_scope_limit"]
        ).any()
    )
    if args.smoke:
        lines.append("`engineering_smoke_completed`")
    else:
        lines.append(
            "`simulator_qualification_completed`"
            if completed
            else "`simulator_qualification_no_primary_instrument_qualified`"
        )
    lines.append("")
    if args.smoke:
        lines.append(
            "This engineering smoke run validates execution, schemas, range "
            "safety, identity preservation, and deterministic reproducibility. "
            "It issues no simulator qualification decision."
        )
    else:
        lines.append(
            "OBS-085b0 qualifies simulation machinery only. It does not "
            "estimate canonical simulated gate-passage probability, observed "
            "power, or a minimum detectable effect."
        )
    lines.append("")
    lines.append("## Frozen lineage")
    lines.append("")
    lines.append(f"- OBS-085a frozen commit anchor: `{lineage['expected_obs085a_commit']}`")
    lines.append(f"- Current repository HEAD: `{lineage['head_commit']}`")
    lines.append(f"- OBS-085a manifest SHA-256: `{lineage['obs085a_manifest_sha256']}`")
    lines.append(
        f"- OBS-085a manifest artifact hashes checked: **{len(lineage['manifest_hashes_checked']):,}**"
    )
    lines.append("")
    lines.append("## Qualification universe")
    lines.append("")
    lines.append(f"- Predicate-indexed address universe: **{universe_counts['address_universe']:,}**")
    lines.append(f"- Both-partition-feasible addresses: **{universe_counts['both_feasible_addresses']:,}**")
    lines.append(
        "- Both-partition-feasible record-scoped supports: "
        f"**{universe_counts['both_feasible_record_scoped_supports']:,}**"
    )
    lines.append(f"- Deterministically selected qualification addresses: **{selected['address_id'].nunique():,}**")
    lines.append("")
    lines.append(markdown_table(predicate_selection, args.max_report_rows))
    lines.append("")
    lines.append(
        "Selection maximized deterministic coverage of relation, carrier, claim "
        "entitlement, support family, support-prevalence stratum, and sealed-"
        "candidate context. No address was added or removed after inspecting "
        "simulation results."
    )
    lines.append("")
    lines.append("## Predicate-specific injection contracts")
    lines.append("")
    contract_display = contract_table[
        [
            "failure_predicate",
            "failure_mode",
            "metric",
            "lowest_valid_artifact_level",
            "mathematical_scale",
            "range_preserving_transform",
            "effect_scale_definition",
        ]
    ]
    lines.append(markdown_table(contract_display, args.max_report_rows))
    lines.append("")
    lines.append(
        "The three probability-loss predicates share a retained diagnostic "
        "probability-pair artifact, but each contract reruns its own frozen "
        "predicate estimator. Missingness uses a separate support-concentrated "
        "measurement-availability mechanism. No universal additive shift to a "
        "final contrast is used."
    )
    lines.append("")
    lines.append("## Simulator family")
    lines.append("")
    lines.append(markdown_table(simulator_table, args.max_report_rows))
    lines.append("")
    lines.append("## Qualification scenarios")
    lines.append("")
    lines.append(markdown_table(scenarios, args.max_report_rows))
    lines.append("")
    lines.append(
        "These cells exist to test null behavior, directionality, covariance, "
        "control response, and reproducibility. Their empirical success rates "
        "are not canonical sensitivity estimates."
    )
    lines.append("")
    lines.append(
        "## Q1–Q10 engineering diagnostics"
        if args.smoke
        else "## Q1–Q10 qualification results"
    )
    lines.append("")
    lines.append(markdown_table(status_counts, args.max_report_rows))
    lines.append("")
    lines.append(markdown_table(gate_failure_table, args.max_report_rows))
    lines.append("")
    lines.append(markdown_table(gates, args.max_report_rows))
    lines.append("")
    lines.append(
        "Q2 is evaluated on the direct target response from the same simulated "
        "null draw, not on the noisy target-control contrast. Q4 establishes "
        "joint target/control construction; covariance preservation is reported "
        "separately when estimable. Q9 treats deterministic leave-one-object-out "
        "estimator loss as an address-level scope limit rather than a simulator "
        "execution failure."
    )
    lines.append("")
    lines.append("## Simulator-family summary")
    lines.append("")
    lines.append(markdown_table(summary, args.max_report_rows))
    lines.append("")
    lines.append("## Null preservation")
    lines.append("")
    lines.append(markdown_table(null_audit, args.max_report_rows))
    lines.append("")
    lines.append("## Directionality and monotonic recoverability")
    lines.append("")
    lines.append(markdown_table(directionality, args.max_report_rows))
    lines.append("")
    lines.append("## Predicate and control-response specificity")
    lines.append("")
    lines.append(markdown_table(specificity, args.max_report_rows))
    lines.append("")
    lines.append("## Joint construction and covariance estimability audit")
    lines.append("")
    lines.append(markdown_table(covariance, args.max_report_rows))
    lines.append("")
    lines.append("## Object influence")
    lines.append("")
    lines.append(markdown_table(influence, args.max_report_rows))
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(markdown_table(reproducibility, args.max_report_rows))
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    lines.append(markdown_table(failures, args.max_report_rows))
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "> A qualified simulator is a validated experimental instrument under "
        "its frozen contract. Qualification is not evidence that the simulator "
        "is the true failure-generating mechanism."
    )
    lines.append("")
    lines.append(
        "> Monte Carlo precision is not instrument-model certainty. Sensitivity "
        "estimates remain conditional on the declared simulation-generating "
        "assumptions, particularly given the small number of observed "
        "independent object clusters."
    )
    lines.append("")
    if args.smoke:
        lines.append(
            "This smoke run does not alter the null FL3 result of OBS-084 and "
            "does not authorize OBS-085b execution for any predicate × "
            "simulator family."
        )
    else:
        lines.append(
            "OBS-085b0 does not alter the null FL3 result of OBS-084 and does "
            "not authorize OBS-085b execution for any predicate × simulator "
            "family marked `not_qualified`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    repo_root: Path,
    output_dir: Path,
    paths: Mapping[str, Path],
    lineage: Mapping[str, Any],
    universe_counts: Mapping[str, int],
    selected: pd.DataFrame,
    scenarios: pd.DataFrame,
    specs: Sequence[SimulatorSpec],
    gates: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_records: list[dict[str, Any]] = []
    for name, path in sorted(paths.items()):
        if name == "obs085b0_manifest.json" or not path.is_file():
            continue
        output_records.append(
            {
                "artifact_path": repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "state": (
            "engineering_smoke_completed"
            if args.smoke
            else "simulator_qualification_completed"
        ),
        "scope": "diagnostic-only instrument metrology",
        "claim_ceiling": (
            "engineering execution diagnostics only; no qualification decision"
            if args.smoke
            else "simulator qualification; no gate-passage probability"
        ),
        "frozen_lineage": dict(lineage),
        "universe_counts": dict(universe_counts),
        "qualification_address_ids": sorted(selected["address_id"].astype(str).unique()),
        "scenario_ids": sorted(scenarios["scenario_id"].astype(str).unique()),
        "simulator_ids": [spec.simulator_id for spec in specs],
        "qualification_decision_enabled": not args.smoke,
        "qualified_predicate_simulators": (
            []
            if args.smoke
            else gates.loc[
                gates["qualification_status"].isin(
                    ["qualified", "qualified_with_scope_limit"]
                ),
                [
                    "failure_predicate",
                    "partition",
                    "simulator_id",
                    "qualification_status",
                ],
            ].to_dict("records")
        ),
        "execution": {
            "master_seed": int(args.master_seed),
            "replicates_per_qualification_cell": int(args.replicates),
            "effect_grid": parse_number_grid(args.effect_grid, name="effect grid"),
            "control_response_grid": parse_number_grid(
                args.control_response_grid,
                name="control-response grid",
            ),
            "smoke": bool(args.smoke),
        },
        "output_artifacts": output_records,
        "mandatory_statement": (
            "Monte Carlo precision is not instrument-model certainty. "
            "Sensitivity estimates remain conditional on the declared "
            "simulation-generating assumptions, particularly given the small "
            "number of observed independent object clusters."
        ),
    }
    manifest_id = sha256_bytes(canonical_json(manifest_core).encode("utf-8"))
    return {"obs085b0_manifest_id": manifest_id, **manifest_core}


# -----------------------------------------------------------------------------
# Self test
# -----------------------------------------------------------------------------


def synthetic_component(record_id: str, seed: int = 1) -> PreparedComponent:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for cluster in ["o1", "o2", "o3", "o4"]:
        for support in [False, True]:
            for regime in ["C", "Cp"]:
                for replicate in range(3):
                    pair_share = np.clip(0.78 + rng.normal(0, 0.04), 0.55, 0.95)
                    pair_mass = 0.95
                    p = pair_mass * pair_share
                    q = pair_mass * (1 - pair_share)
                    margin = p - q
                    rows.append(
                        {
                            "case": regime,
                            "object": cluster,
                            "cluster_id": cluster,
                            "scale_index_from": 1,
                            "scale_index_to": 2,
                            "cohort": "inside" if support else "outside",
                            "observation_id": stable_id(cluster, support, regime, replicate),
                            "observation_key": stable_id("k", cluster, support, regime, replicate),
                            "transition": "1→2",
                            "transition_midpoint": 1.5,
                            "partition": "discovery",
                            "scale_band": "early" if support else "late",
                            "seam_relative_region": "near" if support else "far",
                            "record_id": record_id,
                            "relation": "C_vs_Cp",
                            "carrier": "toy",
                            "true_regime": regime,
                            "predicted_probability": p,
                            "max_other_probability": q,
                            "true_class_margin": margin,
                            "signed_margin": margin,
                            "correct": margin > 0,
                            "misclassification_loss": float(margin <= 0),
                            "margin_loss": -margin,
                            "log_loss": -math.log(p),
                            "predictor_missing_fraction": 0.0,
                            "predictor_missing_any": False,
                        }
                    )
    frame = pd.DataFrame(rows)
    mask = frame["scale_band"] == "early"
    offset, centered, contrast = find_center_offset(frame, mask, "log_loss")
    return PreparedComponent(
        record_id=record_id,
        control_family="target" if record_id == "target" else "relation_control",
        frame=centered,
        support_mask=mask.reset_index(drop=True),
        center_offset=offset,
        baseline_contrast=contrast,
        native_scale=0.5,
    )


def run_self_test() -> int:
    contracts = predicate_contracts()
    specs = simulator_specs()
    target = synthetic_component("target", 1)
    control = synthetic_component("control", 2)
    address = PreparedAddress(
        address_id="self-test-address",
        record_id="target",
        support_id="self-test-support",
        failure_predicate="log_loss_attenuation",
        relation="C_vs_Cp",
        carrier="toy",
        support_definition="scale_band=early",
        support_query_json='[{"column":"scale_band","operator":"eq","value":"early"}]',
        entitlement_status="fl3_entitled",
        partition="discovery",
        target=target,
        controls=[control],
        cluster_ids=("o1", "o2", "o3", "o4"),
        selection_reason="self-test",
    )
    for spec in specs:
        null = simulate_replicate(
            address,
            spec,
            contracts[address.failure_predicate],
            0.0,
            0.0,
            123,
        )
        injected = simulate_replicate(
            address,
            spec,
            contracts[address.failure_predicate],
            1.0,
            0.0,
            123,
        )
        repeat = simulate_replicate(
            address,
            spec,
            contracts[address.failure_predicate],
            1.0,
            0.0,
            123,
        )
        if injected["replicate_hash"] != repeat["replicate_hash"]:
            raise AssertionError(f"Reproducibility failed for {spec.simulator_id}")
        if injected["range_violation_count"] != 0:
            raise AssertionError(f"Range preservation failed for {spec.simulator_id}")
        if injected["identity_violation_count"] != 0:
            raise AssertionError(f"Identity preservation failed for {spec.simulator_id}")
        if not (injected["target_response_from_simulated_null"] > 0):
            raise AssertionError(f"Directionality failed for {spec.simulator_id}")
    print("OBS-085b0 self-test passed: injection direction, ranges, identity, and seeds")
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    args.repo_root = args.repo_root.resolve()
    if args.smoke:
        args.addresses_per_predicate = 1
        args.replicates = 8
        args.effect_grid = "0.00,0.50,1.00"
        args.control_response_grid = "0.00,1.00"

    if args.addresses_per_predicate < 1:
        raise ValueError("--addresses-per-predicate must be at least 1")
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2")
    if args.max_controls < 1:
        raise ValueError("--max-controls must be at least 1")
    if args.specificity_target_response_floor <= 0:
        raise ValueError("--specificity-target-response-floor must be positive")

    effect_grid = parse_number_grid(args.effect_grid, name="effect grid")
    control_grid = parse_number_grid(
        args.control_response_grid,
        name="control-response grid",
    )

    paths = input_paths(args)
    validate_required_inputs(paths)
    lineage = validate_obs085a_lineage(args)

    evidence = read_csv(
        paths["obs085a_evidence_feasibility"],
        dtype={"candidate_id": "string", "address_id": "string"},
    )
    inventory = read_csv(paths["obs085a_support_address_inventory"])
    control_availability = read_csv(paths["obs085a_control_availability"])
    discovery_observations = read_csv(paths["obs084b_discovery_observation_losses"])
    confirmation_observations = read_csv(paths["obs084c_confirmation_observation_losses"])

    require_columns(
        discovery_observations,
        ["record_id", "true_regime", "cluster_id", "predicted_probability", "max_other_probability"],
        "OBS-084b discovery observation losses",
    )
    require_columns(
        confirmation_observations,
        ["record_id", "true_regime", "cluster_id", "predicted_probability", "max_other_probability"],
        "OBS-084c confirmation observation losses",
    )

    universe = merge_address_inputs(evidence, inventory)
    universe_counts = validate_universe_counts(universe, args)
    selected = deterministic_address_selection(
        universe,
        args.addresses_per_predicate,
    )

    print("OBS-085b0 validation complete: frozen OBS-085a lineage valid")
    print(
        "Both-partition-feasible universe: "
        f"{universe_counts['both_feasible_addresses']:,} addresses"
    )
    print(
        "Qualification selection: "
        f"{selected['address_id'].nunique():,} addresses across "
        f"{selected['failure_predicate'].nunique():,} predicates"
    )
    if args.validate_only:
        return 0

    failures: list[QualificationFailure] = []
    contracts = predicate_contracts()
    specs = simulator_specs()
    prepared, baseline_vectors = prepare_selected_addresses(
        selected,
        control_availability,
        discovery_observations,
        confirmation_observations,
        contracts,
        args.max_controls,
        failures,
    )
    if not prepared:
        raise RuntimeError("No selected address could be prepared for qualification")

    runs = run_qualification_simulations(
        prepared,
        specs,
        contracts,
        effect_grid,
        control_grid,
        args.replicates,
        args.master_seed,
    )
    null_audit = null_preservation_audit(runs, baseline_vectors)
    directionality = directionality_audit(runs)
    specificity = specificity_audit(
        runs,
        args.specificity_target_response_floor,
    )
    covariance = covariance_audit(runs, baseline_vectors, specs)
    influence = object_influence_audit(
        prepared,
        specs,
        contracts,
        args.master_seed,
        args.replicates,
        max(effect_grid),
    )
    reproducibility = reproducibility_audit(
        prepared,
        specs,
        contracts,
        args.master_seed,
        max(effect_grid),
    )
    gates = qualification_gate_matrix(
        null_audit,
        directionality,
        specificity,
        covariance,
        influence,
        reproducibility,
        specs,
        args,
    )
    summary = simulator_family_summary(gates)

    if not args.smoke:
        for _, row in gates[
            gates["qualification_status"] == "not_qualified"
        ].iterrows():
            failures.append(
                QualificationFailure(
                    stage="simulator_qualification",
                    scope_id=(
                        f"{row['failure_predicate']}::{row['partition']}::"
                        f"{row['simulator_id']}"
                    ),
                    reason="required_qualification_gate_failed",
                    detail=str(row["failed_gates_json"]),
                    severity="error",
                )
            )

    failures_df = pd.DataFrame([asdict(failure) for failure in failures])
    if failures_df.empty:
        failures_df = pd.DataFrame(
            columns=["stage", "scope_id", "reason", "detail", "severity"]
        )

    contract_table = contract_frame(contracts)
    simulator_table = simulator_frame(specs)
    scenarios = scenario_manifest(
        effect_grid,
        control_grid,
        args.replicates,
        args.master_seed,
    )
    input_manifest = build_input_manifest(paths, args.repo_root)

    output_dir = repo_path(args.repo_root, args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    outputs = output_paths(output_dir)

    selection_columns = [
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
            "support_family_label",
            "support_prevalence",
            "selection_reason",
            "sealed_obs084b_candidate",
        ]
        if column in selected.columns
    ]
    frames: dict[str, pd.DataFrame] = {
        "obs085b0_input_manifest.csv": input_manifest,
        "obs085b0_injection_contracts.csv": contract_table,
        "obs085b0_simulator_specs.csv": simulator_table,
        "obs085b0_scenario_manifest.csv": scenarios,
        "obs085b0_qualification_address_manifest.csv": selected[selection_columns],
        "obs085b0_baseline_object_vectors.csv": baseline_vectors,
        "obs085b0_qualification_runs.csv": runs,
        "obs085b0_null_preservation_audit.csv": null_audit,
        "obs085b0_directionality_audit.csv": directionality,
        "obs085b0_specificity_audit.csv": specificity,
        "obs085b0_covariance_audit.csv": covariance,
        "obs085b0_object_influence_audit.csv": influence,
        "obs085b0_reproducibility_audit.csv": reproducibility,
        "obs085b0_qualification_gate_matrix.csv": gates,
        "obs085b0_simulator_family_summary.csv": summary,
        "obs085b0_failures.csv": failures_df,
    }
    for name, frame in frames.items():
        frame.to_csv(outputs[name], index=False)

    write_report(
        outputs["obs085b0_report.md"],
        lineage,
        universe_counts,
        selected,
        contract_table,
        simulator_table,
        scenarios,
        null_audit,
        directionality,
        specificity,
        covariance,
        influence,
        reproducibility,
        gates,
        summary,
        failures_df,
        args,
    )
    manifest = build_manifest(
        args.repo_root,
        output_dir,
        outputs,
        lineage,
        universe_counts,
        selected,
        scenarios,
        specs,
        gates,
        args,
    )
    outputs["obs085b0_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.smoke:
        passed_vectors = int(
            gates[QUALIFICATION_GATE_COLUMNS]
            .apply(lambda column: column.map(normalize_bool))
            .all(axis=1)
            .sum()
        )
        print(
            "Predicate × partition × simulator engineering diagnostics: "
            f"{passed_vectors:,} passed all Q1–Q10 checks; "
            "no qualification decisions issued"
        )
    else:
        qualified = int(
            gates["qualification_status"].isin(
                ["qualified", "qualified_with_scope_limit"]
            ).sum()
        )
        not_qualified = int(
            (gates["qualification_status"] == "not_qualified").sum()
        )
        print(
            "Predicate × partition × simulator qualifications: "
            f"{qualified:,} qualified/scope-limited; "
            f"{not_qualified:,} not qualified"
        )
    print(f"Qualification replicates written: {len(runs):,}")
    print(f"OBS-085b0 manifest ID: {manifest['obs085b0_manifest_id']}")
    print(f"Outputs: {repo_relative(output_dir, args.repo_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"OBS-085b0 failed: {exc}", file=sys.stderr)
        raise
