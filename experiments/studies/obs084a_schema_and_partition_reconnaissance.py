#!/usr/bin/env python3
"""
obs084a_schema_and_partition_reconnaissance.py

OBS-084a — Schema and Partition Reconnaissance
================================================

Purpose
-------
Audit whether the committed PAM/RIG artifacts through OBS-083 can support the
OBS-084 Direct Failure-Support Witness Protocol before any candidate discovery,
freeze, confirmation, witness assignment, or localization promotion occurs.

This script is deliberately reconnaissance-only. It:

* inventories relevant CSV/JSON/Markdown artifacts;
* records schemas, row counts, hashes, and read status;
* audits candidate join keys without performing analytical joins that create
  new scientific evidence;
* inventories plausible observation and structural-cluster units;
* evaluates deterministic discovery/confirmation/replication partition
  feasibility at the cluster level;
* audits availability of candidate support vocabularies;
* audits relation- and carrier-control feasibility from OBS-083 records;
* audits provenance, leakage-sensitive, and versioning fields;
* writes a conservative reconnaissance report.

It does NOT:

* train classifiers;
* compute out-of-fold losses;
* nominate or freeze failure-support candidates;
* inspect reserved confirmation outcomes;
* assign FL0–FL5 maturity;
* create direct witnesses;
* propose repairs or interventions;
* establish causality, control, actionability, external generalization, or
  formal topology.

Default inputs
--------------
The script searches the repository's ``outputs/`` tree and prioritizes known
OBS-078–083 and RIG-registry artifacts. Explicit paths can be supplied with
``--include``.

Default outputs
---------------
outputs/rig_registry/obs084_direct_failure_witness/reconnaissance/
    obs084a_input_manifest.csv
    obs084a_schema_inventory.csv
    obs084a_join_key_audit.csv
    obs084a_observation_unit_inventory.csv
    obs084a_candidate_support_availability.csv
    obs084a_partition_feasibility.csv
    obs084a_control_feasibility.csv
    obs084a_provenance_completeness.csv
    obs084a_leakage_field_audit.csv
    obs084a_reconnaissance_summary.csv
    obs084a_reconnaissance_report.md

Run
---
PYTHONPATH=src .venv/bin/python \
    experiments/studies/obs084a_schema_and_partition_reconnaissance.py

Scope guardrail
---------------
A positive feasibility result means only that the current artifact schemas
appear capable of supporting a future protocol step. It is not evidence that a
failure support exists, is direct, is causal, or is repairable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCRIPT_VERSION = "1.0.0"
PROTOCOL_NAME = "OBS-084 RIG Direct Failure-Support Witness Protocol"
RECON_SCOPE = "schema_and_partition_reconnaissance_only"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/reconnaissance"
)

# Artifact names are intentionally broad. The script still scans outputs/ for
# likely OBS-078–083 files because filenames may evolve while schemas remain
# scientifically relevant.
PRIORITY_FILENAME_PATTERNS: tuple[str, ...] = (
    "obs078",
    "obs079",
    "obs080",
    "obs081",
    "obs082",
    "obs083",
    "rig_relation_registry",
    "rig_survival_matrix",
    "rig_failure_localization",
    "rig_geometry_needed_ladder",
    "rig_repair_recommendations",
    "stability_core",
    "feature_table",
    "localization",
    "contract",
    "readiness",
    "negative_control",
)

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".md"}

# Candidate identity fields. Aliases are grouped by scientific role rather than
# treated as equivalent without qualification.
FIELD_FAMILIES: dict[str, tuple[str, ...]] = {
    "record_id": (
        "record_id",
        "registry_record_id",
        "rig_record_id",
    ),
    "relation": (
        "relation",
        "relation_id",
        "comparison",
        "comparison_id",
        "task",
        "pair",
    ),
    "carrier": (
        "carrier",
        "carrier_id",
        "carrier_name",
        "feature_set",
        "feature_subset",
        "feature_family_name",
    ),
    "observation_id": (
        "observation_id",
        "row_id",
        "sample_id",
        "case_id",
        "id",
    ),
    "object_id": (
        "object_id",
        "object",
        "object_key",
        "support_id",
        "case_object_id",
    ),
    "route_id": (
        "route_id",
        "route",
        "path_id",
        "path_key",
        "trajectory_id",
        "generator_path_id",
    ),
    "transition_id": (
        "transition_id",
        "transition",
        "transition_key",
        "event_id",
    ),
    "window_id": (
        "window_id",
        "window",
        "window_key",
        "window_index",
    ),
    "cohort": (
        "cohort",
        "cohort_id",
        "path_label_cohort",
        "transition_cohort",
        "status_cohort",
    ),
    "regime": (
        "regime",
        "condition",
        "corpus",
        "corpus_id",
        "class",
        "label",
        "true_regime",
    ),
    "scale": (
        "scale",
        "scale_id",
        "scale_band",
        "diffusion_scale",
        "t",
        "sigma",
    ),
    "feature_family": (
        "feature_family",
        "feature_group",
        "carrier_family",
        "feature_set",
        "projection_family",
    ),
    "contract": (
        "contract",
        "contract_id",
        "contract_family",
        "transformation",
        "transform",
        "transform_name",
        "evaluation_contract",
    ),
    "seam": (
        "distance_to_seam",
        "seam_distance",
        "seam_band",
        "seam_relative_region",
        "near_seam",
        "seam_contact",
    ),
    "boundary": (
        "distance_to_boundary",
        "boundary_distance",
        "boundary_band",
        "boundary_relative_region",
        "near_boundary",
    ),
    "provenance": (
        "provenance_id",
        "provenance",
        "source_artifact",
        "source_file",
        "source_path",
        "corpus_origin",
        "generation_id",
        "run_id",
        "campaign_id",
        "model_id",
        "model_name",
        "prompt_id",
        "preamble_id",
    ),
    "leakage_sensitive": (
        "label",
        "true_label",
        "true_regime",
        "target",
        "class",
        "condition",
        "corpus",
        "record_id",
        "relation",
        "carrier",
        "object_id",
        "route_id",
        "transition_id",
        "window_id",
    ),
}

# Support availability is a schema-level capability audit, not candidate
# generation. Each support family lists aliases whose presence suggests the
# address can potentially be indexed.
SUPPORT_FAMILIES: dict[str, tuple[str, ...]] = {
    "object": FIELD_FAMILIES["object_id"],
    "route_or_path": FIELD_FAMILIES["route_id"],
    "transition": FIELD_FAMILIES["transition_id"],
    "window": FIELD_FAMILIES["window_id"],
    "cohort": FIELD_FAMILIES["cohort"],
    "scale_band": FIELD_FAMILIES["scale"],
    "feature_family": FIELD_FAMILIES["feature_family"],
    "contract_or_transform": FIELD_FAMILIES["contract"],
    "seam_relative": FIELD_FAMILIES["seam"],
    "boundary_relative": FIELD_FAMILIES["boundary"],
    "provenance_slice": FIELD_FAMILIES["provenance"],
}

PARTITION_UNIT_PRIORITY: tuple[str, ...] = (
    "object_id",
    "route_id",
    "transition_id",
    "provenance",
    "observation_id",
)


@dataclass(frozen=True)
class Artifact:
    label: str
    path: Path
    suffix: str
    priority: bool
    size_bytes: int
    sha256: str
    read_status: str
    rows: int | None
    columns: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class PartitionAssessment:
    artifact_label: str
    artifact_path: str
    unit_family: str
    unit_column: str
    row_count: int
    non_null_rows: int
    unique_clusters: int
    singleton_clusters: int
    largest_cluster_rows: int
    median_cluster_rows: float
    regime_column: str
    regime_levels: int
    provenance_column: str
    provenance_levels: int
    feasible_two_way: bool
    feasible_three_way: bool
    recommended_design: str
    limitation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit OBS-078–083 artifact schemas and structural partition "
            "feasibility for OBS-084 without generating candidates or witnesses."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Outputs tree relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Reconnaissance output directory.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Explicit file or directory to include. May be repeated. Explicit "
            "files are inventoried even if their names do not match priority patterns."
        ),
    )
    parser.add_argument(
        "--max-csv-mb",
        type=float,
        default=512.0,
        help="Skip full reads of CSV files larger than this size. Default: 512 MB.",
    )
    parser.add_argument(
        "--min-two-way-clusters",
        type=int,
        default=12,
        help="Minimum independent clusters for discovery/confirmation feasibility.",
    )
    parser.add_argument(
        "--min-three-way-clusters",
        type=int,
        default=24,
        help="Minimum independent clusters for discovery/confirmation/replication feasibility.",
    )
    parser.add_argument(
        "--min-clusters-per-stratum",
        type=int,
        default=3,
        help="Minimum clusters per observed regime stratum for stratification feasibility.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=250_000,
        help=(
            "Maximum rows read per CSV for schema/count reconnaissance. Files with "
            "more rows are marked sampled. Set 0 to read all rows."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if no confirmation-feasible structural unit is found.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def normalize_column(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalized_column_map(columns: Iterable[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for col in columns:
        norm = normalize_column(col)
        result.setdefault(norm, str(col))
    return result


def match_columns(columns: Iterable[Any], aliases: Sequence[str]) -> list[str]:
    cmap = normalized_column_map(columns)
    matches: list[str] = []
    for alias in aliases:
        actual = cmap.get(normalize_column(alias))
        if actual is not None and actual not in matches:
            matches.append(actual)
    return matches


def first_match(columns: Iterable[Any], aliases: Sequence[str]) -> str:
    matches = match_columns(columns, aliases)
    return matches[0] if matches else ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def stable_short_hash(payload: Any, length: int = 16) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def is_priority_path(path: Path) -> bool:
    low = str(path).lower()
    return any(pattern in low for pattern in PRIORITY_FILENAME_PATTERNS)


def discover_paths(
    outputs_root: Path,
    includes: Sequence[Path],
    excluded_output_dir: Path,
) -> list[Path]:
    found: set[Path] = set()

    def add_path(path: Path, explicit: bool) -> None:
        if not path.exists():
            return
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                if explicit or is_priority_path(path):
                    found.add(path.resolve())
            return
        for child in path.rglob("*"):
            if not child.is_file() or child.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                child.resolve().relative_to(excluded_output_dir.resolve())
                continue
            except ValueError:
                pass
            if explicit or is_priority_path(child):
                found.add(child.resolve())

    add_path(outputs_root, explicit=False)
    for include in includes:
        add_path(include, explicit=True)

    return sorted(found, key=lambda p: str(p))


def safe_read_csv(
    path: Path,
    max_mb: float,
    sample_rows: int,
) -> tuple[pd.DataFrame | None, str, str]:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        try:
            header = pd.read_csv(path, nrows=0)
            return header, "header_only_size_limit", f"{size_mb:.1f} MB exceeds {max_mb:.1f} MB"
        except Exception as exc:  # pragma: no cover - defensive
            return None, "read_error", f"{type(exc).__name__}: {exc}"

    try:
        nrows = None if sample_rows <= 0 else sample_rows
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
        status = "ok" if nrows is None or len(df) < nrows else "sampled_row_limit"
        return df, status, ""
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty_csv", ""
    except Exception as exc:
        return None, "read_error", f"{type(exc).__name__}: {exc}"


def inventory_artifacts(
    paths: Sequence[Path],
    repo_root: Path,
    max_csv_mb: float,
    sample_rows: int,
) -> tuple[list[Artifact], dict[str, pd.DataFrame]]:
    artifacts: list[Artifact] = []
    frames: dict[str, pd.DataFrame] = {}
    label_counts: Counter[str] = Counter()

    for path in paths:
        base_label = normalize_column(path.stem) or "artifact"
        label_counts[base_label] += 1
        label = base_label
        if label_counts[base_label] > 1:
            label = f"{base_label}__{label_counts[base_label]}"

        suffix = path.suffix.lower()
        rows: int | None = None
        columns: tuple[str, ...] = ()
        status = "exists_non_tabular"
        error = ""

        if suffix == ".csv":
            df, status, error = safe_read_csv(path, max_csv_mb, sample_rows)
            if df is not None:
                rows = len(df)
                columns = tuple(str(c) for c in df.columns)
                frames[label] = df
        elif suffix in {".json", ".jsonl"}:
            try:
                if suffix == ".jsonl":
                    df = pd.read_json(path, lines=True)
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        df = pd.json_normalize(payload)
                    elif isinstance(payload, dict):
                        df = pd.json_normalize(payload)
                    else:
                        df = pd.DataFrame({"value": [payload]})
                rows = len(df)
                columns = tuple(str(c) for c in df.columns)
                frames[label] = df
                status = "ok"
            except Exception as exc:
                status = "read_error"
                error = f"{type(exc).__name__}: {exc}"

        try:
            rel = path.resolve().relative_to(repo_root.resolve())
            shown_path = rel
        except ValueError:
            shown_path = path.resolve()

        artifact = Artifact(
            label=label,
            path=shown_path,
            suffix=suffix,
            priority=is_priority_path(path),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            read_status=status,
            rows=rows,
            columns=columns,
            error=error,
        )
        artifacts.append(artifact)

    return artifacts, frames


def artifact_manifest(artifacts: Sequence[Artifact]) -> pd.DataFrame:
    rows = []
    for a in artifacts:
        rows.append(
            {
                "artifact_label": a.label,
                "artifact_path": str(a.path),
                "suffix": a.suffix,
                "priority_match": a.priority,
                "size_bytes": a.size_bytes,
                "size_mb": round(a.size_bytes / (1024 * 1024), 6),
                "sha256": a.sha256,
                "rows_read": a.rows,
                "column_count": len(a.columns),
                "read_status": a.read_status,
                "read_error": a.error,
            }
        )
    return pd.DataFrame(rows)


def build_schema_inventory(
    artifacts: Sequence[Artifact],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_by_label = {a.label: a for a in artifacts}

    for label, df in frames.items():
        artifact = artifact_by_label[label]
        for position, col in enumerate(df.columns):
            series = df[col]
            non_null = int(series.notna().sum())
            unique = int(series.nunique(dropna=True)) if len(series) else 0
            sample_values = []
            if non_null:
                sample_values = [str(v)[:120] for v in series.dropna().head(3).tolist()]
            role_matches = [
                family
                for family, aliases in FIELD_FAMILIES.items()
                if normalize_column(col) in {normalize_column(a) for a in aliases}
            ]
            rows.append(
                {
                    "artifact_label": label,
                    "artifact_path": str(artifact.path),
                    "column_position": position,
                    "column_name": str(col),
                    "normalized_column": normalize_column(col),
                    "dtype": str(series.dtype),
                    "rows_read": len(df),
                    "non_null_rows": non_null,
                    "non_null_share": (non_null / len(df)) if len(df) else math.nan,
                    "unique_non_null": unique,
                    "uniqueness_share": (unique / non_null) if non_null else math.nan,
                    "candidate_roles": "|".join(role_matches),
                    "sample_values": " | ".join(sample_values),
                }
            )
    return pd.DataFrame(rows)


def key_profile(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    if not columns:
        return {
            "rows": len(df),
            "complete_rows": 0,
            "complete_share": 0.0,
            "unique_keys": 0,
            "duplicate_complete_rows": 0,
            "unique_on_complete": False,
        }
    subset = df[list(columns)]
    complete_mask = subset.notna().all(axis=1)
    complete = subset.loc[complete_mask]
    unique_keys = int(complete.drop_duplicates().shape[0])
    duplicates = int(len(complete) - unique_keys)
    return {
        "rows": len(df),
        "complete_rows": int(complete_mask.sum()),
        "complete_share": float(complete_mask.mean()) if len(df) else 0.0,
        "unique_keys": unique_keys,
        "duplicate_complete_rows": duplicates,
        "unique_on_complete": bool(len(complete) > 0 and duplicates == 0),
    }


def build_join_key_audit(
    artifacts: Sequence[Artifact],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_by_label = {a.label: a for a in artifacts}

    candidate_key_specs: list[tuple[str, tuple[str, ...]]] = [
        ("record", ("record_id",)),
        ("relation_carrier", ("relation", "carrier")),
        ("observation", ("observation_id",)),
        ("object", ("object_id",)),
        ("route", ("route_id",)),
        ("transition", ("transition_id",)),
        ("window", ("window_id",)),
        ("object_transition", ("object_id", "transition_id")),
        ("route_transition", ("route_id", "transition_id")),
        ("object_route_transition", ("object_id", "route_id", "transition_id")),
        ("record_observation", ("record_id", "observation_id")),
        ("record_object_transition", ("record_id", "object_id", "transition_id")),
    ]

    for label, df in frames.items():
        artifact = artifact_by_label[label]
        resolved: dict[str, str] = {
            family: first_match(df.columns, FIELD_FAMILIES[family])
            for family in (
                "record_id",
                "relation",
                "carrier",
                "observation_id",
                "object_id",
                "route_id",
                "transition_id",
                "window_id",
            )
        }

        for key_name, families in candidate_key_specs:
            actual_cols = [resolved[f] for f in families if resolved.get(f)]
            all_present = len(actual_cols) == len(families)
            profile = key_profile(df, actual_cols if all_present else [])
            rows.append(
                {
                    "artifact_label": label,
                    "artifact_path": str(artifact.path),
                    "key_name": key_name,
                    "required_families": "|".join(families),
                    "resolved_columns": "|".join(actual_cols),
                    "all_required_columns_present": all_present,
                    **profile,
                    "join_use": classify_join_use(key_name, all_present, profile),
                    "caveat": join_caveat(key_name, all_present, profile),
                }
            )

    # Cross-artifact compatibility is schema-level only. It records shared key
    # families and does not perform joins or infer scientific equivalence.
    labels = list(frames)
    for i, left_label in enumerate(labels):
        left = frames[left_label]
        for right_label in labels[i + 1 :]:
            right = frames[right_label]
            shared_families: list[str] = []
            pairs: list[str] = []
            for family in (
                "record_id",
                "relation",
                "carrier",
                "observation_id",
                "object_id",
                "route_id",
                "transition_id",
                "window_id",
                "regime",
                "scale",
            ):
                lcol = first_match(left.columns, FIELD_FAMILIES[family])
                rcol = first_match(right.columns, FIELD_FAMILIES[family])
                if lcol and rcol:
                    shared_families.append(family)
                    pairs.append(f"{lcol}={rcol}")
            if shared_families:
                rows.append(
                    {
                        "artifact_label": f"{left_label} <-> {right_label}",
                        "artifact_path": "cross_artifact_schema_comparison",
                        "key_name": "cross_artifact_shared_families",
                        "required_families": "|".join(shared_families),
                        "resolved_columns": "|".join(pairs),
                        "all_required_columns_present": True,
                        "rows": math.nan,
                        "complete_rows": math.nan,
                        "complete_share": math.nan,
                        "unique_keys": math.nan,
                        "duplicate_complete_rows": math.nan,
                        "unique_on_complete": False,
                        "join_use": "candidate_schema_bridge_only",
                        "caveat": (
                            "Shared field names do not prove semantic key equivalence; "
                            "value-domain and provenance validation remain required."
                        ),
                    }
                )

    return pd.DataFrame(rows)


def classify_join_use(key_name: str, present: bool, profile: Mapping[str, Any]) -> str:
    if not present:
        return "unavailable"
    if profile["complete_rows"] == 0:
        return "present_but_empty"
    if profile["unique_on_complete"]:
        return "candidate_primary_or_one_side_key"
    if key_name in {"object", "route", "transition", "window"}:
        return "candidate_cluster_or_many_side_key"
    return "candidate_many_side_key_requires_cardinality_audit"


def join_caveat(key_name: str, present: bool, profile: Mapping[str, Any]) -> str:
    if not present:
        return "required field family not found"
    if profile["complete_share"] < 0.8:
        return "substantial missing key values"
    if not profile["unique_on_complete"]:
        return "duplicate keys require explicit one-to-many or many-to-many semantics"
    if key_name == "observation":
        return "uniqueness is artifact-local; cross-artifact namespace compatibility unproven"
    return "semantic equivalence and provenance still require validation"


def build_observation_unit_inventory(
    artifacts: Sequence[Artifact],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_by_label = {a.label: a for a in artifacts}

    unit_families = (
        "observation_id",
        "object_id",
        "route_id",
        "transition_id",
        "window_id",
        "cohort",
        "regime",
        "scale",
        "provenance",
    )

    for label, df in frames.items():
        artifact = artifact_by_label[label]
        for family in unit_families:
            columns = match_columns(df.columns, FIELD_FAMILIES[family])
            if not columns:
                rows.append(
                    {
                        "artifact_label": label,
                        "artifact_path": str(artifact.path),
                        "unit_family": family,
                        "column_name": "",
                        "available": False,
                        "rows_read": len(df),
                        "non_null_rows": 0,
                        "non_null_share": 0.0,
                        "unique_units": 0,
                        "median_rows_per_unit": math.nan,
                        "largest_unit_rows": 0,
                        "candidate_role": "unavailable",
                        "limitation": "field family not found",
                    }
                )
                continue

            for col in columns:
                series = df[col]
                valid = series.dropna()
                counts = valid.value_counts(dropna=True)
                unique = int(counts.size)
                median_rows = float(counts.median()) if unique else math.nan
                largest = int(counts.max()) if unique else 0
                role = unit_candidate_role(family, unique, len(df), largest)
                rows.append(
                    {
                        "artifact_label": label,
                        "artifact_path": str(artifact.path),
                        "unit_family": family,
                        "column_name": col,
                        "available": True,
                        "rows_read": len(df),
                        "non_null_rows": int(valid.size),
                        "non_null_share": float(valid.size / len(df)) if len(df) else 0.0,
                        "unique_units": unique,
                        "median_rows_per_unit": median_rows,
                        "largest_unit_rows": largest,
                        "candidate_role": role,
                        "limitation": unit_limitation(family, unique, len(df), largest),
                    }
                )

    return pd.DataFrame(rows)


def unit_candidate_role(family: str, unique: int, rows: int, largest: int) -> str:
    if unique == 0:
        return "unavailable"
    if family == "observation_id":
        return "candidate_observation_key" if unique == rows else "candidate_observation_group"
    if family in {"object_id", "route_id", "transition_id", "provenance"}:
        return "candidate_structural_partition_unit" if unique >= 2 else "insufficient_partition_unit"
    if family == "window_id":
        return "candidate_nested_unit_not_preferred_for_partition"
    if family in {"cohort", "regime", "scale"}:
        return "candidate_stratification_or_support_field"
    return "candidate_descriptor"


def unit_limitation(family: str, unique: int, rows: int, largest: int) -> str:
    if unique == 0:
        return "no valid units"
    if unique == 1:
        return "single observed level cannot support partitioning or comparison"
    if largest == rows and rows:
        return "all rows share one unit"
    if family == "window_id":
        return "overlapping windows may violate independence; keep parent route/transition together"
    if family == "observation_id" and unique < rows:
        return "observation identifiers repeat; namespace or nesting audit required"
    return ""


def cluster_partition_assessment(
    artifact: Artifact,
    df: pd.DataFrame,
    family: str,
    column: str,
    min_two: int,
    min_three: int,
    min_per_stratum: int,
) -> PartitionAssessment:
    valid = df[df[column].notna()].copy()
    counts = valid[column].value_counts(dropna=True)
    unique_clusters = int(counts.size)
    singleton = int((counts == 1).sum()) if unique_clusters else 0
    largest = int(counts.max()) if unique_clusters else 0
    median = float(counts.median()) if unique_clusters else math.nan

    regime_col = first_match(df.columns, FIELD_FAMILIES["regime"])
    provenance_col = first_match(df.columns, FIELD_FAMILIES["provenance"])
    regime_levels = int(valid[regime_col].nunique(dropna=True)) if regime_col else 0
    provenance_levels = int(valid[provenance_col].nunique(dropna=True)) if provenance_col else 0

    stratification_ok = True
    stratification_note = ""
    if regime_col and regime_levels > 1:
        cluster_regime = valid[[column, regime_col]].dropna().drop_duplicates()
        per_regime = cluster_regime.groupby(regime_col)[column].nunique()
        if not per_regime.empty and int(per_regime.min()) < min_per_stratum:
            stratification_ok = False
            stratification_note = (
                f"minimum {int(per_regime.min())} clusters in a regime stratum "
                f"is below required {min_per_stratum}"
            )

    feasible_two = unique_clusters >= min_two and stratification_ok
    feasible_three = unique_clusters >= min_three and stratification_ok

    if feasible_three:
        design = "candidate_discovery_confirmation_replication"
        limitation = ""
    elif feasible_two:
        design = "candidate_discovery_confirmation_plus_cluster_resampling"
        limitation = "insufficient clusters for a stable three-way partition"
    else:
        design = "not_confirmation_feasible"
        reasons = []
        if unique_clusters < min_two:
            reasons.append(f"only {unique_clusters} independent clusters; need at least {min_two}")
        if not stratification_ok:
            reasons.append(stratification_note)
        limitation = "; ".join(reasons) or "partition feasibility not established"

    return PartitionAssessment(
        artifact_label=artifact.label,
        artifact_path=str(artifact.path),
        unit_family=family,
        unit_column=column,
        row_count=len(df),
        non_null_rows=len(valid),
        unique_clusters=unique_clusters,
        singleton_clusters=singleton,
        largest_cluster_rows=largest,
        median_cluster_rows=median,
        regime_column=regime_col,
        regime_levels=regime_levels,
        provenance_column=provenance_col,
        provenance_levels=provenance_levels,
        feasible_two_way=feasible_two,
        feasible_three_way=feasible_three,
        recommended_design=design,
        limitation=limitation,
    )


def build_partition_feasibility(
    artifacts: Sequence[Artifact],
    frames: Mapping[str, pd.DataFrame],
    min_two: int,
    min_three: int,
    min_per_stratum: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_by_label = {a.label: a for a in artifacts}

    for label, df in frames.items():
        artifact = artifact_by_label[label]
        for family in PARTITION_UNIT_PRIORITY:
            for col in match_columns(df.columns, FIELD_FAMILIES[family]):
                result = cluster_partition_assessment(
                    artifact,
                    df,
                    family,
                    col,
                    min_two,
                    min_three,
                    min_per_stratum,
                )
                row = result.__dict__.copy()
                row["partition_priority"] = PARTITION_UNIT_PRIORITY.index(family) + 1
                row["independence_guardrail"] = partition_guardrail(family)
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "artifact_label",
                "artifact_path",
                "unit_family",
                "unit_column",
                "row_count",
                "non_null_rows",
                "unique_clusters",
                "singleton_clusters",
                "largest_cluster_rows",
                "median_cluster_rows",
                "regime_column",
                "regime_levels",
                "provenance_column",
                "provenance_levels",
                "feasible_two_way",
                "feasible_three_way",
                "recommended_design",
                "limitation",
                "partition_priority",
                "independence_guardrail",
            ]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["feasible_three_way", "feasible_two_way", "partition_priority", "unique_clusters"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return out


def partition_guardrail(family: str) -> str:
    if family == "object_id":
        return "keep all rows/windows/routes belonging to an object in one partition"
    if family == "route_id":
        return "keep overlapping windows and transitions from a route in one partition"
    if family == "transition_id":
        return "keep all windows belonging to a transition block in one partition"
    if family == "provenance":
        return "a provenance campaign may be a replication unit, not merely a row label"
    return "observation-level partitioning is last resort and does not establish independence"


def build_support_availability(
    artifacts: Sequence[Artifact],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_by_label = {a.label: a for a in artifacts}

    for support_family, aliases in SUPPORT_FAMILIES.items():
        available_artifacts = 0
        total_non_null = 0
        total_unique = 0
        artifact_details: list[str] = []

        for label, df in frames.items():
            cols = match_columns(df.columns, aliases)
            if not cols:
                continue
            available_artifacts += 1
            artifact = artifact_by_label[label]
            for col in cols:
                non_null = int(df[col].notna().sum())
                unique = int(df[col].nunique(dropna=True))
                total_non_null += non_null
                total_unique += unique
                artifact_details.append(f"{label}:{col}[n={non_null},u={unique}]")
                rows.append(
                    {
                        "support_family": support_family,
                        "artifact_label": label,
                        "artifact_path": str(artifact.path),
                        "column_name": col,
                        "available": True,
                        "non_null_rows": non_null,
                        "unique_values": unique,
                        "candidate_stage_ceiling": "schema_reconnaissance_only",
                        "support_use": support_use(support_family, unique),
                        "limitation": support_limitation(support_family, unique),
                    }
                )

        if available_artifacts == 0:
            rows.append(
                {
                    "support_family": support_family,
                    "artifact_label": "",
                    "artifact_path": "",
                    "column_name": "",
                    "available": False,
                    "non_null_rows": 0,
                    "unique_values": 0,
                    "candidate_stage_ceiling": "unavailable",
                    "support_use": "unavailable",
                    "limitation": "no matching field found in inventoried artifacts",
                }
            )

    return pd.DataFrame(rows)


def support_use(family: str, unique: int) -> str:
    if unique < 2:
        return "descriptor_only_insufficient_contrast"
    if family in {"object", "route_or_path", "transition"}:
        return "candidate_structural_support_and_partition_unit"
    if family == "window":
        return "candidate_nested_support_not_independent_partition"
    if family in {"cohort", "scale_band", "feature_family", "contract_or_transform"}:
        return "candidate_support_or_stratification_field"
    if family in {"seam_relative", "boundary_relative"}:
        return "candidate_geometry_relative_support_requires_density_matching"
    if family == "provenance_slice":
        return "candidate_scope_or_replication_support"
    return "candidate_support_field"


def support_limitation(family: str, unique: int) -> str:
    if unique == 0:
        return "field contains no non-null values"
    if unique == 1:
        return "single level cannot define support versus complement"
    if family == "window":
        return "windows may overlap; parent route or transition must remain grouped"
    if family in {"seam_relative", "boundary_relative"}:
        return "generic geometric difficulty and exposure density must be controlled"
    if family == "provenance_slice":
        return "provenance-local findings remain scope-local and are not external replication"
    return ""


def infer_record_table(frames: Mapping[str, pd.DataFrame]) -> tuple[str, pd.DataFrame] | tuple[str, None]:
    preferred = sorted(
        frames.items(),
        key=lambda item: (
            "diagnostic_subclass_assignments" not in item[0],
            "relation_registry" not in item[0],
            "readiness_scores" not in item[0],
            item[0],
        ),
    )
    for label, df in preferred:
        record_col = first_match(df.columns, FIELD_FAMILIES["record_id"])
        relation_col = first_match(df.columns, FIELD_FAMILIES["relation"])
        carrier_col = first_match(df.columns, FIELD_FAMILIES["carrier"])
        if record_col or (relation_col and carrier_col):
            if len(df) >= 2:
                return label, df
    return "", None


def derive_record_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    record_col = first_match(out.columns, FIELD_FAMILIES["record_id"])
    relation_col = first_match(out.columns, FIELD_FAMILIES["relation"])
    carrier_col = first_match(out.columns, FIELD_FAMILIES["carrier"])

    if record_col:
        out["__record_id"] = out[record_col].astype("string")
    elif relation_col and carrier_col:
        out["__record_id"] = (
            out[relation_col].astype("string") + "__" + out[carrier_col].astype("string")
        )
    else:
        out["__record_id"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if relation_col:
        out["__relation"] = out[relation_col].astype("string")
    else:
        out["__relation"] = out["__record_id"].str.split("__", n=1).str[0]

    if carrier_col:
        out["__carrier"] = out[carrier_col].astype("string")
    else:
        out["__carrier"] = out["__record_id"].str.split("__", n=1).str[1]

    subclass_col = first_match(
        out.columns,
        (
            "subclass",
            "diagnostic_subclass",
            "readiness_subclass",
            "readiness_class",
        ),
    )
    out["__subclass"] = out[subclass_col].astype("string") if subclass_col else "unknown"
    return out


def build_control_feasibility(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    label, raw = infer_record_table(frames)
    if raw is None:
        return pd.DataFrame(
            [
                {
                    "source_artifact": "",
                    "record_id": "",
                    "relation": "",
                    "carrier": "",
                    "subclass": "unknown",
                    "confirmation_eligibility": "unknown",
                    "relation_control_count": 0,
                    "carrier_control_count": 0,
                    "total_candidate_controls": 0,
                    "control_feasibility": "unavailable",
                    "limitation": "no registry-like relation × carrier table found",
                }
            ]
        )

    records = derive_record_fields(raw)
    records = records.dropna(subset=["__record_id", "__relation", "__carrier"])
    records = records.drop_duplicates("__record_id")

    rows: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        record_id = str(row["__record_id"])
        relation = str(row["__relation"])
        carrier = str(row["__carrier"])
        subclass = str(row["__subclass"])

        relation_controls = records[
            (records["__carrier"] == carrier) & (records["__relation"] != relation)
        ]
        carrier_controls = records[
            (records["__relation"] == relation) & (records["__carrier"] != carrier)
        ]
        total = len(relation_controls) + len(carrier_controls)

        c2 = "c2" in subclass.lower() or "localization" in subclass.lower()
        eligibility = "fl3_confirmation_eligible" if c2 else "discovery_only_or_unknown"
        if len(relation_controls) and len(carrier_controls):
            feasibility = "relation_and_carrier_controls_available"
            limitation = "control admissibility and balance still require observation-level evidence"
        elif total:
            feasibility = "single_control_family_available"
            limitation = "multi-family control robustness unavailable from registry structure"
        else:
            feasibility = "no_matched_record_controls"
            limitation = "cannot estimate record-specific site difficulty from registry structure alone"

        rows.append(
            {
                "source_artifact": label,
                "record_id": record_id,
                "relation": relation,
                "carrier": carrier,
                "subclass": subclass,
                "confirmation_eligibility": eligibility,
                "relation_control_count": len(relation_controls),
                "relation_control_record_ids": "|".join(
                    sorted(relation_controls["__record_id"].astype(str).tolist())
                ),
                "carrier_control_count": len(carrier_controls),
                "carrier_control_record_ids": "|".join(
                    sorted(carrier_controls["__record_id"].astype(str).tolist())
                ),
                "total_candidate_controls": total,
                "control_feasibility": feasibility,
                "limitation": limitation,
            }
        )

    return pd.DataFrame(rows)


def build_provenance_completeness(
    artifacts: Sequence[Artifact],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_by_label = {a.label: a for a in artifacts}

    provenance_roles = {
        "source_pointer": ("source_artifact", "source_file", "source_path", "artifact_path"),
        "run_or_campaign": ("run_id", "campaign_id", "generation_id", "experiment_id"),
        "model": ("model_id", "model_name", "model", "checkpoint"),
        "corpus_origin": ("corpus_origin", "corpus", "condition", "preamble_id", "prompt_id"),
        "contract": FIELD_FAMILIES["contract"],
        "record": FIELD_FAMILIES["record_id"],
        "observation": FIELD_FAMILIES["observation_id"],
    }

    for label, df in frames.items():
        artifact = artifact_by_label[label]
        present_roles: list[str] = []
        missing_roles: list[str] = []
        resolved: list[str] = []
        complete_scores: list[float] = []

        for role, aliases in provenance_roles.items():
            col = first_match(df.columns, aliases)
            if col:
                present_roles.append(role)
                resolved.append(f"{role}:{col}")
                complete_scores.append(float(df[col].notna().mean()) if len(df) else 0.0)
            else:
                missing_roles.append(role)

        core_roles = {"source_pointer", "run_or_campaign", "model", "corpus_origin"}
        core_present = len(core_roles.intersection(present_roles))
        if core_present >= 3:
            status = "strong_schema_provenance"
        elif core_present >= 1:
            status = "partial_schema_provenance"
        else:
            status = "artifact_hash_only_provenance"

        rows.append(
            {
                "artifact_label": label,
                "artifact_path": str(artifact.path),
                "artifact_sha256": artifact.sha256,
                "rows_read": len(df),
                "present_provenance_roles": "|".join(present_roles),
                "resolved_columns": "|".join(resolved),
                "missing_provenance_roles": "|".join(missing_roles),
                "mean_present_field_completeness": (
                    sum(complete_scores) / len(complete_scores) if complete_scores else 0.0
                ),
                "provenance_status": status,
                "artifact_hash_available": bool(artifact.sha256),
                "versioning_readiness": (
                    "candidate_source_manifest_ready"
                    if artifact.sha256
                    else "source_hash_missing"
                ),
                "scope_guardrail": (
                    "schema-level provenance only; scientific lineage equivalence remains to be audited"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_leakage_field_audit(
    artifacts: Sequence[Artifact],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_by_label = {a.label: a for a in artifacts}
    sensitive_norms = {normalize_column(v) for v in FIELD_FAMILIES["leakage_sensitive"]}

    for label, df in frames.items():
        artifact = artifact_by_label[label]
        for col in df.columns:
            norm = normalize_column(col)
            if norm not in sensitive_norms and not any(
                token in norm for token in ("label", "target", "condition", "corpus", "record_id")
            ):
                continue
            series = df[col]
            rows.append(
                {
                    "artifact_label": label,
                    "artifact_path": str(artifact.path),
                    "column_name": str(col),
                    "normalized_column": norm,
                    "non_null_share": float(series.notna().mean()) if len(df) else 0.0,
                    "unique_values": int(series.nunique(dropna=True)),
                    "leakage_risk_role": leakage_role(norm),
                    "required_handling": leakage_handling(norm),
                    "guardrail": "presence does not imply leakage; future feature matrices must explicitly exclude/audit it",
                }
            )
    return pd.DataFrame(rows)


def leakage_role(norm: str) -> str:
    if norm in {"label", "true_label", "true_regime", "target", "class", "condition", "corpus"}:
        return "target_or_regime_identity"
    if norm in {"record_id", "relation", "carrier"}:
        return "registry_identity"
    if norm.endswith("_id"):
        return "structural_identity"
    return "potential_identity_or_target_proxy"


def leakage_handling(norm: str) -> str:
    if norm in {"label", "true_label", "true_regime", "target", "class", "condition", "corpus"}:
        return "retain as outcome/stratification only; exclude from predictive carriers"
    if norm in {"record_id", "relation", "carrier"}:
        return "retain as metadata only; never use as observation-level predictive feature"
    return "retain for grouping/matching only unless a specific leakage-safe use is predeclared"


def build_summary(
    manifest: pd.DataFrame,
    schema: pd.DataFrame,
    joins: pd.DataFrame,
    units: pd.DataFrame,
    supports: pd.DataFrame,
    partitions: pd.DataFrame,
    controls: pd.DataFrame,
    provenance: pd.DataFrame,
) -> pd.DataFrame:
    tabular_ok = int(manifest["read_status"].isin(["ok", "sampled_row_limit"]).sum()) if not manifest.empty else 0
    readable_artifacts = int((manifest["read_status"] != "read_error").sum()) if not manifest.empty else 0
    feasible_two = int(partitions["feasible_two_way"].fillna(False).sum()) if not partitions.empty else 0
    feasible_three = int(partitions["feasible_three_way"].fillna(False).sum()) if not partitions.empty else 0
    available_supports = int(
        supports.loc[supports["available"].fillna(False), "support_family"].nunique()
    ) if not supports.empty else 0
    total_supports = len(SUPPORT_FAMILIES)
    fl3_eligible = int(
        controls["confirmation_eligibility"].eq("fl3_confirmation_eligible").sum()
    ) if not controls.empty and "confirmation_eligibility" in controls else 0
    control_ready = int(
        controls["control_feasibility"].eq("relation_and_carrier_controls_available").sum()
    ) if not controls.empty and "control_feasibility" in controls else 0
    provenance_strong = int(
        provenance["provenance_status"].eq("strong_schema_provenance").sum()
    ) if not provenance.empty else 0

    blockers: list[str] = []
    if readable_artifacts == 0:
        blockers.append("no readable priority artifacts")
    if feasible_two == 0:
        blockers.append("no structurally independent discovery/confirmation partition found")
    if available_supports < 3:
        blockers.append("candidate support vocabulary is too sparse")
    if fl3_eligible == 0:
        blockers.append("no C2/localization-limited confirmation-eligible records identified")
    if control_ready == 0:
        blockers.append("no records with both relation and carrier control families")

    if blockers:
        overall = "not_ready_for_candidate_freeze"
    elif feasible_three:
        overall = "schema_ready_for_obs084a_discovery_design_with_three_way_partition_candidate"
    else:
        overall = "schema_ready_for_obs084a_discovery_design_with_two_way_partition_candidate"

    rows = [
        {"metric": "script_version", "value": SCRIPT_VERSION, "interpretation": "reconnaissance implementation version"},
        {"metric": "artifacts_inventoried", "value": len(manifest), "interpretation": "priority and explicitly included artifacts"},
        {"metric": "readable_artifacts", "value": readable_artifacts, "interpretation": "includes non-tabular artifacts"},
        {"metric": "tabular_artifacts_read", "value": tabular_ok, "interpretation": "CSV/JSON frames available for schema audit"},
        {"metric": "schema_columns_inventoried", "value": len(schema), "interpretation": "column-level schema rows"},
        {"metric": "candidate_join_key_rows", "value": len(joins), "interpretation": "artifact-local and cross-artifact schema checks"},
        {"metric": "observation_unit_rows", "value": len(units), "interpretation": "unit-family availability checks"},
        {"metric": "available_support_families", "value": available_supports, "interpretation": f"out of {total_supports} predeclared support families"},
        {"metric": "two_way_partition_candidates", "value": feasible_two, "interpretation": "schema-level discovery/confirmation feasibility rows"},
        {"metric": "three_way_partition_candidates", "value": feasible_three, "interpretation": "schema-level discovery/confirmation/replication feasibility rows"},
        {"metric": "fl3_confirmation_eligible_records", "value": fl3_eligible, "interpretation": "C2/localization-limited records inferred from registry table"},
        {"metric": "records_with_both_control_families", "value": control_ready, "interpretation": "registry-level relation and carrier controls; observation-level admissibility unproven"},
        {"metric": "strong_schema_provenance_artifacts", "value": provenance_strong, "interpretation": "artifact schemas with at least three core provenance roles"},
        {"metric": "overall_reconnaissance_status", "value": overall, "interpretation": "; ".join(blockers) if blockers else "no fatal schema-level blocker found"},
    ]
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def write_report(
    path: Path,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    joins: pd.DataFrame,
    units: pd.DataFrame,
    supports: pd.DataFrame,
    partitions: pd.DataFrame,
    controls: pd.DataFrame,
    provenance: pd.DataFrame,
    leakage: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    status_row = summary.loc[summary["metric"] == "overall_reconnaissance_status"]
    overall = status_row.iloc[0]["value"] if not status_row.empty else "unknown"
    overall_note = status_row.iloc[0]["interpretation"] if not status_row.empty else ""

    manifest_view = manifest[
        [
            "artifact_label",
            "artifact_path",
            "size_mb",
            "rows_read",
            "column_count",
            "read_status",
        ]
    ] if not manifest.empty else pd.DataFrame()

    partition_view = partitions.head(20)[
        [
            "artifact_label",
            "unit_family",
            "unit_column",
            "unique_clusters",
            "feasible_two_way",
            "feasible_three_way",
            "recommended_design",
            "limitation",
        ]
    ] if not partitions.empty else pd.DataFrame()

    support_summary = (
        supports.groupby("support_family", dropna=False)
        .agg(
            available=("available", "max"),
            artifact_count=("artifact_label", lambda s: int((s.astype(str) != "").sum())),
            max_unique_values=("unique_values", "max"),
        )
        .reset_index()
        if not supports.empty
        else pd.DataFrame()
    )

    control_view = controls.head(30)[
        [
            "record_id",
            "subclass",
            "confirmation_eligibility",
            "relation_control_count",
            "carrier_control_count",
            "control_feasibility",
        ]
    ] if not controls.empty and "record_id" in controls else pd.DataFrame()

    report = f"""# OBS-084a — Schema and Partition Reconnaissance

## State

Reconnaissance completed.

Overall status: `{overall}`

Interpretation: {overall_note or 'No additional status note.'}

This is a schema and partition audit only. It performs no candidate generation,
confirmation, witness assignment, FL promotion, repair design, intervention, or
causal analysis.

## Protocol alignment

This script operationalizes the pre-discovery reconnaissance step beneath
**{PROTOCOL_NAME}**.

Canonical guardrails:

> Directness is artifact-direct, not metaphysically direct and not causally direct.

> Localization is not atomization.

> A site is direct only through its witness.

> Discovery nominates a support; reserved evidence earns the localization claim.

A positive feasibility result means only that the artifact schemas appear able
to support a later protocol step. It is not evidence that a failure support
exists.

## Configuration

| setting | value |
|---|---|
| script_version | `{SCRIPT_VERSION}` |
| repo_root | `{args.repo_root}` |
| outputs_root | `{args.outputs_root}` |
| output_dir | `{args.output_dir}` |
| max_csv_mb | {args.max_csv_mb} |
| sample_rows | {args.sample_rows} |
| min_two_way_clusters | {args.min_two_way_clusters} |
| min_three_way_clusters | {args.min_three_way_clusters} |
| min_clusters_per_stratum | {args.min_clusters_per_stratum} |

## Input artifacts

{df_to_markdown(manifest_view, index=False) if not manifest_view.empty else 'No priority artifacts were discovered.'}

## Observation and structural units

- Unit inventory rows: {len(units)}
- Candidate partition assessment rows: {len(partitions)}
- Two-way feasible rows: {int(partitions['feasible_two_way'].fillna(False).sum()) if not partitions.empty else 0}
- Three-way feasible rows: {int(partitions['feasible_three_way'].fillna(False).sum()) if not partitions.empty else 0}

Partition feasibility is schema-level and cluster-count-based. It does not prove
that candidate supports will have adequate matched complements or balanced
reserved evidence.

{df_to_markdown(partition_view, index=False) if not partition_view.empty else 'No candidate structural partition units were found.'}

## Candidate support vocabulary availability

{df_to_markdown(support_summary, index=False) if not support_summary.empty else 'No candidate support fields were found.'}

Support availability means that an address family can potentially be indexed.
It does not nominate a support or establish degradation.

## Join-key audit

- Audit rows: {len(joins)}
- Artifact-local keys marked candidate primary/one-side: {int(joins['join_use'].eq('candidate_primary_or_one_side_key').sum()) if not joins.empty else 0}
- Cross-artifact schema bridges: {int(joins['join_use'].eq('candidate_schema_bridge_only').sum()) if not joins.empty else 0}

Shared field names do not prove semantic key equivalence. Future analytical
joins must audit value domains, cardinality, namespace, and provenance.

## Control feasibility

{df_to_markdown(control_view, index=False) if not control_view.empty else 'No registry-like record table was available for control reconnaissance.'}

Registry-level relation and carrier controls are only candidate control sets.
Observation-level support overlap, baseline balance, contract exposure, and
failure-mode comparability remain unproven.

## Provenance and versioning readiness

- Artifacts audited: {len(provenance)}
- Strong schema-provenance artifacts: {int(provenance['provenance_status'].eq('strong_schema_provenance').sum()) if not provenance.empty else 0}
- Artifact hashes available: {int(provenance['artifact_hash_available'].fillna(False).sum()) if not provenance.empty else 0}

Artifact hashes support a future frozen source manifest. They do not by
themselves establish scientific lineage equivalence.

## Leakage-sensitive fields

- Fields flagged for explicit handling: {len(leakage)}

Identity, regime, label, record, and structural-unit fields may be required for
outcomes, grouping, matching, or provenance. Their presence does not imply
leakage. Future predictive carriers must explicitly exclude or audit them.

## Reconnaissance decision rules

The repository may proceed to OBS-084a candidate-discovery implementation only
when all of the following are supported:

1. at least one defensible structural partition unit exists;
2. discovery and confirmation can be separated at the cluster level;
3. C2/localization-limited records can be identified;
4. relation and/or carrier control sets can be constructed;
5. candidate support fields are addressable;
6. provenance and source hashes can be frozen;
7. leakage-sensitive identity fields can be separated from predictive carriers.

A three-way discovery/confirmation/replication split is preferred only when the
number and balance of independent clusters support it. Otherwise the protocol
should use a two-way split plus dependence-aware structural resampling.

## Outputs

- `obs084a_input_manifest.csv`
- `obs084a_schema_inventory.csv`
- `obs084a_join_key_audit.csv`
- `obs084a_observation_unit_inventory.csv`
- `obs084a_candidate_support_availability.csv`
- `obs084a_partition_feasibility.csv`
- `obs084a_control_feasibility.csv`
- `obs084a_provenance_completeness.csv`
- `obs084a_leakage_field_audit.csv`
- `obs084a_reconnaissance_summary.csv`
- `obs084a_reconnaissance_report.md`

## Limitations

- Filename discovery prioritizes OBS-078–083 and RIG-related artifacts; use
  `--include` for additional sources.
- CSV files may be sampled for reconnaissance. Sampled results must not be used
  as confirmation evidence.
- Schema aliases are heuristics. They do not prove semantic equivalence.
- Cluster counts do not prove statistical independence.
- The script does not inspect hidden or uncommitted artifacts.
- The script does not create or unlock a reserved confirmation partition.
- No FL maturity level is assigned.

## Canonical result statement

OBS-084a reconnaissance audits whether the committed PAM/RIG artifacts contain
sufficient schema, join-key, structural-unit, support-vocabulary, control, and
provenance infrastructure to design a frozen discovery/confirmation study. It
produces feasibility evidence only and establishes no direct failure support,
causal origin, repair target, actionability, external generalization, or formal
topology.
"""
    path.write_text(report, encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    outputs_root = resolve_path(repo_root, args.outputs_root).resolve()
    output_dir = resolve_path(repo_root, args.output_dir).resolve()
    includes = [resolve_path(repo_root, Path(p)).resolve() for p in args.include]

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_paths(outputs_root, includes, output_dir)
    artifacts, frames = inventory_artifacts(
        paths,
        repo_root,
        args.max_csv_mb,
        args.sample_rows,
    )

    manifest = artifact_manifest(artifacts)
    schema = build_schema_inventory(artifacts, frames)
    joins = build_join_key_audit(artifacts, frames)
    units = build_observation_unit_inventory(artifacts, frames)
    supports = build_support_availability(artifacts, frames)
    partitions = build_partition_feasibility(
        artifacts,
        frames,
        args.min_two_way_clusters,
        args.min_three_way_clusters,
        args.min_clusters_per_stratum,
    )
    controls = build_control_feasibility(frames)
    provenance = build_provenance_completeness(artifacts, frames)
    leakage = build_leakage_field_audit(artifacts, frames)
    summary = build_summary(
        manifest,
        schema,
        joins,
        units,
        supports,
        partitions,
        controls,
        provenance,
    )

    outputs = {
        "obs084a_input_manifest.csv": manifest,
        "obs084a_schema_inventory.csv": schema,
        "obs084a_join_key_audit.csv": joins,
        "obs084a_observation_unit_inventory.csv": units,
        "obs084a_candidate_support_availability.csv": supports,
        "obs084a_partition_feasibility.csv": partitions,
        "obs084a_control_feasibility.csv": controls,
        "obs084a_provenance_completeness.csv": provenance,
        "obs084a_leakage_field_audit.csv": leakage,
        "obs084a_reconnaissance_summary.csv": summary,
    }
    for filename, df in outputs.items():
        write_csv(df, output_dir / filename)

    write_report(
        output_dir / "obs084a_reconnaissance_report.md",
        args,
        manifest,
        joins,
        units,
        supports,
        partitions,
        controls,
        provenance,
        leakage,
        summary,
    )

    overall_rows = summary.loc[summary["metric"] == "overall_reconnaissance_status", "value"]
    overall = str(overall_rows.iloc[0]) if not overall_rows.empty else "unknown"
    print(f"OBS-084a reconnaissance complete: {overall}")
    print(f"Artifacts inventoried: {len(manifest)}")
    print(f"Tabular frames read: {len(frames)}")
    print(
        "Two-way / three-way partition candidates: "
        f"{int(partitions['feasible_two_way'].fillna(False).sum()) if not partitions.empty else 0} / "
        f"{int(partitions['feasible_three_way'].fillna(False).sum()) if not partitions.empty else 0}"
    )
    print(f"Outputs: {output_dir}")

    if args.strict and overall == "not_ready_for_candidate_freeze":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
