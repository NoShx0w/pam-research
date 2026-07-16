#!/usr/bin/env python3
"""
obs084b_direct_failure_support_discovery.py

OBS-084b — Direct Failure-Support Discovery
============================================

Repository-stage note
---------------------
The repository-local OBS-084 sequence uses:

* OBS-084a: canonical lineage resolution and discovery/confirmation freeze;
* OBS-084b: blinded candidate discovery on the frozen discovery partition;
* a later reserved-confirmation stage: confirmation and possible FL3 assignment.

This script therefore performs the protocol's FL2 candidate-discovery role. It
never unlocks or evaluates reserved confirmation outcomes.

Purpose
-------
Search the frozen discovery partition for artifact-indexed, predicate-specific
failure-support candidates for the 24 RIG relation × carrier records.

The script:

* refuses to run unless OBS-084a is ``frozen_ready_for_discovery``;
* verifies the frozen package and frozen source hashes;
* reconstructs the six frozen scientific carriers exactly;
* excludes ``row_bootstrap_unit`` and other non-scientific bookkeeping fields;
* fits the diagnostic classifier only inside the discovery partition;
* uses leave-one-object-out predictions within the discovery objects;
* derives discovery-only scale and seam bins under frozen rules;
* searches only the frozen support vocabulary, with conjunction depth <= 2;
* constructs class-balanced site/complement contrasts;
* audits matched complement overlap and exposure balance;
* computes relation- and carrier-control-adjusted contrasts;
* performs object-cluster leave-one-out, cluster bootstrap, and within-cluster
  stratified permutation diagnostics;
* reports multiplicity over the complete tested denominator;
* seals only non-dominated FL2 candidates into a deterministic manifest.

This script does NOT:

* read confirmation outcomes into an analytical frame;
* assign FL3, FL4, or FL5;
* describe any candidate as confirmed, direct, causal, actionable, or repaired;
* perform interventions or repair tests;
* establish external generalization or formal topology.

Default inputs
--------------
outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution/
    obs084a_freeze_manifest.json
    obs084a_reviewed_observation_key_spec.csv
    obs084a_reviewed_carrier_feature_manifest.csv
    obs084a_reviewed_field_roles.csv
    obs084a_cluster_unit_selection.csv
    obs084a_two_way_partition_manifest.csv
    obs084a_partition_balance_final.csv
    obs084a_seam_discretization_protocol.csv
    obs084a_support_vocabulary_freeze.csv
    obs084a_source_hash_manifest.csv

outputs/rig_registry/obs083_negative_control_localization/
    obs083_diagnostic_subclass_assignments.csv
    obs083_relation_control_contrast.csv
    obs083_carrier_control_contrast.csv

outputs/rig_registry/rig_relation_registry.csv

Default outputs
---------------
outputs/rig_registry/obs084_direct_failure_witness/discovery/
    obs084b_input_manifest.csv
    obs084b_discovery_observation_losses.csv
    obs084b_support_thresholds.csv
    obs084b_support_candidate_inventory.csv
    obs084b_support_complement_matching.csv
    obs084b_site_relative_contrasts.csv
    obs084b_control_adjusted_contrasts.csv
    obs084b_cluster_uncertainty.csv
    obs084b_minimal_support_families.csv
    obs084b_multiplicity_audit.csv
    obs084b_candidate_freeze_manifest.csv
    obs084b_candidate_freeze_manifest.json
    obs084b_discovery_failures.csv
    obs084b_discovery_summary.csv
    obs084b_discovery_report.md

Run
---
PYTHONPATH=src .venv/bin/python \
  experiments/studies/obs084b_direct_failure_support_discovery.py

Canonical guardrail
-------------------
A positive OBS-084b result means only that one or more artifact-indexed FL2
candidate supports were nominated on the frozen discovery partition and sealed
for later reserved confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "1.0.0"
MODEL_RANDOM_STATE = 84002

DEFAULT_FREEZE_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/bridge_resolution"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/discovery"
)
DEFAULT_OBS083_DIR = Path(
    "outputs/rig_registry/obs083_negative_control_localization"
)
DEFAULT_REGISTRY_PATH = Path("outputs/rig_registry/rig_relation_registry.csv")

FREEZE_FILES = {
    "observation_key": "obs084a_reviewed_observation_key_spec.csv",
    "carrier_features": "obs084a_reviewed_carrier_feature_manifest.csv",
    "field_roles": "obs084a_reviewed_field_roles.csv",
    "cluster_unit": "obs084a_cluster_unit_selection.csv",
    "partition": "obs084a_two_way_partition_manifest.csv",
    "partition_balance": "obs084a_partition_balance_final.csv",
    "seam_protocol": "obs084a_seam_discretization_protocol.csv",
    "support_vocabulary": "obs084a_support_vocabulary_freeze.csv",
    "source_hashes": "obs084a_source_hash_manifest.csv",
}

OBS083_FILES = {
    "subclasses": "obs083_diagnostic_subclass_assignments.csv",
    "relation_controls": "obs083_relation_control_contrast.csv",
    "carrier_controls": "obs083_carrier_control_contrast.csv",
}

SCIENTIFIC_OBSERVATION_KEY = (
    "case",
    "object",
    "cohort",
    "scale_index_from",
    "scale_index_to",
)
ALIGNMENT_KEY = ("case", "object", "cohort", "candidate_rank")
FORBIDDEN_SCIENTIFIC_FEATURES = {"row_bootstrap_unit"}

PREDICATES: tuple[dict[str, Any], ...] = (
    {
        "failure_predicate": "relation_separation_attenuation",
        "failure_mode": "attenuation",
        "metric": "margin_loss",
        "expected_direction": "site_greater_than_complement",
        "minimum_effect": 0.10,
        "threshold_basis": "predeclared minimum true-class-margin attenuation",
    },
    {
        "failure_predicate": "local_criterion_breach",
        "failure_mode": "threshold_breach",
        "metric": "misclassification_loss",
        "expected_direction": "site_greater_than_complement",
        "minimum_effect": 0.10,
        "threshold_basis": "registry threshold crossed locally while complement remains above threshold",
    },
    {
        "failure_predicate": "log_loss_attenuation",
        "failure_mode": "attenuation",
        "metric": "log_loss",
        "expected_direction": "site_greater_than_complement",
        "minimum_effect": 0.10,
        "threshold_basis": "predeclared minimum class-balanced log-loss increase",
    },
    {
        "failure_predicate": "measurement_missingness_concentration",
        "failure_mode": "missingness_concentration",
        "metric": "predictor_missing_any",
        "expected_direction": "site_greater_than_complement",
        "minimum_effect": 0.10,
        "threshold_basis": "predeclared minimum concentration of undefined carrier measurements",
    },
)

SUPPORT_FAMILY_COLUMNS = {
    "object": ("object",),
    "cohort": ("cohort",),
    "transition": ("transition",),
    "scale_band": ("scale_band",),
    "seam_relative": ("seam_relative_region",),
    "route_or_path": ("dominant_family",),
    # The following frozen families are audited but are not automatically used
    # unless a discrete observation-varying field is available.
    "window": ("window", "window_id", "window_band"),
    "provenance_slice": (
        "provenance_slice",
        "provenance_id",
        "campaign",
        "corpus_origin",
        "generation_campaign",
    ),
}

NON_SITE_SUPPORT_FAMILIES = {
    "contract_or_transform",
    "feature_family",
}


@dataclass(frozen=True)
class DiscoveryFailure:
    stage: str
    record_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class SupportDefinition:
    support_id: str
    depth: int
    families: tuple[str, ...]
    columns: tuple[str, ...]
    values: tuple[str, ...]
    support_query_json: str
    support_definition: str


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    p.add_argument("--obs083-dir", type=Path, default=DEFAULT_OBS083_DIR)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default="logreg", choices=["logreg"])
    p.add_argument("--min-site-rows", type=int, default=8)
    p.add_argument("--min-complement-rows", type=int, default=12)
    p.add_argument("--min-class-rows", type=int, default=2)
    p.add_argument("--min-site-clusters", type=int, default=2)
    p.add_argument("--min-complement-clusters", type=int, default=2)
    p.add_argument("--min-shared-clusters", type=int, default=2)
    p.add_argument("--max-conjunction-depth", type=int, default=2)
    p.add_argument("--max-supports-per-record", type=int, default=250)
    p.add_argument("--max-resampled-tests-per-record-predicate", type=int, default=8,
                   help="Outcome-blind deterministic cap for expensive cluster bootstrap/permutation diagnostics within each record × predicate. Non-resampled tests remain in the multiplicity denominator with p=1.")
    p.add_argument("--n-cluster-bootstrap", type=int, default=300)
    p.add_argument("--n-permutations", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--discovery-fdr", type=float, default=0.20)
    p.add_argument("--min-direction-consistency", type=float, default=0.75)
    p.add_argument("--min-control-adjusted-effect", type=float, default=0.05)
    p.add_argument("--min-positive-control-share", type=float, default=0.50)
    p.add_argument("--minimality-tolerance", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    p.add_argument(
        "--require-repo-commit",
        action="store_true",
        help="Also require the current Git commit to equal the OBS-084a frozen commit. Source hashes are always required.",
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


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "ok"}


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_No rows._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def canonical_observation_key(df: pd.DataFrame) -> pd.Series:
    require_columns(df, SCIENTIFIC_OBSERVATION_KEY, "canonical feature table")
    return df[list(SCIENTIFIC_OBSERVATION_KEY)].astype(str).agg("|".join, axis=1)


def transition_label(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["scale_index_from"], errors="coerce").astype("Int64").astype(str)
        + "→"
        + pd.to_numeric(df["scale_index_to"], errors="coerce").astype("Int64").astype(str)
    )


def parse_relation_classes(relation: str) -> tuple[str, ...]:
    relation = str(relation)
    if relation == "three_way":
        return ("C", "Cp2", "Cp3")
    if "_vs_" in relation:
        a, b = relation.split("_vs_", 1)
        return (a, b)
    raise ValueError(f"Unsupported relation: {relation}")


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().clip(0.0, 1.0)
    if valid.empty:
        return out
    ordered = valid.sort_values()
    m = len(ordered)
    raw = ordered.to_numpy(dtype=float) * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out.loc[ordered.index] = adjusted
    return out


def quantile_ci(values: Sequence[float], alpha: float) -> tuple[float, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def rng_for(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**32 - 1))


# -----------------------------------------------------------------------------
# Freeze validation
# -----------------------------------------------------------------------------


def load_and_validate_freeze(
    repo_root: Path,
    freeze_dir: Path,
    require_repo_commit: bool,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    manifest_path = freeze_dir / "obs084a_freeze_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_ready_for_discovery":
        raise RuntimeError(
            "OBS-084b requires status=frozen_ready_for_discovery; "
            f"found {payload.get('status')!r}"
        )

    freeze_tables: dict[str, pd.DataFrame] = {}
    artifact_hashes = payload.get("artifact_hashes", {})
    for role, name in FREEZE_FILES.items():
        path = freeze_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        expected = artifact_hashes.get(name)
        actual = sha256_file(path)
        if expected and actual != expected:
            raise RuntimeError(
                f"Frozen artifact hash mismatch for {name}: expected {expected}, got {actual}"
            )
        freeze_tables[role] = read_csv_required(path)

    source_manifest = freeze_tables["source_hashes"].copy()
    require_columns(
        source_manifest,
        ["artifact_path", "sha256", "exists"],
        "OBS-084a source hash manifest",
    )
    source_rows: list[dict[str, Any]] = []
    for _, row in source_manifest.iterrows():
        raw = str(row["artifact_path"])
        path = resolve_path(repo_root, raw)
        expected = str(row.get("sha256", ""))
        exists = path.exists() and path.is_file()
        actual = sha256_file(path) if exists else ""
        valid = exists and bool(expected) and actual == expected
        source_rows.append(
            {
                "source_role": str(row.get("source_role", "")),
                "artifact_path": raw,
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_valid": valid,
                "validation_note": "" if valid else "missing or changed since OBS-084a freeze",
            }
        )
    source_validation = pd.DataFrame(source_rows)
    if source_validation.empty or not source_validation["hash_valid"].all():
        bad = source_validation.loc[~source_validation["hash_valid"], "artifact_path"].tolist()
        raise RuntimeError(f"Frozen source validation failed: {bad}")

    if require_repo_commit:
        current = git_commit(repo_root)
        frozen = str(payload.get("repo_commit", "unknown"))
        if current != frozen:
            raise RuntimeError(
                f"Repository commit differs from freeze: frozen={frozen}, current={current}"
            )

    return payload, freeze_tables, source_validation


def validate_key_and_partition(
    feature_df: pd.DataFrame,
    key_spec: pd.DataFrame,
    partition: pd.DataFrame,
) -> pd.DataFrame:
    selected = key_spec.loc[key_spec.get("selected", False).map(normalize_bool)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one frozen observation key; found {len(selected)}")
    row = selected.iloc[0]
    key_columns = json.loads(str(row.get("key_columns_json", "[]")))
    if tuple(key_columns) != SCIENTIFIC_OBSERVATION_KEY:
        raise RuntimeError(
            f"Frozen key differs from OBS-084 scientific key: {key_columns}"
        )
    require_columns(feature_df, SCIENTIFIC_OBSERVATION_KEY, "canonical feature table")
    keys = canonical_observation_key(feature_df)
    if keys.isna().any() or keys.duplicated().any():
        raise RuntimeError("Canonical scientific observation key is incomplete or non-unique")

    require_columns(
        partition,
        ["observation_key", "row_index", "cluster_id", "partition"],
        "OBS-084a partition manifest",
    )
    if partition["observation_key"].astype(str).duplicated().any():
        raise RuntimeError("Partition manifest has duplicate observation keys")
    if partition["row_index"].duplicated().any():
        raise RuntimeError("Partition manifest has duplicate row indices")
    if set(partition["partition"].astype(str).unique()) != {"discovery", "confirmation"}:
        raise RuntimeError("Partition manifest must contain discovery and confirmation roles")

    cross = pd.DataFrame(
        {
            "row_index": np.arange(len(feature_df), dtype=int),
            "observation_key": keys.astype(str),
        }
    ).merge(
        partition[["row_index", "observation_key", "cluster_id", "partition"]],
        on="row_index",
        how="left",
        suffixes=("_feature", "_partition"),
        validate="one_to_one",
    )
    if cross["partition"].isna().any():
        raise RuntimeError("Feature table rows are missing from the frozen partition")
    mismatch = cross["observation_key_feature"] != cross["observation_key_partition"].astype(str)
    if mismatch.any():
        raise RuntimeError(
            f"Observation-key mismatch for {int(mismatch.sum())} partition rows"
        )
    return cross


# -----------------------------------------------------------------------------
# Registry and carrier resolution
# -----------------------------------------------------------------------------


def load_record_catalog(
    registry: pd.DataFrame,
    subclasses: pd.DataFrame,
    partition_balance: pd.DataFrame,
) -> pd.DataFrame:
    rid = first_existing_column(registry, ("relation_id", "record_id"))
    relation = first_existing_column(registry, ("task", "relation"))
    carrier = first_existing_column(registry, ("carrier",))
    threshold = first_existing_column(registry, ("threshold",))
    if not rid or not relation or not carrier:
        raise ValueError("RIG registry lacks record, relation, or carrier fields")
    out = registry[[rid, relation, carrier] + ([threshold] if threshold else [])].copy()
    out = out.rename(columns={rid: "record_id", relation: "relation", carrier: "carrier"})
    out["threshold"] = (
        pd.to_numeric(out[threshold], errors="coerce")
        if threshold and threshold in out.columns
        else np.nan
    )

    sub_rid = first_existing_column(subclasses, ("record_id", "relation_id"))
    subclass_col = first_existing_column(
        subclasses,
        ("diagnostic_subclass", "subclass", "diagnostic_subclass_label"),
    )
    if sub_rid:
        keep = [sub_rid] + ([subclass_col] if subclass_col else [])
        sub = subclasses[keep].copy().rename(columns={sub_rid: "record_id"})
        if subclass_col:
            sub = sub.rename(columns={subclass_col: "subclass"})
        out = out.merge(sub, on="record_id", how="left")
    if "subclass" not in out.columns:
        out["subclass"] = ""

    eligible = (
        partition_balance.loc[
            partition_balance.get("confirmation_eligible", False).map(normalize_bool),
            "record_id",
        ]
        .astype(str)
        .unique()
        .tolist()
        if not partition_balance.empty and "record_id" in partition_balance.columns
        else []
    )
    out["confirmation_eligible"] = out["record_id"].astype(str).isin(eligible)
    out["discovery_maturity_cap"] = "FL2"
    return out.drop_duplicates("record_id").reset_index(drop=True)


def load_carrier_features(carrier_manifest: pd.DataFrame) -> dict[str, list[str]]:
    require_columns(
        carrier_manifest,
        ["carrier", "feature_name", "feature_present", "predictor_allowed"],
        "frozen carrier manifest",
    )
    features: dict[str, list[str]] = {}
    for carrier, group in carrier_manifest.groupby("carrier", sort=True):
        g = group.copy()
        if "feature_index" in g.columns:
            g = g.sort_values("feature_index")
        g = g[
            g["feature_present"].map(normalize_bool)
            & g["predictor_allowed"].map(normalize_bool)
        ]
        vals = [
            str(x)
            for x in g["feature_name"].tolist()
            if str(x) and str(x) not in FORBIDDEN_SCIENTIFIC_FEATURES
        ]
        if not vals:
            raise RuntimeError(f"Frozen carrier {carrier!r} has no scientific features")
        if len(vals) != len(set(vals)):
            raise RuntimeError(f"Frozen carrier {carrier!r} contains duplicate features")
        features[str(carrier)] = vals
    return features


def load_controls(
    relation_controls: pd.DataFrame,
    carrier_controls: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for family, df in (
        ("relation_control", relation_controls),
        ("carrier_control", carrier_controls),
    ):
        if df.empty:
            continue
        target = first_existing_column(df, ("record_id", "target_record_id"))
        control = first_existing_column(df, ("control_record_id",))
        evidence = first_existing_column(df, ("evidence_available",))
        if not target or not control:
            continue
        x = pd.DataFrame(
            {
                "record_id": df[target].astype(str),
                "control_record_id": df[control].astype(str),
                "control_family": family,
                "evidence_available": (
                    df[evidence].map(normalize_bool) if evidence else True
                ),
            }
        )
        parts.append(x)
    if not parts:
        return pd.DataFrame(
            columns=["record_id", "control_record_id", "control_family", "evidence_available"]
        )
    out = pd.concat(parts, ignore_index=True)
    out = out[out["record_id"] != out["control_record_id"]]
    return out.drop_duplicates().reset_index(drop=True)


# -----------------------------------------------------------------------------
# Discovery-only observation-level diagnostic instrument
# -----------------------------------------------------------------------------


def prepare_feature_table(feature_df: pd.DataFrame, partition: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        feature_df,
        [*SCIENTIFIC_OBSERVATION_KEY, "candidate_rank"],
        "canonical feature table",
    )
    out = feature_df.reset_index(drop=True).copy()
    out["row_index"] = np.arange(len(out), dtype=int)
    out["observation_id"] = canonical_observation_key(out).map(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )
    out["observation_key"] = canonical_observation_key(out)
    out["transition"] = transition_label(out)
    out["transition_midpoint"] = (
        pd.to_numeric(out["scale_index_from"], errors="coerce")
        + pd.to_numeric(out["scale_index_to"], errors="coerce")
    ) / 2.0
    if "transition_delta" not in out.columns:
        out["transition_delta"] = (
            pd.to_numeric(out["scale_index_to"], errors="coerce")
            - pd.to_numeric(out["scale_index_from"], errors="coerce")
        )
    out = out.merge(
        partition[["row_index", "cluster_id", "cluster_unit", "partition"]],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    if out["partition"].isna().any():
        raise RuntimeError("Partition join failed for canonical feature table")
    return out


def derive_support_fields(
    discovery_df: pd.DataFrame,
    seam_protocol: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = discovery_df.copy()
    threshold_rows: list[dict[str, Any]] = []

    # Scale bands reproduce the OBS-080b discovery-only quantile design.
    mid = pd.to_numeric(out["transition_midpoint"], errors="coerce")
    valid_mid = mid.dropna()
    if not valid_mid.empty:
        q33 = float(valid_mid.quantile(1 / 3))
        q67 = float(valid_mid.quantile(2 / 3))
        out["scale_band"] = pd.cut(
            mid,
            bins=[-np.inf, q33, q67, np.inf],
            labels=["early", "middle", "late"],
            include_lowest=True,
        ).astype("object")
        threshold_rows.extend(
            [
                {
                    "support_family": "scale_band",
                    "source_field": "transition_midpoint",
                    "threshold_name": "q33",
                    "threshold_value": q33,
                    "fit_partition": "discovery_only",
                    "direction_rule": "ascending transition midpoint",
                },
                {
                    "support_family": "scale_band",
                    "source_field": "transition_midpoint",
                    "threshold_name": "q67",
                    "threshold_value": q67,
                    "fit_partition": "discovery_only",
                    "direction_rule": "ascending transition midpoint",
                },
            ]
        )
    else:
        out["scale_band"] = np.nan

    if seam_protocol.empty:
        out["seam_relative_region"] = np.nan
        return out, pd.DataFrame(threshold_rows)

    protocol = seam_protocol.iloc[0]
    source_field = str(protocol.get("source_field", ""))
    bins = int(protocol.get("bin_count", 3))
    if not source_field or source_field not in out.columns or bins != 3:
        out["seam_relative_region"] = np.nan
        threshold_rows.append(
            {
                "support_family": "seam_relative",
                "source_field": source_field,
                "threshold_name": "unavailable",
                "threshold_value": np.nan,
                "fit_partition": "discovery_only",
                "direction_rule": "source unavailable or unsupported bin count",
            }
        )
        return out, pd.DataFrame(threshold_rows)

    values = pd.to_numeric(out[source_field], errors="coerce")
    finite = values.dropna()
    if finite.nunique() < 3:
        out["seam_relative_region"] = np.nan
        threshold_rows.append(
            {
                "support_family": "seam_relative",
                "source_field": source_field,
                "threshold_name": "unavailable",
                "threshold_value": np.nan,
                "fit_partition": "discovery_only",
                "direction_rule": "fewer than three unique discovery values",
            }
        )
        return out, pd.DataFrame(threshold_rows)

    q33 = float(finite.quantile(1 / 3))
    q67 = float(finite.quantile(2 / 3))
    lname = source_field.lower()
    high_means_near = any(x in lname for x in ("enrichment", "share", "contact")) and "distance" not in lname
    if high_means_near:
        labels = ["far", "intermediate", "near"]
        direction_rule = "higher enrichment/share/contact interpreted as more seam-adjacent"
    else:
        labels = ["near", "intermediate", "far"]
        direction_rule = "lower value interpreted as more seam-adjacent"
    out["seam_relative_region"] = pd.cut(
        values,
        bins=[-np.inf, q33, q67, np.inf],
        labels=labels,
        include_lowest=True,
    ).astype("object")
    threshold_rows.extend(
        [
            {
                "support_family": "seam_relative",
                "source_field": source_field,
                "threshold_name": "q33",
                "threshold_value": q33,
                "fit_partition": "discovery_only",
                "direction_rule": direction_rule,
            },
            {
                "support_family": "seam_relative",
                "source_field": source_field,
                "threshold_name": "q67",
                "threshold_value": q67,
                "fit_partition": "discovery_only",
                "direction_rule": direction_rule,
            },
        ]
    )
    return out, pd.DataFrame(threshold_rows)


def discovery_oof_predictions(
    record: Mapping[str, Any],
    discovery_df: pd.DataFrame,
    features: Sequence[str],
    seed: int,
) -> tuple[pd.DataFrame, list[DiscoveryFailure]]:
    record_id = str(record["record_id"])
    relation = str(record["relation"])
    carrier = str(record["carrier"])
    classes = parse_relation_classes(relation)
    failures: list[DiscoveryFailure] = []

    missing_features = [f for f in features if f not in discovery_df.columns]
    if missing_features:
        failures.append(
            DiscoveryFailure(
                "observation_model",
                record_id,
                "missing_frozen_carrier_features",
                json.dumps(missing_features),
            )
        )
        return pd.DataFrame(), failures

    sub = discovery_df[discovery_df["case"].astype(str).isin(classes)].copy()
    if sub.empty:
        failures.append(
            DiscoveryFailure("observation_model", record_id, "empty_relation_subset")
        )
        return pd.DataFrame(), failures
    if set(sub["case"].astype(str).unique()) != set(classes):
        failures.append(
            DiscoveryFailure(
                "observation_model",
                record_id,
                "missing_relation_class_in_discovery",
                json.dumps(sorted(sub["case"].astype(str).unique())),
            )
        )
        return pd.DataFrame(), failures

    feature_frame = sub[list(features)].apply(pd.to_numeric, errors="coerce")
    sub["predictor_missing_fraction"] = feature_frame.isna().mean(axis=1)
    sub["predictor_missing_any"] = feature_frame.isna().any(axis=1).astype(float)

    pred_parts: list[pd.DataFrame] = []
    cluster_col = "cluster_id"
    discovery_clusters = sorted(sub[cluster_col].dropna().astype(str).unique())
    for fold_index, heldout in enumerate(discovery_clusters):
        test_mask = sub[cluster_col].astype(str) == heldout
        train = sub.loc[~test_mask].copy()
        test = sub.loc[test_mask].copy()
        train_classes = set(train["case"].astype(str).unique())
        if train.empty or test.empty or train_classes != set(classes):
            failures.append(
                DiscoveryFailure(
                    "observation_model",
                    record_id,
                    "invalid_discovery_leave_cluster_fold",
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

        model = make_model(seed + fold_index)
        model.fit(train[list(features)].apply(pd.to_numeric, errors="coerce"), train["case"].astype(str))
        predicted = model.predict(test[list(features)].apply(pd.to_numeric, errors="coerce"))
        probabilities = model.predict_proba(test[list(features)].apply(pd.to_numeric, errors="coerce"))
        model_classes = list(model.named_steps["model"].classes_)
        class_index = {str(c): i for i, c in enumerate(model_classes)}

        true_prob: list[float] = []
        max_other_prob: list[float] = []
        margin: list[float] = []
        for true_label, prob_row in zip(test["case"].astype(str), probabilities):
            if true_label not in class_index:
                true_prob.append(np.nan)
                max_other_prob.append(np.nan)
                margin.append(np.nan)
                continue
            tp = float(prob_row[class_index[true_label]])
            others = [float(prob_row[i]) for c, i in class_index.items() if c != true_label]
            mo = max(others) if others else 0.0
            true_prob.append(tp)
            max_other_prob.append(mo)
            margin.append(tp - mo)

        part = test.copy()
        part["record_id"] = record_id
        part["relation"] = relation
        part["carrier"] = carrier
        part["true_regime"] = part["case"].astype(str)
        part["predicted_regime"] = pd.Series(predicted, index=part.index).astype(str)
        part["predicted_probability"] = true_prob
        part["max_other_probability"] = max_other_prob
        part["true_class_margin"] = margin
        part["signed_margin"] = margin
        part["correct"] = (
            part["true_regime"].astype(str) == part["predicted_regime"].astype(str)
        )
        part["misclassification_loss"] = 1.0 - part["correct"].astype(float)
        part["margin_loss"] = -pd.to_numeric(part["true_class_margin"], errors="coerce")
        part["log_loss"] = -np.log(
            np.clip(pd.to_numeric(part["predicted_probability"], errors="coerce"), 1e-12, 1.0)
        )
        part["fold_id"] = f"discovery_leave_object_out::{heldout}"
        part["heldout_cluster"] = heldout
        part["partition_role"] = "discovery"
        part["diagnostic_model"] = "logreg_balanced_scaled_median_imputed"
        part["carrier_features_json"] = json.dumps(list(features))
        pred_parts.append(part)

    if not pred_parts:
        return pd.DataFrame(), failures
    out = pd.concat(pred_parts, ignore_index=True)
    if (out["partition_role"] != "discovery").any():
        raise RuntimeError("Confirmation rows entered the discovery analytical frame")
    if out["observation_key"].duplicated().any():
        raise RuntimeError(f"Duplicate discovery predictions for record {record_id}")
    return out, failures


# -----------------------------------------------------------------------------
# Support enumeration and masks
# -----------------------------------------------------------------------------


def included_support_families(support_vocab: pd.DataFrame) -> list[str]:
    require_columns(
        support_vocab,
        ["support_family", "included_in_discovery_vocabulary"],
        "frozen support vocabulary",
    )
    return support_vocab.loc[
        support_vocab["included_in_discovery_vocabulary"].map(normalize_bool),
        "support_family",
    ].astype(str).tolist()


def available_support_dimensions(
    observation_df: pd.DataFrame,
    frozen_families: Sequence[str],
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    dimensions: list[tuple[str, str]] = []
    audit: list[dict[str, Any]] = []
    for family in frozen_families:
        if family in NON_SITE_SUPPORT_FAMILIES:
            audit.append(
                {
                    "support_family": family,
                    "support_column": "",
                    "available_for_site_search": False,
                    "reason": "record-constant contract/carrier dimension; usable for controls, not within-record localization",
                }
            )
            continue
        candidates = SUPPORT_FAMILY_COLUMNS.get(family, ())
        selected = next((c for c in candidates if c in observation_df.columns), None)
        if selected is None:
            audit.append(
                {
                    "support_family": family,
                    "support_column": "",
                    "available_for_site_search": False,
                    "reason": "no reviewed discrete observation-level field",
                }
            )
            continue
        unique = int(observation_df[selected].dropna().astype(str).nunique())
        available = unique >= 2
        audit.append(
            {
                "support_family": family,
                "support_column": selected,
                "available_for_site_search": available,
                "unique_discovery_values": unique,
                "reason": "" if available else "field is constant or empty on discovery rows",
            }
        )
        if available:
            dimensions.append((family, selected))
    return dimensions, audit


def support_mask(df: pd.DataFrame, support: SupportDefinition) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, value in zip(support.columns, support.values):
        mask &= df[col].astype(str) == str(value)
    return mask


def make_support_definition(
    families: Sequence[str],
    columns: Sequence[str],
    values: Sequence[Any],
) -> SupportDefinition:
    query = [
        {"support_family": str(f), "column": str(c), "operator": "eq", "value": str(v)}
        for f, c, v in zip(families, columns, values)
    ]
    payload = {
        "families": list(map(str, families)),
        "conditions": query,
    }
    support_id = stable_hash(payload)[:20]
    definition = " AND ".join(f"{f}:{c}={v}" for f, c, v in zip(families, columns, values))
    return SupportDefinition(
        support_id=support_id,
        depth=len(families),
        families=tuple(map(str, families)),
        columns=tuple(map(str, columns)),
        values=tuple(map(str, values)),
        support_query_json=json.dumps(query, sort_keys=True),
        support_definition=definition,
    )


def enumerate_supports(
    record_df: pd.DataFrame,
    dimensions: Sequence[tuple[str, str]],
    max_depth: int,
    min_site_rows: int,
    min_complement_rows: int,
    max_supports: int,
) -> list[SupportDefinition]:
    candidates: list[tuple[SupportDefinition, int, int]] = []
    max_depth = min(max_depth, 2)
    for depth in range(1, max_depth + 1):
        for dimension_combo in combinations(dimensions, depth):
            families = [x[0] for x in dimension_combo]
            columns = [x[1] for x in dimension_combo]
            observed = record_df[columns].dropna().drop_duplicates()
            for values in observed.itertuples(index=False, name=None):
                support = make_support_definition(families, columns, values)
                mask = support_mask(record_df, support)
                n_site = int(mask.sum())
                n_comp = int((~mask).sum())
                if n_site < min_site_rows or n_comp < min_complement_rows:
                    continue
                candidates.append((support, n_site, n_comp))

    # Outcome-blind deterministic cap. Prefer lower depth, larger minimum side,
    # then lexical support identity. This exposes the denominator without using
    # any loss or prediction result to select addresses.
    candidates.sort(
        key=lambda x: (
            x[0].depth,
            -min(x[1], x[2]),
            x[0].support_definition,
        )
    )
    return [x[0] for x in candidates[:max_supports]]


# -----------------------------------------------------------------------------
# Contrasts, matching, uncertainty, and permutations
# -----------------------------------------------------------------------------


def class_balanced_mean(
    df: pd.DataFrame,
    metric: str,
    classes: Sequence[str],
    min_class_rows: int,
) -> tuple[float, dict[str, float], dict[str, int], bool]:
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cls in classes:
        vals = pd.to_numeric(
            df.loc[df["true_regime"].astype(str) == str(cls), metric],
            errors="coerce",
        ).dropna()
        counts[str(cls)] = int(len(vals))
        if len(vals) < min_class_rows:
            return np.nan, means, counts, False
        means[str(cls)] = float(vals.mean())
    return float(np.mean(list(means.values()))), means, counts, True


def class_balanced_accuracy(
    df: pd.DataFrame,
    classes: Sequence[str],
    min_class_rows: int,
) -> float:
    loss, _, _, ok = class_balanced_mean(
        df,
        "misclassification_loss",
        classes,
        min_class_rows,
    )
    return 1.0 - loss if ok and np.isfinite(loss) else np.nan


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return 0.0 if np.isclose(np.mean(x), np.mean(y)) else np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


def compute_site_contrast(
    record_df: pd.DataFrame,
    support: SupportDefinition,
    predicate: Mapping[str, Any],
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    classes = parse_relation_classes(str(record_df["relation"].iloc[0]))
    metric = str(predicate["metric"])
    site_mask = support_mask(record_df, support)
    site = record_df.loc[site_mask].copy()
    complement = record_df.loc[~site_mask].copy()

    site_clusters = set(site["cluster_id"].astype(str))
    comp_clusters = set(complement["cluster_id"].astype(str))
    shared_clusters = site_clusters & comp_clusters

    site_mean, site_class_means, site_class_counts, site_class_ok = class_balanced_mean(
        site, metric, classes, args.min_class_rows
    )
    comp_mean, comp_class_means, comp_class_counts, comp_class_ok = class_balanced_mean(
        complement, metric, classes, args.min_class_rows
    )
    delta = site_mean - comp_mean if np.isfinite(site_mean) and np.isfinite(comp_mean) else np.nan

    site_ba = class_balanced_accuracy(site, classes, args.min_class_rows)
    comp_ba = class_balanced_accuracy(complement, classes, args.min_class_rows)
    threshold_crossed = (
        np.isfinite(threshold)
        and np.isfinite(site_ba)
        and np.isfinite(comp_ba)
        and site_ba < threshold
        and comp_ba >= threshold
    )

    midpoint_smd = standardized_mean_difference(
        site.get("transition_midpoint", pd.Series(dtype=float)),
        complement.get("transition_midpoint", pd.Series(dtype=float)),
    )
    missing_smd = standardized_mean_difference(
        site.get("predictor_missing_fraction", pd.Series(dtype=float)),
        complement.get("predictor_missing_fraction", pd.Series(dtype=float)),
    )

    checks = {
        "site_rows": len(site) >= args.min_site_rows,
        "complement_rows": len(complement) >= args.min_complement_rows,
        "site_clusters": len(site_clusters) >= args.min_site_clusters,
        "complement_clusters": len(comp_clusters) >= args.min_complement_clusters,
        "shared_clusters": len(shared_clusters) >= args.min_shared_clusters,
        "site_class_support": site_class_ok,
        "complement_class_support": comp_class_ok,
    }
    complement_admissible = all(checks.values())
    if not checks["site_rows"] or not checks["complement_rows"]:
        initial_status = "insufficient_support"
    elif not complement_admissible:
        initial_status = "inadmissible_complement"
    else:
        initial_status = "contrast_computable"

    if predicate["failure_mode"] == "threshold_breach" and complement_admissible:
        predicate_semantics_pass = threshold_crossed
    else:
        predicate_semantics_pass = True

    return {
        "support_id": support.support_id,
        "support_depth": support.depth,
        "support_families": "|".join(support.families),
        "support_columns": "|".join(support.columns),
        "support_values": "|".join(support.values),
        "support_definition": support.support_definition,
        "support_query_json": support.support_query_json,
        "complement_definition": f"NOT ({support.support_definition}) within same record and discovery partition",
        "failure_predicate": predicate["failure_predicate"],
        "failure_mode": predicate["failure_mode"],
        "metric": metric,
        "minimum_effect": float(predicate["minimum_effect"]),
        "expected_direction": predicate["expected_direction"],
        "threshold_basis": predicate["threshold_basis"],
        "n_site_rows": int(len(site)),
        "n_complement_rows": int(len(complement)),
        "n_site_clusters": int(len(site_clusters)),
        "n_complement_clusters": int(len(comp_clusters)),
        "n_shared_clusters": int(len(shared_clusters)),
        "site_class_counts_json": json.dumps(site_class_counts, sort_keys=True),
        "complement_class_counts_json": json.dumps(comp_class_counts, sort_keys=True),
        "site_class_means_json": json.dumps(site_class_means, sort_keys=True),
        "complement_class_means_json": json.dumps(comp_class_means, sort_keys=True),
        "site_loss": site_mean,
        "complement_loss": comp_mean,
        "site_relative_contrast": delta,
        "site_balanced_accuracy": site_ba,
        "complement_balanced_accuracy": comp_ba,
        "registry_threshold": threshold,
        "threshold_crossed": threshold_crossed,
        "predicate_semantics_pass": predicate_semantics_pass,
        "transition_midpoint_smd": midpoint_smd,
        "predictor_missing_fraction_smd": missing_smd,
        "matching_variables": "true_regime;object_cluster_overlap;transition_midpoint_audit;predictor_missingness_audit",
        "exposure_normalization": "equal-weight class-balanced loss; site and complement evaluated on discovery-only observations",
        "complement_admissible": complement_admissible,
        "initial_status": initial_status,
        "matching_check_json": json.dumps(checks, sort_keys=True),
    }


def contrast_from_mask(
    df: pd.DataFrame,
    site_mask: pd.Series,
    metric: str,
    classes: Sequence[str],
    min_class_rows: int,
) -> float:
    site_mean, _, _, site_ok = class_balanced_mean(
        df.loc[site_mask], metric, classes, min_class_rows
    )
    comp_mean, _, _, comp_ok = class_balanced_mean(
        df.loc[~site_mask], metric, classes, min_class_rows
    )
    if not site_ok or not comp_ok:
        return np.nan
    return site_mean - comp_mean


def cluster_uncertainty(
    record_df: pd.DataFrame,
    support: SupportDefinition,
    metric: str,
    args: argparse.Namespace,
    key: str,
) -> dict[str, Any]:
    classes = parse_relation_classes(str(record_df["relation"].iloc[0]))
    site_mask = support_mask(record_df, support)
    clusters = sorted(record_df["cluster_id"].astype(str).unique())
    point = contrast_from_mask(
        record_df, site_mask, metric, classes, args.min_class_rows
    )

    loo_values: list[float] = []
    for cluster in clusters:
        keep = record_df["cluster_id"].astype(str) != cluster
        if not keep.any():
            continue
        value = contrast_from_mask(
            record_df.loc[keep].copy(),
            site_mask.loc[keep],
            metric,
            classes,
            args.min_class_rows,
        )
        if np.isfinite(value):
            loo_values.append(float(value))

    rng = rng_for(args.seed, key, "cluster_bootstrap")
    bootstrap_values: list[float] = []
    if clusters:
        grouped = {c: record_df[record_df["cluster_id"].astype(str) == c].copy() for c in clusters}
        for _ in range(args.n_cluster_bootstrap):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            parts: list[pd.DataFrame] = []
            for draw_index, cluster in enumerate(sampled):
                part = grouped[str(cluster)].copy()
                part["__bootstrap_cluster_copy"] = f"{cluster}::{draw_index}"
                parts.append(part)
            boot = pd.concat(parts, ignore_index=True)
            bmask = support_mask(boot, support)
            value = contrast_from_mask(
                boot, bmask, metric, classes, args.min_class_rows
            )
            if np.isfinite(value):
                bootstrap_values.append(float(value))

    ci_low, ci_high = quantile_ci(bootstrap_values, args.alpha)
    direction_consistency = (
        float(np.mean(np.asarray(loo_values) > 0)) if loo_values else np.nan
    )
    bootstrap_positive_share = (
        float(np.mean(np.asarray(bootstrap_values) > 0)) if bootstrap_values else np.nan
    )
    return {
        "point_estimate": point,
        "independent_cluster_count": len(clusters),
        "loo_successful_count": len(loo_values),
        "loo_min": float(min(loo_values)) if loo_values else np.nan,
        "loo_max": float(max(loo_values)) if loo_values else np.nan,
        "direction_consistency": direction_consistency,
        "bootstrap_requested": args.n_cluster_bootstrap,
        "bootstrap_successful_count": len(bootstrap_values),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_positive_share": bootstrap_positive_share,
        "resampling_unit": "object_cluster",
        "uncertainty_method": "leave-one-object-out sensitivity plus object-cluster bootstrap",
    }


def permutation_p_value(
    record_df: pd.DataFrame,
    support: SupportDefinition,
    metric: str,
    args: argparse.Namespace,
    key: str,
) -> tuple[float, int, str]:
    classes = parse_relation_classes(str(record_df["relation"].iloc[0]))
    site_mask = support_mask(record_df, support)
    actual = contrast_from_mask(
        record_df, site_mask, metric, classes, args.min_class_rows
    )
    if not np.isfinite(actual):
        return np.nan, 0, "actual_contrast_unavailable"

    strata_cols = ["cluster_id", "true_regime"]
    strata = list(record_df.groupby(strata_cols, dropna=False).groups.values())
    if not any(site_mask.loc[idx].nunique() > 1 for idx in strata):
        return 1.0, 0, "support_indicator_constant_within_all_object_class_strata"

    rng = rng_for(args.seed, key, "permutation")
    null: list[float] = []
    base = site_mask.to_numpy(dtype=bool)
    positions = {index: pos for pos, index in enumerate(record_df.index)}
    for _ in range(args.n_permutations):
        perm = base.copy()
        for idx in strata:
            loc = np.asarray([positions[x] for x in idx], dtype=int)
            perm[loc] = rng.permutation(perm[loc])
        perm_mask = pd.Series(perm, index=record_df.index)
        value = contrast_from_mask(
            record_df, perm_mask, metric, classes, args.min_class_rows
        )
        if np.isfinite(value):
            null.append(float(value))
    if not null:
        return np.nan, 0, "no_valid_permutations"
    p = (1 + int(np.sum(np.asarray(null) >= actual))) / (1 + len(null))
    return float(p), len(null), "within_object_class_support_label_permutation"


# -----------------------------------------------------------------------------
# Control adjustment and minimality
# -----------------------------------------------------------------------------


def support_from_contrast_row(row: Mapping[str, Any]) -> SupportDefinition:
    families = tuple(str(row["support_families"]).split("|"))
    columns = tuple(str(row["support_columns"]).split("|"))
    values = tuple(str(row["support_values"]).split("|"))
    return SupportDefinition(
        support_id=str(row["support_id"]),
        depth=int(row["support_depth"]),
        families=families,
        columns=columns,
        values=values,
        support_query_json=str(row["support_query_json"]),
        support_definition=str(row["support_definition"]),
    )


def compute_control_adjustment(
    target_row: Mapping[str, Any],
    observation_by_record: Mapping[str, pd.DataFrame],
    controls: pd.DataFrame,
    record_catalog: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_id = str(target_row["record_id"])
    support = support_from_contrast_row(target_row)
    predicate = next(
        p for p in PREDICATES if p["failure_predicate"] == target_row["failure_predicate"]
    )
    target_delta = float(target_row["site_relative_contrast"])
    target_controls = controls[
        (controls["record_id"].astype(str) == target_id)
        & controls["evidence_available"].map(normalize_bool)
    ]
    rows: list[dict[str, Any]] = []
    control_deltas: list[float] = []
    lambdas: list[float] = []
    for _, control in target_controls.iterrows():
        cid = str(control["control_record_id"])
        cdf = observation_by_record.get(cid)
        if cdf is None or cdf.empty:
            rows.append(
                {
                    "record_id": target_id,
                    "support_id": support.support_id,
                    "failure_predicate": predicate["failure_predicate"],
                    "control_record_id": cid,
                    "control_family": control["control_family"],
                    "control_status": "control_observation_evidence_unavailable",
                }
            )
            continue
        catalog = record_catalog[record_catalog["record_id"].astype(str) == cid]
        cthreshold = float(catalog.iloc[0]["threshold"]) if not catalog.empty else np.nan
        cresult = compute_site_contrast(
            cdf,
            support,
            predicate,
            cthreshold,
            args,
        )
        cdelta = cresult["site_relative_contrast"]
        admissible = normalize_bool(cresult["complement_admissible"]) and np.isfinite(cdelta)
        lam = target_delta - float(cdelta) if admissible else np.nan
        if admissible:
            control_deltas.append(float(cdelta))
            lambdas.append(float(lam))
        rows.append(
            {
                "record_id": target_id,
                "support_id": support.support_id,
                "failure_predicate": predicate["failure_predicate"],
                "failure_mode": predicate["failure_mode"],
                "control_record_id": cid,
                "control_family": control["control_family"],
                "control_relation": str(cdf["relation"].iloc[0]),
                "control_carrier": str(cdf["carrier"].iloc[0]),
                "target_site_relative_contrast": target_delta,
                "control_site_relative_contrast": cdelta,
                "control_adjusted_contrast": lam,
                "control_complement_admissible": cresult["complement_admissible"],
                "control_site_rows": cresult["n_site_rows"],
                "control_complement_rows": cresult["n_complement_rows"],
                "control_shared_clusters": cresult["n_shared_clusters"],
                "control_status": "admissible" if admissible else cresult["initial_status"],
                "control_admissibility_rule": "same frozen support query; class-balanced site/complement contrast; minimum row, class, and shared-object overlap",
            }
        )

    if control_deltas:
        median_control = float(np.median(control_deltas))
        median_lambda = target_delta - median_control
        positive_share = float(np.mean(np.asarray(lambdas) > 0))
        robust = (
            median_lambda >= args.min_control_adjusted_effect
            and positive_share >= args.min_positive_control_share
        )
        status = "control_adjusted_signal" if robust else "control_explained"
    else:
        median_control = np.nan
        median_lambda = np.nan
        positive_share = np.nan
        robust = False
        status = "no_admissible_controls"

    aggregate = {
        "record_id": target_id,
        "support_id": support.support_id,
        "failure_predicate": predicate["failure_predicate"],
        "admissible_control_count": len(control_deltas),
        "alternative_control_count": int(len(target_controls)),
        "median_control_site_relative_contrast": median_control,
        "median_control_adjusted_contrast": median_lambda,
        "positive_control_adjusted_share": positive_share,
        "control_robustness_pass": robust,
        "control_robustness_status": status,
        "eligible_control_records_json": json.dumps(
            sorted(target_controls["control_record_id"].astype(str).unique().tolist())
        ),
    }
    return rows, aggregate


def query_conditions(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    return set(zip(str(row["support_columns"]).split("|"), str(row["support_values"]).split("|")))


def assign_minimal_support_status(
    candidates: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["support_dominance_status"] = "non_dominated"
    out["support_relationship"] = "no stricter nominated support found"
    out["equivalent_or_overlapping_supports"] = "[]"
    out["preferred_support_reason"] = "narrowest surviving predeclared support under depth cap"

    for (record_id, predicate), idx in out.groupby(["record_id", "failure_predicate"]).groups.items():
        group = out.loc[idx]
        condition_map = {i: query_conditions(row) for i, row in group.iterrows()}
        for i, row in group.iterrows():
            current = condition_map[i]
            refinements: list[str] = []
            for j, other in group.iterrows():
                if i == j:
                    continue
                other_conditions = condition_map[j]
                if current < other_conditions:  # strict refinement contains all current conditions
                    if float(other["site_relative_contrast"]) >= float(row["site_relative_contrast"]) - tolerance:
                        refinements.append(str(other["candidate_id"]))
            if refinements:
                out.at[i, "support_dominance_status"] = "dominated_by_stricter_support"
                out.at[i, "support_relationship"] = "proper predeclared refinement also survives with equivalent effect"
                out.at[i, "equivalent_or_overlapping_supports"] = json.dumps(sorted(refinements))
                out.at[i, "preferred_support_reason"] = "retain stricter surviving support(s)"
    return out


# -----------------------------------------------------------------------------
# Reporting and manifest sealing
# -----------------------------------------------------------------------------


def build_input_manifest(
    repo_root: Path,
    freeze_dir: Path,
    freeze_payload: Mapping[str, Any],
    source_validation: pd.DataFrame,
    extra_paths: Mapping[str, Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "artifact_role": "obs084a_freeze_manifest",
            "artifact_path": str(freeze_dir / "obs084a_freeze_manifest.json"),
            "exists": True,
            "sha256": sha256_file(freeze_dir / "obs084a_freeze_manifest.json"),
            "freeze_manifest_id": freeze_payload.get("freeze_manifest_id", ""),
            "freeze_status": freeze_payload.get("status", ""),
        }
    ]
    for _, row in source_validation.iterrows():
        rows.append(
            {
                "artifact_role": row["source_role"],
                "artifact_path": row["artifact_path"],
                "exists": row["exists"],
                "sha256": row["actual_sha256"],
                "freeze_manifest_id": freeze_payload.get("freeze_manifest_id", ""),
                "freeze_status": freeze_payload.get("status", ""),
            }
        )
    for role, path in extra_paths.items():
        rows.append(
            {
                "artifact_role": role,
                "artifact_path": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
                "freeze_manifest_id": freeze_payload.get("freeze_manifest_id", ""),
                "freeze_status": freeze_payload.get("status", ""),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["artifact_role", "artifact_path"])


def confirmation_partition_id(partition: pd.DataFrame) -> str:
    keys = sorted(
        partition.loc[
            partition["partition"].astype(str) == "confirmation", "observation_key"
        ].astype(str).tolist()
    )
    return stable_hash({"role": "confirmation", "observation_keys": keys})


def write_report(
    path: Path,
    status: str,
    freeze_payload: Mapping[str, Any],
    input_manifest: pd.DataFrame,
    record_catalog: pd.DataFrame,
    observation_losses: pd.DataFrame,
    thresholds: pd.DataFrame,
    support_audit: pd.DataFrame,
    inventory: pd.DataFrame,
    contrasts: pd.DataFrame,
    control_agg: pd.DataFrame,
    minimal: pd.DataFrame,
    multiplicity: pd.DataFrame,
    failures: pd.DataFrame,
    candidate_manifest_id: str,
) -> None:
    lines: list[str] = []
    lines.append("# OBS-084b — Direct Failure-Support Discovery")
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"Discovery completed with status: `{status}`")
    lines.append("")
    lines.append(f"OBS-084a freeze manifest: `{freeze_payload.get('freeze_manifest_id', '')}`")
    lines.append(f"Candidate manifest ID: `{candidate_manifest_id}`")
    lines.append("")
    lines.append(
        "This stage uses only the frozen discovery partition. It nominates at most FL2 "
        "artifact-indexed candidates and does not inspect reserved confirmation outcomes."
    )
    lines.append("")
    lines.append("## Canonical guardrails")
    lines.append("")
    lines.append("> Directness is artifact-direct, not metaphysically direct and not causally direct.")
    lines.append("")
    lines.append("> Discovery nominates a support; reserved evidence earns the localization claim.")
    lines.append("")
    lines.append("No result in this report is confirmed, actionable, causal, externally generalized, or formally topological.")
    lines.append("")
    lines.append("## Frozen-input validation")
    lines.append("")
    lines.append(markdown_table(input_manifest[[c for c in ["artifact_role", "artifact_path", "exists", "sha256"] if c in input_manifest.columns]], 40))
    lines.append("")
    lines.append("## Discovery-only diagnostic instrument")
    lines.append("")
    lines.append(f"- Registry records: {len(record_catalog)}")
    lines.append(f"- Observation-loss rows: {len(observation_losses)}")
    lines.append(f"- Discovery observations represented: {observation_losses['observation_id'].nunique() if not observation_losses.empty else 0}")
    lines.append(f"- Structural resampling unit: object (`cluster_id`)")
    lines.append(f"- Diagnostic model: discovery-only leave-one-object-out balanced logistic regression")
    lines.append("")
    lines.append("Upstream prediction artifacts are lineage inputs, but their outcomes are not reused as primary evidence because their training folds may span the later reserved partition.")
    lines.append("")
    lines.append("## Discovery-fitted support thresholds")
    lines.append("")
    lines.append(markdown_table(thresholds, 20))
    lines.append("")
    lines.append("## Frozen support-vocabulary execution audit")
    lines.append("")
    lines.append(markdown_table(support_audit, 30))
    lines.append("")
    lines.append("## Candidate denominator")
    lines.append("")
    lines.append(f"- Record-support addresses generated: {len(inventory)}")
    lines.append(f"- Unique support definitions: {inventory['support_id'].nunique() if not inventory.empty else 0}")
    lines.append(f"- Predicate-indexed tests: {len(contrasts)}")
    lines.append(f"- Records with computable contrasts: {contrasts.loc[contrasts.get('complement_admissible', False).map(normalize_bool), 'record_id'].nunique() if not contrasts.empty else 0}")
    lines.append("")
    if not multiplicity.empty:
        denom = (
            multiplicity.groupby("failure_predicate", as_index=False)
            .agg(tests=("candidate_test_id", "count"), finite_p=("permutation_p", lambda x: int(pd.to_numeric(x, errors='coerce').notna().sum())))
        )
        lines.append(markdown_table(denom, 20))
        lines.append("")
    lines.append("## Control adjustment")
    lines.append("")
    if control_agg.empty:
        lines.append("_No admissible target candidates reached control adjustment._")
    else:
        display = control_agg.sort_values(
            "median_control_adjusted_contrast", ascending=False, na_position="last"
        )
        lines.append(markdown_table(display, 40))
    lines.append("")
    lines.append("## FL2 candidate result")
    lines.append("")
    nominated = minimal[
        minimal.get("candidate_status", "").astype(str).str.startswith("fl2_candidate_nominated")
        & (minimal.get("support_dominance_status", "") == "non_dominated")
    ] if not minimal.empty else pd.DataFrame()
    if nominated.empty:
        lines.append(
            "No non-dominated support survived complement admissibility, minimum effect, "
            "object-cluster sensitivity, multiplicity, and control adjustment."
        )
    else:
        cols = [
            "candidate_id", "record_id", "failure_predicate", "support_definition",
            "site_relative_contrast", "bootstrap_ci_low", "permutation_q_record_predicate",
            "median_control_adjusted_contrast", "candidate_status",
        ]
        lines.append(markdown_table(nominated[[c for c in cols if c in nominated.columns]], 80))
    lines.append("")
    lines.append("## Failures and exclusions")
    lines.append("")
    if failures.empty:
        lines.append("_No execution failures._")
    else:
        lines.append(markdown_table(failures, 80))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if nominated.empty:
        lines.append(
            "The valid null outcome is that the frozen discovery search did not nominate an "
            "FL2 support under the declared evidence rules. This does not prove that no localized "
            "support exists; it bounds what the current artifact resolution and four discovery "
            "object clusters can support."
        )
    else:
        lines.append(
            "The nominated rows are artifact-indexed FL2 candidates only. Their support definitions, "
            "predicates, complements, controls, metrics, and confirmation partition identity are "
            "sealed in the candidate manifest. No candidate may be described as a direct witness "
            "unless it survives a later reserved confirmation stage."
        )
    lines.append("")
    lines.append("## Canonical result statement")
    lines.append("")
    lines.append(
        "OBS-084b executes a blinded, discovery-only search over the frozen PAM/RIG evidence spine. "
        "It may nominate artifact-indexed FL2 candidate supports, but establishes no confirmed direct "
        "failure support, causal origin, repair target, actionability, external generalization, or "
        "formal topology."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    freeze_dir = resolve_path(repo_root, args.freeze_dir)
    obs083_dir = resolve_path(repo_root, args.obs083_dir)
    registry_path = resolve_path(repo_root, args.registry)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_conjunction_depth not in (1, 2):
        raise ValueError("--max-conjunction-depth must be 1 or 2")
    if not (0 < args.alpha < 1):
        raise ValueError("--alpha must be between 0 and 1")
    if not (0 < args.discovery_fdr <= 1):
        raise ValueError("--discovery-fdr must be in (0, 1]")

    freeze_payload, freeze_tables, source_validation = load_and_validate_freeze(
        repo_root, freeze_dir, args.require_repo_commit
    )

    canonical_feature_path = resolve_path(
        repo_root, str(freeze_payload["canonical_feature_table"])
    )
    feature_df = read_csv_required(canonical_feature_path)
    key_crosswalk = validate_key_and_partition(
        feature_df,
        freeze_tables["observation_key"],
        freeze_tables["partition"],
    )
    prepared = prepare_feature_table(feature_df, freeze_tables["partition"])
    discovery = prepared[prepared["partition"].astype(str) == "discovery"].copy()
    confirmation_keys = set(
        prepared.loc[
            prepared["partition"].astype(str) == "confirmation", "observation_key"
        ].astype(str)
    )
    if set(discovery["observation_key"].astype(str)) & confirmation_keys:
        raise RuntimeError("Discovery and confirmation observation keys overlap")

    discovery, support_thresholds = derive_support_fields(
        discovery, freeze_tables["seam_protocol"]
    )

    subclasses_path = obs083_dir / OBS083_FILES["subclasses"]
    relation_controls_path = obs083_dir / OBS083_FILES["relation_controls"]
    carrier_controls_path = obs083_dir / OBS083_FILES["carrier_controls"]
    extra_paths = {
        "obs083_subclasses": subclasses_path,
        "obs083_relation_controls": relation_controls_path,
        "obs083_carrier_controls": carrier_controls_path,
        "rig_relation_registry": registry_path,
        "obs084b_script": Path(__file__).resolve(),
    }
    for p in (subclasses_path, relation_controls_path, carrier_controls_path, registry_path):
        if not p.exists():
            raise FileNotFoundError(p)

    registry = read_csv_required(registry_path)
    subclasses = read_csv_required(subclasses_path)
    record_catalog = load_record_catalog(
        registry,
        subclasses,
        freeze_tables["partition_balance"],
    )
    carrier_features = load_carrier_features(freeze_tables["carrier_features"])
    controls = load_controls(
        read_csv_required(relation_controls_path),
        read_csv_required(carrier_controls_path),
    )

    missing_carriers = sorted(set(record_catalog["carrier"].astype(str)) - set(carrier_features))
    if missing_carriers:
        raise RuntimeError(f"Registry carriers missing from frozen carrier manifest: {missing_carriers}")

    input_manifest = build_input_manifest(
        repo_root,
        freeze_dir,
        freeze_payload,
        source_validation,
        extra_paths,
    )

    frozen_families = included_support_families(freeze_tables["support_vocabulary"])
    dimensions, support_audit_rows = available_support_dimensions(
        discovery, frozen_families
    )
    support_audit = pd.DataFrame(support_audit_rows)

    observation_parts: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    observation_by_record: dict[str, pd.DataFrame] = {}
    for _, record in record_catalog.iterrows():
        rid = str(record["record_id"])
        carrier = str(record["carrier"])
        obs, failures = discovery_oof_predictions(
            record,
            discovery,
            carrier_features[carrier],
            args.seed,
        )
        for failure in failures:
            failure_rows.append(failure.__dict__)
        if not obs.empty:
            observation_by_record[rid] = obs
            observation_parts.append(obs)

    observation_losses = (
        pd.concat(observation_parts, ignore_index=True)
        if observation_parts
        else pd.DataFrame()
    )
    if not observation_losses.empty:
        if (observation_losses["partition_role"] != "discovery").any():
            raise RuntimeError("Non-discovery observation entered output")
        if set(observation_losses["observation_key"].astype(str)) & confirmation_keys:
            raise RuntimeError("Confirmation observation leaked into discovery output")

    inventory_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []

    for _, record in record_catalog.iterrows():
        rid = str(record["record_id"])
        rdf = observation_by_record.get(rid)
        if rdf is None or rdf.empty:
            continue
        supports = enumerate_supports(
            rdf,
            dimensions,
            args.max_conjunction_depth,
            args.min_site_rows,
            args.min_complement_rows,
            args.max_supports_per_record,
        )
        for support in supports:
            smask = support_mask(rdf, support)
            inventory_rows.append(
                {
                    "record_id": rid,
                    "relation": record["relation"],
                    "carrier": record["carrier"],
                    "subclass": record["subclass"],
                    "confirmation_eligible": record["confirmation_eligible"],
                    "support_id": support.support_id,
                    "support_depth": support.depth,
                    "support_families": "|".join(support.families),
                    "support_columns": "|".join(support.columns),
                    "support_values": "|".join(support.values),
                    "support_definition": support.support_definition,
                    "support_query_json": support.support_query_json,
                    "n_site_rows": int(smask.sum()),
                    "n_complement_rows": int((~smask).sum()),
                    "generation_status": "generated_from_frozen_vocabulary",
                }
            )
            for predicate in PREDICATES:
                result = compute_site_contrast(
                    rdf,
                    support,
                    predicate,
                    float(record["threshold"]) if pd.notna(record["threshold"]) else np.nan,
                    args,
                )
                candidate_test_id = stable_hash(
                    {
                        "record_id": rid,
                        "failure_predicate": predicate["failure_predicate"],
                        "support_query": json.loads(support.support_query_json),
                        "metric": predicate["metric"],
                        "freeze_manifest_id": freeze_payload["freeze_manifest_id"],
                    }
                )[:24]
                result.update(
                    {
                        "candidate_test_id": candidate_test_id,
                        "record_id": rid,
                        "relation": record["relation"],
                        "carrier": record["carrier"],
                        "subclass": record["subclass"],
                        "confirmation_eligible": record["confirmation_eligible"],
                    }
                )
                matching_rows.append(
                    {
                        k: result[k]
                        for k in [
                            "candidate_test_id", "record_id", "support_id",
                            "failure_predicate", "support_definition",
                            "n_site_rows", "n_complement_rows", "n_site_clusters",
                            "n_complement_clusters", "n_shared_clusters",
                            "site_class_counts_json", "complement_class_counts_json",
                            "transition_midpoint_smd", "predictor_missing_fraction_smd",
                            "matching_variables", "exposure_normalization",
                            "complement_admissible", "initial_status", "matching_check_json",
                        ]
                    }
                )

                # Expensive dependence-aware diagnostics are run in a second,
                # explicitly capped pass after all outcome-blind support addresses
                # have received their basic site/complement contrast. Tests that do
                # not pass the predeclared effect/predicate screen remain in the
                # multiplicity denominator with permutation_p=1.
                result.update(
                    {
                        "permutation_p": 1.0,
                        "permutation_count": 0,
                        "permutation_method": "not_selected_for_resampling",
                        "point_estimate": result["site_relative_contrast"],
                        "independent_cluster_count": int(rdf["cluster_id"].nunique()),
                        "loo_successful_count": 0,
                        "loo_min": np.nan,
                        "loo_max": np.nan,
                        "direction_consistency": np.nan,
                        "bootstrap_requested": args.n_cluster_bootstrap,
                        "bootstrap_successful_count": 0,
                        "bootstrap_ci_low": np.nan,
                        "bootstrap_ci_high": np.nan,
                        "bootstrap_positive_share": np.nan,
                        "resampling_unit": "object_cluster",
                        "uncertainty_method": "not_selected_for_resampling",
                        "resampling_selected": False,
                        "resampling_selection_status": "pending_second_pass",
                    }
                )
                contrast_rows.append(result)

    inventory = pd.DataFrame(inventory_rows)
    matching = pd.DataFrame(matching_rows)
    contrasts = pd.DataFrame(contrast_rows)

    if contrasts.empty:
        uncertainty_df = pd.DataFrame()
        multiplicity = pd.DataFrame()
        control_rows_df = pd.DataFrame()
        control_agg_df = pd.DataFrame()
        minimal = pd.DataFrame()
    else:
        # ------------------------------------------------------------------
        # Two-pass dependence-aware resampling
        # ------------------------------------------------------------------
        # Every predeclared support/predicate test remains in the multiplicity
        # denominator. Expensive cluster bootstrap and permutation diagnostics
        # are applied only to a deterministic, outcome-blind top-K address set
        # within each record × predicate family. Selection uses support structure
        # and complement admissibility only, never observed loss magnitude or
        # predicate success. Nonselected tests receive p=1 and can never become
        # candidates. This is a computational scope gate, not evidence.
        contrasts["effect_pass"] = (
            pd.to_numeric(contrasts["site_relative_contrast"], errors="coerce")
            >= pd.to_numeric(contrasts["minimum_effect"], errors="coerce")
        )
        contrasts["resampling_design_eligible"] = (
            contrasts["complement_admissible"].map(normalize_bool)
        )

        selected_indices: list[int] = []
        cap = max(0, int(args.max_resampled_tests_per_record_predicate))
        if cap > 0:
            prescreened = contrasts[contrasts["resampling_design_eligible"]].copy()
            for _, group in prescreened.groupby(
                ["record_id", "failure_predicate"], sort=True, dropna=False
            ):
                # Outcome-blind round-robin over support-family signatures.
                # This prevents a lexical prefix (for example cohort supports)
                # from exhausting the cap before other frozen address families
                # are represented. Single-family supports precede conjunctions;
                # within a signature, lexical support identity is deterministic.
                ranked = group.sort_values(
                    ["support_depth", "support_families", "support_definition", "candidate_test_id"],
                    ascending=[True, True, True, True],
                    kind="mergesort",
                ).copy()
                ranked["__within_signature_rank"] = ranked.groupby(
                    "support_families", sort=True, dropna=False
                ).cumcount()
                ranked = ranked.sort_values(
                    ["__within_signature_rank", "support_depth", "support_families", "support_definition", "candidate_test_id"],
                    ascending=[True, True, True, True, True],
                    kind="mergesort",
                )
                selected_indices.extend(ranked.head(cap).index.tolist())

        selected_set = set(selected_indices)
        uncertainty_rows = []
        for idx, row in contrasts.iterrows():
            if idx not in selected_set:
                if not normalize_bool(row["complement_admissible"]):
                    selection_status = "not_resampled_inadmissible_complement"
                elif cap <= 0:
                    selection_status = "not_resampled_cap_zero"
                else:
                    selection_status = "not_resampled_predeclared_cap"
                contrasts.at[idx, "resampling_selected"] = False
                contrasts.at[idx, "resampling_selection_status"] = selection_status
                contrasts.at[idx, "permutation_p"] = 1.0
                contrasts.at[idx, "permutation_count"] = 0
                contrasts.at[idx, "permutation_method"] = selection_status
                contrasts.at[idx, "uncertainty_method"] = selection_status
                continue

            rid = str(row["record_id"])
            rdf = observation_by_record.get(rid)
            if rdf is None or rdf.empty:
                contrasts.at[idx, "resampling_selected"] = False
                contrasts.at[idx, "resampling_selection_status"] = "not_resampled_record_observations_unavailable"
                contrasts.at[idx, "permutation_p"] = 1.0
                continue

            support = support_from_contrast_row(row)
            key = str(row["candidate_test_id"])
            uncertainty = cluster_uncertainty(
                rdf,
                support,
                str(row["metric"]),
                args,
                key,
            )
            permutation_p, permutation_count, permutation_method = permutation_p_value(
                rdf,
                support,
                str(row["metric"]),
                args,
                key,
            )
            if not np.isfinite(permutation_p):
                permutation_p = 1.0
                permutation_method = f"{permutation_method};conservative_p_equals_one"

            contrasts.at[idx, "resampling_selected"] = True
            contrasts.at[idx, "resampling_selection_status"] = "selected_by_outcome_blind_record_predicate_top_k"
            contrasts.at[idx, "permutation_p"] = float(permutation_p)
            contrasts.at[idx, "permutation_count"] = int(permutation_count)
            contrasts.at[idx, "permutation_method"] = permutation_method
            for key_name, value in uncertainty.items():
                contrasts.at[idx, key_name] = value

            uncertainty_rows.append(
                {
                    "candidate_test_id": row["candidate_test_id"],
                    "record_id": row["record_id"],
                    "failure_predicate": row["failure_predicate"],
                    "support_id": row["support_id"],
                    "support_definition": row["support_definition"],
                    "resampling_selected": True,
                    "resampling_selection_status": "selected_by_outcome_blind_record_predicate_top_k",
                    "permutation_p": float(permutation_p),
                    "permutation_count": int(permutation_count),
                    "permutation_method": permutation_method,
                    **uncertainty,
                }
            )

        uncertainty_df = pd.DataFrame(uncertainty_rows)

        # All tests, including non-resampled tests with p=1, remain in each BH
        # family and in the global audit denominator.
        contrasts["permutation_q_record_predicate"] = np.nan
        for _, idx in contrasts.groupby(["record_id", "failure_predicate"]).groups.items():
            contrasts.loc[idx, "permutation_q_record_predicate"] = bh_adjust(
                contrasts.loc[idx, "permutation_p"]
            )
        contrasts["permutation_q_global"] = bh_adjust(contrasts["permutation_p"])
        contrasts["cluster_sensitivity_pass"] = (
            contrasts["resampling_selected"].map(normalize_bool)
            & (pd.to_numeric(contrasts["bootstrap_ci_low"], errors="coerce") > 0)
            & (
                pd.to_numeric(contrasts["direction_consistency"], errors="coerce")
                >= args.min_direction_consistency
            )
        )
        contrasts["multiplicity_pass"] = (
            contrasts["resampling_selected"].map(normalize_bool)
            & (
                pd.to_numeric(contrasts["permutation_q_record_predicate"], errors="coerce")
                <= args.discovery_fdr
            )
        )

        # Controls are evaluated only after the target survives all preceding
        # discovery gates. This preserves the full search denominator while
        # avoiding control work for impossible candidates.
        control_rows: list[dict[str, Any]] = []
        control_aggs: list[dict[str, Any]] = []
        eligible_for_control = contrasts[
            contrasts["complement_admissible"].map(normalize_bool)
            & contrasts["predicate_semantics_pass"].map(normalize_bool)
            & contrasts["effect_pass"].map(normalize_bool)
            & contrasts["cluster_sensitivity_pass"].map(normalize_bool)
            & contrasts["multiplicity_pass"].map(normalize_bool)
        ]
        for _, target in eligible_for_control.iterrows():
            rows, agg = compute_control_adjustment(
                target,
                observation_by_record,
                controls,
                record_catalog,
                args,
            )
            control_rows.extend(rows)
            control_aggs.append(agg)
        control_rows_df = pd.DataFrame(control_rows)
        control_agg_df = pd.DataFrame(control_aggs)
        if not control_agg_df.empty:
            contrasts = contrasts.merge(
                control_agg_df,
                on=["record_id", "support_id", "failure_predicate"],
                how="left",
                validate="one_to_one",
            )

        for col, default in (
            ("control_robustness_pass", False),
            ("control_robustness_status", "not_evaluated"),
            ("median_control_adjusted_contrast", np.nan),
            ("positive_control_adjusted_share", np.nan),
            ("eligible_control_records_json", "[]"),
        ):
            if col not in contrasts.columns:
                contrasts[col] = default
            else:
                contrasts[col] = contrasts[col].fillna(default)

        def candidate_status(row: pd.Series) -> str:
            if not normalize_bool(row["complement_admissible"]):
                return str(row["initial_status"])
            if not normalize_bool(row["predicate_semantics_pass"]):
                return "discovery_signal_below_predicate_threshold"
            if not normalize_bool(row["effect_pass"]):
                return "discovery_signal_below_minimum_effect"
            if not normalize_bool(row.get("resampling_selected", False)):
                return str(row.get("resampling_selection_status", "not_resampled_predeclared_cap"))
            if not normalize_bool(row["cluster_sensitivity_pass"]):
                return "unstable_under_cluster_sensitivity"
            if not normalize_bool(row["multiplicity_pass"]):
                return "multiplicity_not_survived"
            if not normalize_bool(row.get("control_robustness_pass", False)):
                return str(row.get("control_robustness_status", "control_explained"))
            return (
                "fl2_candidate_nominated_confirmation_eligible"
                if normalize_bool(row["confirmation_eligible"])
                else "fl2_candidate_nominated_contrast_limited"
            )

        contrasts["candidate_status"] = contrasts.apply(candidate_status, axis=1)
        contrasts["fl_maturity"] = np.where(
            contrasts["candidate_status"].astype(str).str.startswith("fl2_candidate_nominated"),
            "FL2",
            "below_FL2",
        )

        multiplicity = contrasts[
            [
                "candidate_test_id", "record_id", "failure_predicate",
                "support_id", "support_families", "permutation_p",
                "permutation_q_record_predicate", "permutation_q_global",
                "resampling_selected", "resampling_selection_status",
                "multiplicity_pass", "candidate_status",
            ]
        ].copy()
        multiplicity["multiplicity_family"] = (
            multiplicity["record_id"].astype(str)
            + "::"
            + multiplicity["failure_predicate"].astype(str)
        )
        multiplicity["correction_rule"] = "Benjamini-Hochberg within record × predicate; global BH reported; non-resampled tests retained with p=1"
        multiplicity["candidate_denominator"] = len(contrasts)
        multiplicity["resampling_cap_per_record_predicate"] = cap

        nominated = contrasts[
            contrasts["candidate_status"].astype(str).str.startswith("fl2_candidate_nominated")
        ].copy()
        if not nominated.empty:
            nominated["candidate_id"] = nominated.apply(
                lambda r: "OBS084B-" + stable_hash(
                    {
                        "record_id": r["record_id"],
                        "failure_predicate": r["failure_predicate"],
                        "support_query_json": r["support_query_json"],
                        "metric": r["metric"],
                        "freeze_manifest_id": freeze_payload["freeze_manifest_id"],
                    }
                )[:20],
                axis=1,
            )
            minimal = assign_minimal_support_status(
                nominated,
                args.minimality_tolerance,
            )
        else:
            minimal = nominated

    # Seal only non-dominated nominated FL2 candidates.
    if minimal.empty:
        sealed = pd.DataFrame()
    else:
        sealed = minimal[
            (minimal["support_dominance_status"] == "non_dominated")
            & minimal["candidate_status"].astype(str).str.startswith("fl2_candidate_nominated")
        ].copy()

    confirmation_id = confirmation_partition_id(freeze_tables["partition"])
    source_artifacts = sorted(
        input_manifest.loc[input_manifest["exists"].map(normalize_bool), "artifact_path"]
        .astype(str)
        .unique()
        .tolist()
    )
    if not sealed.empty:
        sealed["control_admissibility_rule"] = "OBS-083 relation and carrier controls; same frozen support query; admissible class-balanced complement required"
        sealed["uncertainty_method"] = "leave-one-object-out sensitivity; object-cluster bootstrap; within-object-class permutation"
        sealed["resampling_unit"] = "object"
        sealed["confirmation_partition_id"] = confirmation_id
        sealed["multiplicity_family"] = sealed["record_id"].astype(str) + "::" + sealed["failure_predicate"].astype(str)
        sealed["exclusion_rules"] = "confirmation rows inaccessible; row_bootstrap_unit excluded; incomplete classes/complements/controls excluded; conjunction depth<=2"
        sealed["source_artifacts"] = json.dumps(source_artifacts)
        sealed["candidate_manifest_status"] = "sealed_FL2_discovery_candidates"
        sealed["material_change_rule"] = "any change to support, predicate, metric, complement, controls, threshold, exclusions, or confirmation partition returns candidate to unconfirmed FL2"

    manifest_columns = [
        "candidate_id", "record_id", "relation", "carrier", "subclass",
        "confirmation_eligible", "failure_predicate", "failure_mode",
        "support_definition", "support_query_json", "complement_definition",
        "matching_variables", "exposure_normalization",
        "control_admissibility_rule", "eligible_control_records_json",
        "metric", "expected_direction", "threshold_basis", "minimum_effect",
        "uncertainty_method", "resampling_unit", "confirmation_partition_id",
        "multiplicity_family", "exclusion_rules", "source_artifacts",
        "site_relative_contrast", "bootstrap_ci_low", "bootstrap_ci_high",
        "permutation_p", "permutation_q_record_predicate",
        "median_control_adjusted_contrast", "positive_control_adjusted_share",
        "candidate_status", "fl_maturity", "support_dominance_status",
        "candidate_manifest_status", "material_change_rule",
    ]
    sealed_manifest = (
        sealed[[c for c in manifest_columns if c in sealed.columns]].copy()
        if not sealed.empty
        else pd.DataFrame(columns=manifest_columns)
    )

    candidate_payload = {
        "schema": "obs084b_candidate_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "obs084a_freeze_manifest_id": freeze_payload["freeze_manifest_id"],
        "confirmation_partition_id": confirmation_id,
        "status": "sealed_FL2_candidates" if not sealed_manifest.empty else "sealed_null_discovery_result",
        "candidate_count": int(len(sealed_manifest)),
        "candidates": sealed_manifest.to_dict("records"),
        "search_configuration": {
            "model": args.model,
            "partition_role": "discovery_only",
            "cluster_unit": "object",
            "max_conjunction_depth": args.max_conjunction_depth,
            "minimum_rows": {
                "site": args.min_site_rows,
                "complement": args.min_complement_rows,
                "per_class": args.min_class_rows,
            },
            "minimum_clusters": {
                "site": args.min_site_clusters,
                "complement": args.min_complement_clusters,
                "shared": args.min_shared_clusters,
            },
            "cluster_bootstrap": args.n_cluster_bootstrap,
            "permutations": args.n_permutations,
            "max_resampled_tests_per_record_predicate": args.max_resampled_tests_per_record_predicate,
            "alpha": args.alpha,
            "discovery_fdr": args.discovery_fdr,
        },
    }
    candidate_manifest_id = stable_hash(candidate_payload)
    candidate_payload["candidate_manifest_id"] = candidate_manifest_id

    failures_df = pd.DataFrame(failure_rows)
    status = (
        "fl2_candidates_sealed_for_reserved_confirmation"
        if not sealed_manifest.empty
        else "valid_null_no_fl2_candidate_survived_discovery_rules"
    )

    # Write artifacts.
    output_tables: dict[str, pd.DataFrame] = {
        "obs084b_input_manifest.csv": input_manifest,
        "obs084b_discovery_observation_losses.csv": observation_losses,
        "obs084b_support_thresholds.csv": support_thresholds,
        "obs084b_support_vocabulary_execution_audit.csv": support_audit,
        "obs084b_support_candidate_inventory.csv": inventory,
        "obs084b_support_complement_matching.csv": matching,
        "obs084b_site_relative_contrasts.csv": contrasts,
        "obs084b_control_adjusted_contrasts.csv": control_rows_df,
        "obs084b_cluster_uncertainty.csv": uncertainty_df,
        "obs084b_minimal_support_families.csv": minimal,
        "obs084b_multiplicity_audit.csv": multiplicity,
        "obs084b_candidate_freeze_manifest.csv": sealed_manifest,
        "obs084b_discovery_failures.csv": failures_df,
    }
    for name, df in output_tables.items():
        df.to_csv(output_dir / name, index=False)
    (output_dir / "obs084b_candidate_freeze_manifest.json").write_text(
        json.dumps(candidate_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            {
                "script_version": SCRIPT_VERSION,
                "overall_status": status,
                "obs084a_freeze_manifest_id": freeze_payload["freeze_manifest_id"],
                "candidate_manifest_id": candidate_manifest_id,
                "registry_records": len(record_catalog),
                "records_with_observation_losses": int(observation_losses["record_id"].nunique()) if not observation_losses.empty else 0,
                "discovery_observation_loss_rows": len(observation_losses),
                "discovery_unique_observations": int(observation_losses["observation_id"].nunique()) if not observation_losses.empty else 0,
                "support_addresses_generated": len(inventory),
                "unique_support_definitions": int(inventory["support_id"].nunique()) if not inventory.empty else 0,
                "predicate_tests": len(contrasts),
                "admissible_complement_tests": int(contrasts.get("complement_admissible", False).map(normalize_bool).sum()) if not contrasts.empty else 0,
                "nominated_candidates_before_minimality": int(contrasts.get("candidate_status", "").astype(str).str.startswith("fl2_candidate_nominated").sum()) if not contrasts.empty else 0,
                "sealed_non_dominated_fl2_candidates": len(sealed_manifest),
                "confirmation_rows_read_into_analytical_frame": 0,
                "current_repo_commit": git_commit(repo_root),
            }
        ]
    )
    summary.to_csv(output_dir / "obs084b_discovery_summary.csv", index=False)

    write_report(
        output_dir / "obs084b_discovery_report.md",
        status,
        freeze_payload,
        input_manifest,
        record_catalog,
        observation_losses,
        support_thresholds,
        support_audit,
        inventory,
        contrasts,
        control_agg_df,
        minimal,
        multiplicity,
        failures_df,
        candidate_manifest_id,
    )

    print(f"OBS-084b discovery complete: {status}")
    print(f"Records modeled: {observation_losses['record_id'].nunique() if not observation_losses.empty else 0}/{len(record_catalog)}")
    print(f"Predicate-indexed tests: {len(contrasts)}")
    print(f"Sealed non-dominated FL2 candidates: {len(sealed_manifest)}")
    print(f"Candidate manifest ID: {candidate_manifest_id}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
