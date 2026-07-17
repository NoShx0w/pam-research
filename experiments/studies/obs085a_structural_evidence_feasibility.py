#!/usr/bin/env python3
"""
obs085a_structural_evidence_feasibility.py

OBS-085a — Structural Evidence Feasibility
==========================================

Purpose
-------
Construct the deterministic structural operating-envelope audit required by the
OBS-085 v2 protocol, using the completed OBS-084 study as an immutable input.

This stage answers a metrological question only:

    Which frozen OBS-084 support addresses contain enough support, complement,
    object-cluster, class, matching, control, joint target-control, outcome, and
    multiplicity structure to be statistically evaluated?

The script:
* invokes the authoritative OBS-084c ``--validate-only`` path before analysis;
* reads, but never rewrites or regenerates, OBS-084 artifacts;
* reconstructs the exact 5,736 predicate-indexed OBS-084b address family;
* verifies the discovery structural checks against the frozen OBS-084b matching
  audit before producing any OBS-085a result;
* evaluates the same frozen support addresses on discovery and confirmation
  observation-loss frames without performing a new candidate search;
* treats ``object`` / ``cluster_id`` as the independent evidence unit;
* audits relation and carrier controls under the same frozen support query;
* separates deterministic evidence feasibility G1–G10 from deterministic FL3
  claim entitlement E1;
* produces file-first CSV, JSON, and Markdown outputs under a new OBS-085 path.

This script does NOT:
* rerun OBS-084 discovery or confirmation;
* fit new diagnostic classifiers;
* alter support predicates, thresholds, candidates, or multiplicity rules;
* inject synthetic effects;
* estimate simulated gate-passage probability or power;
* compute observed power;
* promote any record or candidate to FL3;
* identify causal origins, repairs, interventions, or actionability.

Default inputs
--------------
outputs/rig_registry/obs084_direct_failure_witness/
  bridge_resolution/obs084a_freeze_manifest.json
  discovery/obs084b_support_candidate_inventory.csv
  discovery/obs084b_support_complement_matching.csv
  discovery/obs084b_discovery_observation_losses.csv
  discovery/obs084b_candidate_freeze_manifest.csv
  discovery/obs084b_candidate_freeze_manifest.json
  discovery/obs084c_confirmation_opening_lock.json
  confirmation/obs084c_confirmation_observation_losses.csv
  confirmation/obs084c_support_complement_validation.csv
  confirmation/obs084c_candidate_outcomes.csv
  confirmation/obs084c_confirmation_manifest.json

outputs/rig_registry/obs083_negative_control_localization/
  obs083_diagnostic_subclass_assignments.csv
  obs083_relation_control_contrast.csv
  obs083_carrier_control_contrast.csv

outputs/rig_registry/rig_relation_registry.csv

Default outputs
---------------
outputs/rig_registry/obs085_detection_envelope/obs085a_structural_feasibility/
  obs085a_input_manifest.csv
  obs085a_support_address_inventory.csv
  obs085a_support_coverage_matrix.csv
  obs085a_effective_evidence.csv
  obs085a_complement_admissibility.csv
  obs085a_control_availability.csv
  obs085a_joint_target_control_estimability.csv
  obs085a_structural_gate_matrix.csv
  obs085a_evidence_feasibility.csv
  obs085a_claim_entitlement_overlay.csv
  obs085a_detection_envelope_summary.csv
  obs085a_structural_state_sankey_links.csv
  obs085a_structural_state_sankey.html
  obs085a_failures.csv
  obs085a_manifest.json
  obs085a_report.md

Canonical guardrail
-------------------
An OBS-085a address may be evidence-feasible while remaining FL3-entitlement
capped. Claim entitlement is an epistemic ceiling, not a structural gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.1"
SCHEMA_VERSION = "obs085a_structural_evidence_feasibility_v1"

DEFAULT_OBS084_ROOT = Path(
    "outputs/rig_registry/obs084_direct_failure_witness"
)
DEFAULT_OBS083_DIR = Path(
    "outputs/rig_registry/obs083_negative_control_localization"
)
DEFAULT_REGISTRY_PATH = Path("outputs/rig_registry/rig_relation_registry.csv")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs085_detection_envelope/"
    "obs085a_structural_feasibility"
)
DEFAULT_OBS084C_SCRIPT = Path(
    "experiments/studies/obs084c_direct_failure_support_confirmation.py"
)
DEFAULT_PROTOCOL_PATH = Path(
    "docs/05_project/"
    "085_failure_support_detection_power_and_confirmation_feasibility_protocol.md"
)

DISCOVERY_FILES = {
    "support_inventory": "obs084b_support_candidate_inventory.csv",
    "support_matching": "obs084b_support_complement_matching.csv",
    "observation_losses": "obs084b_discovery_observation_losses.csv",
    "candidate_manifest_csv": "obs084b_candidate_freeze_manifest.csv",
    "candidate_manifest_json": "obs084b_candidate_freeze_manifest.json",
    "opening_lock": "obs084c_confirmation_opening_lock.json",
}

CONFIRMATION_FILES = {
    "observation_losses": "obs084c_confirmation_observation_losses.csv",
    "support_validation": "obs084c_support_complement_validation.csv",
    "candidate_outcomes": "obs084c_candidate_outcomes.csv",
    "confirmation_manifest": "obs084c_confirmation_manifest.json",
}

OBS083_FILES = {
    "subclasses": "obs083_diagnostic_subclass_assignments.csv",
    "relation_controls": "obs083_relation_control_contrast.csv",
    "carrier_controls": "obs083_carrier_control_contrast.csv",
}

PREDICATE_METRICS = {
    "relation_separation_attenuation": "margin_loss",
    "local_criterion_breach": "misclassification_loss",
    "log_loss_attenuation": "log_loss",
    "measurement_missingness_concentration": "predictor_missing_any",
}

GATE_NAMES = {
    "g1_support_presence": "G1 support presence",
    "g2_complement_presence": "G2 complement presence",
    "g3_support_cluster_coverage": "G3 support-cluster coverage",
    "g4_complement_cluster_coverage": "G4 complement-cluster coverage",
    "g5_class_bearing_coverage": "G5 class-bearing coverage",
    "g6_matched_complement_admissibility": "G6 matched-complement admissibility",
    "g7_control_availability": "G7 control availability",
    "g8_joint_target_control_estimability": "G8 joint target-control estimability",
    "g9_outcome_estimability": "G9 outcome estimability",
    "g10_multiplicity_family_definition": "G10 multiplicity-family definition",
}

GATE_COLUMNS = tuple(GATE_NAMES)


@dataclass(frozen=True)
class Failure:
    stage: str
    scope_id: str
    reason: str
    detail: str = ""
    severity: str = "error"


@dataclass(frozen=True)
class SupportCondition:
    family: str
    column: str
    value: str


@dataclass(frozen=True)
class SupportQuery:
    support_id: str
    definition: str
    conditions: tuple[SupportCondition, ...]


# -----------------------------------------------------------------------------
# CLI and general utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--obs084-root", type=Path, default=DEFAULT_OBS084_ROOT)
    p.add_argument("--obs083-dir", type=Path, default=DEFAULT_OBS083_DIR)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--obs084c-script", type=Path, default=DEFAULT_OBS084C_SCRIPT)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)

    # These defaults reproduce the frozen OBS-084b structural contract.
    p.add_argument("--min-site-rows", type=int, default=8)
    p.add_argument("--min-complement-rows", type=int, default=12)
    p.add_argument("--min-class-rows", type=int, default=2)
    p.add_argument("--min-site-clusters", type=int, default=2)
    p.add_argument("--min-complement-clusters", type=int, default=2)
    p.add_argument("--min-shared-clusters", type=int, default=2)
    p.add_argument(
        "--min-joint-target-control-clusters",
        type=int,
        default=2,
        help=(
            "Minimum object clusters jointly contributing target and control "
            "support/complement structure for G8."
        ),
    )

    p.add_argument(
        "--require-repo-commit",
        action="store_true",
        help=(
            "Pass the stricter current-commit check to OBS-084c validation. "
            "Source and artifact hashes remain mandatory regardless."
        ),
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the immutable OBS-084 lineage and the reconstructed "
            "discovery structural contract, then exit without writing outputs."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of an existing OBS-085a output bundle.",
    )
    p.add_argument("--max-report-rows", type=int, default=30)
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def repo_relative_path(path: str | Path, repo_root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return candidate.as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
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


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "pass",
        "ok",
    }


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    context: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lowered = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No rows._"
    shown = df.head(max_rows)
    try:
        return shown.to_markdown(index=False)
    except Exception:
        return "```text\n" + shown.to_string(index=False) + "\n```"


def json_list(values: Iterable[Any]) -> str:
    return json.dumps(sorted({str(v) for v in values if str(v)}))


def structural_state_counts(
    evidence_feasibility: pd.DataFrame,
) -> dict[str, int]:
    """Return mutually exclusive cross-partition structural-state counts."""
    required = [
        "discovery_evidence_feasible",
        "confirmation_evidence_feasible",
        "e1_fl3_claim_entitlement",
    ]
    require_columns(
        evidence_feasibility,
        required,
        "OBS-085a evidence feasibility for Sankey",
    )

    discovery = evidence_feasibility[
        "discovery_evidence_feasible"
    ].map(normalize_bool)
    confirmation = evidence_feasibility[
        "confirmation_evidence_feasible"
    ].map(normalize_bool)
    entitled = evidence_feasibility[
        "e1_fl3_claim_entitlement"
    ].map(normalize_bool)

    both = discovery & confirmation
    discovery_only = discovery & ~confirmation
    confirmation_only = ~discovery & confirmation
    neither = ~discovery & ~confirmation
    both_entitled = both & entitled
    both_capped = both & ~entitled

    counts = {
        "all_addresses": int(len(evidence_feasibility)),
        "discovery_feasible": int(discovery.sum()),
        "discovery_infeasible": int((~discovery).sum()),
        "both_feasible": int(both.sum()),
        "discovery_only": int(discovery_only.sum()),
        "confirmation_only": int(confirmation_only.sum()),
        "infeasible_both": int(neither.sum()),
        "both_feasible_fl3_entitled": int(both_entitled.sum()),
        "both_feasible_entitlement_capped": int(both_capped.sum()),
    }

    total = counts["all_addresses"]
    identities = {
        "discovery_partition": (
            counts["discovery_feasible"]
            + counts["discovery_infeasible"]
        ),
        "cross_partition": (
            counts["both_feasible"]
            + counts["discovery_only"]
            + counts["confirmation_only"]
            + counts["infeasible_both"]
        ),
        "discovery_feasible_split": (
            counts["both_feasible"] + counts["discovery_only"]
        ),
        "discovery_infeasible_split": (
            counts["confirmation_only"] + counts["infeasible_both"]
        ),
        "both_feasible_entitlement_split": (
            counts["both_feasible_fl3_entitled"]
            + counts["both_feasible_entitlement_capped"]
        ),
    }
    expected = {
        "discovery_partition": total,
        "cross_partition": total,
        "discovery_feasible_split": counts["discovery_feasible"],
        "discovery_infeasible_split": counts["discovery_infeasible"],
        "both_feasible_entitlement_split": counts["both_feasible"],
    }
    mismatches = {
        key: {"observed": identities[key], "expected": expected[key]}
        for key in identities
        if identities[key] != expected[key]
    }
    if mismatches:
        raise RuntimeError(
            "Structural-state Sankey identities failed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return counts


def build_structural_state_sankey_links(
    evidence_feasibility: pd.DataFrame,
) -> pd.DataFrame:
    """Build the audited link table used by the structural-state Sankey."""
    counts = structural_state_counts(evidence_feasibility)
    total = counts["all_addresses"]
    rows = [
        (
            "all_addresses",
            "All addresses",
            "discovery_feasible",
            "Discovery feasible",
            counts["discovery_feasible"],
            "discovery structural classification",
        ),
        (
            "all_addresses",
            "All addresses",
            "discovery_infeasible",
            "Discovery infeasible",
            counts["discovery_infeasible"],
            "discovery structural classification",
        ),
        (
            "discovery_feasible",
            "Discovery feasible",
            "both_feasible",
            "Feasible in both partitions",
            counts["both_feasible"],
            "cross-partition structural state",
        ),
        (
            "discovery_feasible",
            "Discovery feasible",
            "discovery_only",
            "Discovery feasible only",
            counts["discovery_only"],
            "cross-partition structural state",
        ),
        (
            "discovery_infeasible",
            "Discovery infeasible",
            "confirmation_only",
            "Confirmation feasible only",
            counts["confirmation_only"],
            "cross-partition structural state",
        ),
        (
            "discovery_infeasible",
            "Discovery infeasible",
            "infeasible_both",
            "Infeasible in both partitions",
            counts["infeasible_both"],
            "cross-partition structural state",
        ),
        (
            "both_feasible",
            "Feasible in both partitions",
            "both_feasible_fl3_entitled",
            "Both feasible and FL3-entitled",
            counts["both_feasible_fl3_entitled"],
            "deterministic entitlement overlay",
        ),
        (
            "both_feasible",
            "Feasible in both partitions",
            "both_feasible_entitlement_capped",
            "Both feasible and entitlement capped",
            counts["both_feasible_entitlement_capped"],
            "deterministic entitlement overlay",
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "source_id",
            "source_label",
            "target_id",
            "target_label",
            "value",
            "flow_semantics",
        ],
    )
    frame["fraction_of_universe"] = [
        safe_ratio(float(value), float(total))
        for value in frame["value"]
    ]
    return frame


def write_structural_state_sankey(
    path: Path,
    evidence_feasibility: pd.DataFrame,
    links: pd.DataFrame,
) -> None:
    """Write a dependency-free deterministic SVG Sankey as standalone HTML."""
    counts = structural_state_counts(evidence_feasibility)
    total = counts["all_addresses"]
    if total <= 0:
        raise ValueError("Cannot render structural-state Sankey with zero rows")

    expected_links = build_structural_state_sankey_links(
        evidence_feasibility
    )
    identity_columns = ["source_id", "target_id", "value"]
    observed_identity = links[identity_columns].reset_index(drop=True)
    expected_identity = expected_links[identity_columns].reset_index(drop=True)
    if not observed_identity.equals(expected_identity):
        raise RuntimeError(
            "Sankey HTML link identity does not match the audited link table"
        )

    width = 1320
    height = 840
    top = 120.0
    available = 620.0
    gap = 24.0
    node_width = 24.0
    scale = (available - 3.0 * gap) / float(total)

    h = {key: float(value) * scale for key, value in counts.items()}
    x = {
        "all_addresses": 70.0,
        "discovery_feasible": 340.0,
        "discovery_infeasible": 340.0,
        "both_feasible": 650.0,
        "discovery_only": 650.0,
        "confirmation_only": 650.0,
        "infeasible_both": 650.0,
        "both_feasible_fl3_entitled": 1010.0,
        "both_feasible_entitlement_capped": 1010.0,
    }
    y = {
        "all_addresses": top,
        "discovery_feasible": top,
        "discovery_infeasible": top + h["discovery_feasible"] + gap,
        "both_feasible": top,
        "discovery_only": top + h["both_feasible"] + gap,
        "confirmation_only": (
            top + h["discovery_feasible"] + 2.0 * gap
        ),
        "infeasible_both": (
            top
            + h["discovery_feasible"]
            + h["confirmation_only"]
            + 3.0 * gap
        ),
        "both_feasible_fl3_entitled": top,
        "both_feasible_entitlement_capped": (
            top + h["both_feasible_fl3_entitled"]
        ),
    }
    labels = {
        "all_addresses": "All addresses",
        "discovery_feasible": "Discovery feasible",
        "discovery_infeasible": "Discovery infeasible",
        "both_feasible": "Feasible in both partitions",
        "discovery_only": "Discovery feasible only",
        "confirmation_only": "Confirmation feasible only",
        "infeasible_both": "Infeasible in both partitions",
        "both_feasible_fl3_entitled": "Both feasible + FL3-entitled",
        "both_feasible_entitlement_capped": (
            "Both feasible + entitlement capped"
        ),
    }
    node_colors = {
        "all_addresses": "#334155",
        "discovery_feasible": "#2563eb",
        "discovery_infeasible": "#64748b",
        "both_feasible": "#0f766e",
        "discovery_only": "#3b82f6",
        "confirmation_only": "#7c3aed",
        "infeasible_both": "#64748b",
        "both_feasible_fl3_entitled": "#15803d",
        "both_feasible_entitlement_capped": "#b45309",
    }

    link_specs = [
        (
            "all_addresses",
            "discovery_feasible",
            counts["discovery_feasible"],
            0,
            0,
            "#2563eb",
        ),
        (
            "all_addresses",
            "discovery_infeasible",
            counts["discovery_infeasible"],
            counts["discovery_feasible"],
            0,
            "#64748b",
        ),
        (
            "discovery_feasible",
            "both_feasible",
            counts["both_feasible"],
            0,
            0,
            "#0f766e",
        ),
        (
            "discovery_feasible",
            "discovery_only",
            counts["discovery_only"],
            counts["both_feasible"],
            0,
            "#3b82f6",
        ),
        (
            "discovery_infeasible",
            "confirmation_only",
            counts["confirmation_only"],
            0,
            0,
            "#7c3aed",
        ),
        (
            "discovery_infeasible",
            "infeasible_both",
            counts["infeasible_both"],
            counts["confirmation_only"],
            0,
            "#64748b",
        ),
        (
            "both_feasible",
            "both_feasible_fl3_entitled",
            counts["both_feasible_fl3_entitled"],
            0,
            0,
            "#15803d",
        ),
        (
            "both_feasible",
            "both_feasible_entitlement_capped",
            counts["both_feasible_entitlement_capped"],
            counts["both_feasible_fl3_entitled"],
            0,
            "#b45309",
        ),
    ]

    svg: list[str] = []
    for source, target, value, source_offset, target_offset, color in link_specs:
        if value <= 0:
            continue
        x1 = x[source] + node_width
        x2 = x[target]
        y1 = y[source] + (float(source_offset) + float(value) / 2.0) * scale
        y2 = y[target] + (float(target_offset) + float(value) / 2.0) * scale
        mid = (x1 + x2) / 2.0
        stroke_width = max(1.0, float(value) * scale)
        tooltip = (
            f"{labels[source]} → {labels[target]}: "
            f"{int(value):,} ({100.0 * value / total:.2f}% of universe)"
        )
        svg.append(
            f'<path class="flow" d="M {x1:.2f} {y1:.2f} '
            f'C {mid:.2f} {y1:.2f}, {mid:.2f} {y2:.2f}, '
            f'{x2:.2f} {y2:.2f}" stroke="{color}" '
            f'stroke-width="{stroke_width:.2f}"><title>{tooltip}</title></path>'
        )

    node_order = [
        "all_addresses",
        "discovery_feasible",
        "discovery_infeasible",
        "both_feasible",
        "discovery_only",
        "confirmation_only",
        "infeasible_both",
        "both_feasible_fl3_entitled",
        "both_feasible_entitlement_capped",
    ]
    for node in node_order:
        value = counts[node]
        node_height = max(1.0, h[node])
        tooltip = (
            f"{labels[node]}: {value:,} "
            f"({100.0 * value / total:.2f}% of universe)"
        )
        svg.append(
            f'<rect class="node" x="{x[node]:.2f}" y="{y[node]:.2f}" '
            f'width="{node_width:.2f}" height="{node_height:.2f}" '
            f'fill="{node_colors[node]}"><title>{tooltip}</title></rect>'
        )
        label_x = x[node]
        label_y = max(94.0, y[node] - 10.0)
        svg.append(
            f'<text class="node-label" x="{label_x:.2f}" y="{label_y:.2f}">'
            f'{labels[node]}</text>'
        )
        svg.append(
            f'<text class="node-value" x="{label_x:.2f}" '
            f'y="{label_y + 18.0:.2f}">{value:,} '
            f'({100.0 * value / total:.1f}%)</text>'
        )

    terminal_rows = [
        (
            "Feasible in both partitions",
            counts["both_feasible"],
        ),
        (
            "Discovery feasible only",
            counts["discovery_only"],
        ),
        (
            "Confirmation feasible only",
            counts["confirmation_only"],
        ),
        (
            "Infeasible in both partitions",
            counts["infeasible_both"],
        ),
    ]
    table_rows = "\n".join(
        "<tr>"
        f"<td>{label}</td>"
        f"<td>{value:,}</td>"
        f"<td>{100.0 * value / total:.2f}%</td>"
        "</tr>"
        for label, value in terminal_rows
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OBS-085a structural-state Sankey</title>
<style>
:root {{ color-scheme: light dark; }}
body {{
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  background: #f8fafc;
  color: #0f172a;
}}
main {{ max-width: 1420px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
p {{ max-width: 1050px; line-height: 1.55; color: #334155; }}
.figure {{
  margin-top: 24px;
  overflow-x: auto;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgb(15 23 42 / 0.06);
}}
svg {{ display: block; min-width: 1180px; width: 100%; height: auto; }}
.flow {{ fill: none; opacity: 0.42; }}
.flow:hover {{ opacity: 0.72; }}
.node {{ rx: 3; ry: 3; stroke: #ffffff; stroke-width: 1; }}
.node-label {{ font-size: 14px; font-weight: 650; fill: #0f172a; }}
.node-value {{ font-size: 12px; fill: #475569; }}
.column-title {{ font-size: 13px; font-weight: 700; fill: #475569; }}
.audit {{ margin-top: 24px; max-width: 760px; }}
table {{ border-collapse: collapse; width: 100%; background: #ffffff; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 9px 12px; text-align: left; }}
th {{ font-size: 13px; color: #475569; }}
td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3) {{
  text-align: right;
}}
.note {{ font-size: 14px; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #0f172a; color: #e2e8f0; }}
  p, .note {{ color: #cbd5e1; }}
  .figure, table {{ background: #111827; border-color: #334155; }}
  .node-label {{ fill: #f1f5f9; }}
  .node-value, .column-title {{ fill: #cbd5e1; }}
  th, td {{ border-color: #334155; }}
  th {{ color: #cbd5e1; }}
}}
</style>
</head>
<body>
<main>
<h1>OBS-085a structural-state Sankey</h1>
<p>
Cross-partition structural classification of all predicate-indexed addresses.
Flows represent mutually exclusive feasibility states and a deterministic
claim-entitlement overlay. They do not represent sequential gate passage,
causal attrition, effect existence, or simulated detection probability.
</p>
<div class="figure">
<svg viewBox="0 0 {width} {height}" role="img"
  aria-label="OBS-085a cross-partition structural-state Sankey">
<text class="column-title" x="70" y="48">Address universe</text>
<text class="column-title" x="340" y="48">Discovery classification</text>
<text class="column-title" x="650" y="48">Cross-partition state</text>
<text class="column-title" x="1010" y="48">Entitlement overlay for both-feasible addresses</text>
{''.join(svg)}
</svg>
</div>
<section class="audit">
<h2>Mutually exclusive terminal states</h2>
<table>
<thead><tr><th>State</th><th>Addresses</th><th>Universe share</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<p class="note">
The authoritative machine-readable flow values are stored in
<code>obs085a_structural_state_sankey_links.csv</code>.
</p>
</section>
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def max_share(counts: Mapping[str, int]) -> float:
    total = int(sum(counts.values()))
    if total <= 0:
        return np.nan
    return float(max(counts.values(), default=0) / total)


def as_failure_frame(failures: Sequence[Failure]) -> pd.DataFrame:
    columns = ["stage", "scope_id", "reason", "detail", "severity"]
    if not failures:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([f.__dict__ for f in failures], columns=columns)


# -----------------------------------------------------------------------------
# Input paths and authoritative OBS-084 validation
# -----------------------------------------------------------------------------


def build_input_paths(
    repo_root: Path,
    obs084_root: Path,
    obs083_dir: Path,
    registry_path: Path,
    obs084c_script: Path,
    protocol_path: Path,
) -> dict[str, Path]:
    bridge_dir = obs084_root / "bridge_resolution"
    discovery_dir = obs084_root / "discovery"
    confirmation_dir = obs084_root / "confirmation"

    paths: dict[str, Path] = {
        "obs084a_freeze_manifest": bridge_dir / "obs084a_freeze_manifest.json",
        "rig_registry": registry_path,
        "obs084c_validation_script": obs084c_script,
        "obs085_protocol": protocol_path,
    }
    paths.update(
        {
            f"obs084b_{role}": discovery_dir / filename
            for role, filename in DISCOVERY_FILES.items()
        }
    )
    paths.update(
        {
            f"obs084c_{role}": confirmation_dir / filename
            for role, filename in CONFIRMATION_FILES.items()
        }
    )
    paths.update(
        {
            f"obs083_{role}": obs083_dir / filename
            for role, filename in OBS083_FILES.items()
        }
    )

    # Normalize only the dictionary values. Paths remain absolute internally.
    return {role: path.resolve() for role, path in paths.items()}


def require_input_files(paths: Mapping[str, Path]) -> None:
    missing = [role for role, path in paths.items() if not path.is_file()]
    if missing:
        detail = {role: str(paths[role]) for role in missing}
        raise FileNotFoundError(f"Required OBS-085a inputs are missing: {detail}")


def run_obs084c_validation(
    repo_root: Path,
    obs084c_script: Path,
    require_repo_commit: bool,
) -> dict[str, Any]:
    command = [sys.executable, str(obs084c_script), "--validate-only"]
    if require_repo_commit:
        command.append("--require-repo-commit")

    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    if completed.returncode != 0:
        raise RuntimeError(
            "Authoritative OBS-084c validation failed. "
            f"returncode={completed.returncode}; stdout={output!r}; "
            f"stderr={error!r}"
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": output,
        "stderr": error,
    }


def extract_identity(payload: Mapping[str, Any], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        value = payload.get(candidate)
        if value not in (None, ""):
            return str(value)
    return ""


def build_input_manifest(
    repo_root: Path,
    paths: Mapping[str, Path],
    upstream_validation: Mapping[str, Any],
) -> pd.DataFrame:
    freeze_payload = read_json_required(paths["obs084a_freeze_manifest"])
    candidate_payload = read_json_required(paths["obs084b_candidate_manifest_json"])
    confirmation_payload = read_json_required(paths["obs084c_confirmation_manifest"])
    lock_payload = read_json_required(paths["obs084b_opening_lock"])

    identities = {
        "obs084a_freeze_manifest_id": extract_identity(
            freeze_payload,
            ("freeze_manifest_id", "manifest_id"),
        ),
        "obs084b_candidate_manifest_id": extract_identity(
            candidate_payload,
            ("candidate_manifest_id", "manifest_id"),
        ),
        "obs084c_confirmation_manifest_id": extract_identity(
            confirmation_payload,
            ("confirmation_manifest_id", "manifest_id"),
        ),
        "obs084c_opening_lock_id": extract_identity(
            lock_payload,
            ("opening_lock_id", "lock_id", "confirmation_partition_id"),
        ),
    }

    rows: list[dict[str, Any]] = []
    for role, path in sorted(paths.items()):
        rows.append(
            {
                "artifact_role": role,
                "artifact_path": repo_relative_path(path, repo_root),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else np.nan,
                "sha256": sha256_file(path) if path.is_file() else "",
                **identities,
                "obs084c_validation_returncode": upstream_validation["returncode"],
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Frozen catalogs and support-query parsing
# -----------------------------------------------------------------------------


def parse_relation_classes(relation: str) -> tuple[str, ...]:
    relation = str(relation)
    if relation == "three_way":
        return ("C", "Cp2", "Cp3")
    if "_vs_" in relation:
        left, right = relation.split("_vs_", 1)
        return (left, right)
    raise ValueError(f"Unsupported frozen relation: {relation!r}")


def parse_support_query(
    support_id: str,
    definition: str,
    raw_query: Any,
) -> SupportQuery:
    try:
        payload = json.loads(str(raw_query))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid support_query_json for {support_id}: {exc}"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Support {support_id} has an empty support query")
    if len(payload) > 2:
        raise ValueError(
            f"Support {support_id} exceeds frozen conjunction depth two"
        )

    conditions: list[SupportCondition] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Support {support_id} contains a non-object condition")
        if str(item.get("operator", "")) != "eq":
            raise ValueError(
                f"Support {support_id} uses non-frozen operator "
                f"{item.get('operator')!r}"
            )
        family = str(item.get("support_family", "")).strip()
        column = str(item.get("column", "")).strip()
        value = str(item.get("value", ""))
        if not family or not column:
            raise ValueError(
                f"Support {support_id} has an empty family or column"
            )
        conditions.append(SupportCondition(family, column, value))

    return SupportQuery(
        support_id=str(support_id),
        definition=str(definition),
        conditions=tuple(conditions),
    )


def support_mask(df: pd.DataFrame, query: SupportQuery) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    for condition in query.conditions:
        if condition.column not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        mask &= df[condition.column].astype(str) == condition.value
    return mask


def load_record_catalog(
    registry: pd.DataFrame,
    subclasses: pd.DataFrame,
    support_inventory: pd.DataFrame,
) -> pd.DataFrame:
    rid = first_existing_column(registry, ("relation_id", "record_id"))
    relation = first_existing_column(registry, ("task", "relation"))
    carrier = first_existing_column(registry, ("carrier",))
    threshold = first_existing_column(registry, ("threshold",))
    if not rid or not relation or not carrier:
        raise ValueError("RIG registry lacks record, relation, or carrier fields")

    keep = [rid, relation, carrier] + ([threshold] if threshold else [])
    out = registry[keep].copy().rename(
        columns={rid: "record_id", relation: "relation", carrier: "carrier"}
    )
    if threshold and threshold in out.columns:
        out["threshold"] = pd.to_numeric(out[threshold], errors="coerce")
    else:
        out["threshold"] = np.nan

    require_columns(subclasses, ["record_id", "subclass"], "OBS-083 subclasses")
    sub_keep = [
        c
        for c in (
            "record_id",
            "subclass",
            "readiness_statement",
            "primary_limiter",
            "secondary_limiter",
            "failure_localization_score",
            "repair_specificity_level",
            "repair_specificity_score",
            "c4_evidence_gate_passed",
        )
        if c in subclasses.columns
    ]
    out = out.merge(
        subclasses[sub_keep].drop_duplicates("record_id"),
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    entitlement = (
        support_inventory[
            ["record_id", "confirmation_eligible"]
        ]
        .assign(
            confirmation_eligible=lambda x: x["confirmation_eligible"].map(
                normalize_bool
            )
        )
        .groupby("record_id", as_index=False)["confirmation_eligible"]
        .agg(lambda s: bool(s.all()) if len(s) else False)
    )
    out = out.merge(
        entitlement,
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    out["confirmation_eligible"] = (
        out["confirmation_eligible"]
        .where(out["confirmation_eligible"].notna(), False)
        .astype(bool)
    )
    out["e1_fl3_claim_entitlement"] = out["confirmation_eligible"].map(bool)
    out["entitlement_status"] = np.where(
        out["e1_fl3_claim_entitlement"],
        "fl3_entitled",
        "fl3_entitlement_capped",
    )
    return out.drop_duplicates("record_id").reset_index(drop=True)


def build_control_catalog(
    relation_controls: pd.DataFrame,
    carrier_controls: pd.DataFrame,
    known_records: set[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if not relation_controls.empty:
        require_columns(
            relation_controls,
            ["record_id", "control_record_id"],
            "OBS-083 relation controls",
        )
        evidence_col = first_existing_column(
            relation_controls,
            ("evidence_available",),
        )
        rel = pd.DataFrame(
            {
                "record_id": relation_controls["record_id"].astype(str),
                "control_record_id": relation_controls[
                    "control_record_id"
                ].astype(str),
                "control_family": "relation_control",
                "mapping_evidence_available": (
                    relation_controls[evidence_col].map(normalize_bool)
                    if evidence_col
                    else True
                ),
            }
        )
        rows.append(rel)

    if not carrier_controls.empty:
        require_columns(
            carrier_controls,
            ["record_id", "control_record_id"],
            "OBS-083 carrier controls",
        )
        evidence_col = first_existing_column(
            carrier_controls,
            ("evidence_available",),
        )
        car = pd.DataFrame(
            {
                "record_id": carrier_controls["record_id"].astype(str),
                "control_record_id": carrier_controls[
                    "control_record_id"
                ].astype(str),
                "control_family": "carrier_control",
                "mapping_evidence_available": (
                    carrier_controls[evidence_col].map(normalize_bool)
                    if evidence_col
                    else True
                ),
            }
        )
        rows.append(car)

    if not rows:
        return pd.DataFrame(
            columns=[
                "record_id",
                "control_record_id",
                "control_family",
                "mapping_evidence_available",
                "control_record_present",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    out = out[out["record_id"] != out["control_record_id"]].copy()
    out["control_record_present"] = out["control_record_id"].isin(known_records)
    return out.drop_duplicates(
        ["record_id", "control_record_id", "control_family"]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Structural evidence calculations
# -----------------------------------------------------------------------------


def count_by_class(df: pd.DataFrame, classes: Sequence[str]) -> dict[str, int]:
    counts = Counter(df["true_regime"].astype(str))
    return {str(cls): int(counts.get(str(cls), 0)) for cls in classes}


def count_finite_by_class(
    df: pd.DataFrame,
    classes: Sequence[str],
    metric: str,
) -> dict[str, int]:
    if metric not in df.columns:
        return {str(cls): 0 for cls in classes}
    numeric = pd.to_numeric(df[metric], errors="coerce")
    counts: dict[str, int] = {}
    for cls in classes:
        mask = df["true_regime"].astype(str) == str(cls)
        counts[str(cls)] = int(numeric.loc[mask].notna().sum())
    return counts


def class_bearing_cluster_count(
    df: pd.DataFrame,
    classes: Sequence[str],
) -> int:
    if df.empty:
        return 0
    count = 0
    for _, group in df.groupby("cluster_id", dropna=False):
        present = set(group["true_regime"].astype(str))
        if set(map(str, classes)).issubset(present):
            count += 1
    return count


def clusters_with_both_site_and_complement(
    df: pd.DataFrame,
    mask: pd.Series,
) -> set[str]:
    site = set(df.loc[mask, "cluster_id"].astype(str))
    complement = set(df.loc[~mask, "cluster_id"].astype(str))
    return site & complement


def metric_estimable_clusters(
    df: pd.DataFrame,
    mask: pd.Series,
    metric: str,
) -> set[str]:
    if metric not in df.columns:
        return set()
    numeric = pd.to_numeric(df[metric], errors="coerce")
    site_clusters = set(df.loc[mask & numeric.notna(), "cluster_id"].astype(str))
    comp_clusters = set(df.loc[(~mask) & numeric.notna(), "cluster_id"].astype(str))
    return site_clusters & comp_clusters


def compute_support_structure(
    record_df: pd.DataFrame,
    query: SupportQuery,
    relation: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    classes = parse_relation_classes(relation)
    columns_available = all(
        condition.column in record_df.columns for condition in query.conditions
    )
    mask = support_mask(record_df, query)
    site = record_df.loc[mask].copy()
    complement = record_df.loc[~mask].copy()

    site_clusters = set(site["cluster_id"].astype(str))
    complement_clusters = set(complement["cluster_id"].astype(str))
    shared_clusters = site_clusters & complement_clusters
    site_class_counts = count_by_class(site, classes)
    complement_class_counts = count_by_class(complement, classes)

    checks = {
        "site_rows": len(site) >= args.min_site_rows,
        "complement_rows": len(complement) >= args.min_complement_rows,
        "site_clusters": len(site_clusters) >= args.min_site_clusters,
        "complement_clusters": (
            len(complement_clusters) >= args.min_complement_clusters
        ),
        "shared_clusters": len(shared_clusters) >= args.min_shared_clusters,
        "site_class_support": all(
            site_class_counts[str(cls)] >= args.min_class_rows for cls in classes
        ),
        "complement_class_support": all(
            complement_class_counts[str(cls)] >= args.min_class_rows
            for cls in classes
        ),
    }
    complement_admissible = bool(columns_available and all(checks.values()))

    site_cluster_counts = Counter(site["cluster_id"].astype(str))
    comp_cluster_counts = Counter(complement["cluster_id"].astype(str))

    return {
        "support_columns_available": columns_available,
        "total_rows": int(len(record_df)),
        "unique_scientific_observations": int(
            record_df["observation_key"].astype(str).nunique()
        ),
        "total_clusters": int(record_df["cluster_id"].astype(str).nunique()),
        "n_site_rows": int(len(site)),
        "n_complement_rows": int(len(complement)),
        "n_site_observations": int(site["observation_key"].astype(str).nunique()),
        "n_complement_observations": int(
            complement["observation_key"].astype(str).nunique()
        ),
        "n_site_clusters": int(len(site_clusters)),
        "n_complement_clusters": int(len(complement_clusters)),
        "n_shared_clusters": int(len(shared_clusters)),
        "site_cluster_ids_json": json_list(site_clusters),
        "complement_cluster_ids_json": json_list(complement_clusters),
        "shared_cluster_ids_json": json_list(shared_clusters),
        "site_class_counts_json": json.dumps(site_class_counts, sort_keys=True),
        "complement_class_counts_json": json.dumps(
            complement_class_counts,
            sort_keys=True,
        ),
        "site_class_bearing_clusters": class_bearing_cluster_count(site, classes),
        "complement_class_bearing_clusters": class_bearing_cluster_count(
            complement,
            classes,
        ),
        "support_prevalence": safe_ratio(len(site), len(record_df)),
        "complement_prevalence": safe_ratio(len(complement), len(record_df)),
        "site_max_cluster_row_share": max_share(site_cluster_counts),
        "complement_max_cluster_row_share": max_share(comp_cluster_counts),
        "g1_support_presence": bool(len(site) > 0 and columns_available),
        "g2_complement_presence": bool(len(complement) > 0 and columns_available),
        "g3_support_cluster_coverage": bool(checks["site_clusters"]),
        "g4_complement_cluster_coverage": bool(checks["complement_clusters"]),
        "g5_class_bearing_coverage": bool(
            checks["site_class_support"]
            and checks["complement_class_support"]
        ),
        "g6_matched_complement_admissibility": complement_admissible,
        "matching_check_json": json.dumps(checks, sort_keys=True),
        "complement_admissible": complement_admissible,
    }


def compute_metric_structure(
    record_df: pd.DataFrame,
    query: SupportQuery,
    relation: str,
    metric: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    classes = parse_relation_classes(relation)
    mask = support_mask(record_df, query)
    site = record_df.loc[mask].copy()
    complement = record_df.loc[~mask].copy()

    site_counts = count_finite_by_class(site, classes, metric)
    complement_counts = count_finite_by_class(complement, classes, metric)
    metric_available = metric in record_df.columns
    site_metric_ok = metric_available and all(
        site_counts[str(cls)] >= args.min_class_rows for cls in classes
    )
    complement_metric_ok = metric_available and all(
        complement_counts[str(cls)] >= args.min_class_rows for cls in classes
    )
    finite_clusters = metric_estimable_clusters(record_df, mask, metric)

    return {
        "metric": metric,
        "metric_available": metric_available,
        "site_finite_metric_rows": int(
            pd.to_numeric(site.get(metric), errors="coerce").notna().sum()
        )
        if metric_available
        else 0,
        "complement_finite_metric_rows": int(
            pd.to_numeric(complement.get(metric), errors="coerce").notna().sum()
        )
        if metric_available
        else 0,
        "site_finite_class_counts_json": json.dumps(site_counts, sort_keys=True),
        "complement_finite_class_counts_json": json.dumps(
            complement_counts,
            sort_keys=True,
        ),
        "metric_estimable_cluster_count": int(len(finite_clusters)),
        "metric_estimable_cluster_ids_json": json_list(finite_clusters),
        "g9_outcome_estimability": bool(site_metric_ok and complement_metric_ok),
    }


def frozen_discovery_contract_validation(
    support_structures: pd.DataFrame,
    frozen_matching: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frozen_matching,
        [
            "record_id",
            "support_id",
            "n_site_rows",
            "n_complement_rows",
            "n_site_clusters",
            "n_complement_clusters",
            "n_shared_clusters",
            "complement_admissible",
            "matching_check_json",
        ],
        "OBS-084b support matching",
    )
    discovery = support_structures[
        support_structures["partition"] == "discovery"
    ].copy()
    discovery = discovery.drop_duplicates(["record_id", "support_id"])

    frozen = frozen_matching.drop_duplicates(["record_id", "support_id"])[
        [
            "record_id",
            "support_id",
            "n_site_rows",
            "n_complement_rows",
            "n_site_clusters",
            "n_complement_clusters",
            "n_shared_clusters",
            "complement_admissible",
            "matching_check_json",
        ]
    ].copy()
    frozen = frozen.rename(
        columns={
            c: f"frozen_{c}"
            for c in frozen.columns
            if c not in {"record_id", "support_id"}
        }
    )

    audit = discovery.merge(
        frozen,
        on=["record_id", "support_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    audit["row_match"] = audit["_merge"] == "both"

    numeric_fields = [
        "n_site_rows",
        "n_complement_rows",
        "n_site_clusters",
        "n_complement_clusters",
        "n_shared_clusters",
    ]
    for field in numeric_fields:
        audit[f"{field}_match"] = (
            pd.to_numeric(audit[field], errors="coerce")
            == pd.to_numeric(audit[f"frozen_{field}"], errors="coerce")
        )
    audit["complement_admissible_match"] = (
        audit["complement_admissible"].map(normalize_bool)
        == audit["frozen_complement_admissible"].map(normalize_bool)
    )
    audit["matching_check_match"] = (
        audit["matching_check_json"].fillna("").astype(str)
        == audit["frozen_matching_check_json"].fillna("").astype(str)
    )

    check_columns = [
        "row_match",
        *[f"{field}_match" for field in numeric_fields],
        "complement_admissible_match",
        "matching_check_match",
    ]
    audit["frozen_contract_match"] = audit[check_columns].all(axis=1)
    return audit


# -----------------------------------------------------------------------------
# Control and joint target-control feasibility
# -----------------------------------------------------------------------------


def parse_json_string_set(raw: Any) -> set[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return set()
    try:
        values = json.loads(str(raw))
    except json.JSONDecodeError:
        return set()
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def compute_control_feasibility(
    address_inventory: pd.DataFrame,
    partition_frames: Mapping[str, pd.DataFrame],
    record_catalog: pd.DataFrame,
    controls: pd.DataFrame,
    query_by_support: Mapping[str, SupportQuery],
    structure_lookup: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_by_record = record_catalog.set_index("record_id").to_dict("index")
    observations_by_partition_record: dict[tuple[str, str], pd.DataFrame] = {}
    for partition, frame in partition_frames.items():
        for record_id, group in frame.groupby("record_id", sort=False):
            observations_by_partition_record[(partition, str(record_id))] = group.copy()

    control_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for _, address in address_inventory.iterrows():
        address_id = str(address["address_id"])
        record_id = str(address["record_id"])
        support_id = str(address["support_id"])
        predicate = str(address["failure_predicate"])
        metric = str(address["metric"])
        query = query_by_support[support_id]
        mapped = controls[
            (controls["record_id"].astype(str) == record_id)
            & controls["mapping_evidence_available"].map(normalize_bool)
            & controls["control_record_present"].map(normalize_bool)
        ].copy()

        for partition in ("discovery", "confirmation"):
            target_structure = structure_lookup[(partition, record_id, support_id, predicate)]
            target_shared = parse_json_string_set(
                target_structure["shared_cluster_ids_json"]
            )
            target_metric_clusters = parse_json_string_set(
                target_structure.get("metric_estimable_cluster_ids_json", "[]")
            )
            target_joint_base = target_shared & target_metric_clusters

            admissible_count = 0
            joint_count = 0
            relation_mapped = 0
            carrier_mapped = 0
            relation_admissible = 0
            carrier_admissible = 0
            joint_cluster_counts: list[int] = []
            admissible_control_ids: list[str] = []
            joint_control_ids: list[str] = []

            for _, mapping in mapped.iterrows():
                control_id = str(mapping["control_record_id"])
                family = str(mapping["control_family"])
                if family == "relation_control":
                    relation_mapped += 1
                elif family == "carrier_control":
                    carrier_mapped += 1

                ckey = (partition, control_id, support_id, predicate)
                if ckey not in cache:
                    control_df = observations_by_partition_record.get(
                        (partition, control_id),
                        pd.DataFrame(),
                    )
                    control_catalog = catalog_by_record.get(control_id, {})
                    control_relation = str(control_catalog.get("relation", ""))
                    if control_df.empty or not control_relation:
                        cache[ckey] = {
                            "control_structure_available": False,
                            "control_complement_admissible": False,
                            "control_outcome_estimable": False,
                            "control_shared_cluster_ids_json": "[]",
                            "control_metric_cluster_ids_json": "[]",
                            "control_status": "control_observation_evidence_unavailable",
                        }
                    else:
                        generic = compute_support_structure(
                            control_df,
                            query,
                            control_relation,
                            args,
                        )
                        metric_part = compute_metric_structure(
                            control_df,
                            query,
                            control_relation,
                            metric,
                            args,
                        )
                        cache[ckey] = {
                            "control_structure_available": True,
                            "control_complement_admissible": generic[
                                "g6_matched_complement_admissibility"
                            ],
                            "control_outcome_estimable": metric_part[
                                "g9_outcome_estimability"
                            ],
                            "control_shared_cluster_ids_json": generic[
                                "shared_cluster_ids_json"
                            ],
                            "control_metric_cluster_ids_json": metric_part[
                                "metric_estimable_cluster_ids_json"
                            ],
                            "control_n_site_rows": generic["n_site_rows"],
                            "control_n_complement_rows": generic[
                                "n_complement_rows"
                            ],
                            "control_n_site_clusters": generic[
                                "n_site_clusters"
                            ],
                            "control_n_complement_clusters": generic[
                                "n_complement_clusters"
                            ],
                            "control_n_shared_clusters": generic[
                                "n_shared_clusters"
                            ],
                            "control_matching_check_json": generic[
                                "matching_check_json"
                            ],
                            "control_status": (
                                "admissible"
                                if generic[
                                    "g6_matched_complement_admissibility"
                                ]
                                and metric_part["g9_outcome_estimability"]
                                else "inadmissible_or_outcome_unestimable"
                            ),
                        }

                result = cache[ckey]
                control_shared = parse_json_string_set(
                    result.get("control_shared_cluster_ids_json", "[]")
                )
                control_metric = parse_json_string_set(
                    result.get("control_metric_cluster_ids_json", "[]")
                )
                control_joint_base = control_shared & control_metric
                joint_clusters = target_joint_base & control_joint_base
                admissible = bool(
                    result.get("control_complement_admissible", False)
                    and result.get("control_outcome_estimable", False)
                )
                jointly_estimable = bool(
                    admissible
                    and len(joint_clusters)
                    >= args.min_joint_target_control_clusters
                )

                if admissible:
                    admissible_count += 1
                    admissible_control_ids.append(control_id)
                    if family == "relation_control":
                        relation_admissible += 1
                    elif family == "carrier_control":
                        carrier_admissible += 1
                if jointly_estimable:
                    joint_count += 1
                    joint_control_ids.append(control_id)
                joint_cluster_counts.append(len(joint_clusters))

                control_rows.append(
                    {
                        "address_id": address_id,
                        "record_id": record_id,
                        "support_id": support_id,
                        "failure_predicate": predicate,
                        "metric": metric,
                        "partition": partition,
                        "control_record_id": control_id,
                        "control_family": family,
                        "mapping_evidence_available": bool(
                            mapping["mapping_evidence_available"]
                        ),
                        "control_record_present": bool(
                            mapping["control_record_present"]
                        ),
                        **result,
                        "target_joint_base_cluster_count": len(target_joint_base),
                        "target_joint_base_clusters_json": json_list(
                            target_joint_base
                        ),
                        "joint_target_control_cluster_count": len(joint_clusters),
                        "joint_target_control_clusters_json": json_list(
                            joint_clusters
                        ),
                        "jointly_estimable": jointly_estimable,
                        "joint_estimability_rule": (
                            "target and control both have admissible support/"
                            "complement outcomes in at least "
                            f"{args.min_joint_target_control_clusters} shared "
                            "object clusters"
                        ),
                    }
                )

            aggregate_rows.append(
                {
                    "address_id": address_id,
                    "record_id": record_id,
                    "support_id": support_id,
                    "failure_predicate": predicate,
                    "metric": metric,
                    "partition": partition,
                    "mapped_control_count": int(len(mapped)),
                    "mapped_relation_control_count": relation_mapped,
                    "mapped_carrier_control_count": carrier_mapped,
                    "admissible_control_count": admissible_count,
                    "admissible_relation_control_count": relation_admissible,
                    "admissible_carrier_control_count": carrier_admissible,
                    "jointly_estimable_control_count": joint_count,
                    "maximum_joint_cluster_count": max(
                        joint_cluster_counts,
                        default=0,
                    ),
                    "admissible_control_records_json": json_list(
                        admissible_control_ids
                    ),
                    "jointly_estimable_control_records_json": json_list(
                        joint_control_ids
                    ),
                    "g7_control_availability": bool(admissible_count >= 1),
                    "g8_joint_target_control_estimability": bool(
                        joint_count >= 1
                    ),
                    "control_availability_rule": (
                        "at least one OBS-083 relation or carrier control with "
                        "the same frozen support query, admissible complement, "
                        "and estimable predicate metric"
                    ),
                }
            )

    return pd.DataFrame(aggregate_rows), pd.DataFrame(control_rows)


# -----------------------------------------------------------------------------
# Gate synthesis and summary classes
# -----------------------------------------------------------------------------


def evidence_class(row: Mapping[str, Any]) -> str:
    failed = [gate for gate in GATE_COLUMNS if not normalize_bool(row.get(gate))]
    if not failed:
        return "evidence_feasible"
    mapping = {
        "g1_support_presence": "support_absent",
        "g2_complement_presence": "complement_absent",
        "g3_support_cluster_coverage": "support_cluster_limited",
        "g4_complement_cluster_coverage": "complement_cluster_limited",
        "g5_class_bearing_coverage": "class_coverage_limited",
        "g6_matched_complement_admissibility": "matching_limited",
        "g7_control_availability": "control_limited",
        "g8_joint_target_control_estimability": "joint_estimability_limited",
        "g9_outcome_estimability": "outcome_not_estimable",
        "g10_multiplicity_family_definition": "multiplicity_undefined",
    }
    if len(failed) == 1:
        return mapping[failed[0]]
    return "multiple_structural_limits"


def first_failed_gate(row: Mapping[str, Any]) -> str:
    for gate in GATE_COLUMNS:
        if not normalize_bool(row.get(gate)):
            return gate
    return ""


def failed_gates_json(row: Mapping[str, Any]) -> str:
    return json.dumps(
        [gate for gate in GATE_COLUMNS if not normalize_bool(row.get(gate))]
    )


def build_entitlement_overlay(record_catalog: pd.DataFrame) -> pd.DataFrame:
    columns = [
        c
        for c in (
            "record_id",
            "relation",
            "carrier",
            "subclass",
            "confirmation_eligible",
            "e1_fl3_claim_entitlement",
            "entitlement_status",
            "readiness_statement",
            "primary_limiter",
            "secondary_limiter",
            "failure_localization_score",
            "repair_specificity_level",
            "repair_specificity_score",
            "c4_evidence_gate_passed",
        )
        if c in record_catalog.columns
    ]
    out = record_catalog[columns].copy()
    out["entitlement_basis"] = (
        "frozen OBS-083 subclass and OBS-084 confirmation-eligibility contract"
    )
    out["entitlement_is_simulated"] = False
    return out


def build_detection_summary(
    address_inventory: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    evidence_feasibility: pd.DataFrame,
    entitlement: pd.DataFrame,
    candidate_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    record_scoped_support_count = int(
        address_inventory[["record_id", "support_id"]]
        .drop_duplicates()
        .shape[0]
    )
    sealed = evidence_feasibility[
        evidence_feasibility["sealed_obs084b_candidate"]
    ].copy()

    rows.extend(
        [
            {
                "summary_group": "universe",
                "summary_key": "registry_records",
                "value": int(address_inventory["record_id"].nunique()),
            },
            {
                "summary_group": "universe",
                "summary_key": "global_unique_support_templates",
                "value": int(address_inventory["support_id"].nunique()),
            },
            {
                "summary_group": "universe",
                "summary_key": "record_scoped_support_definitions",
                "value": record_scoped_support_count,
            },
            {
                "summary_group": "universe",
                "summary_key": "predicate_indexed_addresses",
                "value": int(address_inventory["address_id"].nunique()),
            },
            {
                "summary_group": "universe",
                "summary_key": "failure_predicates",
                "value": int(address_inventory["failure_predicate"].nunique()),
            },
            {
                "summary_group": "obs084_context",
                "summary_key": "sealed_candidates",
                "value": int(address_inventory["sealed_obs084b_candidate"].sum()),
            },
            {
                "summary_group": "obs084_context",
                "summary_key": "confirmed_fl3_witnesses",
                "value": int(
                    candidate_outcomes.get(
                        "confirmation_fl_maturity",
                        pd.Series(dtype=str),
                    )
                    .astype(str)
                    .str.startswith("FL3")
                    .sum()
                ),
            },
            {
                "summary_group": "obs084_context",
                "summary_key": "sealed_confirmation_feasible",
                "value": int(sealed["confirmation_evidence_feasible"].sum()),
            },
            {
                "summary_group": "obs084_context",
                "summary_key": "sealed_end_to_end_feasible",
                "value": int(sealed["end_to_end_evidence_feasible"].sum()),
            },
            {
                "summary_group": "obs084_context",
                "summary_key": "sealed_fl3_entitled",
                "value": int(sealed["e1_fl3_claim_entitlement"].sum()),
            },
            {
                "summary_group": "obs084_context",
                "summary_key": "sealed_feasible_and_fl3_entitled",
                "value": int(sealed["fl3_entitled_structural_ceiling"].sum()),
            },
            {
                "summary_group": "entitlement",
                "summary_key": "fl3_entitled_records",
                "value": int(entitlement["e1_fl3_claim_entitlement"].sum()),
            },
            {
                "summary_group": "entitlement",
                "summary_key": "fl3_entitlement_capped_records",
                "value": int((~entitlement["e1_fl3_claim_entitlement"]).sum()),
            },
        ]
    )

    for partition, group in gate_matrix.groupby("partition", sort=True):
        rows.append(
            {
                "summary_group": f"partition::{partition}",
                "summary_key": "evidence_feasible_addresses",
                "value": int(group["evidence_feasible"].sum()),
            }
        )
        rows.append(
            {
                "summary_group": f"partition::{partition}",
                "summary_key": "evidence_infeasible_addresses",
                "value": int((~group["evidence_feasible"]).sum()),
            }
        )
        for gate in GATE_COLUMNS:
            rows.append(
                {
                    "summary_group": f"partition::{partition}",
                    "summary_key": f"{gate}_pass",
                    "value": int(group[gate].sum()),
                }
            )
            rows.append(
                {
                    "summary_group": f"partition::{partition}",
                    "summary_key": f"{gate}_fail",
                    "value": int((~group[gate]).sum()),
                }
            )

    rows.extend(
        [
            {
                "summary_group": "campaign_structure",
                "summary_key": "discovery_and_confirmation_feasible",
                "value": int(
                    evidence_feasibility[
                        "end_to_end_evidence_feasible"
                    ].sum()
                ),
            },
            {
                "summary_group": "campaign_structure",
                "summary_key": "statistically_feasible_but_entitlement_capped",
                "value": int(
                    evidence_feasibility[
                        "end_to_end_feasible_but_entitlement_capped"
                    ].sum()
                ),
            },
        ]
    )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Reporting and output manifest
# -----------------------------------------------------------------------------


def write_report(
    path: Path,
    upstream_validation: Mapping[str, Any],
    input_manifest: pd.DataFrame,
    address_inventory: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    evidence_feasibility: pd.DataFrame,
    entitlement: pd.DataFrame,
    summary: pd.DataFrame,
    failures: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    discovery = gate_matrix[gate_matrix["partition"] == "discovery"]
    confirmation = gate_matrix[gate_matrix["partition"] == "confirmation"]
    record_scoped_support_count = int(
        address_inventory[["record_id", "support_id"]]
        .drop_duplicates()
        .shape[0]
    )
    sankey_counts = structural_state_counts(evidence_feasibility)
    sankey_terminal_states = pd.DataFrame(
        [
            {
                "structural_state": "feasible_in_both_partitions",
                "address_count": sankey_counts["both_feasible"],
            },
            {
                "structural_state": "discovery_feasible_only",
                "address_count": sankey_counts["discovery_only"],
            },
            {
                "structural_state": "confirmation_feasible_only",
                "address_count": sankey_counts["confirmation_only"],
            },
            {
                "structural_state": "infeasible_in_both_partitions",
                "address_count": sankey_counts["infeasible_both"],
            },
        ]
    )

    gate_fail_rows: list[dict[str, Any]] = []
    for partition, group in gate_matrix.groupby("partition", sort=True):
        for gate in GATE_COLUMNS:
            gate_fail_rows.append(
                {
                    "partition": partition,
                    "gate": GATE_NAMES[gate],
                    "failed_addresses": int((~group[gate]).sum()),
                    "passed_addresses": int(group[gate].sum()),
                }
            )
    gate_failures = pd.DataFrame(gate_fail_rows).sort_values(
        ["partition", "failed_addresses"],
        ascending=[True, False],
    )

    structural_classes = (
        gate_matrix.groupby(["partition", "evidence_class"], as_index=False)
        .size()
        .rename(columns={"size": "address_count"})
        .sort_values(["partition", "address_count"], ascending=[True, False])
    )

    gate_relationship_rows: list[dict[str, Any]] = []
    for partition, group in gate_matrix.groupby("partition", sort=True):
        comparisons = [
            (
                "G5 vs G9",
                "g5_class_bearing_coverage",
                "g9_outcome_estimability",
            ),
            (
                "G7 vs G8",
                "g7_control_availability",
                "g8_joint_target_control_estimability",
            ),
            (
                "G6 vs EvidenceFeasible",
                "g6_matched_complement_admissibility",
                "evidence_feasible",
            ),
        ]
        for comparison, left, right in comparisons:
            gate_relationship_rows.append(
                {
                    "partition": partition,
                    "comparison": comparison,
                    "mismatched_addresses": int(
                        (group[left].astype(bool) != group[right].astype(bool)).sum()
                    ),
                }
            )

        other_gates = [
            gate
            for gate in GATE_COLUMNS
            if gate != "g6_matched_complement_admissibility"
        ]
        passes_other_gates = group[other_gates].astype(bool).all(axis=1)
        gate_relationship_rows.append(
            {
                "partition": partition,
                "comparison": "Pass all non-G6 gates",
                "mismatched_addresses": int(passes_other_gates.sum()),
            }
        )
        gate_relationship_rows.append(
            {
                "partition": partition,
                "comparison": "Pass all non-G6 gates and fail G6",
                "mismatched_addresses": int(
                    (
                        passes_other_gates
                        & ~group["g6_matched_complement_admissibility"].astype(bool)
                    ).sum()
                ),
            }
        )
    gate_relationships = pd.DataFrame(gate_relationship_rows)

    predicate_envelope = (
        gate_matrix.groupby(
            ["partition", "failure_predicate"],
            as_index=False,
        )
        .agg(
            addresses=("address_id", "size"),
            class_coverage_pass=("g5_class_bearing_coverage", "sum"),
            outcome_estimability_pass=("g9_outcome_estimability", "sum"),
            matching_pass=("g6_matched_complement_admissibility", "sum"),
            control_available=("g7_control_availability", "sum"),
            jointly_estimable=("g8_joint_target_control_estimability", "sum"),
            evidence_feasible=("evidence_feasible", "sum"),
        )
    )
    for column in predicate_envelope.columns:
        if column not in {"partition", "failure_predicate"}:
            predicate_envelope[column] = predicate_envelope[column].astype(int)
    predicate_count_columns = [
        column
        for column in predicate_envelope.columns
        if column not in {"partition", "failure_predicate"}
    ]
    predicate_counts_identical = all(
        group[predicate_count_columns].nunique().max() == 1
        for _, group in predicate_envelope.groupby("partition", sort=True)
    )

    other_gates = [
        gate
        for gate in GATE_COLUMNS
        if gate != "g6_matched_complement_admissibility"
    ]
    confirmation_other_pass = confirmation[other_gates].astype(bool).all(axis=1)
    confirmation_g6_only = confirmation[
        confirmation_other_pass
        & ~confirmation["g6_matched_complement_admissibility"].astype(bool)
    ].copy()
    g6_only_support_count = int(
        confirmation_g6_only[["record_id", "support_id"]]
        .drop_duplicates()
        .shape[0]
    )
    g6_only_record_count = int(confirmation_g6_only["record_id"].nunique())
    g6_only_checks: dict[str, int] = {}
    if not confirmation_g6_only.empty:
        parsed_checks = pd.json_normalize(
            confirmation_g6_only["matching_check_json"].map(json.loads)
        )
        for check in ("site_rows", "complement_rows", "shared_clusters"):
            g6_only_checks[check] = int(
                (~parsed_checks[check].astype(bool)).sum()
            )

    gate_relaxation_rows: list[dict[str, Any]] = []
    for threshold in sorted(
        {
            max(1, int(args.min_site_rows)),
            max(1, int(args.min_site_rows) - 1),
            max(1, int(args.min_site_rows) - 2),
        },
        reverse=True,
    ):
        eligible = confirmation_g6_only[
            confirmation_g6_only["n_site_rows"] >= threshold
        ]
        gate_relaxation_rows.append(
            {
                "hypothetical_min_site_rows": threshold,
                "additional_record_scoped_supports": int(
                    eligible[["record_id", "support_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "additional_predicate_indexed_addresses": int(len(eligible)),
            }
        )
    gate_relaxation = pd.DataFrame(gate_relaxation_rows)

    entitlement_counts = (
        entitlement.groupby("entitlement_status", as_index=False)
        .size()
        .rename(columns={"size": "record_count"})
    )

    context_columns = [
        c
        for c in (
            "record_id",
            "failure_predicate",
            "support_definition",
            "discovery_evidence_feasible",
            "confirmation_evidence_feasible",
            "end_to_end_evidence_feasible",
            "entitlement_status",
            "obs084c_confirmation_status",
        )
        if c in evidence_feasibility.columns
    ]
    sealed = evidence_feasibility[
        evidence_feasibility["sealed_obs084b_candidate"]
    ].copy()
    sealed_context = sealed[context_columns]
    sealed_total = int(len(sealed))
    sealed_confirmation_feasible = int(
        sealed["confirmation_evidence_feasible"].sum()
    )
    sealed_end_to_end_feasible = int(
        sealed["end_to_end_evidence_feasible"].sum()
    )
    sealed_fl3_entitled = int(sealed["e1_fl3_claim_entitlement"].sum())
    sealed_feasible_and_entitled = int(
        sealed["fl3_entitled_structural_ceiling"].sum()
    )

    discovery_support_feasible = int(
        evidence_feasibility.loc[
            evidence_feasibility["discovery_evidence_feasible"].astype(bool),
            ["record_id", "support_id"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    confirmation_support_feasible = int(
        evidence_feasibility.loc[
            evidence_feasibility["confirmation_evidence_feasible"].astype(bool),
            ["record_id", "support_id"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    end_to_end_support_feasible = int(
        evidence_feasibility.loc[
            evidence_feasibility["end_to_end_evidence_feasible"].astype(bool),
            ["record_id", "support_id"],
        ]
        .drop_duplicates()
        .shape[0]
    )

    lines: list[str] = []
    lines.append("# OBS-085a — Structural Evidence Feasibility")
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append("`structural_evidence_feasibility_completed`")
    lines.append("")
    lines.append(
        "OBS-085a performs a deterministic structural audit only. It does not "
        "inject effects, estimate simulated gate-passage probability, compute "
        "observed power, or modify the completed OBS-084 result."
    )
    lines.append("")
    lines.append("## Immutable OBS-084 validation")
    lines.append("")
    lines.append("```text")
    lines.append(str(upstream_validation.get("stdout", "")))
    lines.append("```")
    lines.append("")
    lines.append(
        "The OBS-084b discovery structural checks were independently "
        "reconstructed from the frozen support queries and discovery "
        "observation-loss frame. Exact agreement with the frozen matching "
        "audit was required before this report could be written."
    )
    lines.append("")
    lines.append("## Address universe")
    lines.append("")
    lines.append(
        f"- Registry records: **{address_inventory['record_id'].nunique():,}**"
    )
    lines.append(
        "- Global unique support templates: "
        f"**{address_inventory['support_id'].nunique():,}**"
    )
    lines.append(
        "- Record-scoped support definitions: "
        f"**{record_scoped_support_count:,}**"
    )
    lines.append(
        f"- Predicate-indexed addresses: **{address_inventory['address_id'].nunique():,}**"
    )
    lines.append(
        f"- Failure predicates: **{address_inventory['failure_predicate'].nunique():,}**"
    )
    lines.append(
        f"- Sealed OBS-084b candidates: **{int(address_inventory['sealed_obs084b_candidate'].sum()):,}**"
    )
    lines.append("")
    lines.append("## Evidence feasibility")
    lines.append("")
    lines.append(
        f"Discovery-feasible addresses: **{int(discovery['evidence_feasible'].sum()):,} "
        f"/ {len(discovery):,}**"
    )
    lines.append(
        f"Confirmation-feasible addresses: **{int(confirmation['evidence_feasible'].sum()):,} "
        f"/ {len(confirmation):,}**"
    )
    lines.append(
        "Evidence-feasible in both partitions: "
        f"**{int(evidence_feasibility['end_to_end_evidence_feasible'].sum()):,} "
        f"/ {len(evidence_feasibility):,}**"
    )
    lines.append("")
    lines.append(
        "At the underlying record-scoped support level, feasible addresses "
        "represented "
        f"**{discovery_support_feasible:,}** discovery supports, "
        f"**{confirmation_support_feasible:,}** confirmation supports, and "
        f"**{end_to_end_support_feasible:,}** supports feasible in both "
        "partitions."
    )
    lines.append("")
    lines.append("## Structural-state Sankey")
    lines.append("")
    lines.append(
        "[Open the deterministic structural-state Sankey]"
        "(obs085a_structural_state_sankey.html)."
    )
    lines.append("")
    lines.append(markdown_table(sankey_terminal_states, args.max_report_rows))
    lines.append("")
    lines.append(
        "The Sankey cross-classifies mutually exclusive discovery and "
        "confirmation feasibility states, then overlays deterministic FL3 "
        "claim entitlement on the both-feasible branch. It is not a "
        "sequential gate-passage, causal-attrition, effect-existence, or "
        "detection-probability diagram. Authoritative link values are stored "
        "in `obs085a_structural_state_sankey_links.csv`."
    )
    lines.append("")
    lines.append("## Gate-failure audit")
    lines.append("")
    lines.append(markdown_table(gate_failures, args.max_report_rows))
    lines.append("")
    lines.append("## Structural gate relationships")
    lines.append("")
    lines.append(markdown_table(gate_relationships, args.max_report_rows))
    lines.append("")
    lines.append(
        "Under the frozen OBS-084 evidence structure, G6 exactly delimited "
        "the evidence-feasible set in both partitions. G5 and G9 were "
        "empirically coextensive, while G7 and G8 separated only in "
        "confirmation. These are observed relationships in this evidence "
        "spine, not claims that the gates are conceptually interchangeable."
    )
    lines.append("")
    lines.append("## Predicate-level structural envelope")
    lines.append("")
    lines.append(markdown_table(predicate_envelope, args.max_report_rows))
    lines.append("")
    if predicate_counts_identical:
        lines.append(
            "All failure predicates had identical structural pass counts. "
            "This reflects coextensive evidence availability under the frozen "
            "artifacts, not equivalence of predicate behavior, effect scale, "
            "or detectability."
        )
    else:
        lines.append(
            "The failure predicates differed in structural pass counts. These "
            "differences describe evidence availability only, not effect "
            "magnitude or detectability."
        )
    lines.append("")
    lines.append("## Confirmation focal-support row-floor diagnostic")
    lines.append("")
    if confirmation_g6_only.empty:
        lines.append(
            "No confirmation address passed every other gate while failing G6."
        )
    else:
        lines.append(
            f"**{len(confirmation_g6_only):,}** predicate-indexed addresses "
            f"across **{g6_only_support_count:,}** record-scoped supports and "
            f"**{g6_only_record_count:,}** records passed every other gate but "
            "failed G6."
        )
        lines.append("")
        lines.append(
            "The failed incremental G6 checks were: "
            f"site rows = **{g6_only_checks.get('site_rows', 0):,}**, "
            f"complement rows = **{g6_only_checks.get('complement_rows', 0):,}**, "
            f"shared clusters = **{g6_only_checks.get('shared_clusters', 0):,}**."
        )
        lines.append("")
        lines.append(
            "These cases are specifically focal-support-row-limited: their "
            "complements, shared-cluster structure, class coverage, controls, "
            "outcomes, and multiplicity definitions remained admissible."
        )
        lines.append("")
        lines.append(markdown_table(gate_relaxation, args.max_report_rows))
        lines.append("")
        lines.append(
            "The table above is a gate-relaxation diagnostic only. It does "
            "not recommend changing the frozen "
            f"{int(args.min_site_rows):,}-row requirement."
        )
    lines.append("")
    lines.append("## Structural classes")
    lines.append("")
    lines.append(markdown_table(structural_classes, args.max_report_rows))
    lines.append("")
    lines.append("## Claim-entitlement overlay")
    lines.append("")
    lines.append(
        "Claim entitlement is not included in `EvidenceFeasible`. It is a "
        "deterministic record-level overlay inherited from OBS-083/084."
    )
    lines.append("")
    lines.append(markdown_table(entitlement_counts, args.max_report_rows))
    lines.append("")
    lines.append(
        "Addresses that are structurally feasible in both partitions but "
        "remain FL3-entitlement capped: "
        f"**{int(evidence_feasibility['end_to_end_feasible_but_entitlement_capped'].sum()):,}**"
    )
    lines.append("")
    lines.append("## Sealed OBS-084 candidate structural ceiling")
    lines.append("")
    sealed_summary = pd.DataFrame(
        [
            {
                "measure": "sealed_candidates",
                "value": sealed_total,
            },
            {
                "measure": "confirmation_feasible",
                "value": sealed_confirmation_feasible,
            },
            {
                "measure": "both_partitions_feasible",
                "value": sealed_end_to_end_feasible,
            },
            {
                "measure": "fl3_entitled",
                "value": sealed_fl3_entitled,
            },
            {
                "measure": "both_feasible_and_fl3_entitled",
                "value": sealed_feasible_and_entitled,
            },
        ]
    )
    lines.append(markdown_table(sealed_summary, args.max_report_rows))
    lines.append("")
    lines.append(
        f"Within the sealed {sealed_total:,}-candidate family, "
        f"{sealed_confirmation_feasible:,} candidates retained confirmation "
        "structural feasibility. "
        "FL3 entitlement applied to "
        f"{sealed_fl3_entitled:,} members of the sealed family, while "
        f"{sealed_feasible_and_entitled:,} members were simultaneously "
        "feasible in both partitions and FL3-entitled. This is a deterministic "
        "structural "
        "ceiling, not a reinterpretation of the realized confirmation "
        "contrasts."
    )
    lines.append("")
    lines.append("## Sealed OBS-084 candidate context")
    lines.append("")
    lines.append(markdown_table(sealed_context, args.max_report_rows))
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(
        markdown_table(
            input_manifest[
                [
                    "artifact_role",
                    "artifact_path",
                    "size_bytes",
                    "sha256",
                ]
            ],
            args.max_report_rows,
        )
    )
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    lines.append(markdown_table(failures, args.max_report_rows))
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "OBS-085a establishes structural evidence feasibility only. A passing "
        "address has enough frozen empirical structure to support later "
        "simulator qualification and conditional gate-passage analysis. It is "
        "not evidence that an artifact-direct effect exists, and it does not "
        "alter the null FL3 result of OBS-084."
    )
    lines.append("")
    lines.append(
        "> Claim entitlement is an epistemic ceiling, not a component of "
        "structural estimability."
    )
    lines.append("")
    lines.append(
        "> OBS-085a does not compute observed power or simulated gate-passage "
        "probability."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    repo_root: Path,
    output_dir: Path,
    input_manifest: pd.DataFrame,
    outputs: Mapping[str, Path],
    upstream_validation: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    identity_columns = [
        "artifact_role",
        "artifact_path",
        "sha256",
        "obs084a_freeze_manifest_id",
        "obs084b_candidate_manifest_id",
        "obs084c_confirmation_manifest_id",
        "obs084c_opening_lock_id",
    ]
    input_identity = input_manifest[
        [c for c in identity_columns if c in input_manifest.columns]
    ].to_dict("records")

    output_hashes = {
        name: sha256_file(path)
        for name, path in outputs.items()
        if path.is_file() and name != "obs085a_manifest.json"
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "status": "structural_evidence_feasibility_completed",
        "repo_commit": git_commit(repo_root),
        "script_path": repo_relative_path(Path(__file__).resolve(), repo_root),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "output_dir": repo_relative_path(output_dir, repo_root),
        "upstream_validation": dict(upstream_validation),
        "input_identity": input_identity,
        "structural_contract": {
            "cluster_unit": "object",
            "min_site_rows": args.min_site_rows,
            "min_complement_rows": args.min_complement_rows,
            "min_class_rows": args.min_class_rows,
            "min_site_clusters": args.min_site_clusters,
            "min_complement_clusters": args.min_complement_clusters,
            "min_shared_clusters": args.min_shared_clusters,
            "min_joint_target_control_clusters": (
                args.min_joint_target_control_clusters
            ),
            "gates": GATE_NAMES,
            "claim_entitlement_separate_from_evidence_feasibility": True,
        },
        "output_hashes": output_hashes,
        "guardrails": [
            "OBS-084 remains immutable",
            "no synthetic injection",
            "no simulated gate-passage probability",
            "no observed power",
            "no FL3 promotion",
            "claim entitlement is a deterministic overlay",
        ],
    }
    payload["obs085a_manifest_id"] = stable_hash(payload)
    return payload


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    obs084_root = resolve_path(repo_root, args.obs084_root).resolve()
    obs083_dir = resolve_path(repo_root, args.obs083_dir).resolve()
    registry_path = resolve_path(repo_root, args.registry).resolve()
    output_dir = resolve_path(repo_root, args.output_dir).resolve()
    obs084c_script = resolve_path(repo_root, args.obs084c_script).resolve()
    protocol_path = resolve_path(repo_root, args.protocol).resolve()

    paths = build_input_paths(
        repo_root,
        obs084_root,
        obs083_dir,
        registry_path,
        obs084c_script,
        protocol_path,
    )
    require_input_files(paths)

    upstream_validation = run_obs084c_validation(
        repo_root,
        obs084c_script,
        args.require_repo_commit,
    )

    # Load immutable inputs.
    support_inventory = read_csv_required(paths["obs084b_support_inventory"])
    frozen_matching = read_csv_required(paths["obs084b_support_matching"])
    discovery_losses = read_csv_required(paths["obs084b_observation_losses"])
    candidate_manifest = read_csv_required(
        paths["obs084b_candidate_manifest_csv"]
    )
    confirmation_losses = read_csv_required(paths["obs084c_observation_losses"])
    confirmation_support = read_csv_required(paths["obs084c_support_validation"])
    candidate_outcomes = read_csv_required(paths["obs084c_candidate_outcomes"])
    subclasses = read_csv_required(paths["obs083_subclasses"])
    relation_controls = read_csv_required(paths["obs083_relation_controls"])
    carrier_controls = read_csv_required(paths["obs083_carrier_controls"])
    registry = read_csv_required(paths["rig_registry"])

    require_columns(
        support_inventory,
        [
            "record_id",
            "relation",
            "carrier",
            "subclass",
            "confirmation_eligible",
            "support_id",
            "support_depth",
            "support_families",
            "support_columns",
            "support_values",
            "support_definition",
            "support_query_json",
        ],
        "OBS-084b support inventory",
    )
    require_columns(
        frozen_matching,
        [
            "candidate_test_id",
            "record_id",
            "support_id",
            "failure_predicate",
        ],
        "OBS-084b matching audit",
    )
    require_columns(
        discovery_losses,
        [
            "record_id",
            "relation",
            "carrier",
            "observation_key",
            "cluster_id",
            "true_regime",
            "partition_role",
        ],
        "OBS-084b observation losses",
    )
    require_columns(
        confirmation_losses,
        [
            "record_id",
            "relation",
            "carrier",
            "observation_key",
            "cluster_id",
            "true_regime",
            "partition_role",
        ],
        "OBS-084c observation losses",
    )
    require_columns(
        candidate_manifest,
        [
            "candidate_id",
            "record_id",
            "support_definition",
            "failure_predicate",
        ],
        "OBS-084b candidate manifest",
    )
    require_columns(
        confirmation_support,
        [
            "candidate_id",
            "record_id",
            "support_definition",
            "protocol_match",
        ],
        "OBS-084c support validation",
    )
    require_columns(
        candidate_outcomes,
        [
            "candidate_id",
            "confirmation_status",
            "confirmation_fl_maturity",
        ],
        "OBS-084c candidate outcomes",
    )

    candidate_ids = set(candidate_manifest["candidate_id"].astype(str))
    support_validation_ids = set(confirmation_support["candidate_id"].astype(str))
    outcome_ids = set(candidate_outcomes["candidate_id"].astype(str))
    if len(candidate_ids) != len(candidate_manifest):
        raise RuntimeError("OBS-084b candidate IDs are not unique")
    if candidate_ids != support_validation_ids or candidate_ids != outcome_ids:
        raise RuntimeError(
            "OBS-084 candidate identity mismatch across discovery manifest, "
            "confirmation support validation, and confirmation outcomes"
        )
    if not confirmation_support["protocol_match"].map(normalize_bool).all():
        raise RuntimeError(
            "OBS-084c support validation contains a protocol mismatch"
        )

    if set(discovery_losses["partition_role"].astype(str).unique()) != {
        "discovery"
    }:
        raise RuntimeError("Discovery losses contain non-discovery rows")
    if set(confirmation_losses["partition_role"].astype(str).unique()) != {
        "confirmation"
    }:
        raise RuntimeError("Confirmation losses contain non-confirmation rows")

    record_catalog = load_record_catalog(registry, subclasses, support_inventory)
    known_records = set(record_catalog["record_id"].astype(str))
    control_catalog = build_control_catalog(
        relation_controls,
        carrier_controls,
        known_records,
    )

    # The frozen address universe is the exact predicate-indexed OBS-084b test
    # family: support definitions from the outcome-blind inventory crossed with
    # the four frozen predicates represented in the matching audit.
    support_meta = support_inventory.drop_duplicates(["record_id", "support_id"])
    address_inventory = frozen_matching.merge(
        support_meta,
        on=["record_id", "support_id"],
        how="left",
        suffixes=("", "_inventory"),
        validate="many_to_one",
    )
    if address_inventory["support_query_json"].isna().any():
        missing = address_inventory.loc[
            address_inventory["support_query_json"].isna(),
            ["record_id", "support_id"],
        ].drop_duplicates()
        raise RuntimeError(
            "Frozen matching rows lack support metadata: "
            f"{missing.to_dict('records')[:10]}"
        )

    address_inventory = address_inventory.rename(
        columns={"candidate_test_id": "address_id"}
    )
    address_inventory["metric"] = address_inventory[
        "failure_predicate"
    ].map(PREDICATE_METRICS)
    if address_inventory["metric"].isna().any():
        unknown = sorted(
            address_inventory.loc[
                address_inventory["metric"].isna(),
                "failure_predicate",
            ]
            .astype(str)
            .unique()
        )
        raise RuntimeError(f"Unknown frozen failure predicates: {unknown}")

    if address_inventory["address_id"].astype(str).duplicated().any():
        raise RuntimeError("Frozen address IDs are not unique")
    expected_cross = (
        address_inventory[["record_id", "support_id"]]
        .drop_duplicates()
        .shape[0]
        * len(PREDICATE_METRICS)
    )
    if len(address_inventory) != expected_cross:
        raise RuntimeError(
            "Frozen support × predicate universe is incomplete: "
            f"rows={len(address_inventory)}, expected={expected_cross}"
        )

    sealed_keys = set(
        zip(
            candidate_manifest["record_id"].astype(str),
            candidate_manifest["support_definition"].astype(str),
            candidate_manifest["failure_predicate"].astype(str),
        )
    )
    candidate_id_lookup = {
        (
            str(row["record_id"]),
            str(row["support_definition"]),
            str(row["failure_predicate"]),
        ): str(row["candidate_id"])
        for _, row in candidate_manifest.iterrows()
    }
    outcome_lookup = (
        candidate_outcomes.set_index("candidate_id").to_dict("index")
        if not candidate_outcomes.empty
        else {}
    )
    address_inventory["sealed_obs084b_candidate"] = [
        (
            str(row.record_id),
            str(row.support_definition),
            str(row.failure_predicate),
        )
        in sealed_keys
        for row in address_inventory.itertuples(index=False)
    ]
    address_inventory["candidate_id"] = [
        candidate_id_lookup.get(
            (
                str(row.record_id),
                str(row.support_definition),
                str(row.failure_predicate),
            ),
            "",
        )
        for row in address_inventory.itertuples(index=False)
    ]
    mapped_candidate_ids = set(
        address_inventory.loc[
            address_inventory["sealed_obs084b_candidate"],
            "candidate_id",
        ].astype(str)
    )
    if mapped_candidate_ids != candidate_ids:
        raise RuntimeError(
            "The sealed OBS-084b candidates do not map one-to-one onto the "
            "frozen OBS-085a address universe"
        )
    if int(address_inventory["sealed_obs084b_candidate"].sum()) != len(candidate_ids):
        raise RuntimeError(
            "The number of sealed address rows differs from the frozen "
            "candidate-manifest family size"
        )
    address_inventory["m5736_defined"] = True
    address_inventory["m13_defined"] = address_inventory[
        "sealed_obs084b_candidate"
    ]
    address_inventory["multiplicity_family_membership"] = np.where(
        address_inventory["m13_defined"],
        "M5736|M13",
        "M5736",
    )
    address_inventory["g10_multiplicity_family_definition"] = True

    query_by_support: dict[str, SupportQuery] = {}
    for _, row in support_meta.iterrows():
        support_id = str(row["support_id"])
        parsed = parse_support_query(
            support_id,
            str(row["support_definition"]),
            row["support_query_json"],
        )
        existing = query_by_support.get(support_id)
        if existing and existing != parsed:
            raise RuntimeError(
                f"Support ID {support_id} maps to inconsistent frozen queries"
            )
        query_by_support[support_id] = parsed

    partition_frames = {
        "discovery": discovery_losses,
        "confirmation": confirmation_losses,
    }
    observations_by_partition_record = {
        (partition, str(record_id)): group.copy()
        for partition, frame in partition_frames.items()
        for record_id, group in frame.groupby("record_id", sort=False)
    }

    # Compute support-level structural coverage once per record/support/partition.
    support_structure_rows: list[dict[str, Any]] = []
    structure_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, support_row in support_meta.iterrows():
        record_id = str(support_row["record_id"])
        support_id = str(support_row["support_id"])
        relation = str(support_row["relation"])
        query = query_by_support[support_id]
        for partition in ("discovery", "confirmation"):
            record_df = observations_by_partition_record.get(
                (partition, record_id),
                pd.DataFrame(),
            )
            if record_df.empty:
                raise RuntimeError(
                    f"No {partition} observation-loss rows for record {record_id}"
                )
            structure = compute_support_structure(
                record_df,
                query,
                relation,
                args,
            )
            row = {
                "record_id": record_id,
                "relation": relation,
                "carrier": str(support_row["carrier"]),
                "support_id": support_id,
                "support_depth": int(support_row["support_depth"]),
                "support_families": str(support_row["support_families"]),
                "support_columns": str(support_row["support_columns"]),
                "support_values": str(support_row["support_values"]),
                "support_definition": str(support_row["support_definition"]),
                "support_query_json": str(support_row["support_query_json"]),
                "partition": partition,
                **structure,
            }
            support_structure_rows.append(row)
            structure_cache[(partition, record_id, support_id)] = row

    support_coverage = pd.DataFrame(support_structure_rows)

    # Before any OBS-085a result is accepted, reconstruct and exactly match the
    # frozen OBS-084b discovery structural contract.
    contract_audit = frozen_discovery_contract_validation(
        support_coverage,
        frozen_matching,
    )
    if contract_audit.empty or not contract_audit["frozen_contract_match"].all():
        bad = contract_audit.loc[
            ~contract_audit["frozen_contract_match"],
            [
                "record_id",
                "support_id",
                "_merge",
                "row_match",
                "n_site_rows_match",
                "n_complement_rows_match",
                "n_site_clusters_match",
                "n_complement_clusters_match",
                "n_shared_clusters_match",
                "complement_admissible_match",
                "matching_check_match",
            ],
        ]
        raise RuntimeError(
            "OBS-085a reconstruction does not exactly match the frozen "
            "OBS-084b discovery structural contract. First mismatches: "
            f"{bad.head(10).to_dict('records')}"
        )

    if args.validate_only:
        print("OBS-085a validation complete: immutable OBS-084 bundle valid")
        print(
            "Frozen discovery structural contract reproduced exactly: "
            f"{len(contract_audit):,} support definitions"
        )
        print(
            "Predicate-indexed address universe: "
            f"{len(address_inventory):,} addresses"
        )
        return 0

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite only to regenerate OBS-085a outputs from the same "
            "immutable input bundle."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predicate-specific effective evidence and outcome estimability.
    effective_rows: list[dict[str, Any]] = []
    structure_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _, address in address_inventory.iterrows():
        address_id = str(address["address_id"])
        record_id = str(address["record_id"])
        support_id = str(address["support_id"])
        relation = str(address["relation"])
        metric = str(address["metric"])
        query = query_by_support[support_id]
        for partition in ("discovery", "confirmation"):
            generic = structure_cache[(partition, record_id, support_id)]
            record_df = observations_by_partition_record[(partition, record_id)]
            metric_part = compute_metric_structure(
                record_df,
                query,
                relation,
                metric,
                args,
            )
            row = {
                "address_id": address_id,
                "record_id": record_id,
                "relation": relation,
                "carrier": str(address["carrier"]),
                "subclass": str(address["subclass"]),
                "support_id": support_id,
                "support_definition": str(address["support_definition"]),
                "support_query_json": str(address["support_query_json"]),
                "failure_predicate": str(address["failure_predicate"]),
                "metric": metric,
                "partition": partition,
                **{
                    key: value
                    for key, value in generic.items()
                    if key
                    not in {
                        "record_id",
                        "relation",
                        "carrier",
                        "support_id",
                        "support_depth",
                        "support_families",
                        "support_columns",
                        "support_values",
                        "support_definition",
                        "support_query_json",
                        "partition",
                    }
                },
                **metric_part,
            }
            effective_rows.append(row)
            structure_lookup[(partition, record_id, support_id, str(address["failure_predicate"]))] = row

    effective_evidence = pd.DataFrame(effective_rows)

    control_availability, joint_estimability = compute_control_feasibility(
        address_inventory,
        partition_frames,
        record_catalog,
        control_catalog,
        query_by_support,
        structure_lookup,
        args,
    )

    gate_matrix = effective_evidence.merge(
        control_availability,
        on=[
            "address_id",
            "record_id",
            "support_id",
            "failure_predicate",
            "metric",
            "partition",
        ],
        how="left",
        validate="one_to_one",
    )
    gate_matrix = gate_matrix.merge(
        address_inventory[
            [
                "address_id",
                "sealed_obs084b_candidate",
                "candidate_id",
                "m5736_defined",
                "m13_defined",
                "multiplicity_family_membership",
                "g10_multiplicity_family_definition",
                "confirmation_eligible",
            ]
        ],
        on="address_id",
        how="left",
        validate="many_to_one",
    )
    entitlement_map = record_catalog.set_index("record_id")[
        ["e1_fl3_claim_entitlement", "entitlement_status"]
    ]
    gate_matrix = gate_matrix.merge(
        entitlement_map,
        left_on="record_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    for gate in GATE_COLUMNS:
        gate_matrix[gate] = gate_matrix[gate].map(normalize_bool)
    gate_matrix["evidence_feasible"] = gate_matrix[list(GATE_COLUMNS)].all(
        axis=1
    )
    gate_matrix["failed_gate_count"] = (
        ~gate_matrix[list(GATE_COLUMNS)]
    ).sum(axis=1)
    gate_matrix["failed_gates_json"] = gate_matrix.apply(
        failed_gates_json,
        axis=1,
    )
    gate_matrix["first_failed_gate"] = gate_matrix.apply(
        first_failed_gate,
        axis=1,
    )
    gate_matrix["evidence_class"] = gate_matrix.apply(evidence_class, axis=1)
    gate_matrix["statistically_feasible_but_entitlement_capped"] = (
        gate_matrix["evidence_feasible"]
        & ~gate_matrix["e1_fl3_claim_entitlement"].map(normalize_bool)
    )

    # Address-level synthesis across immutable discovery and confirmation frames.
    summary_columns = [
        "address_id",
        "partition",
        "evidence_feasible",
        "evidence_class",
        "failed_gates_json",
        *GATE_COLUMNS,
    ]
    discovery_gate = gate_matrix[gate_matrix["partition"] == "discovery"][
        summary_columns
    ].copy()
    confirmation_gate = gate_matrix[
        gate_matrix["partition"] == "confirmation"
    ][summary_columns].copy()
    discovery_gate = discovery_gate.rename(
        columns={
            c: f"discovery_{c}"
            for c in discovery_gate.columns
            if c not in {"address_id", "partition"}
        }
    ).drop(columns="partition")
    confirmation_gate = confirmation_gate.rename(
        columns={
            c: f"confirmation_{c}"
            for c in confirmation_gate.columns
            if c not in {"address_id", "partition"}
        }
    ).drop(columns="partition")

    evidence_feasibility = address_inventory.merge(
        discovery_gate,
        on="address_id",
        how="left",
        validate="one_to_one",
    ).merge(
        confirmation_gate,
        on="address_id",
        how="left",
        validate="one_to_one",
    )
    evidence_feasibility = evidence_feasibility.merge(
        entitlement_map,
        left_on="record_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    evidence_feasibility["end_to_end_evidence_feasible"] = (
        evidence_feasibility["discovery_evidence_feasible"].map(normalize_bool)
        & evidence_feasibility["confirmation_evidence_feasible"].map(
            normalize_bool
        )
    )
    evidence_feasibility[
        "end_to_end_feasible_but_entitlement_capped"
    ] = (
        evidence_feasibility["end_to_end_evidence_feasible"]
        & ~evidence_feasibility["e1_fl3_claim_entitlement"].map(normalize_bool)
    )
    evidence_feasibility["fl3_entitled_structural_ceiling"] = (
        evidence_feasibility["end_to_end_evidence_feasible"]
        & evidence_feasibility["e1_fl3_claim_entitlement"].map(normalize_bool)
    )

    evidence_feasibility["obs084c_confirmation_status"] = [
        str(outcome_lookup.get(str(candidate_id), {}).get("confirmation_status", ""))
        if str(candidate_id)
        else ""
        for candidate_id in evidence_feasibility["candidate_id"]
    ]
    evidence_feasibility["obs084c_confirmation_fl_maturity"] = [
        str(
            outcome_lookup.get(str(candidate_id), {}).get(
                "confirmation_fl_maturity",
                "",
            )
        )
        if str(candidate_id)
        else ""
        for candidate_id in evidence_feasibility["candidate_id"]
    ]

    entitlement = build_entitlement_overlay(record_catalog)

    # Complement output retains the complete gate vector and the original
    # frozen discovery validation for inspection.
    complement_columns = [
        "address_id",
        "record_id",
        "support_id",
        "failure_predicate",
        "partition",
        "n_site_rows",
        "n_complement_rows",
        "n_site_clusters",
        "n_complement_clusters",
        "n_shared_clusters",
        "site_class_counts_json",
        "complement_class_counts_json",
        "matching_check_json",
        "complement_admissible",
        "g1_support_presence",
        "g2_complement_presence",
        "g3_support_cluster_coverage",
        "g4_complement_cluster_coverage",
        "g5_class_bearing_coverage",
        "g6_matched_complement_admissibility",
    ]
    complement_admissibility = gate_matrix[
        [c for c in complement_columns if c in gate_matrix.columns]
    ].copy()

    failures: list[Failure] = []
    missing_control_mappings = gate_matrix[
        gate_matrix["mapped_control_count"].fillna(0).astype(int) == 0
    ]
    for record_id in sorted(missing_control_mappings["record_id"].astype(str).unique()):
        failures.append(
            Failure(
                "control_catalog",
                record_id,
                "no_frozen_control_mapping",
                "No evidence-available OBS-083 relation or carrier control mapping.",
                "warning",
            )
        )
    for partition, group in gate_matrix.groupby("partition"):
        unestimable = group[~group["g9_outcome_estimability"]]
        if not unestimable.empty:
            failures.append(
                Failure(
                    "outcome_estimability",
                    partition,
                    "predicate_metric_unestimable_for_some_addresses",
                    f"address_count={len(unestimable)}",
                    "warning",
                )
            )
    failures_df = as_failure_frame(failures)

    summary = build_detection_summary(
        address_inventory,
        gate_matrix,
        evidence_feasibility,
        entitlement,
        candidate_outcomes,
    )

    input_manifest = build_input_manifest(
        repo_root,
        paths,
        upstream_validation,
    )

    outputs = {
        "obs085a_input_manifest.csv": output_dir / "obs085a_input_manifest.csv",
        "obs085a_support_address_inventory.csv": output_dir
        / "obs085a_support_address_inventory.csv",
        "obs085a_support_coverage_matrix.csv": output_dir
        / "obs085a_support_coverage_matrix.csv",
        "obs085a_effective_evidence.csv": output_dir
        / "obs085a_effective_evidence.csv",
        "obs085a_complement_admissibility.csv": output_dir
        / "obs085a_complement_admissibility.csv",
        "obs085a_control_availability.csv": output_dir
        / "obs085a_control_availability.csv",
        "obs085a_joint_target_control_estimability.csv": output_dir
        / "obs085a_joint_target_control_estimability.csv",
        "obs085a_structural_gate_matrix.csv": output_dir
        / "obs085a_structural_gate_matrix.csv",
        "obs085a_evidence_feasibility.csv": output_dir
        / "obs085a_evidence_feasibility.csv",
        "obs085a_claim_entitlement_overlay.csv": output_dir
        / "obs085a_claim_entitlement_overlay.csv",
        "obs085a_detection_envelope_summary.csv": output_dir
        / "obs085a_detection_envelope_summary.csv",
        "obs085a_structural_state_sankey_links.csv": output_dir
        / "obs085a_structural_state_sankey_links.csv",
        "obs085a_structural_state_sankey.html": output_dir
        / "obs085a_structural_state_sankey.html",
        "obs085a_failures.csv": output_dir / "obs085a_failures.csv",
        "obs085a_report.md": output_dir / "obs085a_report.md",
        "obs085a_manifest.json": output_dir / "obs085a_manifest.json",
        "obs085a_frozen_discovery_contract_audit.csv": output_dir
        / "obs085a_frozen_discovery_contract_audit.csv",
    }

    frames = {
        "obs085a_input_manifest.csv": input_manifest,
        "obs085a_support_address_inventory.csv": address_inventory,
        "obs085a_support_coverage_matrix.csv": support_coverage,
        "obs085a_effective_evidence.csv": effective_evidence,
        "obs085a_complement_admissibility.csv": complement_admissibility,
        "obs085a_control_availability.csv": control_availability,
        "obs085a_joint_target_control_estimability.csv": joint_estimability,
        "obs085a_structural_gate_matrix.csv": gate_matrix,
        "obs085a_evidence_feasibility.csv": evidence_feasibility,
        "obs085a_claim_entitlement_overlay.csv": entitlement,
        "obs085a_detection_envelope_summary.csv": summary,
        "obs085a_structural_state_sankey_links.csv": (
            build_structural_state_sankey_links(evidence_feasibility)
        ),
        "obs085a_failures.csv": failures_df,
        "obs085a_frozen_discovery_contract_audit.csv": contract_audit,
    }
    for name, frame in frames.items():
        frame.to_csv(outputs[name], index=False)

    write_structural_state_sankey(
        outputs["obs085a_structural_state_sankey.html"],
        evidence_feasibility,
        frames["obs085a_structural_state_sankey_links.csv"],
    )

    write_report(
        outputs["obs085a_report.md"],
        upstream_validation,
        input_manifest,
        address_inventory,
        gate_matrix,
        evidence_feasibility,
        entitlement,
        summary,
        failures_df,
        args,
    )

    manifest = build_manifest(
        repo_root,
        output_dir,
        input_manifest,
        outputs,
        upstream_validation,
        args,
    )
    outputs["obs085a_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("OBS-085a structural evidence-feasibility audit complete")
    print(f"Addresses audited: {len(address_inventory):,}")
    print(
        "Discovery evidence-feasible: "
        f"{int(gate_matrix.loc[gate_matrix['partition'] == 'discovery', 'evidence_feasible'].sum()):,}"
    )
    print(
        "Confirmation evidence-feasible: "
        f"{int(gate_matrix.loc[gate_matrix['partition'] == 'confirmation', 'evidence_feasible'].sum()):,}"
    )
    print(
        "Evidence-feasible in both partitions: "
        f"{int(evidence_feasibility['end_to_end_evidence_feasible'].sum()):,}"
    )
    print(f"OBS-085a manifest ID: {manifest['obs085a_manifest_id']}")
    print(
        "Structural-state Sankey: "
        f"{repo_relative_path(outputs['obs085a_structural_state_sankey.html'], repo_root)}"
    )
    print(f"Outputs: {repo_relative_path(output_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
