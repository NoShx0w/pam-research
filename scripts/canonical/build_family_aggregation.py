#!/usr/bin/env python3
"""
Build the canonical family-level aggregation table.

This script aggregates normalized event features and canonical event assignments
into a single family-level artifact:

    outputs/canonical/family_aggregation.csv

It also writes a small human-readable validation report:

    outputs/canonical/validation/family_aggregation_validation.txt

Current scope
-------------
Version 1 is intentionally conservative:
- aggregates assigned events by assigned_family
- summarizes assignment diagnostics
- summarizes event-type shares
- summarizes mapped family-level enrichments
- provides a compact end-to-end coherence surface
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import pandas as pd


CANONICAL_ROUTE_CLASSES = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

AGGREGATION_VERSION = "family_aggregation_v1"


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def mode_or_na(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return pd.NA
    modes = s.mode()
    if modes.empty:
        return pd.NA
    return modes.iloc[0]


def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.NA
    return s.mean()


def build_family_aggregation(features_df: pd.DataFrame, assign_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        features_df,
        required=[
            "event_id",
            "event_type",
            "source_row_type",
            "local_gateway_strength",
            "crossing_rate",
            "effective_temporal_depth",
            "memory_regime_label",
            "forgetting_share",
            "dominant_compression_state",
            "branch_exit_score" if "branch_exit_score" in features_df.columns else "event_id",
        ],
        name="event_family_features",
    )
    require_columns(
        assign_df,
        required=[
            "event_id",
            "assigned_family",
            "assignment_confidence",
            "assignment_ambiguity_flag",
            "manual_review_flag",
            "branch_exit_score",
            "stable_seam_corridor_score",
            "reorganization_heavy_score",
            "assignment_version",
            "source_feature_version",
        ],
        name="event_family_assignment",
    )

    merged = assign_df.merge(
        features_df,
        on="event_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_assign", "_feat"),
    )

    rows = []
    for family, sub in merged.groupby("assigned_family", dropna=False):
        n_events = len(sub)

        core_internal_share = (sub["event_type"] == "core_internal").mean() if n_events else pd.NA
        core_to_escape_share = (sub["event_type"] == "core_to_escape").mean() if n_events else pd.NA

        source_row_type_count = sub["source_row_type"].nunique(dropna=True)

        row = {
            "route_class": family,
            "n_events": n_events,
            "mean_assignment_confidence": safe_mean(sub["assignment_confidence"]),
            "ambiguous_event_share": safe_mean(sub["assignment_ambiguity_flag"].astype(float)),
            "manual_review_event_share": safe_mean(sub["manual_review_flag"].astype(float)),
            "core_internal_share": core_internal_share,
            "core_to_escape_share": core_to_escape_share,
            "mean_local_gateway_strength": safe_mean(sub["local_gateway_strength"]),
            "mean_crossing_rate": safe_mean(sub["crossing_rate"]),
            "mean_effective_temporal_depth": safe_mean(sub["effective_temporal_depth"]),
            "memory_regime_label_mode": mode_or_na(sub["memory_regime_label"]),
            "mean_forgetting_share": safe_mean(sub["forgetting_share"]),
            "dominant_compression_state_mode": mode_or_na(sub["dominant_compression_state"]),
            "mean_branch_exit_score": safe_mean(sub["branch_exit_score"]),
            "mean_stable_seam_corridor_score": safe_mean(sub["stable_seam_corridor_score"]),
            "mean_reorganization_heavy_score": safe_mean(sub["reorganization_heavy_score"]),
            "source_row_type_count": source_row_type_count,
            "aggregation_version": AGGREGATION_VERSION,
            "source_feature_version": mode_or_na(sub["source_feature_version"]),
            "source_assignment_version": mode_or_na(sub["assignment_version"]),
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    out["route_class"] = pd.Categorical(
        out["route_class"],
        categories=CANONICAL_ROUTE_CLASSES,
        ordered=True,
    )
    out = out.sort_values("route_class").reset_index(drop=True)

    preferred_order = [
        "route_class",
        "n_events",
        "mean_assignment_confidence",
        "ambiguous_event_share",
        "manual_review_event_share",
        "core_internal_share",
        "core_to_escape_share",
        "mean_local_gateway_strength",
        "mean_crossing_rate",
        "mean_effective_temporal_depth",
        "memory_regime_label_mode",
        "mean_forgetting_share",
        "dominant_compression_state_mode",
        "mean_branch_exit_score",
        "mean_stable_seam_corridor_score",
        "mean_reorganization_heavy_score",
        "source_row_type_count",
        "aggregation_version",
        "source_feature_version",
        "source_assignment_version",
    ]
    actual_order = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
    return out[actual_order]


def validate_family_aggregation(agg_df: pd.DataFrame, assign_df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

    require_columns(
        agg_df,
        required=[
            "route_class",
            "n_events",
            "mean_assignment_confidence",
            "ambiguous_event_share",
            "manual_review_event_share",
            "core_internal_share",
            "core_to_escape_share",
            "aggregation_version",
            "source_feature_version",
            "source_assignment_version",
        ],
        name="family_aggregation",
    )

    if agg_df["route_class"].duplicated().any():
        raise ValueError("Duplicate route_class rows found in family_aggregation.")

    observed_classes = set(agg_df["route_class"].dropna().astype(str))
    missing_classes = [c for c in CANONICAL_ROUTE_CLASSES if c not in observed_classes]
    extra_classes = [c for c in observed_classes if c not in CANONICAL_ROUTE_CLASSES]

    if missing_classes:
        warnings.append(f"Missing canonical route_class rows: {missing_classes}")
    if extra_classes:
        warnings.append(f"Unexpected non-canonical route_class rows: {extra_classes}")

    total_agg = int(pd.to_numeric(agg_df["n_events"], errors="coerce").sum())
    total_assign = len(assign_df)
    if total_agg != total_assign:
        warnings.append(f"Aggregated event count ({total_agg}) does not match assignment count ({total_assign}).")

    for _, row in agg_df.iterrows():
        fam = row["route_class"]
        ci = pd.to_numeric(pd.Series([row["core_internal_share"]]), errors="coerce").iloc[0]
        ce = pd.to_numeric(pd.Series([row["core_to_escape_share"]]), errors="coerce").iloc[0]
        if pd.notna(ci) and pd.notna(ce):
            total = ci + ce
            if abs(total - 1.0) > 1e-6:
                warnings.append(f"{fam}: core_internal_share + core_to_escape_share = {total:.6f}, expected ~1.0.")

    try:
        by_class = agg_df.set_index("route_class")
        be = float(by_class.loc["branch_exit", "mean_forgetting_share"])
        co = float(by_class.loc["stable_seam_corridor", "mean_forgetting_share"])
        rh = float(by_class.loc["reorganization_heavy", "mean_forgetting_share"])
        if not (be < co < rh):
            warnings.append(
                "mean_forgetting_share ordering is not strictly branch_exit < stable_seam_corridor < reorganization_heavy."
            )
    except Exception:
        warnings.append("Could not verify mean_forgetting_share ordering.")

    return warnings


def write_validation_report(path: Path, agg_df: pd.DataFrame, warnings: List[str]) -> None:
    lines: list[str] = []
    lines.append("family_aggregation validation report")
    lines.append(f"aggregation_version: {AGGREGATION_VERSION}")
    lines.append(f"n_rows: {len(agg_df)}")
    lines.append(f"route_classes: {', '.join(agg_df['route_class'].dropna().astype(str).tolist())}")
    lines.append("")

    if warnings:
        lines.append("warnings:")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("warnings:")
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    features_path = repo_root / "outputs" / "canonical" / "event_family_features.csv"
    assign_path = repo_root / "outputs" / "canonical" / "event_family_assignment.csv"

    out_csv = repo_root / "outputs" / "canonical" / "family_aggregation.csv"
    out_report = repo_root / "outputs" / "canonical" / "validation" / "family_aggregation_validation.txt"

    try:
        features_df = load_csv(features_path)
        assign_df = load_csv(assign_path)

        agg_df = build_family_aggregation(features_df, assign_df)
        warnings = validate_family_aggregation(agg_df, assign_df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(out_csv, index=False)
        write_validation_report(out_report, agg_df, warnings)

        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_report}")
        if warnings:
            print("Validation warnings:")
            for w in warnings:
                print(f" - {w}")
        else:
            print("Validation warnings: none")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
