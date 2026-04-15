#!/usr/bin/env python3
"""
Build the canonical trajectory-level family rollup table.

This script rolls up canonical event features and event assignments into a
trajectory-level artifact:

    outputs/canonical/trajectory_family_rollup.csv

It also writes a small human-readable validation report:

    outputs/canonical/validation/trajectory_family_rollup_validation.txt

Current scope
-------------
Version 1 is intentionally conservative:
- rolls up only rows with non-null trajectory_id
- does not attempt speculative trajectory linkage
- preserves family mixture rather than forcing purity
- functions both as a convenience layer and as a trajectory-linkage status surface
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import pandas as pd


ROLLUP_VERSION = "trajectory_family_rollup_v1"
CANONICAL_ROUTE_CLASSES = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.NA
    return s.mean()


def mode_or_na(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return pd.NA
    modes = s.mode()
    if modes.empty:
        return pd.NA
    return modes.iloc[0]


def shannon_entropy(probs: list[float]) -> float:
    import math
    total = 0.0
    for p in probs:
        if p > 0:
            total -= p * math.log2(p)
    return total


def build_empty_rollup(source_feature_version: str, source_assignment_version: str) -> pd.DataFrame:
    cols = [
        "trajectory_id",
        "n_events",
        "dominant_family",
        "dominant_family_share",
        "family_mixture_entropy",
        "family_mixture_flag",
        "mean_assignment_confidence",
        "ambiguous_event_share",
        "manual_review_event_share",
        "core_internal_share",
        "core_to_escape_share",
        "mean_effective_temporal_depth",
        "mean_forgetting_share",
        "dominant_compression_state_mode",
        "rollup_version",
        "source_feature_version",
        "source_assignment_version",
    ]
    df = pd.DataFrame(columns=cols)
    if len(df) == 0:
        # preserve intended dtypes loosely by writing version metadata only if rows later exist
        pass
    return df


def build_trajectory_family_rollup(features_df: pd.DataFrame, assign_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        features_df,
        required=[
            "event_id",
            "trajectory_id",
            "event_type",
            "effective_temporal_depth",
            "forgetting_share",
            "dominant_compression_state",
            "normalization_version",
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

    # Only roll up cleanly linked trajectory rows.
    merged = merged[merged["trajectory_id"].notna()].copy()
    merged["trajectory_id"] = merged["trajectory_id"].astype(str).str.strip()
    merged = merged[merged["trajectory_id"] != ""].copy()

    source_feature_version = (
        mode_or_na(merged["normalization_version"]) if not merged.empty else mode_or_na(features_df["normalization_version"])
    )
    source_assignment_version = (
        mode_or_na(merged["assignment_version"]) if not merged.empty else mode_or_na(assign_df["assignment_version"])
    )

    if merged.empty:
        return build_empty_rollup(str(source_feature_version), str(source_assignment_version))

    rows = []
    for trajectory_id, sub in merged.groupby("trajectory_id", dropna=False):
        n_events = len(sub)

        fam_counts = (
            sub["assigned_family"]
            .value_counts(dropna=False)
            .reindex(CANONICAL_ROUTE_CLASSES, fill_value=0)
        )
        dominant_family = fam_counts.idxmax()
        dominant_count = int(fam_counts.max())
        dominant_family_share = dominant_count / n_events if n_events else pd.NA

        fam_probs = [(count / n_events) for count in fam_counts.tolist() if n_events]
        family_mixture_entropy = shannon_entropy(fam_probs) if n_events else pd.NA
        family_mixture_flag = sum(count > 0 for count in fam_counts.tolist()) > 1

        core_internal_share = (sub["event_type"] == "core_internal").mean() if n_events else pd.NA
        core_to_escape_share = (sub["event_type"] == "core_to_escape").mean() if n_events else pd.NA

        rows.append(
            {
                "trajectory_id": trajectory_id,
                "n_events": n_events,
                "dominant_family": dominant_family,
                "dominant_family_share": dominant_family_share,
                "family_mixture_entropy": family_mixture_entropy,
                "family_mixture_flag": family_mixture_flag,
                "mean_assignment_confidence": safe_mean(sub["assignment_confidence"]),
                "ambiguous_event_share": safe_mean(sub["assignment_ambiguity_flag"].astype(float)),
                "manual_review_event_share": safe_mean(sub["manual_review_flag"].astype(float)),
                "core_internal_share": core_internal_share,
                "core_to_escape_share": core_to_escape_share,
                "mean_effective_temporal_depth": safe_mean(sub["effective_temporal_depth"]),
                "mean_forgetting_share": safe_mean(sub["forgetting_share"]),
                "dominant_compression_state_mode": mode_or_na(sub["dominant_compression_state"]),
                "rollup_version": ROLLUP_VERSION,
                "source_feature_version": mode_or_na(sub["source_feature_version"]),
                "source_assignment_version": mode_or_na(sub["assignment_version"]),
            }
        )

    out = pd.DataFrame(rows).sort_values("trajectory_id").reset_index(drop=True)
    return out


def validate_trajectory_family_rollup(
    rollup_df: pd.DataFrame,
    features_df: pd.DataFrame,
    assign_df: pd.DataFrame,
) -> List[str]:
    warnings: List[str] = []

    required_cols = [
        "trajectory_id",
        "n_events",
        "dominant_family",
        "dominant_family_share",
        "family_mixture_entropy",
        "family_mixture_flag",
        "mean_assignment_confidence",
        "ambiguous_event_share",
        "manual_review_event_share",
        "core_internal_share",
        "core_to_escape_share",
        "mean_effective_temporal_depth",
        "mean_forgetting_share",
        "dominant_compression_state_mode",
        "rollup_version",
        "source_feature_version",
        "source_assignment_version",
    ]
    missing = [c for c in required_cols if c not in rollup_df.columns]
    if missing:
        raise ValueError(f"trajectory_family_rollup is missing required columns: {missing}")

    if rollup_df["trajectory_id"].duplicated().any():
        raise ValueError("Duplicate trajectory_id values found in trajectory_family_rollup.")

    linked_feature_rows = features_df["trajectory_id"].notna().sum() if "trajectory_id" in features_df.columns else 0
    if linked_feature_rows == 0:
        warnings.append("No non-null trajectory_id values exist in event_family_features; rollup is empty by design.")

    if rollup_df.empty:
        return warnings

    bad_families = sorted(set(rollup_df["dominant_family"].dropna().astype(str)) - set(CANONICAL_ROUTE_CLASSES))
    if bad_families:
        warnings.append(f"Unexpected dominant_family values present: {bad_families}")

    for _, row in rollup_df.iterrows():
        ci = pd.to_numeric(pd.Series([row["core_internal_share"]]), errors="coerce").iloc[0]
        ce = pd.to_numeric(pd.Series([row["core_to_escape_share"]]), errors="coerce").iloc[0]
        if pd.notna(ci) and pd.notna(ce):
            total = ci + ce
            if abs(total - 1.0) > 1e-6:
                warnings.append(
                    f"{row['trajectory_id']}: core_internal_share + core_to_escape_share = {total:.6f}, expected ~1.0."
                )

    # Check event counts against linked events.
    linked = assign_df.merge(
        features_df[["event_id", "trajectory_id"]],
        on="event_id",
        how="inner",
        validate="one_to_one",
    )
    linked = linked[linked["trajectory_id"].notna()].copy()
    linked["trajectory_id"] = linked["trajectory_id"].astype(str).str.strip()
    linked = linked[linked["trajectory_id"] != ""].copy()

    expected_counts = (
        linked.groupby("trajectory_id", dropna=False)
        .size()
        .rename("expected_n_events")
        .reset_index()
    )
    merged_counts = rollup_df.merge(expected_counts, on="trajectory_id", how="left", validate="one_to_one")
    bad = pd.to_numeric(merged_counts["n_events"], errors="coerce") != pd.to_numeric(
        merged_counts["expected_n_events"], errors="coerce"
    )
    if bad.any():
        warnings.append(f"n_events mismatch found for {int(bad.sum())} trajectories.")

    return warnings


def write_validation_report(
    path: Path,
    rollup_df: pd.DataFrame,
    features_df: pd.DataFrame,
    warnings: List[str],
) -> None:
    linked_rows = 0
    linked_trajectories = 0
    if "trajectory_id" in features_df.columns:
        tmp = features_df[features_df["trajectory_id"].notna()].copy()
        if not tmp.empty:
            tmp["trajectory_id"] = tmp["trajectory_id"].astype(str).str.strip()
            tmp = tmp[tmp["trajectory_id"] != ""]
            linked_rows = len(tmp)
            linked_trajectories = tmp["trajectory_id"].nunique()

    lines: list[str] = []
    lines.append("trajectory_family_rollup validation report")
    lines.append(f"rollup_version: {ROLLUP_VERSION}")
    lines.append(f"n_rows: {len(rollup_df)}")
    lines.append(f"linked_event_rows: {linked_rows}")
    lines.append(f"linked_trajectories: {linked_trajectories}")
    lines.append("")

    if not rollup_df.empty:
        dom_fams = sorted(rollup_df["dominant_family"].dropna().astype(str).unique().tolist())
        lines.append(f"dominant_families: {', '.join(dom_fams)}")
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

    out_csv = repo_root / "outputs" / "canonical" / "trajectory_family_rollup.csv"
    out_report = repo_root / "outputs" / "canonical" / "validation" / "trajectory_family_rollup_validation.txt"

    try:
        features_df = load_csv(features_path)
        assign_df = load_csv(assign_path)

        rollup_df = build_trajectory_family_rollup(features_df, assign_df)
        warnings = validate_trajectory_family_rollup(rollup_df, features_df, assign_df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rollup_df.to_csv(out_csv, index=False)
        write_validation_report(out_report, rollup_df, features_df, warnings)

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
