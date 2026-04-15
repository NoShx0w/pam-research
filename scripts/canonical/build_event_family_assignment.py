#!/usr/bin/env python3
"""
Build the canonical event-level family assignment table.

This script converts normalized event rows into a canonical assignment artifact:

    outputs/canonical/event_family_assignment.csv

It also writes a small human-readable validation report:

    outputs/canonical/validation/event_family_assignment_validation.txt

Current scope
-------------
Version 1 is intentionally conservative:
- assignment is derived from the observed normalized `route_class`
- no de novo classifier is claimed
- confidence and ambiguity are diagnostic, not probabilistic
- ambiguity is used to flag weakly supported or weakly matched rows
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

ASSIGNMENT_METHOD = "observed_route_class_with_diagnostic_confidence"
ASSIGNMENT_VERSION = "event_family_assignment_v1"


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def match_branch_exit(row: pd.Series) -> float:
    score = 0.0
    max_score = 4.0

    if str(row.get("memory_regime_label")) == "directed/downstream":
        score += 1.5

    etd = safe_float(row.get("effective_temporal_depth"))
    if etd is not None and etd <= 0.5:
        score += 1.0

    forgetting = safe_float(row.get("forgetting_share"))
    if forgetting is not None and forgetting <= 0.15:
        score += 1.0

    if str(row.get("dominant_compression_state")) == "escape":
        score += 0.5

    return score / max_score


def match_stable_seam_corridor(row: pd.Series) -> float:
    score = 0.0
    max_score = 4.5

    if str(row.get("memory_regime_label")) == "local gateway":
        score += 1.5

    etd = safe_float(row.get("effective_temporal_depth"))
    if etd is not None and 0.5 <= etd <= 1.5:
        score += 1.0

    local_gateway_strength = safe_float(row.get("local_gateway_strength"))
    if local_gateway_strength is not None and local_gateway_strength >= 0.72:
        score += 1.0

    forgetting = safe_float(row.get("forgetting_share"))
    if forgetting is not None and 0.2 <= forgetting <= 0.55:
        score += 0.5

    if str(row.get("dominant_compression_state")) in {"core", "escape"}:
        score += 0.5

    return score / max_score


def match_reorganization_heavy(row: pd.Series) -> float:
    score = 0.0
    max_score = 4.5

    if str(row.get("memory_regime_label")) == "path-context":
        score += 1.5

    etd = safe_float(row.get("effective_temporal_depth"))
    if etd is not None and etd >= 2.0:
        score += 1.0

    forgetting = safe_float(row.get("forgetting_share"))
    if forgetting is not None and forgetting >= 0.55:
        score += 1.0

    if str(row.get("dominant_compression_state")) == "core":
        score += 0.5

    # gateway_context rows are especially natural for reorganization-heavy
    if str(row.get("source_row_type")) == "gateway_context":
        score += 0.5

    return score / max_score


def compute_family_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["branch_exit_score"] = out.apply(match_branch_exit, axis=1)
    out["stable_seam_corridor_score"] = out.apply(match_stable_seam_corridor, axis=1)
    out["reorganization_heavy_score"] = out.apply(match_reorganization_heavy, axis=1)
    return out


def choose_runner_up(row: pd.Series) -> tuple[str | pd.NA, float | pd.NA]:
    scores = {
        "branch_exit": safe_float(row["branch_exit_score"]) or 0.0,
        "stable_seam_corridor": safe_float(row["stable_seam_corridor_score"]) or 0.0,
        "reorganization_heavy": safe_float(row["reorganization_heavy_score"]) or 0.0,
    }
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ordered) < 2:
        return pd.NA, pd.NA
    runner_up_family, runner_up_score = ordered[1]
    return runner_up_family, runner_up_score


def diagnostic_confidence(row: pd.Series) -> tuple[float, bool, bool, str | pd.NA, str | pd.NA]:
    assigned = str(row["route_class"])

    score_cols = {
        "branch_exit": "branch_exit_score",
        "stable_seam_corridor": "stable_seam_corridor_score",
        "reorganization_heavy": "reorganization_heavy_score",
    }

    assigned_score = safe_float(row.get(score_cols[assigned]))
    if assigned_score is None:
        return 0.0, True, True, pd.NA, pd.NA

    runner_up_family, runner_up_score = choose_runner_up(row)
    runner_up_score_f = None if pd.isna(runner_up_score) else float(runner_up_score)
    margin = 0.0 if runner_up_score_f is None else assigned_score - runner_up_score_f

    missing_core_support = any(
        pd.isna(row.get(col))
        for col in [
            "local_gateway_strength",
            "effective_temporal_depth",
            "forgetting_share",
            "memory_regime_label",
        ]
    )

    ambiguity = False
    manual_review = False

    if missing_core_support:
        ambiguity = True

    if margin < 0.15:
        ambiguity = True

    if assigned_score < 0.55:
        ambiguity = True
        manual_review = True

    confidence = max(0.0, min(1.0, 0.7 * assigned_score + 0.3 * max(margin, 0.0)))
    if missing_core_support:
        confidence = min(confidence, 0.6)

    primary_discriminator = pd.NA
    secondary_discriminator = pd.NA

    if assigned == "branch_exit":
        primary_discriminator = "memory_regime_label"
        secondary_discriminator = "forgetting_share"
    elif assigned == "stable_seam_corridor":
        primary_discriminator = "local_gateway_strength"
        secondary_discriminator = "effective_temporal_depth"
    elif assigned == "reorganization_heavy":
        primary_discriminator = "memory_regime_label"
        secondary_discriminator = "forgetting_share"

    return confidence, ambiguity, manual_review, primary_discriminator, secondary_discriminator


def build_event_family_assignment(features_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        features_df,
        required=[
            "event_id",
            "route_class",
            "normalization_version",
            "source_row_type",
            "local_gateway_strength",
            "effective_temporal_depth",
            "forgetting_share",
            "memory_regime_label",
            "dominant_compression_state",
        ],
        name="event_family_features",
    )

    df = compute_family_scores(features_df)

    diagnostics = df.apply(diagnostic_confidence, axis=1, result_type="expand")
    diagnostics.columns = [
        "assignment_confidence",
        "assignment_ambiguity_flag",
        "manual_review_flag",
        "primary_discriminator",
        "secondary_discriminator",
    ]
    df = pd.concat([df, diagnostics], axis=1)

    runner_up = df.apply(choose_runner_up, axis=1, result_type="expand")
    runner_up.columns = ["runner_up_family", "_runner_up_score"]
    df = pd.concat([df, runner_up], axis=1)

    score_lookup = {
        "branch_exit": "branch_exit_score",
        "stable_seam_corridor": "stable_seam_corridor_score",
        "reorganization_heavy": "reorganization_heavy_score",
    }

    def calc_margin(row: pd.Series) -> float:
        assigned = str(row["route_class"])
        assigned_score = safe_float(row.get(score_lookup[assigned])) or 0.0
        runner_up_score = safe_float(row.get("_runner_up_score")) or 0.0
        return assigned_score - runner_up_score

    df["score_margin"] = df.apply(calc_margin, axis=1)

    out = pd.DataFrame()
    out["event_id"] = df["event_id"]
    out["assigned_family"] = df["route_class"]
    out["assignment_method"] = ASSIGNMENT_METHOD
    out["assignment_confidence"] = df["assignment_confidence"].round(6)
    out["assignment_ambiguity_flag"] = df["assignment_ambiguity_flag"]
    out["runner_up_family"] = df["runner_up_family"]
    out["score_margin"] = df["score_margin"].round(6)
    out["manual_review_flag"] = df["manual_review_flag"]
    out["primary_discriminator"] = df["primary_discriminator"]
    out["secondary_discriminator"] = df["secondary_discriminator"]
    out["branch_exit_score"] = df["branch_exit_score"].round(6)
    out["stable_seam_corridor_score"] = df["stable_seam_corridor_score"].round(6)
    out["reorganization_heavy_score"] = df["reorganization_heavy_score"].round(6)
    out["assignment_version"] = ASSIGNMENT_VERSION
    out["source_feature_version"] = df["normalization_version"]

    return out


def validate_event_family_assignment(assign_df: pd.DataFrame, features_df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

    require_columns(
        assign_df,
        required=[
            "event_id",
            "assigned_family",
            "assignment_method",
            "assignment_confidence",
            "assignment_ambiguity_flag",
            "assignment_version",
            "source_feature_version",
        ],
        name="event_family_assignment",
    )

    if assign_df["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id values found in event_family_assignment.")

    feature_ids = set(features_df["event_id"].astype(str))
    assign_ids = set(assign_df["event_id"].astype(str))

    missing_from_features = sorted(assign_ids - feature_ids)
    if missing_from_features:
        raise ValueError(
            f"Some assignment rows do not join to event_family_features: {missing_from_features[:5]}"
        )

    bad_families = sorted(set(assign_df["assigned_family"].dropna().astype(str)) - set(CANONICAL_ROUTE_CLASSES))
    if bad_families:
        warnings.append(f"Unexpected assigned_family values present: {bad_families}")

    if assign_df["assignment_confidence"].isna().any():
        warnings.append("Some rows have null assignment_confidence values.")

    low_conf_share = float((assign_df["assignment_confidence"] < 0.55).mean())
    ambiguity_share = float(assign_df["assignment_ambiguity_flag"].mean())

    if low_conf_share > 0.5:
        warnings.append(f"More than half of rows have low assignment confidence ({low_conf_share:.3f}).")

    if ambiguity_share > 0.75:
        warnings.append(f"Ambiguity share is very high ({ambiguity_share:.3f}).")

    # In v1, assigned family should mirror observed route_class.
    joined = assign_df.merge(
        features_df[["event_id", "route_class"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    mismatch = joined["assigned_family"].astype(str) != joined["route_class"].astype(str)
    if mismatch.any():
        warnings.append("Some assigned_family values do not match observed route_class in v1.")

    return warnings


def write_validation_report(path: Path, assign_df: pd.DataFrame, warnings: List[str]) -> None:
    lines: list[str] = []
    lines.append("event_family_assignment validation report")
    lines.append(f"assignment_version: {ASSIGNMENT_VERSION}")
    lines.append(f"n_rows: {len(assign_df)}")
    lines.append(f"assigned_families: {', '.join(sorted(assign_df['assigned_family'].dropna().astype(str).unique().tolist()))}")
    lines.append(f"assignment_method: {ASSIGNMENT_METHOD}")
    lines.append(f"ambiguity_share: {float(assign_df['assignment_ambiguity_flag'].mean()):.6f}")
    lines.append(f"manual_review_share: {float(assign_df['manual_review_flag'].mean()):.6f}")
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
    out_csv = repo_root / "outputs" / "canonical" / "event_family_assignment.csv"
    out_report = repo_root / "outputs" / "canonical" / "validation" / "event_family_assignment_validation.txt"

    try:
        features_df = load_csv(features_path)
        assign_df = build_event_family_assignment(features_df)
        warnings = validate_event_family_assignment(assign_df, features_df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        assign_df.to_csv(out_csv, index=False)
        write_validation_report(out_report, assign_df, warnings)

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
