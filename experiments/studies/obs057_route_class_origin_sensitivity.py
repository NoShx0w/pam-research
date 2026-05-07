#!/usr/bin/env python3
"""
OBS-057 — Route-class origin sensitivity.

Purpose
-------
Profile and stress-test the small OBS-022 / OBS-030 route-class origin substrate.

OBS-056 established that downstream route_class labels are assigned in OBS-030
from OBS-022 scene-route metadata:

- branch_exit:
    is_branch_away == 1
- stable_seam_corridor:
    is_representative == 1 and path_family == stable_seam_corridor
- reorganization_heavy:
    is_representative == 1 and path_family == reorganization_heavy

This study asks whether that small representative/branch path substrate is
stable enough to support downstream transition/motif/generator analyses.

This file intentionally starts with:
1. origin-path profiling
2. OBS-030-style transition signature computation
3. leave-one-out sensitivity over the selected origin paths
4. full transition-distribution drift under leave-one-out perturbation

Matched-decoy replacement is intentionally left for a later pass.

Inputs
------
outputs/obs022_scene_bundle/scene_routes.csv
outputs/obs028c_canonical_seam_bundle/seam_nodes.csv

Outputs
-------
outputs/obs057_route_class_origin_sensitivity/
  obs057_origin_path_profile.csv
  obs057_origin_class_summary.csv
  obs057_real_transition_signature.csv
  obs057_real_transition_distribution.csv
  obs057_leave_one_out_transition_signature.csv
  obs057_leave_one_out_transition_distribution.csv
  obs057_leave_one_out_drift.csv
  obs057_route_class_origin_sensitivity_report.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

STATE_ORDER = [
    "off_seam",
    "post_exit",
    "seam_resident_low",
    "mixed_seam",
    "relational_flank",
    "anisotropy_flank",
    "shared_core",
]

TARGET_TRANSITIONS = {
    "relational_release": ("relational_flank", "post_exit"),
    "anisotropy_release": ("anisotropy_flank", "post_exit"),
    "core_retention": ("shared_core", "shared_core"),
    "core_to_low": ("shared_core", "seam_resident_low"),
    "off_reentry": ("off_seam", "relational_flank"),
}


@dataclass(frozen=True)
class Config:
    scene_routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    seam_nodes_csv: str = "outputs/obs028c_canonical_seam_bundle/seam_nodes.csv"
    outdir: str = "outputs/obs057_route_class_origin_sensitivity"
    corpus_label: str = ""
    seam_threshold: float = 0.15
    post_exit_threshold: float = 0.50
    profile_only: bool = False


def read_csv_numeric(path: str | Path, text_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    text = set(text_cols)
    for col in df.columns:
        if col not in text:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors OBS-030 classify_routes().

    route_class is assigned from OBS-022 scene-route metadata:
    - branch_exit from is_branch_away
    - stable/reorg from representative path_family
    - other otherwise
    """
    out = routes.copy()
    fam = out.get("path_family", pd.Series(index=out.index, dtype=object)).astype(str)
    is_branch = pd.to_numeric(out.get("is_branch_away", 0), errors="coerce").fillna(0).eq(1)
    is_rep = pd.to_numeric(out.get("is_representative", 0), errors="coerce").fillna(0).eq(1)

    out["route_class"] = np.select(
        [
            is_branch,
            is_rep & fam.eq("stable_seam_corridor"),
            is_rep & fam.eq("reorganization_heavy"),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )
    return out


def merge_seam_node_fields(routes: pd.DataFrame, seam_nodes: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        c
        for c in [
            "node_id",
            "r",
            "alpha",
            "mds1",
            "mds2",
            "distance_to_seam",
            "neighbor_direction_mismatch_mean",
            "sym_traceless_norm",
            "anisotropy_hotspot",
            "relational_hotspot",
            "shared_hotspot",
            "hotspot_class",
            "seam_band",
        ]
        if c in seam_nodes.columns
    ]

    if "node_id" not in keep_cols:
        return routes.copy()

    seam_use = seam_nodes[keep_cols].drop_duplicates(subset=["node_id"]).copy()
    seam_use = seam_use.rename(columns={c: f"{c}_bundle" for c in seam_use.columns if c != "node_id"})

    out = routes.merge(seam_use, on="node_id", how="left")

    for base_col in [
        "r",
        "alpha",
        "mds1",
        "mds2",
        "distance_to_seam",
        "neighbor_direction_mismatch_mean",
        "sym_traceless_norm",
        "anisotropy_hotspot",
        "relational_hotspot",
        "shared_hotspot",
        "hotspot_class",
        "seam_band",
    ]:
        bundle_col = f"{base_col}_bundle"
        if base_col not in out.columns and bundle_col in out.columns:
            out[base_col] = out[bundle_col]
        elif base_col in out.columns and bundle_col in out.columns:
            out[base_col] = out[base_col].where(out[base_col].notna(), out[bundle_col])

    drop_cols = [c for c in out.columns if c.endswith("_bundle")]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out


def assign_state_type(row: pd.Series, cfg: Config) -> str:
    """
    Mirrors OBS-030 state typing.
    """
    d2s = pd.to_numeric(row.get("distance_to_seam"), errors="coerce")
    rel = pd.to_numeric(row.get("neighbor_direction_mismatch_mean"), errors="coerce")
    aniso = pd.to_numeric(row.get("sym_traceless_norm"), errors="coerce")
    shared = int(pd.to_numeric(row.get("shared_hotspot"), errors="coerce") == 1)
    rel_hot = int(pd.to_numeric(row.get("relational_hotspot"), errors="coerce") == 1)
    aniso_hot = int(pd.to_numeric(row.get("anisotropy_hotspot"), errors="coerce") == 1)

    if pd.isna(d2s):
        return "off_seam"

    if d2s > cfg.post_exit_threshold:
        return "post_exit"

    if d2s > cfg.seam_threshold:
        return "off_seam"

    if shared == 1:
        return "shared_core"
    if rel_hot == 1 and aniso_hot == 0:
        return "relational_flank"
    if aniso_hot == 1 and rel_hot == 0:
        return "anisotropy_flank"
    if rel_hot == 1 and aniso_hot == 1:
        return "mixed_seam"

    if np.isfinite(rel) or np.isfinite(aniso):
        return "seam_resident_low"

    return "mixed_seam"


def selected_origin_routes(routes: pd.DataFrame) -> pd.DataFrame:
    return routes[routes["route_class"].isin(CLASS_ORDER)].copy()


def safe_mean(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def safe_min(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.min()) if x.notna().any() else float("nan")


def safe_max(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.max()) if x.notna().any() else float("nan")


def first_or_nan(s: pd.Series):
    return s.iloc[0] if len(s) else np.nan


def last_or_nan(s: pd.Series):
    return s.iloc[-1] if len(s) else np.nan


def transition_key(state_from: str, state_to: str) -> str:
    return f"{state_from} -> {state_to}"


def build_origin_path_profile(routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    selected = selected_origin_routes(routes)

    rows = []
    corpus = cfg.corpus_label or "unspecified"

    for path_id, grp in selected.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy()
        route_class = str(first_or_nan(grp["route_class"]))
        path_family = str(first_or_nan(grp["path_family"])) if "path_family" in grp.columns else ""

        rows.append(
            {
                "corpus": corpus,
                "path_id": path_id,
                "route_class": route_class,
                "path_family": path_family,
                "is_branch_away": int(pd.to_numeric(grp.get("is_branch_away", 0), errors="coerce").fillna(0).max()),
                "is_representative": int(pd.to_numeric(grp.get("is_representative", 0), errors="coerce").fillna(0).max()),
                "n_steps": int(len(grp)),
                "start_node_id": first_or_nan(grp["node_id"]) if "node_id" in grp.columns else np.nan,
                "end_node_id": last_or_nan(grp["node_id"]) if "node_id" in grp.columns else np.nan,
                "mean_distance_to_seam": safe_mean(grp.get("distance_to_seam", pd.Series(dtype=float))),
                "min_distance_to_seam": safe_min(grp.get("distance_to_seam", pd.Series(dtype=float))),
                "max_distance_to_seam": safe_max(grp.get("distance_to_seam", pd.Series(dtype=float))),
                "mean_lazarus_score": safe_mean(grp.get("lazarus_score", pd.Series(dtype=float))),
                "max_lazarus_score": safe_max(grp.get("lazarus_score", pd.Series(dtype=float))),
                "mean_response_strength": safe_mean(grp.get("response_strength", pd.Series(dtype=float))),
                "max_response_strength": safe_max(grp.get("response_strength", pd.Series(dtype=float))),
                "mean_signed_phase": safe_mean(grp.get("signed_phase", pd.Series(dtype=float))),
                "min_signed_phase": safe_min(grp.get("signed_phase", pd.Series(dtype=float))),
                "max_signed_phase": safe_max(grp.get("signed_phase", pd.Series(dtype=float))),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out["route_class_order"] = out["route_class"].map({c: i for i, c in enumerate(CLASS_ORDER)})
        out = out.sort_values(["route_class_order", "path_id"]).drop(columns=["route_class_order"]).reset_index(drop=True)

    return out


def build_origin_class_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        sub = profile[profile["route_class"] == cls].copy()

        rows.append(
            {
                "route_class": cls,
                "n_paths": int(len(sub)),
                "total_steps": int(sub["n_steps"].sum()) if len(sub) else 0,
                "mean_steps_per_path": safe_mean(sub["n_steps"]) if len(sub) else float("nan"),
                "min_steps_per_path": safe_min(sub["n_steps"]) if len(sub) else float("nan"),
                "max_steps_per_path": safe_max(sub["n_steps"]) if len(sub) else float("nan"),
                "mean_distance_to_seam": safe_mean(sub["mean_distance_to_seam"]) if len(sub) else float("nan"),
                "min_distance_to_seam": safe_min(sub["min_distance_to_seam"]) if len(sub) else float("nan"),
                "max_distance_to_seam": safe_max(sub["max_distance_to_seam"]) if len(sub) else float("nan"),
                "mean_lazarus_score": safe_mean(sub["mean_lazarus_score"]) if len(sub) else float("nan"),
                "max_lazarus_score": safe_max(sub["max_lazarus_score"]) if len(sub) else float("nan"),
                "mean_response_strength": safe_mean(sub["mean_response_strength"]) if len(sub) else float("nan"),
                "max_response_strength": safe_max(sub["max_response_strength"]) if len(sub) else float("nan"),
                "mean_signed_phase": safe_mean(sub["mean_signed_phase"]) if len(sub) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def build_transition_steps(routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = selected_origin_routes(routes).copy()
    work = work.sort_values(["path_id", "step"]).reset_index(drop=True)
    work["state_type"] = work.apply(lambda row: assign_state_type(row, cfg), axis=1)

    rows = []
    for path_id, grp in work.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            rows.append(
                {
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "path_family": a.get("path_family", np.nan),
                    "step": pd.to_numeric(a.get("step"), errors="coerce"),
                    "node_id": pd.to_numeric(a.get("node_id"), errors="coerce"),
                    "next_node_id": pd.to_numeric(b.get("node_id"), errors="coerce"),
                    "state_from": a["state_type"],
                    "state_to": b["state_type"],
                    "transition": transition_key(str(a["state_type"]), str(b["state_type"])),
                    "distance_from": pd.to_numeric(a.get("distance_to_seam"), errors="coerce"),
                    "distance_to": pd.to_numeric(b.get("distance_to_seam"), errors="coerce"),
                    "relational_from": pd.to_numeric(a.get("neighbor_direction_mismatch_mean"), errors="coerce"),
                    "anisotropy_from": pd.to_numeric(a.get("sym_traceless_norm"), errors="coerce"),
                    "dx": pd.to_numeric(b.get("mds1"), errors="coerce") - pd.to_numeric(a.get("mds1"), errors="coerce"),
                    "dy": pd.to_numeric(b.get("mds2"), errors="coerce") - pd.to_numeric(a.get("mds2"), errors="coerce"),
                }
            )

    return pd.DataFrame(rows)


def build_transition_distribution(transition_steps: pd.DataFrame, run_id: str, removed_path_id=None, removed_route_class: str = "") -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        sub = transition_steps[transition_steps["route_class"] == cls].copy()
        total = int(len(sub))

        if total == 0:
            continue

        counts = (
            sub.groupby(["state_from", "state_to", "transition"], as_index=False)
            .agg(n_transitions=("path_id", "size"), n_paths=("path_id", "nunique"))
            .sort_values("n_transitions", ascending=False)
            .reset_index(drop=True)
        )
        counts["transition_share"] = counts["n_transitions"] / total
        counts["route_class"] = cls
        counts["run_id"] = run_id
        counts["removed_path_id"] = removed_path_id if removed_path_id is not None else ""
        counts["removed_route_class"] = removed_route_class
        rows.append(counts)

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "removed_route_class",
                "removed_path_id",
                "route_class",
                "state_from",
                "state_to",
                "transition",
                "n_transitions",
                "n_paths",
                "transition_share",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "run_id",
            "removed_route_class",
            "removed_path_id",
            "route_class",
            "state_from",
            "state_to",
            "transition",
            "n_transitions",
            "n_paths",
            "transition_share",
        ]
    ]


def build_transition_signature(transition_steps: pd.DataFrame, run_id: str, removed_path_id=None, removed_route_class: str = "") -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        sub = transition_steps[transition_steps["route_class"] == cls].copy()
        n_transitions = int(len(sub))
        n_paths = int(sub["path_id"].nunique()) if len(sub) else 0

        row = {
            "run_id": run_id,
            "removed_route_class": removed_route_class,
            "removed_path_id": removed_path_id if removed_path_id is not None else "",
            "route_class": cls,
            "n_paths": n_paths,
            "n_transitions": n_transitions,
        }

        for label, (a, b) in TARGET_TRANSITIONS.items():
            hit = sub[(sub["state_from"] == a) & (sub["state_to"] == b)]
            count = int(len(hit))
            row[f"{label}_count"] = count
            row[f"{label}_share"] = float(count / n_transitions) if n_transitions else 0.0

        if len(sub):
            top = (
                sub.groupby(["state_from", "state_to", "transition"], as_index=False)
                .agg(n=("path_id", "size"))
                .sort_values("n", ascending=False)
                .head(3)
            )
            for i in range(3):
                if len(top) > i:
                    row[f"top_transition_{i+1}"] = str(top.iloc[i]["transition"])
                    row[f"top_transition_{i+1}_count"] = int(top.iloc[i]["n"])
                    row[f"top_transition_{i+1}_share"] = float(top.iloc[i]["n"] / n_transitions) if n_transitions else 0.0
                else:
                    row[f"top_transition_{i+1}"] = ""
                    row[f"top_transition_{i+1}_count"] = 0
                    row[f"top_transition_{i+1}_share"] = 0.0
        else:
            for i in range(3):
                row[f"top_transition_{i+1}"] = ""
                row[f"top_transition_{i+1}_count"] = 0
                row[f"top_transition_{i+1}_share"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def distribution_map(dist: pd.DataFrame, route_class: str) -> dict[str, float]:
    sub = dist[dist["route_class"] == route_class]
    return {
        str(row["transition"]): float(row["transition_share"])
        for _, row in sub.iterrows()
    }


def top_transitions(dist: pd.DataFrame, route_class: str, k: int = 3) -> list[str]:
    sub = (
        dist[dist["route_class"] == route_class]
        .sort_values("transition_share", ascending=False)
        .head(k)
    )
    return [str(x) for x in sub["transition"].tolist()]


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def top_k_overlap(a: list[str], b: list[str], k: int = 3) -> int:
    return len(set(a[:k]) & set(b[:k]))


def add_distribution_drift_metrics(
    drift: dict,
    *,
    baseline_dist: pd.DataFrame,
    after_dist: pd.DataFrame,
    route_class: str,
) -> dict:
    p = distribution_map(baseline_dist, route_class)
    q = distribution_map(after_dist, route_class)

    base_top3 = top_transitions(baseline_dist, route_class, k=3)
    after_top3 = top_transitions(after_dist, route_class, k=3)

    base_top1 = base_top3[0] if base_top3 else ""
    after_top1 = after_top3[0] if after_top3 else ""

    drift["transition_tv_distance"] = total_variation_distance(p, q)
    drift["top1_transition_baseline"] = base_top1
    drift["top1_transition_after"] = after_top1
    drift["top1_changed"] = int(base_top1 != after_top1)
    drift["top3_transition_baseline"] = " | ".join(base_top3)
    drift["top3_transition_after"] = " | ".join(after_top3)
    drift["top3_overlap"] = top_k_overlap(base_top3, after_top3, k=3)
    drift["top3_overlap_share"] = drift["top3_overlap"] / 3.0 if base_top3 or after_top3 else 0.0

    return drift


def run_leave_one_out(routes: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = selected_origin_routes(routes)

    real_steps = build_transition_steps(selected, cfg)
    real_sig = build_transition_signature(real_steps, run_id="real")
    real_dist = build_transition_distribution(real_steps, run_id="real")

    all_sigs = []
    all_dists = []
    drift_rows = []

    metric_cols = [f"{label}_share" for label in TARGET_TRANSITIONS]

    for cls in CLASS_ORDER:
        class_paths = (
            selected[selected["route_class"] == cls]["path_id"]
            .drop_duplicates()
            .tolist()
        )

        real_row = real_sig[real_sig["route_class"] == cls].iloc[0].to_dict()

        for path_id in class_paths:
            perturbed = selected[selected["path_id"] != path_id].copy()
            steps = build_transition_steps(perturbed, cfg)

            run_id = f"loo_{cls}_{path_id}"

            sig = build_transition_signature(
                steps,
                run_id=run_id,
                removed_path_id=path_id,
                removed_route_class=cls,
            )
            dist = build_transition_distribution(
                steps,
                run_id=run_id,
                removed_path_id=path_id,
                removed_route_class=cls,
            )

            all_sigs.append(sig)
            all_dists.append(dist)

            row_after = sig[sig["route_class"] == cls].iloc[0].to_dict()

            drift = {
                "removed_route_class": cls,
                "removed_path_id": path_id,
                "baseline_n_paths": int(real_row["n_paths"]),
                "after_n_paths": int(row_after["n_paths"]),
                "baseline_n_transitions": int(real_row["n_transitions"]),
                "after_n_transitions": int(row_after["n_transitions"]),
            }

            abs_diffs = []
            for col in metric_cols:
                before = float(real_row.get(col, 0.0))
                after = float(row_after.get(col, 0.0))
                diff = after - before
                drift[f"{col}_baseline"] = before
                drift[f"{col}_after"] = after
                drift[f"{col}_diff"] = diff
                drift[f"{col}_abs_diff"] = abs(diff)
                abs_diffs.append(abs(diff))

            drift["max_abs_target_share_drift"] = max(abs_diffs) if abs_diffs else 0.0
            drift["mean_abs_target_share_drift"] = float(np.mean(abs_diffs)) if abs_diffs else 0.0

            drift = add_distribution_drift_metrics(
                drift,
                baseline_dist=real_dist,
                after_dist=dist,
                route_class=cls,
            )

            drift_rows.append(drift)

    loo_sig = pd.concat(all_sigs, ignore_index=True) if all_sigs else pd.DataFrame()
    loo_dist = pd.concat(all_dists, ignore_index=True) if all_dists else pd.DataFrame()
    drift_df = pd.DataFrame(drift_rows)

    if len(drift_df):
        drift_df = drift_df.sort_values(
            ["removed_route_class", "transition_tv_distance", "max_abs_target_share_drift", "removed_path_id"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)

    return loo_sig, loo_dist, drift_df


def build_report(
    cfg: Config,
    profile: pd.DataFrame,
    class_summary: pd.DataFrame,
    real_sig: pd.DataFrame | None,
    real_dist: pd.DataFrame | None,
    loo_drift: pd.DataFrame | None,
) -> str:
    lines = [
        "# OBS-057 — Route-class origin sensitivity",
        "",
        f"Corpus label: `{cfg.corpus_label or 'unspecified'}`",
        "",
        "## Purpose",
        "",
        "OBS-057 profiles and stress-tests the small OBS-022 / OBS-030 route-class origin substrate.",
        "",
        "OBS-056 established that `route_class` is assigned in OBS-030 from OBS-022 scene-route metadata:",
        "",
        "- `branch_exit`: `is_branch_away == 1`",
        "- `stable_seam_corridor`: `is_representative == 1` and `path_family == stable_seam_corridor`",
        "- `reorganization_heavy`: `is_representative == 1` and `path_family == reorganization_heavy`",
        "",
        "OBS-057 asks whether this representative/branch path substrate is stable enough to support downstream transition, motif, generator, proto-groupoid, and gateway analyses.",
        "",
        "## Origin path basis",
        "",
    ]

    for _, row in class_summary.iterrows():
        lines.extend(
            [
                f"### {row['route_class']}",
                "",
                f"- n_paths: {int(row['n_paths'])}",
                f"- total_steps: {int(row['total_steps'])}",
                f"- mean_steps_per_path: {float(row['mean_steps_per_path']):.4f}",
                f"- step_count_range: {float(row['min_steps_per_path']):.0f}–{float(row['max_steps_per_path']):.0f}",
                f"- mean_distance_to_seam: {float(row['mean_distance_to_seam']):.6f}",
                f"- mean_lazarus_score: {float(row['mean_lazarus_score']):.6f}",
                f"- mean_response_strength: {float(row['mean_response_strength']):.6f}",
                "",
            ]
        )

    if real_sig is not None and len(real_sig):
        lines.extend(["## Baseline target-transition signatures", ""])
        for _, row in real_sig.iterrows():
            lines.extend(
                [
                    f"### {row['route_class']}",
                    "",
                    f"- n_paths: {int(row['n_paths'])}",
                    f"- n_transitions: {int(row['n_transitions'])}",
                    f"- relational_release_share: {float(row['relational_release_share']):.6f}",
                    f"- anisotropy_release_share: {float(row['anisotropy_release_share']):.6f}",
                    f"- core_retention_share: {float(row['core_retention_share']):.6f}",
                    f"- core_to_low_share: {float(row['core_to_low_share']):.6f}",
                    f"- off_reentry_share: {float(row['off_reentry_share']):.6f}",
                    f"- top_transition_1: {row['top_transition_1']}",
                    "",
                ]
            )

    if real_dist is not None and len(real_dist):
        lines.extend(["## Baseline full transition-distribution top terms", ""])
        for cls in CLASS_ORDER:
            sub = real_dist[real_dist["route_class"] == cls].sort_values("transition_share", ascending=False).head(5)
            lines.extend([f"### {cls}", ""])
            for _, row in sub.iterrows():
                lines.append(
                    f"- {row['transition']}: n={int(row['n_transitions'])}, share={float(row['transition_share']):.6f}"
                )
            lines.append("")

    if loo_drift is not None and len(loo_drift):
        lines.extend(["## Leave-one-out drift summary", ""])
        for cls in CLASS_ORDER:
            sub = loo_drift[loo_drift["removed_route_class"] == cls].copy()
            if len(sub) == 0:
                continue
            worst_tv = sub.sort_values("transition_tv_distance", ascending=False).iloc[0]
            worst_target = sub.sort_values("max_abs_target_share_drift", ascending=False).iloc[0]
            lines.extend(
                [
                    f"### {cls}",
                    "",
                    f"- leave_one_out_runs: {len(sub)}",
                    f"- worst_tv_removed_path_id: `{worst_tv['removed_path_id']}`",
                    f"- worst_transition_tv_distance: {float(worst_tv['transition_tv_distance']):.6f}",
                    f"- mean_transition_tv_distance: {float(sub['transition_tv_distance'].mean()):.6f}",
                    f"- top1_changed_count: {int(sub['top1_changed'].sum())}",
                    f"- mean_top3_overlap_share: {float(sub['top3_overlap_share'].mean()):.6f}",
                    f"- worst_target_removed_path_id: `{worst_target['removed_path_id']}`",
                    f"- worst_max_abs_target_share_drift: {float(worst_target['max_abs_target_share_drift']):.6f}",
                    f"- mean_abs_target_share_drift: {float(sub['mean_abs_target_share_drift'].mean()):.6f}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "This study does not validate route classes globally.",
            "",
            "It tests the sensitivity of the small representative/branch origin substrate used by OBS-030.",
            "",
            "The target-transition shares are OBS-030-theory-salient metrics, but they can be sparse.",
            "",
            "The full transition-distribution drift metrics are therefore included to detect origin sensitivity even when hand-selected target transitions are zero or rare.",
            "",
            "Matching route-class cardinality across artifact stores does not by itself establish matching path content, path geometry, or downstream signature equivalence.",
            "",
            "## Next steps",
            "",
            "- Compare this profile across C and Cp artifact stores.",
            "- Inspect worst leave-one-out paths by class.",
            "- Add matched-decoy replacement within `path_family` after leave-one-out behavior is understood.",
            "- Only then extend perturbations into downstream motif/generator/proto-groupoid/gateway layers.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile and test OBS-030 route-class origin sensitivity.")
    parser.add_argument("--scene-routes-csv", default=Config.scene_routes_csv)
    parser.add_argument("--seam-nodes-csv", default=Config.seam_nodes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--corpus-label", default=Config.corpus_label)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--post-exit-threshold", type=float, default=Config.post_exit_threshold)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        scene_routes_csv=args.scene_routes_csv,
        seam_nodes_csv=args.seam_nodes_csv,
        outdir=args.outdir,
        corpus_label=args.corpus_label,
        seam_threshold=args.seam_threshold,
        post_exit_threshold=args.post_exit_threshold,
        profile_only=args.profile_only,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    routes = read_csv_numeric(
        cfg.scene_routes_csv,
        text_cols={"path_id", "path_family", "hotspot_class", "seam_band"},
    )
    seam_nodes = read_csv_numeric(
        cfg.seam_nodes_csv,
        text_cols={"hotspot_class", "seam_band"},
    )

    routes = classify_routes(routes)
    routes = merge_seam_node_fields(routes, seam_nodes)

    profile = build_origin_path_profile(routes, cfg)
    class_summary = build_origin_class_summary(profile)

    profile_csv = outdir / "obs057_origin_path_profile.csv"
    class_csv = outdir / "obs057_origin_class_summary.csv"
    real_sig_csv = outdir / "obs057_real_transition_signature.csv"
    real_dist_csv = outdir / "obs057_real_transition_distribution.csv"
    loo_sig_csv = outdir / "obs057_leave_one_out_transition_signature.csv"
    loo_dist_csv = outdir / "obs057_leave_one_out_transition_distribution.csv"
    loo_drift_csv = outdir / "obs057_leave_one_out_drift.csv"
    report_md = outdir / "obs057_route_class_origin_sensitivity_report.md"

    profile.to_csv(profile_csv, index=False)
    class_summary.to_csv(class_csv, index=False)

    real_sig = None
    real_dist = None
    loo_sig = None
    loo_dist = None
    loo_drift = None

    if not cfg.profile_only:
        transition_steps = build_transition_steps(routes, cfg)
        real_sig = build_transition_signature(transition_steps, run_id="real")
        real_dist = build_transition_distribution(transition_steps, run_id="real")
        loo_sig, loo_dist, loo_drift = run_leave_one_out(routes, cfg)

        real_sig.to_csv(real_sig_csv, index=False)
        real_dist.to_csv(real_dist_csv, index=False)
        loo_sig.to_csv(loo_sig_csv, index=False)
        loo_dist.to_csv(loo_dist_csv, index=False)
        loo_drift.to_csv(loo_drift_csv, index=False)

    report_md.write_text(
        build_report(cfg, profile, class_summary, real_sig, real_dist, loo_drift),
        encoding="utf-8",
    )

    print(profile_csv)
    print(class_csv)
    if not cfg.profile_only:
        print(real_sig_csv)
        print(real_dist_csv)
        print(loo_sig_csv)
        print(loo_dist_csv)
        print(loo_drift_csv)
    print(report_md)


if __name__ == "__main__":
    main()
