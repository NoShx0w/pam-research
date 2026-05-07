#!/usr/bin/env python3
"""
OBS-060 — Non-exact decoy ensemble route-origin controls.

Purpose
-------
Test whether OBS-059 nearest non-exact decoy survival generalizes beyond the
single nearest eligible substitute.

OBS-059 established nearest non-exact decoy dominant-signature survival across
C and Cp. OBS-060 asks whether that survival persists across:

1. rank-k deterministic non-exact decoy replacement
2. random non-exact decoy replacement ensembles

Scope
-----
This study tests the OBS-022 / OBS-030 route-origin substrate only.

It does not validate route classes globally and does not yet propagate decoys
through downstream motif, generator, proto-groupoid, gateway, or canonical
family layers.

Inputs
------
outputs/obs022_scene_bundle/scene_routes.csv
outputs/obs028c_canonical_seam_bundle/seam_nodes.csv

Outputs
-------
outputs/obs060_nonexact_decoy_ensemble_controls/
  obs060_all_path_profile.csv
  obs060_selected_path_profile.csv
  obs060_decoy_candidate_pool.csv
  obs060_ranked_nonexact_decoy_candidates.csv
  obs060_baseline_transition_signature.csv
  obs060_baseline_transition_distribution.csv
  obs060_rank_replacement_runs.csv
  obs060_rank_replacement_drift.csv
  obs060_random_replacement_runs.csv
  obs060_random_replacement_drift.csv
  obs060_ensemble_summary.csv
  obs060_nonexact_decoy_ensemble_controls_report.md
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

TARGET_TRANSITIONS = {
    "relational_release": ("relational_flank", "post_exit"),
    "anisotropy_release": ("anisotropy_flank", "post_exit"),
    "core_retention": ("shared_core", "shared_core"),
    "core_to_low": ("shared_core", "seam_resident_low"),
    "off_reentry": ("off_seam", "relational_flank"),
}

MATCH_FEATURES = [
    "n_steps",
    "mean_distance_to_seam",
    "max_distance_to_seam",
    "mean_lazarus_score",
    "mean_response_strength",
]


@dataclass(frozen=True)
class Config:
    scene_routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    seam_nodes_csv: str = "outputs/obs028c_canonical_seam_bundle/seam_nodes.csv"
    outdir: str = "outputs/obs060_nonexact_decoy_ensemble_controls"
    corpus_label: str = ""
    seam_threshold: float = 0.15
    post_exit_threshold: float = 0.50
    allow_decoy_reuse: bool = False
    branch_decoy_same_family: bool = False
    min_matching_distance: float = 1e-9
    exact_match_tolerance: float = 1e-12
    rank_list: str = "1,2,3,5,10"
    n_iter: int = 250
    random_seed: int = 42
    max_candidates_per_selected: int = 100


def read_csv_numeric(path: str | Path, text_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    text = set(text_cols)
    for col in df.columns:
        if col not in text:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_rank_list(raw: str) -> list[int]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val > 0:
            out.append(val)
    return sorted(set(out)) or [1]


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
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


def assign_state_type(row: pd.Series, cfg: Config) -> str:
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


def build_path_profile(routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    corpus = cfg.corpus_label or "unspecified"

    for path_id, grp in routes.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy()
        path_family = str(first_or_nan(grp["path_family"])) if "path_family" in grp.columns else ""
        route_class = str(first_or_nan(grp["route_class"])) if "route_class" in grp.columns else "unknown"

        is_branch_away = int(pd.to_numeric(grp.get("is_branch_away", 0), errors="coerce").fillna(0).max())
        is_representative = int(pd.to_numeric(grp.get("is_representative", 0), errors="coerce").fillna(0).max())

        rows.append(
            {
                "corpus": corpus,
                "path_id": path_id,
                "route_class": route_class,
                "path_family": path_family,
                "is_branch_away": is_branch_away,
                "is_representative": is_representative,
                "is_selected_origin": int(route_class in CLASS_ORDER),
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
        out["route_class_order"] = out["route_class"].map({c: i for i, c in enumerate(CLASS_ORDER)}).fillna(99)
        out = (
            out.sort_values(["route_class_order", "path_family", "path_id"])
            .drop(columns=["route_class_order"])
            .reset_index(drop=True)
        )
    return out


def build_decoy_candidate_pool(profile: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    selected_ids = set(profile.loc[profile["is_selected_origin"] == 1, "path_id"].astype(str))

    for target_class in CLASS_ORDER:
        if target_class == "stable_seam_corridor":
            candidates = profile[
                (profile["path_family"] == "stable_seam_corridor")
                & (profile["is_representative"] == 0)
                & (profile["is_branch_away"] == 0)
            ].copy()

        elif target_class == "reorganization_heavy":
            candidates = profile[
                (profile["path_family"] == "reorganization_heavy")
                & (profile["is_representative"] == 0)
                & (profile["is_branch_away"] == 0)
            ].copy()

        elif target_class == "branch_exit":
            candidates = profile[
                (profile["is_branch_away"] == 0)
                & (~profile["path_id"].astype(str).isin(selected_ids))
            ].copy()

            if cfg.branch_decoy_same_family:
                selected_families = set(
                    profile.loc[profile["route_class"] == "branch_exit", "path_family"].astype(str).unique()
                )
                candidates = candidates[candidates["path_family"].astype(str).isin(selected_families)].copy()

        else:
            candidates = pd.DataFrame()

        candidates["target_route_class"] = target_class
        rows.append(candidates)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def feature_scale_table(profile: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for col in MATCH_FEATURES:
        x = pd.to_numeric(profile[col], errors="coerce")
        mu = float(x.mean()) if x.notna().any() else 0.0
        sd = float(x.std(ddof=0)) if x.notna().any() else 1.0
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        out[col] = (mu, sd)
    return out


def standardized_distance(a: pd.Series, b: pd.Series, scales: dict[str, tuple[float, float]]) -> float:
    diffs = []
    for col in MATCH_FEATURES:
        av = pd.to_numeric(a.get(col), errors="coerce")
        bv = pd.to_numeric(b.get(col), errors="coerce")
        if pd.isna(av) or pd.isna(bv):
            continue
        _, sd = scales[col]
        diffs.append(((float(av) - float(bv)) / sd) ** 2)

    if not diffs:
        return float("inf")
    return float(np.sqrt(np.sum(diffs)))


def raw_feature_max_abs_delta(row: dict | pd.Series) -> float:
    deltas = []
    for col in MATCH_FEATURES:
        delta_col = f"delta_{col}"
        val = pd.to_numeric(row.get(delta_col), errors="coerce")
        if pd.notna(val):
            deltas.append(abs(float(val)))
    return max(deltas) if deltas else float("inf")


def build_ranked_nonexact_candidates(
    profile: pd.DataFrame,
    pool: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    selected = selected.sort_values(["route_class", "path_id"]).reset_index(drop=True)

    scales = feature_scale_table(profile)
    rows = []

    for _, sel in selected.iterrows():
        cls = str(sel["route_class"])
        candidates = pool[pool["target_route_class"] == cls].copy()

        if len(candidates) == 0:
            continue

        distances = []
        for _, cand in candidates.iterrows():
            distances.append(standardized_distance(sel, cand, scales))

        candidates = candidates.copy()
        candidates["matching_distance"] = distances

        for col in MATCH_FEATURES:
            candidates[f"selected_{col}"] = sel.get(col, np.nan)
            candidates[f"delta_{col}"] = (
                pd.to_numeric(candidates[col], errors="coerce")
                - pd.to_numeric(sel.get(col), errors="coerce")
            )

        candidates["max_abs_raw_feature_delta"] = candidates.apply(raw_feature_max_abs_delta, axis=1)
        candidates["profile_exact_match"] = candidates["max_abs_raw_feature_delta"] <= cfg.exact_match_tolerance
        candidates["eligible_nonexact"] = (
            (candidates["matching_distance"] > cfg.min_matching_distance)
            & (~candidates["profile_exact_match"])
        )

        candidates = candidates[candidates["eligible_nonexact"]].copy()
        candidates = candidates.sort_values(["matching_distance", "path_id"]).reset_index(drop=True)
        candidates["eligible_decoy_rank"] = np.arange(1, len(candidates) + 1)

        candidates["selected_route_class"] = cls
        candidates["selected_path_id"] = sel["path_id"]
        candidates["selected_path_family"] = sel.get("path_family", "")
        candidates["selected_is_branch_away"] = int(sel.get("is_branch_away", 0))
        candidates["selected_is_representative"] = int(sel.get("is_representative", 0))

        for col in MATCH_FEATURES:
            candidates[f"selected_{col}"] = sel.get(col, np.nan)
            candidates[f"decoy_{col}"] = candidates[col]

        rows.append(candidates)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out


def select_rank_pairs(
    ranked: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    rank: int,
    allow_decoy_reuse: bool,
) -> pd.DataFrame:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    selected = selected.sort_values(["route_class", "path_id"]).reset_index(drop=True)

    used_decoys: set[str] = set()
    rows = []

    for _, sel in selected.iterrows():
        cls = str(sel["route_class"])
        sid = str(sel["path_id"])
        sub = ranked[
            (ranked["selected_route_class"] == cls)
            & (ranked["selected_path_id"].astype(str) == sid)
        ].copy()

        if not allow_decoy_reuse:
            sub = sub[~sub["path_id"].astype(str).isin(used_decoys)].copy()
            sub = sub.sort_values(["matching_distance", "path_id"]).reset_index(drop=True)
            sub["reuse_adjusted_rank"] = np.arange(1, len(sub) + 1)
            candidate = sub[sub["reuse_adjusted_rank"] == rank].copy()
            rank_col = "reuse_adjusted_rank"
        else:
            candidate = sub[sub["eligible_decoy_rank"] == rank].copy()
            rank_col = "eligible_decoy_rank"

        if len(candidate) == 0:
            rows.append(
                {
                    "route_class": cls,
                    "selected_path_id": sid,
                    "decoy_path_id": "",
                    "match_status": "insufficient_rank_candidate",
                    "requested_decoy_rank": rank,
                    "matching_distance": np.inf,
                    "profile_exact_match": False,
                    "max_abs_raw_feature_delta": np.inf,
                }
            )
            continue

        best = candidate.iloc[0]
        did = str(best["path_id"])
        used_decoys.add(did)

        row = {
            "route_class": cls,
            "selected_path_id": sid,
            "decoy_path_id": did,
            "match_status": "matched_rank_nonexact",
            "requested_decoy_rank": rank,
            "selected_rank": int(best[rank_col]),
            "eligible_decoy_rank": int(best["eligible_decoy_rank"]),
            "matching_distance": float(best["matching_distance"]),
            "profile_exact_match": bool(best["profile_exact_match"]),
            "max_abs_raw_feature_delta": float(best["max_abs_raw_feature_delta"]),
            "selected_path_family": sel.get("path_family", ""),
            "decoy_path_family": best.get("path_family", ""),
            "selected_is_branch_away": int(sel.get("is_branch_away", 0)),
            "decoy_is_branch_away": int(best.get("is_branch_away", 0)),
            "selected_is_representative": int(sel.get("is_representative", 0)),
            "decoy_is_representative": int(best.get("is_representative", 0)),
        }

        for col in MATCH_FEATURES:
            row[f"selected_{col}"] = sel.get(col, np.nan)
            row[f"decoy_{col}"] = best.get(col, np.nan)
            row[f"delta_{col}"] = best.get(f"delta_{col}", np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def select_random_pairs(
    ranked: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    rng: np.random.Generator,
    allow_decoy_reuse: bool,
) -> pd.DataFrame:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    selected = selected.sort_values(["route_class", "path_id"]).reset_index(drop=True)

    used_decoys: set[str] = set()
    rows = []

    for _, sel in selected.iterrows():
        cls = str(sel["route_class"])
        sid = str(sel["path_id"])
        sub = ranked[
            (ranked["selected_route_class"] == cls)
            & (ranked["selected_path_id"].astype(str) == sid)
        ].copy()

        if not allow_decoy_reuse:
            sub = sub[~sub["path_id"].astype(str).isin(used_decoys)].copy()

        if len(sub) == 0:
            rows.append(
                {
                    "route_class": cls,
                    "selected_path_id": sid,
                    "decoy_path_id": "",
                    "match_status": "no_random_candidate",
                    "matching_distance": np.inf,
                    "profile_exact_match": False,
                    "max_abs_raw_feature_delta": np.inf,
                }
            )
            continue

        idx = int(rng.integers(0, len(sub)))
        best = sub.iloc[idx]
        did = str(best["path_id"])
        used_decoys.add(did)

        row = {
            "route_class": cls,
            "selected_path_id": sid,
            "decoy_path_id": did,
            "match_status": "matched_random_nonexact",
            "eligible_decoy_rank": int(best["eligible_decoy_rank"]),
            "matching_distance": float(best["matching_distance"]),
            "profile_exact_match": bool(best["profile_exact_match"]),
            "max_abs_raw_feature_delta": float(best["max_abs_raw_feature_delta"]),
            "selected_path_family": sel.get("path_family", ""),
            "decoy_path_family": best.get("path_family", ""),
            "selected_is_branch_away": int(sel.get("is_branch_away", 0)),
            "decoy_is_branch_away": int(best.get("is_branch_away", 0)),
            "selected_is_representative": int(sel.get("is_representative", 0)),
            "decoy_is_representative": int(best.get("is_representative", 0)),
        }

        for col in MATCH_FEATURES:
            row[f"selected_{col}"] = sel.get(col, np.nan)
            row[f"decoy_{col}"] = best.get(col, np.nan)
            row[f"delta_{col}"] = best.get(f"delta_{col}", np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def selected_path_class_map(profile: pd.DataFrame) -> dict[str, str]:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    return {str(row["path_id"]): str(row["route_class"]) for _, row in selected.iterrows()}


def replacement_path_class_map(profile: pd.DataFrame, pairs: pd.DataFrame, replace_class: str) -> dict[str, str]:
    base = selected_path_class_map(profile)

    selected_ids = set(profile.loc[profile["route_class"] == replace_class, "path_id"].astype(str))
    for pid in selected_ids:
        base.pop(pid, None)

    matched = pairs[
        (pairs["route_class"] == replace_class)
        & (pairs["match_status"].astype(str).str.startswith("matched"))
    ].copy()

    for _, row in matched.iterrows():
        base[str(row["decoy_path_id"])] = replace_class

    return base


def build_transition_steps_for_paths(routes: pd.DataFrame, path_class_map: dict[str, str], cfg: Config) -> pd.DataFrame:
    use_ids = set(path_class_map.keys())
    work = routes[routes["path_id"].astype(str).isin(use_ids)].copy()
    work["analysis_route_class"] = work["path_id"].astype(str).map(path_class_map)
    work = work[work["analysis_route_class"].isin(CLASS_ORDER)].copy()

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
                    "route_class": a["analysis_route_class"],
                    "source_route_class": a.get("route_class", ""),
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


def build_transition_distribution(
    transition_steps: pd.DataFrame,
    run_id: str,
    replacement_route_class: str = "",
    run_type: str = "",
    rank: int | None = None,
    iteration: int | None = None,
) -> pd.DataFrame:
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
        counts["run_type"] = run_type
        counts["replacement_route_class"] = replacement_route_class
        counts["rank"] = rank if rank is not None else np.nan
        counts["iteration"] = iteration if iteration is not None else np.nan
        rows.append(counts)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "run_id",
            "run_type",
            "rank",
            "iteration",
            "replacement_route_class",
            "route_class",
            "state_from",
            "state_to",
            "transition",
            "n_transitions",
            "n_paths",
            "transition_share",
        ]
    ]


def build_transition_signature(
    transition_steps: pd.DataFrame,
    run_id: str,
    replacement_route_class: str = "",
    run_type: str = "",
    rank: int | None = None,
    iteration: int | None = None,
) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        sub = transition_steps[transition_steps["route_class"] == cls].copy()
        n_transitions = int(len(sub))
        n_paths = int(sub["path_id"].nunique()) if len(sub) else 0

        row = {
            "run_id": run_id,
            "run_type": run_type,
            "rank": rank if rank is not None else np.nan,
            "iteration": iteration if iteration is not None else np.nan,
            "replacement_route_class": replacement_route_class,
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
                    row[f"top_transition_{i+1}_share"] = float(top.iloc[i]["n"] / n_transitions)
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
    return {str(row["transition"]): float(row["transition_share"]) for _, row in sub.iterrows()}


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


def build_drift_rows(
    baseline_sig: pd.DataFrame,
    baseline_dist: pd.DataFrame,
    replacement_sig: pd.DataFrame,
    replacement_dist: pd.DataFrame,
    *,
    replacement_route_class: str,
    run_id: str,
    run_type: str,
    rank: int | None = None,
    iteration: int | None = None,
) -> pd.DataFrame:
    metric_cols = [f"{label}_share" for label in TARGET_TRANSITIONS]
    rows = []

    for cls in CLASS_ORDER:
        base_hit = baseline_sig[baseline_sig["route_class"] == cls]
        repl_hit = replacement_sig[
            (replacement_sig["run_id"] == run_id)
            & (replacement_sig["route_class"] == cls)
        ]

        if len(base_hit) == 0 or len(repl_hit) == 0:
            continue

        base_row = base_hit.iloc[0].to_dict()
        repl_row = repl_hit.iloc[0].to_dict()

        row = {
            "run_id": run_id,
            "run_type": run_type,
            "rank": rank if rank is not None else np.nan,
            "iteration": iteration if iteration is not None else np.nan,
            "replacement_route_class": replacement_route_class,
            "evaluated_route_class": cls,
            "is_replaced_class": int(cls == replacement_route_class),
            "baseline_n_paths": int(base_row["n_paths"]),
            "replacement_n_paths": int(repl_row["n_paths"]),
            "baseline_n_transitions": int(base_row["n_transitions"]),
            "replacement_n_transitions": int(repl_row["n_transitions"]),
        }

        abs_diffs = []
        for col in metric_cols:
            before = float(base_row.get(col, 0.0))
            after = float(repl_row.get(col, 0.0))
            diff = after - before
            row[f"{col}_baseline"] = before
            row[f"{col}_replacement"] = after
            row[f"{col}_diff"] = diff
            row[f"{col}_abs_diff"] = abs(diff)
            abs_diffs.append(abs(diff))

        row["max_abs_target_share_drift"] = max(abs_diffs) if abs_diffs else 0.0
        row["mean_abs_target_share_drift"] = float(np.mean(abs_diffs)) if abs_diffs else 0.0

        p = distribution_map(baseline_dist, cls)
        q = distribution_map(replacement_dist[replacement_dist["run_id"] == run_id], cls)

        base_top3 = top_transitions(baseline_dist, cls, k=3)
        repl_top3 = top_transitions(replacement_dist[replacement_dist["run_id"] == run_id], cls, k=3)

        base_top1 = base_top3[0] if base_top3 else ""
        repl_top1 = repl_top3[0] if repl_top3 else ""

        row["transition_tv_distance"] = total_variation_distance(p, q)
        row["top1_transition_baseline"] = base_top1
        row["top1_transition_replacement"] = repl_top1
        row["top1_survived"] = int(base_top1 == repl_top1)
        row["top1_changed"] = int(base_top1 != repl_top1)
        row["top3_transition_baseline"] = " | ".join(base_top3)
        row["top3_transition_replacement"] = " | ".join(repl_top3)
        row["top3_overlap"] = top_k_overlap(base_top3, repl_top3, k=3)
        row["top3_overlap_share"] = row["top3_overlap"] / 3.0 if base_top3 or repl_top3 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def baseline_outputs(routes: pd.DataFrame, profile: pd.DataFrame, cfg: Config):
    baseline_map = selected_path_class_map(profile)
    baseline_steps = build_transition_steps_for_paths(routes, baseline_map, cfg)
    baseline_sig = build_transition_signature(baseline_steps, run_id="baseline", run_type="baseline")
    baseline_dist = build_transition_distribution(baseline_steps, run_id="baseline", run_type="baseline")
    return baseline_sig, baseline_dist


def run_replacement_for_pairs(
    routes: pd.DataFrame,
    profile: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: Config,
    *,
    replacement_route_class: str,
    run_id: str,
    run_type: str,
    rank: int | None = None,
    iteration: int | None = None,
):
    repl_map = replacement_path_class_map(profile, pairs, replacement_route_class)
    steps = build_transition_steps_for_paths(routes, repl_map, cfg)
    sig = build_transition_signature(
        steps,
        run_id=run_id,
        replacement_route_class=replacement_route_class,
        run_type=run_type,
        rank=rank,
        iteration=iteration,
    )
    dist = build_transition_distribution(
        steps,
        run_id=run_id,
        replacement_route_class=replacement_route_class,
        run_type=run_type,
        rank=rank,
        iteration=iteration,
    )
    return sig, dist


def run_rank_controls(
    routes: pd.DataFrame,
    profile: pd.DataFrame,
    ranked: pd.DataFrame,
    baseline_sig: pd.DataFrame,
    baseline_dist: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_values = parse_rank_list(cfg.rank_list)
    run_rows = []
    drift_rows = []

    for rank in rank_values:
        pairs = select_rank_pairs(
            ranked,
            profile,
            rank=rank,
            allow_decoy_reuse=cfg.allow_decoy_reuse,
        )

        for cls in CLASS_ORDER:
            run_id = f"rank_{rank}_replace_{cls}"
            sig, dist = run_replacement_for_pairs(
                routes,
                profile,
                pairs,
                cfg,
                replacement_route_class=cls,
                run_id=run_id,
                run_type="rank",
                rank=rank,
            )

            matched = pairs[
                (pairs["route_class"] == cls)
                & (pairs["match_status"] == "matched_rank_nonexact")
            ]
            run_rows.append(
                {
                    "run_id": run_id,
                    "run_type": "rank",
                    "rank": rank,
                    "iteration": np.nan,
                    "replacement_route_class": cls,
                    "matched_pairs": int(len(matched)),
                    "unmatched_pairs": int(8 - len(matched)),
                    "mean_matching_distance": float(matched["matching_distance"].mean()) if len(matched) else np.nan,
                    "max_matching_distance": float(matched["matching_distance"].max()) if len(matched) else np.nan,
                    "mean_max_abs_raw_feature_delta": float(matched["max_abs_raw_feature_delta"].mean()) if len(matched) else np.nan,
                }
            )

            drift = build_drift_rows(
                baseline_sig,
                baseline_dist,
                sig,
                dist,
                replacement_route_class=cls,
                run_id=run_id,
                run_type="rank",
                rank=rank,
            )
            drift_rows.append(drift)

    run_df = pd.DataFrame(run_rows)
    drift_df = pd.concat(drift_rows, ignore_index=True) if drift_rows else pd.DataFrame()
    return run_df, drift_df


def run_random_controls(
    routes: pd.DataFrame,
    profile: pd.DataFrame,
    ranked: pd.DataFrame,
    baseline_sig: pd.DataFrame,
    baseline_dist: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_seed)

    run_rows = []
    drift_rows = []

    for iteration in range(int(cfg.n_iter)):
        pairs = select_random_pairs(
            ranked,
            profile,
            rng=rng,
            allow_decoy_reuse=cfg.allow_decoy_reuse,
        )

        for cls in CLASS_ORDER:
            run_id = f"random_{iteration:04d}_replace_{cls}"
            sig, dist = run_replacement_for_pairs(
                routes,
                profile,
                pairs,
                cfg,
                replacement_route_class=cls,
                run_id=run_id,
                run_type="random",
                iteration=iteration,
            )

            matched = pairs[
                (pairs["route_class"] == cls)
                & (pairs["match_status"] == "matched_random_nonexact")
            ]

            run_rows.append(
                {
                    "run_id": run_id,
                    "run_type": "random",
                    "rank": np.nan,
                    "iteration": iteration,
                    "replacement_route_class": cls,
                    "matched_pairs": int(len(matched)),
                    "unmatched_pairs": int(8 - len(matched)),
                    "mean_matching_distance": float(matched["matching_distance"].mean()) if len(matched) else np.nan,
                    "max_matching_distance": float(matched["matching_distance"].max()) if len(matched) else np.nan,
                    "mean_max_abs_raw_feature_delta": float(matched["max_abs_raw_feature_delta"].mean()) if len(matched) else np.nan,
                }
            )

            drift = build_drift_rows(
                baseline_sig,
                baseline_dist,
                sig,
                dist,
                replacement_route_class=cls,
                run_id=run_id,
                run_type="random",
                iteration=iteration,
            )
            drift_rows.append(drift)

    run_df = pd.DataFrame(run_rows)
    drift_df = pd.concat(drift_rows, ignore_index=True) if drift_rows else pd.DataFrame()
    return run_df, drift_df


def summarize_ensemble(rank_drift: pd.DataFrame, random_drift: pd.DataFrame) -> pd.DataFrame:
    drift = pd.concat([rank_drift, random_drift], ignore_index=True)
    if len(drift) == 0:
        return pd.DataFrame()

    sub = drift[drift["is_replaced_class"] == 1].copy()
    rows = []

    group_cols = ["run_type", "replacement_route_class"]
    if "rank" in sub.columns:
        rank_sub = sub[sub["run_type"] == "rank"].copy()
        for (run_type, cls, rank), grp in rank_sub.groupby(["run_type", "replacement_route_class", "rank"], dropna=False):
            rows.append(summarize_group(grp, run_type=run_type, cls=cls, rank=rank))

    rand_sub = sub[sub["run_type"] == "random"].copy()
    for (run_type, cls), grp in rand_sub.groupby(group_cols, dropna=False):
        rows.append(summarize_group(grp, run_type=run_type, cls=cls, rank=np.nan))

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["run_type", "replacement_route_class", "rank"]).reset_index(drop=True)
    return out


def summarize_group(grp: pd.DataFrame, *, run_type: str, cls: str, rank) -> dict:
    return {
        "run_type": run_type,
        "replacement_route_class": cls,
        "rank": rank,
        "n_runs": int(len(grp)),
        "top1_survival_rate": float(grp["top1_survived"].mean()) if len(grp) else np.nan,
        "mean_top3_overlap_share": float(grp["top3_overlap_share"].mean()) if len(grp) else np.nan,
        "median_top3_overlap_share": float(grp["top3_overlap_share"].median()) if len(grp) else np.nan,
        "mean_transition_tv_distance": float(grp["transition_tv_distance"].mean()) if len(grp) else np.nan,
        "median_transition_tv_distance": float(grp["transition_tv_distance"].median()) if len(grp) else np.nan,
        "p90_transition_tv_distance": float(grp["transition_tv_distance"].quantile(0.90)) if len(grp) else np.nan,
        "max_transition_tv_distance": float(grp["transition_tv_distance"].max()) if len(grp) else np.nan,
        "mean_max_abs_target_share_drift": float(grp["max_abs_target_share_drift"].mean()) if len(grp) else np.nan,
        "p90_max_abs_target_share_drift": float(grp["max_abs_target_share_drift"].quantile(0.90)) if len(grp) else np.nan,
        "max_abs_target_share_drift": float(grp["max_abs_target_share_drift"].max()) if len(grp) else np.nan,
    }


def summarize_selected(profile: pd.DataFrame) -> pd.DataFrame:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    rows = []

    for cls in CLASS_ORDER:
        sub = selected[selected["route_class"] == cls]
        rows.append(
            {
                "route_class": cls,
                "n_paths": int(len(sub)),
                "total_steps": int(sub["n_steps"].sum()) if len(sub) else 0,
                "mean_steps_per_path": safe_mean(sub["n_steps"]) if len(sub) else float("nan"),
                "mean_distance_to_seam": safe_mean(sub["mean_distance_to_seam"]) if len(sub) else float("nan"),
                "mean_lazarus_score": safe_mean(sub["mean_lazarus_score"]) if len(sub) else float("nan"),
                "mean_response_strength": safe_mean(sub["mean_response_strength"]) if len(sub) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def build_report(
    cfg: Config,
    profile: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    ranked: pd.DataFrame,
    baseline_sig: pd.DataFrame,
    rank_runs: pd.DataFrame,
    rank_drift: pd.DataFrame,
    random_runs: pd.DataFrame,
    random_drift: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
) -> str:
    selected_summary = summarize_selected(profile)
    rank_values = parse_rank_list(cfg.rank_list)

    lines = [
        "# OBS-060 — Non-exact decoy ensemble route-origin controls",
        "",
        f"Corpus label: `{cfg.corpus_label or 'unspecified'}`",
        "",
        "## Purpose",
        "",
        "OBS-060 tests whether OBS-059 nearest non-exact decoy survival generalizes across rank-k and random non-exact decoy replacements.",
        "",
        "OBS-059 established nearest non-exact decoy dominant-signature survival. OBS-060 asks whether that result depends on the single nearest eligible non-exact decoy.",
        "",
        "## Ensemble policy",
        "",
        "Matching uses the same standardized path-profile features as OBS-059:",
        "",
    ]

    for feature in MATCH_FEATURES:
        lines.append(f"- `{feature}`")

    lines.extend(
        [
            "",
            "Eligible candidates must be non-exact under the configured tolerances.",
            "",
            f"`min_matching_distance`: `{cfg.min_matching_distance}`",
            f"`exact_match_tolerance`: `{cfg.exact_match_tolerance}`",
            f"`rank_list`: `{','.join(str(x) for x in rank_values)}`",
            f"`n_iter`: `{cfg.n_iter}`",
            f"`random_seed`: `{cfg.random_seed}`",
            f"`allow_decoy_reuse`: `{cfg.allow_decoy_reuse}`",
            f"`branch_decoy_same_family`: `{cfg.branch_decoy_same_family}`",
            "",
            "Rank controls use the k-th eligible non-exact decoy.",
            "",
            "Random controls sample eligible non-exact decoys without reuse within each replacement run unless reuse is explicitly enabled.",
            "",
            "## Selected origin substrate",
            "",
        ]
    )

    for _, row in selected_summary.iterrows():
        lines.extend(
            [
                f"### {row['route_class']}",
                "",
                f"- n_paths: {int(row['n_paths'])}",
                f"- total_steps: {int(row['total_steps'])}",
                f"- mean_steps_per_path: {float(row['mean_steps_per_path']):.4f}",
                f"- mean_distance_to_seam: {float(row['mean_distance_to_seam']):.6f}",
                f"- mean_lazarus_score: {float(row['mean_lazarus_score']):.6f}",
                f"- mean_response_strength: {float(row['mean_response_strength']):.6f}",
                "",
            ]
        )

    lines.extend(["## Candidate-pool sizes", ""])
    for cls in CLASS_ORDER:
        n_pool = int((candidate_pool["target_route_class"] == cls).sum()) if len(candidate_pool) else 0
        n_ranked = int((ranked["selected_route_class"] == cls).sum()) if len(ranked) else 0
        lines.append(f"- {cls}: pool={n_pool}, eligible_nonexact_ranked_candidates={n_ranked}")
    lines.append("")

    lines.extend(["## Baseline top transitions", ""])
    for _, row in baseline_sig.iterrows():
        lines.extend(
            [
                f"### {row['route_class']}",
                "",
                f"- n_paths: {int(row['n_paths'])}",
                f"- n_transitions: {int(row['n_transitions'])}",
                f"- top_transition_1: {row['top_transition_1']}",
                f"- top_transition_1_share: {float(row['top_transition_1_share']):.6f}",
                "",
            ]
        )

    if len(ensemble_summary):
        lines.extend(["## Ensemble summary", ""])
        for _, row in ensemble_summary.iterrows():
            run_type = row["run_type"]
            cls = row["replacement_route_class"]
            rank = row["rank"]

            title = f"{run_type} — replace {cls}"
            if run_type == "rank" and pd.notna(rank):
                title = f"rank {int(rank)} — replace {cls}"

            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"- n_runs: {int(row['n_runs'])}",
                    f"- top1_survival_rate: {float(row['top1_survival_rate']):.6f}",
                    f"- mean_top3_overlap_share: {float(row['mean_top3_overlap_share']):.6f}",
                    f"- median_transition_tv_distance: {float(row['median_transition_tv_distance']):.6f}",
                    f"- p90_transition_tv_distance: {float(row['p90_transition_tv_distance']):.6f}",
                    f"- max_transition_tv_distance: {float(row['max_transition_tv_distance']):.6f}",
                    f"- mean_max_abs_target_share_drift: {float(row['mean_max_abs_target_share_drift']):.6f}",
                    f"- p90_max_abs_target_share_drift: {float(row['p90_max_abs_target_share_drift']):.6f}",
                    "",
                ]
            )

    rank_unmatched = int(rank_runs["unmatched_pairs"].sum()) if len(rank_runs) and "unmatched_pairs" in rank_runs else 0
    random_unmatched = int(random_runs["unmatched_pairs"].sum()) if len(random_runs) and "unmatched_pairs" in random_runs else 0

    lines.extend(
        [
            "## Match completeness",
            "",
            f"- rank_unmatched_pairs_total: {rank_unmatched}",
            f"- random_unmatched_pairs_total: {random_unmatched}",
            "",
            "## Interpretation guardrail",
            "",
            "This study does not validate route classes globally.",
            "",
            "It tests whether OBS-059 nearest non-exact decoy survival generalizes across deterministic rank-k and random non-exact decoy ensembles.",
            "",
            "If top-1 survival remains high across ensembles, the route-origin dominant signature is not dependent on a single nearest non-exact substitute.",
            "",
            "If survival drops under rank-k or random controls, OBS-059 remains valid but should be scoped to nearest-neighbor non-exact replacement.",
            "",
            "This is still an origin-substrate control. It does not establish downstream motif, generator, proto-groupoid, gateway, or canonical-family survival.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-exact decoy ensemble route-origin controls.")
    parser.add_argument("--scene-routes-csv", default=Config.scene_routes_csv)
    parser.add_argument("--seam-nodes-csv", default=Config.seam_nodes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--corpus-label", default=Config.corpus_label)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--post-exit-threshold", type=float, default=Config.post_exit_threshold)
    parser.add_argument("--allow-decoy-reuse", action="store_true")
    parser.add_argument("--branch-decoy-same-family", action="store_true")
    parser.add_argument("--min-matching-distance", type=float, default=Config.min_matching_distance)
    parser.add_argument("--exact-match-tolerance", type=float, default=Config.exact_match_tolerance)
    parser.add_argument("--rank-list", default=Config.rank_list)
    parser.add_argument("--n-iter", type=int, default=Config.n_iter)
    parser.add_argument("--random-seed", type=int, default=Config.random_seed)
    parser.add_argument("--max-candidates-per-selected", type=int, default=Config.max_candidates_per_selected)
    args = parser.parse_args()

    cfg = Config(
        scene_routes_csv=args.scene_routes_csv,
        seam_nodes_csv=args.seam_nodes_csv,
        outdir=args.outdir,
        corpus_label=args.corpus_label,
        seam_threshold=args.seam_threshold,
        post_exit_threshold=args.post_exit_threshold,
        allow_decoy_reuse=args.allow_decoy_reuse,
        branch_decoy_same_family=args.branch_decoy_same_family,
        min_matching_distance=args.min_matching_distance,
        exact_match_tolerance=args.exact_match_tolerance,
        rank_list=args.rank_list,
        n_iter=max(int(args.n_iter), 0),
        random_seed=int(args.random_seed),
        max_candidates_per_selected=max(int(args.max_candidates_per_selected), 25),
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

    profile = build_path_profile(routes, cfg)
    selected_profile = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    candidate_pool = build_decoy_candidate_pool(profile, cfg)
    ranked = build_ranked_nonexact_candidates(profile, candidate_pool, cfg)

    baseline_sig, baseline_dist = baseline_outputs(routes, profile, cfg)

    rank_runs, rank_drift = run_rank_controls(
        routes,
        profile,
        ranked,
        baseline_sig,
        baseline_dist,
        cfg,
    )

    random_runs, random_drift = run_random_controls(
        routes,
        profile,
        ranked,
        baseline_sig,
        baseline_dist,
        cfg,
    )

    ensemble_summary = summarize_ensemble(rank_drift, random_drift)

    all_profile_csv = outdir / "obs060_all_path_profile.csv"
    selected_profile_csv = outdir / "obs060_selected_path_profile.csv"
    candidate_pool_csv = outdir / "obs060_decoy_candidate_pool.csv"
    ranked_candidates_csv = outdir / "obs060_ranked_nonexact_decoy_candidates.csv"
    baseline_sig_csv = outdir / "obs060_baseline_transition_signature.csv"
    baseline_dist_csv = outdir / "obs060_baseline_transition_distribution.csv"
    rank_runs_csv = outdir / "obs060_rank_replacement_runs.csv"
    rank_drift_csv = outdir / "obs060_rank_replacement_drift.csv"
    random_runs_csv = outdir / "obs060_random_replacement_runs.csv"
    random_drift_csv = outdir / "obs060_random_replacement_drift.csv"
    ensemble_summary_csv = outdir / "obs060_ensemble_summary.csv"
    report_md = outdir / "obs060_nonexact_decoy_ensemble_controls_report.md"

    profile.to_csv(all_profile_csv, index=False)
    selected_profile.to_csv(selected_profile_csv, index=False)
    candidate_pool.to_csv(candidate_pool_csv, index=False)

    ranked_to_write = ranked.copy()
    if len(ranked_to_write) and cfg.max_candidates_per_selected > 0:
        ranked_to_write = (
            ranked_to_write.sort_values(["selected_route_class", "selected_path_id", "eligible_decoy_rank"])
            .groupby(["selected_route_class", "selected_path_id"], as_index=False, group_keys=False)
            .head(cfg.max_candidates_per_selected)
        )
    ranked_to_write.to_csv(ranked_candidates_csv, index=False)

    baseline_sig.to_csv(baseline_sig_csv, index=False)
    baseline_dist.to_csv(baseline_dist_csv, index=False)
    rank_runs.to_csv(rank_runs_csv, index=False)
    rank_drift.to_csv(rank_drift_csv, index=False)
    random_runs.to_csv(random_runs_csv, index=False)
    random_drift.to_csv(random_drift_csv, index=False)
    ensemble_summary.to_csv(ensemble_summary_csv, index=False)

    report_md.write_text(
        build_report(
            cfg,
            profile,
            candidate_pool,
            ranked,
            baseline_sig,
            rank_runs,
            rank_drift,
            random_runs,
            random_drift,
            ensemble_summary,
        ),
        encoding="utf-8",
    )

    print(all_profile_csv)
    print(selected_profile_csv)
    print(candidate_pool_csv)
    print(ranked_candidates_csv)
    print(baseline_sig_csv)
    print(baseline_dist_csv)
    print(rank_runs_csv)
    print(rank_drift_csv)
    print(random_runs_csv)
    print(random_drift_csv)
    print(ensemble_summary_csv)
    print(report_md)


if __name__ == "__main__":
    main()
