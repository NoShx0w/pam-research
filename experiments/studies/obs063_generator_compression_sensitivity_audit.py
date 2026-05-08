#!/usr/bin/env python3
"""
OBS-063 — Generator compression sensitivity audit.

Purpose
-------
Diagnose why motif-class signatures can survive route-origin decoy replacement
while completed-generator and composition signatures degrade.

OBS-062 showed a layered result:

- motif classes are highly robust across C and Cp
- completed-generator identities are class/corpus/band sensitive
- generator compositions are often the most fragile layer

OBS-063 audits the compression path:

    motif_class
      -> reduced motif word
      -> generator
      -> generator_resolved
      -> generator_completed
      -> composition

The study is diagnostic, not a rule-tuning step.

Key diagnostics
---------------
1. Reduced-word survival.
2. Generator-stage survival:
   - generator
   - generator_resolved
   - generator_completed
3. Generator top-1/top-2 margin and near-tie sensitivity.
4. Hamming-distance flip cliffs:
   - low Hamming distance + generator flip = compression cliff
   - high Hamming distance + same motif class = coarse motif taxonomy
5. Algebraic anchors:
   - generator-pair compositions that survive better than their marginal
     generator components.

Inputs
------
outputs/obs022_scene_bundle/scene_routes.csv
outputs/obs028c_canonical_seam_bundle/seam_nodes.csv

Outputs
-------
outputs/obs063_generator_compression_sensitivity_audit/
  obs063_all_path_profile.csv
  obs063_selected_path_profile.csv
  obs063_decoy_candidate_pool.csv
  obs063_ranked_nonexact_decoy_candidates.csv
  obs063_replacement_runs.csv
  obs063_word_signature_baseline.csv
  obs063_word_signature_replacement.csv
  obs063_generator_stage_signature_baseline.csv
  obs063_generator_stage_signature_replacement.csv
  obs063_composition_signature_baseline.csv
  obs063_composition_signature_replacement.csv
  obs063_stage_survival_summary.csv
  obs063_generator_margin_summary.csv
  obs063_motif_to_generator_entropy_baseline.csv
  obs063_motif_to_generator_entropy_replacement.csv
  obs063_top_word_generator_deltas.csv
  obs063_hamming_flip_summary.csv
  obs063_algebraic_anchor_candidates.csv
  obs063_generator_compression_sensitivity_audit_report.md
"""

from __future__ import annotations

import argparse
import math
import re
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

MATCH_FEATURES = [
    "n_steps",
    "mean_distance_to_seam",
    "max_distance_to_seam",
    "mean_lazarus_score",
    "mean_response_strength",
]

GENERATOR_STAGE_COLS = [
    "generator",
    "generator_resolved",
    "generator_completed",
]

GENERATOR_ORDER = [
    "g_rel_release",
    "g_aniso_release",
    "g_flank_shuttle",
    "g_low_flank_shuttle",
    "g_low_residency",
    "g_off_persist",
    "g_post_persist",
    "g_reentry",
    "g_core_behavior",
    "g_off_to_post",
    "g_flank_to_off",
    "g_low_to_off",
    "g_off_to_edge",
    "g_edge_to_post",
    "g_post_to_off",
    "g_other",
]


@dataclass(frozen=True)
class Config:
    scene_routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    seam_nodes_csv: str = "outputs/obs028c_canonical_seam_bundle/seam_nodes.csv"
    outdir: str = "outputs/obs063_generator_compression_sensitivity_audit"
    corpus_label: str = ""
    seam_threshold: float = 0.15
    post_exit_threshold: float = 0.50
    allow_decoy_reuse: bool = False
    branch_decoy_same_family: bool = False
    min_matching_distance: float = 1e-9
    exact_match_tolerance: float = 1e-12
    rank_bands: str = "1-10,51-250,all"
    n_iter: int = 250
    random_seed: int = 42
    max_candidates_per_selected: int = 1000
    top_k_delta_terms: int = 12
    anchor_pair_top_k: int = 5
    strong_anchor_pair_survival: float = 0.80
    strong_anchor_pair_drift: float = 0.10
    strong_anchor_component_rank_survival: float = 0.50
    weak_anchor_pair_survival: float = 0.60
    weak_anchor_component_rank_survival: float = 0.75
    low_margin_threshold: float = 0.05


def read_csv_numeric(path: str | Path, text_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    text = set(text_cols)
    for col in df.columns:
        if col not in text:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_rank_bands(raw: str) -> list[dict[str, object]]:
    bands: list[dict[str, object]] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue

        if token.lower() in {"all", "all_nonexact"}:
            bands.append({"band": "all_nonexact", "rank_min": 1, "rank_max": None})
            continue

        m = re.fullmatch(r"(\d+)-(\d+)", token)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2))
            if lo <= 0 or hi < lo:
                raise ValueError(f"Invalid rank band: {token}")
            bands.append({"band": f"rank_{lo}_{hi}", "rank_min": lo, "rank_max": hi})
            continue

        m = re.fullmatch(r"(\d+)\+", token)
        if m:
            lo = int(m.group(1))
            if lo <= 0:
                raise ValueError(f"Invalid rank band: {token}")
            bands.append({"band": f"rank_{lo}_plus", "rank_min": lo, "rank_max": None})
            continue

        raise ValueError(f"Could not parse rank band token: {token}")

    if not bands:
        bands.append({"band": "rank_1_10", "rank_min": 1, "rank_max": 10})

    return bands


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
    seam_use = seam_use.rename(
        columns={c: f"{c}_bundle" for c in seam_use.columns if c != "node_id"}
    )

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


def reduce_state(s: str) -> str:
    mapping = {
        "relational_flank": "R",
        "anisotropy_flank": "A",
        "seam_resident_low": "L",
        "off_seam": "O",
        "post_exit": "P",
        "shared_core": "C",
        "mixed_seam": "L",
    }
    return mapping.get(str(s), "L")


def classify_motif(a: str, b: str, c: str) -> str:
    if a == "relational_flank" and b in {"off_seam", "post_exit"} and c == "post_exit":
        return "relational_release_motif"
    if a == "anisotropy_flank" and b in {"off_seam", "post_exit"} and c == "post_exit":
        return "anisotropy_release_motif"
    if a == "shared_core" and b in {"off_seam", "post_exit", "seam_resident_low"} and c in {"off_seam", "post_exit"}:
        return "core_release_motif"

    if a == "shared_core" and b == "shared_core" and c == "shared_core":
        return "core_retention_motif"
    if a == "seam_resident_low" and b == "seam_resident_low" and c == "seam_resident_low":
        return "low_residency_motif"

    if {a, b, c}.issubset({"relational_flank", "anisotropy_flank"}):
        return "flank_shuttle_motif"

    if a == "off_seam" and b == "off_seam" and c == "off_seam":
        return "off_seam_persistence_motif"
    if a == "post_exit" and b == "post_exit" and c == "post_exit":
        return "post_exit_persistence_motif"

    if a == "off_seam" and b in {"relational_flank", "anisotropy_flank", "shared_core"}:
        return "reentry_motif"
    if a == "post_exit" and b in {"relational_flank", "anisotropy_flank", "shared_core"}:
        return "reentry_motif"

    if a == "shared_core" and b == "seam_resident_low":
        return "core_to_low_motif"
    if a == "seam_resident_low" and c in {"relational_flank", "anisotropy_flank"}:
        return "low_to_flank_motif"

    return "other_motif"


def assign_generator(motif_class: str, a: str, b: str, c: str) -> str:
    ra, rb, rc = reduce_state(a), reduce_state(b), reduce_state(c)

    if motif_class == "relational_release_motif":
        return "g_rel_release"
    if motif_class == "anisotropy_release_motif":
        return "g_aniso_release"
    if motif_class == "flank_shuttle_motif":
        return "g_flank_shuttle"
    if motif_class == "low_residency_motif":
        return "g_low_residency"
    if motif_class == "off_seam_persistence_motif":
        return "g_off_persist"
    if motif_class == "post_exit_persistence_motif":
        return "g_post_persist"
    if motif_class == "reentry_motif":
        return "g_reentry"
    if motif_class in {"core_retention_motif", "core_release_motif", "core_to_low_motif"}:
        return "g_core_behavior"

    if ra == "R" and rc == "P":
        return "g_rel_release"
    if ra == "A" and rc == "P":
        return "g_aniso_release"
    if {ra, rb, rc}.issubset({"R", "A"}):
        return "g_flank_shuttle"
    if ra == rb == rc == "L":
        return "g_low_residency"
    if ra == rb == rc == "O":
        return "g_off_persist"
    if ra == rb == rc == "P":
        return "g_post_persist"
    if ra in {"O", "P"} and rb in {"R", "A", "L", "C"}:
        return "g_reentry"
    if "C" in {ra, rb, rc}:
        return "g_core_behavior"

    return "g_other"


def resolve_other(word: str, current: str) -> str:
    if current != "g_other":
        return current

    if word == "O~O~P":
        return "g_off_to_post"
    if word in {"R~A~O", "A~R~O", "A~A~O", "R~R~O"}:
        return "g_flank_to_off"
    if word in {"L~L~O", "L~O~O"}:
        return "g_low_to_off"
    if word in {"O~R~R", "O~A~A", "O~R~A", "O~A~R"}:
        return "g_off_to_edge"
    if word in {"P~O~O", "P~O~P"}:
        return "g_post_to_off"

    return "g_other"


def finalize_generator(word: str, current: str) -> str:
    if current != "g_other":
        return current

    if word in {"L~L~R", "L~R~L", "R~L~L", "L~L~A", "L~A~L", "A~L~L"}:
        return "g_low_flank_shuttle"

    if word in {"O~O~R", "O~O~A", "O~O~L"}:
        return "g_off_to_edge"

    if word in {"O~P~P", "L~L~P", "L~P~P", "R~P~P", "A~P~P"}:
        return "g_edge_to_post"

    return "g_other"


def word_parts(word: str) -> list[str]:
    return str(word).split("~")


def word_hamming_distance(a: str, b: str) -> int:
    aa = word_parts(a)
    bb = word_parts(b)
    n = max(len(aa), len(bb))
    dist = 0
    for i in range(n):
        av = aa[i] if i < len(aa) else ""
        bv = bb[i] if i < len(bb) else ""
        dist += int(av != bv)
    return dist


def word_changed_positions(a: str, b: str) -> str:
    aa = word_parts(a)
    bb = word_parts(b)
    labels = ["a", "b", "c"]
    n = max(len(aa), len(bb))
    out = []
    for i in range(n):
        av = aa[i] if i < len(aa) else ""
        bv = bb[i] if i < len(bb) else ""
        if av != bv:
            out.append(labels[i] if i < len(labels) else str(i))
    return ",".join(out)


def shannon_entropy_from_counts(counts: pd.Series) -> float:
    x = pd.to_numeric(counts, errors="coerce").fillna(0).to_numpy(dtype=float)
    total = x.sum()
    if total <= 0:
        return 0.0
    p = x[x > 0] / total
    return float(-(p * np.log2(p)).sum())


def build_path_profile(routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    corpus = cfg.corpus_label or "unspecified"

    for path_id, grp in routes.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy()

        path_family = str(first_or_nan(grp["path_family"])) if "path_family" in grp.columns else ""
        route_class = str(first_or_nan(grp["route_class"])) if "route_class" in grp.columns else "unknown"

        is_branch_away = int(
            pd.to_numeric(grp.get("is_branch_away", 0), errors="coerce").fillna(0).max()
        )
        is_representative = int(
            pd.to_numeric(grp.get("is_representative", 0), errors="coerce").fillna(0).max()
        )

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
                    profile.loc[profile["route_class"] == "branch_exit", "path_family"]
                    .astype(str)
                    .unique()
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

    return pd.concat(rows, ignore_index=True)


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


def select_band_pairs(
    ranked: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    band_name: str,
    rank_min: int,
    rank_max: int | None,
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
            & (ranked["eligible_decoy_rank"] >= rank_min)
        ].copy()

        if rank_max is not None:
            sub = sub[sub["eligible_decoy_rank"] <= rank_max].copy()

        if not allow_decoy_reuse:
            sub = sub[~sub["path_id"].astype(str).isin(used_decoys)].copy()

        if len(sub) == 0:
            rows.append(
                {
                    "band": band_name,
                    "route_class": cls,
                    "selected_path_id": sid,
                    "decoy_path_id": "",
                    "match_status": "no_band_candidate",
                    "rank_min": rank_min,
                    "rank_max": rank_max if rank_max is not None else np.nan,
                    "eligible_decoy_rank": np.nan,
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
            "band": band_name,
            "route_class": cls,
            "selected_path_id": sid,
            "decoy_path_id": did,
            "match_status": "matched_band_nonexact",
            "rank_min": rank_min,
            "rank_max": rank_max if rank_max is not None else np.nan,
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
                    "state_from": a["state_type"],
                    "state_to": b["state_type"],
                }
            )

    return pd.DataFrame(rows)


def build_motifs(transition_steps: pd.DataFrame, run_id: str, band: str, iteration: int, replacement_route_class: str) -> pd.DataFrame:
    rows = []

    if len(transition_steps) == 0:
        return pd.DataFrame()

    for path_id, grp in transition_steps.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            if a["state_to"] != b["state_from"]:
                continue

            state_a = str(a["state_from"])
            state_b = str(a["state_to"])
            state_c = str(b["state_to"])
            motif_class = classify_motif(state_a, state_b, state_c)

            rows.append(
                {
                    "run_id": run_id,
                    "band": band,
                    "iteration": iteration,
                    "replacement_route_class": replacement_route_class,
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "step": a["step"],
                    "motif": f"{state_a} -> {state_b} -> {state_c}",
                    "motif_class": motif_class,
                    "state_a": state_a,
                    "state_b": state_b,
                    "state_c": state_c,
                    "state_a_red": reduce_state(state_a),
                    "state_b_red": reduce_state(state_b),
                    "state_c_red": reduce_state(state_c),
                }
            )

    return pd.DataFrame(rows)


def build_generator_assignments(motifs: pd.DataFrame) -> pd.DataFrame:
    if len(motifs) == 0:
        return pd.DataFrame()

    out = motifs.copy()
    out["generator_word"] = out["state_a_red"] + "~" + out["state_b_red"] + "~" + out["state_c_red"]
    out["generator"] = [
        assign_generator(mc, a, b, c)
        for mc, a, b, c in zip(out["motif_class"], out["state_a"], out["state_b"], out["state_c"])
    ]
    out["generator_resolved"] = [
        resolve_other(word, gen)
        for word, gen in zip(out["generator_word"], out["generator"])
    ]
    out["generator_completed"] = [
        finalize_generator(word, gen)
        for word, gen in zip(out["generator_word"], out["generator_resolved"])
    ]
    return out


def build_generator_compositions(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if len(assignments) == 0:
        return pd.DataFrame()

    for (run_id, path_id), grp in assignments.groupby(["run_id", "path_id"], sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            rows.append(
                {
                    "run_id": a["run_id"],
                    "band": a["band"],
                    "iteration": a["iteration"],
                    "replacement_route_class": a["replacement_route_class"],
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "generator_1": a["generator_completed"],
                    "generator_2": b["generator_completed"],
                    "composition": f"{a['generator_completed']} ; {b['generator_completed']}",
                }
            )

    return pd.DataFrame(rows)


def distribution_from_counts(df: pd.DataFrame, group_col: str, count_name: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    rows = []
    for (run_id, band, iteration, replacement_route_class, route_class), grp in df.groupby(
        ["run_id", "band", "iteration", "replacement_route_class", "route_class"],
        dropna=False,
        sort=False,
    ):
        counts = (
            grp.groupby(group_col, as_index=False)
            .agg(count=(group_col, "size"), n_paths=("path_id", "nunique"))
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        total = int(counts["count"].sum())

        for _, row in counts.iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "band": band,
                    "iteration": iteration,
                    "replacement_route_class": replacement_route_class,
                    "route_class": route_class,
                    group_col: row[group_col],
                    count_name: int(row["count"]),
                    "n_paths": int(row["n_paths"]),
                    "share": float(row["count"] / total) if total else 0.0,
                    "rank": int(len(rows) + 1),
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out["rank"] = (
            out.groupby(["run_id", "route_class", "replacement_route_class", "band", "iteration"])["share"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    return out


def word_signature(assignments: pd.DataFrame) -> pd.DataFrame:
    return distribution_from_counts(assignments, "generator_word", "n_words")


def generator_stage_signature(assignments: pd.DataFrame) -> pd.DataFrame:
    if len(assignments) == 0:
        return pd.DataFrame()

    frames = []
    for stage in GENERATOR_STAGE_COLS:
        sig = distribution_from_counts(assignments, stage, "n_generators")
        if len(sig):
            sig = sig.rename(columns={stage: "generator_label"})
            sig["generator_stage"] = stage
            frames.append(sig)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def composition_signature(compositions: pd.DataFrame) -> pd.DataFrame:
    return distribution_from_counts(compositions, "composition", "n_compositions")


def dist_map(sig: pd.DataFrame, run_id: str, route_class: str, value_col: str, extra_filter: dict | None = None) -> dict[str, float]:
    sub = sig[(sig["run_id"] == run_id) & (sig["route_class"] == route_class)]
    if extra_filter:
        for col, val in extra_filter.items():
            sub = sub[sub[col] == val]
    return {str(row[value_col]): float(row["share"]) for _, row in sub.iterrows()}


def top_values(sig: pd.DataFrame, run_id: str, route_class: str, value_col: str, k: int = 3, extra_filter: dict | None = None) -> list[str]:
    sub = sig[(sig["run_id"] == run_id) & (sig["route_class"] == route_class)]
    if extra_filter:
        for col, val in extra_filter.items():
            sub = sub[sub[col] == val]
    sub = sub.sort_values("share", ascending=False).head(k)
    return [str(x) for x in sub[value_col].tolist()]


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def top_overlap(a: list[str], b: list[str], k: int = 3) -> int:
    return len(set(a[:k]) & set(b[:k]))


def n_items(sig: pd.DataFrame, run_id: str, route_class: str, count_col: str, extra_filter: dict | None = None) -> int:
    sub = sig[(sig["run_id"] == run_id) & (sig["route_class"] == route_class)]
    if extra_filter:
        for col, val in extra_filter.items():
            sub = sub[sub[col] == val]
    if len(sub) == 0:
        return 0
    return int(sub[count_col].sum())


def top_margin(sig: pd.DataFrame, run_id: str, route_class: str, value_col: str, extra_filter: dict | None = None) -> dict[str, object]:
    sub = sig[(sig["run_id"] == run_id) & (sig["route_class"] == route_class)]
    if extra_filter:
        for col, val in extra_filter.items():
            sub = sub[sub[col] == val]

    sub = sub.sort_values("share", ascending=False).reset_index(drop=True)

    if len(sub) == 0:
        return {
            "top1_label": "",
            "top1_share": 0.0,
            "top2_label": "",
            "top2_share": 0.0,
            "top1_minus_top2_margin": 0.0,
        }

    top1_label = str(sub.iloc[0][value_col])
    top1_share = float(sub.iloc[0]["share"])
    if len(sub) > 1:
        top2_label = str(sub.iloc[1][value_col])
        top2_share = float(sub.iloc[1]["share"])
    else:
        top2_label = ""
        top2_share = 0.0

    return {
        "top1_label": top1_label,
        "top1_share": top1_share,
        "top2_label": top2_label,
        "top2_share": top2_share,
        "top1_minus_top2_margin": top1_share - top2_share,
    }


def build_stage_drift(
    baseline: pd.DataFrame,
    replacement: pd.DataFrame,
    *,
    run_id: str,
    band: str,
    iteration: int,
    replacement_route_class: str,
    layer: str,
    value_col: str,
    count_col: str,
    extra_filter: dict | None = None,
) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        p = dist_map(baseline, "baseline", cls, value_col, extra_filter=extra_filter)
        q = dist_map(replacement, run_id, cls, value_col, extra_filter=extra_filter)

        base_top3 = top_values(baseline, "baseline", cls, value_col, k=3, extra_filter=extra_filter)
        repl_top3 = top_values(replacement, run_id, cls, value_col, k=3, extra_filter=extra_filter)

        base_top1 = base_top3[0] if base_top3 else ""
        repl_top1 = repl_top3[0] if repl_top3 else ""

        margin = top_margin(baseline, "baseline", cls, value_col, extra_filter=extra_filter)

        rows.append(
            {
                "run_id": run_id,
                "band": band,
                "iteration": iteration,
                "replacement_route_class": replacement_route_class,
                "evaluated_route_class": cls,
                "is_replaced_class": int(cls == replacement_route_class),
                "layer": layer,
                "baseline_n_items": n_items(baseline, "baseline", cls, count_col, extra_filter=extra_filter),
                "replacement_n_items": n_items(replacement, run_id, cls, count_col, extra_filter=extra_filter),
                "distribution_tv_distance": total_variation(p, q),
                "top1_baseline": base_top1,
                "top1_replacement": repl_top1,
                "top1_survived": int(base_top1 == repl_top1),
                "top1_changed": int(base_top1 != repl_top1),
                "top3_baseline": " | ".join(base_top3),
                "top3_replacement": " | ".join(repl_top3),
                "top3_overlap": top_overlap(base_top3, repl_top3, k=3),
                "top3_overlap_share": top_overlap(base_top3, repl_top3, k=3) / 3.0 if (base_top3 or repl_top3) else 0.0,
                "baseline_top1_share": margin["top1_share"],
                "baseline_top2_share": margin["top2_share"],
                "baseline_top1_minus_top2_margin": margin["top1_minus_top2_margin"],
                "baseline_top1_label": margin["top1_label"],
                "baseline_top2_label": margin["top2_label"],
            }
        )

    return pd.DataFrame(rows)


def build_pipeline_outputs(
    routes: pd.DataFrame,
    path_class_map: dict[str, str],
    cfg: Config,
    *,
    run_id: str,
    band: str,
    iteration: int,
    replacement_route_class: str,
):
    steps = build_transition_steps_for_paths(routes, path_class_map, cfg)
    motifs = build_motifs(steps, run_id, band, iteration, replacement_route_class)
    assignments = build_generator_assignments(motifs)
    compositions = build_generator_compositions(assignments)

    word_sig = word_signature(assignments)
    gen_stage_sig = generator_stage_signature(assignments)
    comp_sig = composition_signature(compositions)

    return assignments, compositions, word_sig, gen_stage_sig, comp_sig


def run_controls(
    routes: pd.DataFrame,
    profile: pd.DataFrame,
    ranked: pd.DataFrame,
    baseline_word: pd.DataFrame,
    baseline_gen_stage: pd.DataFrame,
    baseline_comp: pd.DataFrame,
    cfg: Config,
):
    rng = np.random.default_rng(cfg.random_seed)
    bands = parse_rank_bands(cfg.rank_bands)

    run_rows = []
    replacement_assignments = []
    replacement_compositions = []
    word_sigs = []
    gen_stage_sigs = []
    comp_sigs = []
    drift_rows = []

    for band_cfg in bands:
        band_name = str(band_cfg["band"])
        rank_min = int(band_cfg["rank_min"])
        rank_max = band_cfg["rank_max"]
        rank_max_int = int(rank_max) if rank_max is not None else None

        for iteration in range(int(cfg.n_iter)):
            pairs = select_band_pairs(
                ranked,
                profile,
                band_name=band_name,
                rank_min=rank_min,
                rank_max=rank_max_int,
                rng=rng,
                allow_decoy_reuse=cfg.allow_decoy_reuse,
            )

            for cls in CLASS_ORDER:
                run_id = f"{band_name}_iter_{iteration:04d}_replace_{cls}"
                repl_map = replacement_path_class_map(profile, pairs, cls)

                assignments, compositions, word_sig, gen_stage_sig, comp_sig = build_pipeline_outputs(
                    routes,
                    repl_map,
                    cfg,
                    run_id=run_id,
                    band=band_name,
                    iteration=iteration,
                    replacement_route_class=cls,
                )

                replacement_assignments.append(assignments)
                replacement_compositions.append(compositions)
                word_sigs.append(word_sig)
                gen_stage_sigs.append(gen_stage_sig)
                comp_sigs.append(comp_sig)

                matched = pairs[
                    (pairs["route_class"] == cls)
                    & (pairs["match_status"] == "matched_band_nonexact")
                ]

                run_rows.append(
                    {
                        "run_id": run_id,
                        "band": band_name,
                        "rank_min": rank_min,
                        "rank_max": rank_max_int if rank_max_int is not None else np.nan,
                        "iteration": iteration,
                        "replacement_route_class": cls,
                        "matched_pairs": int(len(matched)),
                        "unmatched_pairs": int(8 - len(matched)),
                        "mean_decoy_rank": float(matched["eligible_decoy_rank"].mean()) if len(matched) else np.nan,
                        "min_decoy_rank": float(matched["eligible_decoy_rank"].min()) if len(matched) else np.nan,
                        "max_decoy_rank": float(matched["eligible_decoy_rank"].max()) if len(matched) else np.nan,
                        "mean_matching_distance": float(matched["matching_distance"].mean()) if len(matched) else np.nan,
                        "max_matching_distance": float(matched["matching_distance"].max()) if len(matched) else np.nan,
                        "mean_max_abs_raw_feature_delta": float(matched["max_abs_raw_feature_delta"].mean()) if len(matched) else np.nan,
                    }
                )

                drift_rows.append(
                    build_stage_drift(
                        baseline_word,
                        word_sig,
                        run_id=run_id,
                        band=band_name,
                        iteration=iteration,
                        replacement_route_class=cls,
                        layer="generator_word",
                        value_col="generator_word",
                        count_col="n_words",
                    )
                )

                for stage in GENERATOR_STAGE_COLS:
                    drift_rows.append(
                        build_stage_drift(
                            baseline_gen_stage,
                            gen_stage_sig,
                            run_id=run_id,
                            band=band_name,
                            iteration=iteration,
                            replacement_route_class=cls,
                            layer=stage,
                            value_col="generator_label",
                            count_col="n_generators",
                            extra_filter={"generator_stage": stage},
                        )
                    )

                drift_rows.append(
                    build_stage_drift(
                        baseline_comp,
                        comp_sig,
                        run_id=run_id,
                        band=band_name,
                        iteration=iteration,
                        replacement_route_class=cls,
                        layer="composition",
                        value_col="composition",
                        count_col="n_compositions",
                    )
                )

    runs = pd.DataFrame(run_rows)
    repl_assign = pd.concat(replacement_assignments, ignore_index=True) if replacement_assignments else pd.DataFrame()
    repl_comp = pd.concat(replacement_compositions, ignore_index=True) if replacement_compositions else pd.DataFrame()
    word_repl = pd.concat(word_sigs, ignore_index=True) if word_sigs else pd.DataFrame()
    gen_stage_repl = pd.concat(gen_stage_sigs, ignore_index=True) if gen_stage_sigs else pd.DataFrame()
    comp_repl = pd.concat(comp_sigs, ignore_index=True) if comp_sigs else pd.DataFrame()
    drift = pd.concat(drift_rows, ignore_index=True) if drift_rows else pd.DataFrame()

    return runs, repl_assign, repl_comp, word_repl, gen_stage_repl, comp_repl, drift


def summarize_stage_survival(runs: pd.DataFrame, drift: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if len(drift) == 0:
        return pd.DataFrame()

    replaced = drift[drift["is_replaced_class"] == 1].copy()
    rows = []

    for (band, cls, layer), grp in replaced.groupby(["band", "replacement_route_class", "layer"], dropna=False):
        run_grp = runs[
            (runs["band"] == band)
            & (runs["replacement_route_class"] == cls)
        ].copy()

        top1_survival = float(grp["top1_survived"].mean()) if len(grp) else np.nan
        mean_margin = float(grp["baseline_top1_minus_top2_margin"].mean()) if len(grp) else np.nan

        rows.append(
            {
                "band": band,
                "replacement_route_class": cls,
                "layer": layer,
                "n_runs": int(len(grp)),
                "unmatched_runs": int((run_grp["unmatched_pairs"] > 0).sum()) if len(run_grp) else 0,
                "mean_unmatched_pairs": float(run_grp["unmatched_pairs"].mean()) if len(run_grp) else np.nan,
                "mean_decoy_rank": float(run_grp["mean_decoy_rank"].mean()) if len(run_grp) else np.nan,
                "mean_matching_distance": float(run_grp["mean_matching_distance"].mean()) if len(run_grp) else np.nan,
                "top1_survival_rate": top1_survival,
                "top1_failure_rate": 1.0 - top1_survival if np.isfinite(top1_survival) else np.nan,
                "mean_top3_overlap_share": float(grp["top3_overlap_share"].mean()) if len(grp) else np.nan,
                "median_distribution_tv_distance": float(grp["distribution_tv_distance"].median()) if len(grp) else np.nan,
                "p90_distribution_tv_distance": float(grp["distribution_tv_distance"].quantile(0.90)) if len(grp) else np.nan,
                "max_distribution_tv_distance": float(grp["distribution_tv_distance"].max()) if len(grp) else np.nan,
                "mean_baseline_n_items": float(grp["baseline_n_items"].mean()) if len(grp) else np.nan,
                "mean_replacement_n_items": float(grp["replacement_n_items"].mean()) if len(grp) else np.nan,
                "baseline_top1_label": str(grp["baseline_top1_label"].iloc[0]) if len(grp) else "",
                "baseline_top2_label": str(grp["baseline_top2_label"].iloc[0]) if len(grp) else "",
                "baseline_top1_share": float(grp["baseline_top1_share"].iloc[0]) if len(grp) else np.nan,
                "baseline_top2_share": float(grp["baseline_top2_share"].iloc[0]) if len(grp) else np.nan,
                "baseline_top1_minus_top2_margin": float(grp["baseline_top1_minus_top2_margin"].iloc[0]) if len(grp) else np.nan,
                "margin_class": "low_margin" if np.isfinite(mean_margin) and mean_margin <= cfg.low_margin_threshold else "high_margin",
                "failure_class": (
                    "soft_fail_low_margin"
                    if np.isfinite(mean_margin) and mean_margin <= cfg.low_margin_threshold and np.isfinite(top1_survival) and top1_survival < 1.0
                    else "hard_fail_high_margin"
                    if np.isfinite(mean_margin) and mean_margin > cfg.low_margin_threshold and np.isfinite(top1_survival) and top1_survival < 1.0
                    else "survived"
                ),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        band_order = {str(b["band"]): i for i, b in enumerate(parse_rank_bands(cfg.rank_bands))}
        class_order = {c: i for i, c in enumerate(CLASS_ORDER)}
        layer_order = {
            "generator_word": 0,
            "generator": 1,
            "generator_resolved": 2,
            "generator_completed": 3,
            "composition": 4,
        }

        out["band_order"] = out["band"].map(lambda x: band_order.get(str(x), 999))
        out["class_order"] = out["replacement_route_class"].map(lambda x: class_order.get(str(x), 999))
        out["layer_order"] = out["layer"].map(lambda x: layer_order.get(str(x), 999))
        out = (
            out.sort_values(["band_order", "class_order", "layer_order"])
            .drop(columns=["band_order", "class_order", "layer_order"])
            .reset_index(drop=True)
        )

    return out


def build_generator_margin_summary(baseline_gen_stage: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for cls in CLASS_ORDER:
        for stage in GENERATOR_STAGE_COLS:
            margin = top_margin(
                baseline_gen_stage,
                "baseline",
                cls,
                "generator_label",
                extra_filter={"generator_stage": stage},
            )
            rows.append(
                {
                    "route_class": cls,
                    "generator_stage": stage,
                    **margin,
                    "margin_class": "low_margin"
                    if float(margin["top1_minus_top2_margin"]) <= cfg.low_margin_threshold
                    else "high_margin",
                }
            )
    return pd.DataFrame(rows)


def motif_to_generator_entropy(assignments: pd.DataFrame, mode: str) -> pd.DataFrame:
    if len(assignments) == 0:
        return pd.DataFrame()

    rows = []
    group_cols = ["run_id", "band", "iteration", "replacement_route_class", "route_class", "motif_class"]

    for keys, grp in assignments.groupby(group_cols, dropna=False, sort=False):
        key_dict = dict(zip(group_cols, keys))
        counts = (
            grp.groupby("generator_completed", as_index=False)
            .agg(n=("path_id", "size"))
            .sort_values("n", ascending=False)
            .reset_index(drop=True)
        )
        total = int(counts["n"].sum())
        top = counts.iloc[0] if len(counts) else None

        rows.append(
            {
                **key_dict,
                "mode": mode,
                "n_motifs": total,
                "n_generator_completed": int(counts["generator_completed"].nunique()) if len(counts) else 0,
                "top_generator_completed": str(top["generator_completed"]) if top is not None else "",
                "top_generator_count": int(top["n"]) if top is not None else 0,
                "top_generator_share_within_motif_class": float(top["n"] / total) if top is not None and total else 0.0,
                "generator_entropy_within_motif_class": shannon_entropy_from_counts(counts["n"]) if len(counts) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def signature_table_for_assignments(assignments: pd.DataFrame, value_col: str, count_col: str) -> pd.DataFrame:
    return distribution_from_counts(assignments, value_col, count_col)


def build_top_word_generator_deltas(
    baseline_assign: pd.DataFrame,
    replacement_assign: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if len(baseline_assign) == 0 or len(replacement_assign) == 0:
        return pd.DataFrame()

    baseline = (
        baseline_assign.groupby(["route_class", "motif_class", "generator_word", "generator", "generator_resolved", "generator_completed"], as_index=False)
        .agg(n_baseline=("path_id", "size"))
    )
    baseline_totals = baseline.groupby("route_class")["n_baseline"].transform("sum")
    baseline["share_baseline"] = baseline["n_baseline"] / baseline_totals.clip(lower=1)

    repl = (
        replacement_assign.groupby(["band", "replacement_route_class", "route_class", "motif_class", "generator_word", "generator", "generator_resolved", "generator_completed"], as_index=False)
        .agg(n_replacement_mean=("path_id", "size"))
    )

    # Average replacement shares over runs by first computing run-level shares.
    run_counts = (
        replacement_assign.groupby(
            [
                "run_id",
                "band",
                "replacement_route_class",
                "route_class",
                "motif_class",
                "generator_word",
                "generator",
                "generator_resolved",
                "generator_completed",
            ],
            as_index=False,
        )
        .agg(n=("path_id", "size"))
    )
    run_totals = run_counts.groupby(["run_id", "route_class"])["n"].transform("sum")
    run_counts["share"] = run_counts["n"] / run_totals.clip(lower=1)

    repl_avg = (
        run_counts.groupby(
            [
                "band",
                "replacement_route_class",
                "route_class",
                "motif_class",
                "generator_word",
                "generator",
                "generator_resolved",
                "generator_completed",
            ],
            as_index=False,
        )
        .agg(
            n_replacement_mean=("n", "mean"),
            share_replacement_mean=("share", "mean"),
            share_replacement_p90=("share", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.90))),
        )
    )

    rows = []

    for (band, repl_cls, route_cls), sub in repl_avg.groupby(["band", "replacement_route_class", "route_class"], dropna=False):
        base = baseline[baseline["route_class"] == route_cls].copy()

        # Pair each baseline word with the replacement word in the same motif class
        # that has the largest absolute share change signal.
        for _, b in base.iterrows():
            candidates = sub[sub["motif_class"] == b["motif_class"]].copy()
            if len(candidates) == 0:
                rows.append(
                    {
                        "band": band,
                        "replacement_route_class": repl_cls,
                        "evaluated_route_class": route_cls,
                        "baseline_motif_class": b["motif_class"],
                        "replacement_motif_class": "",
                        "baseline_word": b["generator_word"],
                        "replacement_word": "",
                        "word_hamming_distance": np.nan,
                        "changed_positions": "",
                        "baseline_generator": b["generator"],
                        "replacement_generator": "",
                        "baseline_generator_resolved": b["generator_resolved"],
                        "replacement_generator_resolved": "",
                        "baseline_generator_completed": b["generator_completed"],
                        "replacement_generator_completed": "",
                        "same_motif_class": False,
                        "same_base_generator": False,
                        "same_resolved_generator": False,
                        "same_completed_generator": False,
                        "share_baseline": float(b["share_baseline"]),
                        "share_replacement_mean": 0.0,
                        "delta_share": -float(b["share_baseline"]),
                        "abs_delta_share": abs(float(b["share_baseline"])),
                        "diagnostic_class": "baseline_word_disappeared",
                    }
                )
                continue

            candidates = candidates.copy()
            candidates["delta_vs_baseline"] = candidates["share_replacement_mean"] - float(b["share_baseline"])
            candidates["abs_delta_vs_baseline"] = candidates["delta_vs_baseline"].abs()
            best = candidates.sort_values("abs_delta_vs_baseline", ascending=False).iloc[0]

            ham = word_hamming_distance(str(b["generator_word"]), str(best["generator_word"]))
            same_completed = str(b["generator_completed"]) == str(best["generator_completed"])
            same_motif = str(b["motif_class"]) == str(best["motif_class"])

            if ham <= 1 and not same_completed:
                diagnostic = "compression_cliff_low_hamming_flip"
            elif ham >= 2 and same_motif:
                diagnostic = "coarse_motif_high_hamming_same_class"
            elif ham <= 1 and same_completed:
                diagnostic = "robust_symbolic_basin"
            else:
                diagnostic = "mixed_shift"

            rows.append(
                {
                    "band": band,
                    "replacement_route_class": repl_cls,
                    "evaluated_route_class": route_cls,
                    "baseline_motif_class": b["motif_class"],
                    "replacement_motif_class": best["motif_class"],
                    "baseline_word": b["generator_word"],
                    "replacement_word": best["generator_word"],
                    "word_hamming_distance": ham,
                    "changed_positions": word_changed_positions(str(b["generator_word"]), str(best["generator_word"])),
                    "baseline_generator": b["generator"],
                    "replacement_generator": best["generator"],
                    "baseline_generator_resolved": b["generator_resolved"],
                    "replacement_generator_resolved": best["generator_resolved"],
                    "baseline_generator_completed": b["generator_completed"],
                    "replacement_generator_completed": best["generator_completed"],
                    "same_motif_class": same_motif,
                    "same_base_generator": str(b["generator"]) == str(best["generator"]),
                    "same_resolved_generator": str(b["generator_resolved"]) == str(best["generator_resolved"]),
                    "same_completed_generator": same_completed,
                    "share_baseline": float(b["share_baseline"]),
                    "share_replacement_mean": float(best["share_replacement_mean"]),
                    "delta_share": float(best["share_replacement_mean"] - float(b["share_baseline"])),
                    "abs_delta_share": float(abs(best["share_replacement_mean"] - float(b["share_baseline"]))),
                    "diagnostic_class": diagnostic,
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out["rank_abs_delta_within_context"] = (
            out.groupby(["band", "replacement_route_class", "evaluated_route_class"])["abs_delta_share"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        out = out[out["rank_abs_delta_within_context"] <= cfg.top_k_delta_terms].copy()
        out = out.sort_values(
            ["band", "replacement_route_class", "evaluated_route_class", "rank_abs_delta_within_context"]
        ).reset_index(drop=True)

    return out


def summarize_hamming_flips(top_deltas: pd.DataFrame) -> pd.DataFrame:
    if len(top_deltas) == 0:
        return pd.DataFrame()

    rows = []
    for keys, grp in top_deltas.groupby(["band", "replacement_route_class", "evaluated_route_class", "diagnostic_class"], dropna=False):
        band, repl_cls, eval_cls, diagnostic = keys
        rows.append(
            {
                "band": band,
                "replacement_route_class": repl_cls,
                "evaluated_route_class": eval_cls,
                "diagnostic_class": diagnostic,
                "n_terms": int(len(grp)),
                "mean_abs_delta_share": float(grp["abs_delta_share"].mean()),
                "max_abs_delta_share": float(grp["abs_delta_share"].max()),
                "mean_word_hamming_distance": float(pd.to_numeric(grp["word_hamming_distance"], errors="coerce").mean()),
                "share_same_completed_generator": float(pd.to_numeric(grp["same_completed_generator"], errors="coerce").mean()),
                "share_same_base_generator": float(pd.to_numeric(grp["same_base_generator"], errors="coerce").mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["band", "replacement_route_class", "evaluated_route_class", "diagnostic_class"]
    ).reset_index(drop=True)


def generator_marginal_share(
    sig: pd.DataFrame,
    run_id: str,
    route_class: str,
    generator_label: str,
) -> float:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["generator_stage"] == "generator_completed")
        & (sig["generator_label"] == generator_label)
    ]
    if len(sub) == 0:
        return 0.0
    return float(sub["share"].sum())


def generator_marginal_rank(
    sig: pd.DataFrame,
    run_id: str,
    route_class: str,
    generator_label: str,
) -> int:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["generator_stage"] == "generator_completed")
    ].sort_values("share", ascending=False).reset_index(drop=True)

    hits = sub.index[sub["generator_label"].astype(str) == str(generator_label)].tolist()
    if not hits:
        return 9999
    return int(hits[0] + 1)


def composition_share(
    sig: pd.DataFrame,
    run_id: str,
    route_class: str,
    composition: str,
) -> float:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["composition"] == composition)
    ]
    if len(sub) == 0:
        return 0.0
    return float(sub["share"].sum())


def build_algebraic_anchor_candidates(
    baseline_gen_stage: pd.DataFrame,
    replacement_gen_stage: pd.DataFrame,
    baseline_comp: pd.DataFrame,
    replacement_comp: pd.DataFrame,
    runs: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if len(baseline_comp) == 0 or len(replacement_comp) == 0:
        return pd.DataFrame()

    rows = []
    baseline_top_pairs = (
        baseline_comp.sort_values(["route_class", "share"], ascending=[True, False])
        .groupby("route_class", as_index=False, group_keys=False)
        .head(cfg.anchor_pair_top_k)
    )

    for _, pair_row in baseline_top_pairs.iterrows():
        route_class = str(pair_row["route_class"])
        composition = str(pair_row["composition"])
        if " ; " in composition:
            g1, g2 = composition.split(" ; ", 1)
        else:
            g1, g2 = composition, ""

        baseline_pair_share = float(pair_row["share"])
        g1_base_share = generator_marginal_share(baseline_gen_stage, "baseline", route_class, g1)
        g2_base_share = generator_marginal_share(baseline_gen_stage, "baseline", route_class, g2)
        g1_base_rank = generator_marginal_rank(baseline_gen_stage, "baseline", route_class, g1)
        g2_base_rank = generator_marginal_rank(baseline_gen_stage, "baseline", route_class, g2)

        for (band, repl_cls), run_grp in runs.groupby(["band", "replacement_route_class"], dropna=False):
            run_ids = run_grp["run_id"].astype(str).tolist()

            pair_survived = []
            pair_repl_shares = []
            g1_repl_shares = []
            g2_repl_shares = []
            g1_rank_survived = []
            g2_rank_survived = []
            g1_ranks = []
            g2_ranks = []

            for run_id in run_ids:
                pshare = composition_share(replacement_comp, run_id, route_class, composition)
                pair_repl_shares.append(pshare)
                pair_survived.append(int(pshare > 0))

                g1s = generator_marginal_share(replacement_gen_stage, run_id, route_class, g1)
                g2s = generator_marginal_share(replacement_gen_stage, run_id, route_class, g2)
                g1_repl_shares.append(g1s)
                g2_repl_shares.append(g2s)

                g1r = generator_marginal_rank(replacement_gen_stage, run_id, route_class, g1)
                g2r = generator_marginal_rank(replacement_gen_stage, run_id, route_class, g2)
                g1_ranks.append(g1r)
                g2_ranks.append(g2r)

                g1_rank_survived.append(int(g1r == g1_base_rank))
                g2_rank_survived.append(int(g2r == g2_base_rank))

            pair_survival_rate = float(np.mean(pair_survived)) if pair_survived else np.nan
            mean_pair_share_replacement = float(np.mean(pair_repl_shares)) if pair_repl_shares else np.nan
            mean_abs_pair_share_drift = float(np.mean([abs(x - baseline_pair_share) for x in pair_repl_shares])) if pair_repl_shares else np.nan

            g1_mean_repl = float(np.mean(g1_repl_shares)) if g1_repl_shares else np.nan
            g2_mean_repl = float(np.mean(g2_repl_shares)) if g2_repl_shares else np.nan
            g1_drift = float(np.mean([abs(x - g1_base_share) for x in g1_repl_shares])) if g1_repl_shares else np.nan
            g2_drift = float(np.mean([abs(x - g2_base_share) for x in g2_repl_shares])) if g2_repl_shares else np.nan
            g1_rank_survival = float(np.mean(g1_rank_survived)) if g1_rank_survived else np.nan
            g2_rank_survival = float(np.mean(g2_rank_survived)) if g2_rank_survived else np.nan
            min_component_rank_survival = float(np.nanmin([g1_rank_survival, g2_rank_survival]))

            anchor_score = pair_survival_rate - min_component_rank_survival

            if (
                pair_survival_rate >= cfg.strong_anchor_pair_survival
                and mean_abs_pair_share_drift <= cfg.strong_anchor_pair_drift
                and min_component_rank_survival < cfg.strong_anchor_component_rank_survival
            ):
                anchor_class = "strong_anchor"
            elif (
                pair_survival_rate >= cfg.weak_anchor_pair_survival
                and min_component_rank_survival < cfg.weak_anchor_component_rank_survival
            ):
                anchor_class = "weak_anchor"
            else:
                anchor_class = "not_anchor"

            rows.append(
                {
                    "band": band,
                    "replacement_route_class": repl_cls,
                    "evaluated_route_class": route_class,
                    "baseline_composition": composition,
                    "generator_1": g1,
                    "generator_2": g2,
                    "baseline_pair_share": baseline_pair_share,
                    "pair_survival_rate": pair_survival_rate,
                    "mean_pair_share_replacement": mean_pair_share_replacement,
                    "mean_abs_pair_share_drift": mean_abs_pair_share_drift,
                    "generator_1_baseline_share": g1_base_share,
                    "generator_1_mean_replacement_share": g1_mean_repl,
                    "generator_1_mean_abs_share_drift": g1_drift,
                    "generator_1_baseline_rank": g1_base_rank,
                    "generator_1_mean_replacement_rank": float(np.mean(g1_ranks)) if g1_ranks else np.nan,
                    "generator_1_rank_survival_rate": g1_rank_survival,
                    "generator_2_baseline_share": g2_base_share,
                    "generator_2_mean_replacement_share": g2_mean_repl,
                    "generator_2_mean_abs_share_drift": g2_drift,
                    "generator_2_baseline_rank": g2_base_rank,
                    "generator_2_mean_replacement_rank": float(np.mean(g2_ranks)) if g2_ranks else np.nan,
                    "generator_2_rank_survival_rate": g2_rank_survival,
                    "min_component_rank_survival_rate": min_component_rank_survival,
                    "anchor_score": anchor_score,
                    "anchor_class": anchor_class,
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["anchor_class", "anchor_score", "pair_survival_rate"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return out


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
    runs: pd.DataFrame,
    stage_summary: pd.DataFrame,
    margin_summary: pd.DataFrame,
    entropy_baseline: pd.DataFrame,
    top_deltas: pd.DataFrame,
    hamming_summary: pd.DataFrame,
    anchors: pd.DataFrame,
) -> str:
    selected_summary = summarize_selected(profile)
    bands = parse_rank_bands(cfg.rank_bands)

    lines = [
        "# OBS-063 — Generator compression sensitivity audit",
        "",
        f"Corpus label: `{cfg.corpus_label or 'unspecified'}`",
        "",
        "## Purpose",
        "",
        "OBS-063 diagnoses why motif-class signatures can survive route-origin decoy replacement while completed-generator and composition signatures degrade.",
        "",
        "The study audits the compression path from motif class to reduced word, generator, resolved generator, completed generator, and composition.",
        "",
        "This is a diagnostic audit, not a rule-tuning step.",
        "",
        "## Decoy policy",
        "",
        "Matching uses standardized path-profile features:",
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
            f"`rank_bands`: `{cfg.rank_bands}`",
            f"`n_iter`: `{cfg.n_iter}`",
            f"`random_seed`: `{cfg.random_seed}`",
            f"`allow_decoy_reuse`: `{cfg.allow_decoy_reuse}`",
            f"`branch_decoy_same_family`: `{cfg.branch_decoy_same_family}`",
            f"`low_margin_threshold`: `{cfg.low_margin_threshold}`",
            "",
            "Parsed bands:",
            "",
        ]
    )

    for b in bands:
        hi = b["rank_max"] if b["rank_max"] is not None else "∞"
        lines.append(f"- `{b['band']}`: ranks {b['rank_min']}–{hi}")
    lines.append("")

    lines.extend(["## Selected origin substrate", ""])
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

    lines.extend(["## Baseline generator margins", ""])
    if len(margin_summary):
        for _, row in margin_summary.iterrows():
            lines.extend(
                [
                    f"### {row['route_class']} — {row['generator_stage']}",
                    "",
                    f"- top1_label: {row['top1_label']}",
                    f"- top1_share: {float(row['top1_share']):.6f}",
                    f"- top2_label: {row['top2_label']}",
                    f"- top2_share: {float(row['top2_share']):.6f}",
                    f"- top1_minus_top2_margin: {float(row['top1_minus_top2_margin']):.6f}",
                    f"- margin_class: {row['margin_class']}",
                    "",
                ]
            )

    lines.extend(["## Stage survival summary", ""])
    if len(stage_summary):
        view = stage_summary[
            stage_summary["layer"].isin(
                ["generator_word", "generator", "generator_resolved", "generator_completed", "composition"]
            )
        ].copy()
        for _, row in view.iterrows():
            lines.extend(
                [
                    f"### {row['band']} — replace {row['replacement_route_class']} — {row['layer']}",
                    "",
                    f"- n_runs: {int(row['n_runs'])}",
                    f"- top1_survival_rate: {float(row['top1_survival_rate']):.6f}",
                    f"- mean_top3_overlap_share: {float(row['mean_top3_overlap_share']):.6f}",
                    f"- median_distribution_tv_distance: {float(row['median_distribution_tv_distance']):.6f}",
                    f"- p90_distribution_tv_distance: {float(row['p90_distribution_tv_distance']):.6f}",
                    f"- baseline_top1_label: {row['baseline_top1_label']}",
                    f"- baseline_top2_label: {row['baseline_top2_label']}",
                    f"- baseline_top1_minus_top2_margin: {float(row['baseline_top1_minus_top2_margin']):.6f}",
                    f"- failure_class: {row['failure_class']}",
                    "",
                ]
            )

    lines.extend(["## Motif-to-generator entropy highlights", ""])
    if len(entropy_baseline):
        ent = entropy_baseline.sort_values("generator_entropy_within_motif_class", ascending=False).head(12)
        for _, row in ent.iterrows():
            lines.append(
                f"- {row['route_class']} / {row['motif_class']}: "
                f"n={int(row['n_motifs'])}, "
                f"n_generators={int(row['n_generator_completed'])}, "
                f"top={row['top_generator_completed']} "
                f"({float(row['top_generator_share_within_motif_class']):.3f}), "
                f"entropy={float(row['generator_entropy_within_motif_class']):.3f}"
            )
    lines.append("")

    lines.extend(["## Hamming flip diagnostics", ""])
    if len(hamming_summary):
        hs = hamming_summary.sort_values("max_abs_delta_share", ascending=False).head(18)
        for _, row in hs.iterrows():
            lines.append(
                f"- {row['band']} / replace {row['replacement_route_class']} / eval {row['evaluated_route_class']} / "
                f"{row['diagnostic_class']}: "
                f"n_terms={int(row['n_terms'])}, "
                f"mean_hamming={float(row['mean_word_hamming_distance']):.3f}, "
                f"max_abs_delta={float(row['max_abs_delta_share']):.3f}"
            )
    lines.append("")

    lines.extend(["## Algebraic anchor candidates", ""])
    if len(anchors):
        anchor_view = anchors[anchors["anchor_class"].isin(["strong_anchor", "weak_anchor"])].copy()
        if len(anchor_view) == 0:
            lines.append("- No strong or weak algebraic anchors detected under the configured thresholds.")
        else:
            anchor_view = anchor_view.sort_values(["anchor_class", "anchor_score"], ascending=[True, False]).head(20)
            for _, row in anchor_view.iterrows():
                lines.append(
                    f"- {row['anchor_class']}: {row['band']} / replace {row['replacement_route_class']} / eval {row['evaluated_route_class']} / "
                    f"{row['baseline_composition']} — pair_survival={float(row['pair_survival_rate']):.3f}, "
                    f"anchor_score={float(row['anchor_score']):.3f}, "
                    f"component_rank_min={float(row['min_component_rank_survival_rate']):.3f}"
                )
    lines.append("")

    unmatched_total = int(runs["unmatched_pairs"].sum()) if len(runs) and "unmatched_pairs" in runs else 0
    n_cliffs = int((top_deltas["diagnostic_class"] == "compression_cliff_low_hamming_flip").sum()) if len(top_deltas) else 0
    n_coarse = int((top_deltas["diagnostic_class"] == "coarse_motif_high_hamming_same_class").sum()) if len(top_deltas) else 0

    lines.extend(
        [
            "## Diagnostic totals",
            "",
            f"- unmatched_pairs_total: {unmatched_total}",
            f"- compression_cliff_low_hamming_flip_terms: {n_cliffs}",
            f"- coarse_motif_high_hamming_same_class_terms: {n_coarse}",
            "",
            "## Interpretation guardrail",
            "",
            "OBS-063 does not tune the OBS-030c/d/e generator rules.",
            "",
            "It treats the current generator vocabulary as a hard symbolic compression layer and audits where that layer is stable, near-tied, or cliff-sensitive.",
            "",
            "Low-margin generator flips should be read as soft failures or tie-break sensitivity.",
            "",
            "High-margin generator flips indicate stronger evidence that decoy replacement changes the downstream symbolic logic.",
            "",
            "Low-Hamming generator flips indicate compression cliffs in the reduced-word to generator map.",
            "",
            "High-Hamming shifts inside a stable motif class indicate that the motif taxonomy is coarse relative to reduced-word structure.",
            "",
            "Algebraic anchors mark generator-pair relations that survive better than their marginal generator labels.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generator compression sensitivity under decoy controls.")
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
    parser.add_argument("--rank-bands", default=Config.rank_bands)
    parser.add_argument("--n-iter", type=int, default=Config.n_iter)
    parser.add_argument("--random-seed", type=int, default=Config.random_seed)
    parser.add_argument("--max-candidates-per-selected", type=int, default=Config.max_candidates_per_selected)
    parser.add_argument("--top-k-delta-terms", type=int, default=Config.top_k_delta_terms)
    parser.add_argument("--anchor-pair-top-k", type=int, default=Config.anchor_pair_top_k)
    parser.add_argument("--low-margin-threshold", type=float, default=Config.low_margin_threshold)
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
        rank_bands=args.rank_bands,
        n_iter=max(int(args.n_iter), 0),
        random_seed=int(args.random_seed),
        max_candidates_per_selected=max(int(args.max_candidates_per_selected), 25),
        top_k_delta_terms=max(int(args.top_k_delta_terms), 1),
        anchor_pair_top_k=max(int(args.anchor_pair_top_k), 1),
        low_margin_threshold=float(args.low_margin_threshold),
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

    baseline_map = selected_path_class_map(profile)
    baseline_assign, baseline_compositions, baseline_word, baseline_gen_stage, baseline_comp = build_pipeline_outputs(
        routes,
        baseline_map,
        cfg,
        run_id="baseline",
        band="baseline",
        iteration=-1,
        replacement_route_class="baseline",
    )

    runs, repl_assign, repl_comp_rows, word_repl, gen_stage_repl, comp_repl, drift = run_controls(
        routes,
        profile,
        ranked,
        baseline_word,
        baseline_gen_stage,
        baseline_comp,
        cfg,
    )

    stage_summary = summarize_stage_survival(runs, drift, cfg)
    margin_summary = build_generator_margin_summary(baseline_gen_stage, cfg)

    entropy_baseline = motif_to_generator_entropy(baseline_assign, mode="baseline")
    entropy_replacement = motif_to_generator_entropy(repl_assign, mode="replacement")

    top_deltas = build_top_word_generator_deltas(baseline_assign, repl_assign, cfg)
    hamming_summary = summarize_hamming_flips(top_deltas)

    anchors = build_algebraic_anchor_candidates(
        baseline_gen_stage,
        gen_stage_repl,
        baseline_comp,
        comp_repl,
        runs,
        cfg,
    )

    all_profile_csv = outdir / "obs063_all_path_profile.csv"
    selected_profile_csv = outdir / "obs063_selected_path_profile.csv"
    candidate_pool_csv = outdir / "obs063_decoy_candidate_pool.csv"
    ranked_candidates_csv = outdir / "obs063_ranked_nonexact_decoy_candidates.csv"
    runs_csv = outdir / "obs063_replacement_runs.csv"
    word_baseline_csv = outdir / "obs063_word_signature_baseline.csv"
    word_repl_csv = outdir / "obs063_word_signature_replacement.csv"
    gen_stage_baseline_csv = outdir / "obs063_generator_stage_signature_baseline.csv"
    gen_stage_repl_csv = outdir / "obs063_generator_stage_signature_replacement.csv"
    comp_baseline_csv = outdir / "obs063_composition_signature_baseline.csv"
    comp_repl_csv = outdir / "obs063_composition_signature_replacement.csv"
    stage_summary_csv = outdir / "obs063_stage_survival_summary.csv"
    margin_summary_csv = outdir / "obs063_generator_margin_summary.csv"
    entropy_baseline_csv = outdir / "obs063_motif_to_generator_entropy_baseline.csv"
    entropy_repl_csv = outdir / "obs063_motif_to_generator_entropy_replacement.csv"
    top_deltas_csv = outdir / "obs063_top_word_generator_deltas.csv"
    hamming_summary_csv = outdir / "obs063_hamming_flip_summary.csv"
    anchors_csv = outdir / "obs063_algebraic_anchor_candidates.csv"
    report_md = outdir / "obs063_generator_compression_sensitivity_audit_report.md"

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

    runs.to_csv(runs_csv, index=False)
    baseline_word.to_csv(word_baseline_csv, index=False)
    word_repl.to_csv(word_repl_csv, index=False)
    baseline_gen_stage.to_csv(gen_stage_baseline_csv, index=False)
    gen_stage_repl.to_csv(gen_stage_repl_csv, index=False)
    baseline_comp.to_csv(comp_baseline_csv, index=False)
    comp_repl.to_csv(comp_repl_csv, index=False)
    stage_summary.to_csv(stage_summary_csv, index=False)
    margin_summary.to_csv(margin_summary_csv, index=False)
    entropy_baseline.to_csv(entropy_baseline_csv, index=False)
    entropy_replacement.to_csv(entropy_repl_csv, index=False)
    top_deltas.to_csv(top_deltas_csv, index=False)
    hamming_summary.to_csv(hamming_summary_csv, index=False)
    anchors.to_csv(anchors_csv, index=False)

    report_md.write_text(
        build_report(
            cfg,
            profile,
            candidate_pool,
            ranked,
            runs,
            stage_summary,
            margin_summary,
            entropy_baseline,
            top_deltas,
            hamming_summary,
            anchors,
        ),
        encoding="utf-8",
    )

    print(all_profile_csv)
    print(selected_profile_csv)
    print(candidate_pool_csv)
    print(ranked_candidates_csv)
    print(runs_csv)
    print(word_baseline_csv)
    print(word_repl_csv)
    print(gen_stage_baseline_csv)
    print(gen_stage_repl_csv)
    print(comp_baseline_csv)
    print(comp_repl_csv)
    print(stage_summary_csv)
    print(margin_summary_csv)
    print(entropy_baseline_csv)
    print(entropy_repl_csv)
    print(top_deltas_csv)
    print(hamming_summary_csv)
    print(anchors_csv)
    print(report_md)


if __name__ == "__main__":
    main()
