#!/usr/bin/env python3
"""
OBS-064 — Proto-groupoid symbolic trace cache.

Purpose
-------
Build a reusable symbolic path-trace substrate for downstream proto-groupoid
survival controls.

OBS-062 and OBS-063 repeatedly rebuilt symbolic traces inside every decoy run.
OBS-064 separates the expensive symbolic extraction step from downstream
decoy-control aggregation.

This script computes per-path symbolic traces once:

    scene_routes
      -> state sequences
      -> transition steps
      -> 3-state motifs
      -> generator assignments
      -> generator compositions
      -> proto-groupoid-ready edges / relations

It also writes baseline route-origin proto-groupoid signatures for the selected
OBS-022 / OBS-030 origin substrate.

Inputs
------
outputs/obs022_scene_bundle/scene_routes.csv
outputs/obs028c_canonical_seam_bundle/seam_nodes.csv

Outputs
-------
outputs/obs064_proto_groupoid_symbolic_trace_cache/
  obs064_all_path_profile.csv
  obs064_selected_origin_path_profile.csv
  obs064_path_state_sequences.csv
  obs064_path_transition_steps.csv
  obs064_path_motifs.csv
  obs064_path_generator_assignments.csv
  obs064_path_generator_compositions.csv
  obs064_path_proto_edges.csv
  obs064_path_proto_relations.csv
  obs064_origin_proto_edge_signature.csv
  obs064_origin_proto_relation_signature.csv
  obs064_origin_sector_signature.csv
  obs064_origin_generator_signature.csv
  obs064_origin_composition_signature.csv
  obs064_route_class_summary.csv
  obs064_proto_groupoid_symbolic_trace_cache_report.md

Notes
-----
OBS-064 is a cache/scaffold. It does not run decoy ensembles. OBS-065 should
consume these cached fragments and aggregate selected/decoy path sets.
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
    outdir: str = "outputs/obs064_proto_groupoid_symbolic_trace_cache"
    corpus_label: str = ""
    seam_threshold: float = 0.15
    post_exit_threshold: float = 0.50


def read_csv_numeric(path: str | Path, text_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    text = set(text_cols)
    for col in df.columns:
        if col not in text:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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


def sector_from_state_type(state_type: str) -> str:
    s = str(state_type)
    if s in {"relational_flank", "anisotropy_flank", "shared_core", "mixed_seam", "seam_resident_low"}:
        return "seam_sector"
    if s == "off_seam":
        return "off_seam_sector"
    if s == "post_exit":
        return "post_exit_sector"
    return "unknown_sector"


def sector_from_reduced_state(red: str) -> str:
    r = str(red)
    if r in {"R", "A", "L", "C"}:
        return "seam_sector"
    if r == "O":
        return "off_seam_sector"
    if r == "P":
        return "post_exit_sector"
    return "unknown_sector"


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


def generator_family(generator: str) -> str:
    g = str(generator)

    if g in {"g_rel_release", "g_aniso_release", "g_core_behavior"}:
        return "release_or_core"
    if g in {"g_flank_shuttle", "g_low_flank_shuttle", "g_low_residency"}:
        return "seam_internal"
    if g in {"g_reentry"}:
        return "reentry"
    if g in {"g_off_persist", "g_post_persist"}:
        return "persistence"
    if g in {"g_off_to_post", "g_flank_to_off", "g_low_to_off", "g_off_to_edge", "g_edge_to_post", "g_post_to_off"}:
        return "boundary_transfer"
    return "other"


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


def build_state_sequences(routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = routes.sort_values(["path_id", "step"]).copy().reset_index(drop=True)
    work["state_type"] = work.apply(lambda row: assign_state_type(row, cfg), axis=1)
    work["state_red"] = work["state_type"].map(reduce_state)
    work["state_sector"] = work["state_type"].map(sector_from_state_type)

    cols = [
        "path_id",
        "step",
        "node_id",
        "route_class",
        "path_family",
        "is_branch_away",
        "is_representative",
        "state_type",
        "state_red",
        "state_sector",
        "distance_to_seam",
        "lazarus_score",
        "response_strength",
        "signed_phase",
    ]
    return work[[c for c in cols if c in work.columns]].copy()


def build_transition_steps(states: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for path_id, grp in states.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            rows.append(
                {
                    "path_id": path_id,
                    "transition_index": i,
                    "step_from": a["step"],
                    "step_to": b["step"],
                    "route_class": a.get("route_class", "other"),
                    "path_family": a.get("path_family", ""),
                    "node_from": a.get("node_id", ""),
                    "node_to": b.get("node_id", ""),
                    "state_from": a["state_type"],
                    "state_to": b["state_type"],
                    "state_from_red": a["state_red"],
                    "state_to_red": b["state_red"],
                    "sector_from": a["state_sector"],
                    "sector_to": b["state_sector"],
                    "transition_word": f"{a['state_red']}~{b['state_red']}",
                    "sector_transition": f"{a['state_sector']} -> {b['state_sector']}",
                }
            )

    return pd.DataFrame(rows)


def build_motifs(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if len(transitions) == 0:
        return pd.DataFrame()

    for path_id, grp in transitions.groupby("path_id", sort=False):
        grp = grp.sort_values("transition_index").copy().reset_index(drop=True)
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

            red_a = str(a["state_from_red"])
            red_b = str(a["state_to_red"])
            red_c = str(b["state_to_red"])

            sector_a = sector_from_reduced_state(red_a)
            sector_b = sector_from_reduced_state(red_b)
            sector_c = sector_from_reduced_state(red_c)

            motif_class = classify_motif(state_a, state_b, state_c)

            rows.append(
                {
                    "path_id": path_id,
                    "motif_index": i,
                    "step_start": a["step_from"],
                    "step_mid": a["step_to"],
                    "step_end": b["step_to"],
                    "route_class": a.get("route_class", "other"),
                    "path_family": a.get("path_family", ""),
                    "state_a": state_a,
                    "state_b": state_b,
                    "state_c": state_c,
                    "state_a_red": red_a,
                    "state_b_red": red_b,
                    "state_c_red": red_c,
                    "sector_a": sector_a,
                    "sector_b": sector_b,
                    "sector_c": sector_c,
                    "motif": f"{state_a} -> {state_b} -> {state_c}",
                    "motif_word": f"{red_a}~{red_b}~{red_c}",
                    "sector_word": f"{sector_a} -> {sector_b} -> {sector_c}",
                    "motif_class": motif_class,
                }
            )

    return pd.DataFrame(rows)


def build_generator_assignments(motifs: pd.DataFrame) -> pd.DataFrame:
    if len(motifs) == 0:
        return pd.DataFrame()

    out = motifs.copy()
    out["generator_word"] = out["motif_word"]

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

    out["generator_family"] = out["generator_completed"].map(generator_family)
    out["proto_source"] = out["state_a_red"]
    out["proto_mid"] = out["state_b_red"]
    out["proto_target"] = out["state_c_red"]
    out["proto_source_sector"] = out["sector_a"]
    out["proto_mid_sector"] = out["sector_b"]
    out["proto_target_sector"] = out["sector_c"]
    out["proto_edge"] = (
        out["proto_source"].astype(str)
        + " --"
        + out["generator_completed"].astype(str)
        + "--> "
        + out["proto_target"].astype(str)
    )
    out["proto_sector_edge"] = (
        out["proto_source_sector"].astype(str)
        + " --"
        + out["generator_completed"].astype(str)
        + "--> "
        + out["proto_target_sector"].astype(str)
    )

    cols = [
        "path_id",
        "motif_index",
        "step_start",
        "step_mid",
        "step_end",
        "route_class",
        "path_family",
        "motif_class",
        "generator_word",
        "generator",
        "generator_resolved",
        "generator_completed",
        "generator_family",
        "proto_source",
        "proto_mid",
        "proto_target",
        "proto_source_sector",
        "proto_mid_sector",
        "proto_target_sector",
        "proto_edge",
        "proto_sector_edge",
        "state_a",
        "state_b",
        "state_c",
    ]

    return out[[c for c in cols if c in out.columns]].copy()


def build_generator_compositions(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if len(assignments) == 0:
        return pd.DataFrame()

    for path_id, grp in assignments.groupby("path_id", sort=False):
        grp = grp.sort_values("motif_index").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            rows.append(
                {
                    "path_id": path_id,
                    "composition_index": i,
                    "motif_index_1": a["motif_index"],
                    "motif_index_2": b["motif_index"],
                    "route_class": a.get("route_class", "other"),
                    "path_family": a.get("path_family", ""),
                    "generator_1": a["generator_completed"],
                    "generator_2": b["generator_completed"],
                    "generator_family_1": a["generator_family"],
                    "generator_family_2": b["generator_family"],
                    "composition": f"{a['generator_completed']} ; {b['generator_completed']}",
                    "composition_family": f"{a['generator_family']} ; {b['generator_family']}",
                    "proto_source_1": a["proto_source"],
                    "proto_target_1": a["proto_target"],
                    "proto_source_2": b["proto_source"],
                    "proto_target_2": b["proto_target"],
                    "proto_relation": (
                        f"{a['proto_source']} --{a['generator_completed']}--> {a['proto_target']}"
                        f" ; "
                        f"{b['proto_source']} --{b['generator_completed']}--> {b['proto_target']}"
                    ),
                    "proto_sector_relation": (
                        f"{a['proto_source_sector']} --{a['generator_completed']}--> {a['proto_target_sector']}"
                        f" ; "
                        f"{b['proto_source_sector']} --{b['generator_completed']}--> {b['proto_target_sector']}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_proto_edges(assignments: pd.DataFrame) -> pd.DataFrame:
    if len(assignments) == 0:
        return pd.DataFrame()

    cols = [
        "path_id",
        "motif_index",
        "route_class",
        "path_family",
        "motif_class",
        "generator_word",
        "generator_completed",
        "generator_family",
        "proto_source",
        "proto_mid",
        "proto_target",
        "proto_source_sector",
        "proto_mid_sector",
        "proto_target_sector",
        "proto_edge",
        "proto_sector_edge",
    ]
    return assignments[[c for c in cols if c in assignments.columns]].copy()


def build_proto_relations(compositions: pd.DataFrame) -> pd.DataFrame:
    if len(compositions) == 0:
        return pd.DataFrame()

    cols = [
        "path_id",
        "composition_index",
        "route_class",
        "path_family",
        "generator_1",
        "generator_2",
        "generator_family_1",
        "generator_family_2",
        "composition",
        "composition_family",
        "proto_relation",
        "proto_sector_relation",
        "proto_source_1",
        "proto_target_1",
        "proto_source_2",
        "proto_target_2",
    ]
    return compositions[[c for c in cols if c in compositions.columns]].copy()


def selected_origin_ids(profile: pd.DataFrame) -> set[str]:
    return set(profile.loc[profile["route_class"].isin(CLASS_ORDER), "path_id"].astype(str))


def distribution_signature(
    df: pd.DataFrame,
    value_col: str,
    count_col: str,
    selected_ids: set[str],
) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    work = df[df["path_id"].astype(str).isin(selected_ids)].copy()
    work = work[work["route_class"].isin(CLASS_ORDER)].copy()

    if len(work) == 0:
        return pd.DataFrame()

    rows = []

    for route_class, grp in work.groupby("route_class", sort=False):
        counts = (
            grp.groupby(value_col, as_index=False)
            .agg(
                count=(value_col, "size"),
                n_paths=("path_id", "nunique"),
            )
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        total = int(counts["count"].sum())

        for rank, (_, row) in enumerate(counts.iterrows(), start=1):
            rows.append(
                {
                    "route_class": route_class,
                    value_col: row[value_col],
                    count_col: int(row["count"]),
                    "n_paths": int(row["n_paths"]),
                    "share": float(row["count"] / total) if total else 0.0,
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def route_class_summary(
    profile: pd.DataFrame,
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    motifs: pd.DataFrame,
    assignments: pd.DataFrame,
    compositions: pd.DataFrame,
) -> pd.DataFrame:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    rows = []

    for cls in CLASS_ORDER:
        pids = set(selected.loc[selected["route_class"] == cls, "path_id"].astype(str))

        state_sub = states[states["path_id"].astype(str).isin(pids)]
        transition_sub = transitions[transitions["path_id"].astype(str).isin(pids)]
        motif_sub = motifs[motifs["path_id"].astype(str).isin(pids)]
        assignment_sub = assignments[assignments["path_id"].astype(str).isin(pids)]
        composition_sub = compositions[compositions["path_id"].astype(str).isin(pids)]

        profile_sub = selected[selected["route_class"] == cls]

        rows.append(
            {
                "route_class": cls,
                "n_paths": int(len(profile_sub)),
                "total_steps": int(profile_sub["n_steps"].sum()) if len(profile_sub) else 0,
                "mean_steps_per_path": safe_mean(profile_sub["n_steps"]) if len(profile_sub) else np.nan,
                "n_state_rows": int(len(state_sub)),
                "n_transitions": int(len(transition_sub)),
                "n_motifs": int(len(motif_sub)),
                "n_generator_assignments": int(len(assignment_sub)),
                "n_compositions": int(len(composition_sub)),
                "n_unique_state_types": int(state_sub["state_type"].nunique()) if len(state_sub) else 0,
                "n_unique_reduced_states": int(state_sub["state_red"].nunique()) if len(state_sub) else 0,
                "n_unique_sectors": int(state_sub["state_sector"].nunique()) if len(state_sub) else 0,
                "n_unique_motif_classes": int(motif_sub["motif_class"].nunique()) if len(motif_sub) else 0,
                "n_unique_generator_words": int(assignment_sub["generator_word"].nunique()) if len(assignment_sub) else 0,
                "n_unique_generators_completed": int(assignment_sub["generator_completed"].nunique()) if len(assignment_sub) else 0,
                "n_unique_proto_edges": int(assignment_sub["proto_edge"].nunique()) if len(assignment_sub) else 0,
                "n_unique_proto_sector_edges": int(assignment_sub["proto_sector_edge"].nunique()) if len(assignment_sub) else 0,
                "n_unique_compositions": int(composition_sub["composition"].nunique()) if len(composition_sub) else 0,
                "n_unique_proto_relations": int(composition_sub["proto_relation"].nunique()) if len(composition_sub) else 0,
                "mean_distance_to_seam": safe_mean(profile_sub["mean_distance_to_seam"]) if len(profile_sub) else np.nan,
                "mean_lazarus_score": safe_mean(profile_sub["mean_lazarus_score"]) if len(profile_sub) else np.nan,
                "mean_response_strength": safe_mean(profile_sub["mean_response_strength"]) if len(profile_sub) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_report(
    cfg: Config,
    profile: pd.DataFrame,
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    motifs: pd.DataFrame,
    assignments: pd.DataFrame,
    compositions: pd.DataFrame,
    proto_edges: pd.DataFrame,
    proto_relations: pd.DataFrame,
    route_summary: pd.DataFrame,
    edge_sig: pd.DataFrame,
    relation_sig: pd.DataFrame,
    sector_sig: pd.DataFrame,
    generator_sig: pd.DataFrame,
    composition_sig: pd.DataFrame,
) -> str:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()

    lines = [
        "# OBS-064 — Proto-groupoid symbolic trace cache",
        "",
        f"Corpus label: `{cfg.corpus_label or 'unspecified'}`",
        "",
        "## Purpose",
        "",
        "OBS-064 builds a reusable symbolic path-trace substrate for downstream proto-groupoid survival controls.",
        "",
        "OBS-062 and OBS-063 repeatedly rebuilt motifs, generators, and compositions inside every decoy run. OBS-064 separates that symbolic extraction step from downstream control aggregation.",
        "",
        "The cached trace path is:",
        "",
        "```text",
        "scene_routes",
        "→ state sequences",
        "→ transition steps",
        "→ 3-state motifs",
        "→ generator assignments",
        "→ generator compositions",
        "→ proto-groupoid-ready edges and relations",
        "```",
        "",
        "## Scope",
        "",
        "OBS-064 is a cache/scaffold. It does not run decoy ensembles.",
        "",
        "The intended downstream consumer is OBS-065, which should aggregate these cached per-path fragments under selected/decoy path replacement maps.",
        "",
        "## Trace-cache size",
        "",
        f"- all profiled paths: {profile['path_id'].nunique() if len(profile) else 0}",
        f"- selected origin paths: {selected['path_id'].nunique() if len(selected) else 0}",
        f"- state rows: {len(states)}",
        f"- transition rows: {len(transitions)}",
        f"- motif rows: {len(motifs)}",
        f"- generator assignment rows: {len(assignments)}",
        f"- generator composition rows: {len(compositions)}",
        f"- proto edge rows: {len(proto_edges)}",
        f"- proto relation rows: {len(proto_relations)}",
        "",
        "## Selected origin route-class summary",
        "",
    ]

    for _, row in route_summary.iterrows():
        lines.extend(
            [
                f"### {row['route_class']}",
                "",
                f"- n_paths: {int(row['n_paths'])}",
                f"- total_steps: {int(row['total_steps'])}",
                f"- n_transitions: {int(row['n_transitions'])}",
                f"- n_motifs: {int(row['n_motifs'])}",
                f"- n_generator_assignments: {int(row['n_generator_assignments'])}",
                f"- n_compositions: {int(row['n_compositions'])}",
                f"- n_unique_state_types: {int(row['n_unique_state_types'])}",
                f"- n_unique_reduced_states: {int(row['n_unique_reduced_states'])}",
                f"- n_unique_sectors: {int(row['n_unique_sectors'])}",
                f"- n_unique_motif_classes: {int(row['n_unique_motif_classes'])}",
                f"- n_unique_generator_words: {int(row['n_unique_generator_words'])}",
                f"- n_unique_generators_completed: {int(row['n_unique_generators_completed'])}",
                f"- n_unique_proto_edges: {int(row['n_unique_proto_edges'])}",
                f"- n_unique_proto_sector_edges: {int(row['n_unique_proto_sector_edges'])}",
                f"- n_unique_compositions: {int(row['n_unique_compositions'])}",
                f"- n_unique_proto_relations: {int(row['n_unique_proto_relations'])}",
                f"- mean_distance_to_seam: {float(row['mean_distance_to_seam']):.6f}",
                f"- mean_lazarus_score: {float(row['mean_lazarus_score']):.6f}",
                f"- mean_response_strength: {float(row['mean_response_strength']):.6f}",
                "",
            ]
        )

    def add_top_section(title: str, sig: pd.DataFrame, value_col: str, count_col: str, top_n: int = 5) -> None:
        lines.extend([f"## {title}", ""])
        if len(sig) == 0:
            lines.append("- No rows.")
            lines.append("")
            return

        for cls in CLASS_ORDER:
            sub = sig[sig["route_class"] == cls].sort_values("rank").head(top_n)
            lines.extend([f"### {cls}", ""])
            if len(sub) == 0:
                lines.append("- No rows.")
                lines.append("")
                continue

            for _, row in sub.iterrows():
                lines.append(
                    f"- {row[value_col]}: n={int(row[count_col])}, share={float(row['share']):.6f}"
                )
            lines.append("")

    add_top_section(
        "Baseline proto-edge signatures",
        edge_sig,
        "proto_edge",
        "n_proto_edges",
    )
    add_top_section(
        "Baseline proto-relation signatures",
        relation_sig,
        "proto_relation",
        "n_proto_relations",
    )
    add_top_section(
        "Baseline sector-edge signatures",
        sector_sig,
        "proto_sector_edge",
        "n_proto_sector_edges",
    )
    add_top_section(
        "Baseline generator signatures",
        generator_sig,
        "generator_completed",
        "n_generators",
    )
    add_top_section(
        "Baseline composition signatures",
        composition_sig,
        "composition",
        "n_compositions",
    )

    lines.extend(
        [
            "## Operational consequence",
            "",
            "OBS-064 creates the reusable symbolic substrate needed to make proto-groupoid decoy controls practical.",
            "",
            "Instead of rebuilding state sequences, motifs, generators, and compositions for every decoy run, OBS-065 can select cached path fragments by `path_id` and aggregate them under route-origin replacement maps.",
            "",
            "This should substantially reduce the runtime of downstream proto-groupoid controls and preserve provenance by separating symbolic extraction from decoy aggregation.",
            "",
            "## Recovery note",
            "",
            "OBS-064 does not validate proto-groupoid survival. It prepares the cache and baseline signatures required for that test.",
            "",
            "The next step is OBS-065: proto-groupoid decoy survival controls using these cached symbolic traces.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-064 proto-groupoid symbolic trace cache.")
    parser.add_argument("--scene-routes-csv", default=Config.scene_routes_csv)
    parser.add_argument("--seam-nodes-csv", default=Config.seam_nodes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--corpus-label", default=Config.corpus_label)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--post-exit-threshold", type=float, default=Config.post_exit_threshold)
    args = parser.parse_args()

    cfg = Config(
        scene_routes_csv=args.scene_routes_csv,
        seam_nodes_csv=args.seam_nodes_csv,
        outdir=args.outdir,
        corpus_label=args.corpus_label,
        seam_threshold=float(args.seam_threshold),
        post_exit_threshold=float(args.post_exit_threshold),
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

    states = build_state_sequences(routes, cfg)
    transitions = build_transition_steps(states)
    motifs = build_motifs(transitions)
    assignments = build_generator_assignments(motifs)
    compositions = build_generator_compositions(assignments)
    proto_edges = build_proto_edges(assignments)
    proto_relations = build_proto_relations(compositions)

    selected_ids = selected_origin_ids(profile)

    edge_sig = distribution_signature(
        proto_edges,
        value_col="proto_edge",
        count_col="n_proto_edges",
        selected_ids=selected_ids,
    )
    relation_sig = distribution_signature(
        proto_relations,
        value_col="proto_relation",
        count_col="n_proto_relations",
        selected_ids=selected_ids,
    )
    sector_sig = distribution_signature(
        proto_edges,
        value_col="proto_sector_edge",
        count_col="n_proto_sector_edges",
        selected_ids=selected_ids,
    )
    generator_sig = distribution_signature(
        assignments,
        value_col="generator_completed",
        count_col="n_generators",
        selected_ids=selected_ids,
    )
    composition_sig = distribution_signature(
        compositions,
        value_col="composition",
        count_col="n_compositions",
        selected_ids=selected_ids,
    )

    route_summary = route_class_summary(
        profile,
        states,
        transitions,
        motifs,
        assignments,
        compositions,
    )

    all_profile_csv = outdir / "obs064_all_path_profile.csv"
    selected_profile_csv = outdir / "obs064_selected_origin_path_profile.csv"
    states_csv = outdir / "obs064_path_state_sequences.csv"
    transitions_csv = outdir / "obs064_path_transition_steps.csv"
    motifs_csv = outdir / "obs064_path_motifs.csv"
    assignments_csv = outdir / "obs064_path_generator_assignments.csv"
    compositions_csv = outdir / "obs064_path_generator_compositions.csv"
    proto_edges_csv = outdir / "obs064_path_proto_edges.csv"
    proto_relations_csv = outdir / "obs064_path_proto_relations.csv"
    edge_sig_csv = outdir / "obs064_origin_proto_edge_signature.csv"
    relation_sig_csv = outdir / "obs064_origin_proto_relation_signature.csv"
    sector_sig_csv = outdir / "obs064_origin_sector_signature.csv"
    generator_sig_csv = outdir / "obs064_origin_generator_signature.csv"
    composition_sig_csv = outdir / "obs064_origin_composition_signature.csv"
    route_summary_csv = outdir / "obs064_route_class_summary.csv"
    report_md = outdir / "obs064_proto_groupoid_symbolic_trace_cache_report.md"

    profile.to_csv(all_profile_csv, index=False)
    selected_profile.to_csv(selected_profile_csv, index=False)
    states.to_csv(states_csv, index=False)
    transitions.to_csv(transitions_csv, index=False)
    motifs.to_csv(motifs_csv, index=False)
    assignments.to_csv(assignments_csv, index=False)
    compositions.to_csv(compositions_csv, index=False)
    proto_edges.to_csv(proto_edges_csv, index=False)
    proto_relations.to_csv(proto_relations_csv, index=False)
    edge_sig.to_csv(edge_sig_csv, index=False)
    relation_sig.to_csv(relation_sig_csv, index=False)
    sector_sig.to_csv(sector_sig_csv, index=False)
    generator_sig.to_csv(generator_sig_csv, index=False)
    composition_sig.to_csv(composition_sig_csv, index=False)
    route_summary.to_csv(route_summary_csv, index=False)

    report_md.write_text(
        build_report(
            cfg,
            profile,
            states,
            transitions,
            motifs,
            assignments,
            compositions,
            proto_edges,
            proto_relations,
            route_summary,
            edge_sig,
            relation_sig,
            sector_sig,
            generator_sig,
            composition_sig,
        ),
        encoding="utf-8",
    )

    print(all_profile_csv)
    print(selected_profile_csv)
    print(states_csv)
    print(transitions_csv)
    print(motifs_csv)
    print(assignments_csv)
    print(compositions_csv)
    print(proto_edges_csv)
    print(proto_relations_csv)
    print(edge_sig_csv)
    print(relation_sig_csv)
    print(sector_sig_csv)
    print(generator_sig_csv)
    print(composition_sig_csv)
    print(route_summary_csv)
    print(report_md)


if __name__ == "__main__":
    main()
