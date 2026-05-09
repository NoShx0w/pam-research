#!/usr/bin/env python3
"""
OBS-066 — Gateway and canonical-family decoy survival.

Consumes the OBS-064 proto-groupoid symbolic trace cache and, when available,
reuses OBS-065 replacement pair plans so that gateway/canonical-family survival
is evaluated under exactly the same decoy runs.

OBS-066 asks whether downstream gateway and canonical-family summaries survive
route-origin decoy replacement even when fine proto-groupoid signatures drift.

Derived layers
--------------
gateway_event
    sector movement only

gateway_event_generator
    sector movement plus generator label

gateway_event_relation
    ordered pair of gateway events

gateway_event_generator_relation
    ordered pair of generator-qualified gateway events

canonical_family
    coarse sector-action family

canonical_family_relation
    ordered pair of canonical families

anchor_relation_indicator
    anchor / non_anchor mass, using OBS-065 anchors when available

anchor_canonical_family
    canonical relation family marked by anchor status

Inputs
------
OBS-064 cache directory containing at least:

  obs064_all_path_profile.csv
  obs064_path_proto_edges.csv
  obs064_path_proto_relations.csv

Optional OBS-065 directory containing:

  obs065_replacement_pairs.csv
  obs065_replacement_runs.csv
  obs065_ranked_nonexact_decoy_candidates.csv
  obs065_proto_anchor_candidates.csv

If OBS-065 pair plans are absent, OBS-066 deterministically rebuilds non-exact
decoy candidates and replacement pair plans.

Outputs
-------
  obs066_path_gateway_signature_counts.csv
  obs066_path_canonical_family_counts.csv
  obs066_ranked_nonexact_decoy_candidates.csv
  obs066_replacement_pairs.csv
  obs066_replacement_runs.csv
  obs066_gateway_signature_baseline.csv
  obs066_gateway_signature_replacement.csv       # optional
  obs066_gateway_layer_survival_drift.csv
  obs066_gateway_survival_summary.csv
  obs066_canonical_family_survival_summary.csv
  obs066_anchor_family_survival_summary.csv
  obs066_cross_layer_consequence_modes.csv
  obs066_gateway_canonical_family_survival_report.md
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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

LAYER_ORDER = [
    "gateway_event",
    "gateway_event_generator",
    "gateway_event_relation",
    "gateway_event_generator_relation",
    "canonical_family",
    "canonical_family_relation",
    "anchor_relation_indicator",
    "anchor_canonical_family",
]


@dataclass(frozen=True)
class Config:
    trace_cache_dir: str
    obs065_dir: str
    outdir: str
    corpus_label: str
    min_matching_distance: float = 1e-9
    exact_match_tolerance: float = 1e-12
    rank_bands: str = "1-10,51-250,all"
    n_iter: int = 250
    random_seed: int = 42
    allow_decoy_reuse: bool = False
    branch_decoy_same_family: bool = False
    max_candidates_per_selected: int = 1000
    n_workers: int = 1
    write_replacement_signatures: bool = False
    strong_survival_threshold: float = 0.80
    weak_survival_threshold: float = 0.50


def read_csv_numeric(path: str | Path, text_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    text = set(text_cols)
    for col in df.columns:
        if col not in text:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_rank_bands(raw: str) -> list[dict[str, object]]:
    bands = []
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
        bands = [{"band": "rank_1_10", "rank_min": 1, "rank_max": 10}]

    return bands


def selected_origin_ids(profile: pd.DataFrame) -> set[str]:
    return set(profile.loc[profile["route_class"].isin(CLASS_ORDER), "path_id"].astype(str))


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


def sectorize_state(state: str) -> str:
    s = str(state).strip()
    if s in {"R", "A", "L", "C"}:
        return "seam_sector"
    if s == "O":
        return "off_seam_sector"
    if s == "P":
        return "post_exit_sector"
    if s in {"seam_sector", "off_seam_sector", "post_exit_sector"}:
        return s
    return "unknown_sector"


def gateway_event_from_sectors(src_sector: str, tgt_sector: str) -> str:
    src = str(src_sector)
    tgt = str(tgt_sector)

    if src == "off_seam_sector" and tgt == "seam_sector":
        return "off_to_seam_gateway"
    if src == "seam_sector" and tgt == "off_seam_sector":
        return "seam_to_off_gateway"
    if src == "seam_sector" and tgt == "post_exit_sector":
        return "seam_to_post_gateway"
    if src == "off_seam_sector" and tgt == "post_exit_sector":
        return "off_to_post_gateway"
    if src == "post_exit_sector" and tgt == "off_seam_sector":
        return "post_to_off_gateway"
    if src == "post_exit_sector" and tgt == "post_exit_sector":
        return "post_persistence"
    if src == "seam_sector" and tgt == "seam_sector":
        return "seam_persistence"
    if src == "off_seam_sector" and tgt == "off_seam_sector":
        return "off_persistence"
    if src == "post_exit_sector" and tgt == "seam_sector":
        return "post_to_seam_gateway"
    return "other_gateway_event"


def canonical_family_from_gateway_event(event: str) -> str:
    mapping = {
        "seam_persistence": "seam_internal_family",
        "off_persistence": "off_internal_family",
        "post_persistence": "post_internal_family",
        "off_to_seam_gateway": "off_to_seam_family",
        "seam_to_off_gateway": "seam_to_off_family",
        "seam_to_post_gateway": "seam_to_post_family",
        "off_to_post_gateway": "off_to_post_family",
        "post_to_off_gateway": "post_to_off_family",
        "post_to_seam_gateway": "post_to_seam_family",
    }
    return mapping.get(str(event), "mixed_or_other_family")


def parse_proto_edge(edge: str) -> tuple[str, str, str]:
    m = re.match(r"(.+?) --(.+?)--> (.+)", str(edge))
    if not m:
        return "unknown_state", "g_unknown", "unknown_state"
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


def parse_sector_edge(edge: str) -> tuple[str, str, str]:
    src, gen, tgt = parse_proto_edge(edge)
    return sectorize_state(src), gen, sectorize_state(tgt)


def split_relation(rel: str) -> list[str]:
    if " ; " in str(rel):
        return [x.strip() for x in str(rel).split(" ; ")]
    return [str(rel).strip()]


def relation_to_gateway_parts(proto_relation: str) -> tuple[list[str], list[str], list[str]]:
    events = []
    event_generators = []
    families = []

    for part in split_relation(proto_relation):
        src, gen, tgt = parse_sector_edge(part)
        event = gateway_event_from_sectors(src, tgt)
        family = canonical_family_from_gateway_event(event)
        events.append(event)
        event_generators.append(f"{event}::{gen}")
        families.append(family)

    return events, event_generators, families


def load_anchor_relation_set(obs065_dir: Path) -> set[str]:
    path = obs065_dir / "obs065_proto_anchor_candidates.csv"
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    if "anchor_class" not in df.columns or "baseline_proto_relation" not in df.columns:
        return set()

    anchor_df = df[df["anchor_class"].astype(str).isin(["strong_proto_anchor", "weak_proto_anchor", "sector_anchor"])].copy()
    return set(anchor_df["baseline_proto_relation"].astype(str))


def build_path_gateway_signature_counts(cache_dir: Path, obs065_dir: Path) -> pd.DataFrame:
    edges_path = cache_dir / "obs064_path_proto_edges.csv"
    relations_path = cache_dir / "obs064_path_proto_relations.csv"

    edges = pd.read_csv(edges_path, usecols=["path_id", "proto_sector_edge"])
    relations = pd.read_csv(relations_path, usecols=["path_id", "proto_relation", "proto_sector_relation"])

    frames = []

    # Edge-derived gateway events and canonical families.
    edge_rows = []
    for _, row in edges.dropna(subset=["path_id", "proto_sector_edge"]).iterrows():
        path_id = str(row["path_id"])
        src, gen, tgt = parse_sector_edge(str(row["proto_sector_edge"]))
        event = gateway_event_from_sectors(src, tgt)
        family = canonical_family_from_gateway_event(event)

        edge_rows.append((path_id, "gateway_event", event))
        edge_rows.append((path_id, "gateway_event_generator", f"{event}::{gen}"))
        edge_rows.append((path_id, "canonical_family", family))

    if edge_rows:
        df = pd.DataFrame(edge_rows, columns=["path_id", "layer", "signature_value"])
        frames.append(
            df.groupby(["path_id", "layer", "signature_value"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

    # Relation-derived gateway-event sequences and canonical-family sequences.
    anchor_set = load_anchor_relation_set(obs065_dir)
    relation_rows = []

    for _, row in relations.dropna(subset=["path_id", "proto_relation"]).iterrows():
        path_id = str(row["path_id"])
        proto_relation = str(row["proto_relation"])

        events, event_generators, families = relation_to_gateway_parts(proto_relation)
        event_relation = " ; ".join(events)
        event_generator_relation = " ; ".join(event_generators)
        family_relation = " ; ".join(families)

        anchor_indicator = "anchor_relation" if proto_relation in anchor_set else "non_anchor_relation"
        anchor_family = f"{anchor_indicator}::{family_relation}"

        relation_rows.append((path_id, "gateway_event_relation", event_relation))
        relation_rows.append((path_id, "gateway_event_generator_relation", event_generator_relation))
        relation_rows.append((path_id, "canonical_family_relation", family_relation))
        relation_rows.append((path_id, "anchor_relation_indicator", anchor_indicator))
        relation_rows.append((path_id, "anchor_canonical_family", anchor_family))

    if relation_rows:
        df = pd.DataFrame(relation_rows, columns=["path_id", "layer", "signature_value"])
        frames.append(
            df.groupby(["path_id", "layer", "signature_value"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

    if not frames:
        return pd.DataFrame(columns=["path_id", "layer", "signature_value", "count"])

    out = pd.concat(frames, ignore_index=True)
    out = (
        out.groupby(["path_id", "layer", "signature_value"], as_index=False)["count"]
        .sum()
        .sort_values(["path_id", "layer", "signature_value"])
        .reset_index(drop=True)
    )
    out["path_id"] = out["path_id"].astype(str)
    return out


def feature_scale_table(profile: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for col in MATCH_FEATURES:
        x = pd.to_numeric(profile[col], errors="coerce")
        mu = float(x.mean()) if x.notna().any() else 0.0
        sd = float(x.std(ddof=0)) if x.notna().any() else 1.0
        if not np.isfinite(sd) or sd == 0.0:
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
        val = pd.to_numeric(row.get(f"delta_{col}"), errors="coerce")
        if pd.notna(val):
            deltas.append(abs(float(val)))
    return max(deltas) if deltas else float("inf")


def build_decoy_candidate_pool(profile: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    selected_ids = selected_origin_ids(profile)

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

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_ranked_nonexact_candidates(profile: pd.DataFrame, pool: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    selected = selected.sort_values(["route_class", "path_id"]).reset_index(drop=True)
    scales = feature_scale_table(profile)
    rows = []

    for _, sel in selected.iterrows():
        cls = str(sel["route_class"])
        candidates = pool[pool["target_route_class"] == cls].copy()
        if len(candidates) == 0:
            continue

        candidates["matching_distance"] = [
            standardized_distance(sel, cand, scales) for _, cand in candidates.iterrows()
        ]

        for col in MATCH_FEATURES:
            candidates[f"selected_{col}"] = sel.get(col, np.nan)
            candidates[f"decoy_{col}"] = candidates[col]
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
        candidates["selected_path_id"] = str(sel["path_id"])
        candidates["selected_path_family"] = sel.get("path_family", "")
        candidates["selected_is_branch_away"] = int(sel.get("is_branch_away", 0))
        candidates["selected_is_representative"] = int(sel.get("is_representative", 0))

        rows.append(candidates)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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


def precompute_pair_plans(profile: pd.DataFrame, ranked: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_seed)
    bands = parse_rank_bands(cfg.rank_bands)
    pair_frames = []
    run_rows = []

    for band_cfg in bands:
        band_name = str(band_cfg["band"])
        rank_min = int(band_cfg["rank_min"])
        rank_max = int(band_cfg["rank_max"]) if band_cfg["rank_max"] is not None else None

        for iteration in range(cfg.n_iter):
            pairs = select_band_pairs(
                ranked,
                profile,
                band_name=band_name,
                rank_min=rank_min,
                rank_max=rank_max,
                rng=rng,
                allow_decoy_reuse=cfg.allow_decoy_reuse,
            )
            pairs["iteration"] = iteration
            pair_frames.append(pairs)

            for cls in CLASS_ORDER:
                matched = pairs[
                    (pairs["route_class"] == cls)
                    & (pairs["match_status"] == "matched_band_nonexact")
                ].copy()

                run_rows.append(
                    {
                        "run_id": f"{band_name}_iter_{iteration:04d}_replace_{cls}",
                        "band": band_name,
                        "rank_min": rank_min,
                        "rank_max": rank_max if rank_max is not None else np.nan,
                        "iteration": iteration,
                        "replacement_route_class": cls,
                        "matched_pairs": int(len(matched)),
                        "unmatched_pairs": int(8 - len(matched)),
                        "mean_decoy_rank": float(matched["eligible_decoy_rank"].mean()) if len(matched) else np.nan,
                        "mean_matching_distance": float(matched["matching_distance"].mean()) if len(matched) else np.nan,
                    }
                )

    pairs_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    runs = pd.DataFrame(run_rows)
    return pairs_all, runs


def load_or_build_pair_plans(profile: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs065_dir = Path(cfg.obs065_dir) if cfg.obs065_dir else Path("")
    pairs_path = obs065_dir / "obs065_replacement_pairs.csv"
    runs_path = obs065_dir / "obs065_replacement_runs.csv"
    ranked_path = obs065_dir / "obs065_ranked_nonexact_decoy_candidates.csv"

    if cfg.obs065_dir and pairs_path.exists() and runs_path.exists():
        pairs = pd.read_csv(pairs_path)
        runs = pd.read_csv(runs_path)
        ranked = pd.read_csv(ranked_path) if ranked_path.exists() else pd.DataFrame()
        pairs["selected_path_id"] = pairs["selected_path_id"].astype(str)
        pairs["decoy_path_id"] = pairs["decoy_path_id"].astype(str)
        return ranked, pairs, runs

    pool = build_decoy_candidate_pool(profile, cfg)
    ranked = build_ranked_nonexact_candidates(profile, pool, cfg)
    pairs, runs = precompute_pair_plans(profile, ranked, cfg)
    return ranked, pairs, runs


def aggregate_signature_for_map(
    path_counts: pd.DataFrame,
    path_class_map: dict[str, str],
    *,
    run_id: str,
    band: str,
    iteration: int,
    replacement_route_class: str,
) -> pd.DataFrame:
    if len(path_counts) == 0 or not path_class_map:
        return pd.DataFrame()

    pids = set(path_class_map.keys())
    sub = path_counts[path_counts["path_id"].astype(str).isin(pids)].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    sub["route_class"] = sub["path_id"].astype(str).map(path_class_map)
    sub = sub[sub["route_class"].isin(CLASS_ORDER)].copy()

    grouped = (
        sub.groupby(["route_class", "layer", "signature_value"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "n_items"})
    )

    totals = (
        grouped.groupby(["route_class", "layer"], as_index=False)["n_items"]
        .sum()
        .rename(columns={"n_items": "total_items"})
    )

    grouped = grouped.merge(totals, on=["route_class", "layer"], how="left")
    grouped["share"] = grouped["n_items"] / grouped["total_items"].replace(0, np.nan)
    grouped["share"] = grouped["share"].fillna(0.0)

    grouped = grouped.sort_values(
        ["route_class", "layer", "share", "signature_value"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    grouped["rank"] = grouped.groupby(["route_class", "layer"]).cumcount() + 1

    grouped["run_id"] = run_id
    grouped["band"] = band
    grouped["iteration"] = iteration
    grouped["replacement_route_class"] = replacement_route_class

    return grouped[
        [
            "run_id",
            "band",
            "iteration",
            "replacement_route_class",
            "route_class",
            "layer",
            "signature_value",
            "n_items",
            "share",
            "rank",
        ]
    ].copy()


def dist_map(sig: pd.DataFrame, route_class: str, layer: str) -> dict[str, float]:
    sub = sig[(sig["route_class"] == route_class) & (sig["layer"] == layer)]
    return {str(row["signature_value"]): float(row["share"]) for _, row in sub.iterrows()}


def top_values(sig: pd.DataFrame, route_class: str, layer: str, k: int = 3) -> list[str]:
    sub = sig[(sig["route_class"] == route_class) & (sig["layer"] == layer)].sort_values("rank")
    return [str(x) for x in sub["signature_value"].head(k).tolist()]


def n_items(sig: pd.DataFrame, route_class: str, layer: str) -> int:
    sub = sig[(sig["route_class"] == route_class) & (sig["layer"] == layer)]
    if len(sub) == 0:
        return 0
    return int(pd.to_numeric(sub["n_items"], errors="coerce").sum())


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def top_margin(sig: pd.DataFrame, route_class: str, layer: str) -> dict[str, object]:
    sub = sig[(sig["route_class"] == route_class) & (sig["layer"] == layer)].sort_values("rank").reset_index(drop=True)

    if len(sub) == 0:
        return {
            "top1_label": "",
            "top1_share": 0.0,
            "top2_label": "",
            "top2_share": 0.0,
            "top1_minus_top2_margin": 0.0,
        }

    top1_label = str(sub.iloc[0]["signature_value"])
    top1_share = float(sub.iloc[0]["share"])

    if len(sub) > 1:
        top2_label = str(sub.iloc[1]["signature_value"])
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


def build_layer_drift(
    baseline_sig: pd.DataFrame,
    replacement_sig: pd.DataFrame,
    *,
    run_id: str,
    band: str,
    iteration: int,
    replacement_route_class: str,
) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        for layer in LAYER_ORDER:
            p = dist_map(baseline_sig, cls, layer)
            q = dist_map(replacement_sig, cls, layer)

            base_top3 = top_values(baseline_sig, cls, layer, k=3)
            repl_top3 = top_values(replacement_sig, cls, layer, k=3)

            base_top1 = base_top3[0] if base_top3 else ""
            repl_top1 = repl_top3[0] if repl_top3 else ""
            margin = top_margin(baseline_sig, cls, layer)

            rows.append(
                {
                    "run_id": run_id,
                    "band": band,
                    "iteration": iteration,
                    "replacement_route_class": replacement_route_class,
                    "evaluated_route_class": cls,
                    "is_replaced_class": int(cls == replacement_route_class),
                    "layer": layer,
                    "baseline_n_items": n_items(baseline_sig, cls, layer),
                    "replacement_n_items": n_items(replacement_sig, cls, layer),
                    "distribution_tv_distance": total_variation(p, q),
                    "top1_baseline": base_top1,
                    "top1_replacement": repl_top1,
                    "top1_survived": int(base_top1 == repl_top1),
                    "top1_changed": int(base_top1 != repl_top1),
                    "top3_baseline": " | ".join(base_top3),
                    "top3_replacement": " | ".join(repl_top3),
                    "top3_overlap": len(set(base_top3) & set(repl_top3)),
                    "top3_overlap_share": len(set(base_top3) & set(repl_top3)) / 3.0 if (base_top3 or repl_top3) else 0.0,
                    "baseline_top1_share": margin["top1_share"],
                    "baseline_top2_share": margin["top2_share"],
                    "baseline_top1_minus_top2_margin": margin["top1_minus_top2_margin"],
                    "baseline_top1_label": margin["top1_label"],
                    "baseline_top2_label": margin["top2_label"],
                }
            )

    return pd.DataFrame(rows)


_WORKER_PROFILE = None
_WORKER_COUNTS = None
_WORKER_BASELINE_SIG = None
_WORKER_PAIRS = None
_WORKER_WRITE_SIG = False


def _init_worker(profile: pd.DataFrame, counts: pd.DataFrame, baseline_sig: pd.DataFrame, pairs: pd.DataFrame, write_sig: bool) -> None:
    global _WORKER_PROFILE, _WORKER_COUNTS, _WORKER_BASELINE_SIG, _WORKER_PAIRS, _WORKER_WRITE_SIG
    _WORKER_PROFILE = profile
    _WORKER_COUNTS = counts
    _WORKER_BASELINE_SIG = baseline_sig
    _WORKER_PAIRS = pairs
    _WORKER_WRITE_SIG = write_sig


def _run_one_task(task: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    band = str(task["band"])
    iteration = int(task["iteration"])
    replace_class = str(task["replacement_route_class"])
    run_id = str(task["run_id"])

    pairs_i = _WORKER_PAIRS[
        (_WORKER_PAIRS["band"].astype(str) == band)
        & (pd.to_numeric(_WORKER_PAIRS["iteration"], errors="coerce").astype(int) == iteration)
    ].copy()

    repl_map = replacement_path_class_map(_WORKER_PROFILE, pairs_i, replace_class)

    replacement_sig = aggregate_signature_for_map(
        _WORKER_COUNTS,
        repl_map,
        run_id=run_id,
        band=band,
        iteration=iteration,
        replacement_route_class=replace_class,
    )

    drift = build_layer_drift(
        _WORKER_BASELINE_SIG,
        replacement_sig,
        run_id=run_id,
        band=band,
        iteration=iteration,
        replacement_route_class=replace_class,
    )

    return drift, replacement_sig if _WORKER_WRITE_SIG else None


def run_controls(
    profile: pd.DataFrame,
    counts: pd.DataFrame,
    baseline_sig: pd.DataFrame,
    pairs: pd.DataFrame,
    runs: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = runs[["run_id", "band", "iteration", "replacement_route_class"]].to_dict("records")

    if cfg.n_workers <= 1:
        _init_worker(profile, counts, baseline_sig, pairs, cfg.write_replacement_signatures)
        outputs = [_run_one_task(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=cfg.n_workers,
            initializer=_init_worker,
            initargs=(profile, counts, baseline_sig, pairs, cfg.write_replacement_signatures),
        ) as pool:
            outputs = list(pool.imap_unordered(_run_one_task, tasks, chunksize=10))

    drift_frames = [x[0] for x in outputs if x[0] is not None and len(x[0])]
    sig_frames = [x[1] for x in outputs if x[1] is not None and len(x[1])]

    drift = pd.concat(drift_frames, ignore_index=True) if drift_frames else pd.DataFrame()
    repl_sig = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()
    return drift, repl_sig


def summarize_survival(runs: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    if len(drift) == 0:
        return pd.DataFrame()

    replaced = drift[drift["is_replaced_class"] == 1].copy()
    rows = []

    for (band, cls, layer), grp in replaced.groupby(["band", "replacement_route_class", "layer"], dropna=False):
        run_grp = runs[
            (runs["band"] == band)
            & (runs["replacement_route_class"] == cls)
        ].copy()

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
                "top1_survival_rate": float(grp["top1_survived"].mean()) if len(grp) else np.nan,
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
            }
        )

    out = pd.DataFrame(rows)
    band_order = {str(b["band"]): i for i, b in enumerate(parse_rank_bands("1-10,51-250,all"))}
    class_order = {c: i for i, c in enumerate(CLASS_ORDER)}
    layer_order = {l: i for i, l in enumerate(LAYER_ORDER)}
    out["band_order"] = out["band"].map(lambda x: band_order.get(str(x), 999))
    out["class_order"] = out["replacement_route_class"].map(lambda x: class_order.get(str(x), 999))
    out["layer_order"] = out["layer"].map(lambda x: layer_order.get(str(x), 999))
    return (
        out.sort_values(["band_order", "class_order", "layer_order"])
        .drop(columns=["band_order", "class_order", "layer_order"])
        .reset_index(drop=True)
    )


def split_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gateway_layers = [
        "gateway_event",
        "gateway_event_generator",
        "gateway_event_relation",
        "gateway_event_generator_relation",
    ]
    canonical_layers = ["canonical_family", "canonical_family_relation"]
    anchor_layers = ["anchor_relation_indicator", "anchor_canonical_family"]

    return (
        summary[summary["layer"].isin(gateway_layers)].copy(),
        summary[summary["layer"].isin(canonical_layers)].copy(),
        summary[summary["layer"].isin(anchor_layers)].copy(),
    )


def survival_lookup(summary: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    out = {}
    for _, row in summary.iterrows():
        out[(str(row["band"]), str(row["replacement_route_class"]), str(row["layer"]))] = float(row["top1_survival_rate"])
    return out


def load_obs065_proto_summary(obs065_dir: Path) -> pd.DataFrame:
    p = obs065_dir / "obs065_proto_survival_summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def proto_lookup(obs065_summary: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    out = {}
    if len(obs065_summary) == 0:
        return out
    for _, row in obs065_summary.iterrows():
        out[(str(row["band"]), str(row["replacement_route_class"]), str(row["layer"]))] = float(row["top1_survival_rate"])
    return out


def build_cross_layer_consequence_modes(summary: pd.DataFrame, obs065_summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if len(summary) == 0:
        return pd.DataFrame()

    obs66 = survival_lookup(summary)
    obs65 = proto_lookup(obs065_summary)
    rows = []

    for band in summary["band"].astype(str).unique():
        for cls in CLASS_ORDER:
            proto_rel = obs65.get((band, cls, "proto_relation"), np.nan)
            proto_edge = obs65.get((band, cls, "proto_edge"), np.nan)
            gateway_event = obs66.get((band, cls, "gateway_event"), np.nan)
            gateway_event_gen = obs66.get((band, cls, "gateway_event_generator"), np.nan)
            gateway_relation = obs66.get((band, cls, "gateway_event_relation"), np.nan)
            gateway_gen_relation = obs66.get((band, cls, "gateway_event_generator_relation"), np.nan)
            canonical = obs66.get((band, cls, "canonical_family"), np.nan)
            canonical_relation = obs66.get((band, cls, "canonical_family_relation"), np.nan)
            anchor_indicator = obs66.get((band, cls, "anchor_relation_indicator"), np.nan)
            anchor_family = obs66.get((band, cls, "anchor_canonical_family"), np.nan)

            if np.isfinite(proto_rel) and np.isfinite(gateway_relation) and proto_rel < cfg.weak_survival_threshold and gateway_relation >= cfg.strong_survival_threshold:
                mode = "fine_proto_fails_gateway_survives"
            elif np.isfinite(gateway_event_gen) and np.isfinite(canonical) and gateway_event_gen < cfg.weak_survival_threshold and canonical >= cfg.strong_survival_threshold:
                mode = "gateway_fails_canonical_survives"
            elif np.isfinite(proto_rel) and np.isfinite(anchor_family) and proto_rel < cfg.weak_survival_threshold and anchor_family >= cfg.strong_survival_threshold:
                mode = "anchor_family_survives_relation_fails"
            elif all(np.isfinite(x) and x >= cfg.strong_survival_threshold for x in [gateway_event, canonical, canonical_relation]):
                mode = "broad_downstream_survival"
            elif all(np.isfinite(x) and x < cfg.weak_survival_threshold for x in [gateway_event, gateway_relation, canonical, canonical_relation]):
                mode = "broad_downstream_failure"
            else:
                mode = "mixed_consequence"

            rows.append(
                {
                    "band": band,
                    "replacement_route_class": cls,
                    "obs065_proto_edge_top1_survival": proto_edge,
                    "obs065_proto_relation_top1_survival": proto_rel,
                    "gateway_event_top1_survival": gateway_event,
                    "gateway_event_generator_top1_survival": gateway_event_gen,
                    "gateway_event_relation_top1_survival": gateway_relation,
                    "gateway_event_generator_relation_top1_survival": gateway_gen_relation,
                    "canonical_family_top1_survival": canonical,
                    "canonical_family_relation_top1_survival": canonical_relation,
                    "anchor_relation_indicator_top1_survival": anchor_indicator,
                    "anchor_canonical_family_top1_survival": anchor_family,
                    "consequence_mode": mode,
                }
            )

    return pd.DataFrame(rows)


def summarize_selected(profile: pd.DataFrame) -> pd.DataFrame:
    selected = profile[profile["route_class"].isin(CLASS_ORDER)].copy()
    rows = []
    for cls in CLASS_ORDER:
        sub = selected[selected["route_class"] == cls]
        rows.append(
            {
                "route_class": cls,
                "n_paths": int(len(sub)),
                "total_steps": int(pd.to_numeric(sub["n_steps"], errors="coerce").sum()) if len(sub) else 0,
                "mean_steps_per_path": float(pd.to_numeric(sub["n_steps"], errors="coerce").mean()) if len(sub) else np.nan,
                "mean_distance_to_seam": float(pd.to_numeric(sub["mean_distance_to_seam"], errors="coerce").mean()) if len(sub) else np.nan,
                "mean_lazarus_score": float(pd.to_numeric(sub["mean_lazarus_score"], errors="coerce").mean()) if len(sub) else np.nan,
                "mean_response_strength": float(pd.to_numeric(sub["mean_response_strength"], errors="coerce").mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_report(
    cfg: Config,
    profile: pd.DataFrame,
    counts: pd.DataFrame,
    ranked: pd.DataFrame,
    pairs: pd.DataFrame,
    runs: pd.DataFrame,
    baseline_sig: pd.DataFrame,
    summary: pd.DataFrame,
    gateway_summary: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    consequence_modes: pd.DataFrame,
) -> str:
    selected_summary = summarize_selected(profile)
    bands = parse_rank_bands(cfg.rank_bands)

    lines = [
        "# OBS-066 — Gateway and canonical-family decoy survival",
        "",
        f"Corpus label: `{cfg.corpus_label or 'unspecified'}`",
        "",
        "## Purpose",
        "",
        "OBS-066 tests whether gateway and canonical-family summaries survive route-origin decoy replacement after proto-groupoid signatures are recomputed from the OBS-064 cached symbolic trace substrate.",
        "",
        "OBS-065 showed that proto-groupoid survival is layer-specific. OBS-066 asks whether downstream gateway/canonical summaries absorb or amplify that proto-groupoid drift.",
        "",
        "## Optimization status",
        "",
        f"- path gateway/family signature count rows: {len(counts)}",
        f"- replacement_runs: {len(runs)}",
        f"- n_workers: {cfg.n_workers}",
        f"- write_replacement_signatures: {cfg.write_replacement_signatures}",
        f"- reused_obs065_dir: `{cfg.obs065_dir}`",
        "",
        "## Tested layers",
        "",
    ]

    for layer in LAYER_ORDER:
        lines.append(f"- `{layer}`")
    lines.append("")

    lines.extend(
        [
            "## Decoy policy",
            "",
            "Matching uses the same standardized path-profile features as OBS-065:",
            "",
        ]
    )
    for f in MATCH_FEATURES:
        lines.append(f"- `{f}`")
    lines.extend(
        [
            "",
            f"`min_matching_distance`: `{cfg.min_matching_distance}`",
            f"`exact_match_tolerance`: `{cfg.exact_match_tolerance}`",
            f"`rank_bands`: `{cfg.rank_bands}`",
            f"`n_iter`: `{cfg.n_iter}`",
            f"`random_seed`: `{cfg.random_seed}`",
            f"`allow_decoy_reuse`: `{cfg.allow_decoy_reuse}`",
            f"`branch_decoy_same_family`: `{cfg.branch_decoy_same_family}`",
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
    if len(ranked):
        for cls in CLASS_ORDER:
            if "selected_route_class" in ranked.columns:
                n_ranked = int((ranked["selected_route_class"] == cls).sum())
            else:
                n_ranked = 0
            lines.append(f"- {cls}: eligible_nonexact_ranked_candidates={n_ranked}")
    else:
        lines.append("- Reused OBS-065 pair plans; ranked candidate table unavailable or not rewritten.")
    lines.append("")

    lines.extend(["## Baseline top gateway/canonical signatures", ""])
    for layer in LAYER_ORDER:
        lines.extend([f"### {layer}", ""])
        for cls in CLASS_ORDER:
            sub = baseline_sig[
                (baseline_sig["layer"] == layer)
                & (baseline_sig["route_class"] == cls)
            ].sort_values("rank").head(3)
            lines.append(f"#### {cls}")
            lines.append("")
            if len(sub) == 0:
                lines.append("- No rows.")
            else:
                for _, row in sub.iterrows():
                    lines.append(
                        f"- {row['signature_value']}: n={int(row['n_items'])}, share={float(row['share']):.6f}"
                    )
            lines.append("")

    lines.extend(["## Gateway survival summary", ""])
    for _, row in gateway_summary.iterrows():
        lines.extend(
            [
                f"### {row['band']} — replace {row['replacement_route_class']} — {row['layer']}",
                "",
                f"- n_runs: {int(row['n_runs'])}",
                f"- unmatched_runs: {int(row['unmatched_runs'])}",
                f"- mean_decoy_rank: {float(row['mean_decoy_rank']):.3f}",
                f"- mean_matching_distance: {float(row['mean_matching_distance']):.6f}",
                f"- top1_survival_rate: {float(row['top1_survival_rate']):.6f}",
                f"- mean_top3_overlap_share: {float(row['mean_top3_overlap_share']):.6f}",
                f"- median_distribution_tv_distance: {float(row['median_distribution_tv_distance']):.6f}",
                f"- p90_distribution_tv_distance: {float(row['p90_distribution_tv_distance']):.6f}",
                f"- baseline_top1_label: {row['baseline_top1_label']}",
                f"- baseline_top2_label: {row['baseline_top2_label']}",
                f"- baseline_top1_minus_top2_margin: {float(row['baseline_top1_minus_top2_margin']):.6f}",
                "",
            ]
        )

    lines.extend(["## Canonical-family survival summary", ""])
    for _, row in canonical_summary.iterrows():
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
                "",
            ]
        )

    lines.extend(["## Anchor-family survival summary", ""])
    for _, row in anchor_summary.iterrows():
        lines.extend(
            [
                f"### {row['band']} — replace {row['replacement_route_class']} — {row['layer']}",
                "",
                f"- n_runs: {int(row['n_runs'])}",
                f"- top1_survival_rate: {float(row['top1_survival_rate']):.6f}",
                f"- mean_top3_overlap_share: {float(row['mean_top3_overlap_share']):.6f}",
                f"- median_distribution_tv_distance: {float(row['median_distribution_tv_distance']):.6f}",
                f"- baseline_top1_label: {row['baseline_top1_label']}",
                "",
            ]
        )

    lines.extend(["## Cross-layer consequence modes", ""])
    if len(consequence_modes):
        for _, row in consequence_modes.iterrows():
            lines.append(
                f"- {row['band']} / replace {row['replacement_route_class']}: "
                f"{row['consequence_mode']} "
                f"(proto_relation={float(row['obs065_proto_relation_top1_survival']) if pd.notna(row['obs065_proto_relation_top1_survival']) else float('nan'):.3f}, "
                f"gateway_relation={float(row['gateway_event_relation_top1_survival']) if pd.notna(row['gateway_event_relation_top1_survival']) else float('nan'):.3f}, "
                f"canonical_relation={float(row['canonical_family_relation_top1_survival']) if pd.notna(row['canonical_family_relation_top1_survival']) else float('nan'):.3f}, "
                f"anchor_family={float(row['anchor_canonical_family_top1_survival']) if pd.notna(row['anchor_canonical_family_top1_survival']) else float('nan'):.3f})"
            )
    lines.append("")

    unmatched_total = int(runs["unmatched_pairs"].sum()) if len(runs) and "unmatched_pairs" in runs.columns else 0

    lines.extend(
        [
            "## Diagnostic totals",
            "",
            f"- replacement_runs: {len(runs)}",
            f"- replacement_pairs: {len(pairs)}",
            f"- unmatched_pairs_total: {unmatched_total}",
            "",
            "## Interpretation guardrail",
            "",
            "OBS-066 tests survival of gateway/canonical-family summaries under route-origin decoy replacement.",
            "",
            "It does not test predictive gateway models, does not establish causal gateway mechanisms, and does not revise OBS-030c/d/e generator rules.",
            "",
            "Gateway-event summaries are sector-movement projections of proto-groupoid structure.",
            "",
            "Canonical-family summaries are coarser still. If canonical families survive while gateway or proto-relation signatures fail, the downstream family layer is absorbing fine symbolic drift.",
            "",
            "Anchor-family summaries depend on OBS-065 anchor detection when available. If OBS-065 anchors are absent, all relations are treated as non-anchor relations.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OBS-066 gateway/canonical-family decoy survival.")
    parser.add_argument("--trace-cache-dir", required=True)
    parser.add_argument("--obs065-dir", default="")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--corpus-label", default="")
    parser.add_argument("--min-matching-distance", type=float, default=1e-9)
    parser.add_argument("--exact-match-tolerance", type=float, default=1e-12)
    parser.add_argument("--rank-bands", default="1-10,51-250,all")
    parser.add_argument("--n-iter", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--allow-decoy-reuse", action="store_true")
    parser.add_argument("--branch-decoy-same-family", action="store_true")
    parser.add_argument("--max-candidates-per-selected", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--write-replacement-signatures", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        trace_cache_dir=args.trace_cache_dir,
        obs065_dir=args.obs065_dir,
        outdir=args.outdir,
        corpus_label=args.corpus_label,
        min_matching_distance=float(args.min_matching_distance),
        exact_match_tolerance=float(args.exact_match_tolerance),
        rank_bands=args.rank_bands,
        n_iter=max(int(args.n_iter), 0),
        random_seed=int(args.random_seed),
        allow_decoy_reuse=bool(args.allow_decoy_reuse),
        branch_decoy_same_family=bool(args.branch_decoy_same_family),
        max_candidates_per_selected=max(int(args.max_candidates_per_selected), 25),
        n_workers=max(int(args.n_workers), 1),
        write_replacement_signatures=bool(args.write_replacement_signatures),
    )

    cache_dir = Path(cfg.trace_cache_dir)
    obs065_dir = Path(cfg.obs065_dir) if cfg.obs065_dir else Path("")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading OBS-064 all-path profile...")
    profile = read_csv_numeric(
        cache_dir / "obs064_all_path_profile.csv",
        text_cols={"corpus", "path_id", "route_class", "path_family", "start_node_id", "end_node_id"},
    )
    profile["path_id"] = profile["path_id"].astype(str)

    print("Building path gateway/canonical-family signature counts...")
    counts = build_path_gateway_signature_counts(cache_dir, obs065_dir)

    print("Loading or building replacement pair plans...")
    ranked, pairs, runs = load_or_build_pair_plans(profile, cfg)

    print("Building baseline gateway/canonical signature...")
    baseline_map = selected_path_class_map(profile)
    baseline_sig = aggregate_signature_for_map(
        counts,
        baseline_map,
        run_id="baseline",
        band="baseline",
        iteration=-1,
        replacement_route_class="baseline",
    )

    print(f"Running OBS-066 controls with n_workers={cfg.n_workers}...")
    drift, replacement_sig = run_controls(profile, counts, baseline_sig, pairs, runs, cfg)

    print("Summarizing survival...")
    summary = summarize_survival(runs, drift)
    gateway_summary, canonical_summary, anchor_summary = split_summary(summary)

    obs065_summary = load_obs065_proto_summary(obs065_dir) if cfg.obs065_dir else pd.DataFrame()
    consequence_modes = build_cross_layer_consequence_modes(summary, obs065_summary, cfg)

    ranked_to_write = ranked.copy()
    if len(ranked_to_write) and "selected_route_class" in ranked_to_write.columns:
        ranked_to_write = (
            ranked_to_write.sort_values(["selected_route_class", "selected_path_id", "eligible_decoy_rank"])
            .groupby(["selected_route_class", "selected_path_id"], as_index=False, group_keys=False)
            .head(cfg.max_candidates_per_selected)
        )

    path_counts_csv = outdir / "obs066_path_gateway_signature_counts.csv"
    canonical_counts_csv = outdir / "obs066_path_canonical_family_counts.csv"
    ranked_csv = outdir / "obs066_ranked_nonexact_decoy_candidates.csv"
    pairs_csv = outdir / "obs066_replacement_pairs.csv"
    runs_csv = outdir / "obs066_replacement_runs.csv"
    baseline_csv = outdir / "obs066_gateway_signature_baseline.csv"
    replacement_csv = outdir / "obs066_gateway_signature_replacement.csv"
    drift_csv = outdir / "obs066_gateway_layer_survival_drift.csv"
    summary_csv = outdir / "obs066_gateway_survival_summary.csv"
    canonical_summary_csv = outdir / "obs066_canonical_family_survival_summary.csv"
    anchor_summary_csv = outdir / "obs066_anchor_family_survival_summary.csv"
    modes_csv = outdir / "obs066_cross_layer_consequence_modes.csv"
    report_md = outdir / "obs066_gateway_canonical_family_survival_report.md"

    print("Writing outputs...")
    counts.to_csv(path_counts_csv, index=False)
    counts[counts["layer"].astype(str).str.contains("canonical|anchor", regex=True)].to_csv(canonical_counts_csv, index=False)
    ranked_to_write.to_csv(ranked_csv, index=False)
    pairs.to_csv(pairs_csv, index=False)
    runs.to_csv(runs_csv, index=False)
    baseline_sig.to_csv(baseline_csv, index=False)
    drift.to_csv(drift_csv, index=False)
    gateway_summary.to_csv(summary_csv, index=False)
    canonical_summary.to_csv(canonical_summary_csv, index=False)
    anchor_summary.to_csv(anchor_summary_csv, index=False)
    consequence_modes.to_csv(modes_csv, index=False)

    if cfg.write_replacement_signatures:
        replacement_sig.to_csv(replacement_csv, index=False)

    report_md.write_text(
        build_report(
            cfg,
            profile,
            counts,
            ranked_to_write,
            pairs,
            runs,
            baseline_sig,
            summary,
            gateway_summary,
            canonical_summary,
            anchor_summary,
            consequence_modes,
        ),
        encoding="utf-8",
    )

    print(path_counts_csv)
    print(canonical_counts_csv)
    print(ranked_csv)
    print(pairs_csv)
    print(runs_csv)
    print(baseline_csv)
    if cfg.write_replacement_signatures:
        print(replacement_csv)
    print(drift_csv)
    print(summary_csv)
    print(canonical_summary_csv)
    print(anchor_summary_csv)
    print(modes_csv)
    print(report_md)


if __name__ == "__main__":
    main()
