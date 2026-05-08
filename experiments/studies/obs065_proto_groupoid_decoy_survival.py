#!/usr/bin/env python3
"""
OBS-065 — Proto-groupoid decoy survival controls.

Purpose
-------
Test whether proto-groupoid signatures survive route-origin decoy replacement
using the OBS-064 cached symbolic path traces.

OBS-064 created a reusable per-path symbolic substrate:

    path_id
      -> generator assignments
      -> generator compositions
      -> proto edges
      -> proto relations

OBS-065 consumes that cache and evaluates rank-banded non-exact decoy
replacement controls without rebuilding symbolic traces from raw routes.

Core question
-------------
Do proto-groupoid signatures survive route-origin decoy replacement, and at
which structural resolution do they fail?

Layers
------
- generator_completed
- composition
- proto_edge
- proto_sector_edge
- proto_relation
- proto_sector_relation

Interpretive failure modes
--------------------------
- vocabulary_survives_typed_action_fails:
    generator survives but proto_edge fails

- typed_edge_survives_relation_fails:
    proto_edge survives but proto_relation fails

- sector_survives_state_fails:
    proto_sector_edge survives but proto_edge fails, or
    proto_sector_relation survives but proto_relation fails

- broad_survival:
    relevant fine/coarse layers all survive

Outputs
-------
outputs/obs065_proto_groupoid_decoy_survival/
  obs065_ranked_nonexact_decoy_candidates.csv
  obs065_replacement_pairs.csv
  obs065_replacement_runs.csv
  obs065_proto_signature_baseline.csv
  obs065_proto_signature_replacement.csv
  obs065_proto_layer_survival_drift.csv
  obs065_proto_survival_summary.csv
  obs065_cross_layer_failure_modes.csv
  obs065_proto_anchor_candidates.csv
  obs065_proto_groupoid_decoy_survival_report.md
"""

from __future__ import annotations

import argparse
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

LAYER_SPECS = [
    {
        "layer": "generator_completed",
        "source": "generator_assignments",
        "value_col": "generator_completed",
        "count_col": "n_generators",
    },
    {
        "layer": "composition",
        "source": "generator_compositions",
        "value_col": "composition",
        "count_col": "n_compositions",
    },
    {
        "layer": "proto_edge",
        "source": "proto_edges",
        "value_col": "proto_edge",
        "count_col": "n_proto_edges",
    },
    {
        "layer": "proto_sector_edge",
        "source": "proto_edges",
        "value_col": "proto_sector_edge",
        "count_col": "n_proto_sector_edges",
    },
    {
        "layer": "proto_relation",
        "source": "proto_relations",
        "value_col": "proto_relation",
        "count_col": "n_proto_relations",
    },
    {
        "layer": "proto_sector_relation",
        "source": "proto_relations",
        "value_col": "proto_sector_relation",
        "count_col": "n_proto_sector_relations",
    },
]


@dataclass(frozen=True)
class Config:
    trace_cache_dir: str = "outputs/obs064_proto_groupoid_symbolic_trace_cache"
    outdir: str = "outputs/obs065_proto_groupoid_decoy_survival"
    corpus_label: str = ""
    min_matching_distance: float = 1e-9
    exact_match_tolerance: float = 1e-12
    rank_bands: str = "1-10,51-250,all"
    n_iter: int = 250
    random_seed: int = 42
    allow_decoy_reuse: bool = False
    branch_decoy_same_family: bool = False
    max_candidates_per_selected: int = 1000
    anchor_top_k: int = 10
    strong_anchor_survival: float = 0.80
    weak_anchor_survival: float = 0.60
    anchor_component_survival_threshold: float = 0.75


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

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


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
        candidates["selected_path_id"] = sel["path_id"]
        candidates["selected_path_family"] = sel.get("path_family", "")
        candidates["selected_is_branch_away"] = int(sel.get("is_branch_away", 0))
        candidates["selected_is_representative"] = int(sel.get("is_representative", 0))

        rows.append(candidates)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


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


def get_layer_source(layer_spec: dict[str, str], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return tables[layer_spec["source"]]


def distribution_for_map(
    source_df: pd.DataFrame,
    path_class_map: dict[str, str],
    value_col: str,
    count_col: str,
    *,
    run_id: str,
    band: str,
    iteration: int,
    replacement_route_class: str,
) -> pd.DataFrame:
    if len(source_df) == 0 or not path_class_map:
        return pd.DataFrame()

    pids = set(path_class_map.keys())
    work = source_df[source_df["path_id"].astype(str).isin(pids)].copy()
    if len(work) == 0:
        return pd.DataFrame()

    work["analysis_route_class"] = work["path_id"].astype(str).map(path_class_map)
    work = work[work["analysis_route_class"].isin(CLASS_ORDER)].copy()

    rows = []

    for route_class, grp in work.groupby("analysis_route_class", sort=False):
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
                    "run_id": run_id,
                    "band": band,
                    "iteration": iteration,
                    "replacement_route_class": replacement_route_class,
                    "route_class": route_class,
                    value_col: row[value_col],
                    count_col: int(row["count"]),
                    "n_paths": int(row["n_paths"]),
                    "share": float(row["count"] / total) if total else 0.0,
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def build_all_layer_signatures_for_map(
    tables: dict[str, pd.DataFrame],
    path_class_map: dict[str, str],
    *,
    run_id: str,
    band: str,
    iteration: int,
    replacement_route_class: str,
) -> pd.DataFrame:
    frames = []

    for spec in LAYER_SPECS:
        src = get_layer_source(spec, tables)
        sig = distribution_for_map(
            src,
            path_class_map,
            spec["value_col"],
            spec["count_col"],
            run_id=run_id,
            band=band,
            iteration=iteration,
            replacement_route_class=replacement_route_class,
        )

        if len(sig):
            sig = sig.rename(columns={spec["value_col"]: "signature_value", spec["count_col"]: "n_items"})
            sig["layer"] = spec["layer"]
            frames.append(sig)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def dist_map(sig: pd.DataFrame, run_id: str, route_class: str, layer: str) -> dict[str, float]:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["layer"] == layer)
    ]
    return {str(row["signature_value"]): float(row["share"]) for _, row in sub.iterrows()}


def top_values(sig: pd.DataFrame, run_id: str, route_class: str, layer: str, k: int = 3) -> list[str]:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["layer"] == layer)
    ].sort_values("share", ascending=False)

    return [str(x) for x in sub["signature_value"].head(k).tolist()]


def n_items(sig: pd.DataFrame, run_id: str, route_class: str, layer: str) -> int:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["layer"] == layer)
    ]

    if len(sub) == 0:
        return 0

    return int(pd.to_numeric(sub["n_items"], errors="coerce").sum())


def top_margin(sig: pd.DataFrame, run_id: str, route_class: str, layer: str) -> dict[str, object]:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["layer"] == layer)
    ].sort_values("share", ascending=False).reset_index(drop=True)

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


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def top_overlap(a: list[str], b: list[str], k: int = 3) -> int:
    return len(set(a[:k]) & set(b[:k]))


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
        for spec in LAYER_SPECS:
            layer = spec["layer"]

            p = dist_map(baseline_sig, "baseline", cls, layer)
            q = dist_map(replacement_sig, run_id, cls, layer)

            base_top3 = top_values(baseline_sig, "baseline", cls, layer, k=3)
            repl_top3 = top_values(replacement_sig, run_id, cls, layer, k=3)

            base_top1 = base_top3[0] if base_top3 else ""
            repl_top1 = repl_top3[0] if repl_top3 else ""
            margin = top_margin(baseline_sig, "baseline", cls, layer)

            rows.append(
                {
                    "run_id": run_id,
                    "band": band,
                    "iteration": iteration,
                    "replacement_route_class": replacement_route_class,
                    "evaluated_route_class": cls,
                    "is_replaced_class": int(cls == replacement_route_class),
                    "layer": layer,
                    "baseline_n_items": n_items(baseline_sig, "baseline", cls, layer),
                    "replacement_n_items": n_items(replacement_sig, run_id, cls, layer),
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


def run_controls(
    profile: pd.DataFrame,
    ranked: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    baseline_sig: pd.DataFrame,
    cfg: Config,
):
    rng = np.random.default_rng(cfg.random_seed)
    bands = parse_rank_bands(cfg.rank_bands)

    run_rows = []
    pair_rows = []
    sig_rows = []
    drift_rows = []

    for band_cfg in bands:
        band_name = str(band_cfg["band"])
        rank_min = int(band_cfg["rank_min"])
        rank_max = band_cfg["rank_max"]
        rank_max_int = int(rank_max) if rank_max is not None else None

        for iteration in range(cfg.n_iter):
            pairs = select_band_pairs(
                ranked,
                profile,
                band_name=band_name,
                rank_min=rank_min,
                rank_max=rank_max_int,
                rng=rng,
                allow_decoy_reuse=cfg.allow_decoy_reuse,
            )
            pairs["iteration"] = iteration
            pair_rows.append(pairs)

            for cls in CLASS_ORDER:
                run_id = f"{band_name}_iter_{iteration:04d}_replace_{cls}"
                repl_map = replacement_path_class_map(profile, pairs, cls)

                repl_sig = build_all_layer_signatures_for_map(
                    tables,
                    repl_map,
                    run_id=run_id,
                    band=band_name,
                    iteration=iteration,
                    replacement_route_class=cls,
                )

                sig_rows.append(repl_sig)

                matched = pairs[
                    (pairs["route_class"] == cls)
                    & (pairs["match_status"] == "matched_band_nonexact")
                ].copy()

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
                    build_layer_drift(
                        baseline_sig,
                        repl_sig,
                        run_id=run_id,
                        band=band_name,
                        iteration=iteration,
                        replacement_route_class=cls,
                    )
                )

    runs = pd.DataFrame(run_rows)
    pairs_all = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    repl_sig_all = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()
    drift = pd.concat(drift_rows, ignore_index=True) if drift_rows else pd.DataFrame()

    return runs, pairs_all, repl_sig_all, drift


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

    if len(out):
        band_order = {str(b["band"]): i for i, b in enumerate(parse_rank_bands("1-10,51-250,all"))}
        class_order = {c: i for i, c in enumerate(CLASS_ORDER)}
        layer_order = {spec["layer"]: i for i, spec in enumerate(LAYER_SPECS)}
        out["band_order"] = out["band"].map(lambda x: band_order.get(str(x), 999))
        out["class_order"] = out["replacement_route_class"].map(lambda x: class_order.get(str(x), 999))
        out["layer_order"] = out["layer"].map(lambda x: layer_order.get(str(x), 999))
        out = (
            out.sort_values(["band_order", "class_order", "layer_order"])
            .drop(columns=["band_order", "class_order", "layer_order"])
            .reset_index(drop=True)
        )

    return out


def layer_survival_lookup(summary: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    out = {}
    for _, row in summary.iterrows():
        out[(str(row["band"]), str(row["replacement_route_class"]), str(row["layer"]))] = float(row["top1_survival_rate"])
    return out


def build_cross_layer_failure_modes(summary: pd.DataFrame) -> pd.DataFrame:
    if len(summary) == 0:
        return pd.DataFrame()

    lookup = layer_survival_lookup(summary)
    rows = []

    for band in summary["band"].astype(str).unique():
        for cls in CLASS_ORDER:
            gen = lookup.get((band, cls, "generator_completed"), np.nan)
            comp = lookup.get((band, cls, "composition"), np.nan)
            edge = lookup.get((band, cls, "proto_edge"), np.nan)
            sector_edge = lookup.get((band, cls, "proto_sector_edge"), np.nan)
            relation = lookup.get((band, cls, "proto_relation"), np.nan)
            sector_relation = lookup.get((band, cls, "proto_sector_relation"), np.nan)

            if np.isfinite(gen) and np.isfinite(edge) and gen >= 0.80 and edge < 0.50:
                mode = "vocabulary_survives_typed_action_fails"
            elif np.isfinite(edge) and np.isfinite(relation) and edge >= 0.80 and relation < 0.50:
                mode = "typed_edge_survives_relation_fails"
            elif np.isfinite(sector_edge) and np.isfinite(edge) and sector_edge >= 0.80 and edge < 0.50:
                mode = "sector_edge_survives_state_edge_fails"
            elif np.isfinite(sector_relation) and np.isfinite(relation) and sector_relation >= 0.80 and relation < 0.50:
                mode = "sector_relation_survives_state_relation_fails"
            elif all(np.isfinite(x) and x >= 0.80 for x in [gen, edge, sector_edge]):
                mode = "broad_local_action_survival"
            else:
                mode = "mixed_or_gradual_degradation"

            rows.append(
                {
                    "band": band,
                    "replacement_route_class": cls,
                    "generator_completed_top1_survival": gen,
                    "composition_top1_survival": comp,
                    "proto_edge_top1_survival": edge,
                    "proto_sector_edge_top1_survival": sector_edge,
                    "proto_relation_top1_survival": relation,
                    "proto_sector_relation_top1_survival": sector_relation,
                    "failure_mode": mode,
                }
            )

    return pd.DataFrame(rows)


def signature_share(sig: pd.DataFrame, run_id: str, route_class: str, layer: str, signature_value: str) -> float:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["layer"] == layer)
        & (sig["signature_value"].astype(str) == str(signature_value))
    ]
    if len(sub) == 0:
        return 0.0
    return float(pd.to_numeric(sub["share"], errors="coerce").sum())


def signature_rank(sig: pd.DataFrame, run_id: str, route_class: str, layer: str, signature_value: str) -> int:
    sub = sig[
        (sig["run_id"] == run_id)
        & (sig["route_class"] == route_class)
        & (sig["layer"] == layer)
    ].sort_values("share", ascending=False).reset_index(drop=True)

    hits = sub.index[sub["signature_value"].astype(str) == str(signature_value)].tolist()
    if not hits:
        return 9999
    return int(hits[0] + 1)


def parse_proto_relation_components(proto_relation: str) -> list[str]:
    if " ; " in str(proto_relation):
        return [x.strip() for x in str(proto_relation).split(" ; ", 1)]
    return [str(proto_relation)]


def parse_proto_relation_generators(proto_relation: str) -> list[str]:
    gens = []
    for comp in parse_proto_relation_components(proto_relation):
        m = re.search(r"--(.+?)-->", comp)
        if m:
            gens.append(m.group(1).strip())
    return gens


def sectorize_proto_relation(proto_relation: str) -> str:
    def state_to_sector(state: str) -> str:
        state = state.strip()
        if state in {"R", "A", "L", "C"}:
            return "seam_sector"
        if state == "O":
            return "off_seam_sector"
        if state == "P":
            return "post_exit_sector"
        return "unknown_sector"

    out = []
    for comp in parse_proto_relation_components(proto_relation):
        m = re.match(r"(.+?) --(.+?)--> (.+)", comp)
        if not m:
            out.append(comp)
            continue
        src = state_to_sector(m.group(1))
        gen = m.group(2).strip()
        tgt = state_to_sector(m.group(3))
        out.append(f"{src} --{gen}--> {tgt}")
    return " ; ".join(out)


def build_proto_anchor_candidates(
    baseline_sig: pd.DataFrame,
    replacement_sig: pd.DataFrame,
    runs: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    relation_base = baseline_sig[
        (baseline_sig["run_id"] == "baseline")
        & (baseline_sig["layer"] == "proto_relation")
    ].copy()

    if len(relation_base) == 0 or len(replacement_sig) == 0:
        return pd.DataFrame()

    relation_base = (
        relation_base.sort_values(["route_class", "share"], ascending=[True, False])
        .groupby("route_class", as_index=False, group_keys=False)
        .head(cfg.anchor_top_k)
    )

    rows = []

    for _, base_row in relation_base.iterrows():
        route_class = str(base_row["route_class"])
        proto_relation = str(base_row["signature_value"])
        baseline_relation_share = float(base_row["share"])
        component_edges = parse_proto_relation_components(proto_relation)
        component_generators = parse_proto_relation_generators(proto_relation)
        sector_relation = sectorize_proto_relation(proto_relation)

        for (band, repl_cls), run_grp in runs.groupby(["band", "replacement_route_class"], dropna=False):
            run_ids = run_grp["run_id"].astype(str).tolist()

            relation_survived = []
            sector_relation_survived = []
            relation_shares = []
            sector_relation_shares = []

            component_edge_survival_values = []
            component_gen_survival_values = []
            component_edge_rank_survival_values = []
            component_gen_rank_survival_values = []

            for run_id in run_ids:
                rel_share = signature_share(replacement_sig, run_id, route_class, "proto_relation", proto_relation)
                relation_shares.append(rel_share)
                relation_survived.append(int(rel_share > 0))

                sec_share = signature_share(replacement_sig, run_id, route_class, "proto_sector_relation", sector_relation)
                sector_relation_shares.append(sec_share)
                sector_relation_survived.append(int(sec_share > 0))

                for edge in component_edges:
                    base_edge_rank = signature_rank(baseline_sig, "baseline", route_class, "proto_edge", edge)
                    repl_edge_rank = signature_rank(replacement_sig, run_id, route_class, "proto_edge", edge)
                    edge_share = signature_share(replacement_sig, run_id, route_class, "proto_edge", edge)

                    component_edge_survival_values.append(int(edge_share > 0))
                    component_edge_rank_survival_values.append(int(base_edge_rank == repl_edge_rank))

                for gen in component_generators:
                    base_gen_rank = signature_rank(baseline_sig, "baseline", route_class, "generator_completed", gen)
                    repl_gen_rank = signature_rank(replacement_sig, run_id, route_class, "generator_completed", gen)
                    gen_share = signature_share(replacement_sig, run_id, route_class, "generator_completed", gen)

                    component_gen_survival_values.append(int(gen_share > 0))
                    component_gen_rank_survival_values.append(int(base_gen_rank == repl_gen_rank))

            relation_survival_rate = float(np.mean(relation_survived)) if relation_survived else np.nan
            sector_relation_survival_rate = float(np.mean(sector_relation_survived)) if sector_relation_survived else np.nan
            mean_relation_share = float(np.mean(relation_shares)) if relation_shares else np.nan
            mean_abs_relation_share_drift = (
                float(np.mean([abs(x - baseline_relation_share) for x in relation_shares]))
                if relation_shares
                else np.nan
            )

            component_edge_survival_rate = (
                float(np.mean(component_edge_survival_values)) if component_edge_survival_values else np.nan
            )
            component_generator_survival_rate = (
                float(np.mean(component_gen_survival_values)) if component_gen_survival_values else np.nan
            )
            component_edge_rank_survival_rate = (
                float(np.mean(component_edge_rank_survival_values)) if component_edge_rank_survival_values else np.nan
            )
            component_generator_rank_survival_rate = (
                float(np.mean(component_gen_rank_survival_values)) if component_gen_rank_survival_values else np.nan
            )

            min_component_rank_survival = float(
                np.nanmin([component_edge_rank_survival_rate, component_generator_rank_survival_rate])
            )

            anchor_score = relation_survival_rate - min_component_rank_survival

            if (
                relation_survival_rate >= cfg.strong_anchor_survival
                and min_component_rank_survival < cfg.anchor_component_survival_threshold
            ):
                anchor_class = "strong_proto_anchor"
            elif (
                relation_survival_rate >= cfg.weak_anchor_survival
                and min_component_rank_survival < cfg.anchor_component_survival_threshold
            ):
                anchor_class = "weak_proto_anchor"
            elif (
                sector_relation_survival_rate >= cfg.strong_anchor_survival
                and relation_survival_rate < cfg.weak_anchor_survival
            ):
                anchor_class = "sector_anchor"
            else:
                anchor_class = "not_anchor"

            rows.append(
                {
                    "band": band,
                    "replacement_route_class": repl_cls,
                    "evaluated_route_class": route_class,
                    "baseline_proto_relation": proto_relation,
                    "baseline_proto_sector_relation": sector_relation,
                    "component_proto_edges": " | ".join(component_edges),
                    "component_generators": " | ".join(component_generators),
                    "baseline_relation_share": baseline_relation_share,
                    "relation_survival_rate": relation_survival_rate,
                    "sector_relation_survival_rate": sector_relation_survival_rate,
                    "mean_relation_share_replacement": mean_relation_share,
                    "mean_abs_relation_share_drift": mean_abs_relation_share_drift,
                    "component_edge_survival_rate": component_edge_survival_rate,
                    "component_generator_survival_rate": component_generator_survival_rate,
                    "component_edge_rank_survival_rate": component_edge_rank_survival_rate,
                    "component_generator_rank_survival_rate": component_generator_rank_survival_rate,
                    "min_component_rank_survival_rate": min_component_rank_survival,
                    "anchor_score": anchor_score,
                    "anchor_class": anchor_class,
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["anchor_class", "anchor_score", "relation_survival_rate"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    return out


def load_trace_cache(cache_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    profile = read_csv_numeric(
        cache_dir / "obs064_all_path_profile.csv",
        text_cols={"corpus", "path_id", "route_class", "path_family", "start_node_id", "end_node_id"},
    )

    assignments = read_csv_numeric(
        cache_dir / "obs064_path_generator_assignments.csv",
        text_cols={
            "path_id",
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
        },
    )

    compositions = read_csv_numeric(
        cache_dir / "obs064_path_generator_compositions.csv",
        text_cols={
            "path_id",
            "route_class",
            "path_family",
            "generator_1",
            "generator_2",
            "generator_family_1",
            "generator_family_2",
            "composition",
            "composition_family",
            "proto_source_1",
            "proto_target_1",
            "proto_source_2",
            "proto_target_2",
            "proto_relation",
            "proto_sector_relation",
        },
    )

    proto_edges = read_csv_numeric(
        cache_dir / "obs064_path_proto_edges.csv",
        text_cols={
            "path_id",
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
        },
    )

    proto_relations = read_csv_numeric(
        cache_dir / "obs064_path_proto_relations.csv",
        text_cols={
            "path_id",
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
        },
    )

    tables = {
        "generator_assignments": assignments,
        "generator_compositions": compositions,
        "proto_edges": proto_edges,
        "proto_relations": proto_relations,
    }

    return profile, tables


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
                "mean_steps_per_path": safe_mean(sub["n_steps"]) if len(sub) else np.nan,
                "mean_distance_to_seam": safe_mean(sub["mean_distance_to_seam"]) if len(sub) else np.nan,
                "mean_lazarus_score": safe_mean(sub["mean_lazarus_score"]) if len(sub) else np.nan,
                "mean_response_strength": safe_mean(sub["mean_response_strength"]) if len(sub) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_report(
    cfg: Config,
    profile: pd.DataFrame,
    ranked: pd.DataFrame,
    runs: pd.DataFrame,
    baseline_sig: pd.DataFrame,
    summary: pd.DataFrame,
    failure_modes: pd.DataFrame,
    anchors: pd.DataFrame,
) -> str:
    selected_summary = summarize_selected(profile)
    bands = parse_rank_bands(cfg.rank_bands)

    lines = [
        "# OBS-065 — Proto-groupoid decoy survival controls",
        "",
        f"Corpus label: `{cfg.corpus_label or 'unspecified'}`",
        "",
        "## Purpose",
        "",
        "OBS-065 tests whether proto-groupoid signatures survive route-origin decoy replacement using the OBS-064 cached symbolic trace substrate.",
        "",
        "Unlike OBS-062/063, this study does not rebuild symbolic traces inside each decoy run. It aggregates cached per-path fragments by selected/decoy path maps.",
        "",
        "## Tested layers",
        "",
    ]

    for spec in LAYER_SPECS:
        lines.append(f"- `{spec['layer']}`")
    lines.append("")

    lines.extend(
        [
            "## Decoy policy",
            "",
            "Matching uses standardized path-profile features from the OBS-064 all-path profile:",
            "",
        ]
    )

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
        n_ranked = int((ranked["selected_route_class"] == cls).sum()) if len(ranked) else 0
        lines.append(f"- {cls}: eligible_nonexact_ranked_candidates={n_ranked}")
    lines.append("")

    lines.extend(["## Baseline top signatures", ""])
    for layer in [spec["layer"] for spec in LAYER_SPECS]:
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

    lines.extend(["## Survival summary", ""])
    if len(summary):
        for _, row in summary.iterrows():
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

    lines.extend(["## Cross-layer failure modes", ""])
    if len(failure_modes):
        for _, row in failure_modes.iterrows():
            lines.append(
                f"- {row['band']} / replace {row['replacement_route_class']}: "
                f"{row['failure_mode']} "
                f"(generator={float(row['generator_completed_top1_survival']):.3f}, "
                f"proto_edge={float(row['proto_edge_top1_survival']):.3f}, "
                f"sector_edge={float(row['proto_sector_edge_top1_survival']):.3f}, "
                f"proto_relation={float(row['proto_relation_top1_survival']):.3f}, "
                f"sector_relation={float(row['proto_sector_relation_top1_survival']):.3f})"
            )
    lines.append("")

    lines.extend(["## Proto-anchor candidates", ""])
    if len(anchors):
        anchor_view = anchors[anchors["anchor_class"].isin(["strong_proto_anchor", "weak_proto_anchor", "sector_anchor"])].copy()
        if len(anchor_view) == 0:
            lines.append("- No proto-anchor candidates detected under configured thresholds.")
        else:
            anchor_view = anchor_view.sort_values(
                ["anchor_class", "anchor_score", "relation_survival_rate"],
                ascending=[True, False, False],
            ).head(30)

            for _, row in anchor_view.iterrows():
                lines.append(
                    f"- {row['anchor_class']}: {row['band']} / replace {row['replacement_route_class']} / "
                    f"eval {row['evaluated_route_class']} / {row['baseline_proto_relation']} — "
                    f"relation_survival={float(row['relation_survival_rate']):.3f}, "
                    f"sector_survival={float(row['sector_relation_survival_rate']):.3f}, "
                    f"anchor_score={float(row['anchor_score']):.3f}"
                )
    lines.append("")

    unmatched_total = int(runs["unmatched_pairs"].sum()) if len(runs) and "unmatched_pairs" in runs else 0

    lines.extend(
        [
            "## Diagnostic totals",
            "",
            f"- replacement_runs: {len(runs)}",
            f"- unmatched_pairs_total: {unmatched_total}",
            "",
            "## Interpretation guardrail",
            "",
            "OBS-065 tests proto-groupoid survival under route-origin decoy replacement.",
            "",
            "It does not revise OBS-030c/d/e generator rules and does not test gateway predictors or canonical family summaries.",
            "",
            "Layer interpretation:",
            "",
            "- `generator_completed` tests vocabulary survival.",
            "- `proto_edge` tests typed reduced-state action survival.",
            "- `proto_sector_edge` tests typed sector-action survival.",
            "- `proto_relation` tests fine compositional proto-algebra survival.",
            "- `proto_sector_relation` tests coarse sector-level compositional survival.",
            "",
            "A sector layer surviving while the corresponding reduced-state layer fails indicates coarse proto-algebra robustness with fine-state sensitivity.",
            "",
            "A proto-relation surviving despite unstable component ranks indicates a proto-groupoid algebraic anchor.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OBS-065 proto-groupoid decoy survival controls.")
    parser.add_argument("--trace-cache-dir", default=Config.trace_cache_dir)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--corpus-label", default=Config.corpus_label)
    parser.add_argument("--min-matching-distance", type=float, default=Config.min_matching_distance)
    parser.add_argument("--exact-match-tolerance", type=float, default=Config.exact_match_tolerance)
    parser.add_argument("--rank-bands", default=Config.rank_bands)
    parser.add_argument("--n-iter", type=int, default=Config.n_iter)
    parser.add_argument("--random-seed", type=int, default=Config.random_seed)
    parser.add_argument("--allow-decoy-reuse", action="store_true")
    parser.add_argument("--branch-decoy-same-family", action="store_true")
    parser.add_argument("--max-candidates-per-selected", type=int, default=Config.max_candidates_per_selected)
    parser.add_argument("--anchor-top-k", type=int, default=Config.anchor_top_k)
    args = parser.parse_args()

    cfg = Config(
        trace_cache_dir=args.trace_cache_dir,
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
        anchor_top_k=max(int(args.anchor_top_k), 1),
    )

    cache_dir = Path(cfg.trace_cache_dir)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    profile, tables = load_trace_cache(cache_dir)

    candidate_pool = build_decoy_candidate_pool(profile, cfg)
    ranked = build_ranked_nonexact_candidates(profile, candidate_pool, cfg)

    baseline_map = selected_path_class_map(profile)
    baseline_sig = build_all_layer_signatures_for_map(
        tables,
        baseline_map,
        run_id="baseline",
        band="baseline",
        iteration=-1,
        replacement_route_class="baseline",
    )

    runs, pairs, replacement_sig, drift = run_controls(
        profile,
        ranked,
        tables,
        baseline_sig,
        cfg,
    )

    summary = summarize_survival(runs, drift)
    failure_modes = build_cross_layer_failure_modes(summary)
    anchors = build_proto_anchor_candidates(
        baseline_sig,
        replacement_sig,
        runs,
        cfg,
    )

    ranked_to_write = ranked.copy()
    if len(ranked_to_write) and cfg.max_candidates_per_selected > 0:
        ranked_to_write = (
            ranked_to_write.sort_values(["selected_route_class", "selected_path_id", "eligible_decoy_rank"])
            .groupby(["selected_route_class", "selected_path_id"], as_index=False, group_keys=False)
            .head(cfg.max_candidates_per_selected)
        )

    ranked_csv = outdir / "obs065_ranked_nonexact_decoy_candidates.csv"
    pairs_csv = outdir / "obs065_replacement_pairs.csv"
    runs_csv = outdir / "obs065_replacement_runs.csv"
    baseline_sig_csv = outdir / "obs065_proto_signature_baseline.csv"
    replacement_sig_csv = outdir / "obs065_proto_signature_replacement.csv"
    drift_csv = outdir / "obs065_proto_layer_survival_drift.csv"
    summary_csv = outdir / "obs065_proto_survival_summary.csv"
    failure_modes_csv = outdir / "obs065_cross_layer_failure_modes.csv"
    anchors_csv = outdir / "obs065_proto_anchor_candidates.csv"
    report_md = outdir / "obs065_proto_groupoid_decoy_survival_report.md"

    ranked_to_write.to_csv(ranked_csv, index=False)
    pairs.to_csv(pairs_csv, index=False)
    runs.to_csv(runs_csv, index=False)
    baseline_sig.to_csv(baseline_sig_csv, index=False)
    replacement_sig.to_csv(replacement_sig_csv, index=False)
    drift.to_csv(drift_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    failure_modes.to_csv(failure_modes_csv, index=False)
    anchors.to_csv(anchors_csv, index=False)

    report_md.write_text(
        build_report(
            cfg,
            profile,
            ranked,
            runs,
            baseline_sig,
            summary,
            failure_modes,
            anchors,
        ),
        encoding="utf-8",
    )

    print(ranked_csv)
    print(pairs_csv)
    print(runs_csv)
    print(baseline_sig_csv)
    print(replacement_sig_csv)
    print(drift_csv)
    print(summary_csv)
    print(failure_modes_csv)
    print(anchors_csv)
    print(report_md)


if __name__ == "__main__":
    main()
