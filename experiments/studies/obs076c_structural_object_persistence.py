#!/usr/bin/env python3
"""
obs076c_structural_object_persistence.py

OBS-076c — Structural object persistence over observable-space geometry proxies

v2 patch
--------
v2 separates dynamic scale-space objects from static reference objects.

v1 consumed only OBS-076b node geometry. That table contains dynamic fields
such as energy, density_score, and seam_proxy, but many context columns
(signed_phase, lazarus_score, response_strength, frobenius_T, coupling fields)
may be inherited static node-context fields.

v2 optionally consumes the OBS-076a diffusion bundle and injects diffused
observable columns into the OBS-076b node table as:

    dyn__<observable_name>

When available, v2 defines structural objects from dynamic columns by default:

    phase_band_positive   -> dyn__signed_phase
    phase_band_negative   -> dyn__signed_phase
    lazarus_concentration -> dyn__lazarus_score
    response_ridge        -> dyn__response_strength
    frobenius_ridge       -> dyn__frobenius_T
    coupling_positive     -> dyn__signed_coupling
    coupling_negative     -> dyn__signed_coupling

If a dynamic column is unavailable, v2 falls back to the static/context column
and marks the object as source_kind=static_reference.

Purpose
-------
OBS-076c consumes OBS-076b node-level geometry-by-scale output and tracks
named structural object supports across diffusion scale.

It asks:

    When X(t) is diffused and observable-space geometry is rebuilt,
    which structural objects persist, migrate, overlap, or separate?

Scope discipline
----------------
This script does NOT claim canonical seam, gateway, attractor, or Fisher
geometry persistence.

It tracks structural-object proxies over OBS-076b observable-space geometry.

Inputs
------
Required:
    obs076b_node_geometry_by_scale.csv

Optional:
    obs076a_diffusion_bundle.npz

Expected OBS-076b columns:
    id, scale_index, t, geom_x, geom_y, energy, density_score,
    seam_proxy_score, is_seam_proxy, phase_contrast

Optional context columns:
    r, alpha, signed_phase, distance_to_seam, lazarus_score,
    response_strength, signed_coupling, cosine_alignment,
    trace_T, frobenius_T

Optional dynamic columns are injected from OBS-076a bundle:
    dyn__signed_phase
    dyn__lazarus_score
    dyn__response_strength
    dyn__frobenius_T
    dyn__signed_coupling
    etc.

Outputs
-------
outdir/
  obs076c_input_manifest.csv
  obs076c_object_manifest.csv
  obs076c_object_membership_by_scale.csv
  obs076c_object_persistence.csv
  obs076c_object_context_summary.csv
  obs076c_object_overlap_by_scale.csv
  obs076c_selected_object_overlap_by_scale.csv
  obs076c_object_centroid_drift.csv
  obs076c_report.md

Structural objects
------------------
Objects are defined per scale using fixed quantile thresholds.

Default quantile: 0.85

Supported proxy objects:
    energy_ridge
    density_core
    seam_proxy
    phase_band_positive
    phase_band_negative
    lazarus_concentration
    response_ridge
    frobenius_ridge
    coupling_positive
    coupling_negative

Interpretation
--------------
OBS-076c tests whether proxy objects co-move or separate across scale.

This is designed to diagnose whether Cp3 fine-scale response support
separates from coarse-scale phase/seam-band support.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
DYN_PREFIX = "dyn__"


@dataclass(frozen=True)
class Config:
    obs076b_node_geometry: Path
    outdir: Path
    obs076a_bundle: Path | None
    id_col: str
    quantile: float
    top_mode: str
    required_min_members: int
    object_source_mode: str


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    base_column: str
    direction: str
    description: str
    dynamic_eligible: bool


@dataclass(frozen=True)
class ResolvedObjectSpec:
    name: str
    source_column: str
    base_column: str
    direction: str
    description: str
    source_kind: str
    dynamic_eligible: bool


OBJECT_SPECS: list[ObjectSpec] = [
    ObjectSpec(
        name="energy_ridge",
        base_column="energy",
        direction="top",
        description="Top quantile by OBS-076b node energy ||Z_i(t)||.",
        dynamic_eligible=False,
    ),
    ObjectSpec(
        name="density_core",
        base_column="density_score",
        direction="top",
        description="Top quantile by density score; higher means denser observable-space neighborhood.",
        dynamic_eligible=False,
    ),
    ObjectSpec(
        name="seam_proxy",
        base_column="is_seam_proxy",
        direction="flag",
        description="OBS-076b phase-derived seam proxy flag.",
        dynamic_eligible=False,
    ),
    ObjectSpec(
        name="phase_band_positive",
        base_column="signed_phase",
        direction="top",
        description="Top quantile by signed phase.",
        dynamic_eligible=True,
    ),
    ObjectSpec(
        name="phase_band_negative",
        base_column="signed_phase",
        direction="bottom",
        description="Bottom quantile by signed phase.",
        dynamic_eligible=True,
    ),
    ObjectSpec(
        name="lazarus_concentration",
        base_column="lazarus_score",
        direction="top",
        description="Top quantile by Lazarus score.",
        dynamic_eligible=True,
    ),
    ObjectSpec(
        name="response_ridge",
        base_column="response_strength",
        direction="top",
        description="Top quantile by response strength.",
        dynamic_eligible=True,
    ),
    ObjectSpec(
        name="frobenius_ridge",
        base_column="frobenius_T",
        direction="top",
        description="Top quantile by response tensor Frobenius norm.",
        dynamic_eligible=True,
    ),
    ObjectSpec(
        name="coupling_positive",
        base_column="signed_coupling",
        direction="top",
        description="Top quantile by signed coupling.",
        dynamic_eligible=True,
    ),
    ObjectSpec(
        name="coupling_negative",
        base_column="signed_coupling",
        direction="bottom",
        description="Bottom quantile by signed coupling.",
        dynamic_eligible=True,
    ),
]


CONTEXT_COLUMNS = [
    "geom_x",
    "geom_y",
    "energy",
    "density_score",
    "mean_knn_observable_distance",
    "seam_proxy_score",
    "is_seam_proxy",
    "phase_contrast",
    "r",
    "alpha",
    "mds1",
    "mds2",
    "signed_phase",
    "distance_to_seam",
    "lazarus_score",
    "response_strength",
    "signed_coupling",
    "cosine_alignment",
    "trace_T",
    "frobenius_T",
    "dyn__signed_phase",
    "dyn__distance_to_seam",
    "dyn__lazarus_score",
    "dyn__response_strength",
    "dyn__signed_coupling",
    "dyn__cosine_alignment",
    "dyn__trace_T",
    "dyn__frobenius_T",
]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="OBS-076c structural object persistence over OBS-076b observable-space geometry proxies."
    )
    parser.add_argument("--obs076b-node-geometry", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--obs076a-bundle",
        default=None,
        type=Path,
        help="Optional OBS-076a diffusion bundle. When provided, diffused columns are injected as dyn__<observable>.",
    )
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--quantile", type=float, default=0.85)
    parser.add_argument(
        "--top-mode",
        choices=["quantile", "fixed_count"],
        default="quantile",
        help=(
            "quantile: use threshold at --quantile. "
            "fixed_count: use ceil((1-quantile) * n) nodes per object."
        ),
    )
    parser.add_argument(
        "--required-min-members",
        type=int,
        default=1,
        help="Objects with fewer members than this at any scale are still written, but flagged.",
    )
    parser.add_argument(
        "--object-source-mode",
        choices=["prefer_dynamic", "static_only", "dynamic_only"],
        default="prefer_dynamic",
        help=(
            "prefer_dynamic: use dyn__ columns when available, else static/context columns. "
            "static_only: ignore dyn__ columns. "
            "dynamic_only: skip dynamic-eligible objects unless dyn__ columns are available."
        ),
    )

    args = parser.parse_args()

    if not (0.0 < args.quantile < 1.0):
        raise ValueError("--quantile must be between 0 and 1")

    return Config(
        obs076b_node_geometry=args.obs076b_node_geometry,
        outdir=args.outdir,
        obs076a_bundle=args.obs076a_bundle,
        id_col=args.id_col,
        quantile=args.quantile,
        top_mode=args.top_mode,
        required_min_members=args.required_min_members,
        object_source_mode=args.object_source_mode,
    )


def as_str_array(x: np.ndarray) -> np.ndarray:
    return np.array([str(v) for v in x.tolist()], dtype=str)


def load_node_geometry(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [id_col, "scale_index", "t", "geom_x", "geom_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OBS-076b node geometry missing required columns: {missing}")

    df[id_col] = df[id_col].astype(str)
    df["scale_index"] = pd.to_numeric(df["scale_index"], errors="raise").astype(int)
    df["t"] = pd.to_numeric(df["t"], errors="raise").astype(float)

    dupes = df.duplicated([id_col, "scale_index"])
    if dupes.any():
        bad = df.loc[dupes, [id_col, "scale_index"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate id/scale rows found: {bad}")

    return df


def load_obs076a_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    required = ["ids", "observable_cols", "X_t", "t"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"OBS-076a bundle missing required keys: {missing}")

    ids = as_str_array(data["ids"])
    observable_cols = as_str_array(data["observable_cols"])
    X_t = np.asarray(data["X_t"], dtype=float)
    ts = np.asarray(data["t"], dtype=float)

    if X_t.ndim != 3:
        raise ValueError(f"X_t must be scales × nodes × observables; got {X_t.shape}")
    if X_t.shape[1] != len(ids):
        raise ValueError("X_t node dimension does not match ids")
    if X_t.shape[2] != len(observable_cols):
        raise ValueError("X_t observable dimension does not match observable_cols")
    if X_t.shape[0] != len(ts):
        raise ValueError("X_t scale dimension does not match t")

    return {
        "ids": ids,
        "observable_cols": observable_cols,
        "X_t": X_t,
        "t": ts,
    }


def inject_dynamic_columns_from_bundle(
    df: pd.DataFrame,
    bundle_path: Path | None,
    id_col: str,
) -> tuple[pd.DataFrame, list[str], str]:
    if bundle_path is None:
        return df, [], "not_provided"

    bundle = load_obs076a_bundle(bundle_path)
    ids = bundle["ids"]
    observable_cols = bundle["observable_cols"]
    X_t = bundle["X_t"]
    ts = bundle["t"]

    scale_values = (
        df[["scale_index", "t"]]
        .drop_duplicates()
        .sort_values(["scale_index", "t"])
        .reset_index(drop=True)
    )

    if len(scale_values) != len(ts):
        raise ValueError(
            f"OBS-076b scale count {len(scale_values)} does not match bundle scale count {len(ts)}"
        )

    dyn_rows = []
    for scale_index in range(len(ts)):
        block = pd.DataFrame({id_col: ids})
        block["scale_index"] = scale_index
        block["bundle_t"] = float(ts[scale_index])
        for j, col in enumerate(observable_cols):
            block[f"{DYN_PREFIX}{col}"] = X_t[scale_index, :, j]
        dyn_rows.append(block)

    dyn_df = pd.concat(dyn_rows, ignore_index=True)
    dyn_df[id_col] = dyn_df[id_col].astype(str)

    dyn_cols = [c for c in dyn_df.columns if c.startswith(DYN_PREFIX)]

    merged = df.merge(
        dyn_df,
        on=[id_col, "scale_index"],
        how="left",
        validate="one_to_one",
    )

    return merged, dyn_cols, "ok"


def resolve_object_specs(df: pd.DataFrame, cfg: Config) -> tuple[list[ResolvedObjectSpec], pd.DataFrame]:
    resolved: list[ResolvedObjectSpec] = []
    manifest_rows = []

    for spec in OBJECT_SPECS:
        dyn_col = f"{DYN_PREFIX}{spec.base_column}"
        static_col = spec.base_column

        chosen_col = None
        source_kind = None
        status = None

        if spec.dynamic_eligible and cfg.object_source_mode != "static_only":
            if dyn_col in df.columns:
                chosen_col = dyn_col
                source_kind = "dynamic"
                status = "available_dynamic"
            elif cfg.object_source_mode == "dynamic_only":
                status = "missing_dynamic_column"
            elif static_col in df.columns:
                chosen_col = static_col
                source_kind = "static_reference"
                status = "fallback_static_reference"
            else:
                status = "missing_column"
        else:
            if static_col in df.columns:
                chosen_col = static_col
                source_kind = "dynamic_proxy" if spec.name in {"energy_ridge", "density_core", "seam_proxy"} else "static_reference"
                status = "available"
            else:
                status = "missing_column"

        if chosen_col is not None:
            resolved.append(
                ResolvedObjectSpec(
                    name=spec.name,
                    source_column=chosen_col,
                    base_column=spec.base_column,
                    direction=spec.direction,
                    description=spec.description,
                    source_kind=str(source_kind),
                    dynamic_eligible=spec.dynamic_eligible,
                )
            )

        manifest_rows.append(
            {
                "object": spec.name,
                "base_column": spec.base_column,
                "resolved_source_column": chosen_col if chosen_col is not None else "",
                "direction": spec.direction,
                "source_kind": source_kind if source_kind is not None else "",
                "dynamic_eligible": int(spec.dynamic_eligible),
                "description": spec.description,
                "status": status,
            }
        )

    return resolved, pd.DataFrame(manifest_rows)


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def select_members_by_spec(
    scale_df: pd.DataFrame,
    spec: ResolvedObjectSpec,
    cfg: Config,
) -> tuple[set[str], float | None, str]:
    id_col = cfg.id_col

    if spec.source_column not in scale_df.columns:
        return set(), None, "missing_column"

    if spec.direction == "flag":
        vals = finite_numeric(scale_df[spec.source_column])
        members = set(scale_df.loc[vals.fillna(0) > 0, id_col].astype(str))
        return members, 0.5, "ok" if members else "empty"

    vals = finite_numeric(scale_df[spec.source_column])
    finite = vals[np.isfinite(vals)]

    if finite.empty:
        return set(), None, "no_finite_values"

    n = len(scale_df)

    if cfg.top_mode == "fixed_count":
        k = int(np.ceil((1.0 - cfg.quantile) * n))
        k = max(1, min(k, n))

        valid = scale_df.copy()
        valid["_value"] = vals

        if spec.direction == "top":
            chosen = valid.dropna(subset=["_value"]).sort_values("_value", ascending=False).head(k)
            threshold = float(chosen["_value"].min()) if len(chosen) else np.nan
        elif spec.direction == "bottom":
            chosen = valid.dropna(subset=["_value"]).sort_values("_value", ascending=True).head(k)
            threshold = float(chosen["_value"].max()) if len(chosen) else np.nan
        else:
            raise ValueError(f"Unsupported direction: {spec.direction}")

        members = set(chosen[id_col].astype(str))
        status = "ok" if members else "empty"
        return members, threshold, status

    if spec.direction == "top":
        threshold = float(np.nanquantile(finite, cfg.quantile))
        mask = vals >= threshold
    elif spec.direction == "bottom":
        threshold = float(np.nanquantile(finite, 1.0 - cfg.quantile))
        mask = vals <= threshold
    else:
        raise ValueError(f"Unsupported direction: {spec.direction}")

    members = set(scale_df.loc[mask.fillna(False), id_col].astype(str))
    status = "ok" if members else "empty"
    return members, threshold, status


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def centroid(scale_df: pd.DataFrame, ids: set[str], id_col: str) -> tuple[float, float]:
    sub = scale_df[scale_df[id_col].astype(str).isin(ids)]
    if sub.empty:
        return np.nan, np.nan

    x = pd.to_numeric(sub["geom_x"], errors="coerce")
    y = pd.to_numeric(sub["geom_y"], errors="coerce")
    return float(np.nanmean(x)), float(np.nanmean(y))


def euclidean_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    if not all(np.isfinite([ax, ay, bx, by])):
        return np.nan
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def context_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in CONTEXT_COLUMNS if c in df.columns]
    extra_dyn = [c for c in df.columns if c.startswith(DYN_PREFIX) and c not in cols]
    return cols + sorted(extra_dyn)


def summarize_context(
    scale_df: pd.DataFrame,
    member_ids: set[str],
    spec: ResolvedObjectSpec,
    cfg: Config,
) -> dict:
    id_col = cfg.id_col
    sub = scale_df[scale_df[id_col].astype(str).isin(member_ids)]

    row: dict = {
        "scale_index": int(scale_df["scale_index"].iloc[0]),
        "t": float(scale_df["t"].iloc[0]),
        "object": spec.name,
        "source_column": spec.source_column,
        "base_column": spec.base_column,
        "source_kind": spec.source_kind,
        "n_members": int(len(sub)),
    }

    for col in context_columns(scale_df):
        vals = pd.to_numeric(sub[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        row[f"{col}_mean"] = float(vals.mean())
        row[f"{col}_std"] = float(vals.std()) if vals.notna().sum() > 1 else 0.0
        row[f"{col}_min"] = float(vals.min())
        row[f"{col}_max"] = float(vals.max())

    return row


def build_membership_and_persistence(
    df: pd.DataFrame,
    cfg: Config,
    resolved_specs: list[ResolvedObjectSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_col = cfg.id_col

    scale_values = (
        df[["scale_index", "t"]]
        .drop_duplicates()
        .sort_values(["scale_index", "t"])
        .reset_index(drop=True)
    )

    membership_rows: list[dict] = []
    persistence_rows: list[dict] = []
    context_rows: list[dict] = []
    centroid_rows: list[dict] = []
    overlap_rows: list[dict] = []

    base_sets: dict[str, set[str]] = {}
    prev_sets: dict[str, set[str]] = {}
    base_centroids: dict[str, tuple[float, float]] = {}
    prev_centroids: dict[str, tuple[float, float]] = {}

    first_scale = int(scale_values["scale_index"].min())

    for scale_row in scale_values.itertuples(index=False):
        scale_index = int(scale_row.scale_index)
        t = float(scale_row.t)
        scale_df = df[df["scale_index"] == scale_index].copy()
        object_sets: dict[str, set[str]] = {}
        object_meta: dict[str, ResolvedObjectSpec] = {}

        for spec in resolved_specs:
            members, threshold, status = select_members_by_spec(scale_df, spec, cfg)
            object_sets[spec.name] = members
            object_meta[spec.name] = spec

            if len(members) < cfg.required_min_members:
                status = f"{status}:below_required_min_members"

            if scale_index == first_scale:
                base_sets[spec.name] = set(members)
                prev_sets[spec.name] = set(members)

            c = centroid(scale_df, members, id_col)

            if spec.name not in base_centroids:
                base_centroids[spec.name] = c
            if spec.name not in prev_centroids:
                prev_centroids[spec.name] = c

            persistence_rows.append(
                {
                    "scale_index": scale_index,
                    "t": t,
                    "object": spec.name,
                    "source_column": spec.source_column,
                    "base_column": spec.base_column,
                    "source_kind": spec.source_kind,
                    "direction": spec.direction,
                    "threshold": threshold,
                    "status": status,
                    "n_members": int(len(members)),
                    "jaccard_vs_base": float(jaccard(base_sets[spec.name], members)),
                    "jaccard_vs_previous": float(jaccard(prev_sets[spec.name], members)),
                    "retained_from_base": int(len(base_sets[spec.name] & members)),
                    "gained_vs_base": int(len(members - base_sets[spec.name])),
                    "lost_vs_base": int(len(base_sets[spec.name] - members)),
                }
            )

            centroid_rows.append(
                {
                    "scale_index": scale_index,
                    "t": t,
                    "object": spec.name,
                    "source_column": spec.source_column,
                    "source_kind": spec.source_kind,
                    "centroid_x": c[0],
                    "centroid_y": c[1],
                    "centroid_drift_vs_base": euclidean_dist(base_centroids[spec.name], c),
                    "centroid_drift_vs_previous": euclidean_dist(prev_centroids[spec.name], c),
                    "n_members": int(len(members)),
                }
            )

            context_rows.append(summarize_context(scale_df, members, spec, cfg))

            for node_id in sorted(members):
                membership_rows.append(
                    {
                        "scale_index": scale_index,
                        "t": t,
                        id_col: node_id,
                        "object": spec.name,
                        "source_column": spec.source_column,
                        "base_column": spec.base_column,
                        "source_kind": spec.source_kind,
                        "direction": spec.direction,
                        "threshold": threshold,
                    }
                )

            prev_sets[spec.name] = set(members)
            prev_centroids[spec.name] = c

        names = sorted(object_sets.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                A = object_sets[a]
                B = object_sets[b]
                inter = A & B
                union = A | B
                meta_a = object_meta[a]
                meta_b = object_meta[b]

                overlap_rows.append(
                    {
                        "scale_index": scale_index,
                        "t": t,
                        "object_a": a,
                        "object_b": b,
                        "source_column_a": meta_a.source_column,
                        "source_column_b": meta_b.source_column,
                        "source_kind_a": meta_a.source_kind,
                        "source_kind_b": meta_b.source_kind,
                        "n_a": int(len(A)),
                        "n_b": int(len(B)),
                        "n_intersection": int(len(inter)),
                        "n_union": int(len(union)),
                        "jaccard": float(jaccard(A, B)),
                        "overlap_a_share": float(len(inter) / len(A)) if A else np.nan,
                        "overlap_b_share": float(len(inter) / len(B)) if B else np.nan,
                    }
                )

    membership_df = pd.DataFrame(membership_rows)
    persistence_df = pd.DataFrame(persistence_rows)
    context_df = pd.DataFrame(context_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    centroid_df = pd.DataFrame(centroid_rows)

    return membership_df, persistence_df, context_df, overlap_df, centroid_df


def compact_final_persistence_table(persistence_df: pd.DataFrame) -> pd.DataFrame:
    if persistence_df.empty:
        return persistence_df

    final_scale = int(persistence_df["scale_index"].max())
    cols = [
        "object",
        "source_kind",
        "source_column",
        "direction",
        "n_members",
        "jaccard_vs_base",
        "jaccard_vs_previous",
        "retained_from_base",
        "gained_vs_base",
        "lost_vs_base",
    ]
    return (
        persistence_df[persistence_df["scale_index"] == final_scale][cols]
        .sort_values("object")
        .reset_index(drop=True)
    )


def selected_overlap_table(overlap_df: pd.DataFrame) -> pd.DataFrame:
    if overlap_df.empty:
        return overlap_df

    pairs = {
        tuple(sorted(("energy_ridge", "response_ridge"))),
        tuple(sorted(("energy_ridge", "frobenius_ridge"))),
        tuple(sorted(("energy_ridge", "seam_proxy"))),
        tuple(sorted(("energy_ridge", "phase_band_positive"))),
        tuple(sorted(("energy_ridge", "phase_band_negative"))),
        tuple(sorted(("response_ridge", "seam_proxy"))),
        tuple(sorted(("response_ridge", "lazarus_concentration"))),
        tuple(sorted(("frobenius_ridge", "seam_proxy"))),
        tuple(sorted(("lazarus_concentration", "seam_proxy"))),
        tuple(sorted(("phase_band_positive", "seam_proxy"))),
        tuple(sorted(("phase_band_negative", "seam_proxy"))),
    }

    mask = overlap_df.apply(
        lambda r: tuple(sorted((str(r["object_a"]), str(r["object_b"])))) in pairs,
        axis=1,
    )
    return overlap_df[mask].sort_values(["scale_index", "object_a", "object_b"]).reset_index(drop=True)


def write_input_manifest(
    cfg: Config,
    df: pd.DataFrame,
    object_manifest_df: pd.DataFrame,
    dyn_cols: list[str],
    dyn_status: str,
) -> None:
    rows = [
        {
            "artifact": "obs076b_node_geometry",
            "path": str(cfg.obs076b_node_geometry),
            "role": "input_node_geometry_by_scale",
            "status": "ok",
        },
        {
            "artifact": "obs076a_bundle",
            "path": str(cfg.obs076a_bundle) if cfg.obs076a_bundle else "",
            "role": "optional_dynamic_observable_source",
            "status": dyn_status,
        },
        {
            "artifact": "dynamic_columns_injected",
            "path": "",
            "role": ",".join(dyn_cols),
            "status": "observed" if dyn_cols else "none",
        },
        {
            "artifact": "n_rows",
            "path": "",
            "role": str(len(df)),
            "status": "observed",
        },
        {
            "artifact": "n_nodes",
            "path": "",
            "role": str(df[cfg.id_col].nunique()),
            "status": "observed",
        },
        {
            "artifact": "n_scales",
            "path": "",
            "role": str(df["scale_index"].nunique()),
            "status": "observed",
        },
        {
            "artifact": "quantile",
            "path": "",
            "role": str(cfg.quantile),
            "status": "configured",
        },
        {
            "artifact": "top_mode",
            "path": "",
            "role": cfg.top_mode,
            "status": "configured",
        },
        {
            "artifact": "object_source_mode",
            "path": "",
            "role": cfg.object_source_mode,
            "status": "configured",
        },
    ]

    for row in object_manifest_df.itertuples(index=False):
        rows.append(
            {
                "artifact": f"object:{row.object}",
                "path": "",
                "role": f"{row.resolved_source_column}:{row.direction}:{row.source_kind}",
                "status": row.status,
            }
        )

    pd.DataFrame(rows).to_csv(cfg.outdir / "obs076c_input_manifest.csv", index=False)


def write_report(
    cfg: Config,
    df: pd.DataFrame,
    object_manifest_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    centroid_df: pd.DataFrame,
    dyn_cols: list[str],
    dyn_status: str,
) -> None:
    final_persist = compact_final_persistence_table(persistence_df)
    selected_overlap = selected_overlap_table(overlap_df)

    n_nodes = df[cfg.id_col].nunique()
    n_rows = len(df)
    n_scales = df["scale_index"].nunique()
    scale_min = df["t"].min()
    scale_max = df["t"].max()

    lines: list[str] = [
        "# OBS-076c — Structural object persistence",
        "",
        "## Scope",
        "",
        "OBS-076c tracks named structural-object proxies over OBS-076b observable-space geometry.",
        "",
        "This is not canonical seam, gateway, attractor, or Fisher-geometry persistence.",
        "",
        "## v2 source-kind patch",
        "",
        "v2 distinguishes dynamic scale-space objects from static reference objects.",
        "",
        "When `--obs076a-bundle` is provided, diffused observable columns are injected as `dyn__<observable>`.",
        "",
        "Dynamic-eligible objects use those injected columns when available under `object_source_mode=prefer_dynamic`.",
        "",
        "## Inputs",
        "",
        f"- obs076b_node_geometry: `{cfg.obs076b_node_geometry}`",
        f"- obs076a_bundle: `{cfg.obs076a_bundle if cfg.obs076a_bundle else ''}`",
        f"- obs076a_bundle_status: `{dyn_status}`",
        f"- dynamic columns injected: `{len(dyn_cols)}`",
        f"- rows: `{n_rows}`",
        f"- nodes: `{n_nodes}`",
        f"- scales: `{n_scales}`",
        f"- t range: `{scale_min}` → `{scale_max}`",
        "",
        "## Configuration",
        "",
        f"- quantile: `{cfg.quantile}`",
        f"- top_mode: `{cfg.top_mode}`",
        f"- required_min_members: `{cfg.required_min_members}`",
        f"- object_source_mode: `{cfg.object_source_mode}`",
        "",
        "## Object availability and source kind",
        "",
        "| object | base_column | resolved_source_column | direction | source_kind | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in object_manifest_df.sort_values("object").itertuples(index=False):
        lines.append(
            "| "
            f"{row.object} | "
            f"{row.base_column} | "
            f"{row.resolved_source_column} | "
            f"{row.direction} | "
            f"{row.source_kind} | "
            f"{row.status} |"
        )

    lines.extend(["", "## Final-scale object persistence", ""])

    if final_persist.empty:
        lines.append("No persistence rows were produced.")
    else:
        lines.extend(
            [
                "| object | source_kind | source_column | n_members | jaccard_vs_base | jaccard_vs_previous | retained_from_base | gained_vs_base | lost_vs_base |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in final_persist.itertuples(index=False):
            lines.append(
                "| "
                f"{row.object} | "
                f"{row.source_kind} | "
                f"{row.source_column} | "
                f"{int(row.n_members)} | "
                f"{float(row.jaccard_vs_base):.6g} | "
                f"{float(row.jaccard_vs_previous):.6g} | "
                f"{int(row.retained_from_base)} | "
                f"{int(row.gained_vs_base)} | "
                f"{int(row.lost_vs_base)} |"
            )

    lines.extend(["", "## Selected object overlaps by scale", ""])

    if selected_overlap.empty:
        lines.append("No selected overlap rows were produced.")
    else:
        lines.extend(
            [
                "| scale_index | t | object_a | object_b | source_kind_a | source_kind_b | intersection | jaccard | overlap_a_share | overlap_b_share |",
                "| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in selected_overlap.itertuples(index=False):
            lines.append(
                "| "
                f"{int(row.scale_index)} | "
                f"{float(row.t):.6g} | "
                f"{row.object_a} | "
                f"{row.object_b} | "
                f"{row.source_kind_a} | "
                f"{row.source_kind_b} | "
                f"{int(row.n_intersection)} | "
                f"{float(row.jaccard):.6g} | "
                f"{float(row.overlap_a_share):.6g} | "
                f"{float(row.overlap_b_share):.6g} |"
            )

    lines.extend(["", "## Largest final centroid drifts", ""])

    if centroid_df.empty:
        lines.append("No centroid drift rows were produced.")
    else:
        final_scale = int(centroid_df["scale_index"].max())
        final_centroid = (
            centroid_df[centroid_df["scale_index"] == final_scale]
            .sort_values("centroid_drift_vs_base", ascending=False)
            .head(10)
        )
        lines.extend(
            [
                "| object | source_kind | n_members | centroid_drift_vs_base | centroid_drift_vs_previous |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in final_centroid.itertuples(index=False):
            lines.append(
                "| "
                f"{row.object} | "
                f"{row.source_kind} | "
                f"{int(row.n_members)} | "
                f"{float(row.centroid_drift_vs_base):.6g} | "
                f"{float(row.centroid_drift_vs_previous):.6g} |"
            )

    lines.extend(
        [
            "",
            "## Output artifacts",
            "",
            "- `obs076c_input_manifest.csv`",
            "- `obs076c_object_manifest.csv`",
            "- `obs076c_object_membership_by_scale.csv`",
            "- `obs076c_object_persistence.csv`",
            "- `obs076c_object_context_summary.csv`",
            "- `obs076c_object_overlap_by_scale.csv`",
            "- `obs076c_selected_object_overlap_by_scale.csv`",
            "- `obs076c_object_centroid_drift.csv`",
            "",
            "## Interpretation guardrails",
            "",
            "- Objects are proxy supports defined over OBS-076b observable-space geometry.",
            "- Dynamic source objects use diffused OBS-076a columns injected as `dyn__<observable>`.",
            "- Static-reference objects use node-context columns inherited from OBS-076b.",
            "- Seam proxy is inherited from OBS-076b and is not a canonical seam.",
            "- Overlap persistence indicates co-location of proxy supports, not causal dependence.",
            "- This checkpoint prepares structural targets for later scale-space transfer tests.",
            "",
        ]
    )

    (cfg.outdir / "obs076c_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    df = load_node_geometry(cfg.obs076b_node_geometry, cfg.id_col)
    df, dyn_cols, dyn_status = inject_dynamic_columns_from_bundle(
        df=df,
        bundle_path=cfg.obs076a_bundle,
        id_col=cfg.id_col,
    )

    resolved_specs, object_manifest_df = resolve_object_specs(df, cfg)

    (
        membership_df,
        persistence_df,
        context_df,
        overlap_df,
        centroid_df,
    ) = build_membership_and_persistence(df, cfg, resolved_specs)

    selected_overlap_df = selected_overlap_table(overlap_df)

    write_input_manifest(
        cfg=cfg,
        df=df,
        object_manifest_df=object_manifest_df,
        dyn_cols=dyn_cols,
        dyn_status=dyn_status,
    )

    object_manifest_df.to_csv(cfg.outdir / "obs076c_object_manifest.csv", index=False)
    membership_df.to_csv(cfg.outdir / "obs076c_object_membership_by_scale.csv", index=False)
    persistence_df.to_csv(cfg.outdir / "obs076c_object_persistence.csv", index=False)
    context_df.to_csv(cfg.outdir / "obs076c_object_context_summary.csv", index=False)
    overlap_df.to_csv(cfg.outdir / "obs076c_object_overlap_by_scale.csv", index=False)
    selected_overlap_df.to_csv(
        cfg.outdir / "obs076c_selected_object_overlap_by_scale.csv",
        index=False,
    )
    centroid_df.to_csv(cfg.outdir / "obs076c_object_centroid_drift.csv", index=False)

    write_report(
        cfg=cfg,
        df=df,
        object_manifest_df=object_manifest_df,
        persistence_df=persistence_df,
        overlap_df=overlap_df,
        centroid_df=centroid_df,
        dyn_cols=dyn_cols,
        dyn_status=dyn_status,
    )

    print("OBS-076c complete")
    print("wrote:", cfg.outdir / "obs076c_input_manifest.csv")
    print("wrote:", cfg.outdir / "obs076c_object_manifest.csv")
    print("wrote:", cfg.outdir / "obs076c_object_membership_by_scale.csv")
    print("wrote:", cfg.outdir / "obs076c_object_persistence.csv")
    print("wrote:", cfg.outdir / "obs076c_object_context_summary.csv")
    print("wrote:", cfg.outdir / "obs076c_object_overlap_by_scale.csv")
    print("wrote:", cfg.outdir / "obs076c_selected_object_overlap_by_scale.csv")
    print("wrote:", cfg.outdir / "obs076c_object_centroid_drift.csv")
    print("wrote:", cfg.outdir / "obs076c_report.md")


if __name__ == "__main__":
    main()
