#!/usr/bin/env python3
"""
obs077a_pinch_point_geometry.py

OBS-077a — Pinch-point geometry and support-transition diagnostics

v2 patch
--------
v2 adds score-family ablations:

    support_score:
        jaccard loss
        centroid drift
        support size change

    overlap_score:
        max overlap change with another object

    shape_score:
        participation-ratio change
        anisotropy change
        log-volume change

    id_score:
        TwoNN intrinsic-dimension change
        local MLE / Levina-Bickel intrinsic-dimension change

    pinch_score_total:
        support_score + overlap_score + shape_score + id_score

v2 also adds:

    dominant_family:
        support
        overlap
        shape
        intrinsic_dimension
        mixed
        none

The goal is to distinguish:
    - support transition
    - overlap transition
    - shape transition
    - intrinsic-dimension transition

instead of treating all high pinch scores as one event type.

Purpose
-------
OBS-077a detects candidate scale-local reorganization events in OBS-076
scale-space objects.

It asks:

    Where do structural supports merge, split, migrate, or change local
    geometric character across diffusion scale?

This script consumes OBS-076b and OBS-076c artifacts and computes transition
diagnostics for named structural objects.

It does NOT yet perform path/text projection.
It does NOT claim topological singularities.
It detects pinch-point candidates.

Inputs
------
Required:
    --node-geometry obs076b_node_geometry_by_scale.csv
    --objects obs076c_object_membership_by_scale.csv

Optional:
    --bundle obs076a_diffusion_bundle.npz
        Injects dynamic OBS-076a fields as dyn__<observable>.

    --overlaps obs076c_object_overlap_by_scale.csv
        Adds object-overlap transition changes where available.

Outputs
-------
outdir/
    obs077a_input_manifest.csv
    obs077a_feature_manifest.csv
    obs077a_object_geometry_by_scale.csv
    obs077a_object_transition_events.csv
    obs077a_overlap_transition_events.csv
    obs077a_pinch_point_candidates.csv
    obs077a_report.md

Definitions
-----------
For each object O and scale t:

    support(O, t) = nodes belonging to object O at scale t

For each transition t_i -> t_{i+1}:

    jaccard_previous
    support_size_delta
    centroid_drift
    mean_field_delta
    participation_ratio_delta
    anisotropy_delta
    log_volume_delta
    local_mle_id_delta
    twonn_id_delta
    overlap_delta_max

A pinch candidate is a high-scoring transition under a composite robust
z-score defined over observed transitions.

Guardrails
----------
- "Pinch-point" means candidate scale-local support reorganization.
- It is not a formal singularity.
- It is not a topological defect.
- It is not a linguistic/path-label result.
- Topological language requires future PH/index/winding diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DYN_PREFIX = "dyn__"
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    case: str
    node_geometry: Path
    objects: Path
    bundle: Path | None
    overlaps: Path | None
    outdir: Path
    id_col: str
    k_neighbors: int
    id_k: int
    min_support: int
    top_n: int
    ridge_eps: float
    include_background: bool
    family_dominance_ratio: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="OBS-077a pinch-point geometry diagnostics."
    )

    parser.add_argument("--case", default="case")
    parser.add_argument("--node-geometry", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--bundle", default=None, type=Path)
    parser.add_argument("--overlaps", default=None, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--id-col", default="id")

    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=7,
        help="Neighborhood size for local covariance summaries.",
    )
    parser.add_argument(
        "--id-k",
        type=int,
        default=10,
        help="k for local MLE intrinsic dimension estimator.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=4,
        help="Minimum object support size to compute local geometry.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top pinch candidates to emphasize in report.",
    )
    parser.add_argument(
        "--ridge-eps",
        type=float,
        default=1e-6,
        help="Regularization for covariance log-volume proxy.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Also compute a background pseudo-object over all nodes per scale.",
    )
    parser.add_argument(
        "--family-dominance-ratio",
        type=float,
        default=1.15,
        help=(
            "Dominant family must exceed the second-largest family score by this "
            "ratio. Otherwise dominant_family is mixed."
        ),
    )

    args = parser.parse_args()

    if args.k_neighbors < 2:
        raise ValueError("--k-neighbors must be >= 2")
    if args.id_k < 2:
        raise ValueError("--id-k must be >= 2")
    if args.min_support < 2:
        raise ValueError("--min-support must be >= 2")
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    if args.family_dominance_ratio < 1.0:
        raise ValueError("--family-dominance-ratio must be >= 1.0")

    return Config(
        case=args.case,
        node_geometry=args.node_geometry,
        objects=args.objects,
        bundle=args.bundle,
        overlaps=args.overlaps,
        outdir=args.outdir,
        id_col=args.id_col,
        k_neighbors=args.k_neighbors,
        id_k=args.id_k,
        min_support=args.min_support,
        top_n=args.top_n,
        ridge_eps=args.ridge_eps,
        include_background=bool(args.include_background),
        family_dominance_ratio=float(args.family_dominance_ratio),
    )


def as_str_array(x: np.ndarray) -> np.ndarray:
    return np.array([str(v) for v in x.tolist()], dtype=str)


def safe_float(x) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def robust_z(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce").astype(float)
    med = vals.median(skipna=True)
    mad = (vals - med).abs().median(skipna=True)

    if pd.isna(med):
        return pd.Series(np.zeros(len(vals)), index=vals.index, dtype=float)

    if pd.isna(mad) or mad < EPS:
        std = vals.std(skipna=True)
        if pd.isna(std) or std < EPS:
            return pd.Series(np.zeros(len(vals)), index=vals.index, dtype=float)
        return ((vals - vals.mean(skipna=True)) / (std + EPS)).fillna(0.0)

    return (0.6745 * (vals - med) / (mad + EPS)).fillna(0.0)


def positive_z(x: pd.Series) -> pd.Series:
    return robust_z(x).clip(lower=0.0)


def load_node_geometry(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [id_col, "scale_index", "t", "geom_x", "geom_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"node geometry missing required columns: {missing}")

    df[id_col] = df[id_col].astype(str)
    df["scale_index"] = pd.to_numeric(df["scale_index"], errors="raise").astype(int)
    df["t"] = pd.to_numeric(df["t"], errors="raise").astype(float)
    df["geom_x"] = pd.to_numeric(df["geom_x"], errors="coerce")
    df["geom_y"] = pd.to_numeric(df["geom_y"], errors="coerce")

    if df.duplicated([id_col, "scale_index"]).any():
        dup = df[df.duplicated([id_col, "scale_index"])][[id_col, "scale_index"]].head(10)
        raise ValueError(f"duplicate node/scale rows:\n{dup}")

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
    x_t = np.asarray(data["X_t"], dtype=float)
    ts = np.asarray(data["t"], dtype=float)

    if x_t.ndim != 3:
        raise ValueError(f"X_t must be scales × nodes × observables; got {x_t.shape}")
    if x_t.shape[1] != len(ids):
        raise ValueError("X_t node dimension does not match ids")
    if x_t.shape[2] != len(observable_cols):
        raise ValueError("X_t observable dimension does not match observable_cols")
    if x_t.shape[0] != len(ts):
        raise ValueError("X_t scale dimension does not match t")

    return {
        "ids": ids,
        "observable_cols": observable_cols,
        "X_t": x_t,
        "t": ts,
    }


def inject_bundle_features(
    df: pd.DataFrame,
    bundle_path: Path | None,
    id_col: str,
) -> tuple[pd.DataFrame, list[str], str]:
    if bundle_path is None:
        return df, [], "not_provided"

    bundle = load_obs076a_bundle(bundle_path)
    ids = bundle["ids"]
    obs_cols = bundle["observable_cols"]
    x_t = bundle["X_t"]
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

    rows = []
    for sidx in range(len(ts)):
        block = pd.DataFrame({id_col: ids})
        block["scale_index"] = sidx
        block["bundle_t"] = float(ts[sidx])
        for j, col in enumerate(obs_cols):
            block[f"{DYN_PREFIX}{col}"] = x_t[sidx, :, j]
        rows.append(block)

    dyn = pd.concat(rows, ignore_index=True)
    dyn[id_col] = dyn[id_col].astype(str)

    dyn_cols = [c for c in dyn.columns if c.startswith(DYN_PREFIX)]

    merged = df.merge(
        dyn,
        on=[id_col, "scale_index"],
        how="left",
        validate="one_to_one",
    )

    return merged, dyn_cols, "ok"


def load_object_memberships(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    obj = pd.read_csv(path)

    required = [id_col, "scale_index", "object"]
    missing = [c for c in required if c not in obj.columns]
    if missing:
        raise ValueError(f"object membership missing required columns: {missing}")

    obj[id_col] = obj[id_col].astype(str)
    obj["scale_index"] = pd.to_numeric(obj["scale_index"], errors="raise").astype(int)
    obj["object"] = obj["object"].astype(str)

    return obj[[id_col, "scale_index", "object"]].drop_duplicates()


def add_background_memberships(df: pd.DataFrame, memberships: pd.DataFrame, id_col: str) -> pd.DataFrame:
    bg = df[[id_col, "scale_index"]].drop_duplicates().copy()
    bg["object"] = "__background_all_nodes__"
    return pd.concat([memberships, bg], ignore_index=True)


def available_measure_fields(df: pd.DataFrame) -> list[str]:
    preferred = [
        "energy",
        "density_score",
        "mean_knn_observable_distance",
        "phase_contrast",
        "seam_proxy_score",
        "is_seam_proxy",
        "signed_phase",
        "distance_to_seam",
        "lazarus_score",
        "response_strength",
        "frobenius_T",
        "trace_T",
        "signed_coupling",
        "cosine_alignment",
        "dyn__signed_phase",
        "dyn__distance_to_seam",
        "dyn__lazarus_score",
        "dyn__response_strength",
        "dyn__frobenius_T",
        "dyn__trace_T",
        "dyn__signed_coupling",
        "dyn__cosine_alignment",
        "dyn__grad_signed_phase_norm",
        "dyn__grad_lazarus_score_norm",
    ]

    fields = []
    for col in preferred:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            fields.append(col)

    return fields


def feature_matrix_for_geometry(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Choose numeric fields for local covariance / ID estimation.

    Preference:
      dynamic OBS-076a fields if available;
      otherwise geometry/proxy numeric fields.
    """
    dyn_cols = [
        c
        for c in df.columns
        if c.startswith(DYN_PREFIX)
        and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
    ]

    if dyn_cols:
        cols = sorted(dyn_cols)
    else:
        candidates = [
            "geom_x",
            "geom_y",
            "energy",
            "density_score",
            "mean_knn_observable_distance",
            "phase_contrast",
            "seam_proxy_score",
            "signed_phase",
            "distance_to_seam",
            "lazarus_score",
            "response_strength",
            "frobenius_T",
            "trace_T",
            "signed_coupling",
            "cosine_alignment",
        ]
        cols = [
            c
            for c in candidates
            if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
        ]

    if not cols:
        raise ValueError("No numeric fields available for local geometry")

    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).copy()

    for j in range(X.shape[1]):
        finite = np.isfinite(X[:, j])
        med = float(np.nanmedian(X[finite, j])) if finite.any() else 0.0
        X[~finite, j] = med

    return X, cols


def robust_standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    med = np.nanmedian(X, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    q75 = np.nanpercentile(X, 75, axis=0)
    iqr = q75 - q25
    iqr = np.where(np.abs(iqr) < EPS, 1.0, iqr)
    return (X - med) / iqr


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)


def participation_ratio(evals: np.ndarray) -> float:
    ev = np.asarray(evals, dtype=float)
    ev = ev[np.isfinite(ev)]
    ev = ev[ev > EPS]
    if len(ev) == 0:
        return float("nan")
    s1 = float(ev.sum())
    s2 = float(np.sum(ev * ev))
    if s2 < EPS:
        return float("nan")
    return float((s1 * s1) / s2)


def spectral_anisotropy(evals: np.ndarray) -> float:
    ev = np.asarray(evals, dtype=float)
    ev = ev[np.isfinite(ev)]
    ev = ev[ev > EPS]
    if len(ev) == 0:
        return float("nan")
    return float(np.max(ev) / (np.sum(ev) + EPS))


def log_volume_proxy(evals: np.ndarray, ridge_eps: float) -> float:
    ev = np.asarray(evals, dtype=float)
    ev = ev[np.isfinite(ev)]
    if len(ev) == 0:
        return float("nan")
    return float(np.sum(np.log(np.maximum(ev, 0.0) + ridge_eps)))


def twonn_id_from_distances(D: np.ndarray) -> float:
    """
    TwoNN regional estimator.

    Uses sorted ratios r2/r1 and the standard regression:
        -log(1 - F(mu)) ≈ d * log(mu)

    Returns slope through origin.
    """
    D = np.asarray(D, dtype=float).copy()
    n = D.shape[0]
    if n < 4:
        return float("nan")

    np.fill_diagonal(D, np.inf)
    ds = np.sort(D, axis=1)
    r1 = ds[:, 0]
    r2 = ds[:, 1]
    ok = np.isfinite(r1) & np.isfinite(r2) & (r1 > EPS) & (r2 > r1)
    if ok.sum() < 4:
        return float("nan")

    mu = np.sort(r2[ok] / r1[ok])
    m = len(mu)
    F = (np.arange(1, m + 1) - 0.5) / m

    x = np.log(mu)
    y = -np.log(np.maximum(1.0 - F, EPS))

    ok2 = np.isfinite(x) & np.isfinite(y) & (x > EPS)
    if ok2.sum() < 3:
        return float("nan")

    x = x[ok2]
    y = y[ok2]
    denom = float(np.sum(x * x))
    if denom < EPS:
        return float("nan")

    return float(np.sum(x * y) / denom)


def local_mle_id_from_distances(D: np.ndarray, k: int) -> float:
    """
    Levina-Bickel style local MLE ID averaged over points.

    For each point:
      d_i = 1 / mean_j log(T_k / T_j), j=1..k-1

    Returns the mean over valid local estimates.
    """
    D = np.asarray(D, dtype=float).copy()
    n = D.shape[0]
    if n < max(4, k + 1):
        return float("nan")

    kk = min(k, n - 1)
    np.fill_diagonal(D, np.inf)
    ds = np.sort(D, axis=1)[:, :kk]

    Tk = ds[:, kk - 1]
    inner = ds[:, : kk - 1]

    estimates = []
    for i in range(n):
        if not np.isfinite(Tk[i]) or Tk[i] <= EPS:
            continue
        vals = inner[i]
        ok = np.isfinite(vals) & (vals > EPS) & (vals < Tk[i])
        if ok.sum() < 2:
            continue
        logs = np.log(Tk[i] / vals[ok])
        denom = float(np.mean(logs))
        if denom > EPS and np.isfinite(denom):
            estimates.append(1.0 / denom)

    if not estimates:
        return float("nan")

    return float(np.mean(estimates))


def covariance_spectrum(X: np.ndarray, ridge_eps: float) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if n < 2 or d < 1:
        return np.array([], dtype=float), empty_geometry_metrics()

    Xc = X - np.mean(X, axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(n - 1, 1)
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(np.maximum(evals, 0.0))[::-1]

    lam1 = float(evals[0]) if len(evals) > 0 else float("nan")
    lam2 = float(evals[1]) if len(evals) > 1 else float("nan")
    lam3 = float(evals[2]) if len(evals) > 2 else float("nan")

    gap12 = float(lam1 / (lam2 + EPS)) if np.isfinite(lam1) and np.isfinite(lam2) else float("nan")
    gap23 = float(lam2 / (lam3 + EPS)) if np.isfinite(lam2) and np.isfinite(lam3) else float("nan")

    return evals, {
        "participation_ratio": participation_ratio(evals),
        "anisotropy": spectral_anisotropy(evals),
        "log_volume": log_volume_proxy(evals, ridge_eps=ridge_eps),
        "lambda1": lam1,
        "lambda2": lam2,
        "lambda3": lam3,
        "spectral_gap_1_2": gap12,
        "spectral_gap_2_3": gap23,
    }


def empty_geometry_metrics() -> dict:
    return {
        "participation_ratio": float("nan"),
        "anisotropy": float("nan"),
        "log_volume": float("nan"),
        "lambda1": float("nan"),
        "lambda2": float("nan"),
        "lambda3": float("nan"),
        "spectral_gap_1_2": float("nan"),
        "spectral_gap_2_3": float("nan"),
        "twonn_id": float("nan"),
        "local_mle_id": float("nan"),
    }


def object_support_table(
    df: pd.DataFrame,
    memberships: pd.DataFrame,
    id_col: str,
) -> pd.DataFrame:
    merged = memberships.merge(
        df,
        on=[id_col, "scale_index"],
        how="left",
        validate="many_to_one",
    )
    return merged


def compute_object_geometry(
    df: pd.DataFrame,
    memberships: pd.DataFrame,
    cfg: Config,
    measure_fields: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_df = object_support_table(df, memberships, cfg.id_col)

    geom_rows = []

    scale_feature_cache: dict[int, tuple[pd.DataFrame, np.ndarray, list[str]]] = {}
    for scale_index, sub in df.groupby("scale_index", sort=True):
        X, feature_cols = feature_matrix_for_geometry(sub)
        Xz = robust_standardize(X)
        tmp = sub[[cfg.id_col, "scale_index", "t"]].copy().reset_index(drop=True)
        scale_feature_cache[int(scale_index)] = (tmp, Xz, feature_cols)

    for (obj, scale_index), sub in support_df.groupby(["object", "scale_index"], sort=True):
        sub = sub.copy()
        t = safe_float(sub["t"].iloc[0])
        n = int(len(sub))
        ids = sorted(sub[cfg.id_col].astype(str).dropna().unique().tolist())

        row = {
            "case": cfg.case,
            "object": obj,
            "scale_index": int(scale_index),
            "t": t,
            "support_size": n,
            "node_ids": ",".join(ids),
            "centroid_geom_x": float(pd.to_numeric(sub["geom_x"], errors="coerce").mean()),
            "centroid_geom_y": float(pd.to_numeric(sub["geom_y"], errors="coerce").mean()),
            "std_geom_x": float(pd.to_numeric(sub["geom_x"], errors="coerce").std()),
            "std_geom_y": float(pd.to_numeric(sub["geom_y"], errors="coerce").std()),
        }

        for field in measure_fields:
            vals = pd.to_numeric(sub[field], errors="coerce")
            row[f"mean__{field}"] = float(vals.mean(skipna=True))
            row[f"std__{field}"] = float(vals.std(skipna=True))
            row[f"max__{field}"] = float(vals.max(skipna=True))
            row[f"min__{field}"] = float(vals.min(skipna=True))

        if n >= cfg.min_support:
            tmp, Xz, feature_cols = scale_feature_cache[int(scale_index)]
            id_to_pos = {node_id: i for i, node_id in enumerate(tmp[cfg.id_col].astype(str))}
            positions = [id_to_pos[i] for i in ids if i in id_to_pos]

            if len(positions) >= cfg.min_support:
                X_obj = Xz[positions, :]
                D_obj = pairwise_distances(X_obj)
                _, spec = covariance_spectrum(X_obj, ridge_eps=cfg.ridge_eps)
                row.update(spec)
                row["twonn_id"] = twonn_id_from_distances(D_obj)
                row["local_mle_id"] = local_mle_id_from_distances(D_obj, k=cfg.id_k)
                row["geometry_feature_cols"] = ",".join(feature_cols)
                row["geometry_status"] = "ok"
            else:
                row.update(empty_geometry_metrics())
                row["geometry_feature_cols"] = ",".join(feature_cols)
                row["geometry_status"] = "insufficient_matched_positions"
        else:
            row.update(empty_geometry_metrics())
            row["geometry_feature_cols"] = ""
            row["geometry_status"] = "below_min_support"

        geom_rows.append(row)

    feature_rows = [
        {
            "case": cfg.case,
            "feature_role": "measure_fields",
            "n_features": len(measure_fields),
            "features": ",".join(measure_fields),
        }
    ]

    if scale_feature_cache:
        _, _, feature_cols = next(iter(scale_feature_cache.values()))
        feature_rows.append(
            {
                "case": cfg.case,
                "feature_role": "local_geometry_features",
                "n_features": len(feature_cols),
                "features": ",".join(feature_cols),
            }
        )

    return pd.DataFrame(geom_rows), pd.DataFrame(feature_rows)


def parse_node_set(s: str) -> set[str]:
    if not isinstance(s, str) or not s:
        return set()
    return set(x for x in s.split(",") if x)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    union = a | b
    if not union:
        return float("nan")
    return float(len(a & b) / len(union))


def compute_transitions(object_geom: pd.DataFrame, measure_fields: list[str]) -> pd.DataFrame:
    rows = []

    for obj, sub in object_geom.groupby("object", sort=True):
        sub = sub.sort_values("scale_index").reset_index(drop=True)

        for i in range(len(sub) - 1):
            a = sub.iloc[i]
            b = sub.iloc[i + 1]

            ids_a = parse_node_set(a.get("node_ids", ""))
            ids_b = parse_node_set(b.get("node_ids", ""))

            cx_a = safe_float(a.get("centroid_geom_x"))
            cy_a = safe_float(a.get("centroid_geom_y"))
            cx_b = safe_float(b.get("centroid_geom_x"))
            cy_b = safe_float(b.get("centroid_geom_y"))

            centroid_drift = (
                float(math.sqrt((cx_b - cx_a) ** 2 + (cy_b - cy_a) ** 2))
                if all(np.isfinite([cx_a, cy_a, cx_b, cy_b]))
                else float("nan")
            )

            row = {
                "case": a["case"],
                "object": obj,
                "scale_index_from": int(a["scale_index"]),
                "scale_index_to": int(b["scale_index"]),
                "t_from": safe_float(a["t"]),
                "t_to": safe_float(b["t"]),
                "support_size_from": int(a["support_size"]),
                "support_size_to": int(b["support_size"]),
                "support_size_delta": int(b["support_size"]) - int(a["support_size"]),
                "support_size_ratio": float(
                    int(b["support_size"]) / max(int(a["support_size"]), 1)
                ),
                "jaccard_previous": jaccard(ids_a, ids_b),
                "centroid_drift": centroid_drift,
                "participation_ratio_from": safe_float(a.get("participation_ratio")),
                "participation_ratio_to": safe_float(b.get("participation_ratio")),
                "participation_ratio_delta": safe_float(b.get("participation_ratio"))
                - safe_float(a.get("participation_ratio")),
                "anisotropy_from": safe_float(a.get("anisotropy")),
                "anisotropy_to": safe_float(b.get("anisotropy")),
                "anisotropy_delta": safe_float(b.get("anisotropy"))
                - safe_float(a.get("anisotropy")),
                "log_volume_from": safe_float(a.get("log_volume")),
                "log_volume_to": safe_float(b.get("log_volume")),
                "log_volume_delta": safe_float(b.get("log_volume"))
                - safe_float(a.get("log_volume")),
                "twonn_id_from": safe_float(a.get("twonn_id")),
                "twonn_id_to": safe_float(b.get("twonn_id")),
                "twonn_id_delta": safe_float(b.get("twonn_id"))
                - safe_float(a.get("twonn_id")),
                "local_mle_id_from": safe_float(a.get("local_mle_id")),
                "local_mle_id_to": safe_float(b.get("local_mle_id")),
                "local_mle_id_delta": safe_float(b.get("local_mle_id"))
                - safe_float(a.get("local_mle_id")),
            }

            for field in measure_fields:
                ma = safe_float(a.get(f"mean__{field}"))
                mb = safe_float(b.get(f"mean__{field}"))
                row[f"mean_delta__{field}"] = mb - ma

            rows.append(row)

    return pd.DataFrame(rows)


def load_overlap_table(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        return None, "not_provided"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if not {"scale_index", "object_a", "object_b"}.issubset(df.columns):
        rename = {}
        if "object_1" in df.columns:
            rename["object_1"] = "object_a"
        if "object_2" in df.columns:
            rename["object_2"] = "object_b"
        if rename:
            df = df.rename(columns=rename)

    missing = [c for c in ["scale_index", "object_a", "object_b"] if c not in df.columns]
    if missing:
        return df, f"missing_expected_columns:{missing}"

    df["scale_index"] = pd.to_numeric(df["scale_index"], errors="raise").astype(int)
    df["object_a"] = df["object_a"].astype(str)
    df["object_b"] = df["object_b"].astype(str)

    metric_candidates = [
        "jaccard",
        "jaccard_overlap",
        "overlap_jaccard",
        "intersection_over_union",
        "intersection_size",
        "intersection",
    ]

    metric_col = None
    for c in metric_candidates:
        if c in df.columns:
            metric_col = c
            break

    if metric_col is None:
        if {"n_intersection", "n_union"}.issubset(df.columns):
            df["jaccard"] = pd.to_numeric(df["n_intersection"], errors="coerce") / (
                pd.to_numeric(df["n_union"], errors="coerce") + EPS
            )
            metric_col = "jaccard"
        elif {"intersection_count", "union_count"}.issubset(df.columns):
            df["jaccard"] = pd.to_numeric(df["intersection_count"], errors="coerce") / (
                pd.to_numeric(df["union_count"], errors="coerce") + EPS
            )
            metric_col = "jaccard"
        else:
            return df, "no_overlap_metric"

    df["overlap_metric"] = pd.to_numeric(df[metric_col], errors="coerce")
    df["overlap_metric_col"] = metric_col

    return df, "ok"


def compute_overlap_transitions(overlap_df: pd.DataFrame | None) -> pd.DataFrame:
    if overlap_df is None or overlap_df.empty:
        return pd.DataFrame()

    required = ["scale_index", "object_a", "object_b", "overlap_metric"]
    if any(c not in overlap_df.columns for c in required):
        return pd.DataFrame()

    rows = []
    for (a_obj, b_obj), sub in overlap_df.groupby(["object_a", "object_b"], sort=True):
        sub = sub.sort_values("scale_index").reset_index(drop=True)

        for i in range(len(sub) - 1):
            a = sub.iloc[i]
            b = sub.iloc[i + 1]

            delta = safe_float(b["overlap_metric"]) - safe_float(a["overlap_metric"])

            rows.append(
                {
                    "object_a": a_obj,
                    "object_b": b_obj,
                    "scale_index_from": int(a["scale_index"]),
                    "scale_index_to": int(b["scale_index"]),
                    "overlap_from": safe_float(a["overlap_metric"]),
                    "overlap_to": safe_float(b["overlap_metric"]),
                    "overlap_delta": delta,
                    "abs_overlap_delta": abs(delta),
                    "metric_col": a.get("overlap_metric_col", "overlap_metric"),
                }
            )

    return pd.DataFrame(rows)


def attach_overlap_delta_max(transitions: pd.DataFrame, overlap_transitions: pd.DataFrame) -> pd.DataFrame:
    transitions = transitions.copy()

    if overlap_transitions is None or overlap_transitions.empty:
        transitions["overlap_delta_max"] = 0.0
        transitions["overlap_delta_partner"] = ""
        transitions["overlap_delta_signed"] = 0.0
        return transitions

    max_vals = []
    partners = []
    signed_vals = []

    for row in transitions.itertuples(index=False):
        obj = row.object
        s_from = int(row.scale_index_from)
        s_to = int(row.scale_index_to)

        sub = overlap_transitions[
            (overlap_transitions["scale_index_from"] == s_from)
            & (overlap_transitions["scale_index_to"] == s_to)
            & (
                (overlap_transitions["object_a"] == obj)
                | (overlap_transitions["object_b"] == obj)
            )
        ].copy()

        if sub.empty:
            max_vals.append(0.0)
            partners.append("")
            signed_vals.append(0.0)
            continue

        idx = pd.to_numeric(sub["abs_overlap_delta"], errors="coerce").idxmax()
        best = sub.loc[idx]
        partner = best["object_b"] if best["object_a"] == obj else best["object_a"]

        max_vals.append(safe_float(best["abs_overlap_delta"]))
        partners.append(str(partner))
        signed_vals.append(safe_float(best["overlap_delta"]))

    transitions["overlap_delta_max"] = max_vals
    transitions["overlap_delta_partner"] = partners
    transitions["overlap_delta_signed"] = signed_vals
    return transitions


def dominant_family_from_scores(row: pd.Series, ratio: float) -> str:
    family_scores = {
        "support": safe_float(row.get("support_score")),
        "overlap": safe_float(row.get("overlap_score")),
        "shape": safe_float(row.get("shape_score")),
        "intrinsic_dimension": safe_float(row.get("id_score")),
    }

    family_scores = {
        k: (0.0 if not np.isfinite(v) else max(0.0, v))
        for k, v in family_scores.items()
    }

    ordered = sorted(family_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_val = ordered[0]
    second_val = ordered[1][1] if len(ordered) > 1 else 0.0

    if best_val <= EPS:
        return "none"

    if second_val <= EPS:
        return best_name

    if best_val >= ratio * second_val:
        return best_name

    return "mixed"


def compute_pinch_scores(transitions: pd.DataFrame, family_dominance_ratio: float) -> pd.DataFrame:
    if transitions.empty:
        return transitions

    df = transitions.copy()

    df["jaccard_loss"] = 1.0 - pd.to_numeric(df["jaccard_previous"], errors="coerce")
    df["abs_support_size_delta"] = pd.to_numeric(df["support_size_delta"], errors="coerce").abs()
    df["abs_participation_ratio_delta"] = pd.to_numeric(df["participation_ratio_delta"], errors="coerce").abs()
    df["abs_anisotropy_delta"] = pd.to_numeric(df["anisotropy_delta"], errors="coerce").abs()
    df["abs_log_volume_delta"] = pd.to_numeric(df["log_volume_delta"], errors="coerce").abs()
    df["abs_twonn_id_delta"] = pd.to_numeric(df["twonn_id_delta"], errors="coerce").abs()
    df["abs_local_mle_id_delta"] = pd.to_numeric(df["local_mle_id_delta"], errors="coerce").abs()

    components = {
        "pinch_component_jaccard_loss": positive_z(df["jaccard_loss"]),
        "pinch_component_centroid_drift": positive_z(df["centroid_drift"]),
        "pinch_component_overlap_delta": positive_z(df["overlap_delta_max"]),
        "pinch_component_support_size_delta": positive_z(df["abs_support_size_delta"]),
        "pinch_component_participation_delta": positive_z(df["abs_participation_ratio_delta"]),
        "pinch_component_anisotropy_delta": positive_z(df["abs_anisotropy_delta"]),
        "pinch_component_log_volume_delta": positive_z(df["abs_log_volume_delta"]),
        "pinch_component_twonn_delta": positive_z(df["abs_twonn_id_delta"]),
        "pinch_component_local_mle_delta": positive_z(df["abs_local_mle_id_delta"]),
    }

    for name, vals in components.items():
        df[name] = vals

    df["support_score"] = (
        df["pinch_component_jaccard_loss"]
        + df["pinch_component_centroid_drift"]
        + df["pinch_component_support_size_delta"]
    )
    df["overlap_score"] = df["pinch_component_overlap_delta"]
    df["shape_score"] = (
        df["pinch_component_participation_delta"]
        + df["pinch_component_anisotropy_delta"]
        + df["pinch_component_log_volume_delta"]
    )
    df["id_score"] = (
        df["pinch_component_twonn_delta"]
        + df["pinch_component_local_mle_delta"]
    )

    df["pinch_score_total"] = (
        df["support_score"]
        + df["overlap_score"]
        + df["shape_score"]
        + df["id_score"]
    )

    # Backward-compatible alias.
    df["pinch_score"] = df["pinch_score_total"]

    component_cols = list(components.keys())
    dominant_reasons = []
    for _, row in df.iterrows():
        vals = {c: safe_float(row[c]) for c in component_cols}
        best = max(vals.items(), key=lambda kv: kv[1])
        if best[1] <= 0:
            dominant_reasons.append("no_strong_component")
        else:
            dominant_reasons.append(best[0].replace("pinch_component_", ""))
    df["dominant_reason"] = dominant_reasons

    df["dominant_family"] = df.apply(
        lambda row: dominant_family_from_scores(row, family_dominance_ratio),
        axis=1,
    )

    ordered_cols = [
        "case",
        "object",
        "scale_index_from",
        "scale_index_to",
        "t_from",
        "t_to",
        "pinch_score_total",
        "pinch_score",
        "support_score",
        "overlap_score",
        "shape_score",
        "id_score",
        "dominant_family",
        "dominant_reason",
    ]
    rest = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + rest]

    return df.sort_values("pinch_score_total", ascending=False).reset_index(drop=True)


def markdown_float(x, digits: int = 6) -> str:
    y = safe_float(x)
    if not np.isfinite(y):
        return ""
    return f"{y:.{digits}g}"


def write_report(
    cfg: Config,
    input_manifest: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    object_geom: pd.DataFrame,
    transitions: pd.DataFrame,
    overlap_transitions: pd.DataFrame,
    pinch: pd.DataFrame,
) -> None:
    lines = [
        "# OBS-077a — Pinch-point geometry",
        "",
        "## Scope",
        "",
        "OBS-077a detects candidate scale-local support reorganization events.",
        "",
        "This is a geometry/proxy diagnostic over OBS-076 scale-space artifacts.",
        "",
        "It does not claim formal singularities, topological defects, or path/text semantics.",
        "",
        "## v2 patch",
        "",
        "v2 decomposes the total pinch score into family-level scores:",
        "",
        "```text",
        "support_score = jaccard loss + centroid drift + support size change",
        "overlap_score = max object-overlap change",
        "shape_score   = participation-ratio + anisotropy + log-volume change",
        "id_score      = TwoNN + local MLE intrinsic-dimension change",
        "```",
        "",
        "This separates support transitions, overlap transitions, shape transitions, and intrinsic-dimension transitions.",
        "",
        "## Configuration",
        "",
        f"- case: `{cfg.case}`",
        f"- k_neighbors: `{cfg.k_neighbors}`",
        f"- id_k: `{cfg.id_k}`",
        f"- min_support: `{cfg.min_support}`",
        f"- include_background: `{cfg.include_background}`",
        f"- family_dominance_ratio: `{cfg.family_dominance_ratio}`",
        "",
        "## Inputs",
        "",
        "| artifact | status | details | path |",
        "| --- | --- | --- | --- |",
    ]

    for row in input_manifest.itertuples(index=False):
        lines.append(f"| {row.artifact} | {row.status} | {row.details} | `{row.path}` |")

    lines.extend(
        [
            "",
            "## Feature manifest",
            "",
            "| feature_role | n_features | features |",
            "| --- | ---: | --- |",
        ]
    )

    for row in feature_manifest.itertuples(index=False):
        lines.append(f"| {row.feature_role} | {int(row.n_features)} | `{row.features}` |")

    lines.extend(
        [
            "",
            "## Object geometry coverage",
            "",
            f"- object-scale rows: `{len(object_geom)}`",
            f"- transition rows: `{len(transitions)}`",
            f"- overlap-transition rows: `{len(overlap_transitions) if overlap_transitions is not None else 0}`",
            "",
        ]
    )

    if not object_geom.empty:
        cov = (
            object_geom.groupby("object")
            .agg(
                n_scales=("scale_index", "nunique"),
                min_support=("support_size", "min"),
                max_support=("support_size", "max"),
                mean_support=("support_size", "mean"),
                ok_geometry=("geometry_status", lambda s: int((s == "ok").sum())),
            )
            .reset_index()
            .sort_values("object")
        )

        lines.extend(
            [
                "| object | n_scales | min_support | max_support | mean_support | ok_geometry |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in cov.itertuples(index=False):
            lines.append(
                f"| {row.object} | {int(row.n_scales)} | {int(row.min_support)} | "
                f"{int(row.max_support)} | {float(row.mean_support):.4g} | {int(row.ok_geometry)} |"
            )

    lines.extend(["", "## Top pinch-point candidates", ""])

    if pinch.empty:
        lines.append("No pinch candidates computed.")
    else:
        top = pinch.head(cfg.top_n)
        lines.extend(
            [
                "| rank | object | scale_from | scale_to | t_from | t_to | total | support | overlap | shape | id | dominant_family | dominant_reason | jaccard | drift | partner |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            lines.append(
                "| "
                f"{rank} | "
                f"{row.object} | "
                f"{int(row.scale_index_from)} | "
                f"{int(row.scale_index_to)} | "
                f"{markdown_float(row.t_from)} | "
                f"{markdown_float(row.t_to)} | "
                f"{markdown_float(row.pinch_score_total)} | "
                f"{markdown_float(row.support_score)} | "
                f"{markdown_float(row.overlap_score)} | "
                f"{markdown_float(row.shape_score)} | "
                f"{markdown_float(row.id_score)} | "
                f"{row.dominant_family} | "
                f"{row.dominant_reason} | "
                f"{markdown_float(row.jaccard_previous)} | "
                f"{markdown_float(row.centroid_drift)} | "
                f"{row.overlap_delta_partner} |"
            )

    if not pinch.empty:
        lines.extend(["", "## Dominant family summary", ""])

        fam = (
            pinch.groupby("dominant_family")
            .agg(
                n_events=("dominant_family", "size"),
                max_total=("pinch_score_total", "max"),
                mean_total=("pinch_score_total", "mean"),
            )
            .reset_index()
            .sort_values("max_total", ascending=False)
        )

        lines.extend(
            [
                "| dominant_family | n_events | max_total | mean_total |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in fam.itertuples(index=False):
            lines.append(
                f"| {row.dominant_family} | {int(row.n_events)} | "
                f"{markdown_float(row.max_total)} | {markdown_float(row.mean_total)} |"
            )

        lines.extend(["", "## Top event by object", ""])

        by_obj = (
            pinch.sort_values("pinch_score_total", ascending=False)
            .groupby("object", as_index=False)
            .head(1)
            .sort_values("pinch_score_total", ascending=False)
        )

        lines.extend(
            [
                "| object | scale_from | scale_to | total | support | overlap | shape | id | dominant_family | jaccard | drift | partner |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in by_obj.itertuples(index=False):
            lines.append(
                "| "
                f"{row.object} | "
                f"{int(row.scale_index_from)} | "
                f"{int(row.scale_index_to)} | "
                f"{markdown_float(row.pinch_score_total)} | "
                f"{markdown_float(row.support_score)} | "
                f"{markdown_float(row.overlap_score)} | "
                f"{markdown_float(row.shape_score)} | "
                f"{markdown_float(row.id_score)} | "
                f"{row.dominant_family} | "
                f"{markdown_float(row.jaccard_previous)} | "
                f"{markdown_float(row.centroid_drift)} | "
                f"{row.overlap_delta_partner} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "A high total score indicates a candidate support-reorganization transition.",
            "",
            "Family scores should be read as follows:",
            "",
            "```text",
            "support_score:",
            "  support identity changes, drifts, or changes size",
            "",
            "overlap_score:",
            "  relation to another structural object changes sharply",
            "",
            "shape_score:",
            "  covariance geometry changes without necessarily changing membership",
            "",
            "id_score:",
            "  local kNN distance geometry changes under intrinsic-dimension estimators",
            "```",
            "",
            "Useful event language:",
            "",
            "```text",
            "support-dominant event       = support replacement / migration",
            "overlap-dominant event       = factor relation transition",
            "shape-dominant event         = support-stable geometric deformation",
            "id-dominant event            = intrinsic-dimension / neighborhood-geometry shift",
            "mixed event                  = multiple transition modes active",
            "```",
            "",
            "## Guardrails",
            "",
            "- Pinch-point candidates are not formal singularities.",
            "- Intrinsic-dimension estimates are local diagnostics, not final topology.",
            "- Object supports are OBS-076c quantile/proxy objects.",
            "- This analysis is node-level and scale-space-local.",
            "- Path/text semantics require a later projection layer.",
            "",
            "## Output artifacts",
            "",
            "- `obs077a_input_manifest.csv`",
            "- `obs077a_feature_manifest.csv`",
            "- `obs077a_object_geometry_by_scale.csv`",
            "- `obs077a_object_transition_events.csv`",
            "- `obs077a_overlap_transition_events.csv`",
            "- `obs077a_pinch_point_candidates.csv`",
            "- `obs077a_report.md`",
            "",
            "## Next step",
            "",
            "Compare dominant-family patterns across C, Cp2, and Cp3 before projecting path labels.",
            "",
        ]
    )

    (cfg.outdir / "obs077a_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    node_df = load_node_geometry(cfg.node_geometry, cfg.id_col)
    node_df, dyn_cols, dyn_status = inject_bundle_features(node_df, cfg.bundle, cfg.id_col)

    memberships = load_object_memberships(cfg.objects, cfg.id_col)
    if cfg.include_background:
        memberships = add_background_memberships(node_df, memberships, cfg.id_col)

    overlap_df, overlap_status = load_overlap_table(cfg.overlaps)

    measure_fields = available_measure_fields(node_df)

    object_geom, feature_manifest = compute_object_geometry(
        df=node_df,
        memberships=memberships,
        cfg=cfg,
        measure_fields=measure_fields,
    )

    transitions = compute_transitions(object_geom, measure_fields=measure_fields)
    overlap_transitions = compute_overlap_transitions(overlap_df)
    transitions = attach_overlap_delta_max(transitions, overlap_transitions)
    pinch = compute_pinch_scores(
        transitions,
        family_dominance_ratio=cfg.family_dominance_ratio,
    )

    input_manifest = pd.DataFrame(
        [
            {
                "artifact": "obs076b_node_geometry",
                "path": str(cfg.node_geometry),
                "status": "ok",
                "details": f"rows={len(node_df)}",
            },
            {
                "artifact": "obs076c_objects",
                "path": str(cfg.objects),
                "status": "ok",
                "details": f"membership_rows={len(memberships)}",
            },
            {
                "artifact": "obs076a_bundle",
                "path": str(cfg.bundle) if cfg.bundle else "",
                "status": dyn_status,
                "details": f"dynamic_columns={len(dyn_cols)}",
            },
            {
                "artifact": "obs076c_overlaps",
                "path": str(cfg.overlaps) if cfg.overlaps else "",
                "status": overlap_status,
                "details": "",
            },
        ]
    )

    input_manifest.to_csv(cfg.outdir / "obs077a_input_manifest.csv", index=False)
    feature_manifest.to_csv(cfg.outdir / "obs077a_feature_manifest.csv", index=False)
    object_geom.to_csv(cfg.outdir / "obs077a_object_geometry_by_scale.csv", index=False)
    transitions.to_csv(cfg.outdir / "obs077a_object_transition_events.csv", index=False)
    overlap_transitions.to_csv(cfg.outdir / "obs077a_overlap_transition_events.csv", index=False)
    pinch.to_csv(cfg.outdir / "obs077a_pinch_point_candidates.csv", index=False)

    write_report(
        cfg=cfg,
        input_manifest=input_manifest,
        feature_manifest=feature_manifest,
        object_geom=object_geom,
        transitions=transitions,
        overlap_transitions=overlap_transitions,
        pinch=pinch,
    )

    print("OBS-077a complete")
    print("wrote:", cfg.outdir / "obs077a_input_manifest.csv")
    print("wrote:", cfg.outdir / "obs077a_feature_manifest.csv")
    print("wrote:", cfg.outdir / "obs077a_object_geometry_by_scale.csv")
    print("wrote:", cfg.outdir / "obs077a_object_transition_events.csv")
    print("wrote:", cfg.outdir / "obs077a_overlap_transition_events.csv")
    print("wrote:", cfg.outdir / "obs077a_pinch_point_candidates.csv")
    print("wrote:", cfg.outdir / "obs077a_report.md")


if __name__ == "__main__":
    main()
