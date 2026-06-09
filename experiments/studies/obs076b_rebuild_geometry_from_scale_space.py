#!/usr/bin/env python3
"""
obs076b_rebuild_geometry_from_scale_space.py

OBS-076b — Rebuild observable-space geometry from scale-space substrate

Purpose
-------
OBS-076b consumes an OBS-076a diffusion bundle and rebuilds simple
observable-space geometry at each diffusion scale.

It answers:

    When observable fields are diffused across scale,
    do geometry-level proxies persist, migrate, or collapse?

Scope discipline
----------------
This script does NOT recompute canonical Fisher geometry.

It rebuilds observable-space geometry from X(t):

    X(t)
      -> standardized observable matrix Z(t)
      -> pairwise observable-space distance D_X(t)
      -> 2D embedding E(t)
      -> proxy structural summaries

The resulting seam/phase/density/high-energy structures are proxies.
They are scale-space robustness diagnostics, not replacements for the
canonical PAM pipeline.

Inputs
------
- OBS-076a diffusion bundle:
    obs076a_diffusion_bundle.npz

Required bundle keys:
    ids
    observable_cols
    X0
    X_t
    t

Optional bundle keys:
    W
    L_sym
    degrees

Optional node context:
    CSV with id column and context fields such as r, alpha, mds1, mds2,
    signed_phase, distance_to_seam, lazarus_score, response_strength,
    trace_T, frobenius_T.

Outputs
-------
outdir/
  obs076b_node_geometry_by_scale.csv
  obs076b_scale_geometry_summary.csv
  obs076b_topk_geometry_persistence.csv
  obs076b_seam_proxy_persistence.csv
  obs076b_phase_proxy_summary.csv
  obs076b_report.md

Main proxy metrics
------------------
Per scale:
- pairwise-distance correlation vs base
- embedding correlation vs base distances
- observable variance retained
- mean kNN distance
- energy top-k persistence
- density top-k persistence
- seam proxy persistence when a phase column exists
- phase-sign agreement when a phase column exists

Guardrail
---------
OBS-076b v1 is observable-space geometry rebuild. It is not canonical
Fisher geometry rebuild. It establishes whether OBS-076a support migration
has a geometry-level proxy signature.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class Config:
    bundle: Path
    outdir: Path
    node_context: Path | None
    id_col: str
    k_density: int
    topk_values: list[int]
    seam_quantile: float
    random_state: int
    max_mds_iter: int
    use_mds: bool


def as_str_array(x: np.ndarray) -> np.ndarray:
    return np.array([str(v) for v in x.tolist()], dtype=str)


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    required = ["ids", "observable_cols", "X0", "X_t", "t"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"OBS-076a bundle missing required keys: {missing}")

    bundle = {k: data[k] for k in data.files}

    ids = as_str_array(bundle["ids"])
    obs_cols = as_str_array(bundle["observable_cols"])
    X_t = np.asarray(bundle["X_t"], dtype=float)
    ts = np.asarray(bundle["t"], dtype=float)

    if X_t.ndim != 3:
        raise ValueError(f"X_t must have shape scales × nodes × observables; got {X_t.shape}")
    if X_t.shape[1] != len(ids):
        raise ValueError(f"X_t node dimension {X_t.shape[1]} != len(ids) {len(ids)}")
    if X_t.shape[2] != len(obs_cols):
        raise ValueError(f"X_t observable dimension {X_t.shape[2]} != len(observable_cols) {len(obs_cols)}")
    if X_t.shape[0] != len(ts):
        raise ValueError(f"X_t scale dimension {X_t.shape[0]} != len(t) {len(ts)}")

    bundle["ids"] = ids
    bundle["observable_cols"] = obs_cols
    bundle["X_t"] = X_t
    bundle["t"] = ts
    return bundle


def load_node_context(path: Path | None, ids: np.ndarray, id_col: str) -> pd.DataFrame:
    base = pd.DataFrame({id_col: ids})

    if path is None:
        return base

    if not path.exists():
        raise FileNotFoundError(path)

    ctx = pd.read_csv(path)
    if id_col not in ctx.columns:
        raise ValueError(f"id column {id_col!r} not found in node context {path}")

    ctx[id_col] = ctx[id_col].astype(str)
    if ctx[id_col].duplicated().any():
        dupes = ctx[id_col].value_counts()
        dupes = dupes[dupes > 1].head(10).to_dict()
        raise ValueError(f"node context has duplicate ids: {dupes}")

    return base.merge(ctx, on=id_col, how="left")


def standardize_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Z = X.copy()

    for j in range(Z.shape[1]):
        col = Z[:, j]
        finite = np.isfinite(col)

        if not finite.any():
            Z[:, j] = 0.0
            continue

        med = float(np.nanmedian(col[finite]))
        q25 = float(np.nanquantile(col[finite], 0.25))
        q75 = float(np.nanquantile(col[finite], 0.75))
        scale = q75 - q25

        if not np.isfinite(scale) or scale <= EPS:
            scale = float(np.nanstd(col[finite]))
        if not np.isfinite(scale) or scale <= EPS:
            scale = 1.0

        z = (col - med) / scale
        z[~finite] = 0.0
        Z[:, j] = z

    return Z


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def upper_triangle_values(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)

    if finite.sum() < 2:
        return np.nan

    aa = a[finite]
    bb = b[finite]
    if np.std(aa) <= EPS or np.std(bb) <= EPS:
        return np.nan

    return float(np.corrcoef(aa, bb)[0, 1])


def classical_mds(D: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, float]:
    """
    Deterministic classical MDS.

    Returns:
      embedding, reconstruction correlation between input distances and
      embedding distances.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D * D) @ J
    B = 0.5 * (B + B.T)

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    pos = np.maximum(eigvals[:n_components], 0.0)
    coords = eigvecs[:, :n_components] * np.sqrt(pos)[None, :]

    if coords.shape[1] < n_components:
        padded = np.zeros((n, n_components), dtype=float)
        padded[:, : coords.shape[1]] = coords
        coords = padded

    D_emb = pairwise_distances(coords)
    fit_corr = corr(upper_triangle_values(D), upper_triangle_values(D_emb))
    return coords, fit_corr


def sklearn_metric_mds(D: np.ndarray, random_state: int, max_iter: int) -> tuple[np.ndarray, float]:
    """
    Optional metric MDS. Falls back to classical MDS if sklearn import or fit fails.
    """
    try:
        from sklearn.manifold import MDS
    except Exception:
        return classical_mds(D, n_components=2)

    try:
        model = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress="auto",
            max_iter=max_iter,
            n_init=4,
        )
        coords = model.fit_transform(D)
        D_emb = pairwise_distances(coords)
        fit_corr = corr(upper_triangle_values(D), upper_triangle_values(D_emb))
        return coords, fit_corr
    except Exception:
        return classical_mds(D, n_components=2)


def orient_embedding_to_base(E: np.ndarray, E_base: np.ndarray | None) -> np.ndarray:
    """
    Procrustes-align E to base orientation when possible.
    This stabilizes scale-by-scale embedding coordinates for summaries/plots.
    """
    E = np.asarray(E, dtype=float)

    if E_base is None:
        return E

    A = E - E.mean(axis=0, keepdims=True)
    B = E_base - E_base.mean(axis=0, keepdims=True)

    if A.shape != B.shape or A.shape[1] != 2:
        return E

    try:
        U, _, Vt = np.linalg.svd(A.T @ B)
        R = U @ Vt
        aligned = A @ R
        aligned += E_base.mean(axis=0, keepdims=True)
        return aligned
    except Exception:
        return E


def row_energy(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(X, dtype=float) ** 2, axis=1))


def local_density_score(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Density score is negative mean kNN distance so higher means denser.
    Returns:
      density_score, mean_knn_distance
    """
    n = D.shape[0]
    k = min(max(1, int(k)), n - 1)

    mean_knn = np.zeros(n, dtype=float)
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        vals = np.sort(row[np.isfinite(row)])[:k]
        mean_knn[i] = float(np.mean(vals)) if vals.size else np.nan

    density = -mean_knn
    return density, mean_knn


def topk_set(values: np.ndarray, k: int) -> set[int]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    idx = np.where(finite)[0]

    if idx.size == 0:
        return set()

    k = min(max(1, int(k)), idx.size)
    order = idx[np.argsort(values[idx])[::-1]]
    return set(int(i) for i in order[:k])


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def find_observable_index(observable_cols: np.ndarray, candidates: Iterable[str]) -> int | None:
    cols = [str(c) for c in observable_cols]
    for cand in candidates:
        if cand in cols:
            return cols.index(cand)
    return None


def seam_proxy_from_phase(
    phase: np.ndarray,
    D_emb: np.ndarray,
    seam_quantile: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a seam proxy from phase field.

    Components:
    - abs_phase small
    - local phase contrast high

    Returns:
      seam_score, is_seam_proxy, local_phase_contrast
    """
    phase = np.asarray(phase, dtype=float)
    n = len(phase)
    k = min(max(1, int(k)), n - 1)

    contrast = np.zeros(n, dtype=float)

    for i in range(n):
        row = D_emb[i].copy()
        row[i] = np.inf
        neigh = np.argsort(row)[:k]
        vals = np.abs(phase[i] - phase[neigh])
        vals = vals[np.isfinite(vals)]
        contrast[i] = float(np.mean(vals)) if vals.size else 0.0

    abs_phase = np.abs(phase)
    # Smaller abs_phase is seam-like; higher contrast is seam-like.
    abs_component = 1.0 / (abs_phase + EPS)
    score = abs_component * (contrast + EPS)

    finite = np.isfinite(score)
    if finite.sum() == 0:
        return score, np.zeros(n, dtype=bool), contrast

    q = float(np.nanquantile(score[finite], seam_quantile))
    is_seam = score >= q
    return score, is_seam, contrast


def phase_sign_agreement(base_phase: np.ndarray, phase: np.ndarray) -> float:
    base = np.sign(base_phase)
    cur = np.sign(phase)

    valid = np.isfinite(base_phase) & np.isfinite(phase)
    # Ignore exact zero signs because they are seam-like ambiguous.
    valid &= base != 0

    if valid.sum() == 0:
        return np.nan

    return float(np.mean(base[valid] == cur[valid]))


def safe_context_columns(ctx: pd.DataFrame, id_col: str) -> list[str]:
    preferred = [
        id_col,
        "node_id",
        "i",
        "j",
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
    ]
    return [c for c in preferred if c in ctx.columns]


def compute_scale_geometry(
    ids: np.ndarray,
    observable_cols: np.ndarray,
    X_t: np.ndarray,
    ts: np.ndarray,
    ctx: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    topk_rows: list[dict] = []
    seam_rows: list[dict] = []
    phase_rows: list[dict] = []

    base_D: np.ndarray | None = None
    base_E: np.ndarray | None = None
    base_energy_top: dict[int, set[int]] = {}
    prev_energy_top: dict[int, set[int]] = {}
    base_density_top: dict[int, set[int]] = {}
    prev_density_top: dict[int, set[int]] = {}
    base_seam_set: set[int] | None = None
    prev_seam_set: set[int] | None = None
    base_phase: np.ndarray | None = None

    phase_idx = find_observable_index(
        observable_cols,
        candidates=["signed_phase", "phase", "phase_coordinate"],
    )

    id_col = cfg.id_col
    context_cols = safe_context_columns(ctx, id_col)

    for scale_index, (t, X) in enumerate(zip(ts, X_t)):
        Z = standardize_columns(X)
        D = pairwise_distances(Z)

        if cfg.use_mds:
            E_raw, embedding_fit_corr = sklearn_metric_mds(
                D,
                random_state=cfg.random_state + scale_index,
                max_iter=cfg.max_mds_iter,
            )
        else:
            E_raw, embedding_fit_corr = classical_mds(D, n_components=2)

        E = orient_embedding_to_base(E_raw, base_E)
        D_emb = pairwise_distances(E)

        energy = row_energy(Z)
        density, mean_knn = local_density_score(D, cfg.k_density)

        if base_D is None:
            base_D = D.copy()
        if base_E is None:
            base_E = E.copy()

        distance_corr_vs_base = corr(
            upper_triangle_values(base_D),
            upper_triangle_values(D),
        )

        emb_distance_corr_vs_base = corr(
            upper_triangle_values(pairwise_distances(base_E)),
            upper_triangle_values(D_emb),
        )

        variance_mean = float(np.mean(np.var(Z, axis=0)))
        distance_mean = float(np.mean(upper_triangle_values(D)))
        distance_median = float(np.median(upper_triangle_values(D)))
        mean_knn_distance = float(np.nanmean(mean_knn))

        summary_rows.append(
            {
                "scale_index": int(scale_index),
                "t": float(t),
                "n_nodes": int(len(ids)),
                "n_observables": int(Z.shape[1]),
                "observable_distance_corr_vs_base": distance_corr_vs_base,
                "embedding_distance_corr_vs_base": emb_distance_corr_vs_base,
                "embedding_fit_corr": embedding_fit_corr,
                "mean_observable_variance_after_standardization": variance_mean,
                "mean_pairwise_observable_distance": distance_mean,
                "median_pairwise_observable_distance": distance_median,
                "mean_knn_observable_distance": mean_knn_distance,
                "mean_energy": float(np.nanmean(energy)),
                "max_energy": float(np.nanmax(energy)),
                "mean_density_score": float(np.nanmean(density)),
            }
        )

        for k in cfg.topk_values:
            e_cur = topk_set(energy, k)
            d_cur = topk_set(density, k)

            if scale_index == 0:
                base_energy_top[k] = e_cur
                prev_energy_top[k] = e_cur
                base_density_top[k] = d_cur
                prev_density_top[k] = d_cur

            topk_rows.append(
                {
                    "scale_index": int(scale_index),
                    "t": float(t),
                    "metric": "energy",
                    "topk": int(k),
                    "jaccard_vs_base": float(jaccard(base_energy_top[k], e_cur)),
                    "jaccard_vs_previous": float(jaccard(prev_energy_top[k], e_cur)),
                }
            )
            topk_rows.append(
                {
                    "scale_index": int(scale_index),
                    "t": float(t),
                    "metric": "density",
                    "topk": int(k),
                    "jaccard_vs_base": float(jaccard(base_density_top[k], d_cur)),
                    "jaccard_vs_previous": float(jaccard(prev_density_top[k], d_cur)),
                }
            )

            prev_energy_top[k] = e_cur
            prev_density_top[k] = d_cur

        phase = None
        seam_score = np.full(len(ids), np.nan, dtype=float)
        seam_mask = np.zeros(len(ids), dtype=bool)
        phase_contrast = np.full(len(ids), np.nan, dtype=float)

        if phase_idx is not None:
            phase = X[:, phase_idx].astype(float)
            seam_score, seam_mask, phase_contrast = seam_proxy_from_phase(
                phase=phase,
                D_emb=D_emb,
                seam_quantile=cfg.seam_quantile,
                k=cfg.k_density,
            )

            cur_seam = set(np.where(seam_mask)[0].astype(int).tolist())

            if scale_index == 0:
                base_seam_set = cur_seam
                prev_seam_set = cur_seam
                base_phase = phase.copy()

            assert base_seam_set is not None
            assert prev_seam_set is not None
            assert base_phase is not None

            seam_rows.append(
                {
                    "scale_index": int(scale_index),
                    "t": float(t),
                    "phase_column": str(observable_cols[phase_idx]),
                    "seam_quantile": float(cfg.seam_quantile),
                    "n_seam_proxy_nodes": int(len(cur_seam)),
                    "seam_proxy_jaccard_vs_base": float(jaccard(base_seam_set, cur_seam)),
                    "seam_proxy_jaccard_vs_previous": float(jaccard(prev_seam_set, cur_seam)),
                    "mean_seam_score": float(np.nanmean(seam_score)),
                    "max_seam_score": float(np.nanmax(seam_score)),
                    "mean_phase_contrast": float(np.nanmean(phase_contrast)),
                }
            )

            phase_rows.append(
                {
                    "scale_index": int(scale_index),
                    "t": float(t),
                    "phase_column": str(observable_cols[phase_idx]),
                    "phase_corr_vs_base": corr(base_phase, phase),
                    "phase_sign_agreement_vs_base": phase_sign_agreement(base_phase, phase),
                    "phase_mean": float(np.nanmean(phase)),
                    "phase_std": float(np.nanstd(phase)),
                    "phase_abs_mean": float(np.nanmean(np.abs(phase))),
                    "phase_zero_band_share_abs_lt_0p1": float(np.mean(np.abs(phase) < 0.1)),
                }
            )

            prev_seam_set = cur_seam

        node_df = pd.DataFrame(
            {
                id_col: ids,
                "scale_index": int(scale_index),
                "t": float(t),
                "geom_x": E[:, 0],
                "geom_y": E[:, 1],
                "energy": energy,
                "density_score": density,
                "mean_knn_observable_distance": mean_knn,
                "seam_proxy_score": seam_score,
                "is_seam_proxy": seam_mask.astype(int),
                "phase_contrast": phase_contrast,
            }
        )

        node_df = node_df.merge(ctx[context_cols], on=id_col, how="left")
        node_rows.append(node_df)

    node_all = pd.concat(node_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    topk_df = pd.DataFrame(topk_rows)
    seam_df = pd.DataFrame(seam_rows)
    phase_df = pd.DataFrame(phase_rows)

    return node_all, summary_df, topk_df, seam_df, phase_df


def write_report(
    cfg: Config,
    bundle: dict,
    summary_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    seam_df: pd.DataFrame,
    phase_df: pd.DataFrame,
) -> None:
    ids = bundle["ids"]
    observable_cols = bundle["observable_cols"]
    ts = bundle["t"]

    lines: list[str] = [
        "# OBS-076b — Observable-space geometry rebuild from scale-space",
        "",
        "## Scope",
        "",
        "OBS-076b rebuilds observable-space geometry from OBS-076a diffused fields.",
        "",
        "This is not canonical Fisher geometry rebuild. Seam and density structures are proxies.",
        "",
        "## Inputs",
        "",
        f"- bundle: `{cfg.bundle}`",
        f"- node_context: `{cfg.node_context if cfg.node_context else ''}`",
        f"- nodes: `{len(ids)}`",
        f"- observables: `{len(observable_cols)}`",
        f"- scales: `{len(ts)}`",
        "",
        "## Configuration",
        "",
        f"- k_density: `{cfg.k_density}`",
        f"- topk_values: `{','.join(str(k) for k in cfg.topk_values)}`",
        f"- seam_quantile: `{cfg.seam_quantile}`",
        f"- embedding: `{'metric_mds' if cfg.use_mds else 'classical_mds'}`",
        "",
        "## Scale geometry summary",
        "",
        "| scale_index | t | distance_corr_vs_base | embedding_corr_vs_base | embedding_fit_corr | mean_knn_distance | mean_energy | max_energy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            f"{int(row.scale_index)} | "
            f"{float(row.t):.6g} | "
            f"{float(row.observable_distance_corr_vs_base):.6g} | "
            f"{float(row.embedding_distance_corr_vs_base):.6g} | "
            f"{float(row.embedding_fit_corr):.6g} | "
            f"{float(row.mean_knn_observable_distance):.6g} | "
            f"{float(row.mean_energy):.6g} | "
            f"{float(row.max_energy):.6g} |"
        )

    lines.extend(["", "## Top-k geometry persistence", ""])

    if topk_df.empty:
        lines.append("No top-k rows were produced.")
    else:
        final_scale = int(topk_df["scale_index"].max())
        final = topk_df[topk_df["scale_index"] == final_scale]
        lines.extend(
            [
                "| metric | topk | final_jaccard_vs_base | final_jaccard_vs_previous |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in final.sort_values(["metric", "topk"]).itertuples(index=False):
            lines.append(
                "| "
                f"{row.metric} | "
                f"{int(row.topk)} | "
                f"{float(row.jaccard_vs_base):.6g} | "
                f"{float(row.jaccard_vs_previous):.6g} |"
            )

    lines.extend(["", "## Seam proxy persistence", ""])

    if seam_df.empty:
        lines.append("No phase column was found; seam proxy was not computed.")
    else:
        lines.extend(
            [
                "| scale_index | t | n_seam_proxy_nodes | jaccard_vs_base | jaccard_vs_previous | mean_phase_contrast |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in seam_df.itertuples(index=False):
            lines.append(
                "| "
                f"{int(row.scale_index)} | "
                f"{float(row.t):.6g} | "
                f"{int(row.n_seam_proxy_nodes)} | "
                f"{float(row.seam_proxy_jaccard_vs_base):.6g} | "
                f"{float(row.seam_proxy_jaccard_vs_previous):.6g} | "
                f"{float(row.mean_phase_contrast):.6g} |"
            )

    lines.extend(["", "## Phase proxy summary", ""])

    if phase_df.empty:
        lines.append("No phase proxy summary was produced.")
    else:
        lines.extend(
            [
                "| scale_index | t | phase_corr_vs_base | phase_sign_agreement_vs_base | phase_abs_mean | zero_band_share_abs_lt_0p1 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in phase_df.itertuples(index=False):
            lines.append(
                "| "
                f"{int(row.scale_index)} | "
                f"{float(row.t):.6g} | "
                f"{float(row.phase_corr_vs_base):.6g} | "
                f"{float(row.phase_sign_agreement_vs_base):.6g} | "
                f"{float(row.phase_abs_mean):.6g} | "
                f"{float(row.phase_zero_band_share_abs_lt_0p1):.6g} |"
            )

    lines.extend(
        [
            "",
            "## Output artifacts",
            "",
            "- `obs076b_node_geometry_by_scale.csv`",
            "- `obs076b_scale_geometry_summary.csv`",
            "- `obs076b_topk_geometry_persistence.csv`",
            "- `obs076b_seam_proxy_persistence.csv`",
            "- `obs076b_phase_proxy_summary.csv`",
            "",
            "## Interpretation guardrails",
            "",
            "- This is observable-space geometry, not canonical Fisher geometry.",
            "- Seam structures are proxies derived from diffused `signed_phase` when available.",
            "- Density is based on observable-space kNN distances at each scale.",
            "- Geometry persistence here is a robustness diagnostic, not a replacement for canonical seams or attractors.",
            "- OBS-076c/OBS-076b-v2 should connect this substrate back to canonical Fisher/geodesic rebuilds and path-family structures.",
            "",
        ]
    )

    (cfg.outdir / "obs076b_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(cfg: Config, bundle: dict) -> None:
    rows = [
        {
            "artifact": "obs076a_bundle",
            "path": str(cfg.bundle),
            "role": "input_diffusion_bundle",
            "status": "ok",
        },
        {
            "artifact": "node_context",
            "path": str(cfg.node_context) if cfg.node_context else "",
            "role": "optional_node_context",
            "status": "ok" if cfg.node_context else "not_provided",
        },
        {
            "artifact": "n_nodes",
            "path": "",
            "role": str(len(bundle["ids"])),
            "status": "observed",
        },
        {
            "artifact": "n_observables",
            "path": "",
            "role": str(len(bundle["observable_cols"])),
            "status": "observed",
        },
        {
            "artifact": "n_scales",
            "path": "",
            "role": str(len(bundle["t"])),
            "status": "observed",
        },
        {
            "artifact": "observable_cols",
            "path": "",
            "role": ",".join(str(c) for c in bundle["observable_cols"]),
            "status": "observed",
        },
    ]
    pd.DataFrame(rows).to_csv(cfg.outdir / "obs076b_input_manifest.csv", index=False)


def parse_topk(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("--topk must contain at least one positive integer")
    return vals


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="OBS-076b observable-space geometry rebuild from OBS-076a scale-space bundle."
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--node-context", default=None, type=Path)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--k-density", type=int, default=7)
    parser.add_argument("--topk", default="5,10,20")
    parser.add_argument("--seam-quantile", type=float, default=0.85)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--max-mds-iter", type=int, default=300)
    parser.add_argument(
        "--use-mds",
        action="store_true",
        help="Use sklearn metric MDS. Default is deterministic classical MDS.",
    )

    args = parser.parse_args()

    if not (0.0 < args.seam_quantile < 1.0):
        raise ValueError("--seam-quantile must be between 0 and 1")

    return Config(
        bundle=args.bundle,
        outdir=args.outdir,
        node_context=args.node_context,
        id_col=args.id_col,
        k_density=args.k_density,
        topk_values=parse_topk(args.topk),
        seam_quantile=args.seam_quantile,
        random_state=args.random_state,
        max_mds_iter=args.max_mds_iter,
        use_mds=bool(args.use_mds),
    )


def main() -> None:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(cfg.bundle)
    ctx = load_node_context(cfg.node_context, bundle["ids"], cfg.id_col)

    node_df, summary_df, topk_df, seam_df, phase_df = compute_scale_geometry(
        ids=bundle["ids"],
        observable_cols=bundle["observable_cols"],
        X_t=bundle["X_t"],
        ts=bundle["t"],
        ctx=ctx,
        cfg=cfg,
    )

    write_manifest(cfg, bundle)

    node_df.to_csv(cfg.outdir / "obs076b_node_geometry_by_scale.csv", index=False)
    summary_df.to_csv(cfg.outdir / "obs076b_scale_geometry_summary.csv", index=False)
    topk_df.to_csv(cfg.outdir / "obs076b_topk_geometry_persistence.csv", index=False)
    seam_df.to_csv(cfg.outdir / "obs076b_seam_proxy_persistence.csv", index=False)
    phase_df.to_csv(cfg.outdir / "obs076b_phase_proxy_summary.csv", index=False)

    write_report(
        cfg=cfg,
        bundle=bundle,
        summary_df=summary_df,
        topk_df=topk_df,
        seam_df=seam_df,
        phase_df=phase_df,
    )

    print("OBS-076b complete")
    print("wrote:", cfg.outdir / "obs076b_input_manifest.csv")
    print("wrote:", cfg.outdir / "obs076b_node_geometry_by_scale.csv")
    print("wrote:", cfg.outdir / "obs076b_scale_geometry_summary.csv")
    print("wrote:", cfg.outdir / "obs076b_topk_geometry_persistence.csv")
    print("wrote:", cfg.outdir / "obs076b_seam_proxy_persistence.csv")
    print("wrote:", cfg.outdir / "obs076b_phase_proxy_summary.csv")
    print("wrote:", cfg.outdir / "obs076b_report.md")


if __name__ == "__main__":
    main()
