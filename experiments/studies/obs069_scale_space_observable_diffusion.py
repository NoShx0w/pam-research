#!/usr/bin/env python3
"""
obs069_scale_space_observable_diffusion.py

OBS-069 — Scale-space observable diffusion pilot

Purpose
-------
Construct a PAM scale-space robustness branch by diffusing observable fields
over the canonical-MDS pilot graph.

This script does not smooth figures and does not replace the canonical pipeline.
It creates scale-parameterized observable fields X(t), then writes simple
persistence/drift summaries for later downstream geometry recomputation.

Core contract
-------------
Given:

    X ∈ R^(N × D)
        observable matrix over N parameter/corpus nodes

    D_F ∈ R^(N × N)
        canonical Fisher/geodesic dissimilarity matrix over the same nodes

Build a local self-tuning affinity graph:

    W_ij = exp(-D_F(i,j)^2 / (2 sigma_i sigma_j))

for j in kNN(i), symmetrize W, compute:

    L_sym = I - D^(-1/2) W D^(-1/2)

and diffuse:

    X(t) = exp(-t L_sym) X

via eigendecomposition of L_sym.

Outputs
-------
- obs069_scale_ladder.csv
- obs069_diffused_observables_tXXX.csv
- obs069_observable_drift_summary.csv
- obs069_topk_persistence_summary.csv
- obs069_scale_space_report.md

Guardrail
---------
This is observable-field diffusion over an observed reduced/canonical graph.
Persistence across t is a robustness diagnostic, not proof of ontological
fundamentality. Collapse under t indicates scale-sensitivity, not falsity.
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
class ScaleConfig:
    k: int
    sigma_rank: int
    n_scales: int
    t_min: float
    t_max: float


def parse_cols(raw: str | None, df: pd.DataFrame, id_col: str) -> list[str]:
    if raw:
        cols = [c.strip() for c in raw.split(",") if c.strip()]
    else:
        cols = [
            c
            for c in df.columns
            if c != id_col and pd.api.types.is_numeric_dtype(df[c])
        ]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Observable columns missing from input: {missing}")

    if not cols:
        raise ValueError("No observable columns selected.")

    return cols


def load_observables(path: Path, id_col: str, observable_cols: str | None):
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"id column {id_col!r} not found in {path}")

    cols = parse_cols(observable_cols, df, id_col)

    ids = df[id_col].astype(str).to_numpy()
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(X).any():
        raise ValueError("Observable matrix contains no finite values.")

    # Columnwise finite imputation. This is conservative and documented in report.
    X_imp = X.copy()
    for j in range(X_imp.shape[1]):
        col = X_imp[:, j]
        finite = np.isfinite(col)
        fill = float(np.nanmedian(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        X_imp[:, j] = col

    return df, ids, cols, X_imp


def load_distance_matrix(path: Path, ids: np.ndarray, id_col: str) -> np.ndarray:
    """
    Load a square distance matrix.

    Supported formats:
    1. Square CSV with first column equal to id_col and remaining columns as ids.
    2. Square numeric CSV with shape N x N and same row order as observables.
    3. Long CSV with columns source,target,distance.
    """
    df = pd.read_csv(path)
    n = len(ids)

    long_candidates = {"source", "target", "distance"}
    if long_candidates.issubset(df.columns):
        index = {node_id: i for i, node_id in enumerate(ids)}
        D = np.full((n, n), np.inf, dtype=float)
        np.fill_diagonal(D, 0.0)

        for row in df.itertuples(index=False):
            source = str(getattr(row, "source"))
            target = str(getattr(row, "target"))
            distance = float(getattr(row, "distance"))
            if source in index and target in index:
                i = index[source]
                j = index[target]
                D[i, j] = distance
                D[j, i] = distance

        if not np.isfinite(D).any():
            raise ValueError("Long distance table produced no finite distances.")

        return D

    if id_col in df.columns:
        row_ids = df[id_col].astype(str).to_numpy()
        value_df = df.drop(columns=[id_col])

        missing_cols = [node_id for node_id in ids if node_id not in value_df.columns]
        if missing_cols:
            raise ValueError(
                "Square distance matrix column ids do not match observable ids. "
                f"First missing ids: {missing_cols[:5]}"
            )

        value_df = value_df.loc[:, ids]
        row_order = {node_id: i for i, node_id in enumerate(row_ids)}
        missing_rows = [node_id for node_id in ids if node_id not in row_order]
        if missing_rows:
            raise ValueError(
                "Square distance matrix row ids do not match observable ids. "
                f"First missing ids: {missing_rows[:5]}"
            )

        ordered_rows = [row_order[node_id] for node_id in ids]
        D = value_df.iloc[ordered_rows].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return sanitize_distance_matrix(D)

    numeric = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if numeric.shape != (n, n):
        raise ValueError(
            f"Numeric distance matrix shape {numeric.shape} does not match observables {(n, n)}."
        )

    return sanitize_distance_matrix(numeric)


def sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square; got {D.shape}")

    D = np.where(np.isfinite(D), D, np.inf)
    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0.0)

    finite_offdiag = D[np.isfinite(D) & ~np.eye(D.shape[0], dtype=bool)]
    if finite_offdiag.size == 0:
        raise ValueError("Distance matrix has no finite off-diagonal distances.")

    return D


def build_self_tuning_affinity(D: np.ndarray, *, k: int, sigma_rank: int) -> tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least two nodes for graph diffusion.")

    k = min(max(1, k), n - 1)
    sigma_rank = min(max(1, sigma_rank), k)

    W = np.zeros((n, n), dtype=float)
    sigmas = np.zeros(n, dtype=float)

    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        order = np.argsort(row)
        neigh = [j for j in order if np.isfinite(row[j])][:k]

        if not neigh:
            continue

        sigma_idx = neigh[min(sigma_rank - 1, len(neigh) - 1)]
        sigma = D[i, sigma_idx]
        if not np.isfinite(sigma) or sigma <= 0:
            finite_pos = row[np.isfinite(row) & (row > 0)]
            sigma = float(np.nanmedian(finite_pos)) if finite_pos.size else 1.0

        sigmas[i] = max(float(sigma), EPS)

    # Fill any isolated sigma values with median positive sigma.
    positive_sigmas = sigmas[sigmas > 0]
    fallback_sigma = float(np.median(positive_sigmas)) if positive_sigmas.size else 1.0
    sigmas[sigmas <= 0] = fallback_sigma

    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        order = np.argsort(row)
        neigh = [j for j in order if np.isfinite(row[j])][:k]

        for j in neigh:
            denom = 2.0 * sigmas[i] * sigmas[j]
            wij = np.exp(-(D[i, j] ** 2) / max(denom, EPS))
            W[i, j] = max(W[i, j], wij)

    # Symmetrize kNN graph.
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)

    return W, sigmas


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    positive = deg > EPS
    inv_sqrt[positive] = 1.0 / np.sqrt(deg[positive])

    S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
    L = np.eye(W.shape[0]) - S

    # Symmetry guard against numerical noise.
    return 0.5 * (L + L.T)


def scale_ladder(n_scales: int, t_min: float, t_max: float) -> np.ndarray:
    if n_scales < 1:
        raise ValueError("--n-scales must be >= 1")
    if t_min <= 0 or t_max <= 0:
        raise ValueError("--t-min and --t-max must be positive")
    if n_scales == 1:
        return np.array([t_min], dtype=float)
    return np.geomspace(t_min, t_max, n_scales)


def diffuse_fields(X: np.ndarray, L: np.ndarray, ts: np.ndarray) -> list[np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.maximum(eigvals, 0.0)

    fields = []
    coeff = eigvecs.T @ X
    for t in ts:
        decay = np.exp(-float(t) * eigvals)
        Xt = eigvecs @ (decay[:, None] * coeff)
        fields.append(Xt)

    return fields


def row_norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(a * a, axis=1))


def topk_set(values: np.ndarray, k: int) -> set[int]:
    finite = np.isfinite(values)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return set()
    k = min(k, idx.size)
    order = idx[np.argsort(values[idx])[::-1]]
    return set(int(i) for i in order[:k])


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def write_diffused_fields(
    out_dir: Path,
    ids: np.ndarray,
    observable_cols: list[str],
    ts: np.ndarray,
    fields: list[np.ndarray],
    id_col: str,
) -> None:
    for idx, (t, Xt) in enumerate(zip(ts, fields)):
        df = pd.DataFrame(Xt, columns=observable_cols)
        df.insert(0, id_col, ids)
        df.insert(1, "scale_index", idx)
        df.insert(2, "t", float(t))
        df.to_csv(out_dir / f"obs069_diffused_observables_t{idx:03d}.csv", index=False)


def summarize_drift(
    ids: np.ndarray,
    ts: np.ndarray,
    X0: np.ndarray,
    fields: list[np.ndarray],
    id_col: str,
) -> pd.DataFrame:
    rows = []
    for idx, (t, Xt) in enumerate(zip(ts, fields)):
        diff = Xt - X0
        node_drift = row_norm(diff)
        rows.append(
            {
                "scale_index": idx,
                "t": float(t),
                "mean_node_l2_drift": float(np.mean(node_drift)),
                "median_node_l2_drift": float(np.median(node_drift)),
                "p90_node_l2_drift": float(np.quantile(node_drift, 0.9)),
                "max_node_l2_drift": float(np.max(node_drift)),
                "global_frobenius_drift": float(np.linalg.norm(diff)),
                "mean_observable_variance": float(np.mean(np.var(Xt, axis=0))),
            }
        )
    return pd.DataFrame(rows)


def summarize_topk_persistence(
    ts: np.ndarray,
    X0: np.ndarray,
    fields: list[np.ndarray],
    topk_values: Iterable[int],
) -> pd.DataFrame:
    """
    Track whether high-energy / high-magnitude nodes remain high-magnitude
    after diffusion. This is intentionally generic for v1.

    Later versions can replace magnitude with Lazarus score, seam score,
    curvature, gateway score, etc.
    """
    base_energy = row_norm(X0)
    rows = []

    for k in topk_values:
        base = topk_set(base_energy, int(k))
        prev = base

        for idx, (t, Xt) in enumerate(zip(ts, fields)):
            energy = row_norm(Xt)
            cur = topk_set(energy, int(k))
            rows.append(
                {
                    "topk": int(k),
                    "scale_index": idx,
                    "t": float(t),
                    "jaccard_vs_base": float(jaccard(base, cur)),
                    "jaccard_vs_previous": float(jaccard(prev, cur)),
                }
            )
            prev = cur

    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    config: ScaleConfig,
    n_nodes: int,
    observable_cols: list[str],
    W: np.ndarray,
    sigmas: np.ndarray,
    drift_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> None:
    n_edges = int(np.count_nonzero(np.triu(W > 0, k=1)))
    degrees = np.count_nonzero(W > 0, axis=1)

    lines = [
        "# OBS-069 — Scale-space observable diffusion pilot",
        "",
        "## Purpose",
        "",
        "OBS-069 constructs a scale-space robustness branch by diffusing observable fields over the canonical-MDS pilot graph.",
        "",
        "It diffuses observables, not rendered images, and does not replace the canonical geometry pipeline.",
        "",
        "## Inputs",
        "",
        f"- Observables: `{args.observables_csv}`",
        f"- Distance matrix/table: `{args.distance_csv}`",
        f"- ID column: `{args.id_col}`",
        f"- Observable columns: `{', '.join(observable_cols)}`",
        "",
        "## Configuration",
        "",
        f"- k nearest neighbors: `{config.k}`",
        f"- sigma rank: `{config.sigma_rank}`",
        f"- n scales: `{config.n_scales}`",
        f"- t_min: `{config.t_min}`",
        f"- t_max: `{config.t_max}`",
        "",
        "## Graph diagnostics",
        "",
        f"- nodes: `{n_nodes}`",
        f"- undirected edges: `{n_edges}`",
        f"- mean graph degree: `{float(np.mean(degrees)):.3f}`",
        f"- min graph degree: `{int(np.min(degrees))}`",
        f"- max graph degree: `{int(np.max(degrees))}`",
        f"- median self-tuning sigma: `{float(np.median(sigmas)):.6g}`",
        "",
        "## Drift summary",
        "",
    ]

    for row in drift_df.itertuples(index=False):
        lines.extend(
            [
                f"### t{int(row.scale_index):03d} — t={float(row.t):.6g}",
                "",
                f"- mean node L2 drift: `{float(row.mean_node_l2_drift):.6g}`",
                f"- median node L2 drift: `{float(row.median_node_l2_drift):.6g}`",
                f"- p90 node L2 drift: `{float(row.p90_node_l2_drift):.6g}`",
                f"- max node L2 drift: `{float(row.max_node_l2_drift):.6g}`",
                f"- mean observable variance: `{float(row.mean_observable_variance):.6g}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Top-k persistence",
            "",
        ]
    )

    for topk, sub in topk_df.groupby("topk"):
        final = sub.sort_values("scale_index").iloc[-1]
        lines.extend(
            [
                f"- top-{int(topk)} final Jaccard vs base: `{float(final['jaccard_vs_base']):.6g}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "Scale persistence indicates robustness of observable-field structure under graph diffusion.",
            "Scale collapse indicates scale-sensitivity, not falsity.",
            "This pilot does not recompute the full Fisher geometry, seams, attractors, or symbolic route structures at each scale.",
            "",
            "## Result",
            "",
            "OBS-069 v1 produces scale-parameterized observable fields and first-pass drift/persistence summaries for later multiscale geometry recomputation.",
            "",
        ]
    )

    (out_dir / "obs069_scale_space_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observables-csv", required=True)
    parser.add_argument("--distance-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="id")
    parser.add_argument(
        "--observable-cols",
        default=None,
        help="Comma-separated observable columns. Defaults to all numeric columns except id-col.",
    )
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--sigma-rank", type=int, default=7)
    parser.add_argument("--n-scales", type=int, default=8)
    parser.add_argument("--t-min", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument(
        "--topk",
        default="5,10,20",
        help="Comma-separated top-k node counts for generic magnitude-persistence summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ScaleConfig(
        k=args.k,
        sigma_rank=args.sigma_rank,
        n_scales=args.n_scales,
        t_min=args.t_min,
        t_max=args.t_max,
    )

    obs_df, ids, observable_cols, X = load_observables(
        Path(args.observables_csv),
        id_col=args.id_col,
        observable_cols=args.observable_cols,
    )
    D = load_distance_matrix(Path(args.distance_csv), ids=ids, id_col=args.id_col)

    if D.shape[0] != X.shape[0]:
        raise ValueError(f"Distance matrix rows {D.shape[0]} do not match X rows {X.shape[0]}")

    W, sigmas = build_self_tuning_affinity(D, k=config.k, sigma_rank=config.sigma_rank)
    L = normalized_laplacian(W)
    ts = scale_ladder(config.n_scales, config.t_min, config.t_max)
    fields = diffuse_fields(X, L, ts)

    scale_df = pd.DataFrame(
        {
            "scale_index": list(range(len(ts))),
            "t": ts,
        }
    )
    scale_df.to_csv(out_dir / "obs069_scale_ladder.csv", index=False)

    write_diffused_fields(out_dir, ids, observable_cols, ts, fields, args.id_col)

    drift_df = summarize_drift(ids, ts, X, fields, args.id_col)
    drift_df.to_csv(out_dir / "obs069_observable_drift_summary.csv", index=False)

    topk_values = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    topk_df = summarize_topk_persistence(ts, X, fields, topk_values)
    topk_df.to_csv(out_dir / "obs069_topk_persistence_summary.csv", index=False)

    graph_diag = {
        "n_nodes": int(W.shape[0]),
        "n_observables": int(X.shape[1]),
        "n_undirected_edges": int(np.count_nonzero(np.triu(W > 0, k=1))),
        "mean_degree": float(np.mean(np.count_nonzero(W > 0, axis=1))),
        "min_degree": int(np.min(np.count_nonzero(W > 0, axis=1))),
        "max_degree": int(np.max(np.count_nonzero(W > 0, axis=1))),
        "median_sigma": float(np.median(sigmas)),
        "config": {
            "k": config.k,
            "sigma_rank": config.sigma_rank,
            "n_scales": config.n_scales,
            "t_min": config.t_min,
            "t_max": config.t_max,
        },
    }
    (out_dir / "obs069_graph_diagnostics.json").write_text(
        json.dumps(graph_diag, indent=2),
        encoding="utf-8",
    )

    write_report(
        out_dir=out_dir,
        args=args,
        config=config,
        n_nodes=len(ids),
        observable_cols=observable_cols,
        W=W,
        sigmas=sigmas,
        drift_df=drift_df,
        topk_df=topk_df,
    )

    print("OBS-069 complete")
    print("wrote:", out_dir / "obs069_scale_ladder.csv")
    print("wrote:", out_dir / "obs069_observable_drift_summary.csv")
    print("wrote:", out_dir / "obs069_topk_persistence_summary.csv")
    print("wrote:", out_dir / "obs069_graph_diagnostics.json")
    print("wrote:", out_dir / "obs069_scale_space_report.md")


if __name__ == "__main__":
    main()
