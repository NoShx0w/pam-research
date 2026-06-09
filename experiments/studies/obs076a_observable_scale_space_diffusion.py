#!/usr/bin/env python3
"""
obs076a_observable_scale_space_diffusion.py

OBS-076a — Observable scale-space diffusion substrate

Purpose
-------
Promote the OBS-069 observable-diffusion pilot into a provenance-controlled
scale-space substrate for PAM robustness studies.

This script diffuses observable fields X over a declared graph distance D:

    X ∈ R^(N × P)

where rows are parameter / geometry nodes and columns are observable features.

It constructs a local self-tuning affinity graph,

    W_ij = exp(-D(i,j)^2 / (2 sigma_i sigma_j))

using k-nearest neighborhoods, builds the symmetric normalized Laplacian,

    L_sym = I - D^(-1/2) W D^(-1/2)

and emits

    X(t) = exp(-t L_sym) X

for a geometric scale ladder.

Scope discipline
----------------
OBS-076a does NOT rebuild Fisher geometry, seams, attractors, route families,
or response operators at each scale.

It creates the scale-space substrate required for OBS-076b.

Outputs
-------
outdir/
  obs076a_input_manifest.csv
  obs076a_scale_ladder.csv
  obs076a_graph_diagnostics.json
  obs076a_laplacian_spectrum.csv
  obs076a_diffusion_bundle.npz
  obs076a_diffused_observables_tXXX.csv        optional
  obs076a_observable_drift_summary.csv
  obs076a_topk_persistence_summary.csv
  obs076a_scale_space_report.md

Guardrail
---------
Persistence across diffusion scale is evidence of robustness under the declared
graph/field contract. Collapse under diffusion indicates scale-sensitivity,
not falsity.

This is observable-field diffusion, not image smoothing.
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


GRAPH_DISTANCE_KINDS = {
    "canonical_fisher",
    "canonical_geodesic",
    "canonical_mds_pilot",
    "observable_euclidean",
    "declared_other",
}


@dataclass(frozen=True)
class ScaleConfig:
    k: int
    sigma_rank: int
    n_scales: int
    t_min: float
    t_max: float
    graph_distance_kind: str


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

    non_numeric = [
        c for c in cols if not pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]
    # This check is intentionally soft because object columns containing numeric strings are allowed.
    # Numeric coercion happens later.

    return cols


def load_observables(
    path: Path,
    id_col: str,
    observable_cols: str | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"id column {id_col!r} not found in {path}")

    cols = parse_cols(observable_cols, df, id_col)

    ids = df[id_col].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        dupes = pd.Series(ids).value_counts()
        dupes = dupes[dupes > 1].head(10).to_dict()
        raise ValueError(f"id column {id_col!r} contains duplicate ids: {dupes}")

    X_raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(X_raw).any():
        raise ValueError("Observable matrix contains no finite values.")

    impute_rows = []
    X = X_raw.copy()

    for j, col_name in enumerate(cols):
        col = X[:, j]
        finite = np.isfinite(col)
        n_missing = int((~finite).sum())

        if finite.any():
            fill = float(np.nanmedian(col[finite]))
        else:
            fill = 0.0

        col[~finite] = fill
        X[:, j] = col

        impute_rows.append(
            {
                "observable": col_name,
                "n_missing_or_nonfinite": n_missing,
                "impute_value": fill,
                "raw_min": float(np.nanmin(X_raw[:, j])) if finite.any() else np.nan,
                "raw_max": float(np.nanmax(X_raw[:, j])) if finite.any() else np.nan,
                "raw_median": fill,
            }
        )

    impute_df = pd.DataFrame(impute_rows)
    return df, ids, cols, X, impute_df


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

    negative = finite_offdiag[finite_offdiag < 0]
    if negative.size:
        raise ValueError("Distance matrix contains negative off-diagonal distances.")

    return D


def load_distance_matrix(path: Path, ids: np.ndarray, id_col: str) -> np.ndarray:
    """
    Load a square or long-form distance matrix.

    Supported formats:

    1. Long CSV:
       source,target,distance

    2. Square CSV with id column:
       id_col,node_a,node_b,...

    3. Numeric square CSV:
       N x N, assumed to match observable row order.
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
            distance = pd.to_numeric(getattr(row, "distance"), errors="coerce")

            if not np.isfinite(distance):
                continue

            if source in index and target in index:
                i = index[source]
                j = index[target]
                D[i, j] = float(distance)
                D[j, i] = float(distance)

        return sanitize_distance_matrix(D)

    if id_col in df.columns:
        row_ids = df[id_col].astype(str).to_numpy()
        value_df = df.drop(columns=[id_col])

        missing_cols = [node_id for node_id in ids if node_id not in value_df.columns]
        if missing_cols:
            raise ValueError(
                "Square distance matrix column ids do not match observable ids. "
                f"First missing ids: {missing_cols[:10]}"
            )

        row_order = {node_id: i for i, node_id in enumerate(row_ids)}
        missing_rows = [node_id for node_id in ids if node_id not in row_order]
        if missing_rows:
            raise ValueError(
                "Square distance matrix row ids do not match observable ids. "
                f"First missing ids: {missing_rows[:10]}"
            )

        value_df = value_df.loc[:, ids]
        ordered_rows = [row_order[node_id] for node_id in ids]
        D = value_df.iloc[ordered_rows].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return sanitize_distance_matrix(D)

    numeric = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if numeric.shape != (n, n):
        raise ValueError(
            f"Numeric distance matrix shape {numeric.shape} does not match observables {(n, n)}."
        )

    return sanitize_distance_matrix(numeric)


def build_self_tuning_affinity(
    D: np.ndarray,
    *,
    k: int,
    sigma_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]

    if n < 2:
        raise ValueError("Need at least two nodes for graph diffusion.")

    k = min(max(1, int(k)), n - 1)
    sigma_rank = min(max(1, int(sigma_rank)), k)

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

    positive_sigmas = sigmas[sigmas > 0]
    fallback_sigma = float(np.median(positive_sigmas)) if positive_sigmas.size else 1.0
    sigmas[sigmas <= 0] = fallback_sigma

    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf

        order = np.argsort(row)
        neigh = [j for j in order if np.isfinite(row[j])][:k]

        for j in neigh:
            denom = max(2.0 * sigmas[i] * sigmas[j], EPS)
            wij = float(np.exp(-(D[i, j] ** 2) / denom))
            W[i, j] = max(W[i, j], wij)

    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)

    if np.count_nonzero(W) == 0:
        raise ValueError("Affinity graph has no nonzero edges.")

    return W, sigmas


def normalized_laplacian(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deg = W.sum(axis=1)

    inv_sqrt = np.zeros_like(deg)
    positive = deg > EPS
    inv_sqrt[positive] = 1.0 / np.sqrt(deg[positive])

    S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
    L = np.eye(W.shape[0]) - S

    # Symmetry guard.
    L = 0.5 * (L + L.T)
    return L, deg


def scale_ladder(n_scales: int, t_min: float, t_max: float) -> np.ndarray:
    if n_scales < 1:
        raise ValueError("--n-scales must be >= 1")
    if t_min <= 0 or t_max <= 0:
        raise ValueError("--t-min and --t-max must be positive")
    if t_max < t_min:
        raise ValueError("--t-max must be >= --t-min")

    if n_scales == 1:
        return np.array([t_min], dtype=float)

    return np.geomspace(t_min, t_max, n_scales)


def eigendiffuse_fields(
    X: np.ndarray,
    L: np.ndarray,
    ts: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(L)

    # Numerical guard. L_sym should be positive semidefinite.
    eigvals = np.maximum(eigvals, 0.0)

    coeff = eigvecs.T @ X
    fields = []

    for t in ts:
        decay = np.exp(-float(t) * eigvals)
        Xt = eigvecs @ (decay[:, None] * coeff)
        fields.append(Xt)

    return fields, eigvals, eigvecs


def row_norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(a * a, axis=1))


def matrix_corr_flat(A: np.ndarray, B: np.ndarray) -> float:
    a = A.reshape(-1)
    b = B.reshape(-1)

    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return np.nan

    a = a[finite]
    b = b[finite]

    if np.std(a) <= EPS or np.std(b) <= EPS:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def laplacian_energy(X: np.ndarray, L: np.ndarray) -> float:
    return float(np.trace(X.T @ L @ X))


def topk_set(values: np.ndarray, k: int) -> set[int]:
    finite = np.isfinite(values)
    idx = np.where(finite)[0]

    if idx.size == 0:
        return set()

    k = min(int(k), idx.size)
    order = idx[np.argsort(values[idx])[::-1]]

    return set(int(i) for i in order[:k])


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0

    union = a | b
    if not union:
        return 1.0

    return len(a & b) / len(union)


def summarize_drift(
    ts: np.ndarray,
    X0: np.ndarray,
    fields: list[np.ndarray],
    L: np.ndarray,
) -> pd.DataFrame:
    rows = []

    base_energy = laplacian_energy(X0, L)
    base_variance = float(np.mean(np.var(X0, axis=0)))
    base_norm = float(np.linalg.norm(X0))

    for idx, (t, Xt) in enumerate(zip(ts, fields)):
        diff = Xt - X0
        node_drift = row_norm(diff)

        energy = laplacian_energy(Xt, L)
        variance = float(np.mean(np.var(Xt, axis=0)))
        norm_xt = float(np.linalg.norm(Xt))

        rows.append(
            {
                "scale_index": idx,
                "t": float(t),
                "mean_node_l2_drift": float(np.mean(node_drift)),
                "median_node_l2_drift": float(np.median(node_drift)),
                "p90_node_l2_drift": float(np.quantile(node_drift, 0.9)),
                "max_node_l2_drift": float(np.max(node_drift)),
                "global_frobenius_drift": float(np.linalg.norm(diff)),
                "frobenius_norm": norm_xt,
                "frobenius_norm_ratio_vs_base": norm_xt / max(base_norm, EPS),
                "mean_observable_variance": variance,
                "variance_retained_ratio": variance / max(base_variance, EPS),
                "laplacian_energy": energy,
                "laplacian_energy_ratio_vs_base": energy / max(base_energy, EPS),
                "flat_corr_with_base": matrix_corr_flat(X0, Xt),
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
    Generic top-k persistence based on node observable magnitude.

    This is deliberately substrate-level. OBS-076b can replace/add named
    structures such as seam score, Lazarus score, criticality, attractor basin,
    or route-family membership.
    """
    base_energy = row_norm(X0)
    rows = []

    for k in topk_values:
        base = topk_set(base_energy, int(k))
        prev = base

        for idx, (t, Xt) in enumerate(zip(ts, fields)):
            cur = topk_set(row_norm(Xt), int(k))
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


def write_diffused_csvs(
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
        df.to_csv(out_dir / f"obs076a_diffused_observables_t{idx:03d}.csv", index=False)


def write_input_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    config: ScaleConfig,
    n_nodes: int,
    observable_cols: list[str],
    impute_df: pd.DataFrame,
) -> pd.DataFrame:
    manifest = pd.DataFrame(
        [
            {
                "artifact": "observables_csv",
                "path": args.observables_csv,
                "role": "observable_matrix_source",
                "status": "ok",
            },
            {
                "artifact": "distance_csv",
                "path": args.distance_csv,
                "role": "graph_distance_source",
                "status": "ok",
            },
            {
                "artifact": "graph_distance_kind",
                "path": "",
                "role": config.graph_distance_kind,
                "status": "declared",
            },
            {
                "artifact": "id_col",
                "path": "",
                "role": args.id_col,
                "status": "declared",
            },
            {
                "artifact": "observable_columns",
                "path": "",
                "role": ",".join(observable_cols),
                "status": f"n={len(observable_cols)}",
            },
            {
                "artifact": "n_nodes",
                "path": "",
                "role": str(n_nodes),
                "status": "observed",
            },
            {
                "artifact": "n_imputed_values",
                "path": "",
                "role": str(int(impute_df["n_missing_or_nonfinite"].sum())),
                "status": "observed",
            },
        ]
    )
    manifest.to_csv(out_dir / "obs076a_input_manifest.csv", index=False)
    impute_df.to_csv(out_dir / "obs076a_imputation_summary.csv", index=False)
    return manifest


def write_npz_bundle(
    out_dir: Path,
    ids: np.ndarray,
    observable_cols: list[str],
    X0: np.ndarray,
    fields: list[np.ndarray],
    ts: np.ndarray,
    W: np.ndarray,
    sigmas: np.ndarray,
    L: np.ndarray,
    degrees: np.ndarray,
    eigvals: np.ndarray,
) -> None:
    X_t = np.stack(fields, axis=0)

    np.savez_compressed(
        out_dir / "obs076a_diffusion_bundle.npz",
        ids=ids.astype(str),
        observable_cols=np.array(observable_cols, dtype=object),
        X0=X0,
        X_t=X_t,
        t=ts,
        W=W,
        sigmas=sigmas,
        L_sym=L,
        degrees=degrees,
        laplacian_eigvals=eigvals,
    )


def graph_components(W: np.ndarray) -> list[int]:
    n = W.shape[0]
    seen = np.zeros(n, dtype=bool)
    sizes: list[int] = []

    adjacency = [np.where(W[i] > 0)[0].tolist() for i in range(n)]

    for start in range(n):
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        size = 0

        while stack:
            i = stack.pop()
            size += 1

            for j in adjacency[i]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)

        sizes.append(size)

    return sorted(sizes, reverse=True)


def write_graph_diagnostics(
    out_dir: Path,
    config: ScaleConfig,
    W: np.ndarray,
    sigmas: np.ndarray,
    degrees: np.ndarray,
    eigvals: np.ndarray,
    n_observables: int,
) -> dict:
    binary_degrees = np.count_nonzero(W > 0, axis=1)
    components = graph_components(W)

    diagnostics = {
        "n_nodes": int(W.shape[0]),
        "n_observables": int(n_observables),
        "graph_distance_kind": config.graph_distance_kind,
        "n_undirected_edges": int(np.count_nonzero(np.triu(W > 0, k=1))),
        "mean_binary_degree": float(np.mean(binary_degrees)),
        "min_binary_degree": int(np.min(binary_degrees)),
        "max_binary_degree": int(np.max(binary_degrees)),
        "mean_weighted_degree": float(np.mean(degrees)),
        "min_weighted_degree": float(np.min(degrees)),
        "max_weighted_degree": float(np.max(degrees)),
        "n_connected_components": int(len(components)),
        "component_sizes": components,
        "median_sigma": float(np.median(sigmas)),
        "min_sigma": float(np.min(sigmas)),
        "max_sigma": float(np.max(sigmas)),
        "laplacian_eig_min": float(np.min(eigvals)),
        "laplacian_eig_max": float(np.max(eigvals)),
        "laplacian_eig_second": float(eigvals[1]) if len(eigvals) > 1 else np.nan,
        "config": {
            "k": config.k,
            "sigma_rank": config.sigma_rank,
            "n_scales": config.n_scales,
            "t_min": config.t_min,
            "t_max": config.t_max,
        },
    }

    (out_dir / "obs076a_graph_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "eig_index": np.arange(len(eigvals), dtype=int),
            "laplacian_eigenvalue": eigvals,
        }
    ).to_csv(out_dir / "obs076a_laplacian_spectrum.csv", index=False)

    return diagnostics


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    config: ScaleConfig,
    observable_cols: list[str],
    graph_diag: dict,
    drift_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# OBS-076a — Observable scale-space diffusion substrate",
        "",
        "## Scope",
        "",
        "OBS-076a diffuses observable fields over a declared graph distance and emits scale-parameterized observable matrices.",
        "",
        "It does not rebuild Fisher geometry, seams, attractors, response operators, or route-family structures at each scale.",
        "",
        "This run establishes the substrate for OBS-076b.",
        "",
        "## Inputs",
        "",
        f"- Observables: `{args.observables_csv}`",
        f"- Distance matrix/table: `{args.distance_csv}`",
        f"- Graph distance kind: `{config.graph_distance_kind}`",
        f"- ID column: `{args.id_col}`",
        f"- Observable count: `{len(observable_cols)}`",
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
        f"- nodes: `{graph_diag['n_nodes']}`",
        f"- observables: `{graph_diag['n_observables']}`",
        f"- undirected edges: `{graph_diag['n_undirected_edges']}`",
        f"- connected components: `{graph_diag['n_connected_components']}`",
        f"- component sizes: `{graph_diag['component_sizes']}`",
        f"- mean binary degree: `{graph_diag['mean_binary_degree']:.6g}`",
        f"- min binary degree: `{graph_diag['min_binary_degree']}`",
        f"- max binary degree: `{graph_diag['max_binary_degree']}`",
        f"- mean weighted degree: `{graph_diag['mean_weighted_degree']:.6g}`",
        f"- median self-tuning sigma: `{graph_diag['median_sigma']:.6g}`",
        f"- second Laplacian eigenvalue: `{graph_diag['laplacian_eig_second']:.6g}`",
        "",
        "## Drift / energy summary",
        "",
        "| scale_index | t | mean_node_l2_drift | variance_retained_ratio | laplacian_energy_ratio_vs_base | flat_corr_with_base |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in drift_df.itertuples(index=False):
        lines.append(
            "| "
            f"{int(row.scale_index)} | "
            f"{float(row.t):.6g} | "
            f"{float(row.mean_node_l2_drift):.6g} | "
            f"{float(row.variance_retained_ratio):.6g} | "
            f"{float(row.laplacian_energy_ratio_vs_base):.6g} | "
            f"{float(row.flat_corr_with_base):.6g} |"
        )

    lines.extend(
        [
            "",
            "## Top-k persistence",
            "",
            "| topk | final_jaccard_vs_base | final_jaccard_vs_previous |",
            "| ---: | ---: | ---: |",
        ]
    )

    if not topk_df.empty:
        for topk, sub in topk_df.groupby("topk"):
            final = sub.sort_values("scale_index").iloc[-1]
            lines.append(
                "| "
                f"{int(topk)} | "
                f"{float(final['jaccard_vs_base']):.6g} | "
                f"{float(final['jaccard_vs_previous']):.6g} |"
            )

    lines.extend(
        [
            "",
            "## Output artifacts",
            "",
            "- `obs076a_input_manifest.csv`",
            "- `obs076a_scale_ladder.csv`",
            "- `obs076a_graph_diagnostics.json`",
            "- `obs076a_laplacian_spectrum.csv`",
            "- `obs076a_diffusion_bundle.npz`",
            "- `obs076a_observable_drift_summary.csv`",
            "- `obs076a_topk_persistence_summary.csv`",
            "",
            "## Interpretation guardrails",
            "",
            "- This is observable-field diffusion, not plot smoothing.",
            "- Persistence across t indicates robustness under the declared graph-distance contract.",
            "- Collapse across t indicates scale-sensitivity, not falsity.",
            "- Graph-distance provenance matters; `canonical_mds_pilot` is weaker than `canonical_fisher` or `canonical_geodesic`.",
            "- OBS-076a does not claim that scale-diffused observables define canonical PAM geometry.",
            "- Geometry rebuilds and structural-object persistence belong to OBS-076b.",
            "",
        ]
    )

    (out_dir / "obs076a_scale_space_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OBS-076a observable scale-space diffusion substrate."
    )

    parser.add_argument("--observables-csv", required=True)
    parser.add_argument("--distance-csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--id-col", default="id")
    parser.add_argument(
        "--observable-cols",
        default=None,
        help="Comma-separated observable columns. Defaults to all numeric columns except id-col.",
    )

    parser.add_argument(
        "--graph-distance-kind",
        required=True,
        choices=sorted(GRAPH_DISTANCE_KINDS),
        help="Declared provenance of the distance matrix used to build the diffusion graph.",
    )

    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--sigma-rank", type=int, default=7)
    parser.add_argument("--n-scales", type=int, default=8)
    parser.add_argument("--t-min", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=10.0)

    parser.add_argument(
        "--topk",
        default="5,10,20",
        help="Comma-separated top-k node counts for generic magnitude persistence.",
    )
    parser.add_argument(
        "--write-csv-fields",
        action="store_true",
        help="Also write one CSV per diffused scale. The NPZ bundle is always written.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ScaleConfig(
        k=args.k,
        sigma_rank=args.sigma_rank,
        n_scales=args.n_scales,
        t_min=args.t_min,
        t_max=args.t_max,
        graph_distance_kind=args.graph_distance_kind,
    )

    obs_df, ids, observable_cols, X0, impute_df = load_observables(
        Path(args.observables_csv),
        id_col=args.id_col,
        observable_cols=args.observable_cols,
    )

    D = load_distance_matrix(
        Path(args.distance_csv),
        ids=ids,
        id_col=args.id_col,
    )

    if D.shape[0] != X0.shape[0]:
        raise ValueError(
            f"Distance matrix rows {D.shape[0]} do not match observable rows {X0.shape[0]}."
        )

    W, sigmas = build_self_tuning_affinity(
        D,
        k=config.k,
        sigma_rank=config.sigma_rank,
    )
    L, degrees = normalized_laplacian(W)

    ts = scale_ladder(
        n_scales=config.n_scales,
        t_min=config.t_min,
        t_max=config.t_max,
    )

    fields, eigvals, eigvecs = eigendiffuse_fields(X0, L, ts)

    pd.DataFrame(
        {
            "scale_index": np.arange(len(ts), dtype=int),
            "t": ts,
        }
    ).to_csv(out_dir / "obs076a_scale_ladder.csv", index=False)

    write_input_manifest(
        out_dir=out_dir,
        args=args,
        config=config,
        n_nodes=len(ids),
        observable_cols=observable_cols,
        impute_df=impute_df,
    )

    drift_df = summarize_drift(ts, X0, fields, L)
    drift_df.to_csv(out_dir / "obs076a_observable_drift_summary.csv", index=False)

    topk_values = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    topk_df = summarize_topk_persistence(ts, X0, fields, topk_values)
    topk_df.to_csv(out_dir / "obs076a_topk_persistence_summary.csv", index=False)

    graph_diag = write_graph_diagnostics(
        out_dir=out_dir,
        config=config,
        W=W,
        sigmas=sigmas,
        degrees=degrees,
        eigvals=eigvals,
        n_observables=len(observable_cols),
    )

    write_npz_bundle(
        out_dir=out_dir,
        ids=ids,
        observable_cols=observable_cols,
        X0=X0,
        fields=fields,
        ts=ts,
        W=W,
        sigmas=sigmas,
        L=L,
        degrees=degrees,
        eigvals=eigvals,
    )

    if args.write_csv_fields:
        write_diffused_csvs(
            out_dir=out_dir,
            ids=ids,
            observable_cols=observable_cols,
            ts=ts,
            fields=fields,
            id_col=args.id_col,
        )

    write_report(
        out_dir=out_dir,
        args=args,
        config=config,
        observable_cols=observable_cols,
        graph_diag=graph_diag,
        drift_df=drift_df,
        topk_df=topk_df,
    )

    print("OBS-076a complete")
    print("wrote:", out_dir / "obs076a_input_manifest.csv")
    print("wrote:", out_dir / "obs076a_scale_ladder.csv")
    print("wrote:", out_dir / "obs076a_graph_diagnostics.json")
    print("wrote:", out_dir / "obs076a_laplacian_spectrum.csv")
    print("wrote:", out_dir / "obs076a_diffusion_bundle.npz")
    print("wrote:", out_dir / "obs076a_observable_drift_summary.csv")
    print("wrote:", out_dir / "obs076a_topk_persistence_summary.csv")
    print("wrote:", out_dir / "obs076a_scale_space_report.md")


if __name__ == "__main__":
    main()
