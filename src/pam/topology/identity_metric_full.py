from __future__ import annotations

"""
PAM Identity Metric (Full Local Quadratic Form)

Estimate a first-pass full local metric induced by identity distance using
a small lattice patch around each node.

Model:
    d^2 ≈ g_rr (Δr)^2 + 2 g_ra (Δr)(Δα) + g_aa (Δα)^2

where:
- d is identity distance between the center node identity graph and a local patch node
- Δr, Δα are parameter displacements relative to the center node

Outputs per node:
- identity_g_rr
- identity_g_ra
- identity_g_aa
- identity_metric_det
- identity_metric_trace
- identity_metric_eig1
- identity_metric_eig2
- identity_metric_cond
- identity_metric_valid
- identity_metric_n_samples
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pam.topology.identity import IdentityGraph, identity_distance


@dataclass(frozen=True)
class IdentityMetricFullConfig:
    patch_radius: int = 1
    ridge: float = 1e-8
    min_samples: int = 3
    normalized_distance: bool = False


def load_identity_metric_full_inputs(
    *,
    identity_nodes_csv: str | Path,
    identity_proxy_nodes_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load node metadata needed to build local patch stencils.

    Required columns:
    - node_id
    - i
    - j
    - r
    - alpha

    identity_nodes_csv is typically:
      outputs/fim_identity/identity_field_nodes.csv
    """
    df = pd.read_csv(identity_nodes_csv).copy()

    required = {"node_id", "i", "j", "r", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"identity nodes CSV missing required columns: {sorted(missing)}"
        )

    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("Int64").astype(str)
    for col in ["i", "j"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    for col in ["r", "alpha"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _build_grid_lookup(nodes_df: pd.DataFrame) -> dict[tuple[int, int], dict]:
    return {
        (int(row["i"]), int(row["j"])): row
        for _, row in nodes_df.iterrows()
    }


def _local_patch_rows(
    center_row: dict,
    grid_lookup: dict[tuple[int, int], dict],
    *,
    patch_radius: int,
) -> list[dict]:
    ci = int(center_row["i"])
    cj = int(center_row["j"])

    out: list[dict] = []
    for di in range(-patch_radius, patch_radius + 1):
        for dj in range(-patch_radius, patch_radius + 1):
            if di == 0 and dj == 0:
                continue
            key = (ci + di, cj + dj)
            if key in grid_lookup:
                out.append(grid_lookup[key])
    return out


def _fit_full_quadratic_metric(
    samples: list[tuple[float, float, float]],
    *,
    ridge: float,
    min_samples: int,
) -> dict[str, float | int]:
    """
    samples: list of (delta_r, delta_alpha, d)

    Fits:
        d^2 ≈ g_rr Δr^2 + 2 g_ra Δr Δα + g_aa Δα^2
    """
    if len(samples) < min_samples:
        return {
            "identity_g_rr": np.nan,
            "identity_g_ra": np.nan,
            "identity_g_aa": np.nan,
            "identity_metric_det": np.nan,
            "identity_metric_trace": np.nan,
            "identity_metric_eig1": np.nan,
            "identity_metric_eig2": np.nan,
            "identity_metric_cond": np.nan,
            "identity_metric_valid": 0,
            "identity_metric_n_samples": int(len(samples)),
        }

    X = []
    y = []

    for delta_r, delta_alpha, d in samples:
        X.append(
            [
                delta_r ** 2,
                2.0 * delta_r * delta_alpha,
                delta_alpha ** 2,
            ]
        )
        y.append(d ** 2)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # ridge-regularized least squares
    XtX = X.T @ X
    Xty = X.T @ y
    reg = ridge * np.eye(3, dtype=float)

    try:
        beta = np.linalg.solve(XtX + reg, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    g_rr = float(beta[0])
    g_ra = float(beta[1])
    g_aa = float(beta[2])

    G = np.array([[g_rr, g_ra], [g_ra, g_aa]], dtype=float)

    try:
        eigvals = np.linalg.eigvalsh(G)
        eig1, eig2 = float(eigvals[0]), float(eigvals[1])
    except np.linalg.LinAlgError:
        eig1, eig2 = np.nan, np.nan

    det = float(np.linalg.det(G)) if np.all(np.isfinite(G)) else np.nan
    trace = float(np.trace(G)) if np.all(np.isfinite(G)) else np.nan

    valid = bool(
        np.isfinite(g_rr)
        and np.isfinite(g_ra)
        and np.isfinite(g_aa)
        and np.isfinite(det)
        and np.isfinite(eig1)
        and np.isfinite(eig2)
        and eig1 > 0
        and eig2 > 0
    )

    if valid:
        lo = min(eig1, eig2)
        hi = max(eig1, eig2)
        cond = float(hi / lo) if lo > 0 else np.inf
    else:
        cond = np.nan

    return {
        "identity_g_rr": g_rr,
        "identity_g_ra": g_ra,
        "identity_g_aa": g_aa,
        "identity_metric_det": det,
        "identity_metric_trace": trace,
        "identity_metric_eig1": eig1,
        "identity_metric_eig2": eig2,
        "identity_metric_cond": cond,
        "identity_metric_valid": int(valid),
        "identity_metric_n_samples": int(len(samples)),
    }


def estimate_full_identity_metric_table(
    nodes_df: pd.DataFrame,
    identity_graphs: dict[str, IdentityGraph],
    *,
    config: IdentityMetricFullConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = IdentityMetricFullConfig()

    work = nodes_df.copy()
    work["node_id"] = pd.to_numeric(work["node_id"], errors="coerce").astype("Int64").astype(str)
    grid_lookup = _build_grid_lookup(work)

    rows: list[dict] = []

    for _, center_row in work.iterrows():
        center_id = str(center_row["node_id"])
        center_graph = identity_graphs[center_id]

        patch_rows = _local_patch_rows(
            center_row,
            grid_lookup,
            patch_radius=config.patch_radius,
        )

        samples: list[tuple[float, float, float]] = []

        for nbr_row in patch_rows:
            nbr_id = str(nbr_row["node_id"])
            nbr_graph = identity_graphs[nbr_id]

            delta_r = float(nbr_row["r"] - center_row["r"])
            delta_alpha = float(nbr_row["alpha"] - center_row["alpha"])

            d = float(
                identity_distance(
                    center_graph,
                    nbr_graph,
                    normalized=config.normalized_distance,
                )
            )
            samples.append((delta_r, delta_alpha, d))

        metric = _fit_full_quadratic_metric(
            samples,
            ridge=config.ridge,
            min_samples=config.min_samples,
        )

        out = {
            "node_id": center_id,
            "i": int(center_row["i"]),
            "j": int(center_row["j"]),
            "r": float(center_row["r"]),
            "alpha": float(center_row["alpha"]),
        }
        out.update(metric)
        rows.append(out)

    return pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)
