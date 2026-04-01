from __future__ import annotations

"""
PAM Identity Metric

Estimate a first-pass local diagonal metric induced by identity distance.

Given local axis-aligned identity-distance measurements:
    d^2 ≈ g_rr (Δr)^2 + g_aa (Δα)^2

this module estimates, per node:

- g_rr
- g_ra = 0.0   (first-pass diagonal metric)
- g_aa
- identity_metric_det
- identity_metric_trace
- identity_metric_anisotropy
- identity_metric_valid

This is intentionally conservative and matches the current neighbor stencil.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IdentityMetricConfig:
    axis_tol: float = 1e-12
    min_samples_per_axis: int = 1


def load_identity_metric_inputs(
    *,
    identity_nodes_csv: str | Path,
    identity_edges_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(identity_nodes_csv)
    edges = pd.read_csv(identity_edges_csv)

    required_nodes = {"node_id", "i", "j", "r", "alpha", "identity_spin"}
    required_edges = {
        "src_node_id",
        "dst_node_id",
        "src_r",
        "src_alpha",
        "dst_r",
        "dst_alpha",
        "identity_distance",
    }

    missing_nodes = required_nodes - set(nodes.columns)
    if missing_nodes:
        raise ValueError(f"identity nodes CSV missing required columns: {sorted(missing_nodes)}")

    missing_edges = required_edges - set(edges.columns)
    if missing_edges:
        raise ValueError(f"identity edges CSV missing required columns: {sorted(missing_edges)}")

    nodes = nodes.copy()
    edges = edges.copy()

    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64").astype(str)
    edges["src_node_id"] = pd.to_numeric(edges["src_node_id"], errors="coerce").astype("Int64").astype(str)
    edges["dst_node_id"] = pd.to_numeric(edges["dst_node_id"], errors="coerce").astype("Int64").astype(str)

    numeric_cols_nodes = ["i", "j", "r", "alpha", "identity_spin"]
    for col in numeric_cols_nodes:
        nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

    numeric_cols_edges = ["src_r", "src_alpha", "dst_r", "dst_alpha", "identity_distance"]
    for col in numeric_cols_edges:
        edges[col] = pd.to_numeric(edges[col], errors="coerce")

    return nodes, edges


def _incident_edge_samples(
    node_id: str,
    edges: pd.DataFrame,
) -> list[tuple[float, float, float]]:
    """
    Returns a list of local samples relative to the center node:
        (delta_r, delta_alpha, identity_distance)
    """
    rows: list[tuple[float, float, float]] = []

    src_rows = edges[edges["src_node_id"] == node_id]
    for _, row in src_rows.iterrows():
        rows.append(
            (
                float(row["dst_r"] - row["src_r"]),
                float(row["dst_alpha"] - row["src_alpha"]),
                float(row["identity_distance"]),
            )
        )

    dst_rows = edges[edges["dst_node_id"] == node_id]
    for _, row in dst_rows.iterrows():
        rows.append(
            (
                float(row["src_r"] - row["dst_r"]),
                float(row["src_alpha"] - row["dst_alpha"]),
                float(row["identity_distance"]),
            )
        )

    return rows


def estimate_local_identity_metric(
    node_id: str,
    edges: pd.DataFrame,
    *,
    config: IdentityMetricConfig | None = None,
) -> dict[str, float | int | bool]:
    if config is None:
        config = IdentityMetricConfig()

    samples = _incident_edge_samples(node_id, edges)

    rr_samples: list[float] = []
    aa_samples: list[float] = []

    for delta_r, delta_alpha, d in samples:
        d2 = float(d) ** 2

        if abs(delta_r) > config.axis_tol and abs(delta_alpha) <= config.axis_tol:
            rr_samples.append(d2 / (delta_r ** 2))

        elif abs(delta_alpha) > config.axis_tol and abs(delta_r) <= config.axis_tol:
            aa_samples.append(d2 / (delta_alpha ** 2))

    g_rr = float(np.mean(rr_samples)) if len(rr_samples) >= config.min_samples_per_axis else np.nan
    g_aa = float(np.mean(aa_samples)) if len(aa_samples) >= config.min_samples_per_axis else np.nan
    g_ra = 0.0

    valid = bool(np.isfinite(g_rr) and np.isfinite(g_aa) and g_rr >= 0 and g_aa >= 0)

    det = float(g_rr * g_aa) if valid else np.nan
    trace = float(g_rr + g_aa) if valid else np.nan

    if valid:
        lo = min(g_rr, g_aa)
        hi = max(g_rr, g_aa)
        anisotropy = float(hi / lo) if lo > config.axis_tol else np.inf
    else:
        anisotropy = np.nan

    return {
        "node_id": node_id,
        "identity_g_rr": g_rr,
        "identity_g_ra": g_ra,
        "identity_g_aa": g_aa,
        "identity_metric_det": det,
        "identity_metric_trace": trace,
        "identity_metric_anisotropy": anisotropy,
        "identity_metric_valid": int(valid),
        "identity_metric_n_rr": int(len(rr_samples)),
        "identity_metric_n_aa": int(len(aa_samples)),
    }


def build_identity_metric_table(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    config: IdentityMetricConfig | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []

    for node_id in nodes["node_id"].astype(str):
        rows.append(
            estimate_local_identity_metric(
                node_id=node_id,
                edges=edges,
                config=config,
            )
        )

    metric_df = pd.DataFrame(rows)

    out = (
        nodes.merge(metric_df, on="node_id", how="left")
        .sort_values(["i", "j"])
        .reset_index(drop=True)
    )
    return out
