from __future__ import annotations

"""
PAM Identity Transport / Holonomy

First-pass local loop residuals for structural identity.

This module avoids strong formal transport claims and instead computes
cell-based path inconsistency over elementary grid loops.

For each cell with corners:

    A = (i, j)
    B = (i, j+1)
    C = (i+1, j+1)
    D = (i+1, j)

we compare two paths from A to C:

    path_ab_bc = d(A, B) + d(B, C)
    path_ad_dc = d(A, D) + d(D, C)

and define:

    holonomy_residual = path_ab_bc - path_ad_dc

This is a first-pass, loop-based path-dependence observable.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pam.topology.identity import IdentityGraph, identity_distance


@dataclass(frozen=True)
class IdentityHolonomyConfig:
    normalized_distance: bool = True


def load_identity_transport_nodes(
    *,
    identity_nodes_csv: str | Path,
) -> pd.DataFrame:
    df = pd.read_csv(identity_nodes_csv).copy()

    required = {"node_id", "i", "j", "r", "alpha", "identity_spin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"identity nodes CSV missing required columns: {sorted(missing)}"
        )

    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("Int64").astype(str)
    for col in ["i", "j"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    for col in ["r", "alpha", "identity_spin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _grid_lookup(nodes_df: pd.DataFrame) -> dict[tuple[int, int], dict]:
    return {
        (int(row["i"]), int(row["j"])): row
        for _, row in nodes_df.iterrows()
    }


def build_identity_holonomy_table(
    nodes_df: pd.DataFrame,
    identity_graphs: dict[str, IdentityGraph],
    *,
    config: IdentityHolonomyConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = IdentityHolonomyConfig()

    work = nodes_df.copy()
    work["node_id"] = pd.to_numeric(work["node_id"], errors="coerce").astype("Int64").astype(str)
    lookup = _grid_lookup(work)

    max_i = int(work["i"].max())
    max_j = int(work["j"].max())

    rows: list[dict[str, float | str | int]] = []

    for i in range(max_i):
        for j in range(max_j):
            corners = {
                "A": lookup.get((i, j)),
                "B": lookup.get((i, j + 1)),
                "C": lookup.get((i + 1, j + 1)),
                "D": lookup.get((i + 1, j)),
            }
            if any(v is None for v in corners.values()):
                continue

            A, B, C, D = corners["A"], corners["B"], corners["C"], corners["D"]

            gA = identity_graphs[str(A["node_id"])]
            gB = identity_graphs[str(B["node_id"])]
            gC = identity_graphs[str(C["node_id"])]
            gD = identity_graphs[str(D["node_id"])]

            d_ab = float(identity_distance(gA, gB, normalized=config.normalized_distance))
            d_bc = float(identity_distance(gB, gC, normalized=config.normalized_distance))
            d_ad = float(identity_distance(gA, gD, normalized=config.normalized_distance))
            d_dc = float(identity_distance(gD, gC, normalized=config.normalized_distance))

            path_ab_bc = d_ab + d_bc
            path_ad_dc = d_ad + d_dc

            holonomy_residual = path_ab_bc - path_ad_dc
            abs_holonomy_residual = abs(holonomy_residual)

            corner_spins = [
                float(A["identity_spin"]),
                float(B["identity_spin"]),
                float(C["identity_spin"]),
                float(D["identity_spin"]),
            ]

            rows.append(
                {
                    "cell_i": i,
                    "cell_j": j,
                    "A_node_id": str(A["node_id"]),
                    "B_node_id": str(B["node_id"]),
                    "C_node_id": str(C["node_id"]),
                    "D_node_id": str(D["node_id"]),
                    "r_center": float(np.mean([A["r"], B["r"], C["r"], D["r"]])),
                    "alpha_center": float(np.mean([A["alpha"], B["alpha"], C["alpha"], D["alpha"]])),
                    "d_ab": d_ab,
                    "d_bc": d_bc,
                    "d_ad": d_ad,
                    "d_dc": d_dc,
                    "path_ab_bc": path_ab_bc,
                    "path_ad_dc": path_ad_dc,
                    "holonomy_residual": holonomy_residual,
                    "abs_holonomy_residual": abs_holonomy_residual,
                    "mean_abs_corner_spin": float(np.mean(np.abs(corner_spins))),
                    "max_abs_corner_spin": float(np.max(np.abs(corner_spins))),
                }
            )

    return pd.DataFrame(rows).sort_values(["cell_i", "cell_j"]).reset_index(drop=True)
