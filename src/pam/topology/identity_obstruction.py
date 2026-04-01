from __future__ import annotations

"""
PAM Identity Obstruction

Derive a local obstruction field from canonical holonomy cells.

Primary principle
-----------------
Holonomy is the invariant transport-obstruction object.
This module derives node-local obstruction summaries from incident cells.

Outputs per node may include:
- obstruction_mean_abs_holonomy
- obstruction_max_abs_holonomy
- obstruction_mean_signed_holonomy
- obstruction_n_incident_cells
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IdentityObstructionConfig:
    use_absolute_primary: bool = True


def load_identity_obstruction_inputs(
    *,
    identity_nodes_csv: str | Path,
    holonomy_cells_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(identity_nodes_csv).copy()
    cells = pd.read_csv(holonomy_cells_csv).copy()

    required_nodes = {"node_id", "i", "j", "r", "alpha"}
    required_cells = {
        "cell_i",
        "cell_j",
        "A_node_id",
        "B_node_id",
        "C_node_id",
        "D_node_id",
        "holonomy_residual",
        "abs_holonomy_residual",
    }

    missing_nodes = required_nodes - set(nodes.columns)
    if missing_nodes:
        raise ValueError(f"identity nodes CSV missing required columns: {sorted(missing_nodes)}")

    missing_cells = required_cells - set(cells.columns)
    if missing_cells:
        raise ValueError(f"holonomy cells CSV missing required columns: {sorted(missing_cells)}")

    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64").astype(str)
    for col in ["i", "j"]:
        nodes[col] = pd.to_numeric(nodes[col], errors="raise").astype(int)
    for col in ["r", "alpha"]:
        nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

    for col in ["A_node_id", "B_node_id", "C_node_id", "D_node_id"]:
        cells[col] = pd.to_numeric(cells[col], errors="coerce").astype("Int64").astype(str)

    for col in ["cell_i", "cell_j"]:
        cells[col] = pd.to_numeric(cells[col], errors="raise").astype(int)

    for col in ["holonomy_residual", "abs_holonomy_residual"]:
        cells[col] = pd.to_numeric(cells[col], errors="coerce")

    return nodes, cells


def build_identity_obstruction_table(
    nodes: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    config: IdentityObstructionConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = IdentityObstructionConfig()

    incident: dict[str, list[dict[str, float]]] = {
        str(node_id): []
        for node_id in nodes["node_id"].astype(str)
    }

    for _, row in cells.iterrows():
        cell_payload = {
            "holonomy_residual": float(row["holonomy_residual"]),
            "abs_holonomy_residual": float(row["abs_holonomy_residual"]),
        }
        for node_col in ["A_node_id", "B_node_id", "C_node_id", "D_node_id"]:
            node_id = str(row[node_col])
            incident.setdefault(node_id, []).append(cell_payload)

    rows: list[dict[str, float | int | str]] = []

    for _, row in nodes.iterrows():
        node_id = str(row["node_id"])
        vals = incident.get(node_id, [])

        hol = np.array([v["holonomy_residual"] for v in vals], dtype=float) if vals else np.array([], dtype=float)
        abs_hol = np.array([v["abs_holonomy_residual"] for v in vals], dtype=float) if vals else np.array([], dtype=float)

        out = {
            "node_id": node_id,
            "i": int(row["i"]),
            "j": int(row["j"]),
            "r": float(row["r"]),
            "alpha": float(row["alpha"]),
            "obstruction_n_incident_cells": int(len(vals)),
            "obstruction_mean_holonomy": float(np.mean(hol)) if len(hol) else np.nan,
            "obstruction_mean_abs_holonomy": float(np.mean(abs_hol)) if len(abs_hol) else np.nan,
            "obstruction_max_abs_holonomy": float(np.max(abs_hol)) if len(abs_hol) else np.nan,
        }
        rows.append(out)

    return pd.DataFrame(rows).sort_values(["i", "j"]).reset_index(drop=True)
