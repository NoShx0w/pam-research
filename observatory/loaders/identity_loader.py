from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class IdentityData:
    identity_nodes_df: pd.DataFrame
    holonomy_cells_df: pd.DataFrame
    identity_nodes_mtime: float | None
    holonomy_cells_mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_identity_data(outputs_root: str | Path = "outputs") -> IdentityData:
    outputs_root = Path(outputs_root)

    identity_nodes_csv = outputs_root / "fim_identity" / "identity_field_nodes.csv"
    obstruction_csv = outputs_root / "fim_identity_obstruction" / "identity_obstruction_nodes.csv"
    obstruction_signed_csv = outputs_root / "fim_identity_obstruction" / "identity_obstruction_signed_nodes.csv"
    holonomy_cells_csv = outputs_root / "fim_identity_holonomy" / "identity_holonomy_cells.csv"

    identity_nodes = pd.read_csv(identity_nodes_csv).copy() if identity_nodes_csv.exists() else pd.DataFrame()
    obstruction = pd.read_csv(obstruction_csv).copy() if obstruction_csv.exists() else pd.DataFrame()
    obstruction_signed = pd.read_csv(obstruction_signed_csv).copy() if obstruction_signed_csv.exists() else pd.DataFrame()
    holonomy_cells = pd.read_csv(holonomy_cells_csv).copy() if holonomy_cells_csv.exists() else pd.DataFrame()

    for df in [identity_nodes, obstruction, obstruction_signed]:
        if not df.empty:
            if "node_id" in df.columns:
                df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("Int64").astype(str)
            for col in ["r", "alpha"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    if not identity_nodes.empty:
        for col in ["identity_magnitude", "identity_spin"]:
            if col in identity_nodes.columns:
                identity_nodes[col] = pd.to_numeric(identity_nodes[col], errors="coerce")

    if not obstruction.empty:
        for col in [
            "obstruction_mean_holonomy",
            "obstruction_mean_abs_holonomy",
            "obstruction_max_abs_holonomy",
        ]:
            if col in obstruction.columns:
                obstruction[col] = pd.to_numeric(obstruction[col], errors="coerce")

    if not obstruction_signed.empty:
        for col in [
            "obstruction_mean_holonomy",
            "obstruction_signed_sum_holonomy",
            "obstruction_signed_weighted_holonomy",
            "obstruction_mean_abs_holonomy",
            "obstruction_max_abs_holonomy",
        ]:
            if col in obstruction_signed.columns:
                obstruction_signed[col] = pd.to_numeric(obstruction_signed[col], errors="coerce")

    merged = identity_nodes.copy()

    if not obstruction.empty:
        cols = [
            c
            for c in [
                "node_id",
                "obstruction_mean_holonomy",
                "obstruction_mean_abs_holonomy",
                "obstruction_max_abs_holonomy",
            ]
            if c in obstruction.columns
        ]
        merged = merged.merge(obstruction[cols], on="node_id", how="left")

    if not obstruction_signed.empty:
        cols = [
            c
            for c in [
                "node_id",
                "obstruction_signed_sum_holonomy",
                "obstruction_signed_weighted_holonomy",
            ]
            if c in obstruction_signed.columns
        ]
        merged = merged.merge(obstruction_signed[cols], on="node_id", how="left")

    if "obstruction_mean_abs_holonomy" in merged.columns:
        merged["absolute_holonomy_node"] = merged["obstruction_mean_abs_holonomy"]
    else:
        merged["absolute_holonomy_node"] = pd.NA

    merged = merged.sort_values(["r", "alpha"]).reset_index(drop=True) if not merged.empty else merged

    if not holonomy_cells.empty:
        for col in ["r_center", "alpha_center", "holonomy_residual", "abs_holonomy_residual"]:
            if col in holonomy_cells.columns:
                holonomy_cells[col] = pd.to_numeric(holonomy_cells[col], errors="coerce")

    return IdentityData(
        identity_nodes_df=merged,
        holonomy_cells_df=holonomy_cells,
        identity_nodes_mtime=_safe_mtime(identity_nodes_csv),
        holonomy_cells_mtime=_safe_mtime(holonomy_cells_csv),
    )
