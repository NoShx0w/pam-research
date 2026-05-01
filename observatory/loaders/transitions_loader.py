from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TransitionsData:
    transitions_df: pd.DataFrame
    mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_obs051_window_divergence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = _coerce_numeric(
        out,
        [
            "mean_lambda_local",
            "bounded_share",
            "mean_delta_d",
            "mean_d_start",
            "mean_d_end",
            "median_lambda_local",
            "max_lambda_local",
        ],
    )

    keep = [
        c
        for c in [
            "path_id",
            "path_family",
            "outcome_group",
            "seam_band",
            "coupling_class",
            "mean_lambda_local",
            "bounded_share",
            "mean_delta_d",
            "mean_d_start",
            "mean_d_end",
            "median_lambda_local",
            "max_lambda_local",
        ]
        if c in out.columns
    ]
    return out[keep].copy()


def _project_obs051_to_nodes(obs051_df: pd.DataFrame, path_nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Project window-level OBS-051 metrics onto nodes through path visitation.

    Each path-level/window-level row is joined to all nodes visited by that path,
    then aggregated to node_id / r / alpha.
    """
    if obs051_df.empty or path_nodes_df.empty:
        return pd.DataFrame()

    paths = path_nodes_df.copy()
    paths = _coerce_numeric(paths, ["node_id", "r", "alpha", "step"])

    keep_path_cols = [c for c in ["path_id", "node_id", "r", "alpha"] if c in paths.columns]
    if not {"path_id", "node_id", "r", "alpha"}.issubset(keep_path_cols):
        return pd.DataFrame()

    paths = paths[keep_path_cols].dropna(subset=["path_id", "node_id", "r", "alpha"]).copy()
    paths["node_id"] = paths["node_id"].astype(int)

    merged = obs051_df.merge(paths, on="path_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    agg = {}
    for col in merged.columns:
        if col in {"node_id", "r", "alpha"}:
            continue
        if col in {
            "mean_lambda_local",
            "bounded_share",
            "mean_delta_d",
            "mean_d_start",
            "mean_d_end",
            "median_lambda_local",
            "max_lambda_local",
        }:
            agg[col] = "mean"
        else:
            agg[col] = "first"

    out = (
        merged.groupby(["node_id", "r", "alpha"], dropna=False)
        .agg(agg)
        .reset_index()
    )
    return out


def _normalize_obs052_node_basin_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = _coerce_numeric(
        out,
        [
            "node_id",
            "r",
            "alpha",
            "attractor_score",
            "mean_lambda_local",
            "mean_roughness",
            "mean_abs_m_seam",
            "n_unique_paths",
            "recovering_landings",
            "nonrecovering_landings",
        ],
    )

    keep = [
        c
        for c in [
            "node_id",
            "r",
            "alpha",
            "attractor_score",
            "mean_lambda_local",
            "mean_roughness",
            "mean_abs_m_seam",
            "n_unique_paths",
            "recovering_landings",
            "nonrecovering_landings",
            "basin_class",
            "seam_band",
        ]
        if c in out.columns
    ]
    return out[keep].copy()


def _normalize_obs052_recovery_landing_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = _coerce_numeric(
        out,
        [
            "node_id",
            "r",
            "alpha",
            "recovering_landings",
            "nonrecovering_landings",
            "total_landings",
        ],
    )

    if "total_landings" not in out.columns:
        if {"recovering_landings", "nonrecovering_landings"}.issubset(out.columns):
            out["total_landings"] = (
                out["recovering_landings"].fillna(0) + out["nonrecovering_landings"].fillna(0)
            )

    keep = [
        c
        for c in [
            "node_id",
            "r",
            "alpha",
            "recovering_landings",
            "nonrecovering_landings",
            "total_landings",
        ]
        if c in out.columns
    ]
    return out[keep].copy()


def _merge_on_keys(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return extra.copy()
    if extra.empty:
        return base.copy()

    keys = [c for c in ["node_id", "r", "alpha"] if c in base.columns and c in extra.columns]
    if not keys:
        return base.copy()

    merged = base.merge(extra, on=keys, how="outer", suffixes=("", "_dup"))

    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    for dup_col in dup_cols:
        base_col = dup_col[:-4]
        if base_col not in merged.columns:
            merged = merged.rename(columns={dup_col: base_col})
        else:
            merged[base_col] = merged[base_col].combine_first(merged[dup_col])
            merged = merged.drop(columns=[dup_col])

    return merged


def load_transitions_data(
    outputs_root: str | Path = "outputs",
    observatory_root: str | Path = "observatory",
) -> TransitionsData:
    outputs_root = Path(outputs_root)

    obs051_path = outputs_root / "obs051_local_divergence_in_coupled_windows" / "obs051_window_divergence.csv"
    path_nodes_path = outputs_root / "scales" / "100000" / "family_substrate" / "path_nodes_for_family.csv"
    obs052_node_path = outputs_root / "obs052_attractor_basin_mapping" / "obs052_node_basin_table.csv"
    obs052_landings_path = outputs_root / "obs052_attractor_basin_mapping" / "obs052_recovery_landing_table.csv"

    obs051_df = pd.read_csv(obs051_path) if obs051_path.exists() else pd.DataFrame()
    path_nodes_df = pd.read_csv(path_nodes_path) if path_nodes_path.exists() else pd.DataFrame()
    obs052_node_df = pd.read_csv(obs052_node_path) if obs052_node_path.exists() else pd.DataFrame()
    obs052_landings_df = pd.read_csv(obs052_landings_path) if obs052_landings_path.exists() else pd.DataFrame()

    obs051_windows = _normalize_obs051_window_divergence(obs051_df)
    obs051_nodes = _project_obs051_to_nodes(obs051_windows, path_nodes_df)
    obs052_nodes = _normalize_obs052_node_basin_table(obs052_node_df)
    obs052_landings = _normalize_obs052_recovery_landing_table(obs052_landings_df)

    merged = pd.DataFrame()
    merged = _merge_on_keys(merged, obs051_nodes)
    merged = _merge_on_keys(merged, obs052_nodes)
    merged = _merge_on_keys(merged, obs052_landings)

    if not merged.empty:
        merged = _coerce_numeric(
            merged,
            [
                "node_id",
                "r",
                "alpha",
                "mean_lambda_local",
                "bounded_share",
                "mean_delta_d",
                "mean_d_start",
                "mean_d_end",
                "median_lambda_local",
                "max_lambda_local",
                "recovering_landings",
                "nonrecovering_landings",
                "total_landings",
                "attractor_score",
                "mean_roughness",
                "mean_abs_m_seam",
                "n_unique_paths",
            ],
        )

        if "total_landings" not in merged.columns:
            if {"recovering_landings", "nonrecovering_landings"}.issubset(merged.columns):
                merged["total_landings"] = (
                    merged["recovering_landings"].fillna(0) + merged["nonrecovering_landings"].fillna(0)
                )

        sort_cols = [c for c in ["node_id", "r", "alpha"] if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)

    mtimes = [
        _safe_mtime(p)
        for p in [obs051_path, path_nodes_path, obs052_node_path, obs052_landings_path]
        if p.exists()
    ]
    mtime = max(mtimes) if mtimes else None

    return TransitionsData(
        transitions_df=merged,
        mtime=mtime,
    )