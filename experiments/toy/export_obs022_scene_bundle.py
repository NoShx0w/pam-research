#!/usr/bin/env python3
"""
export_obs022_scene_bundle.py

Assemble a canonical scene bundle for PAM Observatory rendering.

Purpose
-------
Collect the key observatory objects into one stable export directory so that:
- 2D figures
- 3D scenes
- future observatory UI / TUI layers

can all render from the same scene contract.

Exports
-------
<outdir>/
  scene_nodes.csv
  scene_edges.csv
  scene_seam.csv
  scene_hubs.csv
  scene_routes.csv
  scene_glyphs.csv
  scene_metadata.txt

Primary inputs
--------------
- signed phase coordinates
- Fisher edge graph
- seam backprojection
- response operator nodes
- Lazarus scores
- family assignments
- path nodes

Default assumptions
-------------------
- uses the scale-100000 route-family outputs
- uses signed phase as the canonical z-lift field
- extracts top routing hubs from path occupancy
- extracts sparse glyphs from response-operator eigensystems

Notes
-----
This script assembles and normalizes data; it does not perform new scientific
analysis beyond:
- occupancy aggregation for hubs
- representative route flagging
- eigendecomposition of 2x2 response tensors
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    fim_csv: str = "outputs/fim/fim_surface.csv"
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    edges_csv: str = "outputs/fim_distance/fisher_edges.csv"
    seam_csv: str = "outputs/fim_phase/phase_boundary_mds_backprojected.csv"
    response_csv: str = "outputs/fim_response_operator/response_operator_nodes.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    family_csv: str = "outputs/scales/100000/family_substrate/path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/family_substrate/path_nodes_for_family.csv"
    outdir: str = "outputs/obs022_scene_bundle"
    top_hubs: int = 20
    glyph_top_k: int = 16
    reps_per_family: int = 8


FAMILY_ORDER = [
    "stable_seam_corridor",
    "reorganization_heavy",
    "settled_distant",
    "off_seam_reorganizing",
]


HOLONOMY_NODE_CANDIDATES = [
    "absolute_holonomy",
    "unsigned_obstruction",
    "node_holonomy",
    "holonomy",
    "spin",
]

HOLONOMY_EDGE_CANDIDATES = [
    "edge_holonomy",
    "absolute_holonomy",
    "unsigned_obstruction",
    "edge_unsigned_obstruction",
    "holonomy",
    "spin",
]

HOLONOMY_SOURCE_CANDIDATES = [
    "outputs/scales/100000/toy_scaled_probe_path_diagnostics/path_node_diagnostics.csv",
    "outputs/scales/100000/toy_scaled_probe_path_diagnostics/path_diagnostics.csv",
    "outputs/toy_geodesic_path_diagnostics/path_node_diagnostics.csv",
    "outputs/toy_geodesic_path_diagnostics/path_diagnostics.csv",
    "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv",
]

NODE_PROXY_SOURCES = [
    "outputs/fim_identity_obstruction/identity_obstruction_nodes.csv",
    "outputs/fim_identity_diagnostics/identity_diagnostics_nodes.csv",
]

NODE_PROXY_CANDIDATES = [
    "obstruction_mean_abs_holonomy",
    "obstruction_max_abs_holonomy",
    "obstruction_mean_holonomy",
    "obstruction_signed_weighted_holonomy",
    "obstruction_signed_sum_holonomy",
    "unsigned_obstruction",
    "absolute_holonomy",
    "holonomy",
    "spin",
]

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def read_csv_required(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def to_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def normalize_edge_columns(edges: pd.DataFrame) -> pd.DataFrame:
    edges = edges.copy()

    src_col = first_existing(list(edges.columns), ["src", "u", "src_id"])
    dst_col = first_existing(list(edges.columns), ["dst", "v", "dst_id"])
    w_col = first_existing(
        list(edges.columns),
        ["distance", "weight", "edge_cost", "fisher_distance", "dist", "length"],
    )

    if src_col is None or dst_col is None:
        raise ValueError(
            "Could not identify source/destination columns in edges file. "
            f"Found columns: {list(edges.columns)}"
        )

    rename_map = {src_col: "src_id", dst_col: "dst_id"}
    if w_col is not None:
        rename_map[w_col] = "edge_cost"

    edges = edges.rename(columns=rename_map)
    to_numeric_inplace(edges, ["src_id", "dst_id", "edge_cost"])
    edges = edges.dropna(subset=["src_id", "dst_id"]).copy()
    edges["src_id"] = edges["src_id"].astype(int)
    edges["dst_id"] = edges["dst_id"].astype(int)

    if "edge_cost" not in edges.columns:
        edges["edge_cost"] = np.nan

    return edges


def normalize_path_columns(paths: pd.DataFrame) -> pd.DataFrame:
    paths = paths.copy()
    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})
    required = ["path_id", "step", "r", "alpha", "mds1", "mds2"]
    missing = [c for c in required if c not in paths.columns]
    if missing:
        raise ValueError(f"Path nodes csv missing required columns: {missing}")
    return paths


def principal_response_eig(
    t_xx: np.ndarray,
    t_xy: np.ndarray,
    t_yx: np.ndarray,
    t_yy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      eig_major, eig_minor, theta_major
    where theta_major is the orientation of the principal eigenvector.
    """
    n = len(t_xx)
    eig_major = np.full(n, np.nan)
    eig_minor = np.full(n, np.nan)
    theta_major = np.full(n, np.nan)

    for i in range(n):
        M = np.array([[t_xx[i], t_xy[i]], [t_yx[i], t_yy[i]]], dtype=float)
        if not np.isfinite(M).all():
            continue

        vals, vecs = np.linalg.eig(M)
        order = np.argsort(np.abs(vals))[::-1]
        vals = np.real(vals[order])
        vecs = np.real(vecs[:, order])

        v = vecs[:, 0]
        eig_major[i] = float(vals[0])
        eig_minor[i] = float(vals[1])
        theta_major[i] = float(np.arctan2(v[1], v[0]))

    return eig_major, eig_minor, theta_major


def family_rank_key(series: pd.Series) -> pd.Series:
    order = {fam: i for i, fam in enumerate(FAMILY_ORDER)}
    return series.map(lambda x: order.get(x, 999))


def first_existing(cols: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(cols)
    return next((c for c in candidates if c in cols), None)


def existing_paths(paths: list[str]) -> list[Path]:
    return [Path(p) for p in paths if Path(p).exists()]


def load_best_holonomy_sources() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Best-effort loader for node/edge holonomy-like proxies from available upstream CSVs.

    Returns
    -------
    node_proxy_df, edge_proxy_df
      node_proxy_df expected keys: node_id or (r, alpha)
      edge_proxy_df expected keys: src_id/dst_id or equivalent
    """
    node_proxy_df = None
    edge_proxy_df = None

    for path in existing_paths(HOLONOMY_SOURCE_CANDIDATES):
        df = pd.read_csv(path)

        node_col = first_existing(df.columns, HOLONOMY_NODE_CANDIDATES)
        edge_col = first_existing(df.columns, HOLONOMY_EDGE_CANDIDATES)

        # node-level candidate
        if node_proxy_df is None and node_col is not None:
            join_cols = [c for c in ["node_id", "r", "alpha"] if c in df.columns]
            if join_cols:
                keep = join_cols + [node_col]
                tmp = df[keep].copy()

                # aggregate if repeated rows
                grp_cols = [c for c in ["node_id", "r", "alpha"] if c in tmp.columns]
                tmp = (
                    tmp.groupby(grp_cols, as_index=False)
                    .agg(node_holonomy_proxy=(node_col, "mean"))
                )
                node_proxy_df = tmp

        # edge-level candidate
        if edge_proxy_df is None and edge_col is not None:
            src_col = first_existing(df.columns, ["src_id", "src", "u"])
            dst_col = first_existing(df.columns, ["dst_id", "dst", "v"])
            if src_col is not None and dst_col is not None:
                tmp = df[[src_col, dst_col, edge_col]].copy()
                tmp = tmp.rename(columns={src_col: "src_id", dst_col: "dst_id", edge_col: "edge_holonomy_proxy"})
                to_numeric_inplace(tmp, ["src_id", "dst_id", "edge_holonomy_proxy"])
                tmp = tmp.dropna(subset=["src_id", "dst_id"]).copy()
                tmp["src_id"] = tmp["src_id"].astype(int)
                tmp["dst_id"] = tmp["dst_id"].astype(int)

                # average repeated edges
                tmp = (
                    tmp.groupby(["src_id", "dst_id"], as_index=False)
                    .agg(edge_holonomy_proxy=("edge_holonomy_proxy", "mean"))
                )
                edge_proxy_df = tmp

        if node_proxy_df is not None and edge_proxy_df is not None:
            break

    return node_proxy_df, edge_proxy_df


def load_best_node_proxy() -> pd.DataFrame | None:
    for path in NODE_PROXY_SOURCES:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)

        value_col = first_existing(df.columns, NODE_PROXY_CANDIDATES)
        if value_col is None:
            continue

        join_cols = [c for c in ["node_id", "r", "alpha"] if c in df.columns]
        if not join_cols:
            continue

        out = df[join_cols + [value_col]].copy()
        grp_cols = [c for c in ["node_id", "r", "alpha"] if c in out.columns]
        out = out.groupby(grp_cols, as_index=False).agg(node_holonomy_proxy=(value_col, "mean"))
        return out

    return None


def check_required_inputs(cfg: Config) -> None:
    required = [
        ("fim_csv", cfg.fim_csv),
        ("phase_csv", cfg.phase_csv),
        ("edges_csv", cfg.edges_csv),
        ("seam_csv", cfg.seam_csv),
        ("response_csv", cfg.response_csv),
        ("lazarus_csv", cfg.lazarus_csv),
        ("family_csv", cfg.family_csv),
        ("path_nodes_csv", cfg.path_nodes_csv),
    ]

    missing: list[tuple[str, str]] = []
    for label, path in required:
        if not Path(path).exists():
            missing.append((label, path))

    if not missing:
        return

    lines = [
        "Missing required inputs for export_obs022_scene_bundle.py:",
        "",
    ]
    for label, path in missing:
        lines.append(f"  - {label}: {path}")

    lines.extend(
        [
            "",
            "This export depends on upstream study artifacts in addition to the core pipeline.",
            "Typical prerequisite chain includes:",
            "  1. bash scripts/run_full_pipeline.sh",
            "  2. PYTHONPATH=src python experiments/studies/fim_response_operator.py",
            "  3. scale-specific family/path preparation artifacts",
            "",
            "You may also override inputs explicitly, e.g.:",
            "  --family-csv <path>",
            "  --path-nodes-csv <path>",
        ]
    )

    raise FileNotFoundError("\n".join(lines))


# ---------------------------------------------------------------------
# loaders / mergers
# ---------------------------------------------------------------------


def load_nodes(
    phase_csv: str,
    lazarus_csv: str,
    response_csv: str,
    fim_csv: str,
) -> pd.DataFrame:
    phase = read_csv_required(phase_csv).copy()
    laz = read_csv_required(lazarus_csv).copy()
    rsp = read_csv_required(response_csv).copy()
    fim = read_csv_required(fim_csv).copy()

    keep_phase = [
        c for c in [
            "node_id", "r", "alpha", "mds1", "mds2",
            "signed_phase", "distance_to_seam"
        ]
        if c in phase.columns
    ]
    nodes = phase[keep_phase].copy()

    if "node_id" not in nodes.columns:
        nodes = nodes.reset_index(drop=True)
        nodes["node_id"] = nodes.index.astype(int)

    # merge lazarus
    keep_laz = [
        c for c in ["node_id", "r", "alpha", "lazarus_score", "lazarus_hit"]
        if c in laz.columns
    ]
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in nodes.columns and c in laz.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]
    if keep_laz:
        nodes = nodes.merge(laz[keep_laz], on=join_cols, how="left")

    # merge response operator
    keep_rsp = [
        c for c in [
            "node_id", "r", "alpha",
            "response_strength",
            "T_xx", "T_xy", "T_yx", "T_yy",
            "trace_T", "frobenius_T",
        ]
        if c in rsp.columns
    ]
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in nodes.columns and c in rsp.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]
    if keep_rsp:
        nodes = nodes.merge(rsp[keep_rsp], on=join_cols, how="left")

    # merge FIM geometry / connection proxy
    keep_fim = [
        c for c in [
            "r", "alpha",
            "fim_theta", "fim_eig1", "fim_eig2", "fim_cond", "fim_trace"
        ]
        if c in fim.columns
    ]
    if keep_fim:
        nodes = nodes.merge(fim[keep_fim], on=["r", "alpha"], how="left")

    to_numeric_inplace(
        nodes,
        [
            "node_id", "r", "alpha", "mds1", "mds2",
            "signed_phase", "distance_to_seam",
            "lazarus_score", "lazarus_hit",
            "response_strength",
            "T_xx", "T_xy", "T_yx", "T_yy",
            "trace_T", "frobenius_T",
            "fim_theta", "fim_eig1", "fim_eig2", "fim_cond", "fim_trace",
        ],
    )

    # derive response eigensystem if tensor exists
    if {"T_xx", "T_xy", "T_yx", "T_yy"}.issubset(nodes.columns):
        eig1, eig2, theta = principal_response_eig(
            nodes["T_xx"].to_numpy(dtype=float),
            nodes["T_xy"].to_numpy(dtype=float),
            nodes["T_yx"].to_numpy(dtype=float),
            nodes["T_yy"].to_numpy(dtype=float),
        )
        nodes["rsp_eig_major"] = eig1
        nodes["rsp_eig_minor"] = eig2
        nodes["rsp_theta"] = theta

        den = np.abs(eig2)
        num = np.abs(eig1)
        ratio = np.full_like(num, np.nan, dtype=float)
        mask = den > 1e-12
        ratio[mask] = num[mask] / den[mask]
        nodes["rsp_anisotropy"] = ratio
    else:
        nodes["rsp_eig_major"] = np.nan
        nodes["rsp_eig_minor"] = np.nan
        nodes["rsp_theta"] = np.nan
        nodes["rsp_anisotropy"] = np.nan

    # best-effort holonomy / obstruction proxy merge
    node_proxy_df = load_best_node_proxy()
    if node_proxy_df is not None:
        join_cols = [c for c in ["node_id", "r", "alpha"] if c in nodes.columns and c in node_proxy_df.columns]
        if not join_cols:
            join_cols = [c for c in ["r", "alpha"] if c in nodes.columns and c in node_proxy_df.columns]
        if join_cols:
            nodes = nodes.merge(node_proxy_df, on=join_cols, how="left")
    else:
        nodes["node_holonomy_proxy"] = np.nan
    print("node_holonomy_proxy non-null:", int(nodes["node_holonomy_proxy"].notna().sum()))

    return nodes.sort_values(["r", "alpha"]).reset_index(drop=True)


def load_seam(seam_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    seam = read_csv_required(seam_csv).copy()

    join_cols = [c for c in ["r", "alpha"] if c in seam.columns and c in nodes.columns]
    if "mds1" not in seam.columns or "mds2" not in seam.columns:
        if not join_cols:
            raise ValueError("Seam csv lacks mds1/mds2 and cannot be joined on r/alpha.")
        seam = seam.merge(
            nodes[join_cols + [c for c in ["mds1", "mds2", "signed_phase", "distance_to_seam"] if c in nodes.columns]],
            on=join_cols,
            how="left",
        )
    else:
        enrich_cols = [c for c in ["signed_phase", "distance_to_seam"] if c in nodes.columns]
        if join_cols and enrich_cols:
            seam = seam.merge(
                nodes[join_cols + enrich_cols].drop_duplicates(),
                on=join_cols,
                how="left",
            )

    keep = [c for c in ["r", "alpha", "mds1", "mds2", "signed_phase", "distance_to_seam"] if c in seam.columns]
    seam = seam[keep].copy()
    to_numeric_inplace(seam, keep)
    return seam.sort_values(["mds1", "mds2"]).reset_index(drop=True)


def load_edges(edges_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    edges = normalize_edge_columns(read_csv_required(edges_csv))
    node_lookup = nodes.set_index("node_id")

    src = node_lookup.loc[edges["src_id"], ["mds1", "mds2", "signed_phase", "distance_to_seam"]].reset_index(drop=True)
    dst = node_lookup.loc[edges["dst_id"], ["mds1", "mds2", "signed_phase", "distance_to_seam"]].reset_index(drop=True)

    out = edges.copy()
    out["src_mds1"] = src["mds1"].to_numpy()
    out["src_mds2"] = src["mds2"].to_numpy()
    out["dst_mds1"] = dst["mds1"].to_numpy()
    out["dst_mds2"] = dst["mds2"].to_numpy()

    out["src_signed_phase"] = src["signed_phase"].to_numpy()
    out["dst_signed_phase"] = dst["signed_phase"].to_numpy()
    out["src_distance_to_seam"] = src["distance_to_seam"].to_numpy()
    out["dst_distance_to_seam"] = dst["distance_to_seam"].to_numpy()

    out["edge_phase_mid"] = 0.5 * (out["src_signed_phase"] + out["dst_signed_phase"])
    out["edge_seam_mid"] = 0.5 * (out["src_distance_to_seam"] + out["dst_distance_to_seam"])
    out["edge_phase_delta"] = out["dst_signed_phase"] - out["src_signed_phase"]
    out["edge_seam_delta"] = out["dst_distance_to_seam"] - out["src_distance_to_seam"]

    # best-effort edge holonomy / obstruction proxy merge
    _, edge_proxy_df = load_best_holonomy_sources()
    if edge_proxy_df is not None:
        out = out.merge(edge_proxy_df, on=["src_id", "dst_id"], how="left")

        # fill reverse direction if needed
        rev = edge_proxy_df.rename(columns={"src_id": "dst_id", "dst_id": "src_id"})
        out = out.merge(
            rev,
            on=["src_id", "dst_id"],
            how="left",
            suffixes=("", "_rev"),
        )
        if "edge_holonomy_proxy_rev" in out.columns:
            out["edge_holonomy_proxy"] = out["edge_holonomy_proxy"].fillna(out["edge_holonomy_proxy_rev"])
            out = out.drop(columns=["edge_holonomy_proxy_rev"])
    else:
        out["edge_holonomy_proxy"] = np.nan

    return out.reset_index(drop=True)


def seam_run_lengths(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    cur = 0
    for x in mask:
        if bool(x):
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def pick_representative_path_ids(
    routes: pd.DataFrame,
    reps_per_family: int,
    min_steps: int = 5,
    min_unique_nodes: int = 5,
    min_start_end_dist: float = 0.35,
    near_dup_tol: float = 0.12,
    seam_threshold: float = 0.15,
) -> list[str]:
    """
    Select visually and scientifically representative paths.

    Upgrade
    -------
    In addition to path length / Lazarus / seam distance, this version uses
    seam-residency structure so representatives are more family-typical:

    stable_seam_corridor:
      - seam-near
      - high Lazarus / response
      - coherent seam traversal
      - fewer seam episodes
      - longer seam runs

    reorganization_heavy:
      - seam-near
      - high Lazarus
      - longer / rougher
      - more fragmented seam traversal
      - stronger phase reorganization
    """
    if len(routes) == 0:
        return []

    path_rows = []
    for path_id, grp in routes.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy()

        fam = str(grp["path_family"].iloc[0]) if "path_family" in grp.columns else "unknown"
        n_rows = len(grp)
        n_steps = int(pd.to_numeric(grp["step"], errors="coerce").max()) if "step" in grp.columns else n_rows - 1
        n_unique_nodes = int(grp["node_id"].nunique()) if "node_id" in grp.columns else n_rows

        x = pd.to_numeric(grp["mds1"], errors="coerce")
        y = pd.to_numeric(grp["mds2"], errors="coerce")
        x0, y0 = float(x.iloc[0]), float(y.iloc[0])
        x1, y1 = float(x.iloc[-1]), float(y.iloc[-1])
        start_end_dist = float(np.hypot(x1 - x0, y1 - y0))

        laz = pd.to_numeric(grp.get("lazarus_score"), errors="coerce")
        rsp = pd.to_numeric(grp.get("response_strength"), errors="coerce")
        d2s = pd.to_numeric(grp.get("distance_to_seam"), errors="coerce")
        phase = pd.to_numeric(grp.get("signed_phase"), errors="coerce")

        mean_lazarus = float(laz.mean()) if laz.notna().any() else np.nan
        max_lazarus = float(laz.max()) if laz.notna().any() else np.nan
        mean_response = float(rsp.mean()) if rsp.notna().any() else np.nan
        mean_distance_to_seam = float(d2s.mean()) if d2s.notna().any() else np.nan
        min_distance_to_seam = float(d2s.min()) if d2s.notna().any() else np.nan

        phase_start = float(phase.iloc[0]) if phase.notna().any() else np.nan
        phase_end = float(phase.iloc[-1]) if phase.notna().any() else np.nan
        phase_span = phase_end - phase_start if np.isfinite(phase_start) and np.isfinite(phase_end) else np.nan

        # seam residency
        seam_mask = (d2s <= seam_threshold).fillna(False).to_numpy(dtype=bool)
        runs = seam_run_lengths(seam_mask)
        n_seam_points = int(seam_mask.sum())
        seam_fraction = float(n_seam_points / max(len(seam_mask), 1))
        n_seam_episodes = int(len(runs))
        mean_seam_run_length = float(np.mean(runs)) if runs else 0.0
        max_seam_run_length = int(max(runs)) if runs else 0

        path_rows.append(
            {
                "path_id": path_id,
                "path_family": fam,
                "n_rows": n_rows,
                "n_steps": n_steps,
                "n_unique_nodes": n_unique_nodes,
                "start_x": x0,
                "start_y": y0,
                "end_x": x1,
                "end_y": y1,
                "start_end_dist": start_end_dist,
                "mean_lazarus": mean_lazarus,
                "max_lazarus": max_lazarus,
                "mean_response": mean_response,
                "mean_distance_to_seam": mean_distance_to_seam,
                "min_distance_to_seam": min_distance_to_seam,
                "phase_span": phase_span,
                "seam_fraction": seam_fraction,
                "n_seam_episodes": n_seam_episodes,
                "mean_seam_run_length": mean_seam_run_length,
                "max_seam_run_length": max_seam_run_length,
            }
        )

    path_df = pd.DataFrame(path_rows)

    if len(path_df) == 0:
        return []

    keep = (
        (path_df["n_steps"] >= min_steps)
        & (path_df["n_unique_nodes"] >= min_unique_nodes)
        & (path_df["start_end_dist"] >= min_start_end_dist)
    )
    path_df = path_df[keep].copy()

    if len(path_df) == 0:
        return []

    selected_ids: list[str] = []

    for fam in ["stable_seam_corridor", "reorganization_heavy"]:
        sub = path_df[path_df["path_family"] == fam].copy()
        if len(sub) == 0:
            continue

        def _rank(col: str, ascending: bool) -> pd.Series:
            s = pd.to_numeric(sub[col], errors="coerce")
            if s.notna().sum() == 0:
                return pd.Series(np.zeros(len(sub)), index=sub.index)
            return s.rank(pct=True, ascending=ascending)

        if fam == "stable_seam_corridor":
            # corridor = coherent, privileged seam traversal
            sub["score"] = (
                0.20 * _rank("mean_lazarus", ascending=True)
                + 0.18 * _rank("mean_response", ascending=True)
                + 0.16 * _rank("seam_fraction", ascending=True)
                + 0.16 * _rank("mean_seam_run_length", ascending=True)
                + 0.12 * _rank("max_seam_run_length", ascending=True)
                + 0.10 * _rank("n_steps", ascending=True)
                + 0.08 * _rank("mean_distance_to_seam", ascending=False)
                - 0.12 * _rank("n_seam_episodes", ascending=True)
            )
        else:
            # reorganization-heavy = seam-rich, long, fragmented, reorganizing
            sub["score"] = (
                0.20 * _rank("mean_lazarus", ascending=True)
                + 0.16 * _rank("seam_fraction", ascending=True)
                + 0.14 * _rank("n_seam_episodes", ascending=True)
                + 0.12 * _rank("mean_seam_run_length", ascending=True)
                + 0.12 * _rank("n_steps", ascending=True)
                + 0.10 * _rank("start_end_dist", ascending=True)
                + 0.10 * _rank("mean_distance_to_seam", ascending=False)
                + 0.06 * _rank("phase_span", ascending=False)
            )

        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)

        fam_selected: list[str] = []
        fam_anchors: list[tuple[float, float, float, float]] = []

        for _, row in sub.iterrows():
            cand = (
                float(row["start_x"]),
                float(row["start_y"]),
                float(row["end_x"]),
                float(row["end_y"]),
            )

            too_close = False
            for prev in fam_anchors:
                d = np.sqrt(
                    (cand[0] - prev[0]) ** 2
                    + (cand[1] - prev[1]) ** 2
                    + (cand[2] - prev[2]) ** 2
                    + (cand[3] - prev[3]) ** 2
                )
                if d < near_dup_tol:
                    too_close = True
                    break

            if too_close:
                continue

            fam_selected.append(str(row["path_id"]))
            fam_anchors.append(cand)

            if len(fam_selected) >= reps_per_family:
                break

        selected_ids.extend(fam_selected)

    return selected_ids


def pick_branch_away_path_ids(
    routes: pd.DataFrame,
    n_paths: int = 8,
    min_steps: int = 6,
    min_unique_nodes: int = 6,
    seam_threshold: float = 0.15,
    min_exit_gain: float = 0.25,
    min_start_end_dist: float = 0.45,
    near_dup_tol: float = 0.14,
) -> list[str]:
    """
    Select seam-exit / branch-away exemplars.

    Desired pattern:
    - path comes near the seam
    - path does not simply remain a seam-resident exemplar
    - path clearly moves away from seam contact afterward
    """
    rows = []

    for path_id, grp in routes.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy()

        n_steps = int(pd.to_numeric(grp["step"], errors="coerce").max()) if "step" in grp.columns else len(grp) - 1
        n_unique_nodes = int(grp["node_id"].nunique()) if "node_id" in grp.columns else len(grp)

        x = pd.to_numeric(grp["mds1"], errors="coerce")
        y = pd.to_numeric(grp["mds2"], errors="coerce")
        d2s = pd.to_numeric(grp["distance_to_seam"], errors="coerce")
        laz = pd.to_numeric(grp.get("lazarus_score"), errors="coerce")

        if d2s.notna().sum() == 0:
            continue

        start_end_dist = float(np.hypot(float(x.iloc[-1] - x.iloc[0]), float(y.iloc[-1] - y.iloc[0])))
        min_idx = int(d2s.idxmin())
        min_distance = float(d2s.min())
        end_distance = float(d2s.iloc[-1])
        exit_gain = float(end_distance - min_distance)
        seam_fraction = float((d2s <= seam_threshold).fillna(False).mean())
        mean_lazarus = float(laz.mean()) if laz.notna().any() else np.nan

        rows.append(
            {
                "path_id": path_id,
                "n_steps": n_steps,
                "n_unique_nodes": n_unique_nodes,
                "start_x": float(x.iloc[0]),
                "start_y": float(y.iloc[0]),
                "end_x": float(x.iloc[-1]),
                "end_y": float(y.iloc[-1]),
                "start_end_dist": start_end_dist,
                "min_distance_to_seam": min_distance,
                "end_distance_to_seam": end_distance,
                "exit_gain": exit_gain,
                "seam_fraction": seam_fraction,
                "mean_lazarus": mean_lazarus,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return []

    keep = (
        (df["n_steps"] >= min_steps)
        & (df["n_unique_nodes"] >= min_unique_nodes)
        & (df["start_end_dist"] >= min_start_end_dist)
        & (df["min_distance_to_seam"] <= seam_threshold)
        & (df["exit_gain"] >= min_exit_gain)
    )
    df = df[keep].copy()
    if len(df) == 0:
        return []

    def _rank(col: str, ascending: bool) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            return pd.Series(np.zeros(len(df)), index=df.index)
        return s.rank(pct=True, ascending=ascending)

    # Prefer real seam contact, then strong departure, then visible path extent
    df["score"] = (
        0.30 * _rank("exit_gain", ascending=True)
        + 0.22 * _rank("start_end_dist", ascending=True)
        + 0.18 * _rank("n_steps", ascending=True)
        + 0.15 * _rank("min_distance_to_seam", ascending=False)
        + 0.10 * _rank("mean_lazarus", ascending=True)
        - 0.10 * _rank("seam_fraction", ascending=True)
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    selected: list[str] = []
    anchors: list[tuple[float, float, float, float]] = []

    for _, row in df.iterrows():
        cand = (
            float(row["start_x"]),
            float(row["start_y"]),
            float(row["end_x"]),
            float(row["end_y"]),
        )

        too_close = False
        for prev in anchors:
            d = np.sqrt(
                (cand[0] - prev[0]) ** 2
                + (cand[1] - prev[1]) ** 2
                + (cand[2] - prev[2]) ** 2
                + (cand[3] - prev[3]) ** 2
            )
            if d < near_dup_tol:
                too_close = True
                break

        if too_close:
            continue

        selected.append(str(row["path_id"]))
        anchors.append(cand)

        if len(selected) >= n_paths:
            break

    return selected


def load_routes(
    path_nodes_csv: str,
    family_csv: str,
    nodes: pd.DataFrame,
    reps_per_family: int,
) -> pd.DataFrame:
    paths = normalize_path_columns(read_csv_required(path_nodes_csv))
    fam = read_csv_required(family_csv).copy()

    if "path_family" not in fam.columns:
        raise ValueError("Family csv must contain 'path_family'.")

    fam_cols = [c for c in ["path_id", "path_family"] if c in fam.columns]

    paths["path_id"] = paths["path_id"].astype(str)
    fam["path_id"] = fam["path_id"].astype(str)
    routes = paths.merge(fam[fam_cols], on="path_id", how="left")

    n_total = len(routes)
    n_labeled = int(routes["path_family"].notna().sum()) if "path_family" in routes.columns else 0
    print(f"route rows: {n_total}, labeled rows: {n_labeled}")
    print(f"unique path ids in paths: {paths['path_id'].nunique(dropna=True)}")
    print(f"unique path ids in fam: {fam['path_id'].nunique(dropna=True)}")
    print(f"unique labeled path ids: {routes.loc[routes['path_family'].notna(), 'path_id'].nunique(dropna=True) if 'path_family' in routes.columns else 0}")

    # enrich missing fields from canonical nodes table
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in routes.columns and c in nodes.columns]
    enrich_cols = [
        c for c in [
            "signed_phase", "distance_to_seam", "lazarus_score",
            "response_strength", "rsp_theta", "rsp_eig_major",
            "rsp_eig_minor", "rsp_anisotropy",
        ] if c in nodes.columns and c not in routes.columns
    ]
    if join_cols and enrich_cols:
        routes = routes.merge(nodes[join_cols + enrich_cols], on=join_cols, how="left")

    to_numeric_inplace(
        routes,
        [
            "step", "node_id", "r", "alpha", "mds1", "mds2",
            "signed_phase", "distance_to_seam", "lazarus_score",
            "response_strength", "rsp_theta", "rsp_eig_major",
            "rsp_eig_minor", "rsp_anisotropy",
        ],
    )

    # mark representative paths
    grouped = (
        routes.groupby(["path_id", "path_family"], as_index=False)
        .agg(
            n_steps=("step", "max"),
            mean_lazarus=("lazarus_score", "mean"),
            mean_distance_to_seam=("distance_to_seam", "mean"),
            mean_response=("response_strength", "mean"),
        )
    )

    rep_ids = pick_representative_path_ids(
        routes,
        reps_per_family=reps_per_family,
        min_steps=5,
        min_unique_nodes=5,
        min_start_end_dist=0.35,
        near_dup_tol=0.12,
        seam_threshold=0.15,
    )

    routes["is_representative"] = routes["path_id"].isin(rep_ids).astype(int)

    branch_ids = pick_branch_away_path_ids(
        routes,
        n_paths=max(6, reps_per_family),
        min_steps=6,
        min_unique_nodes=6,
        seam_threshold=0.15,
        min_exit_gain=0.25,
        min_start_end_dist=0.45,
        near_dup_tol=0.14,
    )

    routes["is_branch_away"] = routes["path_id"].isin(branch_ids).astype(int)

    routes = routes.sort_values(
        by=["path_family", "path_id", "step"],
        key=lambda s: family_rank_key(s) if s.name == "path_family" else s,
    ).reset_index(drop=True)

    return routes


def build_hubs(routes: pd.DataFrame, nodes: pd.DataFrame, top_hubs: int) -> pd.DataFrame:
    traffic = (
        routes.groupby(["node_id", "r", "alpha"], as_index=False)
        .agg(
            n_visits=("path_id", "size"),
            n_unique_paths=("path_id", "nunique"),
        )
        .sort_values(["n_unique_paths", "n_visits"], ascending=False)
        .reset_index(drop=True)
    )

    n_total_paths = max(int(routes["path_id"].nunique()), 1)
    traffic["path_occupancy"] = traffic["n_unique_paths"] / n_total_paths

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in traffic.columns and c in nodes.columns]
    hubs = traffic.merge(
        nodes[
            join_cols
            + [c for c in [
                "mds1", "mds2", "signed_phase", "distance_to_seam",
                "lazarus_score", "response_strength",
                "rsp_theta", "rsp_eig_major", "rsp_eig_minor", "rsp_anisotropy",
            ] if c in nodes.columns]
        ],
        on=join_cols,
        how="left",
    )

    return hubs.head(top_hubs).reset_index(drop=True)


def build_glyphs(nodes: pd.DataFrame, glyph_top_k: int) -> pd.DataFrame:
    """
    Sparse glyph set for future rendering.

    Selection policy:
    Prefer nodes that are seam-near and response-anisotropic.
    """
    work = nodes.copy()

    score_terms = []
    if "distance_to_seam" in work.columns:
        seam_term = -pd.to_numeric(work["distance_to_seam"], errors="coerce").fillna(np.inf)
        score_terms.append(seam_term.rank(pct=True))
    if "rsp_anisotropy" in work.columns:
        aniso_term = pd.to_numeric(work["rsp_anisotropy"], errors="coerce").fillna(-np.inf)
        score_terms.append(aniso_term.rank(pct=True))
    if "response_strength" in work.columns:
        resp_term = pd.to_numeric(work["response_strength"], errors="coerce").fillna(-np.inf)
        score_terms.append(resp_term.rank(pct=True))

    if score_terms:
        score = sum(score_terms) / len(score_terms)
    else:
        score = pd.Series(np.zeros(len(work)), index=work.index)

    work["glyph_score"] = score
    work = work.sort_values("glyph_score", ascending=False).head(glyph_top_k).copy()

    keep = [
        c for c in [
            "node_id", "r", "alpha", "mds1", "mds2",
            "signed_phase", "distance_to_seam",
            "lazarus_score", "response_strength",
            "rsp_theta", "rsp_eig_major", "rsp_eig_minor", "rsp_anisotropy",
            "glyph_score",
        ] if c in work.columns
    ]
    return work[keep].reset_index(drop=True)




def write_metadata(
    outpath: Path,
    cfg: Config,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seam: pd.DataFrame,
    hubs: pd.DataFrame,
    routes: pd.DataFrame,
    glyphs: pd.DataFrame,
) -> None:
    lines = [
        "=== OBS-022 Scene Bundle Metadata ===",
        "",
        "Inputs",
        f"phase_csv={cfg.phase_csv}",
        f"edges_csv={cfg.edges_csv}",
        f"seam_csv={cfg.seam_csv}",
        f"response_csv={cfg.response_csv}",
        f"lazarus_csv={cfg.lazarus_csv}",
        f"family_csv={cfg.family_csv}",
        f"path_nodes_csv={cfg.path_nodes_csv}",
        "",
        "Bundle counts",
        f"n_nodes={len(nodes)}",
        f"n_edges={len(edges)}",
        f"n_seam_points={len(seam)}",
        f"n_hubs={len(hubs)}",
        f"n_route_rows={len(routes)}",
        f"n_unique_paths={routes['path_id'].nunique()}",
        f"n_glyphs={len(glyphs)}",
        "",
        "Route families",
    ]
    fam_counts = routes[["path_id", "path_family"]].drop_duplicates()["path_family"].value_counts(dropna=False)
    for fam, n in fam_counts.items():
        lines.append(f"  {fam}: {n}")
    lines.extend(
        [
            "",
            "Bundle semantics",
            "- scene_nodes.csv: canonical node table with scalar fields and tensor summaries",
            "- scene_edges.csv: adjacency web with endpoint coordinates and midpoint field values",
            "- scene_seam.csv: seam backprojection in MDS coordinates",
            "- scene_hubs.csv: top routing hubs by path occupancy",
            "- scene_routes.csv: long route table with family labels and representative flags",
            "- scene_glyphs.csv: sparse anisotropy/operator glyph anchors",
        ]
    )
    outpath.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical OBS-022 scene bundle.")
    parser.add_argument("--fim-csv", default=Config.fim_csv)
    parser.add_argument("--phase-csv", default=Config.phase_csv)
    parser.add_argument("--edges-csv", default=Config.edges_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--response-csv", default=Config.response_csv)
    parser.add_argument("--lazarus-csv", default=Config.lazarus_csv)
    parser.add_argument("--family-csv", default=Config.family_csv)
    parser.add_argument("--path-nodes-csv", default=Config.path_nodes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--top-hubs", type=int, default=Config.top_hubs)
    parser.add_argument("--glyph-top-k", type=int, default=Config.glyph_top_k)
    parser.add_argument("--reps-per-family", type=int, default=Config.reps_per_family)
    args = parser.parse_args()

    cfg = Config(
        fim_csv=args.fim_csv,
        phase_csv=args.phase_csv,
        edges_csv=args.edges_csv,
        seam_csv=args.seam_csv,
        response_csv=args.response_csv,
        lazarus_csv=args.lazarus_csv,
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        outdir=args.outdir,
        top_hubs=args.top_hubs,
        glyph_top_k=args.glyph_top_k,
        reps_per_family=args.reps_per_family,
    )
    
    check_required_inputs(cfg)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(cfg.phase_csv, cfg.lazarus_csv, cfg.response_csv, cfg.fim_csv)
    seam = load_seam(cfg.seam_csv, nodes)
    edges = load_edges(cfg.edges_csv, nodes)
    routes = load_routes(cfg.path_nodes_csv, cfg.family_csv, nodes, cfg.reps_per_family)
    hubs = build_hubs(routes, nodes, cfg.top_hubs)
    glyphs = build_glyphs(nodes, cfg.glyph_top_k)

    nodes.to_csv(outdir / "scene_nodes.csv", index=False)
    edges.to_csv(outdir / "scene_edges.csv", index=False)
    seam.to_csv(outdir / "scene_seam.csv", index=False)
    hubs.to_csv(outdir / "scene_hubs.csv", index=False)
    routes.to_csv(outdir / "scene_routes.csv", index=False)
    glyphs.to_csv(outdir / "scene_glyphs.csv", index=False)
    write_metadata(
        outdir / "scene_metadata.txt",
        cfg,
        nodes,
        edges,
        seam,
        hubs,
        routes,
        glyphs,
    )

    print(outdir / "scene_nodes.csv")
    print(outdir / "scene_edges.csv")
    print(outdir / "scene_seam.csv")
    print(outdir / "scene_hubs.csv")
    print(outdir / "scene_routes.csv")
    print(outdir / "scene_glyphs.csv")
    print(outdir / "scene_metadata.txt")


if __name__ == "__main__":
    main()
