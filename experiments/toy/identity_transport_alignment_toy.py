#!/usr/bin/env python3
"""
identity_transport_alignment_toy.py

Toy experiment: transport-aware alignment of the response principal-direction field
on the PAM manifold.

Core idea
---------
Compare local response directions across neighboring nodes *after transporting*
one direction to the other's tangent frame using an identity-angle connection proxy.

This script computes:
  1. principal response direction at each node
  2. identity-angle transport across graph edges
  3. axial transport misalignment per edge and per node
  4. simple correlations with seam distance / holonomy proxy columns if present
  5. a compact diagnostic plate and CSV exports

Definitions
-----------
Given neighboring nodes i -> j:
  transport(v_i, i->j) = R(theta_j - theta_i) @ v_i

where theta_* is an identity-angle / connection proxy.
By default this is taken from:
  - fim_theta
  - identity_theta
  - theta_identity
  - rsp_theta
  - phase_tangent_theta

in that order of preference.

Because eigenvectors are axial rather than directed, misalignment uses the
smaller of angle(u, v) and angle(u, -v), giving values in [0, pi/2].

Expected outputs
----------------
<outdir>/
  node_transport_alignment.csv
  edge_transport_alignment.csv
  identity_transport_alignment_summary.txt
  identity_transport_alignment_plate.png

Usage
-----
PYTHONPATH=src .venv/bin/python experiments/toy/identity_transport_alignment_toy.py

Optional bundle-driven usage:
PYTHONPATH=src .venv/bin/python experiments/toy/identity_transport_alignment_toy.py \
  --bundle-dir outputs/obs022_scene_bundle

Notes
-----
- This is intentionally a toy / observatory diagnostic, not a final canonical module.
- It works from either the scene bundle or directly from base CSV inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    # bundle mode
    bundle_dir: str | None = "outputs/obs022_scene_bundle"

    # direct-input mode
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    edges_csv: str = "outputs/fim_distance/fisher_edges.csv"
    seam_csv: str = "outputs/fim_phase/phase_boundary_mds_backprojected.csv"
    response_csv: str = "outputs/fim_response_operator/response_operator_nodes.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"

    outdir: str = "outputs/toy_identity_transport_alignment"
    seam_threshold: float = 0.15
    use_axial_comparison: bool = True


THETA_CANDIDATES = [
    "fim_theta",
    "identity_theta",
    "theta_identity",
    "rsp_theta",
    "phase_tangent_theta",
]

EDGE_HOLONOMY_CANDIDATES = [
    "edge_holonomy",
    "holonomy",
    "absolute_holonomy",
    "unsigned_obstruction",
    "edge_unsigned_obstruction",
]

NODE_HOLONOMY_CANDIDATES = [
    "node_holonomy_proxy",
    "obstruction_mean_abs_holonomy",
    "obstruction_max_abs_holonomy",
    "obstruction_mean_holonomy",
    "obstruction_signed_weighted_holonomy",
    "obstruction_signed_sum_holonomy",
    "absolute_holonomy",
    "unsigned_obstruction",
    "node_holonomy",
    "holonomy",
    "spin",
]

SRC_CANDIDATES = ["src_id", "src", "u"]
DST_CANDIDATES = ["dst_id", "dst", "v"]


# ---------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------


def first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    return next((c for c in candidates if c in cols), None)


def read_csv_required(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def to_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def angle_from_theta(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)], dtype=float)


def angle_between_vectors(u: np.ndarray, v: np.ndarray, axial: bool = True) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if not np.isfinite([nu, nv]).all() or nu <= 1e-12 or nv <= 1e-12:
        return np.nan

    uu = u / nu
    vv = v / nv
    dot = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    ang = float(np.arccos(dot))

    if axial:
        # identify v and -v
        ang = min(ang, np.pi - ang)

    return ang


def principal_response_eig(
    t_xx: np.ndarray,
    t_xy: np.ndarray,
    t_yx: np.ndarray,
    t_yy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      eig_major, eig_minor, theta_major
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


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


# ---------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------


def load_from_bundle(bundle_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle_dir = Path(bundle_dir)
    nodes = read_csv_required(bundle_dir / "scene_nodes.csv").copy()
    edges = read_csv_required(bundle_dir / "scene_edges.csv").copy()
    seam = read_csv_required(bundle_dir / "scene_seam.csv").copy()

    # standardize edge endpoint names if needed
    src_col = first_existing(edges.columns, SRC_CANDIDATES)
    dst_col = first_existing(edges.columns, DST_CANDIDATES)
    if src_col is None or dst_col is None:
        raise ValueError(f"Could not identify edge endpoint columns in bundle edges: {list(edges.columns)}")
    if src_col != "src_id" or dst_col != "dst_id":
        edges = edges.rename(columns={src_col: "src_id", dst_col: "dst_id"})

    return nodes, edges, seam


def load_direct(
    phase_csv: str,
    edges_csv: str,
    seam_csv: str,
    response_csv: str,
    lazarus_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase = read_csv_required(phase_csv).copy()
    edges = read_csv_required(edges_csv).copy()
    seam = read_csv_required(seam_csv).copy()
    rsp = read_csv_required(response_csv).copy()
    laz = read_csv_required(lazarus_csv).copy()

    # nodes
    keep_phase = [c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase", "distance_to_seam", "fim_theta"] if c in phase.columns]
    nodes = phase[keep_phase].copy()
    if "node_id" not in nodes.columns:
        nodes = nodes.reset_index(drop=True)
        nodes["node_id"] = nodes.index.astype(int)

    # response enrich
    keep_rsp = [c for c in ["node_id", "r", "alpha", "response_strength", "T_xx", "T_xy", "T_yx", "T_yy", "rsp_theta"] if c in rsp.columns]
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in nodes.columns and c in rsp.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]
    nodes = nodes.merge(rsp[keep_rsp], on=join_cols, how="left")

    # lazarus enrich
    keep_laz = [c for c in ["node_id", "r", "alpha", "lazarus_score"] if c in laz.columns]
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in nodes.columns and c in laz.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]
    nodes = nodes.merge(laz[keep_laz], on=join_cols, how="left")

    # derive response theta if needed
    if "rsp_theta" not in nodes.columns and {"T_xx", "T_xy", "T_yx", "T_yy"}.issubset(nodes.columns):
        eig1, eig2, theta = principal_response_eig(
            pd.to_numeric(nodes["T_xx"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(nodes["T_xy"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(nodes["T_yx"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(nodes["T_yy"], errors="coerce").to_numpy(dtype=float),
        )
        nodes["rsp_eig_major"] = eig1
        nodes["rsp_eig_minor"] = eig2
        nodes["rsp_theta"] = theta

    # standardize edge endpoints
    src_col = first_existing(edges.columns, SRC_CANDIDATES)
    dst_col = first_existing(edges.columns, DST_CANDIDATES)
    if src_col is None or dst_col is None:
        raise ValueError(f"Could not identify edge endpoint columns: {list(edges.columns)}")
    if src_col != "src_id" or dst_col != "dst_id":
        edges = edges.rename(columns={src_col: "src_id", dst_col: "dst_id"})

    return nodes, edges, seam


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cfg.bundle_dir:
        bundle_path = Path(cfg.bundle_dir)
        if bundle_path.exists():
            return load_from_bundle(bundle_path)

    return load_direct(
        phase_csv=cfg.phase_csv,
        edges_csv=cfg.edges_csv,
        seam_csv=cfg.seam_csv,
        response_csv=cfg.response_csv,
        lazarus_csv=cfg.lazarus_csv,
    )


# ---------------------------------------------------------------------
# alignment computation
# ---------------------------------------------------------------------


def ensure_response_theta(nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()

    if "rsp_theta" in nodes.columns and pd.to_numeric(nodes["rsp_theta"], errors="coerce").notna().any():
        nodes["rsp_theta"] = pd.to_numeric(nodes["rsp_theta"], errors="coerce")
        return nodes

    if {"T_xx", "T_xy", "T_yx", "T_yy"}.issubset(nodes.columns):
        eig1, eig2, theta = principal_response_eig(
            pd.to_numeric(nodes["T_xx"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(nodes["T_xy"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(nodes["T_yx"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(nodes["T_yy"], errors="coerce").to_numpy(dtype=float),
        )
        nodes["rsp_eig_major"] = eig1
        nodes["rsp_eig_minor"] = eig2
        nodes["rsp_theta"] = theta
        return nodes

    raise ValueError("Could not determine response principal direction. Need rsp_theta or T_xx/T_xy/T_yx/T_yy.")


def choose_transport_theta_column(nodes: pd.DataFrame) -> str:
    col = first_existing(nodes.columns, THETA_CANDIDATES)
    if col is None:
        raise ValueError(
            "Could not identify a transport-angle column. "
            f"Tried: {THETA_CANDIDATES}. Available columns: {list(nodes.columns)}"
        )
    return col


def compute_edge_alignment(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    theta_col: str,
    use_axial_comparison: bool,
) -> pd.DataFrame:
    lookup = nodes.set_index("node_id")

    rows: list[dict] = []
    for _, row in edges.iterrows():
        i = int(row["src_id"])
        j = int(row["dst_id"])
        if i not in lookup.index or j not in lookup.index:
            continue

        ni = lookup.loc[i]
        nj = lookup.loc[j]

        theta_i = pd.to_numeric(ni[theta_col], errors="coerce")
        theta_j = pd.to_numeric(nj[theta_col], errors="coerce")
        rsp_i = pd.to_numeric(ni["rsp_theta"], errors="coerce")
        rsp_j = pd.to_numeric(nj["rsp_theta"], errors="coerce")

        if not np.isfinite([theta_i, theta_j, rsp_i, rsp_j]).all():
            continue

        dtheta = float(wrap_angle(theta_j - theta_i))
        v_i = angle_from_theta(float(rsp_i))
        v_j = angle_from_theta(float(rsp_j))
        v_i_trans = rotation_matrix(dtheta) @ v_i

        misalignment_rad = angle_between_vectors(v_i_trans, v_j, axial=use_axial_comparison)
        misalignment_deg = float(np.degrees(misalignment_rad)) if np.isfinite(misalignment_rad) else np.nan

        out = {
            "src_id": i,
            "dst_id": j,
            "src_mds1": float(pd.to_numeric(ni["mds1"], errors="coerce")) if "mds1" in ni.index else np.nan,
            "src_mds2": float(pd.to_numeric(ni["mds2"], errors="coerce")) if "mds2" in ni.index else np.nan,
            "dst_mds1": float(pd.to_numeric(nj["mds1"], errors="coerce")) if "mds1" in nj.index else np.nan,
            "dst_mds2": float(pd.to_numeric(nj["mds2"], errors="coerce")) if "mds2" in nj.index else np.nan,
            "src_theta_transport": float(theta_i),
            "dst_theta_transport": float(theta_j),
            "src_rsp_theta": float(rsp_i),
            "dst_rsp_theta": float(rsp_j),
            "transport_delta_theta": dtheta,
            "transport_delta_theta_deg": float(np.degrees(dtheta)),
            "misalignment_rad": misalignment_rad,
            "misalignment_deg": misalignment_deg,
        }

        # optional scalar enrichments
        for col in [
            "signed_phase",
            "distance_to_seam",
            "lazarus_score",
            "response_strength",
        ]:
            if col in ni.index:
                out[f"src_{col}"] = float(pd.to_numeric(ni[col], errors="coerce"))
            if col in nj.index:
                out[f"dst_{col}"] = float(pd.to_numeric(nj[col], errors="coerce"))

        out["edge_distance_to_seam_mid"] = np.nanmean(
            [out.get("src_distance_to_seam", np.nan), out.get("dst_distance_to_seam", np.nan)]
        )
        out["edge_signed_phase_mid"] = np.nanmean(
            [out.get("src_signed_phase", np.nan), out.get("dst_signed_phase", np.nan)]
        )
        out["edge_response_strength_mid"] = np.nanmean(
            [out.get("src_response_strength", np.nan), out.get("dst_response_strength", np.nan)]
        )
        out["edge_lazarus_mid"] = np.nanmean(
            [out.get("src_lazarus_score", np.nan), out.get("dst_lazarus_score", np.nan)]
        )

        hol_col = first_existing(edges.columns, EDGE_HOLONOMY_CANDIDATES)
        if hol_col is not None and hol_col in row.index:
            out["edge_holonomy_proxy"] = float(pd.to_numeric(row[hol_col], errors="coerce"))
        else:
            out["edge_holonomy_proxy"] = np.nan

        rows.append(out)

        # symmetric comparison j -> i for node-level averaging fairness
        dtheta_rev = float(wrap_angle(theta_i - theta_j))
        v_j_trans = rotation_matrix(dtheta_rev) @ v_j
        mis_rev = angle_between_vectors(v_j_trans, v_i, axial=use_axial_comparison)
        mis_rev_deg = float(np.degrees(mis_rev)) if np.isfinite(mis_rev) else np.nan

        out_rev = {
            "src_id": j,
            "dst_id": i,
            "src_mds1": out["dst_mds1"],
            "src_mds2": out["dst_mds2"],
            "dst_mds1": out["src_mds1"],
            "dst_mds2": out["src_mds2"],
            "src_theta_transport": float(theta_j),
            "dst_theta_transport": float(theta_i),
            "src_rsp_theta": float(rsp_j),
            "dst_rsp_theta": float(rsp_i),
            "transport_delta_theta": dtheta_rev,
            "transport_delta_theta_deg": float(np.degrees(dtheta_rev)),
            "misalignment_rad": mis_rev,
            "misalignment_deg": mis_rev_deg,
            "src_signed_phase": out.get("dst_signed_phase", np.nan),
            "dst_signed_phase": out.get("src_signed_phase", np.nan),
            "src_distance_to_seam": out.get("dst_distance_to_seam", np.nan),
            "dst_distance_to_seam": out.get("src_distance_to_seam", np.nan),
            "src_lazarus_score": out.get("dst_lazarus_score", np.nan),
            "dst_lazarus_score": out.get("src_lazarus_score", np.nan),
            "src_response_strength": out.get("dst_response_strength", np.nan),
            "dst_response_strength": out.get("src_response_strength", np.nan),
            "edge_distance_to_seam_mid": out["edge_distance_to_seam_mid"],
            "edge_signed_phase_mid": out["edge_signed_phase_mid"],
            "edge_response_strength_mid": out["edge_response_strength_mid"],
            "edge_lazarus_mid": out["edge_lazarus_mid"],
            "edge_holonomy_proxy": out["edge_holonomy_proxy"],
        }
        rows.append(out_rev)

    return pd.DataFrame(rows)


def compute_node_alignment(nodes: pd.DataFrame, edge_alignment: pd.DataFrame) -> pd.DataFrame:
    if len(edge_alignment) == 0:
        out = nodes.copy()
        out["transport_align_mean_deg"] = np.nan
        out["transport_align_max_deg"] = np.nan
        out["transport_align_std_deg"] = np.nan
        out["transport_align_n_neighbors"] = 0

        if "node_holonomy_proxy" in out.columns:
            out["node_holonomy_proxy"] = pd.to_numeric(out["node_holonomy_proxy"], errors="coerce")
        else:
            out["node_holonomy_proxy"] = np.nan

        return out

    agg = (
        edge_alignment.groupby("src_id", as_index=False)
        .agg(
            transport_align_mean_deg=("misalignment_deg", "mean"),
            transport_align_max_deg=("misalignment_deg", "max"),
            transport_align_std_deg=("misalignment_deg", "std"),
            transport_align_n_neighbors=("dst_id", "count"),
            transport_holonomy_edge_mean=("edge_holonomy_proxy", "mean"),
            transport_edge_seam_mid_mean=("edge_distance_to_seam_mid", "mean"),
        )
        .rename(columns={"src_id": "node_id"})
    )

    out = nodes.merge(agg, on="node_id", how="left")

    if "node_holonomy_proxy" in out.columns:
        out["node_holonomy_proxy"] = pd.to_numeric(out["node_holonomy_proxy"], errors="coerce")
    else:
        hol_col = first_existing(out.columns, NODE_HOLONOMY_CANDIDATES)
        if hol_col is not None:
            out["node_holonomy_proxy"] = pd.to_numeric(out[hol_col], errors="coerce")
        else:
            out["node_holonomy_proxy"] = np.nan

    return out

# ---------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------


def write_summary(
    outpath: Path,
    theta_col: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seam_threshold: float,
) -> None:
    n_nodes = len(nodes)
    n_edges = len(edges) // 2 if len(edges) else 0  # undirected original count
    corr_node_seam = safe_corr(nodes["transport_align_mean_deg"], nodes["distance_to_seam"]) if "distance_to_seam" in nodes.columns else np.nan
    corr_node_hol = safe_corr(nodes["transport_align_mean_deg"], nodes["node_holonomy_proxy"])
    corr_edge_seam = safe_corr(edges["misalignment_deg"], edges["edge_distance_to_seam_mid"]) if len(edges) else np.nan
    corr_edge_hol = safe_corr(edges["misalignment_deg"], edges["edge_holonomy_proxy"]) if len(edges) else np.nan

    seam_mask = (
        pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold
        if "distance_to_seam" in nodes.columns
        else pd.Series(False, index=nodes.index)
    )
    seam_mean = float(pd.to_numeric(nodes.loc[seam_mask, "transport_align_mean_deg"], errors="coerce").mean()) if seam_mask.any() else np.nan
    off_mean = float(pd.to_numeric(nodes.loc[~seam_mask, "transport_align_mean_deg"], errors="coerce").mean()) if (~seam_mask).any() else np.nan

    lines = [
        "=== Identity Transport Alignment Toy Summary ===",
        "",
        f"transport_theta_column = {theta_col}",
        f"n_nodes = {n_nodes}",
        f"n_edges_undirected = {n_edges}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Node-level means",
        f"  mean transport_align_mean_deg = {float(pd.to_numeric(nodes['transport_align_mean_deg'], errors='coerce').mean()):.4f}",
        f"  mean transport_align_max_deg  = {float(pd.to_numeric(nodes['transport_align_max_deg'], errors='coerce').mean()):.4f}",
        f"  seam-band mean misalignment   = {seam_mean:.4f}",
        f"  off-seam mean misalignment    = {off_mean:.4f}",
        "",
        "Correlations",
        f"  corr(node misalignment, distance_to_seam) = {corr_node_seam:.4f}",
        f"  corr(node misalignment, node holonomy)    = {corr_node_hol:.4f}",
        f"  corr(edge misalignment, edge seam mid)    = {corr_edge_seam:.4f}",
        f"  corr(edge misalignment, edge holonomy)    = {corr_edge_hol:.4f}",
        "",
        "Highest-misalignment nodes",
    ]

    top = nodes.sort_values("transport_align_mean_deg", ascending=False).head(10)
    keep_cols = [c for c in ["node_id", "r", "alpha", "transport_align_mean_deg", "distance_to_seam", "node_holonomy_proxy"] if c in top.columns]
    for _, row in top[keep_cols].iterrows():
        lines.append(
            "  "
            + ", ".join(
                f"{col}={float(row[col]):.4f}" if col not in {"node_id"} else f"{col}={int(row[col])}"
                for col in keep_cols
            )
        )

    outpath.write_text("\n".join(lines), encoding="utf-8")


def render_plate(
    outpath: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seam: pd.DataFrame,
    theta_col: str,
    seam_threshold: float,
) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.0, 1.4], height_ratios=[1.0, 1.0])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_sc1 = fig.add_subplot(gs[0, 1])
    ax_sc2 = fig.add_subplot(gs[1, 1])

    # edge web
    for _, row in edges.iterrows():
        if int(row["src_id"]) > int(row["dst_id"]):
            continue  # avoid drawing both symmetric copies
        arr = [
            row["src_mds1"], row["src_mds2"], row["dst_mds1"], row["dst_mds2"], row["edge_signed_phase_mid"]
        ]
        if not np.isfinite(arr).all():
            continue
        ax_main.plot(
            [row["src_mds1"], row["dst_mds1"]],
            [row["src_mds2"], row["dst_mds2"]],
            color=plt.cm.coolwarm((float(row["edge_signed_phase_mid"]) + 1.0) / 2.0),
            alpha=0.28,
            linewidth=1.0,
            zorder=1,
        )

    # seam
    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.2, alpha=0.65, zorder=2)
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.5, alpha=0.95, zorder=3)

    # nodes colored by transport misalignment
    sc = ax_main.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce"),
        s=82,
        cmap="viridis",
        alpha=0.95,
        linewidths=0.4,
        edgecolors="white",
        zorder=4,
    )
    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.045, pad=0.03)
    cbar.set_label("mean transport misalignment (deg)")

    # seam-band ring
    if "distance_to_seam" in nodes.columns:
        seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold
        seam_nodes = nodes[seam_mask].copy()
        if len(seam_nodes):
            ax_main.scatter(
                seam_nodes["mds1"],
                seam_nodes["mds2"],
                s=170,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                zorder=5,
            )

    ax_main.set_title("Transport-aware response-field misalignment")
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.text(
        0.02,
        0.97,
        f"transport angle = {theta_col}\nblack rings = seam neighborhood",
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )

    # scatter 1: seam
    if "distance_to_seam" in nodes.columns:
        x = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
        y = pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce")
        ax_sc1.scatter(x, y, s=34, alpha=0.85)
        ax_sc1.set_xlabel("distance to seam")
        ax_sc1.set_ylabel("mean misalignment (deg)")
        ax_sc1.set_title("Misalignment vs seam distance")
        ax_sc1.grid(alpha=0.15)

    # scatter 2: holonomy proxy if present
    hol = pd.to_numeric(nodes["node_holonomy_proxy"], errors="coerce")
    mis = pd.to_numeric(nodes["transport_align_mean_deg"], errors="coerce")
    mask = hol.notna() & mis.notna()
    if int(mask.sum()) >= 3:
        ax_sc2.scatter(hol[mask], mis[mask], s=34, alpha=0.85)
        ax_sc2.set_xlabel("node holonomy proxy")
        ax_sc2.set_ylabel("mean misalignment (deg)")
        ax_sc2.set_title("Misalignment vs holonomy proxy")
    else:
        ax_sc2.text(
            0.5, 0.5, "No node holonomy proxy available",
            ha="center", va="center", fontsize=11,
        )
        ax_sc2.set_title("Misalignment vs holonomy proxy")
        ax_sc2.set_xticks([])
        ax_sc2.set_yticks([])
    ax_sc2.grid(alpha=0.15)

    fig.suptitle("Identity Transport Alignment Toy", fontsize=18)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy transport-aware alignment on the PAM manifold.")
    parser.add_argument("--bundle-dir", default=Config.bundle_dir)
    parser.add_argument("--phase-csv", default=Config.phase_csv)
    parser.add_argument("--edges-csv", default=Config.edges_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--response-csv", default=Config.response_csv)
    parser.add_argument("--lazarus-csv", default=Config.lazarus_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--no-axial", action="store_true", help="Use directed instead of axial vector comparison.")
    args = parser.parse_args()

    cfg = Config(
        bundle_dir=args.bundle_dir,
        phase_csv=args.phase_csv,
        edges_csv=args.edges_csv,
        seam_csv=args.seam_csv,
        response_csv=args.response_csv,
        lazarus_csv=args.lazarus_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        use_axial_comparison=not args.no_axial,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, seam = load_data(cfg)
    nodes = ensure_response_theta(nodes)
    theta_col = choose_transport_theta_column(nodes)

    # normalize key columns
    to_numeric_inplace(
        nodes,
        [
            "node_id", "r", "alpha", "mds1", "mds2", "signed_phase",
            "distance_to_seam", "lazarus_score", "response_strength",
            "rsp_theta", theta_col,
        ],
    )
    to_numeric_inplace(edges, ["src_id", "dst_id"])
    if "src_id" not in edges.columns or "dst_id" not in edges.columns:
        raise ValueError("Edges must contain src_id and dst_id after normalization.")

    edge_alignment = compute_edge_alignment(
        nodes=nodes,
        edges=edges,
        theta_col=theta_col,
        use_axial_comparison=cfg.use_axial_comparison,
    )
    node_alignment = compute_node_alignment(nodes, edge_alignment)

    node_csv = outdir / "node_transport_alignment.csv"
    edge_csv = outdir / "edge_transport_alignment.csv"
    txt_out = outdir / "identity_transport_alignment_summary.txt"
    png_out = outdir / "identity_transport_alignment_plate.png"

    node_alignment.to_csv(node_csv, index=False)
    edge_alignment.to_csv(edge_csv, index=False)
    write_summary(
        outpath=txt_out,
        theta_col=theta_col,
        nodes=node_alignment,
        edges=edge_alignment,
        seam_threshold=cfg.seam_threshold,
    )
    render_plate(
        outpath=png_out,
        nodes=node_alignment,
        edges=edge_alignment,
        seam=seam,
        theta_col=theta_col,
        seam_threshold=cfg.seam_threshold,
    )

    print(node_csv)
    print(edge_csv)
    print(txt_out)
    print(png_out)


if __name__ == "__main__":
    main()
