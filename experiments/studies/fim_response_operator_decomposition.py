#!/usr/bin/env python3
"""
fim_response_operator_decomposition.py

Decompose the local response operator T into canonical 2x2 parts and test
which component drives complex-compatibility failure near the seam.

For each node, decompose:

    T = scalar_part + symmetric_traceless_part + antisymmetric_part

where:
- scalar_part            = (tr(T)/2) I
- symmetric_traceless    = 0.5 * (T + T^T) - (tr(T)/2) I
- antisymmetric_part     = 0.5 * (T - T^T)

We compute:
- Frobenius norm of each part
- commutator norm with canonical complex structure J
- seam localization summaries
- correlations with seam distance, holonomy proxy, and transport mismatch

Inputs
------
outputs/obs022_scene_bundle/scene_nodes.csv
outputs/obs023_transport_misalignment/obs023_transport_misalignment_nodes.csv
outputs/fim_response_complex_compatibility/response_complex_compatibility_nodes.csv
outputs/obs022_scene_bundle/scene_seam.csv

Outputs
-------
outputs/fim_response_operator_decomposition/
  response_operator_decomposition_nodes.csv
  response_operator_decomposition_summary.txt
  response_operator_decomposition_panel.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    transport_nodes_csv: str = "outputs/obs023_transport_misalignment/obs023_transport_misalignment_nodes.csv"
    compatibility_nodes_csv: str = "outputs/fim_response_complex_compatibility/response_complex_compatibility_nodes.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/fim_response_operator_decomposition"
    seam_threshold: float = 0.15
    top_k_labels: int = 10


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def fro_norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, ord="fro"))


def canonical_J() -> np.ndarray:
    return np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def response_matrix(row: pd.Series) -> np.ndarray | None:
    cols = ["T_xx", "T_xy", "T_yx", "T_yy"]
    if not all(c in row.index for c in cols):
        return None
    vals = [pd.to_numeric(row[c], errors="coerce") for c in cols]
    if any(pd.isna(v) for v in vals):
        return None
    return np.array([[float(vals[0]), float(vals[1])], [float(vals[2]), float(vals[3])]], dtype=float)


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    transport = pd.read_csv(cfg.transport_nodes_csv)
    compat = pd.read_csv(cfg.compatibility_nodes_csv)
    seam = pd.read_csv(cfg.seam_csv)

    for df in (nodes, transport, compat, seam):
        for col in df.columns:
            if col not in {"path_id", "path_family"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    keep_transport = [c for c in ["node_id", "transport_align_mean_deg", "transport_align_max_deg"] if c in transport.columns]
    keep_compat = [c for c in ["node_id", "commutator_norm_rsp", "anticommutator_norm_rsp"] if c in compat.columns]

    merged = nodes.merge(transport[keep_transport], on="node_id", how="left")
    merged = merged.merge(compat[keep_compat], on="node_id", how="left")

    return merged, seam


def decompose_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    J = canonical_J()
    rows: list[dict] = []

    for _, row in nodes.iterrows():
        node_id = int(row["node_id"])
        T = response_matrix(row)

        out = {"node_id": node_id}

        if T is None:
            out.update(
                {
                    "scalar_norm": np.nan,
                    "sym_traceless_norm": np.nan,
                    "antisymmetric_norm": np.nan,
                    "total_norm": np.nan,
                    "scalar_trace_coeff": np.nan,
                    "anisotropy_ratio": np.nan,
                    "scalar_commutator_norm": np.nan,
                    "sym_traceless_commutator_norm": np.nan,
                    "antisymmetric_commutator_norm": np.nan,
                    "dominant_component": np.nan,
                }
            )
            rows.append(out)
            continue

        tr = float(np.trace(T))
        I = np.eye(2, dtype=float)

        scalar = 0.5 * tr * I
        sym = 0.5 * (T + T.T)
        antisym = 0.5 * (T - T.T)
        sym_traceless = sym - scalar

        scalar_norm = fro_norm(scalar)
        sym_traceless_norm = fro_norm(sym_traceless)
        antisymmetric_norm = fro_norm(antisym)
        total_norm = fro_norm(T)

        comp_map = {
            "scalar": scalar_norm,
            "sym_traceless": sym_traceless_norm,
            "antisymmetric": antisymmetric_norm,
        }
        dominant_component = max(comp_map, key=comp_map.get)

        out.update(
            {
                "scalar_norm": scalar_norm,
                "sym_traceless_norm": sym_traceless_norm,
                "antisymmetric_norm": antisymmetric_norm,
                "total_norm": total_norm,
                "scalar_trace_coeff": 0.5 * tr,
                "anisotropy_ratio": sym_traceless_norm / total_norm if total_norm > 1e-12 else np.nan,
                "scalar_commutator_norm": fro_norm(commutator(scalar, J)),
                "sym_traceless_commutator_norm": fro_norm(commutator(sym_traceless, J)),
                "antisymmetric_commutator_norm": fro_norm(commutator(antisym, J)),
                "dominant_component": dominant_component,
            }
        )
        rows.append(out)

    return nodes.merge(pd.DataFrame(rows), on="node_id", how="left")


def build_summary(nodes: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    components = [
        ("scalar_norm", "scalar"),
        ("sym_traceless_norm", "sym_traceless"),
        ("antisymmetric_norm", "antisymmetric"),
    ]

    lines = [
        "=== FIM Response Operator Decomposition Summary ===",
        "",
        f"n_nodes = {len(nodes)}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Mean component norms",
    ]
    for col, label in components:
        lines.append(f"  mean {label} = {float(pd.to_numeric(nodes[col], errors='coerce').mean()):.6f}")

    lines.extend(
        [
            "",
            "Seam localization",
            f"  seam-band mean scalar_norm         = {float(pd.to_numeric(nodes.loc[seam_mask, 'scalar_norm'], errors='coerce').mean()):.6f}",
            f"  off-seam mean scalar_norm          = {float(pd.to_numeric(nodes.loc[~seam_mask, 'scalar_norm'], errors='coerce').mean()):.6f}",
            f"  seam-band mean sym_traceless_norm  = {float(pd.to_numeric(nodes.loc[seam_mask, 'sym_traceless_norm'], errors='coerce').mean()):.6f}",
            f"  off-seam mean sym_traceless_norm   = {float(pd.to_numeric(nodes.loc[~seam_mask, 'sym_traceless_norm'], errors='coerce').mean()):.6f}",
            f"  seam-band mean antisymmetric_norm  = {float(pd.to_numeric(nodes.loc[seam_mask, 'antisymmetric_norm'], errors='coerce').mean()):.6f}",
            f"  off-seam mean antisymmetric_norm   = {float(pd.to_numeric(nodes.loc[~seam_mask, 'antisymmetric_norm'], errors='coerce').mean()):.6f}",
            "",
            "Commutator decomposition",
            f"  mean scalar_commutator_norm        = {float(pd.to_numeric(nodes['scalar_commutator_norm'], errors='coerce').mean()):.6f}",
            f"  mean sym_traceless_commutator_norm = {float(pd.to_numeric(nodes['sym_traceless_commutator_norm'], errors='coerce').mean()):.6f}",
            f"  mean antisymmetric_commutator_norm = {float(pd.to_numeric(nodes['antisymmetric_commutator_norm'], errors='coerce').mean()):.6f}",
            "",
            "Correlations with response / seam observables",
            f"  corr(sym_traceless_norm, commutator_norm_rsp)   = {safe_corr(nodes['sym_traceless_norm'], nodes['commutator_norm_rsp']):.4f}",
            f"  corr(antisymmetric_norm, commutator_norm_rsp)   = {safe_corr(nodes['antisymmetric_norm'], nodes['commutator_norm_rsp']):.4f}",
            f"  corr(scalar_norm, commutator_norm_rsp)          = {safe_corr(nodes['scalar_norm'], nodes['commutator_norm_rsp']):.4f}",
            f"  corr(sym_traceless_norm, transport_align_mean)  = {safe_corr(nodes['sym_traceless_norm'], nodes['transport_align_mean_deg']):.4f}",
            f"  corr(antisymmetric_norm, transport_align_mean)  = {safe_corr(nodes['antisymmetric_norm'], nodes['transport_align_mean_deg']):.4f}",
            f"  corr(sym_traceless_norm, distance_to_seam)      = {safe_corr(nodes['sym_traceless_norm'], nodes['distance_to_seam']):.4f}",
            f"  corr(antisymmetric_norm, distance_to_seam)      = {safe_corr(nodes['antisymmetric_norm'], nodes['distance_to_seam']):.4f}",
            "",
            "Dominant components",
        ]
    )

    dom_counts = nodes["dominant_component"].value_counts(dropna=False)
    for k, v in dom_counts.items():
        lines.append(f"  {k}: {int(v)}")

    lines.extend(["", "Top anisotropy / incompatibility nodes"])
    top = nodes.sort_values(["sym_traceless_norm", "commutator_norm_rsp"], ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, "
            f"r={float(row['r']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"sym_traceless_norm={float(row['sym_traceless_norm']):.6f}, "
            f"antisymmetric_norm={float(row['antisymmetric_norm']):.6f}, "
            f"commutator_norm_rsp={float(row['commutator_norm_rsp']):.6f}, "
            f"distance_to_seam={float(row['distance_to_seam']):.4f}"
        )

    return "\n".join(lines)


def render_panel(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.0, 1.3], height_ratios=[1.0, 1.0])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[1, 1])

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=6.0, alpha=0.65, zorder=1)
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.8, alpha=0.96, zorder=2)

    sc = ax_main.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["sym_traceless_norm"], errors="coerce"),
        cmap="viridis",
        s=95,
        alpha=0.96,
        linewidths=0.35,
        edgecolors="white",
        zorder=3,
    )

    seam_nodes = nodes[seam_mask]
    if len(seam_nodes):
        ax_main.scatter(
            seam_nodes["mds1"],
            seam_nodes["mds2"],
            s=170,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            zorder=4,
        )

    top = nodes.sort_values(["sym_traceless_norm", "commutator_norm_rsp"], ascending=False).head(cfg.top_k_labels)
    for _, row in top.iterrows():
        ax_main.scatter(
            [row["mds1"]],
            [row["mds2"]],
            s=145,
            facecolors="none",
            edgecolors="#FFD166",
            linewidths=1.8,
            zorder=5,
        )
        ax_main.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=6,
        )

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.038, pad=0.02)
    cbar.set_label("symmetric traceless norm")

    ax_main.set_title("Response anisotropy on the phase manifold", fontsize=17, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.text(
        0.02,
        0.97,
        "black seam = detected phase boundary\nblack rings = seam neighborhood\nyellow labels = top anisotropy / incompatibility nodes",
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )

    vals = [
        float(pd.to_numeric(nodes.loc[seam_mask, "sym_traceless_norm"], errors="coerce").mean()),
        float(pd.to_numeric(nodes.loc[~seam_mask, "sym_traceless_norm"], errors="coerce").mean()),
    ]
    ax_bar.bar(["seam-band", "off-seam"], vals, alpha=0.9)
    ax_bar.set_ylabel("mean symmetric traceless norm")
    ax_bar.set_title("Seam localization", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    x = pd.to_numeric(nodes["sym_traceless_norm"], errors="coerce")
    y = pd.to_numeric(nodes["commutator_norm_rsp"], errors="coerce")
    mask = x.notna() & y.notna()
    ax_sc.scatter(x[mask], y[mask], s=38, alpha=0.88)
    ax_sc.set_xlabel("symmetric traceless norm")
    ax_sc.set_ylabel("commutator norm")
    ax_sc.set_title("Anisotropy vs incompatibility", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)
    ax_sc.text(
        0.98,
        0.05,
        f"corr = {safe_corr(x, y):.3f}",
        transform=ax_sc.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("PAM Observatory — Response Operator Decomposition", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose response operator and test seam incompatibility drivers.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--transport-nodes-csv", default=Config.transport_nodes_csv)
    parser.add_argument("--compatibility-nodes-csv", default=Config.compatibility_nodes_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        transport_nodes_csv=args.transport_nodes_csv,
        compatibility_nodes_csv=args.compatibility_nodes_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, seam = load_inputs(cfg)
    nodes = decompose_nodes(nodes)

    csv_path = outdir / "response_operator_decomposition_nodes.csv"
    txt_path = outdir / "response_operator_decomposition_summary.txt"
    png_path = outdir / "response_operator_decomposition_panel.png"

    nodes.to_csv(csv_path, index=False)
    txt_path.write_text(build_summary(nodes, cfg.seam_threshold), encoding="utf-8")
    render_panel(cfg, nodes, seam, png_path)

    print(csv_path)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
