#!/usr/bin/env python3
"""
OBS-028b — Diffusion mode analysis for the PAM observatory.

Purpose
-------
Give diffusion coordinates a fair, diffusion-native evaluation.

Rather than scoring diffusion maps mainly on local-neighborhood retention or
class separation alone, this study asks whether the leading diffusion modes
capture:

1. seam distance
2. branch-exit tendency
3. corridor residency tendency
4. hotspot / bottleneck concentration
5. multiscale behavior across diffusion time t

Inputs
------
outputs/obs022_scene_bundle/scene_nodes.csv
outputs/obs022_scene_bundle/scene_edges.csv
outputs/obs022_scene_bundle/scene_routes.csv

Outputs
-------
outputs/obs028b_diffusion_mode_analysis/
  diffusion_modes_nodes.csv
  diffusion_mode_scores.csv
  obs028b_diffusion_mode_analysis_summary.txt
  obs028b_diffusion_mode_analysis.png
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
    edges_csv: str = "outputs/obs022_scene_bundle/scene_edges.csv"
    routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    outdir: str = "outputs/obs028b_diffusion_mode_analysis"
    seam_threshold: float = 0.15
    hotspot_quantile: float = 0.85
    diffusion_times: tuple[int, ...] = (1, 2, 4, 8)
    kernel_scale: str = "median"  # {"median", "mean"}
    top_k_labels: int = 8


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def safe_corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    aa = pd.to_numeric(pd.Series(a), errors="coerce")
    bb = pd.to_numeric(pd.Series(b), errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def safe_mean(s: pd.Series | np.ndarray) -> float:
    ss = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(ss.mean()) if ss.notna().any() else float("nan")


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
    out = routes.copy()
    fam = out.get("path_family", pd.Series(index=out.index, dtype=object))
    is_branch = pd.to_numeric(out.get("is_branch_away", 0), errors="coerce").fillna(0).eq(1)
    is_rep = pd.to_numeric(out.get("is_representative", 0), errors="coerce").fillna(0).eq(1)

    out["route_class"] = np.select(
        [
            is_branch,
            is_rep & fam.eq("stable_seam_corridor"),
            is_rep & fam.eq("reorganization_heavy"),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )
    return out


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    edges = pd.read_csv(cfg.edges_csv)
    routes = pd.read_csv(cfg.routes_csv)

    for df in (nodes, edges, routes):
        for col in df.columns:
            if col not in {"path_id", "path_family", "route_class", "hotspot_class"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    if "is_branch_away" not in routes.columns:
        routes["is_branch_away"] = 0
    if "is_representative" not in routes.columns:
        routes["is_representative"] = 0

    routes = classify_routes(routes)
    return nodes, edges, routes


def build_node_augmented_table(nodes: pd.DataFrame, routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = nodes.copy()

    # Family occupancy fields
    family_touch = (
        routes[routes["route_class"].isin(CLASS_ORDER)]
        .groupby(["node_id", "route_class"], as_index=False)
        .agg(n_rows=("path_id", "size"), n_paths=("path_id", "nunique"))
    )

    pivot = (
        family_touch.pivot(index="node_id", columns="route_class", values="n_rows")
        .fillna(0.0)
        .reset_index()
    )
    pivot.columns.name = None
    for cls in CLASS_ORDER:
        if cls not in pivot.columns:
            pivot[cls] = 0.0

    out = out.merge(pivot, on="node_id", how="left")
    for cls in CLASS_ORDER:
        out[cls] = pd.to_numeric(out[cls], errors="coerce").fillna(0.0)

    # Hotspots from available seam-side fields if not already present
    if "anisotropy_hotspot" not in out.columns and "sym_traceless_norm" in out.columns:
        thr_a = float(pd.to_numeric(out["sym_traceless_norm"], errors="coerce").quantile(cfg.hotspot_quantile))
        out["anisotropy_hotspot"] = (
            pd.to_numeric(out["sym_traceless_norm"], errors="coerce") >= thr_a
        ).astype(int)
    elif "anisotropy_hotspot" not in out.columns:
        out["anisotropy_hotspot"] = 0

    if "relational_hotspot" not in out.columns and "neighbor_direction_mismatch_mean" in out.columns:
        thr_r = float(
            pd.to_numeric(out["neighbor_direction_mismatch_mean"], errors="coerce").quantile(cfg.hotspot_quantile)
        )
        out["relational_hotspot"] = (
            pd.to_numeric(out["neighbor_direction_mismatch_mean"], errors="coerce") >= thr_r
        ).astype(int)
    elif "relational_hotspot" not in out.columns:
        out["relational_hotspot"] = 0

    if "shared_hotspot" not in out.columns:
        out["shared_hotspot"] = (
            (pd.to_numeric(out["anisotropy_hotspot"], errors="coerce").fillna(0) == 1)
            & (pd.to_numeric(out["relational_hotspot"], errors="coerce").fillna(0) == 1)
        ).astype(int)

    return out


def compute_graph_distance(nodes: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    node_ids = pd.to_numeric(nodes["node_id"], errors="coerce").astype(int).tolist()
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    D = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)

    for _, row in edges.iterrows():
        try:
            u = idx[int(row["src_id"])]
            v = idx[int(row["dst_id"])]
        except Exception:
            continue

        w = np.nan
        if "edge_cost" in edges.columns and pd.notna(row.get("edge_cost", np.nan)):
            w = float(row["edge_cost"])
        elif all(c in edges.columns for c in ["src_mds1", "src_mds2", "dst_mds1", "dst_mds2"]):
            try:
                w = float(np.hypot(float(row["dst_mds1"]) - float(row["src_mds1"]),
                                  float(row["dst_mds2"]) - float(row["src_mds2"])))
            except Exception:
                w = np.nan

        if not np.isfinite(w):
            w = 1.0

        D[u, v] = min(D[u, v], w)
        D[v, u] = min(D[v, u], w)

    # Floyd-Warshall, n is small
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])

    finite = D[np.isfinite(D) & (D > 0)]
    fill = float(finite.max()) if finite.size else 1.0
    D[~np.isfinite(D)] = fill
    return D


def build_diffusion_operator(D: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    positive = D[D > 0]
    if positive.size == 0:
        sigma = 1.0
    elif cfg.kernel_scale == "mean":
        sigma = float(np.mean(positive))
    else:
        sigma = float(np.median(positive))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    K = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(K, 0.0)

    row_sum = K.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 1e-12] = 1.0
    P = K / row_sum
    return P, sigma


def diffusion_eigendecomposition(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eig(P)
    vals = np.real(vals)
    vecs = np.real(vecs)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs


def diffusion_coords(vals: np.ndarray, vecs: np.ndarray, t: int) -> np.ndarray:
    # skip trivial first eigenvector
    lam = vals[1:3]
    psi = vecs[:, 1:3]
    return psi * (lam ** t)


def seam_separation_score(df: pd.DataFrame, seam_threshold: float, xcol: str, ycol: str) -> float:
    seam = pd.to_numeric(df["distance_to_seam"], errors="coerce") <= seam_threshold
    A = df.loc[seam, [xcol, ycol]].to_numpy(dtype=float)
    B = df.loc[~seam, [xcol, ycol]].to_numpy(dtype=float)
    if len(A) < 2 or len(B) < 2:
        return float("nan")
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    varA = float(np.mean(np.sum((A - muA) ** 2, axis=1)))
    varB = float(np.mean(np.sum((B - muB) ** 2, axis=1)))
    denom = np.sqrt(max(varA + varB, 1e-12))
    return float(np.linalg.norm(muA - muB) / denom)


def family_centroid_score(df: pd.DataFrame, xcol: str, ycol: str) -> float:
    centroids = []
    X = df[[xcol, ycol]].to_numpy(dtype=float)
    for cls in CLASS_ORDER:
        w = pd.to_numeric(df.get(cls, 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.sum(w) <= 0:
            continue
        mu = (X * w[:, None]).sum(axis=0) / np.sum(w)
        centroids.append(mu)
    if len(centroids) < 2:
        return float("nan")
    centroids = np.vstack(centroids)
    diffs = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    iu = np.triu_indices_from(dist, k=1)
    return float(np.mean(dist[iu]))


def shared_hotspot_compactness(df: pd.DataFrame, xcol: str, ycol: str) -> float:
    shared = pd.to_numeric(df.get("shared_hotspot", 0), errors="coerce").fillna(0) == 1
    X = df.loc[shared, [xcol, ycol]].to_numpy(dtype=float)
    if len(X) < 2:
        return float("nan")
    mu = X.mean(axis=0)
    return float(np.mean(np.sqrt(np.sum((X - mu) ** 2, axis=1))))


def bottleneck_proxy_score(df: pd.DataFrame, xcol: str, ycol: str) -> float:
    """
    Fairer diffusion-native proxy:
    shared hotspots should occupy a narrower subregion than non-shared nodes.
    Smaller ratio is better.
    """
    shared = pd.to_numeric(df.get("shared_hotspot", 0), errors="coerce").fillna(0) == 1
    Xs = df.loc[shared, [xcol, ycol]].to_numpy(dtype=float)
    Xn = df.loc[~shared, [xcol, ycol]].to_numpy(dtype=float)
    if len(Xs) < 2 or len(Xn) < 2:
        return float("nan")
    mu_s = Xs.mean(axis=0)
    mu_n = Xn.mean(axis=0)
    spread_s = float(np.mean(np.sqrt(np.sum((Xs - mu_s) ** 2, axis=1))))
    spread_n = float(np.mean(np.sqrt(np.sum((Xn - mu_n) ** 2, axis=1))))
    if spread_n <= 1e-12:
        return float("nan")
    return spread_s / spread_n


def axis_alignment_scores(df: pd.DataFrame, xcol: str, ycol: str) -> dict[str, float]:
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    seam = pd.to_numeric(df["distance_to_seam"], errors="coerce")
    branch = pd.to_numeric(df.get("branch_exit", 0), errors="coerce").fillna(0.0)
    corridor = pd.to_numeric(df.get("stable_seam_corridor", 0), errors="coerce").fillna(0.0)
    shared = pd.to_numeric(df.get("shared_hotspot", 0), errors="coerce").fillna(0.0)

    candidates = {
        "seam_distance": max(abs(safe_corr(x, seam)), abs(safe_corr(y, seam))),
        "branch_exit": max(abs(safe_corr(x, branch)), abs(safe_corr(y, branch))),
        "corridor": max(abs(safe_corr(x, corridor)), abs(safe_corr(y, corridor))),
        "shared_hotspot": max(abs(safe_corr(x, shared)), abs(safe_corr(y, shared))),
    }
    return candidates


def render_panel(nodes: pd.DataFrame, scores: pd.DataFrame, vals: np.ndarray, outpath: Path, cfg: Config) -> None:
    times = list(scores["diffusion_time"])
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 1.2, 1.0, 1.0])

    # top row: two selected diffusion times
    chosen = [times[0], times[min(2, len(times) - 1)]]
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold

    for i, t in enumerate(chosen):
        ax = fig.add_subplot(gs[0, i])
        xcol, ycol = f"diff1_t{t}", f"diff2_t{t}"
        sc = ax.scatter(
            nodes[xcol], nodes[ycol],
            c=pd.to_numeric(nodes["distance_to_seam"], errors="coerce"),
            cmap="magma_r",
            s=80, alpha=0.95, linewidths=0.35, edgecolors="white",
        )
        ax.scatter(
            nodes.loc[seam_mask, xcol],
            nodes.loc[seam_mask, ycol],
            s=135, facecolors="none", edgecolors="black", linewidths=1.0,
        )
        shared = pd.to_numeric(nodes.get("shared_hotspot", 0), errors="coerce").fillna(0) == 1
        top = nodes.loc[shared].copy().head(cfg.top_k_labels)
        for _, row in top.iterrows():
            ax.text(
                float(row[xcol]) + 0.02,
                float(row[ycol]) + 0.02,
                f"{int(row['node_id'])}",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.8", alpha=0.9),
            )
        ax.set_title(f"Diffusion coordinates (t={t})", fontsize=14, pad=8)
        ax.set_xlabel("diff 1")
        ax.set_ylabel("diff 2")
        ax.grid(alpha=0.12)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    # eigenvalues
    ax_eig = fig.add_subplot(gs[0, 2])
    lam = vals[:10]
    ax_eig.plot(np.arange(len(lam)), lam, marker="o")
    ax_eig.set_title("Leading diffusion eigenvalues", fontsize=14, pad=8)
    ax_eig.set_xlabel("index")
    ax_eig.set_ylabel("eigenvalue")
    ax_eig.grid(alpha=0.15)

    # mode scores by t
    ax_score = fig.add_subplot(gs[0, 3])
    ax_score.plot(scores["diffusion_time"], scores["seam_separation"], marker="o", label="seam sep")
    ax_score.plot(scores["diffusion_time"], scores["family_centroid_separation"], marker="o", label="family sep")
    ax_score.plot(scores["diffusion_time"], scores["bottleneck_proxy"], marker="o", label="bottleneck ratio")
    ax_score.set_title("Diffusion-time score trends", fontsize=14, pad=8)
    ax_score.set_xlabel("diffusion time t")
    ax_score.grid(alpha=0.15)
    ax_score.legend()

    # bottom left: axis alignments
    ax_align = fig.add_subplot(gs[1, 0:2])
    for label in ["axis_align_seam_distance", "axis_align_branch_exit", "axis_align_corridor", "axis_align_shared_hotspot"]:
        ax_align.plot(scores["diffusion_time"], scores[label], marker="o", label=label.replace("axis_align_", ""))
    ax_align.set_title("Axis-alignment by diffusion time", fontsize=14, pad=8)
    ax_align.set_xlabel("diffusion time t")
    ax_align.set_ylabel("max |corr(axis, field)|")
    ax_align.grid(alpha=0.15)
    ax_align.legend()

    # bottom middle: seam & family separation
    ax_sep = fig.add_subplot(gs[1, 2])
    width = 0.36
    xx = np.arange(len(scores))
    ax_sep.bar(xx - width / 2, scores["seam_separation"], width, label="seam")
    ax_sep.bar(xx + width / 2, scores["family_centroid_separation"], width, label="family")
    ax_sep.set_xticks(xx)
    ax_sep.set_xticklabels([str(int(t)) for t in scores["diffusion_time"]])
    ax_sep.set_title("Separation scores", fontsize=14, pad=8)
    ax_sep.set_xlabel("t")
    ax_sep.grid(alpha=0.15, axis="y")
    ax_sep.legend()

    # bottom right: interpretation box
    ax_diag = fig.add_subplot(gs[1, 3])
    ax_diag.axis("off")
    best_seam_row = scores.sort_values("seam_separation", ascending=False).iloc[0]
    best_branch_row = scores.sort_values("axis_align_branch_exit", ascending=False).iloc[0]
    best_corridor_row = scores.sort_values("axis_align_corridor", ascending=False).iloc[0]
    text = (
        "OBS-028b diagnostics\n\n"
        f"best seam sep t: {int(best_seam_row['diffusion_time'])}\n"
        f"best seam score: {best_seam_row['seam_separation']:.3f}\n\n"
        f"best branch axis t: {int(best_branch_row['diffusion_time'])}\n"
        f"branch align: {best_branch_row['axis_align_branch_exit']:.3f}\n\n"
        f"best corridor axis t: {int(best_corridor_row['diffusion_time'])}\n"
        f"corridor align: {best_corridor_row['axis_align_corridor']:.3f}\n\n"
        "Interpretation:\n"
        "diffusion should be judged\n"
        "by slow-mode alignment and\n"
        "bottleneck structure, not\n"
        "only local-neighborhood fit."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-028b diffusion mode analysis", fontsize=20)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def build_summary(scores: pd.DataFrame, sigma: float) -> str:
    lines = [
        "=== OBS-028b Diffusion Mode Analysis Summary ===",
        "",
        f"kernel_sigma = {sigma:.6f}",
        "",
        "Scores by diffusion time",
    ]
    for _, row in scores.iterrows():
        lines.append(
            f"  t={int(row['diffusion_time'])}: "
            f"seam_separation={row['seam_separation']:.4f}, "
            f"family_centroid_separation={row['family_centroid_separation']:.4f}, "
            f"shared_hotspot_compactness={row['shared_hotspot_compactness']:.4f}, "
            f"bottleneck_proxy={row['bottleneck_proxy']:.4f}, "
            f"axis_align_seam_distance={row['axis_align_seam_distance']:.4f}, "
            f"axis_align_branch_exit={row['axis_align_branch_exit']:.4f}, "
            f"axis_align_corridor={row['axis_align_corridor']:.4f}, "
            f"axis_align_shared_hotspot={row['axis_align_shared_hotspot']:.4f}"
        )

    best_seam = scores.sort_values("seam_separation", ascending=False).iloc[0]
    best_branch = scores.sort_values("axis_align_branch_exit", ascending=False).iloc[0]
    best_corridor = scores.sort_values("axis_align_corridor", ascending=False).iloc[0]
    best_bottle = scores.sort_values("bottleneck_proxy", ascending=True).iloc[0]

    lines.extend(
        [
            "",
            "Best-by-diffusion-native criterion",
            f"  seam separation: t={int(best_seam['diffusion_time'])}",
            f"  branch-exit axis alignment: t={int(best_branch['diffusion_time'])}",
            f"  corridor axis alignment: t={int(best_corridor['diffusion_time'])}",
            f"  bottleneck concentration: t={int(best_bottle['diffusion_time'])}",
            "",
            "Interpretive guide",
            "- high seam separation means diffusion coordinates isolate seam regime well",
            "- high branch/corridor alignment means a leading diffusion axis tracks family tendency",
            "- lower bottleneck_proxy means shared hotspots concentrate more tightly than the background",
            "- diffusion may be useful even if it is not the best general-purpose 2D display embedding",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair diffusion-native analysis of observatory structure.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--edges-csv", default=Config.edges_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    parser.add_argument("--kernel-scale", default=Config.kernel_scale, choices=["median", "mean"])
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        routes_csv=args.routes_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        hotspot_quantile=args.hotspot_quantile,
        kernel_scale=args.kernel_scale,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, routes = load_inputs(cfg)
    nodes = build_node_augmented_table(nodes, routes, cfg)
    D = compute_graph_distance(nodes, edges)
    P, sigma = build_diffusion_operator(D, cfg)
    vals, vecs = diffusion_eigendecomposition(P)

    score_rows = []
    for t in cfg.diffusion_times:
        coords = diffusion_coords(vals, vecs, t)
        nodes[f"diff1_t{t}"] = coords[:, 0]
        nodes[f"diff2_t{t}"] = coords[:, 1]

        row = {
            "diffusion_time": t,
            "seam_separation": seam_separation_score(nodes, cfg.seam_threshold, f"diff1_t{t}", f"diff2_t{t}"),
            "family_centroid_separation": family_centroid_score(nodes, f"diff1_t{t}", f"diff2_t{t}"),
            "shared_hotspot_compactness": shared_hotspot_compactness(nodes, f"diff1_t{t}", f"diff2_t{t}"),
            "bottleneck_proxy": bottleneck_proxy_score(nodes, f"diff1_t{t}", f"diff2_t{t}"),
        }
        row.update(axis_alignment_scores(nodes, f"diff1_t{t}", f"diff2_t{t}"))
        row = {
            **row,
            "axis_align_seam_distance": row["seam_distance"],
            "axis_align_branch_exit": row["branch_exit"],
            "axis_align_corridor": row["corridor"],
            "axis_align_shared_hotspot": row["shared_hotspot"],
        }
        del row["seam_distance"]
        del row["branch_exit"]
        del row["corridor"]
        del row["shared_hotspot"]
        score_rows.append(row)

    scores = pd.DataFrame(score_rows)
    nodes.to_csv(outdir / "diffusion_modes_nodes.csv", index=False)
    scores.to_csv(outdir / "diffusion_mode_scores.csv", index=False)

    summary = build_summary(scores, sigma)
    (outdir / "obs028b_diffusion_mode_analysis_summary.txt").write_text(summary, encoding="utf-8")

    render_panel(nodes, scores, vals, outdir / "obs028b_diffusion_mode_analysis.png", cfg)

    print(outdir / "diffusion_modes_nodes.csv")
    print(outdir / "diffusion_mode_scores.csv")
    print(outdir / "obs028b_diffusion_mode_analysis_summary.txt")
    print(outdir / "obs028b_diffusion_mode_analysis.png")


if __name__ == "__main__":
    main()
