#!/usr/bin/env python3
"""
OBS-028 — Embedding comparison for the PAM observatory.

Compare multiple low-dimensional embeddings built from the observatory graph:

1. MDS                  (distance-faithful baseline)
2. Isomap               (graph-geodesic manifold embedding)
3. Laplacian Eigenmaps  (local graph structure / boundary-sensitive)
4. Diffusion coordinates (slow connectivity modes)
5. UMAP                 (optional exploratory view, if installed)

Evaluation is observatory-specific, not purely visual.

We report for each embedding:
- seam-band separation score
- family centroid separation score
- hotspot compactness score
- trustworthiness-like nearest-neighbor retention score

Inputs
------
outputs/obs022_scene_bundle/scene_nodes.csv
outputs/obs022_scene_bundle/scene_edges.csv
outputs/obs022_scene_bundle/scene_routes.csv

Outputs
-------
outputs/obs028_embedding_comparison/
  embedding_scores.csv
  embedding_nodes_mds.csv
  embedding_nodes_isomap.csv
  embedding_nodes_laplacian.csv
  embedding_nodes_diffusion.csv
  embedding_nodes_umap.csv          (if UMAP available)
  obs028_embedding_comparison_summary.txt
  obs028_embedding_comparison.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import Isomap, MDS, trustworthiness
from sklearn.neighbors import NearestNeighbors

try:
    from sklearn.manifold import SpectralEmbedding
except Exception:
    SpectralEmbedding = None

try:
    import umap  # type: ignore
except Exception:
    umap = None


@dataclass(frozen=True)
class Config:
    nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    edges_csv: str = "outputs/obs022_scene_bundle/scene_edges.csv"
    routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    outdir: str = "outputs/obs028_embedding_comparison"
    seam_threshold: float = 0.15
    hotspot_quantile: float = 0.85
    n_neighbors: int = 8
    random_state: int = 42
    include_umap: bool = True


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def pairwise_dist(X: np.ndarray) -> np.ndarray:
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    edges = pd.read_csv(cfg.edges_csv)
    routes = pd.read_csv(cfg.routes_csv)

    for df in (nodes, edges, routes):
        for col in df.columns:
            if col not in {"path_id", "path_family"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    if "is_branch_away" not in routes.columns:
        routes["is_branch_away"] = 0
    if "is_representative" not in routes.columns:
        routes["is_representative"] = 0

    return nodes, edges, routes


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
    out = routes.copy()
    fam = out.get("path_family", pd.Series(index=out.index, dtype=object))
    out["route_class"] = np.select(
        [
            pd.to_numeric(out["is_branch_away"], errors="coerce").fillna(0).eq(1),
            pd.to_numeric(out["is_representative"], errors="coerce").fillna(0).eq(1) & fam.eq("stable_seam_corridor"),
            pd.to_numeric(out["is_representative"], errors="coerce").fillna(0).eq(1) & fam.eq("reorganization_heavy"),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )
    return out


def build_feature_table(nodes: pd.DataFrame, routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    routes = classify_routes(routes)

    family_touch = (
        routes[routes["route_class"].isin(CLASS_ORDER)]
        .groupby(["node_id", "route_class"], as_index=False)
        .agg(n_rows=("path_id", "size"), n_paths=("path_id", "nunique"))
    )

    pivot_rows = (
        family_touch.pivot(index="node_id", columns="route_class", values="n_rows")
        .fillna(0.0)
        .reset_index()
    )
    pivot_rows.columns.name = None
    for cls in CLASS_ORDER:
        if cls not in pivot_rows.columns:
            pivot_rows[cls] = 0.0

    out = nodes.merge(pivot_rows, on="node_id", how="left")
    for cls in CLASS_ORDER:
        out[cls] = pd.to_numeric(out[cls], errors="coerce").fillna(0.0)

    # hotspot labels if available, otherwise derive from existing fields
    if "shared_hotspot" not in out.columns:
        if "sym_traceless_norm" in out.columns:
            thr_a = float(pd.to_numeric(out["sym_traceless_norm"], errors="coerce").quantile(cfg.hotspot_quantile))
            out["anisotropy_hotspot"] = (pd.to_numeric(out["sym_traceless_norm"], errors="coerce") >= thr_a).astype(int)
        else:
            out["anisotropy_hotspot"] = 0
        if "neighbor_direction_mismatch_mean" in out.columns:
            thr_r = float(pd.to_numeric(out["neighbor_direction_mismatch_mean"], errors="coerce").quantile(cfg.hotspot_quantile))
            out["relational_hotspot"] = (pd.to_numeric(out["neighbor_direction_mismatch_mean"], errors="coerce") >= thr_r).astype(int)
        else:
            out["relational_hotspot"] = 0
        out["shared_hotspot"] = (
            (pd.to_numeric(out["anisotropy_hotspot"], errors="coerce").fillna(0) == 1)
            & (pd.to_numeric(out["relational_hotspot"], errors="coerce").fillna(0) == 1)
        ).astype(int)

    feature_cols = [
        c for c in [
            "signed_phase",
            "distance_to_seam",
            "lazarus_score",
            "response_strength",
            "sym_traceless_norm",
            "neighbor_direction_mismatch_mean",
            "node_holonomy_proxy",
            "stable_seam_corridor",
            "reorganization_heavy",
            "branch_exit",
        ] if c in out.columns
    ]

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # standardize
    X = out[feature_cols].to_numpy(dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    Xz = (X - mu) / sigma

    out.attrs["feature_cols"] = feature_cols
    out.attrs["Xz"] = Xz
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

        if "edge_cost" in edges.columns and pd.notna(row.get("edge_cost", np.nan)):
            w = float(row["edge_cost"])
        else:
            # fallback to Euclidean in MDS plane if edge_cost absent
            try:
                x1, y1 = float(row["src_mds1"]), float(row["src_mds2"])
                x2, y2 = float(row["dst_mds1"]), float(row["dst_mds2"])
                w = float(np.hypot(x2 - x1, y2 - y1))
            except Exception:
                w = 1.0

        if not np.isfinite(w):
            continue

        D[u, v] = min(D[u, v], w)
        D[v, u] = min(D[v, u], w)

    # Floyd-Warshall is fine for n=75
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])

    # fill any disconnected pairs with max finite distance
    finite = D[np.isfinite(D)]
    max_finite = float(finite.max()) if finite.size else 1.0
    D[~np.isfinite(D)] = max_finite
    return D


def embed_mds(D: np.ndarray, cfg: Config) -> np.ndarray:
    model = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=cfg.random_state,
        normalized_stress="auto",
    )
    return model.fit_transform(D)


def embed_isomap(D: np.ndarray, cfg: Config) -> np.ndarray:
    model = Isomap(n_components=2, metric="precomputed", n_neighbors=cfg.n_neighbors)
    return model.fit_transform(D)


def embed_laplacian(D: np.ndarray, cfg: Config) -> np.ndarray:
    if SpectralEmbedding is None:
        raise RuntimeError("SpectralEmbedding unavailable")
    sigma = np.median(D[D > 0])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    A = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    model = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        random_state=cfg.random_state,
    )
    return model.fit_transform(A)


def embed_diffusion(D: np.ndarray, cfg: Config) -> np.ndarray:
    sigma = np.median(D[D > 0])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    K = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(K, 0.0)

    row_sum = K.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 1e-12] = 1.0
    P = K / row_sum

    vals, vecs = np.linalg.eig(P)
    vals = np.real(vals)
    vecs = np.real(vecs)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # skip trivial first eigenvector
    coords = vecs[:, 1:3] * vals[1:3]
    return coords


def embed_umap_from_features(Xz: np.ndarray, cfg: Config) -> np.ndarray:
    if umap is None:
        raise RuntimeError("UMAP unavailable")
    model = umap.UMAP(
        n_components=2,
        n_neighbors=cfg.n_neighbors,
        random_state=cfg.random_state,
        min_dist=0.15,
    )
    return model.fit_transform(Xz)


def seam_separation_score(df: pd.DataFrame, seam_threshold: float) -> float:
    seam = pd.to_numeric(df["distance_to_seam"], errors="coerce") <= seam_threshold
    A = df.loc[seam, ["emb1", "emb2"]].to_numpy(dtype=float)
    B = df.loc[~seam, ["emb1", "emb2"]].to_numpy(dtype=float)
    if len(A) < 2 or len(B) < 2:
        return float("nan")
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    varA = float(np.mean(np.sum((A - muA) ** 2, axis=1)))
    varB = float(np.mean(np.sum((B - muB) ** 2, axis=1)))
    denom = np.sqrt(max(varA + varB, 1e-12))
    return float(np.linalg.norm(muA - muB) / denom)


def family_centroid_score(df: pd.DataFrame) -> float:
    rows = []
    for cls in CLASS_ORDER:
        weight = pd.to_numeric(df.get(cls, 0), errors="coerce").fillna(0.0)
        if float(weight.sum()) <= 0:
            continue
        X = df[["emb1", "emb2"]].to_numpy(dtype=float)
        mu = (X * weight.to_numpy()[:, None]).sum(axis=0) / float(weight.sum())
        rows.append(mu)
    if len(rows) < 2:
        return float("nan")
    rows = np.vstack(rows)
    d = pairwise_dist(rows)
    iu = np.triu_indices_from(d, k=1)
    return float(np.mean(d[iu]))


def hotspot_compactness_score(df: pd.DataFrame) -> float:
    shared = pd.to_numeric(df.get("shared_hotspot", 0), errors="coerce").fillna(0) == 1
    Xs = df.loc[shared, ["emb1", "emb2"]].to_numpy(dtype=float)
    if len(Xs) < 2:
        return float("nan")
    mu = Xs.mean(axis=0)
    return float(np.mean(np.sqrt(np.sum((Xs - mu) ** 2, axis=1))))


def nn_retention_score(Xref: np.ndarray, Y: np.ndarray, k: int = 8) -> float:
    nn_ref = NearestNeighbors(n_neighbors=min(k + 1, len(Xref))).fit(Xref)
    nn_emb = NearestNeighbors(n_neighbors=min(k + 1, len(Y))).fit(Y)

    idx_ref = nn_ref.kneighbors(return_distance=False)[:, 1:]
    idx_emb = nn_emb.kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    for a, b in zip(idx_ref, idx_emb):
        overlaps.append(len(set(map(int, a)) & set(map(int, b))) / max(len(a), 1))
    return float(np.mean(overlaps))


def score_embedding(df: pd.DataFrame, Xref: np.ndarray, cfg: Config) -> dict[str, float]:
    Y = df[["emb1", "emb2"]].to_numpy(dtype=float)
    return {
        "seam_separation": seam_separation_score(df, cfg.seam_threshold),
        "family_centroid_separation": family_centroid_score(df),
        "shared_hotspot_compactness": hotspot_compactness_score(df),
        "trustworthiness": float(trustworthiness(Xref, Y, n_neighbors=min(cfg.n_neighbors, len(df) - 1))),
        "nn_retention": nn_retention_score(Xref, Y, k=cfg.n_neighbors),
    }


def render_panel(embed_tables: dict[str, pd.DataFrame], scores: pd.DataFrame, outpath: Path) -> None:
    names = list(embed_tables.keys())
    n = len(names)
    fig = plt.figure(figsize=(5.4 * n, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, n)

    for i, name in enumerate(names):
        df = embed_tables[name]

        ax1 = fig.add_subplot(gs[0, i])
        sc = ax1.scatter(
            df["emb1"], df["emb2"],
            c=pd.to_numeric(df["distance_to_seam"], errors="coerce"),
            cmap="magma_r",
            s=70, alpha=0.95, linewidths=0.35, edgecolors="white",
        )
        ax1.set_title(f"{name} — seam distance", fontsize=14, pad=8)
        ax1.set_xlabel("dim 1")
        ax1.set_ylabel("dim 2")
        ax1.grid(alpha=0.12)
        ax1.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.02)

        ax2 = fig.add_subplot(gs[1, i])
        base = df.copy()
        ax2.scatter(base["emb1"], base["emb2"], s=40, c="lightgray", alpha=0.55, linewidths=0)
        aniso_only = base[(pd.to_numeric(base["anisotropy_hotspot"], errors="coerce").fillna(0) == 1)
                          & (pd.to_numeric(base["shared_hotspot"], errors="coerce").fillna(0) == 0)]
        rel_only = base[(pd.to_numeric(base["relational_hotspot"], errors="coerce").fillna(0) == 1)
                        & (pd.to_numeric(base["shared_hotspot"], errors="coerce").fillna(0) == 0)]
        shared = base[pd.to_numeric(base["shared_hotspot"], errors="coerce").fillna(0) == 1]

        if len(aniso_only):
            ax2.scatter(aniso_only["emb1"], aniso_only["emb2"], s=90, c="#2A9D8F", alpha=0.95)
        if len(rel_only):
            ax2.scatter(rel_only["emb1"], rel_only["emb2"], s=90, c="#B23A48", alpha=0.95)
        if len(shared):
            ax2.scatter(shared["emb1"], shared["emb2"], s=120, c="#FFD166", edgecolors="black", linewidths=1.0, alpha=0.98)

        ax2.set_title(f"{name} — hotspot structure", fontsize=14, pad=8)
        ax2.set_xlabel("dim 1")
        ax2.set_ylabel("dim 2")
        ax2.grid(alpha=0.12)
        ax2.set_aspect("equal", adjustable="box")

        row = scores[scores["embedding"] == name].iloc[0]
        txt = (
            f"seam sep: {row['seam_separation']:.2f}\n"
            f"family sep: {row['family_centroid_separation']:.2f}\n"
            f"shared compact: {row['shared_hotspot_compactness']:.2f}\n"
            f"trust: {row['trustworthiness']:.2f}\n"
            f"nn keep: {row['nn_retention']:.2f}"
        )
        ax2.text(
            0.98, 0.02, txt,
            transform=ax2.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.82", alpha=0.9),
        )

    fig.suptitle("PAM Observatory — OBS-028 embedding comparison", fontsize=20)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def build_summary(scores: pd.DataFrame, feature_cols: list[str]) -> str:
    lines = [
        "=== OBS-028 Embedding Comparison Summary ===",
        "",
        "Features used",
        f"  {', '.join(feature_cols)}",
        "",
        "Scores",
    ]
    for _, row in scores.iterrows():
        lines.append(
            f"  {row['embedding']}: "
            f"seam_separation={row['seam_separation']:.4f}, "
            f"family_centroid_separation={row['family_centroid_separation']:.4f}, "
            f"shared_hotspot_compactness={row['shared_hotspot_compactness']:.4f}, "
            f"trustworthiness={row['trustworthiness']:.4f}, "
            f"nn_retention={row['nn_retention']:.4f}"
        )

    best_seam = scores.sort_values("seam_separation", ascending=False).iloc[0]["embedding"]
    best_family = scores.sort_values("family_centroid_separation", ascending=False).iloc[0]["embedding"]
    best_trust = scores.sort_values("trustworthiness", ascending=False).iloc[0]["embedding"]

    lines.extend(
        [
            "",
            "Best-by-metric",
            f"  seam separation: {best_seam}",
            f"  family separation: {best_family}",
            f"  trustworthiness: {best_trust}",
            "",
            "Interpretive guide",
            "- MDS is the distance-faithful baseline",
            "- Isomap emphasizes intrinsic graph-geodesic structure",
            "- Laplacian / diffusion coordinates emphasize connectivity modes",
            "- UMAP is exploratory and should not automatically replace geometry-faithful embeddings",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MDS against alternative observatory embeddings.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--edges-csv", default=Config.edges_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    parser.add_argument("--n-neighbors", type=int, default=Config.n_neighbors)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--no-umap", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        routes_csv=args.routes_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        hotspot_quantile=args.hotspot_quantile,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
        include_umap=not args.no_umap,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, routes = load_inputs(cfg)
    feat_df = build_feature_table(nodes, routes, cfg)
    Xz = feat_df.attrs["Xz"]
    feature_cols = feat_df.attrs["feature_cols"]

    D = compute_graph_distance(feat_df, edges)

    embeddings: dict[str, np.ndarray] = {}
    embeddings["mds"] = embed_mds(D, cfg)
    embeddings["isomap"] = embed_isomap(D, cfg)
    embeddings["laplacian"] = embed_laplacian(D, cfg)
    embeddings["diffusion"] = embed_diffusion(D, cfg)

    if cfg.include_umap and umap is not None:
        embeddings["umap"] = embed_umap_from_features(Xz, cfg)

    embed_tables: dict[str, pd.DataFrame] = {}
    score_rows: list[dict[str, float | str]] = []

    for name, Y in embeddings.items():
        df = feat_df.copy()
        df["emb1"] = Y[:, 0]
        df["emb2"] = Y[:, 1]
        embed_tables[name] = df

        df.to_csv(outdir / f"embedding_nodes_{name}.csv", index=False)

        sc = score_embedding(df, Xz, cfg)
        sc["embedding"] = name
        score_rows.append(sc)

    scores = pd.DataFrame(score_rows)
    score_cols = ["embedding", "seam_separation", "family_centroid_separation", "shared_hotspot_compactness", "trustworthiness", "nn_retention"]
    scores = scores[score_cols].sort_values("embedding").reset_index(drop=True)
    scores.to_csv(outdir / "embedding_scores.csv", index=False)

    summary = build_summary(scores, feature_cols)
    (outdir / "obs028_embedding_comparison_summary.txt").write_text(summary, encoding="utf-8")

    render_panel(embed_tables, scores, outdir / "obs028_embedding_comparison.png")

    print(outdir / "embedding_scores.csv")
    print(outdir / "obs028_embedding_comparison_summary.txt")
    print(outdir / "obs028_embedding_comparison.png")


if __name__ == "__main__":
    main()
