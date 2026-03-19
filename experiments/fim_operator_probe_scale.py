import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    src_candidates = ["src", "u", "src_id"]
    dst_candidates = ["dst", "v", "dst_id"]
    weight_candidates = ["distance", "weight", "edge_cost", "fisher_distance", "dist", "length"]

    src_col = next((c for c in src_candidates if c in edges.columns), None)
    dst_col = next((c for c in dst_candidates if c in edges.columns), None)
    w_col = next((c for c in weight_candidates if c in edges.columns), None)

    if src_col is None or dst_col is None:
        raise ValueError(f"Missing source/destination columns in fisher_edges.csv: {list(edges.columns)}")
    if w_col is None:
        raise ValueError(f"Missing edge weight column in fisher_edges.csv: {list(edges.columns)}")

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(int(row[src_col]), int(row[dst_col]), weight=float(row[w_col]))
    return G


def load_node_table(
    mds_csv: str | Path,
    signed_phase_csv: str | Path,
    curvature_csv: str | Path,
    lazarus_csv: str | Path,
) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    phase = pd.read_csv(signed_phase_csv)
    curv = pd.read_csv(curvature_csv)
    laz = pd.read_csv(lazarus_csv)

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in mds.columns and c in phase.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    keep_phase = [c for c in ["node_id", "r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]
    df = mds.merge(phase[keep_phase], on=join_cols, how="left")

    if "scalar_curvature" in curv.columns:
        df = df.merge(
            curv[[c for c in ["r", "alpha", "scalar_curvature"] if c in curv.columns]],
            on=["r", "alpha"],
            how="left",
        )

    keep_laz = [c for c in ["r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns]
    df = df.merge(laz[keep_laz], on=["r", "alpha"], how="left")

    if "node_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["node_id"] = df.index.astype(int)

    return df


def path_phase_flip_count(path_df: pd.DataFrame) -> int:
    signed = path_df["signed_phase"].dropna().tolist()
    if len(signed) < 2:
        return 0
    flips = 0
    prev_sign = 0
    for val in signed:
        sign = -1 if val < 0 else (1 if val > 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            flips += 1
        prev_sign = sign
    return flips


def annotate_path(path_df: pd.DataFrame, path_length: float) -> Dict:
    signed = path_df["signed_phase"].dropna()
    d2s = path_df["distance_to_seam"].dropna()
    curv = path_df["scalar_curvature"].abs().dropna() if "scalar_curvature" in path_df.columns else pd.Series(dtype=float)
    laz = path_df["lazarus_score"].dropna() if "lazarus_score" in path_df.columns else pd.Series(dtype=float)
    laz_hit = path_df["lazarus_hit"].dropna() if "lazarus_hit" in path_df.columns else pd.Series(dtype=float)

    crosses_seam = int((len(signed) > 0) and (signed.min() < 0) and (signed.max() > 0))
    phase_flip_count = path_phase_flip_count(path_df)
    min_distance_to_seam = float(d2s.min()) if len(d2s) else float("nan")
    max_curvature_along_path = float(curv.max()) if len(curv) else float("nan")
    mean_curvature_along_path = float(curv.mean()) if len(curv) else float("nan")
    phase_start = float(signed.iloc[0]) if len(signed) else float("nan")
    phase_end = float(signed.iloc[-1]) if len(signed) else float("nan")
    phase_span = float(phase_end - phase_start) if len(signed) else float("nan")
    path_lazarus_max = float(laz.max()) if len(laz) else float("nan")
    path_lazarus_mean = float(laz.mean()) if len(laz) else float("nan")
    path_lazarus_hit_any = int((laz_hit > 0).any()) if len(laz_hit) else 0

    return {
        "crosses_seam": crosses_seam,
        "phase_flip_count": phase_flip_count,
        "min_distance_to_seam": min_distance_to_seam,
        "max_curvature_along_path": max_curvature_along_path,
        "mean_curvature_along_path": mean_curvature_along_path,
        "path_length_fisher": float(path_length),
        "phase_start": phase_start,
        "phase_end": phase_end,
        "phase_span": phase_span,
        "path_lazarus_max": path_lazarus_max,
        "path_lazarus_mean": path_lazarus_mean,
        "path_lazarus_hit_any": path_lazarus_hit_any,
    }


def stratified_endpoint_pools(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    work = df.dropna(subset=["signed_phase", "mds1", "mds2"]).copy()

    pos = work[work["signed_phase"] > 0].copy()
    neg = work[work["signed_phase"] < 0].copy()
    seam = work[work["distance_to_seam"].notna()].sort_values("distance_to_seam").head(max(10, len(work) // 5)).copy()

    return {"pos": pos, "neg": neg, "seam": seam}


def sample_probe_pairs(df: pd.DataFrame, n_pairs: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    pools = stratified_endpoint_pools(df)
    pos, neg, seam = pools["pos"], pools["neg"], pools["seam"]

    pairs: List[Dict] = []
    classes = [
        "basin_to_basin",
        "same_phase_positive",
        "same_phase_negative",
        "seam_to_positive",
        "seam_to_negative",
    ]

    for i in range(n_pairs):
        cls = classes[i % len(classes)]

        if cls == "basin_to_basin" and len(pos) and len(neg):
            a = neg.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            b = pos.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

        elif cls == "same_phase_positive" and len(pos) >= 2:
            s = pos.sample(n=2, random_state=int(rng.integers(0, 1_000_000)))
            a, b = s.iloc[0], s.iloc[1]

        elif cls == "same_phase_negative" and len(neg) >= 2:
            s = neg.sample(n=2, random_state=int(rng.integers(0, 1_000_000)))
            a, b = s.iloc[0], s.iloc[1]

        elif cls == "seam_to_positive" and len(seam) and len(pos):
            a = seam.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            b = pos.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

        elif cls == "seam_to_negative" and len(seam) and len(neg):
            a = seam.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            b = neg.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

        else:
            # fallback random pair
            s = df.sample(n=2, random_state=int(rng.integers(0, 1_000_000)))
            a, b = s.iloc[0], s.iloc[1]
            cls = "random_fallback"

        pairs.append(
            {
                "path_id": f"P{i+1:03d}",
                "probe_class": cls,
                "start_node": int(a["node_id"]),
                "end_node": int(b["node_id"]),
            }
        )

    return pairs


def render_sample_plot(node_df: pd.DataFrame, path_rows: pd.DataFrame, outpath: Path, max_paths: int = 12):
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    sc = ax.scatter(
        node_df["mds1"],
        node_df["mds2"],
        c=node_df["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=56,
        alpha=0.55,
    )
    fig.colorbar(sc, ax=ax, label="signed phase")

    shown_ids = path_rows["path_id"].drop_duplicates().head(max_paths).tolist()
    sample = path_rows[path_rows["path_id"].isin(shown_ids)].copy()

    for pid, grp in sample.groupby("path_id"):
        grp = grp.sort_values("step")
        ax.plot(grp["mds1"], grp["mds2"], linewidth=1.8, alpha=0.9)
        ax.scatter(grp.iloc[[0]]["mds1"], grp.iloc[[0]]["mds2"], s=70, marker="o")
        ax.scatter(grp.iloc[[-1]]["mds1"], grp.iloc[[-1]]["mds2"], s=85, marker="X")

    ax.set_title("Scaled GE probe sample on PAM manifold")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Scaled GE probe experiment over the PAM manifold.")
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--outdir", default="outputs/fim_ops")
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.edges_csv)
    G = build_graph(edges)
    node_df = load_node_table(args.mds_csv, args.signed_phase_csv, args.curvature_csv, args.lazarus_csv)
    probes = sample_probe_pairs(node_df, n_pairs=args.n_paths, seed=args.seed)

    probes_df = pd.DataFrame(probes)
    probes_df.to_csv(outdir / "scaled_probe_pairs.csv", index=False)

    node_lookup = node_df.set_index("node_id")
    path_tables = []
    summary_rows = []

    for probe in probes:
        path_id = probe["path_id"]
        start_node = int(probe["start_node"])
        end_node = int(probe["end_node"])

        try:
            path = nx.shortest_path(G, start_node, end_node, weight="weight")
            path_length = nx.shortest_path_length(G, start_node, end_node, weight="weight")
        except nx.NetworkXNoPath:
            summary_rows.append(
                {
                    "path_id": path_id,
                    "probe_class": probe["probe_class"],
                    "start_node": start_node,
                    "end_node": end_node,
                    "no_path": 1,
                }
            )
            continue

        rows = []
        for step, node in enumerate(path):
            row = node_lookup.loc[node]
            rows.append(
                {
                    "path_id": path_id,
                    "probe_class": probe["probe_class"],
                    "step": step,
                    "node_id": int(node),
                    "r": float(row["r"]),
                    "alpha": float(row["alpha"]),
                    "mds1": float(row["mds1"]),
                    "mds2": float(row["mds2"]),
                    "signed_phase": float(row["signed_phase"]) if pd.notna(row["signed_phase"]) else float("nan"),
                    "distance_to_seam": float(row["distance_to_seam"]) if "distance_to_seam" in row.index and pd.notna(row["distance_to_seam"]) else float("nan"),
                    "scalar_curvature": float(row["scalar_curvature"]) if "scalar_curvature" in row.index and pd.notna(row["scalar_curvature"]) else float("nan"),
                    "lazarus_score": float(row["lazarus_score"]) if "lazarus_score" in row.index and pd.notna(row["lazarus_score"]) else float("nan"),
                    "lazarus_hit": int(row["lazarus_hit"]) if "lazarus_hit" in row.index and pd.notna(row["lazarus_hit"]) else 0,
                }
            )

        path_df = pd.DataFrame(rows)
        path_tables.append(path_df)

        metrics = annotate_path(path_df, float(path_length))
        summary_rows.append(
            {
                "path_id": path_id,
                "probe_class": probe["probe_class"],
                "start_node": start_node,
                "end_node": end_node,
                "start_r": float(path_df.iloc[0]["r"]),
                "start_alpha": float(path_df.iloc[0]["alpha"]),
                "end_r": float(path_df.iloc[-1]["r"]),
                "end_alpha": float(path_df.iloc[-1]["alpha"]),
                "num_steps": int(len(path_df)),
                "no_path": 0,
                **metrics,
            }
        )

    all_paths = pd.concat(path_tables, ignore_index=True) if path_tables else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    all_paths.to_csv(outdir / "scaled_probe_paths.csv", index=False)
    summary_df.to_csv(outdir / "scaled_probe_metrics.csv", index=False)

    # predictive summary: median split on path_lazarus_max
    valid = summary_df[summary_df["no_path"] == 0].copy()
    if not valid.empty and "path_lazarus_max" in valid.columns:
        med = valid["path_lazarus_max"].median()
        valid["lazarus_group"] = np.where(valid["path_lazarus_max"] >= med, "high", "low")

        pred_summary = (
            valid.groupby("lazarus_group", as_index=False)
            .agg(
                n_paths=("path_id", "count"),
                seam_cross_rate=("crosses_seam", "mean"),
                mean_phase_flip_count=("phase_flip_count", "mean"),
                mean_min_distance_to_seam=("min_distance_to_seam", "mean"),
                mean_max_curvature=("max_curvature_along_path", "mean"),
                mean_path_length=("path_length_fisher", "mean"),
                mean_lazarus_max=("path_lazarus_max", "mean"),
            )
        )
        pred_summary.to_csv(outdir / "scaled_probe_predictive_summary.csv", index=False)

    render_sample_plot(node_df, all_paths, outdir / "scaled_probe_sample_on_mds.png")

    print(outdir / "scaled_probe_pairs.csv")
    print(outdir / "scaled_probe_paths.csv")
    print(outdir / "scaled_probe_metrics.csv")
    if (outdir / "scaled_probe_predictive_summary.csv").exists():
        print(outdir / "scaled_probe_predictive_summary.csv")
    print(outdir / "scaled_probe_sample_on_mds.png")


if __name__ == "__name__":
    main()
