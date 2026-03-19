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
        raise ValueError(
            f"fisher_edges.csv must contain a source and destination column. Found columns: {list(edges.columns)}"
        )
    if w_col is None:
        raise ValueError(
            f"fisher_edges.csv must contain a distance/weight column. Found columns: {list(edges.columns)}"
        )

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(
            int(row[src_col]),
            int(row[dst_col]),
            weight=float(row[w_col]),
        )
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


def sample_probe_pairs(df: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    work = df.dropna(subset=["signed_phase", "mds1", "mds2"]).copy()

    pos = work[work["signed_phase"] > 0].copy()
    neg = work[work["signed_phase"] < 0].copy()
    seamish = work[work["distance_to_seam"].notna()].sort_values("distance_to_seam").copy()

    rows: List[Dict] = []

    if len(pos) == 0 or len(neg) == 0 or len(seamish) == 0:
        raise ValueError("Need positive, negative, and seam-near nodes to sample scaled probes.")

    families = [
        "basin_to_basin",
        "same_phase_positive",
        "same_phase_negative",
        "seam_to_positive",
        "seam_to_negative",
    ]

    for i in range(n_pairs):
        fam = families[i % len(families)]

        if fam == "basin_to_basin":
            a = neg.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            b = pos.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
        elif fam == "same_phase_positive":
            pair = pos.sample(n=2, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
            a, b = pair.iloc[0], pair.iloc[1]
        elif fam == "same_phase_negative":
            pair = neg.sample(n=2, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
            a, b = pair.iloc[0], pair.iloc[1]
        elif fam == "seam_to_positive":
            a = seamish.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            b = pos.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
        else:  # seam_to_negative
            a = seamish.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            b = neg.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]

        rows.append(
            {
                "probe_id": f"P{i+1:03d}",
                "family": fam,
                "start_node": int(a["node_id"]),
                "end_node": int(b["node_id"]),
                "start_r": float(a["r"]),
                "start_alpha": float(a["alpha"]),
                "end_r": float(b["r"]),
                "end_alpha": float(b["alpha"]),
            }
        )

    return pd.DataFrame(rows)


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
    lazarus_max = float(laz.max()) if len(laz) else float("nan")
    lazarus_mean = float(laz.mean()) if len(laz) else float("nan")
    lazarus_hit_any = int((laz_hit >= 1).any()) if len(laz_hit) else 0
    constraint_strength = float(max_curvature_along_path / max(min_distance_to_seam, 1e-9)) if pd.notna(min_distance_to_seam) and pd.notna(max_curvature_along_path) else float("nan")

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
        "lazarus_max": lazarus_max,
        "lazarus_mean": lazarus_mean,
        "lazarus_hit_any": lazarus_hit_any,
        "constraint_strength": constraint_strength,
    }


def summarize_groups(metrics_df: pd.DataFrame) -> pd.DataFrame:
    med = metrics_df["lazarus_max"].median()
    out = metrics_df.copy()
    out["lazarus_group"] = out["lazarus_max"].apply(lambda x: "high" if x >= med else "low")

    summary = (
        out.groupby("lazarus_group", as_index=False)
        .agg(
            n_paths=("probe_id", "count"),
            seam_cross_rate=("crosses_seam", "mean"),
            mean_phase_flip_count=("phase_flip_count", "mean"),
            mean_min_distance_to_seam=("min_distance_to_seam", "mean"),
            mean_max_curvature=("max_curvature_along_path", "mean"),
            mean_path_length=("path_length_fisher", "mean"),
            mean_constraint_strength=("constraint_strength", "mean"),
        )
    )
    return out, summary


def render_probe_cloud(node_df: pd.DataFrame, seam_df: pd.DataFrame, all_paths: pd.DataFrame, outpath: Path, max_draw: int = 25):
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    sc = ax.scatter(
        node_df["mds1"],
        node_df["mds2"],
        c=node_df["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=46,
        alpha=0.60,
    )
    fig.colorbar(sc, ax=ax, label="signed phase")

    if not seam_df.empty and {"mds1", "mds2"}.issubset(seam_df.columns):
        seam_ord = seam_df.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.2, alpha=0.85)

    # draw only a subset for readability
    keep_ids = list(all_paths["probe_id"].drop_duplicates().head(max_draw))
    draw_df = all_paths[all_paths["probe_id"].isin(keep_ids)].copy()

    for probe_id, grp in draw_df.groupby("probe_id"):
        grp = grp.sort_values("step")
        ax.plot(grp["mds1"], grp["mds2"], linewidth=1.4, alpha=0.8)

    ax.set_title("Scaled GE probes on PAM manifold")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def render_predictive_bar(summary_df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.bar(summary_df["lazarus_group"], summary_df["seam_cross_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(crosses_seam)")
    ax.set_title("Seam crossing rate by Lazarus exposure")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Scale GE probe experiment to many endpoint pairs and test Lazarus predictive structure."
    )
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--outdir", default="outputs/fim_ops_scaled")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-draw", type=int, default=25)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.edges_csv)
    G = build_graph(edges)
    node_df = load_node_table(args.mds_csv, args.signed_phase_csv, args.curvature_csv, args.lazarus_csv)

    seam = pd.read_csv(args.seam_csv) if Path(args.seam_csv).exists() else pd.DataFrame()
    if not seam.empty and not {"mds1", "mds2"}.issubset(seam.columns):
        seam = seam.merge(node_df[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")

    probes_df = sample_probe_pairs(node_df, args.n_pairs, args.seed)
    probes_df.to_csv(outdir / "scaled_probe_pairs.csv", index=False)

    node_lookup = node_df.set_index("node_id")
    path_tables = []
    metric_rows = []

    for _, probe in probes_df.iterrows():
        probe_id = str(probe["probe_id"])
        start_node = int(probe["start_node"])
        end_node = int(probe["end_node"])

        path = nx.shortest_path(G, start_node, end_node, weight="weight")
        path_length = float(nx.shortest_path_length(G, start_node, end_node, weight="weight"))

        rows = []
        for step, node in enumerate(path):
            row = node_lookup.loc[node]
            rows.append(
                {
                    "probe_id": probe_id,
                    "family": str(probe["family"]),
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

        metrics = annotate_path(path_df, path_length)
        metric_rows.append(
            {
                "probe_id": probe_id,
                "family": str(probe["family"]),
                "start_node": start_node,
                "end_node": end_node,
                "start_r": float(probe["start_r"]),
                "start_alpha": float(probe["start_alpha"]),
                "end_r": float(probe["end_r"]),
                "end_alpha": float(probe["end_alpha"]),
                "num_steps": int(len(path_df)),
                **metrics,
            }
        )

    all_paths = pd.concat(path_tables, ignore_index=True) if path_tables else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)

    predictive_df, predictive_summary = summarize_groups(metrics_df)

    all_paths.to_csv(outdir / "scaled_probe_paths.csv", index=False)
    metrics_df.to_csv(outdir / "scaled_probe_metrics.csv", index=False)
    predictive_df.to_csv(outdir / "scaled_probe_predictive_test.csv", index=False)
    predictive_summary.to_csv(outdir / "scaled_probe_predictive_summary.csv", index=False)

    render_probe_cloud(node_df, seam, all_paths, outdir / "scaled_probe_cloud_on_mds.png", max_draw=args.max_draw)
    render_predictive_bar(predictive_summary, outdir / "scaled_probe_predictive_bar.png")

    print(outdir / "scaled_probe_pairs.csv")
    print(outdir / "scaled_probe_paths.csv")
    print(outdir / "scaled_probe_metrics.csv")
    print(outdir / "scaled_probe_predictive_test.csv")
    print(outdir / "scaled_probe_predictive_summary.csv")
    print(outdir / "scaled_probe_cloud_on_mds.png")
    print(outdir / "scaled_probe_predictive_bar.png")


if __name__ == "__main__":
    main()
