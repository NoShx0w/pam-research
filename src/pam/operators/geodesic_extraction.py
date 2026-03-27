"""Canonical Operator S / geodesic extraction stage for the PAM operators layer."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
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
    mds_csv,
    signed_phase_csv,
    curvature_csv,
) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    phase = pd.read_csv(signed_phase_csv)
    curv = pd.read_csv(curvature_csv)

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

    if "node_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["node_id"] = df.index.astype(int)

    return df


def load_seam(seam_csv, node_df: pd.DataFrame) -> pd.DataFrame:
    seam_path = Path(seam_csv)
    if not seam_path.exists():
        return pd.DataFrame()

    seam = pd.read_csv(seam_path)
    if not {"mds1", "mds2"}.issubset(seam.columns):
        seam = seam.merge(
            node_df[["r", "alpha", "mds1", "mds2"]],
            on=["r", "alpha"],
            how="left",
        )
    return seam.dropna(subset=["mds1", "mds2"]).copy()


def choose_default_probes(df: pd.DataFrame) -> List[Dict]:
    work = df.dropna(subset=["signed_phase", "mds1", "mds2"]).copy()
    pos = work[work["signed_phase"] > 0].copy()
    neg = work[work["signed_phase"] < 0].copy()
    seamish = work[work["distance_to_seam"].notna()].sort_values("distance_to_seam").copy()

    probes: List[Dict] = []

    if not pos.empty and not neg.empty:
        pos_far = pos.sort_values("signed_phase", ascending=False).iloc[0]
        neg_far = neg.sort_values("signed_phase", ascending=True).iloc[0]
        probes.append({
            "path_id": "S01",
            "start_node": int(neg_far["node_id"]),
            "end_node": int(pos_far["node_id"]),
            "label": "basin_to_basin_extreme",
        })

    if len(pos) >= 2:
        pos2 = pos.sort_values("signed_phase", ascending=False).head(2)
        probes.append({
            "path_id": "S02",
            "start_node": int(pos2.iloc[1]["node_id"]),
            "end_node": int(pos2.iloc[0]["node_id"]),
            "label": "same_phase_positive_control",
        })

    if len(neg) >= 2:
        neg2 = neg.sort_values("signed_phase", ascending=True).head(2)
        probes.append({
            "path_id": "S03",
            "start_node": int(neg2.iloc[1]["node_id"]),
            "end_node": int(neg2.iloc[0]["node_id"]),
            "label": "same_phase_negative_control",
        })

    if not seamish.empty and not pos.empty:
        seam0 = seamish.iloc[0]
        pos_mid = pos.iloc[len(pos) // 2]
        probes.append({
            "path_id": "S04",
            "start_node": int(seam0["node_id"]),
            "end_node": int(pos_mid["node_id"]),
            "label": "seam_to_positive",
        })

    if not seamish.empty and not neg.empty:
        seam1 = seamish.iloc[min(1, len(seamish) - 1)]
        neg_mid = neg.iloc[len(neg) // 2]
        probes.append({
            "path_id": "S05",
            "start_node": int(seam1["node_id"]),
            "end_node": int(neg_mid["node_id"]),
            "label": "seam_to_negative",
        })

    return probes


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

    crosses_seam = int((len(signed) > 0) and (signed.min() < 0) and (signed.max() > 0))
    phase_flip_count = path_phase_flip_count(path_df)
    min_distance_to_seam = float(d2s.min()) if len(d2s) else float("nan")
    max_curvature_along_path = float(curv.max()) if len(curv) else float("nan")
    mean_curvature_along_path = float(curv.mean()) if len(curv) else float("nan")
    phase_start = float(signed.iloc[0]) if len(signed) else float("nan")
    phase_end = float(signed.iloc[-1]) if len(signed) else float("nan")
    phase_span = float(phase_end - phase_start) if len(signed) else float("nan")

    lazarus_hit = 0
    if len(curv) and len(d2s):
        k_thresh = curv.quantile(0.80)
        d_thresh = d2s.quantile(0.20)
        lazarus_hit = int(
            ((path_df["scalar_curvature"].abs() >= k_thresh) &
             (path_df["distance_to_seam"] <= d_thresh)).any()
        )

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
        "lazarus_hit": lazarus_hit,
    }


def render_plot(node_df: pd.DataFrame, seam_df: pd.DataFrame, all_paths: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    sc = ax.scatter(
        node_df["mds1"],
        node_df["mds2"],
        c=node_df["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=70,
        alpha=0.70,
    )
    fig.colorbar(sc, ax=ax, label="signed phase")

    if not seam_df.empty:
        seam_ord = seam_df.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.4, alpha=0.9)

    for path_id, grp in all_paths.groupby("path_id"):
        grp = grp.sort_values("step")
        ax.plot(grp["mds1"], grp["mds2"], linewidth=2.4, alpha=0.95, label=path_id)
        ax.scatter(grp.iloc[[0]]["mds1"], grp.iloc[[0]]["mds2"], s=115, marker="o")
        ax.scatter(grp.iloc[[-1]]["mds1"], grp.iloc[[-1]]["mds2"], s=135, marker="X")

    ax.set_title("Operator S on the PAM manifold")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def run_geodesic_extraction(
    edges_csv,
    mds_csv,
    signed_phase_csv,
    curvature_csv,
    seam_csv,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(edges_csv)
    G = build_graph(edges)
    node_df = load_node_table(mds_csv, signed_phase_csv, curvature_csv)
    seam_df = load_seam(seam_csv, node_df)
    probes = choose_default_probes(node_df)

    probes_df = pd.DataFrame(probes)
    probes_df.to_csv(outdir / "operator_S_probes.csv", index=False)

    node_lookup = node_df.set_index("node_id")
    path_tables = []
    summary_rows = []

    for probe in probes:
        path_id = probe["path_id"]
        start_node = int(probe["start_node"])
        end_node = int(probe["end_node"])

        path = nx.shortest_path(G, start_node, end_node, weight="weight")
        path_length = nx.shortest_path_length(G, start_node, end_node, weight="weight")

        rows = []
        for step, node in enumerate(path):
            row = node_lookup.loc[node]
            rows.append({
                "path_id": path_id,
                "label": probe["label"],
                "step": step,
                "node_id": int(node),
                "r": float(row["r"]),
                "alpha": float(row["alpha"]),
                "mds1": float(row["mds1"]),
                "mds2": float(row["mds2"]),
                "signed_phase": float(row["signed_phase"]) if pd.notna(row["signed_phase"]) else float("nan"),
                "distance_to_seam": float(row["distance_to_seam"]) if "distance_to_seam" in row.index and pd.notna(row["distance_to_seam"]) else float("nan"),
                "scalar_curvature": float(row["scalar_curvature"]) if "scalar_curvature" in row.index and pd.notna(row["scalar_curvature"]) else float("nan"),
            })

        path_df = pd.DataFrame(rows)
        path_tables.append(path_df)

        metrics = annotate_path(path_df, float(path_length))
        summary_rows.append({
            "path_id": path_id,
            "label": probe["label"],
            "start_node": start_node,
            "end_node": end_node,
            "start_r": float(path_df.iloc[0]["r"]),
            "start_alpha": float(path_df.iloc[0]["alpha"]),
            "end_r": float(path_df.iloc[-1]["r"]),
            "end_alpha": float(path_df.iloc[-1]["alpha"]),
            "num_steps": int(len(path_df)),
            **metrics,
        })

    all_paths = pd.concat(path_tables, ignore_index=True) if path_tables else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    all_paths.to_csv(outdir / "operator_S_paths.csv", index=False)
    summary_df.to_csv(outdir / "operator_S_metrics.csv", index=False)

    render_plot(node_df, seam_df, all_paths, outdir / "operator_S_on_mds.png")

    print(outdir / "operator_S_probes.csv")
    print(outdir / "operator_S_paths.csv")
    print(outdir / "operator_S_metrics.csv")
    print(outdir / "operator_S_on_mds.png")

    return {
        "probes": probes_df,
        "paths": all_paths,
        "metrics": summary_df,
    }
