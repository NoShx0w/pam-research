import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def load_graph(edges_csv: str | Path) -> nx.Graph:
    edges = pd.read_csv(edges_csv)
    G = nx.Graph()
    src_col = "src" if "src" in edges.columns else "u"
    dst_col = "dst" if "dst" in edges.columns else "v"
    w_col = "distance" if "distance" in edges.columns else "weight"
    for _, row in edges.iterrows():
        G.add_edge(int(row[src_col]), int(row[dst_col]), weight=float(row[w_col]))
    return G


def load_nodes(mds_csv: str | Path, signed_phase_csv: str | Path, curvature_csv: str | Path) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    phase = pd.read_csv(signed_phase_csv)
    curv = pd.read_csv(curvature_csv)

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in mds.columns and c in phase.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    keep_phase = [c for c in ["node_id", "r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]
    df = mds.merge(phase[keep_phase], on=join_cols, how="left")

    if "scalar_curvature" in curv.columns:
        df = df.merge(curv[[c for c in ["r", "alpha", "scalar_curvature"] if c in curv.columns]], on=["r", "alpha"], how="left")

    if "node_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["node_id"] = df.index.astype(int)

    return df


def choose_probe_endpoints(df: pd.DataFrame) -> List[Dict]:
    work = df.dropna(subset=["signed_phase", "mds1", "mds2"]).copy()
    pos = work[work["signed_phase"] > 0].copy()
    neg = work[work["signed_phase"] < 0].copy()
    seamish = work[work["distance_to_seam"].notna()].sort_values("distance_to_seam").copy()

    probes: List[Dict] = []

    if not pos.empty and not neg.empty:
        pos_far = pos.sort_values("signed_phase", ascending=False).iloc[0]
        neg_far = neg.sort_values("signed_phase", ascending=True).iloc[0]
        probes.append({"probe_id": "basin_to_basin_extreme", "start_node": int(neg_far["node_id"]), "end_node": int(pos_far["node_id"])})

    if len(pos) >= 2:
        pos2 = pos.sort_values("signed_phase", ascending=False).head(2)
        probes.append({"probe_id": "same_phase_positive_control", "start_node": int(pos2.iloc[1]["node_id"]), "end_node": int(pos2.iloc[0]["node_id"])})

    if len(neg) >= 2:
        neg2 = neg.sort_values("signed_phase", ascending=True).head(2)
        probes.append({"probe_id": "same_phase_negative_control", "start_node": int(neg2.iloc[1]["node_id"]), "end_node": int(neg2.iloc[0]["node_id"])})

    if not seamish.empty and not pos.empty:
        seam0 = seamish.iloc[0]
        pos_mid = pos.iloc[len(pos) // 2]
        probes.append({"probe_id": "seam_to_positive", "start_node": int(seam0["node_id"]), "end_node": int(pos_mid["node_id"])})

    if not seamish.empty and not neg.empty:
        seam1 = seamish.iloc[min(1, len(seamish) - 1)]
        neg_mid = neg.iloc[len(neg) // 2]
        probes.append({"probe_id": "seam_to_negative", "start_node": int(seam1["node_id"]), "end_node": int(neg_mid["node_id"])})

    return probes


def path_metrics(path_df: pd.DataFrame) -> Dict:
    signed = path_df["signed_phase"].dropna()
    curv = path_df["scalar_curvature"].abs().dropna() if "scalar_curvature" in path_df.columns else pd.Series(dtype=float)
    d2s = path_df["distance_to_seam"].dropna() if "distance_to_seam" in path_df.columns else pd.Series(dtype=float)

    crosses_seam = int((len(signed) > 0) and (signed.min() < 0) and (signed.max() > 0))
    phase_start = float(signed.iloc[0]) if len(signed) else float("nan")
    phase_end = float(signed.iloc[-1]) if len(signed) else float("nan")
    phase_span = float(phase_end - phase_start) if len(signed) else float("nan")
    min_distance_to_seam = float(d2s.min()) if len(d2s) else float("nan")
    max_abs_curvature = float(curv.max()) if len(curv) else float("nan")

    lazarus_hit = 0
    if len(curv) and len(d2s):
        k_thresh = curv.quantile(0.80)
        d_thresh = d2s.quantile(0.20)
        lazarus_hit = int(((path_df["scalar_curvature"].abs() >= k_thresh) & (path_df["distance_to_seam"] <= d_thresh)).any())

    return {
        "crosses_seam": crosses_seam,
        "phase_start": phase_start,
        "phase_end": phase_end,
        "phase_span": phase_span,
        "min_distance_to_seam": min_distance_to_seam,
        "max_abs_curvature": max_abs_curvature,
        "lazarus_hit": lazarus_hit,
    }


def render_probe_plot(df: pd.DataFrame, seam_df: pd.DataFrame, path_rows: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    sc = ax.scatter(df["mds1"], df["mds2"], c=df["signed_phase"], cmap="coolwarm", vmin=-1, vmax=1, s=56, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="signed phase")

    if not seam_df.empty and {"mds1", "mds2"}.issubset(seam_df.columns):
        seam_ord = seam_df.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.0, alpha=0.85)

    for probe_id, grp in path_rows.groupby("probe_id"):
        grp = grp.sort_values("step")
        ax.plot(grp["mds1"], grp["mds2"], linewidth=2.2, alpha=0.95, label=probe_id)
        ax.scatter(grp.iloc[[0]]["mds1"], grp.iloc[[0]]["mds2"], s=90, marker="o")
        ax.scatter(grp.iloc[[-1]]["mds1"], grp.iloc[[-1]]["mds2"], s=110, marker="X")

    ax.set_title("Canonical GE probes on PAM manifold")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Canonical GE probe set over the PAM Fisher graph.")
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--outdir", default="outputs/fim_ops")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    G = load_graph(args.edges_csv)
    df = load_nodes(args.mds_csv, args.signed_phase_csv, args.curvature_csv)
    seam = pd.read_csv(args.seam_csv) if Path(args.seam_csv).exists() else pd.DataFrame()

    if not seam.empty and not {"mds1", "mds2"}.issubset(seam.columns):
        seam = seam.merge(df[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")

    probes = choose_probe_endpoints(df)
    probes_df = pd.DataFrame(probes)
    probes_df.to_csv(outdir / "canonical_probes.csv", index=False)

    path_rows = []
    summaries = []
    node_lookup = df.set_index("node_id")

    for probe in probes:
        probe_id = probe["probe_id"]
        start_node = int(probe["start_node"])
        end_node = int(probe["end_node"])

        path = nx.shortest_path(G, start_node, end_node, weight="weight")
        length = float(nx.shortest_path_length(G, start_node, end_node, weight="weight"))

        rows = []
        for step, node in enumerate(path):
            row = node_lookup.loc[node]
            rows.append({
                "probe_id": probe_id,
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
        path_rows.append(path_df)

        m = path_metrics(path_df)
        summaries.append({
            "probe_id": probe_id,
            "start_node": start_node,
            "end_node": end_node,
            "start_r": float(path_df.iloc[0]["r"]),
            "start_alpha": float(path_df.iloc[0]["alpha"]),
            "end_r": float(path_df.iloc[-1]["r"]),
            "end_alpha": float(path_df.iloc[-1]["alpha"]),
            "path_length": length,
            "num_steps": len(path_df),
            **m,
        })

    all_paths = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)

    all_paths.to_csv(outdir / "canonical_probe_paths.csv", index=False)
    summary_df.to_csv(outdir / "canonical_probe_summary.csv", index=False)

    render_probe_plot(df, seam, all_paths, outdir / "canonical_probes_on_mds.png")

    print(outdir / "canonical_probes.csv")
    print(outdir / "canonical_probe_paths.csv")
    print(outdir / "canonical_probe_summary.csv")
    print(outdir / "canonical_probes_on_mds.png")


if __name__ == "__main__":
    main()
