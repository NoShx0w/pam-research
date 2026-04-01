from __future__ import annotations

"""
PAM Identity Proxy

Build first-pass local structural IdentityGraph objects from real PAM manifold
artifacts using:
- node graph structure
- seam distance
- criticality
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pam.topology.identity import Edge, IdentityGraph, Node


@dataclass(frozen=True)
class IdentityProxyConfig:
    seam_eps: float = 0.15
    criticality_quantile: float = 0.9


def load_identity_proxy_inputs(
    *,
    nodes_csv: str | Path,
    edges_csv: str | Path,
    criticality_csv: str | Path,
    phase_distance_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    criticality = pd.read_csv(criticality_csv)
    phase_dist = pd.read_csv(phase_distance_csv)

    required_nodes = {"node_id", "i", "j", "r", "alpha"}
    required_edges = {"src_id", "dst_id"}
    required_crit = {"r", "alpha", "criticality"}
    required_phase = {"node_id", "distance_to_seam"}

    for name, df, required in [
        ("nodes", nodes, required_nodes),
        ("edges", edges, required_edges),
        ("criticality", criticality, required_crit),
        ("phase_distance", phase_dist, required_phase),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing required columns: {sorted(missing)}")

    node_df = (
        nodes.merge(
            phase_dist[["node_id", "distance_to_seam"]],
            on="node_id",
            how="left",
        )
        .merge(
            criticality[["r", "alpha", "criticality"]],
            on=["r", "alpha"],
            how="left",
        )
        .copy()
    )

    return node_df, edges


def classify_patch_node(
    row: pd.Series,
    *,
    seam_eps: float,
    criticality_threshold: float,
) -> str:
    d = pd.to_numeric(row.get("distance_to_seam"), errors="coerce")
    c = pd.to_numeric(row.get("criticality"), errors="coerce")

    if pd.notna(d) and float(d) <= seam_eps:
        return "seam"
    if pd.notna(c) and float(c) >= criticality_threshold:
        return "critical"
    return "stable"


def build_local_identity_graphs(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    *,
    config: IdentityProxyConfig | None = None,
) -> dict[str, IdentityGraph]:
    if config is None:
        config = IdentityProxyConfig()

    work = node_df.copy()
    work["node_id"] = work["node_id"].astype(str)
    work["criticality"] = pd.to_numeric(work["criticality"], errors="coerce")
    work["distance_to_seam"] = pd.to_numeric(work["distance_to_seam"], errors="coerce")

    edge_df = edge_df.copy()
    edge_df["src_id"] = edge_df["src_id"].astype(str)
    edge_df["dst_id"] = edge_df["dst_id"].astype(str)

    criticality_threshold = float(work["criticality"].quantile(config.criticality_quantile))
    node_lookup = work.set_index("node_id", drop=False).to_dict(orient="index")

    neighbors: dict[str, set[str]] = {}
    for _, row in edge_df.iterrows():
        a = str(row["src_id"])
        b = str(row["dst_id"])
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    edge_pairs = {
        tuple(sorted((str(row["src_id"]), str(row["dst_id"]))))
        for _, row in edge_df.iterrows()
    }

    out: dict[str, IdentityGraph] = {}

    for center_id in work["node_id"].astype(str):
        patch_ids = {center_id} | neighbors.get(center_id, set())

        nodes: dict[str, Node] = {}
        for node_id in patch_ids:
            row = node_lookup[node_id]
            kind = classify_patch_node(
                pd.Series(row),
                seam_eps=config.seam_eps,
                criticality_threshold=criticality_threshold,
            )
            nodes[node_id] = Node(
                id=node_id,
                kind=kind,
                attributes={
                    "criticality": float(row["criticality"]) if pd.notna(row["criticality"]) else 0.0,
                    "distance_to_seam": float(row["distance_to_seam"]) if pd.notna(row["distance_to_seam"]) else float("nan"),
                    "is_center": 1.0 if node_id == center_id else 0.0,
                },
            )

        edges: list[Edge] = []
        patch_list = sorted(patch_ids)
        for i, a in enumerate(patch_list):
            for b in patch_list[i + 1:]:
                if tuple(sorted((a, b))) in edge_pairs:
                    edges.append(Edge(a, b, kind="adjacent"))

        out[center_id] = IdentityGraph(nodes=nodes, edges=tuple(edges))

    return out


def identity_grid_from_node_graphs(
    node_df: pd.DataFrame,
    identity_graphs: dict[str, IdentityGraph],
) -> list[list[IdentityGraph]]:
    work = node_df.copy()
    work["node_id"] = work["node_id"].astype(str)
    work["i"] = pd.to_numeric(work["i"], errors="raise").astype(int)
    work["j"] = pd.to_numeric(work["j"], errors="raise").astype(int)

    n_i = int(work["i"].max()) + 1
    n_j = int(work["j"].max()) + 1

    grid: list[list[IdentityGraph | None]] = [
        [None for _ in range(n_j)]
        for _ in range(n_i)
    ]

    for _, row in work.iterrows():
        node_id = str(row["node_id"])
        grid[int(row["i"])][int(row["j"])] = identity_graphs[node_id]

    if any(cell is None for row in grid for cell in row):
        raise ValueError("Identity grid is incomplete; some nodes were not assigned.")

    return [[cell for cell in row] for row in grid]
