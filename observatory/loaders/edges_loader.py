from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class EdgesData:
    edges_df: pd.DataFrame
    mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_edges_data(
    outputs_root: str | Path = "outputs",
    observatory_root: str | Path = "observatory",
) -> EdgesData:
    outputs_root = Path(outputs_root)
    observatory_root = Path(observatory_root)

    candidates = [
        observatory_root / "derived" / "geometry" / "graph" / "edges.csv",
        observatory_root / "derived" / "geometry" / "graph" / "fisher_edges.csv",
        outputs_root / "fim_distance" / "fisher_edges.csv",
    ]

    edges_csv = next((p for p in candidates if p.exists()), None)
    if edges_csv is None:
        return EdgesData(edges_df=pd.DataFrame(), mtime=None)

    df = pd.read_csv(edges_csv).copy()

    # normalize endpoint columns
    if {"src_id", "dst_id"}.issubset(df.columns):
        df = df.rename(columns={"src_id": "source_node_id", "dst_id": "target_node_id"})
    elif {"src_node_id", "dst_node_id"}.issubset(df.columns):
        df = df.rename(columns={"src_node_id": "source_node_id", "dst_node_id": "target_node_id"})
    elif {"src", "dst"}.issubset(df.columns):
        df = df.rename(columns={"src": "source_node_id", "dst": "target_node_id"})
    elif {"source", "target"}.issubset(df.columns):
        df = df.rename(columns={"source": "source_node_id", "target": "target_node_id"})
    elif {"node_u", "node_v"}.issubset(df.columns):
        df = df.rename(columns={"node_u": "source_node_id", "node_v": "target_node_id"})

    # normalize weight column
    if "edge_weight" not in df.columns:
        for candidate in ["edge_cost", "weight", "distance", "fisher_distance", "length"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "edge_weight"})
                break

    if "edge_weight" not in df.columns:
        df["edge_weight"] = 1.0

    required = {"source_node_id", "target_node_id"}
    if not required.issubset(df.columns):
        return EdgesData(edges_df=pd.DataFrame(), mtime=_safe_mtime(edges_csv))

    df["source_node_id"] = pd.to_numeric(df["source_node_id"], errors="coerce")
    df["target_node_id"] = pd.to_numeric(df["target_node_id"], errors="coerce")
    df["edge_weight"] = pd.to_numeric(df["edge_weight"], errors="coerce").fillna(1.0)

    df = df.dropna(subset=["source_node_id", "target_node_id"]).copy()
    df["source_node_id"] = df["source_node_id"].astype(int)
    df["target_node_id"] = df["target_node_id"].astype(int)

    return EdgesData(
        edges_df=df[["source_node_id", "target_node_id", "edge_weight"]].reset_index(drop=True),
        mtime=_safe_mtime(edges_csv),
    )