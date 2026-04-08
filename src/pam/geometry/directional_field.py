from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def axial_angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """
    Axial angle difference in radians.

    Treats directions theta and theta + pi as equivalent.
    Output lies in [0, pi/2].
    """
    d = np.abs(wrap_angle(np.asarray(a) - np.asarray(b)))
    return np.minimum(d, np.pi - d)


def axial_angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.degrees(axial_angle_diff(a, b))


def unit_vector(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)], dtype=float)


@dataclass(frozen=True)
class DirectionalField:
    """
    Canonical directional field on the PAM manifold.

    Stores:
    - node geometry / coordinates
    - response-direction field
    - connection proxy angle field
    - graph edges
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame

    node_id_col: str = "node_id"
    src_col: str = "src_id"
    dst_col: str = "dst_id"
    x_col: str = "mds1"
    y_col: str = "mds2"
    z_col: str = "signed_phase"
    connection_theta_col: str = "fim_theta"
    response_theta_col: str = "rsp_theta"

    @classmethod
    def from_csv(
        cls,
        nodes_csv: str | Path,
        edges_csv: str | Path,
        *,
        node_id_col: str = "node_id",
        src_col: str = "src_id",
        dst_col: str = "dst_id",
        x_col: str = "mds1",
        y_col: str = "mds2",
        z_col: str = "signed_phase",
        connection_theta_col: str = "fim_theta",
        response_theta_col: str = "rsp_theta",
    ) -> "DirectionalField":
        nodes = pd.read_csv(nodes_csv)
        edges = pd.read_csv(edges_csv)
        return cls.from_frames(
            nodes,
            edges,
            node_id_col=node_id_col,
            src_col=src_col,
            dst_col=dst_col,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            connection_theta_col=connection_theta_col,
            response_theta_col=response_theta_col,
        )

    @classmethod
    def from_frames(
        cls,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        *,
        node_id_col: str = "node_id",
        src_col: str = "src_id",
        dst_col: str = "dst_id",
        x_col: str = "mds1",
        y_col: str = "mds2",
        z_col: str = "signed_phase",
        connection_theta_col: str = "fim_theta",
        response_theta_col: str = "rsp_theta",
    ) -> "DirectionalField":
        nodes = nodes.copy()
        edges = edges.copy()

        required_node_cols = [node_id_col, x_col, y_col, connection_theta_col, response_theta_col]
        required_edge_cols = [src_col, dst_col]

        missing_nodes = [c for c in required_node_cols if c not in nodes.columns]
        missing_edges = [c for c in required_edge_cols if c not in edges.columns]
        if missing_nodes:
            raise ValueError(f"Missing node columns: {missing_nodes}")
        if missing_edges:
            raise ValueError(f"Missing edge columns: {missing_edges}")

        for c in [node_id_col, x_col, y_col, z_col, connection_theta_col, response_theta_col]:
            if c in nodes.columns:
                nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
        for c in [src_col, dst_col]:
            edges[c] = pd.to_numeric(edges[c], errors="coerce")

        nodes = nodes.dropna(subset=[node_id_col, x_col, y_col, connection_theta_col, response_theta_col]).copy()
        nodes[node_id_col] = nodes[node_id_col].astype(int)

        edges = edges.dropna(subset=[src_col, dst_col]).copy()
        edges[src_col] = edges[src_col].astype(int)
        edges[dst_col] = edges[dst_col].astype(int)

        valid_ids = set(nodes[node_id_col].tolist())
        edges = edges[
            edges[src_col].isin(valid_ids) & edges[dst_col].isin(valid_ids)
        ].copy()

        return cls(
            nodes=nodes.reset_index(drop=True),
            edges=edges.reset_index(drop=True),
            node_id_col=node_id_col,
            src_col=src_col,
            dst_col=dst_col,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            connection_theta_col=connection_theta_col,
            response_theta_col=response_theta_col,
        )

    @property
    def node_ids(self) -> np.ndarray:
        return self.nodes[self.node_id_col].to_numpy(dtype=int)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def node_index(self) -> pd.Index:
        return pd.Index(self.node_ids, name=self.node_id_col)

    def node_lookup(self) -> pd.DataFrame:
        return self.nodes.set_index(self.node_id_col, drop=False)

    def adjacency(self, undirected: bool = True) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {int(n): [] for n in self.node_ids}
        for _, row in self.edges.iterrows():
            u = int(row[self.src_col])
            v = int(row[self.dst_col])
            adj[u].append(v)
            if undirected:
                adj[v].append(u)
        return adj

    def node_positions(self) -> pd.DataFrame:
        cols = [self.node_id_col, self.x_col, self.y_col]
        if self.z_col in self.nodes.columns:
            cols.append(self.z_col)
        return self.nodes[cols].copy()

    def node_angles(self) -> pd.DataFrame:
        return self.nodes[
            [self.node_id_col, self.connection_theta_col, self.response_theta_col]
        ].copy()

    def local_direction_mismatch(self, degrees: bool = True) -> pd.DataFrame:
        out = self.nodes[[self.node_id_col]].copy()
        val = axial_angle_diff(
            self.nodes[self.connection_theta_col].to_numpy(dtype=float),
            self.nodes[self.response_theta_col].to_numpy(dtype=float),
        )
        out["local_direction_mismatch"] = np.degrees(val) if degrees else val
        return out

    def edge_angle_changes(self, degrees: bool = True) -> pd.DataFrame:
        lookup = self.node_lookup()

        rows: list[dict] = []
        for _, edge in self.edges.iterrows():
            u = int(edge[self.src_col])
            v = int(edge[self.dst_col])

            theta_conn_u = float(lookup.at[u, self.connection_theta_col])
            theta_conn_v = float(lookup.at[v, self.connection_theta_col])
            theta_rsp_u = float(lookup.at[u, self.response_theta_col])
            theta_rsp_v = float(lookup.at[v, self.response_theta_col])

            d_conn = float(wrap_angle(theta_conn_v - theta_conn_u))
            d_rsp = float(wrap_angle(theta_rsp_v - theta_rsp_u))
            d_mis = float(axial_angle_diff(d_rsp, d_conn))

            rows.append(
                {
                    self.src_col: u,
                    self.dst_col: v,
                    "delta_connection_theta": np.degrees(d_conn) if degrees else d_conn,
                    "delta_response_theta": np.degrees(d_rsp) if degrees else d_rsp,
                    "edge_direction_mismatch": np.degrees(d_mis) if degrees else d_mis,
                }
            )

        return pd.DataFrame(rows)

    def node_neighbor_mismatch(self, degrees: bool = True) -> pd.DataFrame:
        edge_df = self.edge_angle_changes(degrees=degrees)
        val_col = "edge_direction_mismatch"

        out = (
            edge_df.groupby(self.src_col, as_index=False)
            .agg(
                neighbor_direction_mismatch_mean=(val_col, "mean"),
                neighbor_direction_mismatch_max=(val_col, "max"),
                neighbor_direction_mismatch_std=(val_col, "std"),
                neighbor_direction_mismatch_n=(val_col, "size"),
            )
            .rename(columns={self.src_col: self.node_id_col})
        )
        return out

    def attach_node_metrics(self, *dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
        out = self.nodes.copy()
        for df in dfs:
            if self.node_id_col not in df.columns:
                raise ValueError(f"Expected {self.node_id_col} in metric frame")
            out = out.merge(df, on=self.node_id_col, how="left")
        return out
