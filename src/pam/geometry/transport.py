from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pam.geometry.directional_field import (
    DirectionalField,
    axial_angle_diff,
    unit_vector,
    wrap_angle,
)


def rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def transport_angle(theta: float, delta_connection: float) -> float:
    return float(wrap_angle(theta + delta_connection))


def transport_vector(vec: np.ndarray, delta_connection: float) -> np.ndarray:
    return rotation_matrix(delta_connection) @ np.asarray(vec, dtype=float)


@dataclass(frozen=True)
class EdgeTransportResult:
    src_id: int
    dst_id: int
    transported_response_theta_deg: float
    dst_response_theta_deg: float
    misalignment_deg: float
    delta_connection_theta_deg: float
    delta_response_theta_deg: float


def transport_response_across_edge(
    field: DirectionalField,
    src_id: int,
    dst_id: int,
) -> EdgeTransportResult:
    lookup = field.node_lookup()

    theta_conn_src = float(lookup.at[src_id, field.connection_theta_col])
    theta_conn_dst = float(lookup.at[dst_id, field.connection_theta_col])
    theta_rsp_src = float(lookup.at[src_id, field.response_theta_col])
    theta_rsp_dst = float(lookup.at[dst_id, field.response_theta_col])

    delta_conn = float(wrap_angle(theta_conn_dst - theta_conn_src))
    delta_rsp = float(wrap_angle(theta_rsp_dst - theta_rsp_src))

    theta_transported = transport_angle(theta_rsp_src, delta_conn)
    misalignment = float(axial_angle_diff(theta_transported, theta_rsp_dst))

    return EdgeTransportResult(
        src_id=src_id,
        dst_id=dst_id,
        transported_response_theta_deg=float(np.degrees(theta_transported)),
        dst_response_theta_deg=float(np.degrees(theta_rsp_dst)),
        misalignment_deg=float(np.degrees(misalignment)),
        delta_connection_theta_deg=float(np.degrees(delta_conn)),
        delta_response_theta_deg=float(np.degrees(delta_rsp)),
    )


def edge_transport_table(field: DirectionalField) -> pd.DataFrame:
    rows: list[dict] = []
    for _, edge in field.edges.iterrows():
        src_id = int(edge[field.src_col])
        dst_id = int(edge[field.dst_col])
        res = transport_response_across_edge(field, src_id, dst_id)
        rows.append(res.__dict__)
    return pd.DataFrame(rows)


def node_transport_summary(field: DirectionalField) -> pd.DataFrame:
    edge_df = edge_transport_table(field)
    out = (
        edge_df.groupby("src_id", as_index=False)
        .agg(
            transport_align_mean_deg=("misalignment_deg", "mean"),
            transport_align_max_deg=("misalignment_deg", "max"),
            transport_align_std_deg=("misalignment_deg", "std"),
            transport_align_n_neighbors=("misalignment_deg", "size"),
        )
        .rename(columns={"src_id": field.node_id_col})
    )
    return out


def transport_along_path(field: DirectionalField, path_node_ids: list[int]) -> dict[str, float]:
    if len(path_node_ids) < 2:
        return {
            "path_transport_total_misalignment_deg": 0.0,
            "path_transport_mean_misalignment_deg": 0.0,
            "path_transport_n_edges": 0,
        }

    lookup = field.node_lookup()
    transported_theta = float(lookup.at[int(path_node_ids[0]), field.response_theta_col])

    misalignments: list[float] = []
    for a, b in zip(path_node_ids[:-1], path_node_ids[1:]):
        a = int(a)
        b = int(b)

        theta_conn_a = float(lookup.at[a, field.connection_theta_col])
        theta_conn_b = float(lookup.at[b, field.connection_theta_col])
        theta_rsp_b = float(lookup.at[b, field.response_theta_col])

        delta_conn = float(wrap_angle(theta_conn_b - theta_conn_a))
        transported_theta = transport_angle(transported_theta, delta_conn)

        mis = float(axial_angle_diff(transported_theta, theta_rsp_b))
        misalignments.append(np.degrees(mis))

        # reset to realized field after measuring local mismatch
        transported_theta = theta_rsp_b

    arr = np.asarray(misalignments, dtype=float)
    return {
        "path_transport_total_misalignment_deg": float(arr.sum()),
        "path_transport_mean_misalignment_deg": float(arr.mean()) if len(arr) else 0.0,
        "path_transport_n_edges": int(len(arr)),
    }


def loop_transport_residual(field: DirectionalField, loop_node_ids: list[int]) -> dict[str, float]:
    """
    Transport a response direction around a loop and compare with the start.

    The loop should include the start node again at the end, e.g. [A, B, C, D, A].
    """
    if len(loop_node_ids) < 4:
        return {
            "loop_transport_residual_deg": np.nan,
            "loop_total_connection_turn_deg": np.nan,
            "loop_n_edges": 0,
        }

    lookup = field.node_lookup()
    start_id = int(loop_node_ids[0])
    theta_start = float(lookup.at[start_id, field.response_theta_col])
    theta = theta_start
    total_conn = 0.0

    for a, b in zip(loop_node_ids[:-1], loop_node_ids[1:]):
        a = int(a)
        b = int(b)
        theta_conn_a = float(lookup.at[a, field.connection_theta_col])
        theta_conn_b = float(lookup.at[b, field.connection_theta_col])
        delta_conn = float(wrap_angle(theta_conn_b - theta_conn_a))
        theta = transport_angle(theta, delta_conn)
        total_conn += delta_conn

    residual = float(axial_angle_diff(theta, theta_start))
    return {
        "loop_transport_residual_deg": float(np.degrees(residual)),
        "loop_total_connection_turn_deg": float(np.degrees(total_conn)),
        "loop_n_edges": int(len(loop_node_ids) - 1),
    }
