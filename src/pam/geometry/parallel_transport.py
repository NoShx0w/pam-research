from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pam.geometry.directional_field import DirectionalField, axial_angle_diff, unit_vector, wrap_angle


def rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def transport_angle(theta: float, delta_connection: float) -> float:
    """
    Transport an axial direction angle by a connection increment.
    """
    return float(wrap_angle(theta + delta_connection))


def transport_vector(vec: np.ndarray, delta_connection: float) -> np.ndarray:
    """
    Rotate a 2D vector by the connection increment.
    """
    return rotation_matrix(delta_connection) @ np.asarray(vec, dtype=float)


@dataclass(frozen=True)
class EdgeParallelTransportResult:
    src_id: int
    dst_id: int
    src_connection_theta_deg: float
    dst_connection_theta_deg: float
    src_response_theta_deg: float
    dst_response_theta_deg: float
    delta_connection_theta_deg: float
    delta_response_theta_deg: float
    transported_response_theta_deg: float
    response_misalignment_deg: float


@dataclass(frozen=True)
class PathParallelTransportResult:
    path_node_ids: tuple[int, ...]
    path_transport_total_misalignment_deg: float
    path_transport_mean_misalignment_deg: float
    path_transport_max_misalignment_deg: float
    path_transport_n_edges: int
    final_transported_theta_deg: float
    final_realized_theta_deg: float
    endpoint_residual_deg: float


@dataclass(frozen=True)
class LoopParallelTransportResult:
    loop_node_ids: tuple[int, ...]
    loop_n_edges: int
    loop_total_connection_turn_deg: float
    loop_transport_residual_deg: float


def _lookup_thetas(field: DirectionalField, node_id: int) -> tuple[float, float]:
    lookup = field.node_lookup()
    theta_conn = float(lookup.at[int(node_id), field.connection_theta_col])
    theta_rsp = float(lookup.at[int(node_id), field.response_theta_col])
    return theta_conn, theta_rsp


def _delta_connection(field: DirectionalField, src_id: int, dst_id: int) -> float:
    theta_conn_src, _ = _lookup_thetas(field, src_id)
    theta_conn_dst, _ = _lookup_thetas(field, dst_id)
    return float(wrap_angle(theta_conn_dst - theta_conn_src))


def _delta_response(field: DirectionalField, src_id: int, dst_id: int) -> float:
    _, theta_rsp_src = _lookup_thetas(field, src_id)
    _, theta_rsp_dst = _lookup_thetas(field, dst_id)
    return float(wrap_angle(theta_rsp_dst - theta_rsp_src))


def parallel_transport_across_edge(
    field: DirectionalField,
    src_id: int,
    dst_id: int,
) -> EdgeParallelTransportResult:
    src_id = int(src_id)
    dst_id = int(dst_id)

    theta_conn_src, theta_rsp_src = _lookup_thetas(field, src_id)
    theta_conn_dst, theta_rsp_dst = _lookup_thetas(field, dst_id)

    delta_conn = float(wrap_angle(theta_conn_dst - theta_conn_src))
    delta_rsp = float(wrap_angle(theta_rsp_dst - theta_rsp_src))
    theta_transported = transport_angle(theta_rsp_src, delta_conn)
    misalignment = float(axial_angle_diff(theta_transported, theta_rsp_dst))

    return EdgeParallelTransportResult(
        src_id=src_id,
        dst_id=dst_id,
        src_connection_theta_deg=float(np.degrees(theta_conn_src)),
        dst_connection_theta_deg=float(np.degrees(theta_conn_dst)),
        src_response_theta_deg=float(np.degrees(theta_rsp_src)),
        dst_response_theta_deg=float(np.degrees(theta_rsp_dst)),
        delta_connection_theta_deg=float(np.degrees(delta_conn)),
        delta_response_theta_deg=float(np.degrees(delta_rsp)),
        transported_response_theta_deg=float(np.degrees(theta_transported)),
        response_misalignment_deg=float(np.degrees(misalignment)),
    )


def edge_parallel_transport_table(field: DirectionalField) -> pd.DataFrame:
    rows: list[dict] = []
    for _, edge in field.edges.iterrows():
        src_id = int(edge[field.src_col])
        dst_id = int(edge[field.dst_col])
        rows.append(parallel_transport_across_edge(field, src_id, dst_id).__dict__)
    return pd.DataFrame(rows)


def node_parallel_transport_summary(field: DirectionalField) -> pd.DataFrame:
    edge_df = edge_parallel_transport_table(field)
    return (
        edge_df.groupby("src_id", as_index=False)
        .agg(
            transport_align_mean_deg=("response_misalignment_deg", "mean"),
            transport_align_max_deg=("response_misalignment_deg", "max"),
            transport_align_std_deg=("response_misalignment_deg", "std"),
            transport_align_n_neighbors=("response_misalignment_deg", "size"),
        )
        .rename(columns={"src_id": field.node_id_col})
    )


def parallel_transport_along_path(
    field: DirectionalField,
    path_node_ids: Sequence[int],
) -> PathParallelTransportResult:
    path = tuple(int(x) for x in path_node_ids)
    if len(path) < 2:
        _, theta0 = _lookup_thetas(field, path[0]) if len(path) == 1 else (np.nan, np.nan)
        return PathParallelTransportResult(
            path_node_ids=path,
            path_transport_total_misalignment_deg=0.0,
            path_transport_mean_misalignment_deg=0.0,
            path_transport_max_misalignment_deg=0.0,
            path_transport_n_edges=0,
            final_transported_theta_deg=float(np.degrees(theta0)) if np.isfinite(theta0) else np.nan,
            final_realized_theta_deg=float(np.degrees(theta0)) if np.isfinite(theta0) else np.nan,
            endpoint_residual_deg=0.0,
        )

    _, theta_start = _lookup_thetas(field, path[0])
    transported_theta = float(theta_start)
    misalignments_deg: list[float] = []

    for a, b in zip(path[:-1], path[1:]):
        delta_conn = _delta_connection(field, a, b)
        _, theta_rsp_b = _lookup_thetas(field, b)

        transported_theta = transport_angle(transported_theta, delta_conn)
        mis = float(axial_angle_diff(transported_theta, theta_rsp_b))
        mis_deg = float(np.degrees(mis))
        misalignments_deg.append(mis_deg)

        # Reset to realized field after local comparison.
        transported_theta = float(theta_rsp_b)

    _, theta_end = _lookup_thetas(field, path[-1])
    endpoint_residual = float(axial_angle_diff(transported_theta, theta_end))

    arr = np.asarray(misalignments_deg, dtype=float)
    return PathParallelTransportResult(
        path_node_ids=path,
        path_transport_total_misalignment_deg=float(arr.sum()) if len(arr) else 0.0,
        path_transport_mean_misalignment_deg=float(arr.mean()) if len(arr) else 0.0,
        path_transport_max_misalignment_deg=float(arr.max()) if len(arr) else 0.0,
        path_transport_n_edges=int(len(arr)),
        final_transported_theta_deg=float(np.degrees(transported_theta)),
        final_realized_theta_deg=float(np.degrees(theta_end)),
        endpoint_residual_deg=float(np.degrees(endpoint_residual)),
    )


def path_parallel_transport_table(
    field: DirectionalField,
    paths: Iterable[Sequence[int]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        rows.append(parallel_transport_along_path(field, path).__dict__)
    return pd.DataFrame(rows)


def parallel_transport_around_loop(
    field: DirectionalField,
    loop_node_ids: Sequence[int],
) -> LoopParallelTransportResult:
    loop = tuple(int(x) for x in loop_node_ids)
    if len(loop) < 4:
        return LoopParallelTransportResult(
            loop_node_ids=loop,
            loop_n_edges=0,
            loop_total_connection_turn_deg=np.nan,
            loop_transport_residual_deg=np.nan,
        )

    start_id = loop[0]
    _, theta_start = _lookup_thetas(field, start_id)
    theta = float(theta_start)
    total_conn = 0.0

    for a, b in zip(loop[:-1], loop[1:]):
        delta_conn = _delta_connection(field, a, b)
        theta = transport_angle(theta, delta_conn)
        total_conn += delta_conn

    residual = float(axial_angle_diff(theta, theta_start))
    return LoopParallelTransportResult(
        loop_node_ids=loop,
        loop_n_edges=int(len(loop) - 1),
        loop_total_connection_turn_deg=float(np.degrees(total_conn)),
        loop_transport_residual_deg=float(np.degrees(residual)),
    )


def loop_parallel_transport_table(
    field: DirectionalField,
    loops: Iterable[Sequence[int]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for loop in loops:
        rows.append(parallel_transport_around_loop(field, loop).__dict__)
    return pd.DataFrame(rows)


def transported_response_vectors(field: DirectionalField) -> pd.DataFrame:
    """
    Convenience edge table with transported source response vectors and
    destination response vectors, useful for plotting / diagnostics.
    """
    rows: list[dict] = []
    for _, edge in field.edges.iterrows():
        src_id = int(edge[field.src_col])
        dst_id = int(edge[field.dst_col])

        _, theta_rsp_src = _lookup_thetas(field, src_id)
        _, theta_rsp_dst = _lookup_thetas(field, dst_id)
        delta_conn = _delta_connection(field, src_id, dst_id)

        vec_src = unit_vector(theta_rsp_src)
        vec_dst = unit_vector(theta_rsp_dst)
        vec_transported = transport_vector(vec_src, delta_conn)

        rows.append(
            {
                "src_id": src_id,
                "dst_id": dst_id,
                "transported_vec_x": float(vec_transported[0]),
                "transported_vec_y": float(vec_transported[1]),
                "dst_vec_x": float(vec_dst[0]),
                "dst_vec_y": float(vec_dst[1]),
                "response_misalignment_deg": float(
                    np.degrees(axial_angle_diff(np.arctan2(vec_transported[1], vec_transported[0]), theta_rsp_dst))
                ),
            }
        )
    return pd.DataFrame(rows)
