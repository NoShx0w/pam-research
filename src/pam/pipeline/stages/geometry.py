from __future__ import annotations

from pam.geometry.curvature import run_curvature
from pam.geometry.distance_graph import run_distance_graph
from pam.geometry.embedding import run_embedding
from pam.geometry.fisher_metric import run_fisher_metric
from pam.geometry.geodesics import run_geodesic, run_geodesic_fan
from pam.pipeline.artifacts import mirror_file, mirror_optional_file
from pam.pipeline.state import PipelineState


def run_geometry_stage(
    state: PipelineState,
    *,
    corpus: str | None = None,
    observables: list[str] | None = None,
    ridge_eps: float = 1e-8,
    neighbor_mode: str = "4",
    cost_mode: str = "midpoint",
    anchor_r: float | None = None,
    anchor_alpha: float | None = None,
    color_by: str = "fim_det",
    run_single_geodesic: bool = False,
    geodesic_start_r: float | None = None,
    geodesic_start_alpha: float | None = None,
    geodesic_end_r: float | None = None,
    geodesic_end_alpha: float | None = None,
    run_geodesic_fan_stage: bool = False,
    fan_start_r: float | None = None,
    fan_start_alpha: float | None = None,
    fan_target_r: float | None = None,
) -> PipelineState:
    """
    Canonical geometry stage.

    Runs:
      1. Fisher metric
      2. Fisher distance graph
      3. MDS embedding
      4. Scalar curvature

    Notes
    -----
    - This stage remains legacy-compatible by writing to outputs/ first.
    - It mirrors first-pass canonical geometry artifacts into observatory/.
    - It returns an updated PipelineState with geometry metadata only.
    """

    if observables is None:
        observables = ["piF_tail", "H_joint_mean"]

    # ------------------------------------------------------------------
    # Legacy-active writes
    # ------------------------------------------------------------------

    run_fisher_metric(
        index_csv=state.outputs.index_csv,
        outdir=state.outputs.fim_dir,
        corpus=corpus,
        observables=observables,
        ridge_eps=ridge_eps,
    )

    run_distance_graph(
        fim_csv=state.outputs.fim_surface_csv,
        outdir=state.outputs.fim_distance_dir,
        neighbor_mode=neighbor_mode,
        cost_mode=cost_mode,
        anchor_r=anchor_r,
        anchor_alpha=anchor_alpha,
    )

    run_embedding(
        distance_csv=state.outputs.fisher_distance_matrix_csv,
        nodes_csv=state.outputs.fisher_nodes_csv,
        edges_csv=state.outputs.fisher_edges_csv,
        fim_csv=state.outputs.fim_surface_csv,
        outdir=state.outputs.fim_mds_dir,
        color_by=color_by,
    )

    nodes_csv = state.outputs.fisher_nodes_csv
    edges_csv = state.outputs.fisher_edges_csv
    coords_csv = state.outputs.mds_coords_csv

    if run_single_geodesic:
        if None in (
            geodesic_start_r,
            geodesic_start_alpha,
            geodesic_end_r,
            geodesic_end_alpha,
        ):
            raise ValueError(
                "Single geodesic requested, but geodesic_start/end parameters were not fully specified."
            )

        run_geodesic(
            nodes_csv=nodes_csv,
            edges_csv=edges_csv,
            coords_csv=coords_csv,
            r0=geodesic_start_r,
            a0=geodesic_start_alpha,
            r1=geodesic_end_r,
            a1=geodesic_end_alpha,
            outdir=state.outputs.fim_geodesic_dir,
        )

    if run_geodesic_fan_stage:
        if None in (
            fan_start_r,
            fan_start_alpha,
            fan_target_r,
        ):
            raise ValueError(
                "Geodesic fan requested, but fan_start/fan_target parameters were not fully specified."
            )

        run_geodesic_fan(
            nodes_csv=nodes_csv,
            edges_csv=edges_csv,
            coords_csv=coords_csv,
            r0=fan_start_r,
            a0=fan_start_alpha,
            r1=fan_target_r,
            outdir=state.outputs.fim_geodesic_fan_dir,
        )

    run_curvature(
        fim_csv=state.outputs.fim_surface_csv,
        outdir=state.outputs.fim_curvature_dir,
    )

    # ------------------------------------------------------------------
    # Pass 1 canonical mirrors
    # ------------------------------------------------------------------

    mirror_optional_file(
        state.outputs.fim_surface_csv,
        state.observatory.geometry_metric_surface_csv,
    )
    mirror_optional_file(
        state.outputs.fim_metadata_txt,
        state.observatory.geometry_metric_metadata_txt,
    )
    mirror_file(
        state.outputs.fisher_nodes_csv,
        state.observatory.geometry_graph_nodes_csv,
    )
    mirror_file(
        state.outputs.fisher_edges_csv,
        state.observatory.geometry_graph_edges_csv,
    )
    mirror_file(
        state.outputs.fisher_distance_matrix_csv,
        state.observatory.geometry_distance_matrix_csv,
    )
    mirror_file(
        state.outputs.mds_coords_csv,
        state.observatory.geometry_mds_coords_csv,
    )
    mirror_optional_file(
        state.outputs.curvature_surface_csv,
        state.observatory.geometry_curvature_surface_csv,
    )

    return state.with_metadata(
        geometry={
            "corpus": corpus,
            "observables": list(observables),
            "ridge_eps": ridge_eps,
            "neighbor_mode": neighbor_mode,
            "cost_mode": cost_mode,
            "color_by": color_by,
            "run_single_geodesic": run_single_geodesic,
            "geodesic_start_r": geodesic_start_r,
            "geodesic_start_alpha": geodesic_start_alpha,
            "geodesic_end_r": geodesic_end_r,
            "geodesic_end_alpha": geodesic_end_alpha,
            "run_geodesic_fan_stage": run_geodesic_fan_stage,
            "fan_start_r": fan_start_r,
            "fan_start_alpha": fan_start_alpha,
            "fan_target_r": fan_target_r,
        }
    )