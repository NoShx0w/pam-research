from __future__ import annotations

from pam.geometry.curvature import run_curvature
from pam.geometry.distance_graph import run_distance_graph
from pam.geometry.embedding import run_embedding
from pam.geometry.fisher_metric import run_fisher_metric
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
    - This stage is file-first: it writes artifacts into the existing outputs root.
    - It returns an updated PipelineState with geometry metadata only.
    """

    if observables is None:
        observables = ["piF_tail", "H_joint_mean"]

    run_fisher_metric(
        index_csv=state.outputs.index_csv,
        outdir=state.outputs.fim_dir,
        corpus=corpus,
        observables=observables,
        ridge_eps=ridge_eps,
    )

    run_distance_graph(
        fim_csv=state.outputs.fim_dir / "fim_surface.csv",
        outdir=state.outputs.fim_distance_dir,
        neighbor_mode=neighbor_mode,
        cost_mode=cost_mode,
        anchor_r=anchor_r,
        anchor_alpha=anchor_alpha,
    )

    run_embedding(
        distance_csv=state.outputs.fim_distance_dir / "fisher_distance_matrix.csv",
        nodes_csv=state.outputs.fim_distance_dir / "fisher_nodes.csv",
        edges_csv=state.outputs.fim_distance_dir / "fisher_edges.csv",
        fim_csv=state.outputs.fim_dir / "fim_surface.csv",
        outdir=state.outputs.fim_mds_dir,
        color_by=color_by,
    )

    run_curvature(
        fim_csv=state.outputs.fim_dir / "fim_surface.csv",
        outdir=state.outputs.fim_curvature_dir,
    )

    return state.with_metadata(
        geometry={
            "corpus": corpus,
            "observables": list(observables),
            "ridge_eps": ridge_eps,
            "neighbor_mode": neighbor_mode,
            "cost_mode": cost_mode,
            "color_by": color_by,
        }
    )
