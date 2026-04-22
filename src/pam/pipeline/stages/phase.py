from __future__ import annotations

from pam.phase.seam import run_seam_extraction
from pam.phase.seam_distance import run_seam_distance
from pam.phase.seam_embedding import run_seam_embedding
from pam.phase.signed_phase import run_signed_phase
from pam.pipeline.artifacts import mirror_file
from pam.pipeline.state import PipelineState


def run_phase_stage(
    state: PipelineState,
    *,
    seam_threshold: float = 10.0,
    seam_samples: int = 100,
) -> PipelineState:
    """
    Canonical phase stage.

    Runs:
      1. Seam extraction from curvature
      2. Seam embedding / backprojection in MDS space
      3. Distance to seam
      4. Signed phase coordinate

    Notes
    -----
    - File-first orchestration over the existing outputs root.
    - Preserves current artifact contracts under outputs/fim_phase.
    - Mirrors first-pass canonical phase artifacts into observatory/.
    """

    # ------------------------------------------------------------------
    # Legacy-active writes
    # ------------------------------------------------------------------

    run_seam_extraction(
        curvature_csv=state.outputs.curvature_surface_csv,
        outdir=state.outputs.fim_phase_dir,
        threshold=seam_threshold,
    )

    run_seam_embedding(
        boundary_csv=state.outputs.fim_phase_dir / "phase_boundary_points.csv",
        mds_csv=state.outputs.mds_coords_csv,
        outdir=state.outputs.fim_phase_dir,
        n_samples=seam_samples,
    )

    run_seam_distance(
        distance_csv=state.outputs.fisher_distance_matrix_csv,
        nodes_csv=state.outputs.fisher_nodes_csv,
        seam_csv=state.outputs.phase_boundary_backprojected_csv,
        outdir=state.outputs.fim_phase_dir,
    )

    run_signed_phase(
        mds_csv=state.outputs.mds_coords_csv,
        seam_csv=state.outputs.phase_boundary_backprojected_csv,
        phase_distance_csv=state.outputs.phase_distance_to_seam_csv,
        outdir=state.outputs.fim_phase_dir,
    )

    # ------------------------------------------------------------------
    # Pass 1 canonical mirrors
    # ------------------------------------------------------------------

    mirror_file(
        state.outputs.phase_boundary_backprojected_csv,
        state.observatory.phase_boundary_csv,
    )
    mirror_file(
        state.outputs.phase_distance_to_seam_csv,
        state.observatory.phase_distance_to_seam_csv,
    )
    mirror_file(
        state.outputs.signed_phase_coords_csv,
        state.observatory.phase_signed_phase_csv,
    )

    return state.with_metadata(
        phase={
            "seam_threshold": seam_threshold,
            "seam_samples": seam_samples,
        }
    )