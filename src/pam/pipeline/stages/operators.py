from __future__ import annotations

from pam.operators.geodesic_extraction import run_geodesic_extraction
from pam.operators.lazarus import run_lazarus
from pam.operators.probes import run_probes
from pam.operators.scaled_probes import run_scaled_probes
from pam.operators.transition_rate import run_transition_rate
from pam.pipeline.artifacts import mirror_file
from pam.pipeline.state import PipelineState


def run_operators_stage(
    state: PipelineState,
    *,
    lazarus_threshold_quantile: float = 0.85,
    scaled_n_pairs: int = 100,
    scaled_seed: int = 42,
    scaled_max_draw: int = 25,
    transition_within_k: int = 2,
) -> PipelineState:
    """
    Canonical operators stage.

    Runs:
      1. Lazarus regime
      2. Operator S / geodesic extraction
      3. Canonical probe set
      4. Scaled probes
      5. Transition-rate analysis

    Notes
    -----
    - File-first orchestration over the existing outputs root.
    - Preserves current artifact contracts under outputs/.
    - Mirrors first-pass canonical operator artifacts into observatory/.
    """

    # ------------------------------------------------------------------
    # Legacy-active writes
    # ------------------------------------------------------------------

    run_lazarus(
        signed_phase_csv=state.outputs.signed_phase_coords_csv,
        curvature_csv=state.outputs.curvature_surface_csv,
        seam_csv=state.outputs.phase_boundary_backprojected_csv,
        outdir=state.outputs.fim_lazarus_dir,
        threshold_quantile=lazarus_threshold_quantile,
    )

    run_geodesic_extraction(
        edges_csv=state.outputs.fisher_edges_csv,
        mds_csv=state.outputs.mds_coords_csv,
        signed_phase_csv=state.outputs.signed_phase_coords_csv,
        curvature_csv=state.outputs.curvature_surface_csv,
        seam_csv=state.outputs.phase_boundary_backprojected_csv,
        outdir=state.outputs.fim_ops_dir,
    )

    run_probes(
        edges_csv=state.outputs.fisher_edges_csv,
        mds_csv=state.outputs.mds_coords_csv,
        signed_phase_csv=state.outputs.signed_phase_coords_csv,
        curvature_csv=state.outputs.curvature_surface_csv,
        seam_csv=state.outputs.phase_boundary_backprojected_csv,
        outdir=state.outputs.fim_ops_dir,
    )

    run_scaled_probes(
        edges_csv=state.outputs.fisher_edges_csv,
        mds_csv=state.outputs.mds_coords_csv,
        signed_phase_csv=state.outputs.signed_phase_coords_csv,
        curvature_csv=state.outputs.curvature_surface_csv,
        lazarus_csv=state.outputs.lazarus_scores_csv,
        seam_csv=state.outputs.phase_boundary_backprojected_csv,
        outdir=state.outputs.fim_ops_scaled_dir,
        n_pairs=scaled_n_pairs,
        seed=scaled_seed,
        max_draw=scaled_max_draw,
    )

    run_transition_rate(
        paths_csv=state.outputs.fim_ops_scaled_dir / "scaled_probe_paths.csv",
        outdir=state.outputs.fim_transition_rate_dir,
        within_k=transition_within_k,
    )

    # ------------------------------------------------------------------
    # Pass 1 canonical mirrors
    # ------------------------------------------------------------------

    mirror_file(
        state.outputs.lazarus_scores_csv,
        state.observatory.operators_lazarus_scores_csv,
    )
    mirror_file(
        state.outputs.lazarus_summary_csv,
        state.observatory.operators_lazarus_summary_csv,
    )
    mirror_file(
        state.outputs.transition_rate_states_csv,
        state.observatory.operators_transition_rate_states_csv,
    )
    mirror_file(
        state.outputs.transition_rate_labeled_csv,
        state.observatory.operators_transition_rate_labeled_csv,
    )
    mirror_file(
        state.outputs.transition_rate_summary_csv,
        state.observatory.operators_transition_rate_summary_csv,
    )

    return state.with_metadata(
        operators={
            "lazarus_threshold_quantile": lazarus_threshold_quantile,
            "scaled_n_pairs": scaled_n_pairs,
            "scaled_seed": scaled_seed,
            "scaled_max_draw": scaled_max_draw,
            "transition_within_k": transition_within_k,
        }
    )