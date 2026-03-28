from __future__ import annotations

from pam.pipeline.state import PipelineState
from pam.topology.critical_points import run_critical_points
from pam.topology.field import run_field_alignment
from pam.topology.flow import run_gradient_alignment
from pam.topology.organization import run_organization


def run_topology_stage(
    state: PipelineState,
    *,
    critical_top_k: int = 5,
) -> PipelineState:
    """
    Canonical topology stage.

    Runs:
      1. Field alignment
      2. Gradient alignment
      3. Critical points
      4. Organizational topology / phase selection map

    Notes
    -----
    - File-first orchestration over the existing outputs root.
    - Preserves current artifact contracts under outputs/.
    """

    run_field_alignment(
        mds_csv=state.outputs.fim_mds_dir / "mds_coords.csv",
        phase_csv=state.outputs.fim_phase_dir / "signed_phase_coords.csv",
        lazarus_csv=state.outputs.fim_lazarus_dir / "lazarus_scores.csv",
        paths_csv=state.outputs.fim_ops_scaled_dir / "scaled_probe_paths.csv",
        outdir=state.outputs.fim_field_alignment_dir,
    )

    run_gradient_alignment(
        mds_csv=state.outputs.fim_mds_dir / "mds_coords.csv",
        phase_csv=state.outputs.fim_phase_dir / "signed_phase_coords.csv",
        lazarus_csv=state.outputs.fim_lazarus_dir / "lazarus_scores.csv",
        outdir=state.outputs.fim_gradient_alignment_dir,
    )

    run_critical_points(
        fim_csv=state.outputs.fim_dir / "fim_surface.csv",
        curvature_csv=state.outputs.fim_curvature_dir / "curvature_surface.csv",
        phase_distance_csv=state.outputs.fim_phase_dir / "phase_distance_to_seam.csv",
        outdir=state.outputs.fim_critical_dir,
        top_k=critical_top_k,
    )

    run_organization(
        summary_csv=state.outputs.fim_initial_conditions_dir / "initial_conditions_outcome_summary.csv",
        outdir=state.outputs.fim_initial_conditions_dir,
    )

    return state.with_metadata(
        topology={
            "critical_top_k": critical_top_k,
        }
    )
