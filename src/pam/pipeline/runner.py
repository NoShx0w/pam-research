from __future__ import annotations

from pathlib import Path

from pam.pipeline.stages.geometry import run_geometry_stage
from pam.pipeline.stages.operators import run_operators_stage
from pam.pipeline.stages.phase import run_phase_stage
from pam.pipeline.stages.topology import run_topology_stage
from pam.pipeline.state import PipelineState


def run_pipeline(
    *,
    outputs_root: str | Path = "outputs",
    observatory_root: str | Path = "observatory",
    corpus: str | None = None,
    geometry_observables: list[str] | None = None,
    geometry_ridge_eps: float = 1e-8,
    geometry_neighbor_mode: str = "4",
    geometry_cost_mode: str = "midpoint",
    geometry_anchor_r: float | None = None,
    geometry_anchor_alpha: float | None = None,
    geometry_color_by: str = "fim_det",
    phase_seam_threshold: float = 10.0,
    phase_seam_samples: int = 100,
    operators_lazarus_threshold_quantile: float = 0.85,
    operators_scaled_n_pairs: int = 100,
    operators_scaled_seed: int = 42,
    operators_scaled_max_draw: int = 25,
    operators_transition_within_k: int = 2,
    topology_critical_top_k: int = 5,
) -> PipelineState:
    """
    Canonical PAM pipeline runner.

    Stage order:
      1. geometry
      2. phase
      3. operators
      4. topology

    Notes
    -----
    - File-first orchestration over the existing outputs root.
    - Returns the final PipelineState with accumulated metadata.
    """

    state = PipelineState.from_roots(
        outputs_root=outputs_root,
        observatory_root=observatory_root,
    )

    state = run_geometry_stage(
        state,
        corpus=corpus,
        observables=geometry_observables,
        ridge_eps=geometry_ridge_eps,
        neighbor_mode=geometry_neighbor_mode,
        cost_mode=geometry_cost_mode,
        anchor_r=geometry_anchor_r,
        anchor_alpha=geometry_anchor_alpha,
        color_by=geometry_color_by,
    )

    state = run_phase_stage(
        state,
        seam_threshold=phase_seam_threshold,
        seam_samples=phase_seam_samples,
    )

    state = run_operators_stage(
        state,
        lazarus_threshold_quantile=operators_lazarus_threshold_quantile,
        scaled_n_pairs=operators_scaled_n_pairs,
        scaled_seed=operators_scaled_seed,
        scaled_max_draw=operators_scaled_max_draw,
        transition_within_k=operators_transition_within_k,
    )

    state = run_topology_stage(
        state,
        critical_top_k=topology_critical_top_k,
    )

    return state
