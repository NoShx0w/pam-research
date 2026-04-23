from __future__ import annotations

from pam.pipeline.artifacts import mirror_file, mirror_optional_file
from pam.pipeline.state import PipelineState
from pam.topology.initial_conditions import run_initial_conditions_summary


def run_initial_conditions_stage(
    state: PipelineState,
    *,
    graze_threshold: float = 0.15,
    corpora_py: str = "src/corpora.py",
) -> PipelineState:
    """
    Canonical initial-conditions stage.

    Produces the initial-condition outcome artifacts required by topology.
    """

    outputs = run_initial_conditions_summary(
        paths_csv=state.outputs.fim_ops_scaled_dir / "scaled_probe_paths.csv",
        metrics_csv=state.outputs.fim_ops_scaled_dir / "scaled_probe_metrics.csv",
        corpora_py=corpora_py,
        graze_threshold=graze_threshold,
        outdir=state.outputs.fim_initial_conditions_dir,
    )

    mirror_file(
        state.outputs.initial_conditions_outcome_summary_csv,
        state.observatory.topology_initial_conditions_summary_csv,
    )
    mirror_optional_file(
        state.outputs.fim_initial_conditions_dir / "initial_conditions_outcomes.csv",
        state.observatory.topology_initial_conditions_dir / "initial_conditions_outcomes.csv",
    )
    mirror_optional_file(
        state.outputs.fim_initial_conditions_dir / "link_geometry_summary.csv",
        state.observatory.topology_initial_conditions_dir / "link_geometry_summary.csv",
    )

    return state.with_metadata(
        initial_conditions={
            "graze_threshold": graze_threshold,
            "corpora_py": corpora_py,
            **outputs,
        }
    )
