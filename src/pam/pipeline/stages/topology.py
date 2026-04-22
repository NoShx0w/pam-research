from __future__ import annotations

from pathlib import Path

import pandas as pd

from pam.pipeline.artifacts import mirror_file
from pam.pipeline.state import PipelineState
from pam.topology.critical_points import run_critical_points
from pam.topology.field import run_field_alignment
from pam.topology.flow import run_gradient_alignment
from pam.topology.organization import run_organization

from pam.topology.identity_field import compute_identity_field
from pam.topology.identity_proxy import (
    IdentityProxyConfig,
    build_local_identity_graphs,
    identity_grid_from_node_graphs,
    load_identity_proxy_inputs,
)
from pam.topology.identity_transport import (
    IdentityHolonomyConfig,
    build_identity_holonomy_table,
    load_identity_transport_nodes,
)
from pam.topology.identity_obstruction import (
    IdentityObstructionConfig,
    build_identity_obstruction_table,
    load_identity_obstruction_inputs,
)


def _run_identity_topology_outputs(
    state: PipelineState,
    *,
    seam_eps: float = 0.15,
    criticality_quantile: float = 0.90,
    normalized_distance: bool = True,
) -> dict[str, str]:
    """
    Produce canonical identity field + holonomy artifacts inside the topology stage.

    Writes:
      outputs/fim_identity/
        - identity_field_nodes.csv
        - identity_field_edges.csv
        - identity_spin.csv

      outputs/fim_identity_holonomy/
        - identity_holonomy_cells.csv
        - identity_holonomy_alignment.csv

      outputs/fim_identity_obstruction/
        - identity_obstruction_nodes.csv
        - identity_obstruction_signed_nodes.csv
    """
    fim_identity_dir = state.outputs.identity_dir
    fim_identity_holonomy_dir = state.outputs.identity_holonomy_dir
    fim_identity_dir.mkdir(parents=True, exist_ok=True)
    fim_identity_holonomy_dir.mkdir(parents=True, exist_ok=True)

    proxy_node_df, proxy_edge_df = load_identity_proxy_inputs(
        nodes_csv=state.outputs.fisher_nodes_csv,
        edges_csv=state.outputs.fisher_edges_csv,
        criticality_csv=state.outputs.criticality_surface_csv,
        phase_distance_csv=state.outputs.phase_distance_to_seam_csv,
    )

    identity_graphs = build_local_identity_graphs(
        node_df=proxy_node_df,
        edge_df=proxy_edge_df,
        config=IdentityProxyConfig(
            seam_eps=seam_eps,
            criticality_quantile=criticality_quantile,
        ),
    )

    identity_grid = identity_grid_from_node_graphs(
        node_df=proxy_node_df,
        identity_graphs=identity_graphs,
    )
    field = compute_identity_field(
        identity_grid,
        normalized=normalized_distance,
    )

    work = proxy_node_df.copy().sort_values(["i", "j"]).reset_index(drop=True)
    work["node_id"] = pd.to_numeric(work["node_id"], errors="coerce").astype("Int64").astype(str)

    lookup = {
        (int(row["i"]), int(row["j"])): row
        for _, row in work.iterrows()
    }

    field_rows = []
    spin_rows = []
    edge_rows = []

    n_i, n_j = field.magnitude.shape
    for i in range(n_i):
        for j in range(n_j):
            row = lookup[(i, j)]
            field_rows.append(
                {
                    "node_id": row["node_id"],
                    "i": i,
                    "j": j,
                    "r": float(row["r"]),
                    "alpha": float(row["alpha"]),
                    "identity_vx": float(field.vx[i, j]),
                    "identity_vy": float(field.vy[i, j]),
                    "identity_magnitude": float(field.magnitude[i, j]),
                    "identity_spin": float(field.spin[i, j]),
                }
            )
            spin_rows.append(
                {
                    "node_id": row["node_id"],
                    "i": i,
                    "j": j,
                    "r": float(row["r"]),
                    "alpha": float(row["alpha"]),
                    "identity_spin": float(field.spin[i, j]),
                }
            )

            if j < n_j - 1:
                dst = lookup[(i, j + 1)]
                edge_rows.append(
                    {
                        "direction": "alpha",
                        "src_node_id": row["node_id"],
                        "dst_node_id": dst["node_id"],
                        "src_i": i,
                        "src_j": j,
                        "dst_i": i,
                        "dst_j": j + 1,
                        "src_r": float(row["r"]),
                        "src_alpha": float(row["alpha"]),
                        "dst_r": float(dst["r"]),
                        "dst_alpha": float(dst["alpha"]),
                        "identity_distance": float(field.vx[i, j]),
                    }
                )

            if i < n_i - 1:
                dst = lookup[(i + 1, j)]
                edge_rows.append(
                    {
                        "direction": "r",
                        "src_node_id": row["node_id"],
                        "dst_node_id": dst["node_id"],
                        "src_i": i,
                        "src_j": j,
                        "dst_i": i + 1,
                        "dst_j": j,
                        "src_r": float(row["r"]),
                        "src_alpha": float(row["alpha"]),
                        "dst_r": float(dst["r"]),
                        "dst_alpha": float(dst["alpha"]),
                        "identity_distance": float(field.vy[i, j]),
                    }
                )

    pd.DataFrame(field_rows).to_csv(state.outputs.identity_field_nodes_csv, index=False)
    pd.DataFrame(edge_rows).to_csv(state.outputs.identity_field_edges_csv, index=False)
    pd.DataFrame(spin_rows).to_csv(state.outputs.identity_spin_csv, index=False)

    holonomy_nodes_df = load_identity_transport_nodes(
        identity_nodes_csv=state.outputs.identity_field_nodes_csv,
    )
    hol_df = build_identity_holonomy_table(
        nodes_df=holonomy_nodes_df,
        identity_graphs=identity_graphs,
        config=IdentityHolonomyConfig(
            normalized_distance=normalized_distance,
        ),
    )

    hol_df.to_csv(state.outputs.identity_holonomy_cells_csv, index=False)

    def _corr_summary(df: pd.DataFrame, x: str, y: str) -> dict[str, float]:
        work = df[[x, y]].dropna()
        if len(work) < 3:
            return {
                "metric_x": x,
                "metric_y": y,
                "n": len(work),
                "pearson": float("nan"),
                "spearman": float("nan"),
            }
        return {
            "metric_x": x,
            "metric_y": y,
            "n": int(len(work)),
            "pearson": float(work[x].corr(work[y], method="pearson")),
            "spearman": float(work[x].corr(work[y], method="spearman")),
        }

    hol_align = pd.DataFrame(
        [
            _corr_summary(hol_df, "holonomy_residual", "mean_abs_corner_spin"),
            _corr_summary(hol_df, "holonomy_residual", "max_abs_corner_spin"),
            _corr_summary(hol_df, "abs_holonomy_residual", "mean_abs_corner_spin"),
            _corr_summary(hol_df, "abs_holonomy_residual", "max_abs_corner_spin"),
        ]
    )
    hol_align.to_csv(state.outputs.identity_holonomy_alignment_csv, index=False)

    obstruction_nodes_df, obstruction_cells_df = load_identity_obstruction_inputs(
        identity_nodes_csv=state.outputs.identity_field_nodes_csv,
        holonomy_cells_csv=state.outputs.identity_holonomy_cells_csv,
    )

    obstruction_df = build_identity_obstruction_table(
        nodes=obstruction_nodes_df,
        cells=obstruction_cells_df,
        config=IdentityObstructionConfig(
            weight_signed_by_abs_holonomy=True,
        ),
    )

    state.outputs.identity_obstruction_dir.mkdir(parents=True, exist_ok=True)
    obstruction_df.to_csv(state.outputs.identity_obstruction_nodes_csv, index=False)
    obstruction_df.to_csv(state.outputs.identity_obstruction_signed_nodes_csv, index=False)

    return {
        "identity_field_nodes_csv": str(state.outputs.identity_field_nodes_csv),
        "identity_field_edges_csv": str(state.outputs.identity_field_edges_csv),
        "identity_spin_csv": str(state.outputs.identity_spin_csv),
        "identity_holonomy_cells_csv": str(state.outputs.identity_holonomy_cells_csv),
        "identity_holonomy_alignment_csv": str(state.outputs.identity_holonomy_alignment_csv),
        "identity_obstruction_nodes_csv": str(state.outputs.identity_obstruction_nodes_csv),
        "identity_obstruction_signed_nodes_csv": str(state.outputs.identity_obstruction_signed_nodes_csv),
    }


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
      5. Identity field / holonomy / obstruction

    Notes
    -----
    - File-first orchestration over the existing outputs root.
    - Preserves current artifact contracts under outputs/.
    - Mirrors first-pass canonical topology artifacts into observatory/.
    """

    # ------------------------------------------------------------------
    # Legacy-active writes
    # ------------------------------------------------------------------

    run_field_alignment(
        mds_csv=state.outputs.mds_coords_csv,
        phase_csv=state.outputs.signed_phase_coords_csv,
        lazarus_csv=state.outputs.lazarus_scores_csv,
        paths_csv=state.outputs.fim_ops_scaled_dir / "scaled_probe_paths.csv",
        outdir=state.outputs.fim_field_alignment_dir,
    )

    run_gradient_alignment(
        mds_csv=state.outputs.mds_coords_csv,
        phase_csv=state.outputs.signed_phase_coords_csv,
        lazarus_csv=state.outputs.lazarus_scores_csv,
        outdir=state.outputs.fim_gradient_alignment_dir,
    )

    run_critical_points(
        fim_csv=state.outputs.fim_surface_csv,
        curvature_csv=state.outputs.curvature_surface_csv,
        phase_distance_csv=state.outputs.phase_distance_to_seam_csv,
        outdir=state.outputs.fim_critical_dir,
        top_k=critical_top_k,
    )

    run_organization(
        summary_csv=state.outputs.initial_conditions_outcome_summary_csv,
        outdir=state.outputs.fim_initial_conditions_dir,
    )

    identity_outputs = _run_identity_topology_outputs(state)

    # ------------------------------------------------------------------
    # Pass 1 canonical mirrors
    # ------------------------------------------------------------------

    mirror_file(
        state.outputs.criticality_surface_csv,
        state.observatory.topology_criticality_surface_csv,
    )
    mirror_file(
        state.outputs.critical_points_csv,
        state.observatory.topology_critical_points_csv,
    )
    mirror_file(
        state.outputs.initial_conditions_outcome_summary_csv,
        state.observatory.topology_initial_conditions_summary_csv,
    )
    mirror_file(
        state.outputs.identity_field_nodes_csv,
        state.observatory.topology_identity_field_nodes_csv,
    )
    mirror_file(
        state.outputs.identity_field_edges_csv,
        state.observatory.topology_identity_field_edges_csv,
    )
    mirror_file(
        state.outputs.identity_spin_csv,
        state.observatory.topology_identity_spin_csv,
    )
    mirror_file(
        state.outputs.identity_holonomy_cells_csv,
        state.observatory.topology_identity_holonomy_cells_csv,
    )
    mirror_file(
        state.outputs.identity_holonomy_alignment_csv,
        state.observatory.topology_identity_holonomy_alignment_csv,
    )
    mirror_file(
        state.outputs.identity_obstruction_nodes_csv,
        state.observatory.topology_identity_obstruction_nodes_csv,
    )
    mirror_file(
        state.outputs.identity_obstruction_signed_nodes_csv,
        state.observatory.topology_identity_obstruction_signed_nodes_csv,
    )

    return state.with_metadata(
        topology={
            "critical_top_k": critical_top_k,
            "identity_outputs_written": True,
            **identity_outputs,
        }
    )