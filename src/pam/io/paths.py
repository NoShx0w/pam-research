from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    """File-first view over the legacy-active outputs artifact store."""

    root: Path

    @property
    def index_csv(self) -> Path:
        return self.root / "index.csv"

    @property
    def trajectories_dir(self) -> Path:
        return self.root / "trajectories"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    # -------------------------------------------------------------------------
    # Core geometry / phase / topology / operators directories
    # -------------------------------------------------------------------------

    @property
    def fim_dir(self) -> Path:
        return self.root / "fim"

    @property
    def fim_distance_dir(self) -> Path:
        return self.root / "fim_distance"

    @property
    def fim_mds_dir(self) -> Path:
        return self.root / "fim_mds"

    @property
    def fim_curvature_dir(self) -> Path:
        return self.root / "fim_curvature"

    @property
    def fim_geodesic_dir(self) -> Path:
        return self.root / "fim_geodesic"

    @property
    def fim_geodesic_fan_dir(self) -> Path:
        return self.root / "fim_geodesic_fan"

    @property
    def fim_phase_dir(self) -> Path:
        return self.root / "fim_phase"

    @property
    def fim_field_alignment_dir(self) -> Path:
        return self.root / "fim_field_alignment"

    @property
    def fim_gradient_alignment_dir(self) -> Path:
        return self.root / "fim_gradient_alignment"

    @property
    def fim_critical_dir(self) -> Path:
        return self.root / "fim_critical"

    @property
    def fim_ops_dir(self) -> Path:
        return self.root / "fim_ops"

    @property
    def fim_ops_scaled_dir(self) -> Path:
        return self.root / "fim_ops_scaled"

    @property
    def fim_lazarus_dir(self) -> Path:
        return self.root / "fim_lazarus"

    @property
    def fim_transition_rate_dir(self) -> Path:
        return self.root / "fim_transition_rate"

    @property
    def fim_initial_conditions_dir(self) -> Path:
        return self.root / "fim_initial_conditions"

    # -------------------------------------------------------------------------
    # First-pass legacy file contracts
    # -------------------------------------------------------------------------

    # Geometry
    @property
    def fim_surface_csv(self) -> Path:
        return self.fim_dir / "fim_surface.csv"

    @property
    def fim_metadata_txt(self) -> Path:
        return self.fim_dir / "fim_metadata.txt"

    @property
    def fisher_nodes_csv(self) -> Path:
        return self.fim_distance_dir / "fisher_nodes.csv"

    @property
    def fisher_edges_csv(self) -> Path:
        return self.fim_distance_dir / "fisher_edges.csv"

    @property
    def fisher_distance_matrix_csv(self) -> Path:
        return self.fim_distance_dir / "fisher_distance_matrix.csv"

    @property
    def mds_coords_csv(self) -> Path:
        return self.fim_mds_dir / "mds_coords.csv"

    @property
    def curvature_surface_csv(self) -> Path:
        # Adjust if your geometry stage writes a different canonical curvature filename.
        return self.fim_curvature_dir / "curvature_surface.csv"

    # Phase
    @property
    def phase_boundary_backprojected_csv(self) -> Path:
        return self.fim_phase_dir / "phase_boundary_mds_backprojected.csv"

    @property
    def phase_distance_to_seam_csv(self) -> Path:
        return self.fim_phase_dir / "phase_distance_to_seam.csv"

    @property
    def signed_phase_coords_csv(self) -> Path:
        return self.fim_phase_dir / "signed_phase_coords.csv"

    # Operators
    @property
    def lazarus_scores_csv(self) -> Path:
        return self.fim_lazarus_dir / "lazarus_scores.csv"

    @property
    def lazarus_summary_csv(self) -> Path:
        return self.fim_lazarus_dir / "lazarus_summary.csv"

    @property
    def transition_rate_states_csv(self) -> Path:
        return self.fim_transition_rate_dir / "transition_rate_states.csv"

    @property
    def transition_rate_labeled_csv(self) -> Path:
        return self.fim_transition_rate_dir / "transition_rate_labeled.csv"

    @property
    def transition_rate_summary_csv(self) -> Path:
        return self.fim_transition_rate_dir / "transition_rate_summary.csv"

    # Topology
    @property
    def criticality_surface_csv(self) -> Path:
        return self.fim_critical_dir / "criticality_surface.csv"

    @property
    def critical_points_csv(self) -> Path:
        return self.fim_critical_dir / "critical_points.csv"

    @property
    def initial_conditions_outcome_summary_csv(self) -> Path:
        return self.fim_initial_conditions_dir / "initial_conditions_outcome_summary.csv"

    @property
    def identity_dir(self) -> Path:
        return self.root / "fim_identity"

    @property
    def identity_holonomy_dir(self) -> Path:
        return self.root / "fim_identity_holonomy"

    @property
    def identity_obstruction_dir(self) -> Path:
        return self.root / "fim_identity_obstruction"

    @property
    def identity_field_nodes_csv(self) -> Path:
        return self.identity_dir / "identity_field_nodes.csv"

    @property
    def identity_field_edges_csv(self) -> Path:
        return self.identity_dir / "identity_field_edges.csv"

    @property
    def identity_spin_csv(self) -> Path:
        return self.identity_dir / "identity_spin.csv"

    @property
    def identity_holonomy_cells_csv(self) -> Path:
        return self.identity_holonomy_dir / "identity_holonomy_cells.csv"

    @property
    def identity_holonomy_alignment_csv(self) -> Path:
        return self.identity_holonomy_dir / "identity_holonomy_alignment.csv"

    @property
    def identity_obstruction_nodes_csv(self) -> Path:
        return self.identity_obstruction_dir / "identity_obstruction_nodes.csv"

    @property
    def identity_obstruction_signed_nodes_csv(self) -> Path:
        return self.identity_obstruction_dir / "identity_obstruction_signed_nodes.csv"


@dataclass(frozen=True)
class ObservatoryPaths:
    """Canonical observatory artifact root for promoted first-class surfaces."""

    root: Path

    # -------------------------------------------------------------------------
    # Top-level structure
    # -------------------------------------------------------------------------

    @property
    def corpora_dir(self) -> Path:
        return self.root / "corpora"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    # -------------------------------------------------------------------------
    # Derived layer directories
    # -------------------------------------------------------------------------

    @property
    def geometry_dir(self) -> Path:
        return self.derived_dir / "geometry"

    @property
    def phase_dir(self) -> Path:
        return self.derived_dir / "phase"

    @property
    def topology_dir(self) -> Path:
        return self.derived_dir / "topology"

    @property
    def operators_dir(self) -> Path:
        return self.derived_dir / "operators"

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    @property
    def geometry_metric_dir(self) -> Path:
        return self.geometry_dir / "metric"

    @property
    def geometry_graph_dir(self) -> Path:
        return self.geometry_dir / "graph"

    @property
    def geometry_distance_dir(self) -> Path:
        return self.geometry_dir / "distance"

    @property
    def geometry_mds_dir(self) -> Path:
        return self.geometry_dir / "mds"

    @property
    def geometry_curvature_dir(self) -> Path:
        return self.geometry_dir / "curvature"

    @property
    def geometry_metric_surface_csv(self) -> Path:
        return self.geometry_metric_dir / "fim_surface.csv"

    @property
    def geometry_metric_metadata_txt(self) -> Path:
        return self.geometry_metric_dir / "fim_metadata.txt"

    @property
    def geometry_graph_nodes_csv(self) -> Path:
        return self.geometry_graph_dir / "nodes.csv"

    @property
    def geometry_graph_edges_csv(self) -> Path:
        return self.geometry_graph_dir / "edges.csv"

    @property
    def geometry_distance_matrix_csv(self) -> Path:
        return self.geometry_distance_dir / "distance_matrix.csv"

    @property
    def geometry_mds_coords_csv(self) -> Path:
        return self.geometry_mds_dir / "coords.csv"

    @property
    def geometry_curvature_surface_csv(self) -> Path:
        return self.geometry_curvature_dir / "curvature_surface.csv"

    # -------------------------------------------------------------------------
    # Phase
    # -------------------------------------------------------------------------

    @property
    def phase_boundary_dir(self) -> Path:
        return self.phase_dir / "boundary"

    @property
    def phase_signed_phase_csv(self) -> Path:
        return self.phase_dir / "signed_phase.csv"

    @property
    def phase_distance_to_seam_csv(self) -> Path:
        return self.phase_dir / "distance_to_seam.csv"

    @property
    def phase_boundary_csv(self) -> Path:
        return self.phase_boundary_dir / "seam_nodes.csv"

    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------

    @property
    def operators_lazarus_dir(self) -> Path:
        return self.operators_dir / "lazarus"

    @property
    def operators_transition_rate_dir(self) -> Path:
        return self.operators_dir / "transition_rate"

    @property
    def operators_lazarus_scores_csv(self) -> Path:
        return self.operators_lazarus_dir / "lazarus_scores.csv"

    @property
    def operators_lazarus_summary_csv(self) -> Path:
        return self.operators_lazarus_dir / "lazarus_summary.csv"

    @property
    def operators_transition_rate_states_csv(self) -> Path:
        return self.operators_transition_rate_dir / "transition_rate_states.csv"

    @property
    def operators_transition_rate_labeled_csv(self) -> Path:
        return self.operators_transition_rate_dir / "transition_rate_labeled.csv"

    @property
    def operators_transition_rate_summary_csv(self) -> Path:
        return self.operators_transition_rate_dir / "transition_rate_summary.csv"

    # -------------------------------------------------------------------------
    # Topology
    # -------------------------------------------------------------------------

    @property
    def topology_criticality_dir(self) -> Path:
        return self.topology_dir / "criticality"

    @property
    def topology_initial_conditions_dir(self) -> Path:
        return self.topology_dir / "initial_conditions"

    @property
    def topology_identity_dir(self) -> Path:
        return self.topology_dir / "identity"

    @property
    def topology_criticality_surface_csv(self) -> Path:
        return self.topology_criticality_dir / "criticality_surface.csv"

    @property
    def topology_critical_points_csv(self) -> Path:
        return self.topology_criticality_dir / "critical_points.csv"

    @property
    def topology_initial_conditions_summary_csv(self) -> Path:
        return self.topology_initial_conditions_dir / "initial_conditions_outcome_summary.csv"

    @property
    def topology_identity_magnitude_csv(self) -> Path:
        return self.topology_identity_dir / "identity_magnitude.csv"

    @property
    def topology_absolute_holonomy_csv(self) -> Path:
        return self.topology_identity_dir / "absolute_holonomy.csv"

    @property
    def topology_unsigned_obstruction_csv(self) -> Path:
        return self.topology_identity_dir / "unsigned_obstruction.csv"

    @property
    def topology_signed_obstruction_csv(self) -> Path:
        return self.topology_identity_dir / "signed_obstruction.csv"

    @property
    def topology_identity_field_nodes_csv(self) -> Path:
        return self.topology_identity_dir / "identity_field_nodes.csv"

    @property
    def topology_identity_field_edges_csv(self) -> Path:
        return self.topology_identity_dir / "identity_field_edges.csv"

    @property
    def topology_identity_spin_csv(self) -> Path:
        return self.topology_identity_dir / "identity_spin.csv"

    @property
    def topology_identity_holonomy_cells_csv(self) -> Path:
        return self.topology_identity_dir / "identity_holonomy_cells.csv"

    @property
    def topology_identity_holonomy_alignment_csv(self) -> Path:
        return self.topology_identity_dir / "identity_holonomy_alignment.csv"

    @property
    def topology_identity_obstruction_nodes_csv(self) -> Path:
        return self.topology_identity_dir / "identity_obstruction_nodes.csv"

    @property
    def topology_identity_obstruction_signed_nodes_csv(self) -> Path:
        return self.topology_identity_dir / "identity_obstruction_signed_nodes.csv"

    @property
    def topology_annotations_dir(self) -> Path:
        return self.topology_dir / "annotations"

    @property
    def topology_hub_nodes_csv(self) -> Path:
        return self.topology_annotations_dir / "hub_nodes.csv"

    @property
    def topology_hotspot_nodes_csv(self) -> Path:
        return self.topology_annotations_dir / "hotspot_nodes.csv"

    @property
    def topology_seam_bundle_nodes_csv(self) -> Path:
        return self.topology_annotations_dir / "seam_bundle_nodes.csv"

    @property
    def topology_seam_bundle_embedding_summary_csv(self) -> Path:
        return self.topology_annotations_dir / "seam_bundle_embedding_summary.csv"

    @property
    def topology_seam_bundle_family_summary_csv(self) -> Path:
        return self.topology_annotations_dir / "seam_bundle_family_summary.csv"