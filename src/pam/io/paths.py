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


@dataclass(frozen=True)
class ObservatoryPaths:
    """Future canonical artifact root. Placeholder-only for now."""

    root: Path

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

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"
