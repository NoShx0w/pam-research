from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SweepSpec:
    r_values: list[float]
    alpha_values: list[float]
    seeds_per_cell: int

    @property
    def expected_total(self) -> int:
        return len(self.r_values) * len(self.alpha_values) * self.seeds_per_cell


@dataclass
class Snapshot:
    row_count: int
    completed: int
    expected_total: int
    percent: float
    last_modified: str
    latest_metrics_text: str
    coverage_heatmap_text: str
    sweep_spec_text: str
    observed_grid_text: str
