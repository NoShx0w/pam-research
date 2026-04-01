from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObservatoryState:
    mode: str = "Run"
    view_space: str = "grid"
    overlay: str = "coverage"

    selected_i: int = 0
    selected_j: int = 0
    grid_rows: int = 10
    grid_cols: int = 10

    refresh_enabled: bool = True
    status_message: str = "Ready"

    outputs_root: str = "outputs"

    def clamp_selection(self) -> None:
        self.selected_i = max(0, min(self.selected_i, self.grid_rows - 1))
        self.selected_j = max(0, min(self.selected_j, self.grid_cols - 1))

    @property
    def selected_node_id(self) -> str:
        return f"{self.selected_i}:{self.selected_j}"
