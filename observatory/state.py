from __future__ import annotations

from dataclasses import dataclass, field


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
    observatory_root: str = "observatory"

    right_pane_mode: str = "detail"  # detail | ranking
    ranking_index: int = 0
    marker_mode: str = "off"  # off | seam | critical | obstruction | lazarus

    show_webbing: bool = True
    webbing_mode: str = "all"   # "all" | "local"

    def clamp_selection(self) -> None:
        self.selected_i = max(0, min(self.selected_i, self.grid_rows - 1))
        self.selected_j = max(0, min(self.selected_j, self.grid_cols - 1))

    def clamp_ranking_index(self, n_rows: int) -> None:
        if n_rows <= 0:
            self.ranking_index = 0
        else:
            self.ranking_index = max(0, min(self.ranking_index, n_rows - 1))

    @property
    def selected_node_id(self) -> str:
        return f"{self.selected_i}:{self.selected_j}"