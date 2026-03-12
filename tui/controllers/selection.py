from __future__ import annotations

from dataclasses import dataclass

from tui.models import SweepSpec


@dataclass
class SelectionState:
    r_index: int = 0
    alpha_index: int = 0
    mode: str = "row"  # row, cell, trajectory

    @property
    def is_row_mode(self) -> bool:
        return self.mode == "row"

    @property
    def is_cell_mode(self) -> bool:
        return self.mode == "cell"

    @property
    def is_trajectory_mode(self) -> bool:
        return self.mode == "trajectory"

    def selected_r(self, spec: SweepSpec) -> float:
        return spec.r_values[self.r_index]

    def selected_alpha(self, spec: SweepSpec) -> float:
        return spec.alpha_values[self.alpha_index]

    def move_up(self) -> None:
        if self.r_index > 0:
            self.r_index -= 1

    def move_down(self, spec: SweepSpec) -> None:
        if self.r_index < len(spec.r_values) - 1:
            self.r_index += 1

    def move_left(self) -> None:
        if self.alpha_index > 0:
            self.alpha_index -= 1

    def move_right(self, spec: SweepSpec) -> None:
        if self.alpha_index < len(spec.alpha_values) - 1:
            self.alpha_index += 1

    def toggle_mode(self) -> None:
        self.mode = "cell" if self.mode == "row" else "row"

    def set_trajectory_mode(self) -> None:
        self.mode = "trajectory"
