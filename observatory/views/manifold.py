from __future__ import annotations

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static

from observatory.state import ObservatoryState


class ManifoldView(Static):
    def _render_grid(self, state: ObservatoryState) -> Table:
        table = Table.grid(padding=(0, 1))
        for _ in range(state.grid_cols):
            table.add_column(justify="center", width=2)

        for i in range(state.grid_rows):
            row = []
            for j in range(state.grid_cols):
                if i == state.selected_i and j == state.selected_j:
                    row.append("[black on bright_white]●[/]")
                else:
                    row.append("[dim]·[/]")
            table.add_row(*row)
        return table

    def _render_mds_placeholder(self, state: ObservatoryState) -> Table:
        table = Table.grid(padding=(0, 1))
        cols = max(12, state.grid_cols + 4)
        rows = max(8, state.grid_rows)
        for _ in range(cols):
            table.add_column(justify="center", width=2)

        points = {
            (1, 2), (2, 4), (2, 9), (3, 6), (3, 12),
            (4, 3), (4, 10), (5, 7), (6, 5), (6, 11),
            (7, 8), (7, 13),
        }

        sel = (
            min(rows - 1, state.selected_i % rows),
            min(cols - 1, state.selected_j % cols),
        )

        for i in range(rows):
            row = []
            for j in range(cols):
                if (i, j) == sel:
                    row.append("[black on bright_white]●[/]")
                elif (i, j) in points:
                    row.append("[cyan]•[/]")
                else:
                    row.append(" ")
            table.add_row(*row)
        return table

    def render_from_state(self, state: ObservatoryState) -> None:
        content = (
            self._render_grid(state)
            if state.view_space == "grid"
            else self._render_mds_placeholder(state)
        )

        title = f"Manifold — {state.mode} / {state.view_space.upper()} / {state.overlay}"
        self.update(
            Panel(
                Align.center(content, vertical="middle"),
                title=title,
                border_style="green",
            )
        )