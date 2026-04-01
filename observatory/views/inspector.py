from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static

from observatory.state import ObservatoryState


class InspectorView(Static):
    def render_from_state(self, state: ObservatoryState) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_row("Mode", state.mode)
        table.add_row("View", state.view_space.upper())
        table.add_row("Overlay", state.overlay)
        table.add_row("Node", state.selected_node_id)
        table.add_row("i", str(state.selected_i))
        table.add_row("j", str(state.selected_j))
        table.add_row("Grid", f"{state.grid_rows}×{state.grid_cols}")
        table.add_row("Refresh", "ON" if state.refresh_enabled else "OFF")
        table.add_row("Status", state.status_message)

        self.update(Panel(table, title="Inspector", border_style="cyan"))