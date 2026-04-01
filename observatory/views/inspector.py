from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static

from observatory.state import ObservatoryState


class InspectorView(Static):
    def render_from_state(self, state: ObservatoryState, run_summary: dict | None = None, index_mtime: float | None = None) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_row("Mode", state.mode)
        table.add_row("View", state.view_space.upper())
        table.add_row("Overlay", state.overlay)
        table.add_row("Node", state.selected_node_id)
        table.add_row("i", str(state.selected_i))
        table.add_row("j", str(state.selected_j))
        table.add_row("Grid", f"{state.grid_rows}×{state.grid_cols}")

        if run_summary:
            table.add_row("r", "—" if run_summary["r"] is None else f"{run_summary['r']:.3f}")
            table.add_row("α", "—" if run_summary["alpha"] is None else f"{run_summary['alpha']:.6f}")
            table.add_row("Rows", str(run_summary["n_rows"]))
            table.add_row("Seeds", str(run_summary["n_seeds"]))

        table.add_row("Refresh", "ON" if state.refresh_enabled else "OFF")
        table.add_row("Status", state.status_message)

        self.update(Panel(table, title="Inspector", border_style="cyan"))
