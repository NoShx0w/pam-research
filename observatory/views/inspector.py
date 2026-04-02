from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static

from observatory.state import ObservatoryState


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
    except Exception:
        pass
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


class InspectorView(Static):
    def render_from_state(
        self,
        state: ObservatoryState,
        run_summary: dict | None = None,
        geometry_summary: dict | None = None,
        phase_summary: dict | None = None,
        topology_summary: dict | None = None,
        operators_summary: dict | None = None,
        index_mtime: float | None = None,
    ) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_row("Mode", state.mode)
        table.add_row("View", state.view_space.upper())
        table.add_row("Overlay", state.overlay)
        table.add_row("Node", state.selected_node_id)
        table.add_row("i", str(state.selected_i))
        table.add_row("j", str(state.selected_j))
        table.add_row("Grid", f"{state.grid_rows}×{state.grid_cols}")

        if run_summary:
            table.add_row("r", _fmt(run_summary["r"], 3))
            table.add_row("α", _fmt(run_summary["alpha"], 6))
            table.add_row("Rows", str(run_summary["n_rows"]))
            table.add_row("Seeds", str(run_summary["n_seeds"]))

        if geometry_summary:
            table.add_row("r", _fmt(geometry_summary["r"], 3))
            table.add_row("α", _fmt(geometry_summary["alpha"], 6))
            table.add_row("Curv", _fmt(geometry_summary["scalar_curvature"], 3))
            table.add_row("Det", _fmt(geometry_summary["fim_det"], 3))
            table.add_row("Cond", _fmt(geometry_summary["fim_cond"], 3))

        if phase_summary:
            table.add_row("r", _fmt(phase_summary["r"], 3))
            table.add_row("α", _fmt(phase_summary["alpha"], 6))
            table.add_row("Phase", _fmt(phase_summary["signed_phase"], 3))
            table.add_row("Seam d", _fmt(phase_summary["distance_to_seam"], 3))

        if topology_summary:
            table.add_row("r", _fmt(topology_summary["r"], 3))
            table.add_row("α", _fmt(topology_summary["alpha"], 6))
            table.add_row("Crit", _fmt(topology_summary["criticality"], 3))

        if operators_summary:
            table.add_row("r", _fmt(operators_summary["r"], 3))
            table.add_row("α", _fmt(operators_summary["alpha"], 6))
            table.add_row("Lazarus", _fmt(operators_summary["lazarus_score"], 3))

        table.add_row("Refresh", "ON" if state.refresh_enabled else "OFF")
        table.add_row("Status", state.status_message)

        self.update(Panel(table, title="Inspector", border_style="cyan"))