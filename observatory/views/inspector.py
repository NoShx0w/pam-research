from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from textual.widgets import Static

from observatory.state import ObservatoryState
from observatory.views.formatting import fmt_value, mode_label, overlay_label


class InspectorView(Static):
    def render_from_state(
        self,
        state: ObservatoryState,
        run_summary: dict | None = None,
        geometry_summary: dict | None = None,
        phase_summary: dict | None = None,
        topology_summary: dict | None = None,
        operators_summary: dict | None = None,
        identity_summary: dict | None = None,
        index_mtime: float | None = None,
    ) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_row("Mode", mode_label(state.mode))
        table.add_row("View", state.view_space.upper())
        table.add_row("Overlay", overlay_label(state.overlay))
        table.add_row("Node", state.selected_node_id)
        table.add_row("i", str(state.selected_i))
        table.add_row("j", str(state.selected_j))
        table.add_row("Grid", f"{state.grid_rows}×{state.grid_cols}")

        if run_summary:
            table.add_row("r", fmt_value(run_summary["r"], 3))
            table.add_row("α", fmt_value(run_summary["alpha"], 6))
            table.add_row("Rows", str(run_summary["n_rows"]))
            table.add_row("Seeds", str(run_summary["n_seeds"]))

        if geometry_summary:
            table.add_row("r", fmt_value(geometry_summary["r"], 3))
            table.add_row("α", fmt_value(geometry_summary["alpha"], 6))
            table.add_row("Curv", fmt_value(geometry_summary["scalar_curvature"], 3))
            table.add_row("Det", fmt_value(geometry_summary["fim_det"], 3))
            table.add_row("Cond", fmt_value(geometry_summary["fim_cond"], 3))

        if phase_summary:
            table.add_row("r", fmt_value(phase_summary["r"], 3))
            table.add_row("α", fmt_value(phase_summary["alpha"], 6))
            table.add_row("Phase", fmt_value(phase_summary["signed_phase"], 3))
            table.add_row("Seam d", fmt_value(phase_summary["distance_to_seam"], 3))

        if topology_summary:
            table.add_row("r", fmt_value(topology_summary["r"], 3))
            table.add_row("α", fmt_value(topology_summary["alpha"], 6))
            table.add_row("Crit", fmt_value(topology_summary["criticality"], 3))

        if operators_summary:
            table.add_row("r", fmt_value(operators_summary["r"], 3))
            table.add_row("α", fmt_value(operators_summary["alpha"], 6))
            table.add_row("Lazarus", fmt_value(operators_summary["lazarus_score"], 3))

        if identity_summary:
            table.add_row("r", fmt_value(identity_summary["r"], 3))
            table.add_row("α", fmt_value(identity_summary["alpha"], 6))
            table.add_row("Mag", fmt_value(identity_summary["identity_magnitude"], 3))
            table.add_row("Hol | |", fmt_value(identity_summary["absolute_holonomy_node"], 3))
            table.add_row("Obs | |", fmt_value(identity_summary["obstruction_mean_abs_holonomy"], 3))
            table.add_row("Obs ±", fmt_value(identity_summary["obstruction_signed_sum_holonomy"], 3))
            table.add_row("Spin*", fmt_value(identity_summary["identity_spin"], 3))

        table.add_row("Refresh", "ON" if state.refresh_enabled else "OFF")
        table.add_row("Status", state.status_message)

        self.update(Panel(table, title="Inspector", border_style="cyan"))