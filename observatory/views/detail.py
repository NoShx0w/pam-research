from __future__ import annotations

from rich.markdown import Markdown
from rich.panel import Panel
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
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


class DetailView(Static):
    def render_from_state(
        self,
        state: ObservatoryState,
        run_summary: dict | None = None,
        geometry_summary: dict | None = None,
        phase_summary: dict | None = None,
        topology_summary: dict | None = None,
        operators_summary: dict | None = None,
    ) -> None:
        if state.mode == "Run":
            body = Markdown(
                f"""
### Selected cell
- node_id: `{state.selected_node_id}`
- grid: `({state.selected_i}, {state.selected_j})`
- r: `{_fmt(run_summary["r"], 3) if run_summary else "—"}`
- alpha: `{_fmt(run_summary["alpha"], 6) if run_summary else "—"}`
- rows: `{run_summary["n_rows"] if run_summary else 0}`
- seeds: `{run_summary["n_seeds"] if run_summary else 0}`

### Run mode
- real coverage from `outputs/index.csv`
- trajectory lens deferred to later PR
"""
            )
        elif state.mode == "Geometry":
            body = Markdown(
                f"""
### Selected geometry node
- node_id: `{state.selected_node_id}`
- r: `{_fmt(geometry_summary["r"], 3) if geometry_summary else "—"}`
- alpha: `{_fmt(geometry_summary["alpha"], 6) if geometry_summary else "—"}`
- scalar curvature: `{_fmt(geometry_summary["scalar_curvature"], 3) if geometry_summary else "—"}`
- fim det: `{_fmt(geometry_summary["fim_det"], 3) if geometry_summary else "—"}`
- fim cond: `{_fmt(geometry_summary["fim_cond"], 3) if geometry_summary else "—"}`

### Active overlay
`{state.overlay}`
"""
            )
        elif state.mode == "Phase":
            body = Markdown(
                f"""
### Selected phase node
- node_id: `{state.selected_node_id}`
- r: `{_fmt(phase_summary["r"], 3) if phase_summary else "—"}`
- alpha: `{_fmt(phase_summary["alpha"], 6) if phase_summary else "—"}`
- signed phase: `{_fmt(phase_summary["signed_phase"], 3) if phase_summary else "—"}`
- distance to seam: `{_fmt(phase_summary["distance_to_seam"], 3) if phase_summary else "—"}`

### Active overlay
`{state.overlay}`
"""
            )
        elif state.mode == "Topology":
            body = Markdown(
                f"""
### Selected topology node
- node_id: `{state.selected_node_id}`
- r: `{_fmt(topology_summary["r"], 3) if topology_summary else "—"}`
- alpha: `{_fmt(topology_summary["alpha"], 6) if topology_summary else "—"}`
- criticality: `{_fmt(topology_summary["criticality"], 3) if topology_summary else "—"}`

### Active overlay
`{state.overlay}`
"""
            )
        elif state.mode == "Operators":
            body = Markdown(
                f"""
### Selected operator node
- node_id: `{state.selected_node_id}`
- r: `{_fmt(operators_summary["r"], 3) if operators_summary else "—"}`
- alpha: `{_fmt(operators_summary["alpha"], 6) if operators_summary else "—"}`
- lazarus: `{_fmt(operators_summary["lazarus_score"], 3) if operators_summary else "—"}`

### Active overlay
`{state.overlay}`
"""
            )
        elif state.mode == "Identity":
            body = Markdown(
                f"""
### Selected node
- node_id: `{state.selected_node_id}`
- grid: `({state.selected_i}, {state.selected_j})`

### Primary identity overlays
- identity magnitude
- absolute holonomy
- unsigned local obstruction
- signed local obstruction

### Comparison overlay
- legacy spin

### Active overlay
`{state.overlay}`
"""
            )
        else:
            body = Markdown(
                f"""
### Selected node
- node_id: `{state.selected_node_id}`
- grid: `({state.selected_i}, {state.selected_j})`

### Active mode
`{state.mode}`

### Active overlay
`{state.overlay}`
"""
            )

        self.update(Panel(body, title="Detail", border_style="magenta"))