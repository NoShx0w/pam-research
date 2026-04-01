from __future__ import annotations

from rich.markdown import Markdown
from rich.panel import Panel
from textual.widgets import Static

from observatory.state import ObservatoryState


class DetailView(Static):
    def render_from_state(self, state: ObservatoryState, run_summary: dict | None = None) -> None:
        if state.mode == "Run":
            body = Markdown(
                f'''
### Selected cell
- node_id: `{state.selected_node_id}`
- grid: `({state.selected_i}, {state.selected_j})`
- r: `{"—" if not run_summary or run_summary["r"] is None else f"{run_summary["r"]:.3f}"}`
- alpha: `{"—" if not run_summary or run_summary["alpha"] is None else f"{run_summary["alpha"]:.6f}"}`
- rows: `{"0" if not run_summary else run_summary["n_rows"]}`
- seeds: `{"0" if not run_summary else run_summary["n_seeds"]}`

### Run mode
- real coverage from `outputs/index.csv`
- placeholder trajectory lens deferred to later PR
'''
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
