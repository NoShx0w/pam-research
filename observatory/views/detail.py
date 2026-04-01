from __future__ import annotations

from rich.markdown import Markdown
from rich.panel import Panel
from textual.widgets import Static

from observatory.state import ObservatoryState


class DetailView(Static):
    def render_from_state(self, state: ObservatoryState) -> None:
        if state.mode == "Identity":
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

### Artifact status
- loaders: placeholder
- data source: PR A shell
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

### Artifact status
- loaders: placeholder
- data source: PR A shell
"""
            )

        self.update(Panel(body, title="Detail", border_style="magenta"))