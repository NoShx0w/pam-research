from __future__ import annotations

from rich.markdown import Markdown
from rich.panel import Panel
from textual.widgets import Static

from observatory.state import ObservatoryState
from observatory.views.formatting import fmt_value, overlay_label, overlay_meta


class DetailView(Static):
    def render_from_state(
        self,
        state: ObservatoryState,
        run_summary: dict | None = None,
        geometry_summary: dict | None = None,
        phase_summary: dict | None = None,
        topology_summary: dict | None = None,
        operators_summary: dict | None = None,
        identity_summary: dict | None = None,
        transitions_summary: dict | None = None,
    ) -> None:
        meta = overlay_meta(state.overlay)
        cue_block = f"""
### Field cue
- overlay: `{overlay_label(state.overlay)}`
- type: `{meta["kind"]}`
- encoding: `{meta["encoding"]}`
- meaning: {meta["meaning"]}
"""

        if state.mode == "Run":
            body = Markdown(
                f"""
### Selected cell
- node_id: `{state.selected_node_id}`
- grid: `({state.selected_i}, {state.selected_j})`
- r: `{fmt_value(run_summary["r"], 3) if run_summary else "—"}`
- alpha: `{fmt_value(run_summary["alpha"], 6) if run_summary else "—"}`
- rows: `{run_summary["n_rows"] if run_summary else 0}`
- seeds: `{run_summary["n_seeds"] if run_summary else 0}`

### Run mode
- file-backed coverage from `outputs/index.csv`
- trajectory lens deferred
{cue_block}
"""
            )

        elif state.mode == "Geometry":
            body = Markdown(
                f"""
### Selected geometry node
- node_id: `{state.selected_node_id}`
- r: `{fmt_value(geometry_summary["r"], 3) if geometry_summary else "—"}`
- alpha: `{fmt_value(geometry_summary["alpha"], 6) if geometry_summary else "—"}`
- scalar curvature: `{fmt_value(geometry_summary["scalar_curvature"], 3) if geometry_summary else "—"}`
- fim determinant: `{fmt_value(geometry_summary["fim_det"], 3) if geometry_summary else "—"}`
- fim condition #: `{fmt_value(geometry_summary["fim_cond"], 3) if geometry_summary else "—"}`

### Active overlay
`{overlay_label(state.overlay)}`
{cue_block}
"""
            )

        elif state.mode == "Phase":
            body = Markdown(
                f"""
### Selected phase node
- node_id: `{state.selected_node_id}`
- r: `{fmt_value(phase_summary["r"], 3) if phase_summary else "—"}`
- alpha: `{fmt_value(phase_summary["alpha"], 6) if phase_summary else "—"}`
- signed phase: `{fmt_value(phase_summary["signed_phase"], 3) if phase_summary else "—"}`
- distance to seam: `{fmt_value(phase_summary["distance_to_seam"], 3) if phase_summary else "—"}`

### Active overlay
`{overlay_label(state.overlay)}`
{cue_block}
"""
            )

        elif state.mode == "Topology":
            body = Markdown(
                f"""
### Selected topology node
- node_id: `{state.selected_node_id}`
- r: `{fmt_value(topology_summary["r"], 3) if topology_summary else "—"}`
- alpha: `{fmt_value(topology_summary["alpha"], 6) if topology_summary else "—"}`
- criticality: `{fmt_value(topology_summary["criticality"], 3) if topology_summary else "—"}`

### Active overlay
`{overlay_label(state.overlay)}`
{cue_block}
"""
            )

        elif state.mode == "Operators":
            body = Markdown(
                f"""
### Selected operator node
- node_id: `{state.selected_node_id}`
- r: `{fmt_value(operators_summary["r"], 3) if operators_summary else "—"}`
- alpha: `{fmt_value(operators_summary["alpha"], 6) if operators_summary else "—"}`
- lazarus: `{fmt_value(operators_summary["lazarus_score"], 3) if operators_summary else "—"}`

### Active overlay
`{overlay_label(state.overlay)}`
{cue_block}
"""
            )

        elif state.mode == "Identity":
            body = Markdown(
                f"""
### Selected identity node
- node_id: `{state.selected_node_id}`
- r: `{fmt_value(identity_summary["r"], 3) if identity_summary else "—"}`
- alpha: `{fmt_value(identity_summary["alpha"], 6) if identity_summary else "—"}`
- identity magnitude: `{fmt_value(identity_summary["identity_magnitude"], 3) if identity_summary else "—"}`
- absolute holonomy: `{fmt_value(identity_summary["absolute_holonomy_node"], 3) if identity_summary else "—"}`
- unsigned obstruction: `{fmt_value(identity_summary["obstruction_mean_abs_holonomy"], 3) if identity_summary else "—"}`
- max unsigned obstruction: `{fmt_value(identity_summary["obstruction_max_abs_holonomy"], 3) if identity_summary else "—"}`
- signed obstruction: `{fmt_value(identity_summary["obstruction_signed_sum_holonomy"], 3) if identity_summary else "—"}`
- weighted signed obstruction: `{fmt_value(identity_summary["obstruction_signed_weighted_holonomy"], 3) if identity_summary else "—"}`
- legacy spin: `{fmt_value(identity_summary["identity_spin"], 3) if identity_summary else "—"}`

### Active overlay
`{overlay_label(state.overlay)}`
{cue_block}

### Identity hierarchy
- primary: magnitude / holonomy / obstruction
- comparison: legacy spin
"""
            )

        elif state.mode == "Transitions":
            body = Markdown(
                f"""
### Selected transition node
- node_id: `{state.selected_node_id}`
- r: `{fmt_value(transitions_summary["r"], 3) if transitions_summary else "—"}`
- alpha: `{fmt_value(transitions_summary["alpha"], 6) if transitions_summary else "—"}`
- seam band: `{transitions_summary["seam_band"] if transitions_summary and transitions_summary.get("seam_band") is not None else "—"}`
- coupling class: `{transitions_summary["coupling_class"] if transitions_summary and transitions_summary.get("coupling_class") is not None else "—"}`
- mean lambda local: `{fmt_value(transitions_summary["mean_lambda_local"], 3) if transitions_summary else "—"}`
- bounded share: `{fmt_value(transitions_summary["bounded_share"], 3) if transitions_summary else "—"}`
- recovering landings: `{fmt_value(transitions_summary["recovering_landings"], 0) if transitions_summary else "—"}`
- nonrecovering landings: `{fmt_value(transitions_summary["nonrecovering_landings"], 0) if transitions_summary else "—"}`
- total landings: `{fmt_value(transitions_summary["total_landings"], 0) if transitions_summary else "—"}`
- attractor score: `{fmt_value(transitions_summary["attractor_score"], 3) if transitions_summary else "—"}`
- basin class: `{transitions_summary["basin_class"] if transitions_summary and transitions_summary.get("basin_class") is not None else "—"}`

### Active overlay
`{overlay_label(state.overlay)}`
{cue_block}

### Transition layer notes
- `mean_lambda_local` and `bounded_share` are OBS-051-derived
- `attractor_score` / `basin_class` are OBS-052-derived and provisional
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
`{overlay_label(state.overlay)}`
"""
            )

        self.update(Panel(body, title="Detail", border_style="magenta"))