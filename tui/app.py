from __future__ import annotations

from time import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header

from tui.config import INDEX_PATH, REFRESH_SECONDS, SWEEP_SPEC_PATH
from tui.controllers.selection import SelectionState
from tui.data_loader import (
    load_cell_detail,
    load_or_create_sweep_spec,
    load_phase_metric,
    load_row_detail,
    load_snapshot,
    load_trajectory_detail,
)
from tui.widgets.coverage_heatmap import CoverageHeatmap
from tui.widgets.detail_view import DetailView
from tui.widgets.panel import Panel
from tui.widgets.phase_diagram import PhaseDiagram


class PAMTUI(App):
    BINDINGS = [
        Binding("up", "prev_r", "Prev r"),
        Binding("down", "next_r", "Next r"),
        Binding("left", "prev_alpha", "Prev α"),
        Binding("right", "next_alpha", "Next α"),
        Binding("enter", "toggle_mode", "Toggle mode"),
        Binding("t", "trajectory_mode", "Trajectory"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #left { width: 42; min-width: 40; }
    #right { width: 1fr; }

    Panel, CoverageHeatmap, DetailView, PhaseDiagram {
        border: round $primary;
        padding: 1 2;
        margin: 0 1 1 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    #coverage { height: auto; }
    #phase { height: auto; }
    #detail { height: 1fr; }

    #latest, #status, #spec { height: auto; }
    """

    refresh_started_at = reactive(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sweep_spec = load_or_create_sweep_spec(SWEEP_SPEC_PATH)
        self.selection = SelectionState()

        self.last_refresh_time = None
        self.last_completed = None
        self.qph_estimate = None

    @property
    def selected_r(self) -> float:
        return self.selection.selected_r(self.sweep_spec)

    @property
    def selected_alpha(self) -> float:
        return self.selection.selected_alpha(self.sweep_spec)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main"):
            with Vertical(id="left"):
                self.status_panel = Panel("Run status", id="status")
                self.spec_panel = Panel("Sweep spec", id="spec")
                self.latest_panel = Panel("Latest row", id="latest")
                yield self.status_panel
                yield self.spec_panel
                yield self.latest_panel

            with Vertical(id="right"):
                self.coverage_panel = CoverageHeatmap(spec=self.sweep_spec, id="coverage")
                self.phase_panel = PhaseDiagram(spec=self.sweep_spec, metric_name="piF_tail_mean", id="phase")
                self.detail_panel = DetailView(spec=self.sweep_spec, id="detail")
                yield self.coverage_panel
                yield self.phase_panel
                yield self.detail_panel

        yield Footer()

    def on_mount(self) -> None:
        self.title = "PAM Observatory"
        self.sub_title = "Live batch monitor"
        self.refresh_started_at = time()
        self.refresh_data()
        self.set_interval(REFRESH_SECONDS, self.refresh_data)

    def action_prev_r(self) -> None:
        self.selection.move_up()
        self.refresh_data()

    def action_next_r(self) -> None:
        self.selection.move_down(self.sweep_spec)
        self.refresh_data()

    def action_prev_alpha(self) -> None:
        self.selection.move_left()
        self.refresh_data()

    def action_next_alpha(self) -> None:
        self.selection.move_right(self.sweep_spec)
        self.refresh_data()

    def action_toggle_mode(self) -> None:
        self.selection.toggle_mode()
        self.refresh_data()

    def action_trajectory_mode(self) -> None:
        self.selection.set_trajectory_mode()
        self.refresh_data()

    def _format_kv(self, key: str, value: str, key_width: int = 14) -> str:
        return f"{key:<{key_width}} {value}"

    def _update_qph(self, completed: int) -> float:
        now = time()

        if self.last_refresh_time is None or self.last_completed is None:
            self.last_refresh_time = now
            self.last_completed = completed
            return 0.0

        dt = now - self.last_refresh_time
        dc = completed - self.last_completed

        if dt > 0 and dc >= 0:
            inst_qph = dc * 3600.0 / dt
            if self.qph_estimate is None:
                self.qph_estimate = inst_qph
            else:
                self.qph_estimate = 0.7 * self.qph_estimate + 0.3 * inst_qph

        self.last_refresh_time = now
        self.last_completed = completed
        return 0.0 if self.qph_estimate is None else self.qph_estimate

    def refresh_data(self) -> None:
        snap, lookup = load_snapshot(INDEX_PATH, self.sweep_spec)
        phase_values = load_phase_metric(INDEX_PATH, self.sweep_spec, "piF_tail")

        if self.selection.is_row_mode:
            detail = load_row_detail(INDEX_PATH, self.sweep_spec, self.selected_r)
        elif self.selection.is_cell_mode:
            detail = load_cell_detail(INDEX_PATH, self.sweep_spec, self.selected_r, self.selected_alpha)
        else:
            detail = load_trajectory_detail(INDEX_PATH, self.selected_r, self.selected_alpha)

        qph = self._update_qph(snap.completed)
        mode_label = self.selection.mode

        status_lines = [
            self._format_kv("index path", str(INDEX_PATH)),
            self._format_kv("rows loaded", f"{snap.row_count:>6}"),
            self._format_kv("completed", f"{snap.completed:>4} / {snap.expected_total:<4}"),
            self._format_kv("progress", f"{snap.percent:>7.2f}%"),
            self._format_kv("throughput", f"{qph:>9.2f} q/h"),
            self._format_kv("observed r", f"{snap.observed_grid_text.splitlines()[0].split()[-3]} / {snap.observed_grid_text.splitlines()[0].split()[-1]}"),
            self._format_kv("observed α", f"{snap.observed_grid_text.splitlines()[1].split()[-3]} / {snap.observed_grid_text.splitlines()[1].split()[-1]}"),
            self._format_kv("mode", mode_label),
            self._format_kv("selected r", f"{self.selected_r:>7.3f}"),
            self._format_kv("selected α", f"{self.selected_alpha:>7.3f}"),
            self._format_kv("controls", "↑↓ r   ←→ α"),
            self._format_kv("enter", "row/cell toggle"),
            self._format_kv("t", "trajectory mode"),
            self._format_kv("last modified", snap.last_modified),
            self._format_kv("refresh every", f"{REFRESH_SECONDS:.1f}s"),
        ]

        self.status_panel.set_body("\n".join(status_lines))
        self.spec_panel.set_body(snap.sweep_spec_text)
        self.latest_panel.set_body(snap.latest_metrics_text)

        self.coverage_panel.set_lookup(lookup)
        self.coverage_panel.set_selected(self.selected_r, self.selected_alpha)

        self.phase_panel.set_values(phase_values)
        self.phase_panel.set_selected(self.selected_r, self.selected_alpha)

        if self.selection.is_row_mode:
            self.detail_panel.show_row_detail(detail)
        elif self.selection.is_cell_mode:
            self.detail_panel.show_cell_detail(detail)
        else:
            self.detail_panel.show_trajectory_detail(detail)


if __name__ == "__main__":
    PAMTUI().run()
