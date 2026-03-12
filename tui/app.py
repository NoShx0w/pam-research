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
    load_row_detail,
    load_snapshot,
)
from tui.widgets.coverage_heatmap import CoverageHeatmap
from tui.widgets.detail_view import DetailView
from tui.widgets.panel import Panel


class PAMTUI(App):
    BINDINGS = [
        Binding("up", "prev_r", "Prev r"),
        Binding("down", "next_r", "Next r"),
        Binding("left", "prev_alpha", "Prev α"),
        Binding("right", "next_alpha", "Next α"),
        Binding("enter", "toggle_mode", "Toggle mode"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #left {
        width: 42;
        min-width: 40;
    }

    #right {
        width: 1fr;
    }

    Panel, CoverageHeatmap, DetailView {
        border: round $primary;
        padding: 1 2;
        margin: 0 1 1 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    #coverage {
        height: auto;
    }

    #detail {
        height: 1fr;
    }

    #latest {
        height: auto;
    }

    #status {
        height: auto;
    }

    #spec {
        height: auto;
    }
    """

    refresh_started_at = reactive(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sweep_spec = load_or_create_sweep_spec(SWEEP_SPEC_PATH)
        self.selection = SelectionState()

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
                self.detail_panel = DetailView(spec=self.sweep_spec, id="detail")
                yield self.coverage_panel
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

    def refresh_data(self) -> None:
        snap, lookup = load_snapshot(INDEX_PATH, self.sweep_spec)

        if self.selection.is_row_mode:
            row_detail = load_row_detail(INDEX_PATH, self.sweep_spec, self.selected_r)
        else:
            cell_detail = load_cell_detail(
                INDEX_PATH,
                self.sweep_spec,
                self.selected_r,
                self.selected_alpha,
            )

        elapsed = time() - self.refresh_started_at
        qph = snap.completed / elapsed * 3600.0 if elapsed > 0 else 0.0
        mode_label = "row" if self.selection.is_row_mode else "cell"

        status_text = (
            f"index path      {INDEX_PATH}\n"
            f"rows loaded     {snap.row_count}\n"
            f"completed       {snap.completed} / {snap.expected_total}\n"
            f"progress        {snap.percent:6.2f}%\n"
            f"throughput      {qph:8.2f} q/h\n"
            f"{snap.observed_grid_text}\n"
            f"mode            {mode_label}\n"
            f"selected r      {self.selected_r:.3f}\n"
            f"selected α      {self.selected_alpha:.3f}\n"
            f"controls        ↑↓ r   ←→ α   enter mode\n"
            f"last modified   {snap.last_modified}\n"
            f"refresh every   {REFRESH_SECONDS:.1f}s"
        )

        self.status_panel.set_body(status_text)
        self.spec_panel.set_body(snap.sweep_spec_text)
        self.latest_panel.set_body(snap.latest_metrics_text)
        self.coverage_panel.set_lookup(lookup)
        self.coverage_panel.set_selected(self.selected_r, self.selected_alpha)

        if self.selection.is_row_mode:
            self.detail_panel.show_row_detail(row_detail)
        else:
            self.detail_panel.show_cell_detail(cell_detail)


if __name__ == "__main__":
    PAMTUI().run()
