from __future__ import annotations

from time import time

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header

from tui.config import INDEX_PATH, REFRESH_SECONDS, SWEEP_SPEC_PATH
from tui.data_loader import load_or_create_sweep_spec, load_snapshot
from tui.widgets.coverage_heatmap import CoverageHeatmap
from tui.widgets.panel import Panel


class PAMTUI(App):
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

    Panel, CoverageHeatmap {
        border: round $primary;
        padding: 1 2;
        margin: 0 1 1 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    #coverage {
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

            with Container(id="right"):
                self.coverage_panel = CoverageHeatmap(
                    spec=self.sweep_spec,
                    id="coverage",
                )
                yield self.coverage_panel

        yield Footer()

    def on_mount(self) -> None:
        self.title = "PAM Observatory"
        self.sub_title = "Live batch monitor"
        self.refresh_started_at = time()
        self.refresh_data()
        self.set_interval(REFRESH_SECONDS, self.refresh_data)

    def refresh_data(self) -> None:
        snap, lookup = load_snapshot(INDEX_PATH, self.sweep_spec)
        elapsed = time() - self.refresh_started_at
        qph = snap.completed / elapsed * 3600.0 if elapsed > 0 else 0.0

        status_text = (
            f"index path      {INDEX_PATH}\n"
            f"rows loaded     {snap.row_count}\n"
            f"completed       {snap.completed} / {snap.expected_total}\n"
            f"progress        {snap.percent:6.2f}%\n"
            f"throughput      {qph:8.2f} q/h\n"
            f"{snap.observed_grid_text}\n"
            f"last modified   {snap.last_modified}\n"
            f"refresh every   {REFRESH_SECONDS:.1f}s"
        )

        self.status_panel.set_body(status_text)
        self.spec_panel.set_body(snap.sweep_spec_text)
        self.latest_panel.set_body(snap.latest_metrics_text)
        self.coverage_panel.set_lookup(lookup)


if __name__ == "__main__":
    PAMTUI().run()
