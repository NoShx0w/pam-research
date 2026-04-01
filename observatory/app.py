from __future__ import annotations

import pandas as pd

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding

from observatory.loaders import load_run_data
from observatory.modes import DEFAULT_OVERLAY_BY_MODE, MODES, OVERLAYS_BY_MODE
from observatory.state import ObservatoryState
from observatory.views.detail import DetailView
from observatory.views.footer import FooterView
from observatory.views.inspector import InspectorView
from observatory.views.manifold import ManifoldView


class ObservatoryApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
        width: 100%;
    }

    #left-pane {
        width: 28;
        height: 100%;
    }

    #center-pane {
        width: 1fr;
        height: 100%;
    }

    #right-pane {
        width: 38;
        height: 100%;
    }

    #footer-pane {
        height: 3;
    }

    InspectorView,
    ManifoldView,
    DetailView,
    FooterView {
        width: 100%;
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("1", "set_mode('Run')", "Run"),
        Binding("2", "set_mode('Geometry')", "Geometry"),
        Binding("3", "set_mode('Phase')", "Phase"),
        Binding("4", "set_mode('Topology')", "Topology"),
        Binding("5", "set_mode('Operators')", "Operators"),
        Binding("6", "set_mode('Identity')", "Identity"),
        Binding("left", "move_selection(-1, 0)", "Left"),
        Binding("right", "move_selection(1, 0)", "Right"),
        Binding("up", "move_selection(0, -1)", "Up"),
        Binding("down", "move_selection(0, 1)", "Down"),
        Binding("g", "toggle_view", "Grid/MDS"),
        Binding("o", "next_overlay", "Next overlay"),
        Binding("shift+o", "prev_overlay", "Prev overlay"),
        Binding("r", "refresh_state", "Refresh"),
        Binding("f", "toggle_refresh", "Freeze"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = ObservatoryState()
        self.run_data = load_run_data(self.state.outputs_root)

    def _update_grid_shape_from_run_data(self) -> None:
        coverage = self.run_data.coverage_df
        if coverage.empty:
            self.state.grid_rows = 10
            self.state.grid_cols = 10
            self.state.clamp_selection()
            return

        r_vals = sorted(pd.to_numeric(coverage["r"], errors="coerce").dropna().unique())
        a_vals = sorted(pd.to_numeric(coverage["alpha"], errors="coerce").dropna().unique())

        self.state.grid_rows = max(1, len(r_vals))
        self.state.grid_cols = max(1, len(a_vals))
        self.state.clamp_selection()


    def _selected_run_cell_summary(self) -> dict[str, object]:
        coverage = self.run_data.coverage_df
        if coverage.empty:
            return {
                "r": None,
                "alpha": None,
                "n_rows": 0,
                "n_seeds": 0,
            }

        r_vals = sorted(pd.to_numeric(coverage["r"], errors="coerce").dropna().unique())
        a_vals = sorted(pd.to_numeric(coverage["alpha"], errors="coerce").dropna().unique())

        if not r_vals or not a_vals:
            return {"r": None, "alpha": None, "n_rows": 0, "n_seeds": 0}

        r = r_vals[self.state.selected_i]
        a = a_vals[self.state.selected_j]

        row = coverage[(coverage["r"] == r) & (coverage["alpha"] == a)]
        if row.empty:
            return {"r": r, "alpha": a, "n_rows": 0, "n_seeds": 0}

        rec = row.iloc[0]
        return {
            "r": float(rec["r"]),
            "alpha": float(rec["alpha"]),
            "n_rows": int(rec["n_rows"]),
            "n_seeds": int(rec["n_seeds"]),
        }

    def compose(self) -> ComposeResult:
        yield Vertical(
            Horizontal(
                Container(InspectorView(id="inspector"), id="left-pane"),
                Container(ManifoldView(id="manifold"), id="center-pane"),
                Container(DetailView(id="detail"), id="right-pane"),
                id="main",
            ),
            Container(FooterView(id="footer"), id="footer-pane"),
        )

    def on_mount(self) -> None:
        self._update_grid_shape_from_run_data()
        self._render_all()

    def _render_all(self) -> None:
        run_summary = self._selected_run_cell_summary()
        self.query_one("#inspector", InspectorView).render_from_state(
            self.state,
            run_summary=run_summary,
            index_mtime=self.run_data.index_mtime,
        )
        self.query_one("#manifold", ManifoldView).render_from_state(
            self.state,
            run_data=self.run_data if self.state.mode == "Run" else None,
        )
        self.query_one("#detail", DetailView).render_from_state(
            self.state,
            run_summary=run_summary if self.state.mode == "Run" else None,
        )
        self.query_one("#footer", FooterView).render_from_state(self.state)

    def action_set_mode(self, mode: str) -> None:
        if mode not in MODES:
            return
        self.state.mode = mode
        self.state.overlay = DEFAULT_OVERLAY_BY_MODE[mode]
        self.state.status_message = f"Mode switched to {mode}"
        self._render_all()

    def action_move_selection(self, dx: int, dy: int) -> None:
        self.state.selected_j += dx
        self.state.selected_i += dy
        self.state.clamp_selection()
        self.state.status_message = f"Selected {self.state.selected_node_id}"
        self._render_all()

    def action_toggle_view(self) -> None:
        self.state.view_space = "mds" if self.state.view_space == "grid" else "grid"
        self.state.status_message = f"View set to {self.state.view_space.upper()}"
        self._render_all()

    def action_next_overlay(self) -> None:
        overlays = OVERLAYS_BY_MODE[self.state.mode]
        idx = overlays.index(self.state.overlay) if self.state.overlay in overlays else -1
        self.state.overlay = overlays[(idx + 1) % len(overlays)]
        self.state.status_message = f"Overlay set to {self.state.overlay}"
        self._render_all()

    def action_prev_overlay(self) -> None:
        overlays = OVERLAYS_BY_MODE[self.state.mode]
        idx = overlays.index(self.state.overlay) if self.state.overlay in overlays else 0
        self.state.overlay = overlays[(idx - 1) % len(overlays)]
        self.state.status_message = f"Overlay set to {self.state.overlay}"
        self._render_all()

    def action_refresh_state(self) -> None:
        self.run_data = load_run_data(self.state.outputs_root)
        self._update_grid_shape_from_run_data()
        self.state.status_message = "Refreshed"
        self._render_all()

    def action_toggle_refresh(self) -> None:
        self.state.refresh_enabled = not self.state.refresh_enabled
        self.state.status_message = "Refresh ON" if self.state.refresh_enabled else "Refresh OFF"
        self._render_all()


def main() -> None:
    ObservatoryApp().run()


if __name__ == "__main__":
    main()
