from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, Grid

from pam.observatory.state import ObservatoryState
from pam.observatory.widgets.embedding import EmbeddingWidget
from pam.observatory.widgets.footer import FooterWidget
from pam.observatory.widgets.geodesic import GeodesicWidget
from pam.observatory.widgets.manifold import ParameterManifoldWidget
from pam.observatory.widgets.status import StatusWidget
from pam.observatory.widgets.trajectory import TrajectoryWidget

from pam.observatory.data.adapter import AdapterConfig, load_state_from_index
from pam.observatory.data.embedding_adapter import (
    EmbeddingAdapterConfig,
    load_embedding_points,
)
from pam.observatory.data.trajectory_adapter import (
    TrajectoryAdapterConfig,
    load_trajectory_series,
)
from pam.observatory.data.geodesic_adapter import (
    GeodesicAdapterConfig,
    load_geodesic_probe,
)


class PamObservatory(App):
    CSS_PATH = "assets/observatory.css"

    BINDINGS = [
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("left", "move_left", "Left"),
        ("right", "move_right", "Right"),
        ("m", "cycle_color", "Manifold"),
        ("e", "cycle_embedding", "Embedding"),
        ("g", "cycle_probe", "Probe"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, outputs_dir: str = "outputs") -> None:
        super().__init__()
        self.outputs_dir = Path(outputs_dir)

        self.state: ObservatoryState = load_state_from_index(
            AdapterConfig(
                index_csv=self.outputs_dir / "index.csv",
                dataset_total=750,
            )
        )

        self.embedding_points = []
        self.trajectory_series = None
        self.geodesic_probe = None
        self._load_external_data()

    def _load_external_data(self) -> None:
        self.embedding_points = load_embedding_points(
            EmbeddingAdapterConfig(outputs_dir=self.outputs_dir),
            r_values=self.state.r_values or [],
            alpha_values=self.state.alpha_values or [],
        )

        self.trajectory_series = load_trajectory_series(
            TrajectoryAdapterConfig(
                trajectories_dir=self.outputs_dir / "trajectories",
                max_points=48,
            ),
            r=self.state.selected_r,
            alpha=self.state.selected_alpha,
        )

        self.geodesic_probe = load_geodesic_probe(
            GeodesicAdapterConfig(outputs_dir=self.outputs_dir),
            r=self.state.selected_r,
            alpha=self.state.selected_alpha,
        )

    def compose(self) -> ComposeResult:
        with Container(id="root"):
            with Container(id="status"):
                yield StatusWidget()

            with Grid(id="observatory-grid"):
                yield ParameterManifoldWidget()
                yield EmbeddingWidget()
                yield GeodesicWidget()
                yield TrajectoryWidget()

            with Container(id="footer"):
                yield FooterWidget()

    def _refresh_instrument(self) -> None:
        self._load_external_data()
        for cls in (
            StatusWidget,
            FooterWidget,
            ParameterManifoldWidget,
            EmbeddingWidget,
            GeodesicWidget,
            TrajectoryWidget,
        ):
            self.query_one(cls).refresh()

    def action_move_up(self) -> None:
        self.state.move_selection(dr=+1)
        self._refresh_instrument()

    def action_move_down(self) -> None:
        self.state.move_selection(dr=-1)
        self._refresh_instrument()

    def action_move_left(self) -> None:
        self.state.move_selection(da=-1)
        self._refresh_instrument()

    def action_move_right(self) -> None:
        self.state.move_selection(da=+1)
        self._refresh_instrument()

    def action_cycle_color(self) -> None:
        self.state.cycle_color_mode()
        self._refresh_instrument()

    def action_cycle_embedding(self) -> None:
        self.state.cycle_embedding_mode()
        self._refresh_instrument()

    def action_cycle_probe(self) -> None:
        self.state.cycle_probe_mode()
        self._refresh_instrument()

    def action_quit(self) -> None:
        self.exit()


if __name__ == "__main__":
    PamObservatory().run()