from __future__ import annotations

from rich.panel import Panel
from textual.widgets import Static

from observatory.keymap import KEY_HINTS
from observatory.state import ObservatoryState
from observatory.views.formatting import mode_label, overlay_label


class FooterView(Static):
    def render_from_state(self, state: ObservatoryState) -> None:
        text = " • ".join(KEY_HINTS)
        title = f"{mode_label(state.mode)} | {overlay_label(state.overlay)}"
        self.update(Panel(text, title=title, border_style="yellow"))