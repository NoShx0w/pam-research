from __future__ import annotations

from rich.panel import Panel
from textual.widgets import Static

from observatory.keymap import KEY_HINTS
from observatory.state import ObservatoryState
from observatory.views.formatting import mode_label, overlay_label, overlay_meta


class FooterView(Static):
    def render_from_state(self, state: ObservatoryState) -> None:
        text = " • ".join(KEY_HINTS)
        meta = overlay_meta(state.overlay)
        title = f"{mode_label(state.mode)} | {overlay_label(state.overlay)} | {meta['kind']}"
        self.update(Panel(text, title=title, border_style="yellow"))