from __future__ import annotations

from rich.panel import Panel
from textual.widgets import Static

from observatory.keymap import KEY_HINTS
from observatory.state import ObservatoryState


class FooterView(Static):
    def render_from_state(self, state: ObservatoryState) -> None:
        text = " • ".join(KEY_HINTS)
        self.update(Panel(text, title=f"{state.mode} | {state.overlay}", border_style="yellow"))
