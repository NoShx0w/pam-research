from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class Panel(Static):
    """Simple titled text panel used throughout the PAM Observatory TUI."""

    def __init__(self, title: str, body: str = "", **kwargs):
        super().__init__("", **kwargs)
        self.title = title
        self.body = body

    def set_body(self, text: str) -> None:
        self.body = text
        self.refresh()

    def render(self) -> Text:
        out = Text()
        out.append(self.title, style="bold")
        if self.body:
            out.append("\n\n")
            out.append(self.body)
        return out
