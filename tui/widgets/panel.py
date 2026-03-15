from __future__ import annotations

from textual.widgets import Static
from textual.reactive import reactive


class Panel(Static):
    """
    Simple bordered text panel used throughout the PAM Observatory TUI.

    Provides:
    - title
    - body text
    - set_body() helper
    """

    title = reactive("")
    body = reactive("")

    def __init__(self, title: str, body: str = "", **kwargs):
        super().__init__("", **kwargs)
        self.title = title
        self.body = body

    def set_body(self, body: str) -> None:
        self.body = body
        self.refresh()

    def render(self) -> str:
        if self.body:
            return f"[bold]{self.title}[/bold]\n\n{self.body}"
        return f"[bold]{self.title}[/bold]"
