from __future__ import annotations

from textual.widgets import Static


class Panel(Static):
    """Simple titled text panel used throughout the PAM Observatory TUI."""

    def __init__(self, title: str, body: str = "", **kwargs):
        super().__init__("", **kwargs)
        self.title = title
        self.body = body
        self.update(self._render())

    def _render(self) -> str:
        return f"[bold]{self.title}[/bold]\n\n{self.body}" if self.body else f"[bold]{self.title}[/bold]"

    def set_body(self, text: str) -> None:
        self.body = text
        self.update(self._render())
