from textual.widgets import Static


class Panel(Static):
    def __init__(self, title: str, body: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.body = body

    def set_body(self, body: str) -> None:
        self.body = body
        self.update(self.render_text())

    def render_text(self) -> str:
        return f"[b]{self.title}[/b]\n\n{self.body}"

    def on_mount(self) -> None:
        self.update(self.render_text())
