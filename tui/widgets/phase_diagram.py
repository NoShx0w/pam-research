from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from tui.models import SweepSpec


def display_float(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.rstrip("0").rstrip(".")


class PhaseDiagram(Static):
    def __init__(self, spec: SweepSpec, metric_name: str = "piF_tail_mean", **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.metric_name = metric_name
        self.values: dict[tuple[float, float], float | None] = {}

    def set_values(self, values: dict[tuple[float, float], float | None]) -> None:
        self.values = values
        self.update(self.render_diagram())

    def _cell_style_and_char(self, value: float | None) -> tuple[str, str]:
        if value is None:
            return ("dim", "·")

        # Assuming piF is in [0, 1]
        v = max(0.0, min(1.0, float(value)))

        if v < 0.10:
            return ("dim", "·")
        if v < 0.25:
            return ("cyan", "░")
        if v < 0.50:
            return ("green", "▒")
        if v < 0.75:
            return ("yellow", "▓")
        return ("bold red", "█")

    def preferred_height(self) -> int:
        return 2 + 2 + len(self.spec.r_values) + 2 + 2

    def render_diagram(self) -> Text:
        row_label_width = 7
        col_width = 6

        text = Text()
        text.append(f"Phase diagram: {self.metric_name}", style="bold")
        text.append("\n\n")

        alpha_labels = [display_float(a, 3) for a in self.spec.alpha_values]
        header = f"{'r \\ α':>{row_label_width}}" + "".join(f"{label:>{col_width}}" for label in alpha_labels)
        text.append(header)
        text.append("\n")
        text.append("-" * len(header), style="dim")
        text.append("\n")

        for r in self.spec.r_values:
            text.append("  ")
            label = f"{display_float(r, 3):>5}"
            text.append(label)

            for a in self.spec.alpha_values:
                value = self.values.get((round(r, 12), round(a, 12)))
                style, char = self._cell_style_and_char(value)
                text.append(f"{char:>{col_width}}", style=style)

            text.append("\n")

        text.append("\n")
        text.append("Legend: ", style="dim")
        text.append("·", style="dim")
        text.append(" low/none  ", style="dim")
        text.append("░", style="cyan")
        text.append(" low  ", style="dim")
        text.append("▒", style="green")
        text.append(" mid  ", style="dim")
        text.append("▓", style="yellow")
        text.append(" high  ", style="dim")
        text.append("█", style="bold red")
        text.append(" very high", style="dim")
        text.append("\n")
        text.append("Current metric aggregated live from index.csv", style="dim")

        return text

    def on_mount(self) -> None:
        self.styles.height = self.preferred_height()
        self.update(self.render_diagram())
