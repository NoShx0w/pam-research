from __future__ import annotations

from typing import Optional

from rich.text import Text
from textual.widgets import Static

from tui.models import SweepSpec


def display_float(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.rstrip("0").rstrip(".")


class CoverageHeatmap(Static):
    def __init__(self, spec: SweepSpec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.lookup: dict[tuple[float, float], int] = {}

    def set_lookup(self, lookup: dict[tuple[float, float], int]) -> None:
        self.lookup = lookup
        self.update(self.render_heatmap())

    def coverage_style_and_char(self, count: int, total: int) -> tuple[str, str]:
        frac = 0.0 if total <= 0 else count / total

        if frac <= 0.0:
            return ("dim", "·")
        if frac < 0.25:
            return ("cyan", "░")
        if frac < 0.50:
            return ("green", "▒")
        if frac < 0.75:
            return ("yellow", "▓")
        if frac < 1.0:
            return ("magenta", "█")
        return ("bold red", "█")

    def render_heatmap(self) -> Text:
        row_label_width = 7
        col_width = 5

        text = Text()
        text.append("Seed coverage", style="bold")
        text.append("\n\n")

        alpha_labels = [display_float(a, 3) for a in self.spec.alpha_values]

        header = f"{'r \\ α':>{row_label_width}}" + "".join(
            f"{label:>{col_width}}" for label in alpha_labels
        )
        text.append(header)
        text.append("\n")
        text.append("-" * len(header), style="dim")
        text.append("\n")

        for r in self.spec.r_values:
            text.append(f"{display_float(r, 3):>{row_label_width}}")
            for a in self.spec.alpha_values:
                n = self.lookup.get((round(r, 12), round(a, 12)), 0)
                style, char = self.coverage_style_and_char(n, self.spec.seeds_per_cell)
                cell = f"{char:>{col_width}}"
                text.append(cell, style=style)
            text.append("\n")

        text.append("\n")
        text.append("Legend: ", style="dim")
        text.append("·", style="dim")
        text.append(" empty  ", style="dim")
        text.append("░", style="cyan")
        text.append(" <25%  ", style="dim")
        text.append("▒", style="green")
        text.append(" <50%  ", style="dim")
        text.append("▓", style="yellow")
        text.append(" <75%  ", style="dim")
        text.append("█", style="magenta")
        text.append(" <100%  ", style="dim")
        text.append("█", style="bold red")
        text.append(" full", style="dim")
        text.append("\n")

        text.append(
            f"Grid: {len(self.spec.r_values)} × {len(self.spec.alpha_values)} × {self.spec.seeds_per_cell}",
            style="dim",
        )

        return text

    def on_mount(self) -> None:
        self.update(self.render_heatmap())
