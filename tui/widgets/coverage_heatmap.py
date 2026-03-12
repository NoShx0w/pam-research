from __future__ import annotations

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
        self.selected_r: float | None = None
        self.selected_alpha: float | None = None

    def preferred_height(self) -> int:
        r_rows = len(self.spec.r_values)
        return 2 + 2 + r_rows + 2 + 2

    def set_lookup(self, lookup: dict[tuple[float, float], int]) -> None:
        self.lookup = lookup
        self.update(self.render_heatmap())

    def set_selected(self, r_value: float, alpha_value: float) -> None:
        self.selected_r = r_value
        self.selected_alpha = alpha_value
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
        col_width = 6

        text = Text()
        text.append("Seed coverage", style="bold")
        text.append("\n\n")

        alpha_labels = [display_float(a, 3) for a in self.spec.alpha_values]
        header = f"{'r \\ α':>{row_label_width}}" + "".join(f"{label:>{col_width}}" for label in alpha_labels)
        text.append(header)
        text.append("\n")
        text.append("-" * len(header), style="dim")
        text.append("\n")

        for r in self.spec.r_values:
            row_selected = self.selected_r is not None and abs(r - self.selected_r) < 1e-9

            text.append("▶ ", style="bold yellow" if row_selected else "")
            label = f"{display_float(r, 3):>5}"
            text.append(label, style="bold yellow" if row_selected else "")

            for a in self.spec.alpha_values:
                n = self.lookup.get((round(r, 12), round(a, 12)), 0)
                style, char = self.coverage_style_and_char(n, self.spec.seeds_per_cell)
                cell_selected = (
                    self.selected_r is not None
                    and self.selected_alpha is not None
                    and abs(r - self.selected_r) < 1e-9
                    and abs(a - self.selected_alpha) < 1e-9
                )

                if cell_selected:
                    text.append(f"{char:>{col_width}}", style="reverse bold")
                elif row_selected:
                    text.append(f"{char:>{col_width}}", style=f"{style} bold")
                else:
                    text.append(f"{char:>{col_width}}", style=style)

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
        self.styles.height = self.preferred_height()
        self.update(self.render_heatmap())
