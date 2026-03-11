from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from tui.models import SweepSpec


def display_float(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.rstrip("0").rstrip(".")


def sparkbar(value: float, vmin: float, vmax: float) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if vmax <= vmin:
        return blocks[0]
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    idx = min(len(blocks) - 1, int(round(t * (len(blocks) - 1))))
    return blocks[idx]


class DetailView(Static):
    def __init__(self, spec: SweepSpec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.mode = "empty"
        self.payload: dict = {}

    def show_empty(self) -> None:
        self.mode = "empty"
        self.payload = {}
        self.update(self.render_detail())

    def show_row_detail(self, row_summary: dict) -> None:
        self.mode = "row"
        self.payload = row_summary
        self.update(self.render_detail())

    def _build_metric_lines(
        self,
        series: list[tuple[float, float | None]],
        label: str,
    ) -> list[str]:
        lines = [label]

        values = [v for _, v in series if v is not None]
        if not values:
            lines.append("  no data")
            return lines

        vmin = min(values)
        vmax = max(values)

        for alpha, value in series:
            if value is None:
                lines.append(f"{display_float(alpha, 3):>6}  ·")
            else:
                glyph = sparkbar(float(value), float(vmin), float(vmax))
                lines.append(f"{display_float(alpha, 3):>6}  {glyph}  {value:.4f}")

        return lines

    def _merge_columns(
        self,
        left_lines: list[str],
        right_lines: list[str],
        left_width: int = 34,
        gap: str = "    ",
    ) -> list[str]:
        n = max(len(left_lines), len(right_lines))
        out = []

        for i in range(n):
            left = left_lines[i] if i < len(left_lines) else ""
            right = right_lines[i] if i < len(right_lines) else ""
            out.append(f"{left:<{left_width}}{gap}{right}")

        return out

    def render_empty(self) -> Text:
        text = Text()
        text.append("Detail view", style="bold")
        text.append("\n\n")
        text.append("No selection yet.", style="dim")
        text.append("\n")
        text.append("Use ↑ and ↓ to inspect fixed-r α sweeps.", style="dim")
        return text

    def render_row_detail(self) -> Text:
        text = Text()
        row_summary = self.payload

        r_value = row_summary.get("r")
        metrics = row_summary.get("metrics", {})
        coverage = row_summary.get("coverage", {})
        alpha_values = row_summary.get("alpha_values", self.spec.alpha_values)

        text.append(f"Detail: r = {display_float(r_value, 3)}", style="bold")
        text.append("\n\n")

        completed_cells = coverage.get("completed_cells", 0)
        total_cells = coverage.get("total_cells", len(alpha_values))
        min_seed_count = coverage.get("min_seed_count", 0)
        max_seed_count = coverage.get("max_seed_count", 0)

        text.append(
            f"α cells observed: {completed_cells} / {total_cells}",
            style="cyan",
        )
        text.append("    ")
        text.append(
            f"seed coverage: min {min_seed_count}   max {max_seed_count}",
            style="cyan",
        )
        text.append("\n\n")

        left_top = self._build_metric_lines(
            metrics.get("piF_tail_mean", []),
            "πF_tail vs α",
        )
        right_top = self._build_metric_lines(
            metrics.get("H_joint_mean_mean", []),
            "H_joint vs α",
        )
        left_bottom = self._build_metric_lines(
            metrics.get("best_corr_mean", []),
            "best_corr vs α",
        )
        right_bottom = self._build_metric_lines(
            metrics.get("delta_r2_freeze_mean", []),
            "ΔR²_freeze vs α",
        )

        top_block = self._merge_columns(left_top, right_top)
        bottom_block = self._merge_columns(left_bottom, right_bottom)

        for line in top_block:
            text.append(line)
            text.append("\n")

        text.append("\n")

        for line in bottom_block:
            text.append(line)
            text.append("\n")

        return text

    def render_detail(self) -> Text:
        if self.mode == "row":
            return self.render_row_detail()
        return self.render_empty()

    def on_mount(self) -> None:
        self.update(self.render_detail())
