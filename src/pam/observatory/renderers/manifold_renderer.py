from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DENSITY_GLYPHS = ["·", "░", "▒", "▓", "█"]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _value_for_mode(cell, mode: str) -> float:
    if mode == "coverage":
        return cell.coverage
    if mode == "curvature":
        return cell.curvature
    if mode == "piF_tail":
        return cell.piF_tail
    if mode == "h_joint_mean":
        return cell.h_joint_mean
    return cell.coverage


def _glyph_for_value(value: float) -> str:
    value = _clamp01(value)
    idx = min(int(value * len(DENSITY_GLYPHS)), len(DENSITY_GLYPHS) - 1)
    return DENSITY_GLYPHS[idx]


def _style_for_value(value: float, mode: str, selected: bool) -> str:
    value = _clamp01(value)

    if selected:
        return "bold black on bright_cyan"

    if mode == "curvature":
        if value < 0.20:
            return "grey50"
        if value < 0.40:
            return "khaki3"
        if value < 0.65:
            return "gold3"
        if value < 0.85:
            return "dark_orange3"
        return "bold bright_yellow"

    if value < 0.20:
        return "grey50"
    if value < 0.40:
        return "cyan4"
    if value < 0.65:
        return "cyan3"
    if value < 0.85:
        return "bright_cyan"
    return "bold white"


def _render_cell(cell, mode: str, selected: bool) -> Text:
    if not cell.present and mode == "coverage":
        glyph = "·"
        style = "grey35"
    else:
        value = _value_for_mode(cell, mode)
        glyph = _glyph_for_value(value)
        style = _style_for_value(value, mode, selected)

    if selected:
        """
        return Text.assemble(
            ("[", "bright_cyan"),
            (glyph, style),
            ("]", "bright_cyan"),
        )"""
        return Text(f" {glyph} ", style="bold reverse")
    return Text(f" {glyph} ", style=style)


def render_parameter_manifold_panel(state) -> Panel:
    r_values = state.r_values or []
    alpha_values = state.alpha_values or []

    table = Table.grid(expand=True)
    table.add_column(justify="right", width=7, no_wrap=True)

    for _ in alpha_values:
        table.add_column(justify="center", width=3, no_wrap=True)

    # compact alpha header: 0.03 0.05 0.06 ...
    header = [Text("α →", style="bold white")]
    for alpha in alpha_values:
        header.append(Text(f"{alpha:.2f}".replace("0.", "."), style="grey70"))
    table.add_row(*header)

    for i, r in enumerate(sorted(r_values, reverse=True)):
        row_label = Text(f"{r:.2f}", style="bold white")
        if i == 0:
            row_label = Text.assemble(
                ("r ↓ ", "bold white"),
                (f"{r:.2f}", "bold white"),
            )

        row = [row_label]

        for alpha in alpha_values:
            cell = state.get_cell(r, alpha)
            selected = (r == state.selected_r and alpha == state.selected_alpha)
            row.append(_render_cell(cell, state.color_mode, selected))

        table.add_row(*row)

    subtitle = Text.assemble(
        ("mode: ", "grey62"),
        (state.color_mode, "bold bright_cyan"),
        ("   selected: ", "grey62"),
        (f"r={state.selected_r:.2f} α={state.selected_alpha:.3f}", "bold white"),
    )

    return Panel(
        table,
        title="Parameter Manifold",
        subtitle=subtitle,
        border_style="grey35",
        padding=(0, 1),
    )