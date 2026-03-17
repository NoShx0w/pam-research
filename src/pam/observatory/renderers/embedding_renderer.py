from __future__ import annotations

from rich.panel import Panel
from rich.text import Text


def _normalize(values, out_min=0.0, out_max=1.0):
    values = list(values)
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        mid = (out_min + out_max) / 2
        return [mid for _ in values]
    return [out_min + (v - vmin) * (out_max - out_min) / (vmax - vmin) for v in values]


def _style_for_point(state, point) -> str:
    if point.r == state.selected_r and point.alpha == state.selected_alpha:
        return "bold black on bright_cyan"

    mode = state.embedding_mode
    if mode == "r":
        if point.r <= 0.10:
            return "cyan4"
        if point.r <= 0.15:
            return "cyan3"
        if point.r <= 0.20:
            return "green3"
        if point.r <= 0.25:
            return "gold3"
        return "bright_yellow"

    if mode == "alpha":
        if point.alpha < 0.045:
            return "grey62"
        if point.alpha < 0.065:
            return "cyan3"
        if point.alpha < 0.085:
            return "magenta"
        return "bright_white"

    cell = state.get_cell(point.r, point.alpha)
    value = max(0.0, min(1.0, cell.curvature))
    if value < 0.20:
        return "grey50"
    if value < 0.40:
        return "khaki3"
    if value < 0.65:
        return "gold3"
    if value < 0.85:
        return "dark_orange3"
    return "bold bright_yellow"


def render_embedding_panel(state, points, width: int = 40, height: int = 12) -> Panel:
    if not points:
        return Panel("no embedding data", title="Manifold Embedding", border_style="grey35")

    xs = _normalize([p.x for p in points], 1, width - 2)
    ys = _normalize([p.y for p in points], height - 2, 1)

    canvas = [[Text(" ", style="white") for _ in range(width)] for _ in range(height)]

    for point, x, y in zip(points, xs, ys):
        xi = max(0, min(width - 1, int(round(x))))
        yi = max(0, min(height - 1, int(round(y))))
        glyph = "◎" if (point.r == state.selected_r and point.alpha == state.selected_alpha) else "•"
        canvas[yi][xi] = Text(glyph, style=_style_for_point(state, point))

    lines = [Text.assemble(*row) for row in canvas]
    body = Text("\n").join(lines)

    source_kind = getattr(points[0], "source_kind", "unknown")
    subtitle = Text.assemble(
        ("mode: ", "grey62"),
        (state.embedding_mode, "bold bright_cyan"),
        ("   source: ", "grey62"),
        (source_kind, "bold white"),
    )

    return Panel(
        body,
        title="Manifold Embedding",
        subtitle=subtitle,
        border_style="grey35",
        padding=(0, 1),
    )
