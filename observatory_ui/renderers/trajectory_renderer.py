from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


def _normalize_series(values):
    values = list(values)
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _sparkline(values):
    norm = _normalize_series(values)
    out = []
    for value in norm:
        idx = min(int(value * len(SPARK_GLYPHS)), len(SPARK_GLYPHS) - 1)
        out.append(SPARK_GLYPHS[idx])
    return "".join(out)


def render_trajectory_panel(state, series) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(justify="left", width=12, no_wrap=True)
    table.add_column(justify="left")

    rows = {
        "F_raw": getattr(series, "f_raw", []),
        "H_joint": getattr(series, "h_joint", []),
        "K": getattr(series, "k_series", []),
        "πF_smooth": getattr(series, "pif_smooth", []),
    }

    styles = {
        "F_raw": "bright_cyan",
        "H_joint": "green3",
        "K": "gold3",
        "πF_smooth": "magenta",
    }

    for label, values in rows.items():
        table.add_row(
            Text(label, style="bold white"),
            Text(_sparkline(values), style=styles.get(label, "white")),
        )

    source_kind = getattr(series, "source_kind", "unknown")
    source_path = getattr(series, "source_path", None)

    subtitle = Text.assemble(
        ("selected: ", "grey62"),
        (f"r={state.selected_r:.2f} α={state.selected_alpha:.3f}", "bold white"),
        ("   source: ", "grey62"),
        (source_kind, "bold white"),
    )
    if source_path:
        subtitle.append("   file: ", style="grey62")
        subtitle.append(source_path.name, style="grey70")

    return Panel(
        table,
        title="Trajectory Signatures",
        subtitle=subtitle,
        border_style="grey35",
        padding=(0, 1),
    )
