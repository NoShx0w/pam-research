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


def _render_path(probe, width: int = 28, height: int = 9) -> Text:
    points = probe.path_points or []
    if not points:
        return Text("no path data", style="grey62")

    rs = [p[0] for p in points]
    alphas = [p[1] for p in points]
    xs = _normalize(alphas, 2, width - 3)
    ys = _normalize(rs, 1, height - 2)

    canvas = [[" " for _ in range(width)] for _ in range(height)]

    for i, (x, y) in enumerate(zip(xs, ys)):
        xi = max(0, min(width - 1, int(round(x))))
        yi = max(0, min(height - 1, int(round(y))))
        canvas[yi][xi] = "⊙" if i == 0 else "•"

    lines = []
    for row in canvas:
        line = Text()
        for char in row:
            style = "grey70"
            if char == "⊙":
                style = "bold bright_cyan"
            elif char == "•":
                style = "white"
            line.append(char, style=style)
        lines.append(line)
    return Text("\n").join(lines)


def _render_fan(probe, width: int = 28, height: int = 9) -> Text:
    rays = probe.fan_rays or []
    all_points = [pt for ray in rays for pt in ray]
    if not all_points:
        return Text("no fan data", style="grey62")

    rs = [p[0] for p in all_points]
    alphas = [p[1] for p in all_points]
    xs = _normalize(alphas, 2, width - 3)
    ys = _normalize(rs, 1, height - 2)

    point_map = {}
    idx = 0
    for ray in rays:
        ray_points = []
        for _ in ray:
            ray_points.append((xs[idx], ys[idx]))
            idx += 1
        point_map[id(ray)] = ray_points

    canvas = [[" " for _ in range(width)] for _ in range(height)]

    for ray, scaled in zip(rays, point_map.values()):
        for i, (x, y) in enumerate(scaled):
            xi = max(0, min(width - 1, int(round(x))))
            yi = max(0, min(height - 1, int(round(y))))
            canvas[yi][xi] = "⊙" if i == 0 else "•"

    if probe.shear_level == "high":
        canvas[min(height - 1, height // 2 + 2)][min(width - 1, width // 2)] = "↯"

    lines = []
    for row in canvas:
        line = Text()
        for char in row:
            style = "grey70"
            if char == "⊙":
                style = "bold bright_cyan"
            elif char == "•":
                style = "gold3"
            elif char == "↯":
                style = "bold bright_yellow"
            line.append(char, style=style)
        lines.append(line)
    return Text("\n").join(lines)


def render_geodesic_panel(state, probe) -> Panel:
    if probe is None:
        return Panel(
            Text("no probe data", style="grey62"),
            title="Geodesic Probe",
            border_style="grey35",
            padding=(0, 1),
        )
    mode = state.probe_mode
    body = _render_path(probe) if mode == "path" else _render_fan(probe)

    subtitle = Text.assemble(
        ("mode: ", "grey62"),
        (mode, "bold bright_cyan"),
        ("   source: ", "grey62"),
        (getattr(probe, "source_kind", "unknown"), "bold white"),
        ("   shear: ", "grey62"),
        (getattr(probe, "shear_level", "unknown"), "gold3"),
    )

    return Panel(
        body,
        title="Geodesic Probe",
        subtitle=subtitle,
        border_style="grey35",
        padding=(0, 1),
    )
