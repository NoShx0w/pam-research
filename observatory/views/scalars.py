from __future__ import annotations

import math


def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except Exception:
        return True


def render_unsigned_cell(value, vmin: float, vmax: float, selected: bool = False) -> str:
    if selected:
        return "[black on bright_white]●[/]"
    if _is_missing(value):
        return "[dim]·[/]"

    span = max(vmax - vmin, 1e-12)
    z = (float(value) - vmin) / span

    if z < 0.20:
        return "[#3b3b3b]·[/]"
    if z < 0.40:
        return "[#5f875f]▪[/]"
    if z < 0.60:
        return "[#87af5f]■[/]"
    if z < 0.80:
        return "[#afdf5f]■[/]"
    return "[#d7ff5f]■[/]"


def render_signed_cell(value, vabs: float, selected: bool = False) -> str:
    if selected:
        return "[black on bright_white]●[/]"
    if _is_missing(value):
        return "[dim]·[/]"

    x = float(value)
    vabs = max(vabs, 1e-12)

    if abs(x) < 0.10 * vabs:
        return "[#808080]·[/]"

    if x < 0:
        if abs(x) < 0.35 * vabs:
            return "[#5f87d7]▪[/]"
        if abs(x) < 0.65 * vabs:
            return "[#5f87ff]■[/]"
        return "[#3f5fff]■[/]"

    if abs(x) < 0.35 * vabs:
        return "[#d78787]▪[/]"
    if abs(x) < 0.65 * vabs:
        return "[#ff875f]■[/]"
    return "[#ff5f5f]■[/]"
