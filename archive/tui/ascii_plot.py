from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _nice_num(x: float) -> float:
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    frac = x / (10**exp)
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.0:
        nice = 2.0
    elif frac < 7.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10**exp)


def _resample(series: np.ndarray, width: int) -> np.ndarray:
    if len(series) <= width:
        return series
    xs_old = np.linspace(0, 1, len(series))
    xs_new = np.linspace(0, 1, width)
    return np.interp(xs_new, xs_old, series)


def render_ascii_plot(
    values: Iterable[float],
    *,
    title: str,
    width: int = 56,
    height: int = 10,
) -> list[str]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return [title, "(no data)"]

    arr = _resample(arr, width)

    y_min = float(arr.min())
    y_max = float(arr.max())

    if y_max <= y_min:
        y_max = y_min + 1e-9

    span = y_max - y_min
    step = _nice_num(span / max(1, height - 1))
    y0 = math.floor(y_min / step) * step
    y1 = math.ceil(y_max / step) * step

    rows = []
    for row in range(height):
        frac = row / max(1, height - 1)
        y = y1 - frac * (y1 - y0)
        chars = []
        for v in arr:
            if y0 == y1:
                level = 0.0
            else:
                level = (v - y0) / (y1 - y0)
            plot_row = int(round((1.0 - level) * (height - 1)))
            chars.append("█" if plot_row == row else " ")
        rows.append(f"{y:>8.3f} │{''.join(chars)}")

    axis = f"{'':>8} └" + "─" * len(arr)
    ticks = f"{'':>10}0{' ' * max(1, len(arr) - 6)}{len(values)-1:>4}"

    return [title, *rows, axis, ticks]
