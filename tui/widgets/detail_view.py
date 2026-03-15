from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from textual.widgets import Static

from .panel import Panel


TRAJECTORY_DIR = Path("outputs/trajectories")


@dataclass
class DetailSelection:
    mode: str  # "row" | "cell" | "trajectory"
    selected_r: float | None = None
    selected_alpha: float | None = None
    selected_seed: int | None = 0


def display_float(x: float | int | np.floating | np.integer, digits: int = 3) -> str:
    x = float(x)
    s = f"{x:.{digits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sorted_unique_numeric(series: pd.Series) -> list[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    return sorted(float(v) for v in vals)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "·"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _bar(value: float, lo: float, hi: float, width: int = 10) -> str:
    if not np.isfinite(value):
        return "·"
    if hi <= lo:
        return "█"
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    n = max(1, int(round(frac * width)))
    return "█" * n


def _metric_block(
    frame: pd.DataFrame,
    metric: str,
    alpha_col: str = "alpha",
    bar_width: int = 10,
) -> list[str]:
    if metric not in frame.columns:
        return [f"{metric}", "missing"]

    sub = frame[[alpha_col, metric]].dropna().copy()
    if sub.empty:
        return [metric, "no data"]

    values = sub[metric].astype(float).to_numpy()
    lo = float(np.min(values))
    hi = float(np.max(values))

    lines = [metric]
    for _, row in sub.iterrows():
        a = display_float(float(row[alpha_col]), 3)
        v = float(row[metric])
        bar = _bar(v, lo, hi, width=bar_width)
        lines.append(f"{a:>6}  {bar:<{bar_width}}  {v:>8.4f}")
    return lines


def _hstack_blocks(blocks: Sequence[list[str]], gap: int = 6) -> str:
    widths = [max((len(line) for line in block), default=0) for block in blocks]
    height = max((len(block) for block in blocks), default=0)
    out: list[str] = []
    for i in range(height):
        parts = []
        for width, block in zip(widths, blocks):
            line = block[i] if i < len(block) else ""
            parts.append(line.ljust(width))
        out.append((" " * gap).join(parts).rstrip())
    return "\n".join(out)


def _compose_grid_2x2(
    top_left: str,
    top_right: str,
    bottom_left: str,
    bottom_right: str,
    gap: int = 4,
) -> str:
    def lines(s: str) -> list[str]:
        return s.splitlines() or [""]

    tl = lines(top_left)
    tr = lines(top_right)
    bl = lines(bottom_left)
    br = lines(bottom_right)

    top_width = max((len(line) for line in tl), default=0)
    bot_width = max((len(line) for line in bl), default=0)
    left_width = max(top_width, bot_width)

    top_h = max(len(tl), len(tr))
    bot_h = max(len(bl), len(br))

    out: list[str] = []

    for i in range(top_h):
        l = tl[i] if i < len(tl) else ""
        r = tr[i] if i < len(tr) else ""
        out.append(l.ljust(left_width) + (" " * gap) + r)

    out.append("")

    for i in range(bot_h):
        l = bl[i] if i < len(bl) else ""
        r = br[i] if i < len(br) else ""
        out.append(l.ljust(left_width) + (" " * gap) + r)

    return "\n".join(line.rstrip() for line in out)


def _series_plot(
    values: Sequence[float],
    title: str,
    width: int = 34,
    height: int = 8,
) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return f"{title}\n(no data)"

    # Downsample / resample to plot width.
    xs = np.linspace(0, len(arr) - 1, min(width, len(arr))).astype(int)
    sample = arr[xs]

    lo = float(np.min(sample))
    hi = float(np.max(sample))
    span = hi - lo if hi > lo else 1.0

    canvas = [[" " for _ in range(len(sample))] for _ in range(height)]
    for x, y in enumerate(sample):
        frac = (y - lo) / span
        row = height - 1 - int(round(frac * (height - 1)))
        row = max(0, min(height - 1, row))
        canvas[row][x] = "█"

    lines = [title]
    for i, row in enumerate(canvas):
        label_val = hi - (span * i / max(1, height - 1))
        lines.append(f"{label_val:>7.3f} │ " + "".join(row))
    lines.append(f"{'':>7} └" + "─" * (len(sample) + 1))
    lines.append(f"{'':>9}0{' ' * max(0, len(sample)-8)}{len(arr)-1}")
    return "\n".join(lines)


def _load_trajectory_npz(
    selected_r: float,
    selected_alpha: float,
    selected_seed: int | None,
) -> dict[str, np.ndarray] | None:
    if not TRAJECTORY_DIR.exists():
        return None

    seed = 0 if selected_seed is None else int(selected_seed)

    patterns = [
        f"*r{selected_r:.2f}*a{selected_alpha:.3f}*seed{seed}*.npz",
        f"*r{selected_r:.2f}*alpha{selected_alpha:.3f}*seed{seed}*.npz",
        f"*r{selected_r:.3f}*a{selected_alpha:.3f}*seed{seed}*.npz",
        f"*seed{seed}*.npz",
    ]

    matches: list[Path] = []
    for pattern in patterns:
        matches = sorted(TRAJECTORY_DIR.glob(pattern))
        if matches:
            break

    if not matches:
        return None

    try:
        with np.load(matches[0], allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception:
        return None


def _first_available(data: dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            return arr
    return None


def build_row_mode_text(
    df: pd.DataFrame,
    selected_r: float,
    alpha_values: Sequence[float] | None = None,
) -> tuple[str, str]:
    work = _safe_numeric(
        df,
        ["r", "alpha", "seed", "piF_tail", "H_joint_mean", "best_corr", "delta_r2_freeze"],
    )
    row = work[np.isclose(work["r"], selected_r, atol=1e-12)].copy()

    if row.empty:
        return f"Detail: row mode   r = {display_float(selected_r, 3)}", "No data for selected r."

    grouped = (
        row.groupby("alpha", dropna=True)
        .agg(
            piF_tail=("piF_tail", "mean"),
            H_joint=("H_joint_mean", "mean"),
            best_corr=("best_corr", "mean"),
            dR2_freeze=("delta_r2_freeze", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values("alpha")
    )

    observed = len(grouped)
    total = len(alpha_values) if alpha_values is not None else observed
    min_seed = int(grouped["seeds"].min()) if not grouped.empty else 0
    max_seed = int(grouped["seeds"].max()) if not grouped.empty else 0

    title = f"Detail: row mode   r = {display_float(selected_r, 3)}"
    header = (
        f"α cells observed: {observed} / {total}    "
        f"seed coverage: min {min_seed}    max {max_seed}"
    )

    blocks = [
        _metric_block(grouped, "piF_tail"),
        _metric_block(grouped, "H_joint"),
        _metric_block(grouped, "best_corr"),
        _metric_block(grouped, "dR2_freeze"),
    ]

    body = header + "\n\n" + _hstack_blocks(blocks, gap=6)
    return title, body


def build_cell_mode_text(df: pd.DataFrame, selected_r: float, selected_alpha: float) -> tuple[str, str]:
    work = _safe_numeric(
        df,
        ["r", "alpha", "seed", "piF_tail", "H_joint_mean", "best_corr", "corr0", "delta_r2_freeze", "delta_r2_entropy", "K_max"],
    )
    cell = work[
        np.isclose(work["r"], selected_r, atol=1e-12)
        & np.isclose(work["alpha"], selected_alpha, atol=1e-12)
    ].copy()

    title = (
        "Detail: cell mode   "
        f"r = {display_float(selected_r, 3)}   α = {display_float(selected_alpha, 3)}"
    )

    if cell.empty:
        return title, "No data for selected cell."

    means = {
        "πF_tail": cell["piF_tail"].mean() if "piF_tail" in cell.columns else np.nan,
        "H_joint": cell["H_joint_mean"].mean() if "H_joint_mean" in cell.columns else np.nan,
        "best_corr": cell["best_corr"].mean() if "best_corr" in cell.columns else np.nan,
        "corr0": cell["corr0"].mean() if "corr0" in cell.columns else np.nan,
        "ΔR²_freeze": cell["delta_r2_freeze"].mean() if "delta_r2_freeze" in cell.columns else np.nan,
        "ΔR²_entropy": cell["delta_r2_entropy"].mean() if "delta_r2_entropy" in cell.columns else np.nan,
        "K_max": cell["K_max"].mean() if "K_max" in cell.columns else np.nan,
    }

    lines = [f"seed coverage: {len(cell)} / {len(cell)}", "", "cell means"]
    for key, value in means.items():
        lines.append(f"{key:<12} {_format_value(value)}")

    lines.extend(["", "per-seed"])
    wanted = ["seed", "piF_tail", "H_joint_mean", "best_corr", "delta_r2_freeze"]
    present = [c for c in wanted if c in cell.columns]
    table = cell[present].sort_values("seed")
    header = "  ".join(f"{c:>12}" for c in present)
    lines.append(header)
    for _, row in table.iterrows():
        lines.append("  ".join(f"{_format_value(row[c]):>12}" for c in present))

    return title, "\n".join(lines)


def build_trajectory_mode_text(
    df: pd.DataFrame,
    selected_r: float,
    selected_alpha: float,
    selected_seed: int | None = 0,
) -> tuple[str, str]:
    title = (
        "Detail: trajectory mode   "
        f"r = {display_float(selected_r, 3)}   α = {display_float(selected_alpha, 3)}   "
        f"seed = {selected_seed if selected_seed is not None else 0}"
    )

    data = _load_trajectory_npz(selected_r, selected_alpha, selected_seed)
    if data is None:
        return title, "No trajectory file found for selected cell."

    f_raw = _first_available(data, ["F_raw", "freeze", "pi_raw"])
    h_joint = _first_available(data, ["H_joint", "H_joint_series", "entropy"])
    k_series = _first_available(data, ["K", "K_series", "complexity"])
    pi_sm = _first_available(data, ["pi", "pi_smooth", "piF_smooth", "F_smooth"])

    tl = _series_plot(f_raw if f_raw is not None else [], "F_raw(t)")
    tr = _series_plot(h_joint if h_joint is not None else [], "H_joint(t)")
    bl = _series_plot(k_series if k_series is not None else [], "K(t)")
    br = _series_plot(pi_sm if pi_sm is not None else [], "πF_smooth(t)")

    body = _compose_grid_2x2(tl, tr, bl, br, gap=4)
    return title, body


class DetailView(Panel):
    """Detail panel for row / cell / trajectory inspection.

    Assumptions:
    - `df` is the current index.csv dataframe
    - `selection.mode` is one of: row, cell, trajectory
    - trajectory files live under outputs/trajectories/
    - `panel.py` exposes Panel(title: str, body: str = "")
    """

    def __init__(self, **kwargs):
        super().__init__("Detail", id="detail", **kwargs)

    def update_detail(
        self,
        df: pd.DataFrame,
        selection: DetailSelection,
        alpha_values: Sequence[float] | None = None,
    ) -> None:
        if selection.selected_r is None:
            self.title = "Detail"
            self.set_body("No selection.")
            return

        mode = selection.mode.lower().strip()

        if mode == "row":
            title, body = build_row_mode_text(df, selection.selected_r, alpha_values=alpha_values)
        elif mode == "cell":
            if selection.selected_alpha is None:
                title, body = "Detail: cell mode", "No α selected."
            else:
                title, body = build_cell_mode_text(df, selection.selected_r, selection.selected_alpha)
        elif mode == "trajectory":
            if selection.selected_alpha is None:
                title, body = "Detail: trajectory mode", "No α selected."
            else:
                title, body = build_trajectory_mode_text(
                    df,
                    selection.selected_r,
                    selection.selected_alpha,
                    selection.selected_seed,
                )
        else:
            title, body = "Detail", f"Unknown mode: {selection.mode}"

        self.title = title
        self.set_body(body)
