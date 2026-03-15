from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .panel import Panel


TRAJECTORY_DIR = Path("outputs/trajectories")


@dataclass
class DetailSelection:
    mode: str  # "row" | "cell" | "trajectory"
    selected_r: float | None
    selected_alpha: float | None
    selected_seed: int | None = 0


def display_float(x: float | int, digits: int = 3) -> str:
    s = f"{float(x):.{digits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = max(1, int(round(frac * width)))
    return "█" * n


def _metric_block(frame: pd.DataFrame, metric: str, alpha_col: str = "alpha", bar_width: int = 10) -> list[str]:
    if metric not in frame.columns:
        return [metric, "missing"]

    sub = frame[[alpha_col, metric]].dropna().copy()
    if sub.empty:
        return [metric, "no data"]

    values = sub[metric].astype(float).to_numpy()
    lo, hi = float(np.min(values)), float(np.max(values))

    lines = [metric]
    for _, row in sub.iterrows():
        a = display_float(float(row[alpha_col]), 3)
        v = float(row[metric])
        lines.append(f"{a:>6}  {_bar(v, lo, hi, width=bar_width):<{bar_width}}  {v:>8.4f}")
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


def _compose_grid_2x2(top_left: str, top_right: str, bottom_left: str, bottom_right: str, gap: int = 4) -> str:
    def lines(s: str) -> list[str]:
        return s.splitlines() or [""]

    tl, tr, bl, br = lines(top_left), lines(top_right), lines(bottom_left), lines(bottom_right)
    left_width = max(
        max((len(line) for line in tl), default=0),
        max((len(line) for line in bl), default=0),
    )

    out: list[str] = []
    top_h = max(len(tl), len(tr))
    for i in range(top_h):
        l = tl[i] if i < len(tl) else ""
        r = tr[i] if i < len(tr) else ""
        out.append(l.ljust(left_width) + (" " * gap) + r.rstrip())

    out.append("")

    bot_h = max(len(bl), len(br))
    for i in range(bot_h):
        l = bl[i] if i < len(bl) else ""
        r = br[i] if i < len(br) else ""
        out.append(l.ljust(left_width) + (" " * gap) + r.rstrip())

    return "\n".join(out).rstrip()


def _first_available(data: dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=float)
    return None


def _load_trajectory_npz(selected_r: float, selected_alpha: float, selected_seed: int | None) -> dict[str, np.ndarray] | None:
    if not TRAJECTORY_DIR.exists():
        return None

    seed = 0 if selected_seed is None else int(selected_seed)

    patterns = [
        f"*r{selected_r:.2f}*a{selected_alpha:.3f}*seed{seed}*.npz",
        f"*r{selected_r:.2f}*alpha{selected_alpha:.3f}*seed{seed}*.npz",
        f"*seed{seed}*.npz",
    ]

    for pattern in patterns:
        matches = sorted(TRAJECTORY_DIR.glob(pattern))
        if matches:
            try:
                with np.load(matches[0], allow_pickle=False) as data:
                    return {k: data[k] for k in data.files}
            except Exception:
                return None
    return None


def _ascii_plot(values, title: str, width: int = 40, height: int = 8) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return f"{title}\n(no data)"

    width = max(24, int(width))
    height = max(6, int(height))

    # resample to target width
    idx = np.linspace(0, len(arr) - 1, width).astype(int)
    sample = arr[idx]

    lo = float(np.min(sample))
    hi = float(np.max(sample))
    span = hi - lo if hi > lo else 1.0

    # 8-level ramp for nicer vertical fill
    glyphs = " ▁▂▃▄▅▆▇█"

    lines = [title]

    for row in range(height):
        # top row = hi, bottom row = lo
        y_top = hi - span * (row / height)
        y_bot = hi - span * ((row + 1) / height)

        chars = []
        for v in sample:
            # map value into this row's vertical band
            frac = (v - y_bot) / (y_top - y_bot) if y_top > y_bot else 0.0
            frac = max(0.0, min(1.0, frac))
            level = int(round(frac * (len(glyphs) - 1)))
            chars.append(glyphs[level])

        label = f"{(hi - span * row / max(1, height - 1)):>7.3f}" if row in (0, height - 1) else f"{'':>7}"
        lines.append(f"{label} │ " + "".join(chars))

    lines.append(f"{'':>7} └" + "─" * (len(sample) + 1))
    lines.append(f"{'':>9}0{' ' * max(0, len(sample)-8)}{len(arr)-1}")

    return "\n".join(lines)


def build_row_mode_text(df: pd.DataFrame, selected_r: float, alpha_values: Sequence[float] | None = None) -> tuple[str, str]:
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
    header = f"α cells observed: {observed} / {total}    seed coverage: min {min_seed}    max {max_seed}"

    blocks = [
        _metric_block(grouped, "piF_tail"),
        _metric_block(grouped, "H_joint"),
        _metric_block(grouped, "best_corr"),
        _metric_block(grouped, "dR2_freeze"),
    ]
    return title, header + "\n\n" + _hstack_blocks(blocks, gap=6)


def build_cell_mode_text(df: pd.DataFrame, selected_r: float, selected_alpha: float) -> tuple[str, str]:
    work = _safe_numeric(
        df,
        ["r", "alpha", "seed", "piF_tail", "H_joint_mean", "best_corr", "corr0", "delta_r2_freeze", "delta_r2_entropy", "K_max"],
    )
    cell = work[
        np.isclose(work["r"], selected_r, atol=1e-12)
        & np.isclose(work["alpha"], selected_alpha, atol=1e-12)
    ].copy()

    title = f"Detail: cell mode   r = {display_float(selected_r, 3)}   α = {display_float(selected_alpha, 3)}"
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
    lines.append("  ".join(f"{c:>12}" for c in present))
    for _, row in table.iterrows():
        lines.append("  ".join(f"{_format_value(row[c]):>12}" for c in present))

    return title, "\n".join(lines)


def build_trajectory_mode_text(df: pd.DataFrame, selected_r: float, selected_alpha: float, selected_seed: int | None, panel_width: int, panel_height: int) -> tuple[str, str]:
    title = (
        "Detail: trajectory mode   "
        f"r = {display_float(selected_r, 3)}   α = {display_float(selected_alpha, 3)}   "
        f"seed = {selected_seed if selected_seed is not None else 0}"
    )

    data = _load_trajectory_npz(selected_r, selected_alpha, selected_seed)
    if data is None:
        return title, "No trajectory file found for selected cell."

    plot_width = max(32, panel_width // 2 - 8)
    plot_height = min(16, max(6, panel_height // 4 )) #if panel_height > 0 else 8))

    f_raw = _first_available(data, ["F_raw", "freeze", "pi_raw"])
    h_joint = _first_available(data, ["H_joint", "H_joint_series", "entropy"])
    k_series = _first_available(data, ["K", "K_series", "complexity"])
    pi_sm = _first_available(data, ["pi", "pi_smooth", "piF_smooth", "F_smooth"])

    tl = _ascii_plot(f_raw if f_raw is not None else [], "F_raw(t)", width=plot_width, height=plot_height)
    tr = _ascii_plot(h_joint if h_joint is not None else [], "H_joint(t)", width=plot_width, height=plot_height)
    bl = _ascii_plot(k_series if k_series is not None else [], "K(t)", width=plot_width, height=plot_height)
    br = _ascii_plot(pi_sm if pi_sm is not None else [], "πF_smooth(t)", width=plot_width, height=plot_height)

    return title, _compose_grid_2x2(tl, tr, bl, br, gap=4)


class DetailView(Panel):
    """Detail panel for row / cell / trajectory inspection."""

    def __init__(self, **kwargs):
        super().__init__("Detail", id="detail", **kwargs)

    def update_detail(self, df: pd.DataFrame, selection: DetailSelection, alpha_values: Sequence[float] | None = None) -> None:
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
                    panel_width=max(60, self.size.width if self.size else 80),
                    panel_height=max(16, self.size.height if self.size else 32),
                )
        else:
            title, body = "Detail", f"Unknown mode: {selection.mode}"

        self.title = title
        self.set_body(body)
